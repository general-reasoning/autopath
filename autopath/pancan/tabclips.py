import contextlib
import functools
import gc
import itertools
import os
from dataclasses import dataclass, fields

import fsspec
import numpy as np
import torch
import tqdm
import dbx
from dbx.datablocks import DATAFILE, DATADIR, DIR, DIRTOPIC
from dbx.datapoints import (
    Datablock,
    DatapointTab,
    DatapointTable,
    DatapointPartition,
    DatapointFold,
    DATASLICE,
    SLICETOPIC,
)

from autopath.pancan.annotations import extract_case_id, get_annotations
from autopath.pancan.tools.tfrecord import PancanTFRecordDataset, get_tfrecord_parser
from autopath.tools import read_mds_samples


class PancanTileBag(DatapointTab):
    VERSION = 1

    TOPICS = {
        'tiles': DATASLICE(tile='ndarray:uint8'),
        'annotations': DATASLICE(annotations=dict(
            case_id='str',
            cohort='str',
            gdc_clinical='json',
            gdc_pathology='json',
            gdc_exposure='json',
            gdc_follow_up='json',
            qupath='json',
            mutations='json',
        )),
        'bag_name': DATASLICE(bag_name='str'),
        'tile_index': DATASLICE(tile_index='int32'),
    }

    @dataclass
    class VAR(Datablock.VAR):
        source: str
        datapoints_per_row: int = 1
        shard_size: int = 256

    def __post_init__(self):
        root, tail = self.var.source.split('/tfrecords/')
        self._label = root.split('/')[-1] # cancer
        self.resolution, records = tail.split('/')
        self._name, _  = os.path.splitext(records)
        # Legacy source paths used for reading raw TFRecords.
        tilesfile = os.path.basename(self.var.source)
        indexfile = tilesfile.split('.')[0] + '.index.npz'
        self._source_dirpath = os.path.dirname(self.var.source)
        self._source_tilesfile = tilesfile
        self._source_indexfile = indexfile

    def __build__(self):
        """Read tiles from source TFRecord and repack as MDS shards."""
        label = self._label
        name = self._name

        # Determine which declared or owed topics need to be built
        owed = set(self.owedtopics()) if hasattr(self, 'owedtopics') else set()
        invalid = {t for t in self.topics() if not self.valid_topic(t)}
        needed = invalid | owed

        # Only decode source TFRecord if 'tiles' is needed
        tiles_needed = 'tiles' in needed or not self.valid_topic('tiles')
        if tiles_needed:
            tiles_tensor = self._source_tensor()
            n_tiles = tiles_tensor.shape[0]
            tiles_np = tiles_tensor.numpy().astype(np.uint8)
        else:
            tiles_tensor = tiles_np = None
            n_tiles = self.n_rows('tiles') if hasattr(self, 'n_rows') and self.valid_topic('tiles') else self._source_len()

        # Resolve annotations if annotations is needed or doing a full build
        annotations = None
        if 'annotations' in needed or tiles_needed:
            case_id = extract_case_id(name)
            if case_id is not None:
                clip_root = self._clip_source_root()
                annotations = get_annotations(clip_root, case_id, label)
            else:
                self.log.info(
                    f"Could not extract case ID from bag name {name!r}; "
                    f"annotations will be empty"
                )

        # Write only the needed slices (or all slices if full build)
        slices_to_write = list(needed) if not tiles_needed else None

        with self.slice_writers(only=slices_to_write, flush_every=self.var.shard_size, size_limit=None) as writers:
            for i in range(n_tiles):
                if 'tiles' in writers:
                    writers['tiles'].write({
                        'tile': tiles_np[i],
                    })
                if 'annotations' in writers:
                    writers['annotations'].write({
                        'annotations': annotations,
                    })
                if 'bag_name' in writers:
                    writers['bag_name'].write({
                        'bag_name': name,
                    })
                if 'tile_index' in writers:
                    writers['tile_index'].write({
                        'tile_index': np.int32(i),
                    })

        self.log.verbose(
            f"Wrote MDS slices {list(writers.keys())}: "
            f"{n_tiles} tiles, label={label!r}, name={name!r}, "
            f"has_annotations={annotations is not None}"
        )

        del tiles_tensor, tiles_np
        gc.collect()
        return self

    @property
    def label(self):
        return self._label

    @property
    def name(self):
        return self._name

    def __len__(self):
        return self._source_len()

    @property
    def size(self):
        return len(self)
    
    @property
    def tensor(self):
        return self._source_tensor()

    @property
    def tiles(self):
        return self._source_tensor()

    @property
    def labels(self):
        return np.array([self._label] * len(self))

    @property
    def _source_tiles_path(self):
        return os.path.join(self._source_dirpath, self._source_tilesfile)

    @property
    def _source_index_path(self):
        return os.path.join(self._source_dirpath, self._source_indexfile)

    def _source_dataset(self):
        """Load the source TFRecord as a dataset (heavy — reads from disk)."""
        parser = get_tfrecord_parser(
                self._source_tiles_path,
                ('image_raw',),
                to_numpy=True,
                decode_images=True
        )
    
        def transform(*args, **kwargs):
            return parser(*args, **kwargs)[0]
        dataset = PancanTFRecordDataset(self._source_tiles_path, self._source_index_path, transform=transform)
        return dataset

    def _source_tensor(self):
        """Load all tiles from source TFRecord as a tensor (heavy — full decode)."""
        tensors = list(self._source_dataset())
        tensor = torch.stack(tensors).permute(0, 3, 1, 2)
        return tensor

    def _source_len(self):
        index = np.load(self._source_index_path)['arr_0']
        return len(index)

    def _clip_source_root(self):
        cohort_dir, _ = self.var.source.split('/tfrecords/')
        return os.path.dirname(cohort_dir)


class PancanTileClip(DatapointTable):
    VERSION = 1
    TAB = PancanTileBag

    TOPICS = {
        'tabs': DATADIR,
        'done': DATAFILE('done'),
        'bag_lens': DATAFILE('bag_lens.npz'),
    }

    def __init__(self, *args, filter_built_tabs: bool = True, **kwargs):
        super().__init__(*args, filter_built_tabs=filter_built_tabs, **kwargs)

    @dataclass
    class VAR(Datablock.VAR):
        source: str
        resolution: str
        datapoints_per_row: int = 1

    def __tab__(self, idx: int, tag=None):
        bagpath = self.bagpaths[idx]
        relpath = os.path.relpath(bagpath, self.var.source)
        source_specline = self.spec.get('source')
        if source_specline is not None and self.is_specline(source_specline):
            source_expr = source_specline[1:]  # strip leading '$' from specline
            source = f"$os.path.join({source_expr}, '{relpath}')"
        else:
            source = bagpath
        if tag is None:
            tag = os.path.join(relpath.split(os.sep)[0], os.path.splitext(os.path.basename(bagpath))[0])
        return super().__tab__(
            idx,
            tag=tag,
            source=source,
        )

    def __stack__(self, results=None):
        self.log.info(f"Stacking {self.n_tabs} bags of {self.__class__.__name__}")
        super().__stack__(results)
        
        bag_lens = [len(self.tab(i)) for i in tqdm.tqdm(range(self.n_tabs), desc="Stacking bag lens")]
        dbx.write_npz(self.path('bag_lens', ensure_dirpath=True), bag_lens=bag_lens)
        
        self.log.info(f"Build complete: {self.__class__.__name__}")
        return self

    def __read__(self, *topicpath):
        topicpath = self._normtopic(topicpath)
        if topicpath == ('bag_lens',):
            return dbx.read_npz(self.path('bag_lens'), 'bag_lens')['bag_lens']
        return super().__read__(*topicpath)

    # 2. Properties and Accessors ───────────────────────────────────

    @functools.cached_property
    def bagpaths(self) -> list[str]:
        def _is_tfrecords_dir(fs, d, resolution):
            return (
                fs.isdir(d) and
                'tfrecords' in [os.path.basename(f) for f in fs.ls(d)] and
                fs.isdir(os.path.join(d, 'tfrecords', resolution))
            )

        def _tfrecords_paths(fs, d, resolution):
            dd = os.path.join(d, 'tfrecords', resolution)
            return [f for f in fs.ls(dd) if f.endswith('.tfrecords')]

        fs, _ = fsspec.core.url_to_fs(self.root)
        return list(itertools.chain.from_iterable(
            [
                _tfrecords_paths(fs, d, resolution=self.var.resolution)
                for d in fs.ls(self.var.source)
                if _is_tfrecords_dir(fs, d, resolution=self.var.resolution)
            ]
        ))

    @property
    def n_tabs(self) -> int:
        return len(self.bagpaths)

    @property
    def bag_lens(self):
        return self.read('bag_lens')

    def __len__(self):
        return sum(self.bag_lens)
        
    def bag(self, idx: int):
        return self.tab(idx)
        
    @property
    def bags(self):
        return self.tabs()
        
    @property
    def n_bags(self):
        return self.n_tabs


class PancanTilePartition(DatapointPartition):
    VERSION = 1


class PancanTileFold(DatapointFold):
    VERSION = 1

    # 2. Accessors ───────────────────────────────────────────────────

    def bag(self, idx: int):
        return self.tab(idx)

    @property
    def bags(self):
        return self.tabs()

    @property
    def n_bags(self):
        return self.n_tabs

