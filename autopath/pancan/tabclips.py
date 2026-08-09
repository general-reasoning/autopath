import contextlib
import gc
import itertools
import os
from dataclasses import dataclass

import fsspec
import numpy as np
import torch
from streaming import Stream, StreamingDataset

import dbx
from dbx.datablocks import DIRTOPIC
from dbx.datasamples import DatasampleTab, DatasampleTable
from dbx.datastreams import ZipStreamingDataset, ZipIterableStreamingDatasets, concat_data

from autopath.autobits import Partition, Fold
from autopath.pancan.annotations import extract_case_id, get_annotations
from autopath.pancan.clips import PancanTFRecordDataset
from autopath.pancan.tools.tfrecord import get_tfrecord_parser
from autopath.tools import read_mds_samples


class PancanTileBag(DatasampleTab):
    VERSION = 1

    SLICES = (
        ('tiles',       'ndarray'),
        ('annotations', 'object'),
        'bag_name',    
        ('tile_index',  'int32'),
    )

    @dataclass
    class VAR(DatasampleTab.VAR):
        source: str
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
        tiles_tensor = self._source_tensor()
        n_tiles = tiles_tensor.shape[0]
        tiles_np = tiles_tensor.numpy().astype(np.uint8)

        label = self._label
        name = self._name

        # ── Resolve annotations ───────────────────────────────────
        case_id = extract_case_id(name)
        if case_id is not None:
            clip_root = self._clip_source_root()
            annotations = get_annotations(clip_root, case_id, label)
        else:
            self.log.info(
                f"Could not extract case ID from bag name {name!r}; "
                f"annotations will be empty"
            )
            annotations = None

        columns = {
            'tiles': {
                'tile': 'ndarray:uint8',
            },
            'annotations': {
                'annotations': 'json',
            },
            'bag_name': {
                'bag_name': 'str',
            },
            'tile_index': {
                'tile_index': 'int32',
            }
        }

        with self.slice_writers(columns, flush_every=self.var.shard_size, size_limit=None) as writers:
            for i in range(n_tiles):
                writers['tiles'].write({
                    'tile': tiles_np[i],
                })
                writers['annotations'].write({
                    'annotations': annotations,
                })
                writers['bag_name'].write({
                    'bag_name': name,
                })
                writers['tile_index'].write({
                    'tile_index': np.int32(i),
                })

        self.log.verbose(
            f"Wrote MDS slices: "
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

    @property
    def _is_local_fs(self):
        """True when this bag's storage is on a local filesystem."""
        protocol = self.fs.protocol if isinstance(self.fs.protocol, str) else self.fs.protocol[0]
        return protocol in ('file', 'local', '')

    def _clip_source_root(self):
        cohort_dir, _ = self.var.source.split('/tfrecords/')
        return os.path.dirname(cohort_dir)


class PancanTileClip(DatasampleTable):
    VERSION = 1
    TAB = PancanTileBag

    TOPICS = {'tabs': DIRTOPIC, 'done': 'done', 'bag_lens': 'bag_lens.npz'}

    @dataclass
    class VAR(DatasampleTable.VAR):
        source: str
        resolution: str

    def __post_init__(self):
        def _is_tfrecords_dir(fs, d, resolution):
            is_tf_records_dir = (
                fs.isdir(d) and
                'tfrecords' in [os.path.basename(f) for f in fs.ls(d)] and
                fs.isdir(os.path.join(d, 'tfrecords', resolution))
            )
            return is_tf_records_dir
        def _tfrecords_paths(fs, d, resolution):
            dd = os.path.join(d, 'tfrecords', resolution)
            ff = [f for f in fs.ls(dd) if f.endswith('.tfrecords')]
            return ff
        fs, _ = fsspec.core.url_to_fs(self.root)
        self.bagpaths = list(itertools.chain.from_iterable(
            [
                _tfrecords_paths(fs, d, resolution=self.var.resolution) 
                for d in fs.ls(self.var.source) 
                if _is_tfrecords_dir(fs, d, resolution=self.var.resolution)
            ]
        ))
        return self

    def __tab__(self, idx: int):
        bagpath = self.bagpaths[idx]
        relpath = os.path.relpath(bagpath, self.var.source)
        source_expr = self.spec['source'][1:]  # strip leading '$' from specline
        source = f"$os.path.join({source_expr}, '{relpath}')"
        tag = os.path.join(relpath.split(os.sep)[0], os.path.splitext(os.path.basename(bagpath))[0])
        return super().__tab__(
            idx,
            tag=tag,
            source=source,
        )

    def __stack__(self, results=None):
        self.log.info(f"Stacking {self.n_tabs} bags of {self.__class__.__name__}")
        super().__stack__(results)
        
        import tqdm
        bag_lens = [len(self.tab(i)) for i in tqdm.tqdm(range(self.n_tabs), desc="Stacking bag lens")]
        dbx.write_npz(self.path('bag_lens', ensure_dirpath=True), bag_lens=bag_lens)
        
        self.log.info(f"Build complete: {self.__class__.__name__}")
        return self

    def __read__(self, *topicpath):
        topicpath = self._normtopic(topicpath)
        if topicpath == ('bag_lens',):
            return dbx.read_npz(self.path('bag_lens'), 'bag_lens')['bag_lens']
        return super().__read__(*topicpath)

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


class PancanTilePartition(Partition):
    VERSION = 1
    
    @dataclass
    class VAR(Partition.CONFIG):
        clip: PancanTileClip


class PancanTileFold(Fold):
    VERSION = 1
    
    @dataclass
    class VAR(Fold.CONFIG):
        partition: PancanTilePartition

    def dataset(self, *slices, mode='map', columns=None, shared=None, validate_shared=False, on_conflict='last', skip_none=True, zip_validator=None, **streaming_kwargs):
        if not slices:
            slices = self.var.partition.var.clip.SLICES
        
        datasets = []
        for slice_name in slices:
            streams = []
            for i in range(self.n_blocks):
                tab = self.block(i)
                if tab._is_local_fs:
                    streams.append(Stream(local=tab.path(tab.DATA, slice_name)))
                else:
                    streams.append(Stream(remote=tab.path(tab.DATA, slice_name)))
            
            ds = StreamingDataset(streams=streams, **streaming_kwargs)
            datasets.append(ds)
            
        if len(datasets) == 1:
            return datasets[0]
            
        if columns is not None:
            cols = [columns.get(s) for s in slices]
        else:
            cols = None
            
        zip_kwargs = dict(
            columns=cols,
            shared=shared,
            validate_shared=validate_shared,
            on_conflict=on_conflict,
            skip_none=skip_none,
            zip_validator=zip_validator
        )
        
        if mode == 'map':
            return ZipStreamingDataset(*datasets, **zip_kwargs)
        elif mode == 'iter':
            return ZipIterableStreamingDatasets(*datasets, check_alignment=True, **zip_kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def data(self, *slices, concat: bool = False, **kwargs):
        if not slices:
            slices = self.var.partition.var.clip.SLICES
            
        def _read_slice(slice_name):
            samples = []
            for i in range(self.n_blocks):
                samples.extend(self.block(i).data(slice_name, **kwargs))
            return samples
            
        if len(slices) == 1:
            res = _read_slice(slices[0])
            return concat_data(res) if concat else res
        res = {name: _read_slice(name) for name in slices}
        if concat:
            return {name: concat_data(val) for name, val in res.items()}
        return res
