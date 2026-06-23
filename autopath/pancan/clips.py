import contextlib
from dataclasses import dataclass
import functools
import gc
import itertools
import math
import os
from typing import Optional

import tqdm

import fsspec
import numpy as np

import torch
import torchvision

from streaming import MDSWriter, Stream, StreamingDataset

import dbx
from dbx import Logger, Datablock

from autopath.autobits import Bag, TileBag, Clip, Partition, Fold, sanitize_collate
from autopath.pancan.annotations import extract_case_id, get_annotations
from autopath.pancan.tools.tfrecord import TFRecordDataset, get_tfrecord_parser
from autopath.tools import read_mds_samples


logger = Logger()



class TileClipDatasetBuilder(Datablock):
    """Factory that builds a PyTorch Dataset from a :class:`PancanTileClip`.

    	Uses the Bag/Clip model: the clip is partitioned into bags (blocks),
    and the inner :class:`Dataset` lazily indexes into them via
    cumulative-sum bounds.  Supports optional per-sample transforms,
    target transforms, and deterministic bag-order shuffling.
    """

    @dataclass
    class CONFIG:
        clip: 'PancanTileClip'
        transform: Optional[torchvision.transforms.Compose] = None
        target_transform: Optional[torchvision.transforms.Compose] = None
        shuffle_seed: Optional[int] = None

    def __post_init__(self):
        self.n_bags = self.cfg.clip.n_bags
        self.log.debug(f"INITIALIZING TileClipDatasetBuilder using {self.n_bags} bags: BEGIN")
        _bag_indices = np.arange(self.n_bags)
        if self.cfg.shuffle_seed is not None:
            self.log.verbose(f"Shuffling bag indices with seed {self.cfg.shuffle_seed}")
            rng = np.random.default_rng(self.cfg.shuffle_seed)
            rng.shuffle(_bag_indices)
        self._bag_indices = [int(i) for i in _bag_indices]
        self.bag_lens = [self.cfg.clip.bag_lens[i] for i in self._bag_indices]
        self.log.debug("Computing bag_bounds...")
        self.bag_bounds = np.cumsum(self.bag_lens)
        self.log.debug("Computing bag_bounds... DONE")
        self.log.detailed(f"{self.n_bags=}, {self.bag_bounds=}")
        self._bag_idx = None
        self._bag = None
        self._bag_label = None
        self._bag_slide = None
        self.log.debug(f"INITIALIZING TileClipDatasetBuilder using {self.n_bags} bags: END")

    @functools.lru_cache(maxsize=3)
    def bag(self, bag_idx):
        if bag_idx != self._bag_idx:
            self._bag = None
            self._bag_label = None
            self._bag_slide = None
            gc.collect()
            self._bag_idx = bag_idx
            self._bag = self.cfg.clip.bag(self._bag_indices[self._bag_idx])
        return self._bag

    def __len__(self):
        return self.bag_bounds[-1]

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, builder):
            self.dataset_builder = builder

        def __len__(self):
            return len(self.dataset_builder)

        def __getitem__(self, index):
            self.dataset_builder.log.silent(f"GETTING item {index} from dataset")
            bag_idx = np.searchsorted(self.dataset_builder.bag_bounds, index, side='right')
            bag_lo = self.dataset_builder.bag_bounds[bag_idx-1] if bag_idx > 0 else 0
            bag_hi = self.dataset_builder.bag_bounds[bag_idx]
            bag_len = self.dataset_builder.bag_lens[bag_idx]
            idx = index - bag_lo
            self.dataset_builder.log.silent(f"----------------------> {index=}, {bag_idx=}, {bag_lo=}, {bag_len=}, {bag_hi=}, {idx=}")
            tensor = self.dataset_builder.bag(bag_idx).tensor
            sample = tensor[idx]
            if self.dataset_builder.cfg.transform is not None:
                sample = self.dataset_builder.cfg.transform(sample)
            labels = self.dataset_builder.bag(bag_idx).labels
            label = labels[idx]
            self.dataset_builder.log.silent("APPLYING target_transform")
            if self.dataset_builder.cfg.target_transform is not None:
                label = self.dataset_builder.cfg.target_transform(label)
            return sample, label

    def dataset(self):
        return self.Dataset(self)


class TileClipDataLoaderBuilder(Datablock):
    """Factory that builds a DataLoader from a :class:`TileClipDatasetBuilder`."""

    @dataclass
    class CONFIG:
        clip_dataset_builder: TileClipDatasetBuilder
        batch_size: int
        shuffle: bool = False

    def __init__(self, spec: dict, *, dataloader_kwargs):
        Datablock.__init__(self, spec=spec, dataloader_kwargs=dataloader_kwargs)
        self.dataset = self.cfg.clip_dataset_builder.dataset()

    def __post_init__(self):
        self.dataloader_kwargs['shuffle'] = self.spec['shuffle']
        self.dataloader_kwargs['batch_size'] = self.spec['batch_size']

    def dataloader(self):
        self.log.debug(f"Initializing TileClipDataLoaderBuilder dataloader with kwargs: {self.dataloader_kwargs}")
        return torch.utils.data.DataLoader(dataset=self.dataset, **self.dataloader_kwargs)




	
class PancanTFRecordDataset(TFRecordDataset):
		def __init__(self, tfrecords_path, index_path, transform):
			self.index = np.load(index_path)['arr_0']
			super().__init__(tfrecords_path, self.index, transform=transform)

		def __len__(self):
			return len(self.index)


class PancanTileBag(TileBag):
	"""A bag of pathology tiles repacked from TFRecords into MDS blocks.

	During ``__build__``, reads all tiles from the source TFRecord,
	duplicates the slide-level label for every tile, and writes them
	as MDS samples together with the bag ``name``.

	Topics
	------
	shards
		MDS shards directory.  ``dataset()`` returns a
		:class:`StreamingDataset`.
	"""

	VERSION = 4

	TOPICS = ['shards']

	@dataclass
	class CONFIG(Datablock.CONFIG):
		source: str
		shard_size: int = 256

	def __init__(self, *args, **kwargs):
		TileBag.__init__(self, *args, **kwargs)

	def __post_init__(self):
		root, tail = self.config.source.split('/tfrecords/')
		self._label = root.split('/')[-1] #cancer
		self.resolution, records = tail.split('/')
		self._name, _  = os.path.splitext(records)
		# Legacy source paths used for reading raw TFRecords.
		tilesfile = os.path.basename(self.config.source)
		indexfile = tilesfile.split('.')[0] + '.index.npz'
		self._source_dirpath = os.path.dirname(self.config.source)
		self._source_tilesfile = tilesfile
		self._source_indexfile = indexfile

	@property
	def label(self):
		return self._label

	@property
	def name(self):
		return self._name


	# ── Source TFRecord access (for build) ──────────────────────────

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

	# ── Validity ────────────────────────────────────────────────────

	def validtopic(self, topic=None):
		if topic == 'shards':
			# MDS directory is valid when index.json has been written.
			return self.fs.exists(
				os.path.join(self.path('shards'), 'index.json')
			)
		return super().validtopic(topic)

	# ── Build ───────────────────────────────────────────────────────

	@property
	def _is_local_fs(self):
		"""True when this bag's storage is on a local filesystem."""
		protocol = self.fs.protocol if isinstance(self.fs.protocol, str) else self.fs.protocol[0]
		return protocol in ('file', 'local', '')

	def _clip_source_root(self):
		"""Derive the clip source root from this bag's source path.

		The bag source has the form
		``{clip_root}/{cohort}/tfrecords/{resolution}/{slide}.tfrecords``.
		Navigating up past ``/tfrecords/`` and then one more level gives
		the clip source root.
		"""
		cohort_dir, _ = self.config.source.split('/tfrecords/')
		return os.path.dirname(cohort_dir)

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
			'tile': 'ndarray:uint8',
			'bag_name': 'str',
			'tile_index': 'int32',
			'annotations': 'json',
		}

		shards_dir = self.path('shards', ensure_dirpath=True)

		sample_bytes = tiles_np[0].nbytes + len(label) + len(name) + 4
		size_limit = self.config.shard_size * sample_bytes

		with MDSWriter(
			out=shards_dir,
			columns=columns,
			size_limit=size_limit,
			exist_ok=True,
		) as writer:
			for i in range(n_tiles):
				writer.write({
					'tile': tiles_np[i],
					'bag_name': name,
					'tile_index': np.int32(i),
					'annotations': annotations,
				})

		self.log.verbose(
			f"Wrote MDS shards to {shards_dir}: "
			f"{n_tiles} tiles, label={label!r}, name={name!r}, "
			f"has_annotations={annotations is not None}"
		)

		del tiles_tensor, tiles_np
		gc.collect()
		return self

	# ── Read / Dataset ──────────────────────────────────────────────

	@contextlib.contextmanager
	def dataset(self, *, shuffle: bool = False, batch_size: int = 1, **kwargs):
		"""Yield a :class:`StreamingDataset` over this bag's MDS shards.

		Use as a context manager to ensure shared-memory file
		descriptors are released promptly::

		    with bag.dataset() as ds:
		        tile = ds[0]['tile']

		Each sample is a dict with keys ``tile`` (ndarray), ``label``
		(str), ``bag_name`` (str), and ``tile_index`` (int32).
		"""
		ds_kwargs = dict(shuffle=shuffle, batch_size=batch_size)
		if self._is_local_fs:
			ds = StreamingDataset(local=self.path('shards'), **ds_kwargs)
		else:
			ds = StreamingDataset(remote=self.path('shards'), **ds_kwargs)
		try:
			yield ds
		finally:
			del ds
			gc.collect()

	def __read__(self, topic):
		if topic == 'shards':
			return self.fs.ls(self.path('shards'))
		raise ValueError(f"Unknown topic: {topic!r}")

	def data(self):
		"""Bulk-read all MDS samples as a list of dicts.

		Uses :class:`MDSReader` (plain file I/O) rather than
		:class:`StreamingDataset` to avoid shared-memory leaks
		when reading many bags.
		"""
		return list(read_mds_samples(self.path('shards')))

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

	def verify_tile_integrity(self, *, n_tiles=None):
		"""Compare MDS tiles against source TFRecords.

		Reads tiles from the MDS shards and from the source TFRecords
		and checks for pixel-exact equality.  Also verifies that
		``bag_name`` and ``tile_index`` metadata are correct.

		Parameters
		----------
		n_tiles : int | None
			Number of tiles to check.  ``None`` checks all tiles.

		Returns
		-------
		dict
			``n_checked``, ``n_pixel_mismatched``, ``n_meta_mismatched``,
			``max_pixel_diff``, ``details`` (list of per-tile dicts for
			any mismatched tiles).
		"""
		tiles_tfr = self.tiles  # (N, C, H, W) uint8 from TFRecords
		mds_samples = list(read_mds_samples(self.path('shards')))

		n_total = min(len(tiles_tfr), len(mds_samples))
		if n_tiles is not None:
			n_total = min(n_total, n_tiles)

		expected_name = self.name
		n_pixel_mismatch = 0
		n_meta_mismatch = 0
		max_pixel_diff = 0
		details = []

		for i in range(n_total):
			tile_tf = tiles_tfr[i].numpy()  # (C, H, W) uint8
			sample = mds_samples[i]
			tile_mds = sample['tile']  # ndarray from MDS
			if isinstance(tile_mds, torch.Tensor):
				tile_mds = tile_mds.numpy()

			# MDS stores as (H, W, C); TFRecords as (C, H, W)
			# Normalise to (C, H, W) for comparison.
			if tile_mds.ndim == 3 and tile_mds.shape[0] != tile_tf.shape[0]:
				tile_mds_chw = np.transpose(tile_mds, (2, 0, 1))
			else:
				tile_mds_chw = tile_mds

			pixel_diff = int(np.abs(
				tile_tf.astype(np.int16) - tile_mds_chw.astype(np.int16)
			).max())
			max_pixel_diff = max(max_pixel_diff, pixel_diff)

			# Metadata checks
			mds_name = sample.get('bag_name', '')
			mds_idx = int(sample.get('tile_index', -1))
			name_ok = (mds_name == expected_name)
			idx_ok = (mds_idx == i)

			if pixel_diff > 0:
				n_pixel_mismatch += 1
			if not name_ok or not idx_ok:
				n_meta_mismatch += 1

			if pixel_diff > 0 or not name_ok or not idx_ok:
				details.append(dict(
					tile_index=i,
					pixel_diff=pixel_diff,
					shape_mds=tile_mds.shape,
					shape_tfr=tile_tf.shape,
					bag_name_mds=mds_name,
					bag_name_expected=expected_name,
					tile_index_mds=mds_idx,
				))

		result = dict(
			n_checked=n_total,
			n_pixel_mismatched=n_pixel_mismatch,
			n_meta_mismatched=n_meta_mismatch,
			max_pixel_diff=max_pixel_diff,
			details=details,
		)
		status = 'PASSED' if (n_pixel_mismatch == 0 and n_meta_mismatch == 0) else 'FAILED'
		self.log.info(
			f"verify_tile_integrity {status}: "
			f"{n_total} checked, {n_pixel_mismatch} pixel mismatches, "
			f"{n_meta_mismatch} metadata mismatches, "
			f"max_pixel_diff={max_pixel_diff}"
		)
		return result

	def UNSAFE_clear(self, *topics, OVERRIDE: bool = False, clear_dirpath: bool = False):
		"""Allow clearing the MDS shards (they are derived, not source data)."""
		return super().UNSAFE_clear(*topics, OVERRIDE=OVERRIDE, clear_dirpath=clear_dirpath)


class PancanTileClip(Clip):
	VERSION = 2
	@dataclass
	class CONFIG:
		source: str
		resolution: str

	def __init__(self, *args, **kwargs):
		super().__init__(*args, v2=True, **kwargs)

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
				_tfrecords_paths(fs, d, resolution=self.config.resolution) 
				for d in fs.ls(self.config.source) 
				if _is_tfrecords_dir(fs, d, resolution=self.config.resolution)
			]
		))
		return self

	@property
	def n_blocks(self):
		return len(self.bagpaths)

	def __block__(self, idx: int):
		bagpath = self.bagpaths[idx]
		relpath = os.path.relpath(bagpath, self.config.source)
		source_expr = self.spec['source'][1:]  # strip leading '$' from specline
		source = f"$os.path.join({source_expr}, '{relpath}')"
		tag = os.path.join(relpath.split(os.sep)[0], os.path.splitext(os.path.basename(bagpath))[0])
		return PancanTileBag(
				url=self.url,
				spec=dict(source=source,),
				tag=tag,
				keyby=self.keyby,
				revision=self.revision,
				verbose=self.verbose,
				debug=self.debug,
			)

	class BlockMaker(Datastack.BlockMaker):
		"""Build a single PancanTileBag block.

		Carries the bag index. Takes the clip (stack) as the first arg
		via the executor, instantiates the bag and builds it.
		"""
		def __init__(self, idx: int):
			super().__init__(idx)

		def __call__(self, stack, *, build=True):
			block = stack.__block__(self.idx)
			if build:
				block.build()
			del block
			gc.collect()

	def __split__(self, *args, **kwargs):
		"""Return one BlockMaker per bag for parallel building."""
		makers = [self.BlockMaker(idx) for idx in range(self.n_blocks)]
		callable_kwargs = dict(build=True)
		self.log.info(
			f"Split {self.__class__.__name__}: {len(makers)} bags to build"
		)
		return makers, callable_kwargs

	def __stack__(self, results=None):
		"""Persist bag_lens after all parallel bag builds complete."""
		self.log.info(f"Stacking {self.n_blocks} bags of {self.__class__.__name__}")
		result = super().__stack__(results)
		self.log.info(f"Build complete: {self.__class__.__name__}")
		return result

	@property
	def bag_lens(self):
		return self.read()

	def __len__(self):
		return len(self.bags)

	def __read__(self):
		bag_lens = dbx.read_npz(self.path('bag_lens'), 'bag_lens')['bag_lens']
		return bag_lens

	def dataset(
		self,
		*,
		shuffle: bool = False,
		skip_invalid_bags: bool = False,
		batch_size: int = 1,
		**streaming_kwargs,
	):
		"""Return a unified dataset over all bags using MDS multi-stream API.

		Each sample dict contains the fields written by
		:meth:`PancanTileBag.__build__`:

		* ``tile`` — uint8 tile image ndarray
		* ``bag_name`` — bag/slide name string
		* ``tile_index`` — index into the originating bag
		* ``annotations`` — clinical/pathological metadata dict

		Parameters
		----------
		shuffle : bool
			Whether to shuffle within the streaming dataset.
		skip_invalid_bags : bool
			If ``True``, silently skip bags whose MDS shards have not
			been built yet.
		batch_size : int | None
			Per-device batch size passed to :class:`StreamingDataset`.
		"""
		streams = []
		n_skipped = 0

		for i in range(self.n_blocks):
			bag = self.block(i)
			if skip_invalid_bags and not bag.valid(topic=None):
				n_skipped += 1
				continue
			if bag._is_local_fs:
				streams.append(Stream(local=bag.path('shards')))
			else:
				streams.append(Stream(remote=bag.path('shards')))

		if n_skipped:
			self.log.info(
				f"Skipped {n_skipped}/{n_skipped + len(streams)} "
				f"unbuilt bags in {self.__class__.__name__}.dataset()"
			)

		return StreamingDataset(streams=streams, shuffle=shuffle, batch_size=batch_size, **streaming_kwargs)



class PancanTilePartition(Partition):
	VERSION = 2


class PancanTileFold(Fold):
	VERSION = 2

	def dataset(self, *, shuffle: bool = False, skip_invalid_bags: bool = False, batch_size: int = 1, **streaming_kwargs):
		"""Delegate to :meth:`PancanTileClip.dataset` (same shard interface)."""
		return PancanTileClip.dataset(self, shuffle=shuffle, skip_invalid_bags=skip_invalid_bags, batch_size=batch_size, **streaming_kwargs)

