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

from autopath.databits import Bag, TileBag, Clip, Partition, Fold
from autopath.pancan.tools.tfrecord import TFRecordDataset, get_tfrecord_parser


logger = Logger()



class TileClipDatasetBuilder(Datablock):
    """Factory that builds a PyTorch Dataset from a :class:`PancanTileClip`.

    Uses the Bag/Clip model: the clip is partitioned into bags (shards),
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
	"""A bag of pathology tiles repacked from TFRecords into MDS shards.

	During ``__build__``, reads all tiles from the source TFRecord,
	duplicates the slide-level label for every tile, and writes them
	as MDS samples together with the bag ``name``.

	Topics
	------
	shards
		MDS shards directory.  ``dataset()`` returns a
		:class:`StreamingDataset`.
	"""

	VERSION = 2

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

	def __build__(self):
		"""Read tiles from source TFRecord and repack as MDS shards."""
		tiles_tensor = self._source_tensor()
		n_tiles = tiles_tensor.shape[0]
		tiles_np = tiles_tensor.numpy().astype(np.uint8)

		label = self._label
		name = self._name

		columns = {
			'tile': 'ndarray:uint8',
			'label': 'str',
			'bag_name': 'str',
			'tile_index': 'int32',
		}

		shards_dir = self.path('shards', ensure_dirpath=True)

		sample_bytes = tiles_np[0].nbytes + len(label) + len(name) + 4
		size_limit = self.config.shard_size * sample_bytes

		with MDSWriter(
			out=shards_dir,
			columns=columns,
			size_limit=size_limit,
		) as writer:
			for i in range(n_tiles):
				writer.write({
					'tile': tiles_np[i],
					'label': label,
					'bag_name': name,
					'tile_index': np.int32(i),
				})

		self.log.verbose(
			f"Wrote MDS shards to {shards_dir}: "
			f"{n_tiles} tiles, label={label!r}, name={name!r}"
		)

		del tiles_tensor, tiles_np
		gc.collect()
		return self

	# ── Read / Dataset ──────────────────────────────────────────────

	def dataset(self, *, shuffle: bool = False) -> StreamingDataset:
		"""Return a :class:`StreamingDataset` over this bag's MDS shards.

		Each sample is a dict with keys ``tile`` (ndarray), ``label``
		(str), ``bag_name`` (str), and ``tile_index`` (int32).
		"""
		if self._is_local_fs:
			return StreamingDataset(local=self.path('shards'), shuffle=shuffle)
		else:
			return StreamingDataset(remote=self.path('shards'), shuffle=shuffle)

	def __read__(self, topic=None):
		if topic == 'shards':
			return self.fs.ls(self.path('shards'))
		raise ValueError(f"Unknown topic: {topic!r}")

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

	def UNSAFE_clear(self, *topics, OVERRIDE: bool = False, clear_dirpath: bool = False):
		"""Allow clearing the MDS shards (they are derived, not source data)."""
		return super().UNSAFE_clear(*topics, OVERRIDE=OVERRIDE, clear_dirpath=clear_dirpath)


class PancanTileClip(Clip):
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
	def n_shards(self):
		return len(self.bagpaths)

	def __shard__(self, idx: int):
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

	class ShardMaker(Clip.ShardMaker):
		"""Build a single PancanTileBag shard.

		Carries the bag index. Takes the clip (stack) as the first arg
		via the executor, instantiates the bag and builds it.
		"""
		def __init__(self, idx: int):
			super().__init__(idx)

		def __call__(self, stack, *, build=True):
			shard = stack.__shard__(self.idx)
			if build:
				shard.build()
			del shard
			gc.collect()

	def __split__(self, *args, **kwargs):
		"""Return one ShardMaker per bag for parallel building."""
		makers = [self.ShardMaker(idx) for idx in range(self.n_shards)]
		callable_kwargs = dict(build=True)
		self.log.info(
			f"Split {self.__class__.__name__}: {len(makers)} bags to build"
		)
		return makers, callable_kwargs

	def __stack__(self, results=None):
		"""Persist bag_lens after all parallel bag builds complete."""
		self.log.info(f"Stacking {self.n_shards} bags of {self.__class__.__name__}")
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
	):
		"""Return a unified dataset over all bags using MDS multi-stream API.

		Each sample dict contains:

		* ``tile`` — uint8 tile image ndarray
		* ``label`` — cancer type label string
		* ``bag_name`` — bag/slide name string
		* ``tile_index`` — index into the originating bag
		* ``bag_index`` — index of the bag within this clip

		Parameters
		----------
		shuffle : bool
			Whether to shuffle within the streaming dataset.
		skip_invalid_bags : bool
			If ``True``, silently skip bags whose MDS shards have not
			been built yet.
		"""
		from autopath.deep.features import LabeledStreamDataset

		streams = []
		bag_labels = []
		bag_lens_list = []
		n_skipped = 0

		for i in range(self.n_shards):
			bag = self.shard(i)
			if skip_invalid_bags and not bag.valid():
				n_skipped += 1
				continue
			if bag._is_local_fs:
				streams.append(Stream(local=bag.path('shards')))
			else:
				streams.append(Stream(remote=bag.path('shards')))
			bag_labels.append(bag.label)
			bag_lens_list.append(len(bag))

		if n_skipped:
			self.log.info(
				f"Skipped {n_skipped}/{n_skipped + len(streams)} "
				f"unbuilt bags in {self.__class__.__name__}.dataset()"
			)

		base = StreamingDataset(streams=streams, shuffle=shuffle)
		result = LabeledStreamDataset(base, bag_lens_list, bag_labels)
		return result



class PancanTilePartition(Partition):
	pass


class PancanTileFold(Fold):
	pass

