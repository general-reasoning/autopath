from dataclasses import dataclass
import functools
import gc
import itertools
import os
from typing import Optional

import tqdm

import fsspec
import numpy as np

import torch
import torchvision


import dbx
from dbx import Logger, Datablock

from autopath.databits import Bag, Clip, Partition, Fold
from autopath.pancan.tools.tfrecord import TFRecordDataset, get_tfrecord_parser


logger = Logger()


class TileBag(Bag):
    """A bag of image tiles backed by a tensor.

    Combines the former ``TileShard`` (tensor-backed tile storage) and
    ``TileBag`` (Bag subclass) into a single class.
    """

    @functools.cached_property
    def tiles(self):
        return self.tensor

    @property
    def labels(self):
        raise NotImplementedError()


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
	@dataclass
	class CONFIG(Datablock.CONFIG):
		source: str

	def __init__(self, *args, **kwargs):
		TileBag.__init__(self, *args, **kwargs)

	def __post_init__(self):
		root, tail = self.config.source.split('/tfrecords/')
		self._label = root.split('/')[-1] #cancer
		self.resolution, records = tail.split('/')
		self._name, _  = os.path.splitext(records)
		tilesfile = os.path.basename(self.config.source)
		indexfile = tilesfile.split('.')[0] + '.index.npz'
		self.TOPICFILES = {'index': indexfile, 'tiles': tilesfile, 'labels': None}
		self._dirpath = os.path.dirname(self.config.source)

	@property
	def label(self):
		return self._label

	@property
	def name(self):
		return self._name

	def dirpath(self, topic=None, *, ensure: bool = False): 
		return self._dirpath

	def path(self, topic=None, *, ensure_dirpath: bool = False):
		if topic is None:
			return self._dirpath
		return os.path.join(self._dirpath, self.TOPICFILES[topic]) if self.TOPICFILES[topic] is not None else None

	def UNSAFE_clear(self, *topics, OVERRIDE: bool = False, clear_dirpath: bool = False):
		raise ValueError(f"Read-Only datablock: {self}")

	def __read__(self, topic):
		if topic == 'index':
			result = np.load(self.path(topic))['arr_0']
		elif topic == 'tiles':
			result = self.tiles
		elif topic == 'labels':
			result = np.array([self.label]*len(self))
		else:
			raise ValueError(f"Unknown topic: {topic}")
		return result

	def __len__(self):
		return len(self.read('index'))

	@property
	def size(self):
		return len(self)
	
	@property
	def dataset(self):
		parser = get_tfrecord_parser(
				self.path('tiles'),
				('image_raw',),
				to_numpy=True,
				decode_images=True
		)
	
		def transform(*args, **kwargs):
			return parser(*args, **kwargs)[0]
		dataset = PancanTFRecordDataset(self.path('tiles'), self.path('index'), transform=transform)
		return dataset
	
	@property
	def tensor(self):
		tensors = list(self.dataset)
		tensor = torch.stack(tensors).permute(0, 3, 1, 2)
		return tensor

	@property
	def tiles(self):
		return self.tensor

	@property
	def labels(self):
		return self.read('labels')


class PancanTileClip(Clip):
	@dataclass
	class CONFIG:
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
		tag = os.path.splitext(os.path.basename(bagpath))[0]
		return PancanTileBag(
				url=self.url,
				spec=dict(source=source,),
				tag=tag,
				revision=self.revision,
				verbose=self.verbose,
				debug=self.debug,
			)

	def __build__(self):
		"""Bags are pre-existing TFRecords — skip parallel shard building."""
		self.__stack__()
		return self

	@property
	def bag_lens(self):
		return self.read()

	def __len__(self):
		return len(self.bags)

	def __read__(self):
		bag_lens = dbx.read_npz(self.path('bag_lens'), 'bag_lens')[0]
		return bag_lens



class PancanTilePartition(Partition):
	pass


class PancanTileFold(Fold):
	pass


def pancan_tile_clip_dataset_builder(
	*,
	clip: PancanTileClip,
	transform: Optional[torchvision.transforms.Compose] = None,
	debug: bool = False,
	verbose: bool = False,
	log = None,
):
		clip = dbx.eval(clip)
		transform = dbx.eval(transform)
		kwargs = dict(
			debug=debug, 
			verbose=verbose,
		)
		if log is not None:
			kwargs['log'] = log

		return TileClipDatasetBuilder(spec=dict(clip=clip, transform=transform,), **kwargs)
