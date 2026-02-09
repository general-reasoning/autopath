from dataclasses import dataclass
import functools
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

from autopath.databits import Clip, Split, Partition, Fold, ClipDataset
from autopath.tiles import TileShard, TileBag
from autopath.pancan.tools.tfrecord import TFRecordDataset, get_tfrecord_parser


logger = Logger()

	
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
		TileShard.__init__(self, *args, **kwargs)

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

	@property
	def dirpath(self, topic, *, ensure: bool = False): 
		return self._dirpath

	def path(self, topic, *, ensure_dirpath: bool = False):
		return os.path.join(self._dirpath, self.TOPICFILES[topic]) if self.TOPICFILES[topic] is not None else None

	def UNSAFE_clear(self):
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


class PancanTileBagClip(Clip):
	TOPICFILE = "bag_lens.npz"
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

	@functools.cached_property
	def bags(self):
		return [
			PancanTileBag(
				root=self._root_,
				spec=dict(source=bagpath,),
				revision=self.revision,
				verbose=self.verbose,
				debug=self.debug,
			)
			for bagpath in self.bagpaths
		]

	def bag(self, idx: int):
		bagpath = self.bagpaths[idx]
		bag = PancanTileBag(
				root=self._root_,
				spec=dict(source=bagpath,),
				revision=self.revision,
				verbose=self.verbose,
				debug=self.debug,
			)
		return bag

	def shard(self, idx: int):
		return self.bag(idx)

	@property
	def bag_lens(self):
		return self.read()

	def __len__(self):
		return len(self.bags)
	
	def __build__(self):
		self.log.verbose(f"Computing bag lens")
		if self.verbose:
			bagsitor = tqdm.tqdm(self.bags)
		else:
			bagsitor = self.bags
		bag_lens = [len(shard) for shard in bagsitor]
		dbx.write_npz(self.path(), bag_lens=bag_lens)
		return self

	def __read__(self):
		bag_lens = dbx.read_npz(self.path(), 'bag_lens')[0]
		return bag_lens

	@functools.cached_property
	def shards(self):
		return self.bags

	@functools.cached_property
	def shard_lens(self):
		return self.bag_lens


#DEPRECATE in favor of PancanTileBagPartition
class PancanTileBagSplit(Split):
	def bag(self, idx: int):
		return self.shard(idx)


class PancanTileBagPartition(Partition):
	def bag(self, idx: int):
		return self.shard(idx)


class PancanTileBagFold(Fold):
	def bag(self, idx: int):
		return self.shard(idx)


def pancan_tilebagset_builder(clip: PancanTileBagClip,
				   transform: Optional[torchvision.transforms.Compose] = None,
				   *,
				   debug: bool = False,
               	   verbose: bool = False,
                   log = None,
):
		clip = dbx.eval_term(clip)
		transform = dbx.eval_term(transform)
		return ClipDatasetBuilder(spec=dict(clip=clip, transform=transform,),
								  debug=debug, 
								  verbose=verbose,
								  log=log,
		)
