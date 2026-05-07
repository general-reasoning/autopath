import copy
from dataclasses import dataclass
import functools
import gc
import itertools
import math
import traceback as tb
from typing import Callable, Literal

import tqdm
import numpy as np


import torch

import dbx
from dbx import (
    Datablock, 
    Datastack,
    TorchMultithreadingDatablocksBuilder,
    TorchMultiprocessingDatablocksBuilder,
    MultiprocessingCallableExecutor,
    MultiprocessingDatablocksBuilder,
    MultithreadingCallableExecutor,
    MultithreadingDatablocksBuilder,
    RayDatablocksBuilder,
    RayCallableExecutor,
    InlineCallableExecutor,
    InlineDatablocksBuilder,
    write_npz,
    read_npz,
    write_tensor,
    read_tensor,
    write_pickle,
    read_pickle,
)

def get_executor_cls(parallelization):
    return {
        'ray': RayCallableExecutor,
        'multiprocessing': MultiprocessingCallableExecutor,
        'multithreading': MultithreadingCallableExecutor,
        'inline': InlineCallableExecutor,
    }[parallelization.lower() if parallelization is not None else 'inline']

from autopath.databits import Shard, Bag, Clip, ClipDatasetBuilder
from .tiles import TileBag

from autopath.tools import tensors_to_device, cat_tensor_dicts


class FeatureBagMaker:
    def __init__(self, clip, idx):
        self.clip = clip
        self.idx = idx
    def __call__(self):
        return dbx.eval(self.clip).bag(self.idx)
    def __repr__(self):
        return f"FeatureBagMaker({dbx.quote(self.clip)}, {self.idx})"
        

class BipolarFeatureBagMaker:
    def __init__(self, clip, idx):
        self.clip = clip
        self.idx = idx
    def __call__(self):
        return dbx.eval(self.clip).bag(self.idx)
    def __repr__(self):
        return f"BipolarFeatureBagMaker({dbx.quote(self.clip)}, {self.idx})"


class FeatureBag(Bag):
    VERSION = 1

    @dataclass
    class CONFIG(Datablock.CONFIG):
        tilebag: TileBag
        extractor: Callable

    def __init__(self, *args, gpu_batch_size: int = 16, **kwargs):
        Datablock.__init__(self, 
                     *args, 
                     gpu_batch_size=gpu_batch_size, 
                     **kwargs)

    def __post_init__(self):
        self.TOPICFILES = {
            'features': 'features.npy',
        }
        if self.has_sideband:
            for layer in self.cfg.extractor.sideband_layers:
                self.TOPICFILES[f'sideband_{layer}' ] = \
                    f'sideband_{layer}.npy'
        return self

    def __len__(self):
        return len(self.labels)
    
    @property
    def has_sideband(self):
        return hasattr(self.cfg.extractor, 'sideband_layers')
    
    @property
    def name(self):
        return self.cfg.tilebag.name
    
    @property
    def label(self):
        return self.cfg.tilebag.label

    def __build__(self, extractor=None):
        if extractor is None:
            extractor = self.cfg.extractor
            
        tilebag = self.cfg.tilebag
        feature_list = []
        sideband_list = []
        n_tiles = len(tilebag.tiles)
        for k in range(math.ceil(len(tilebag.tiles)/self.gpu_batch_size)):
            m = k*self.gpu_batch_size
            n = min((k+1)*self.gpu_batch_size, n_tiles)
            batch = tilebag.tiles[m:n].to(self.device)
            self.log.verbose(f"Evaluating batch {k}: {m}:{n} out of {n_tiles} on device: {self.device}")
            feature = extractor(batch)
            feature_list.append(feature.to('cpu'))
            self.log.debug(f"Evaluated batch to a feature of shape {feature.shape} on device: {self.device}")
            del batch
            del feature
            gc.collect()
            torch.cuda.empty_cache()
            if self.has_sideband:
                _sideband = tensors_to_device(extractor.sideband, 'cpu', detach=True)
                extractor.clear_sideband()
                assert set(_sideband.keys()) == set(extractor.sideband_layers), f"_sideband keys must match sideband_layers: {_sideband.keys()} != {extractor.sideband_layers}"
                _sideband_shapes = {k: v.shape for k, v in _sideband.items()}
                self.log.debug(f"Captured _sideband with shapes {_sideband_shapes} on device: {self.device}")
                sideband_list.append(_sideband)
                del _sideband
                gc.collect
                torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
        self.log.debug(f"Concatenating {len(feature_list)} device batch features on device: {self.device}")
        features = torch.cat(feature_list)
        assert len(features) == n_tiles, f"Number of features does not match the number of tiles: {len(features)} != {n_tiles}"
        del feature_list
        gc.collect()
        torch.cuda.empty_cache()
        self.log.debug(f"Storing features of shape {features.shape} on device: {self.device}")
        dbx.write_tensor(features, self.path('features', ensure_dirpath=True))
        del features
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(extractor, 'sideband'):
            self.log.debug(f"Concatenating {len(sideband_list)} device batch sidebands on device: {self.device}")
            sideband = cat_tensor_dicts(sideband_list)
            del sideband_list
            gc.collect()
            torch.cuda.empty_cache()
            if self.debug:
                sideband_shapes = {k: v.shape for k, v in sideband.items()}
                self.log.debug(f"Storing sidebands of shapes {sideband_shapes} on device: {self.device}")
            for lyr, sbd in sideband.items():
                dbx.write_tensor(sbd, self.path(f'sideband_{lyr}', ensure_dirpath=True))
            del sbd
            del sideband
            gc.collect()
            torch.cuda.empty_cache()
        self._len = n_tiles
        return self

    def read(self, topic):
        return dbx.read_tensor(self.path(topic))

    @functools.cached_property
    def features(self):
        self.log.silent(f"features: {''.join(tb.format_stack())}")
        self.log.silent(f"Reading features from {self.path('features')}: BEGIN")
        features = self.read('features')
        self.log.silent(f"Reading features from {self.path('features')}: END")
        return features
    
    def sideband(self, layer):
        return self.read(f'sideband_{layer}')
    
    def layer(self, layer):
        return self.sideband(layer) if layer is not None else self.features
        
    @property
    def tensor(self):
        return self.features
    
    @functools.cached_property
    def labels(self):
        self.log.silent(f"Assemblying labels from {self.cfg.tilebag}: BEGIN")
        labels = list(zip(self.cfg.tilebag.labels, self.cfg.tilebag.tiles))
        self.log.silent(f"Assemblying labels from {self.cfg.tilebag}: END")
        return labels


class FeatureBagClip(Clip):
    VERSION = 1
    TOPICFILES = {"bag_lens": "bag_lens.npy"}
    @dataclass
    class CONFIG:
        tilebagclip: Clip
        extractor: Callable

    class FeatureBagLengthComputer:
        def __init__(self, featurebag):
            self.featurebag = featurebag
        def __call__(self):
            return len(self.featurebag)
        def __repr__(self):
            return f"FeatureBagLengthComputer({self.featurebag})"

    def __init__(self, *, tag: str | None = None, n_workers: int = 1, devices: list[str] = ["cuda"], gpu_batch_size: int = 16, cpu_batch_size: int|None = None, skip_unreadable: bool = True, 
                 gpu_parallelization: str = 'multithreading', 
                 cpu_parallelization: str|None = None, 
                 bag_n_workers: int = 1,
                 bag_cpu_batch_size: int|None = None,
                 bag_cpu_parallelization: str|None = None,
                 **kwargs
    ):
        super().__init__(n_workers=n_workers, devices=devices, gpu_batch_size=gpu_batch_size, cpu_batch_size=cpu_batch_size, skip_unreadable=skip_unreadable, gpu_parallelization=gpu_parallelization, cpu_parallelization=cpu_parallelization, tag=tag, **kwargs)
        self.bag_n_workers = bag_n_workers
        self.bag_cpu_batch_size = bag_cpu_batch_size
        self.bag_cpu_parallelization = bag_cpu_parallelization
        self.executor_cls = get_executor_cls(cpu_parallelization)
        self.bag_executor_cls = get_executor_cls(bag_cpu_parallelization)
        self.builder_cls = {
            'multiprocessing': TorchMultiprocessingDatablocksBuilder,
            'multithreading': TorchMultithreadingDatablocksBuilder,
        }[gpu_parallelization.lower()]

    def __build__(self):
        bags = self.bags
        self.log.verbose(f"Formed {len(bags)} FeatureBags.  Looking for missing bags.")
        missing_bags = [bag for bag in bags if not bag.valid()]
        self.log.verbose(f"Found {len(missing_bags)} missing bags")
        self.log.verbose(f"Building {len(missing_bags)} missing features bags using devices {self.devices} and gpu_batch_size {self.gpu_batch_size}")
        built_bags = self.builder_cls(devices=self.devices, log=self.log).build_blocks(missing_bags, self.cfg.extractor)
        self.log.verbose(f"Built {len(built_bags)} missing features bags")
        self.log.verbose(f"Building {len(bags)} bag_lens using {self.cpu_parallelization} parallelization with {self.n_workers} workers: BEGIN")
        self.log.detailed(f"Building bag_lens for bags with hash paths {[bag.hashpath() for bag in bags]}")
        executor = self.executor_cls(n_workers=self.n_workers, batch_size=self.cpu_batch_size, log=self.log)
        executables = [FeatureBagClip.FeatureBagLengthComputer(bag) for bag in bags]
        bag_lens_list = executor.execute(executables)
        bag_lens = torch.tensor(bag_lens_list)
        self.log.debug(f"bag_lens: {bag_lens}")
        self.log.verbose(f"Building {len(bags)} bag_lens: using {self.cpu_parallelization} parallelization with {self.n_workers} workers: END")
        dbx.write_tensor(bag_lens, self.path("bag_lens", ensure_dirpath=True))
        return self
    
    def __read__(self, topic):
        if topic == "bag_lens":
            result = dbx.read_tensor(self.path("bag_lens"))
        else:
            raise ValueError(f"Unknown {topic=}")
        return result
    
    def features(self):
        self.log.debug(f"Reading features from {self.n_bags} feature bags")
        feature_list = []
        for featurebag in self.bags:
            try:
                feature_list.append(featurebag.features)
            except Exception as e:
                if self.skip_unreadable:
                    continue
                else:
                    raise(e)
        features = torch.stack(feature_list)
        self.log.debug(f"Combined features: shape: {features.shape}")
        return features
    
    def tiles(self):
        self.log.debug(f"Reading tiles from {self.n_bags} feature bags")
        tile_list = []
        for featurebag in self.bags:
            try:
                _tilebag_tiles = featurebag.cfg.tilebag.tiles
                tilebag_tiles = (
                    self.cfg.extractor.transform(_tilebag_tiles) 
                    if self.cfg.extractor.transform is not None 
                    else _tilebag_tiles
                )
                del _tilebag_tiles
                tile_list.append(tilebag_tiles)
            except Exception as e:
                if self.skip_unreadable:
                    continue
                else:
                    raise(e)
        tiles = torch.stack(tile_list)
        self.log.debug(f"Combined tiles: shape: {tiles.shape}")
        return tiles
    
    def sideband(self, layer):
        sideband_list = []
        for featurebag in self.bags:
            try:
                sideband_list.append(featurebag.sideband(layer))
            except Exception as e:
                if self.skip_unreadable:
                    continue
                else:
                    raise(e)
        sideband = torch.cat(sideband_list)
        return sideband
    
    def layer(self, layer=None):
        return self.sideband(layer) if layer is not None else self.features()
    
    @property
    def bags(self):
        if not hasattr(self, '_bags'):
            self.log.verbose(f"FORMING FeatureBags: BEGIN")
            n_bags = self.cfg.tilebagclip.n_bags
            executables = [FeatureBagMaker(self, idx) for idx in range(n_bags)]
            results = self.bag_executor_cls(n_workers=self.bag_n_workers, log=self.log, batch_size=self.bag_cpu_batch_size, tag='FeatureBagClip bag formation').execute(executables)
            self._bags = list(results)
            self.log.verbose(f"FORMING FeatureBags: END")
        return self._bags

    def bag(self, idx: int):
        if hasattr(self, '_bags'):
            return self._bags[idx]
        tilebag = self.cfg.tilebagclip.bag(idx)
        bag = FeatureBag(
            root=self._root_, 
            spec=dict(tilebag=dbx.quote(tilebag), extractor=self.spec['extractor'],),
            gpu_batch_size=self.gpu_batch_size,
            revision=self.revision,
            tag=self.tag,
        )
        return bag

    def shard(self, idx: int):
        return self.bag(idx)
    
    @property
    def shards(self):
        return self.bags
    
    @property
    def n_bags(self):
        return self.cfg.tilebagclip.n_bags
    
    @property
    def n_shards(self):
        return self.n_bags
    
    @functools.cached_property
    def bag_lens(self):
        return self.read("bag_lens")
    
    @property
    def shard_lens(self):
        return self.bag_lens
    
    def labels(self):
        self.log.debug(f"Reading labels from {self.n_bags} feature bags")
        label_list = []
        for featurebag in self.bags:
            try:
                label_list.append(featurebag.labels)  
            except Exception as e:
                if self.skip_unreadable:
                    continue
                else:
                    raise(e)
        labels = np.concatenate(label_list)
        self.log.debug(f"Combined labels: shape: {labels.shape}")
        return labels
     

class FeaturesToFloat:
    def __init__(self, dtype='float64'):
        self.dtype = dtype
    
    def __call__(self, tensor_or_array):
        if isinstance(tensor_or_array, torch.Tensor):
            tensor_or_array = tensor_or_array.numpy()
            return torch.from_numpy(tensor_or_array.astype(self.dtype))
        return tensor_or_array.astype(self.dtype)


class FeaturesLabelTileToFloat:
    def __init__(self, dtype='float64'):
        self.dtype = dtype
    
    def __call__(self, label_tuple_with_tensor_or_array_tile):
        tile = label_tuple_with_tensor_or_array_tile[1]
        if isinstance(tile, torch.Tensor):
            tile = torch.tensor(tile.numpy().astype(self.dtype))
        else:
            tile = tile.astype(self.dtype)
        return tile
    
        

def featurebag_dataset(featurebagclip: Clip, *, transform=None, target_transform=None, bags_shuffle_seed: int = None):
    return ClipDatasetBuilder(spec=dict(clip=featurebagclip, transform=transform, target_transform=target_transform, shuffle_seed=bags_shuffle_seed)).dataset()


class BipolarFeatureBag(Bag):
    from autopath.pancan.probes import BipolarFeatureBagProbe

    VERSION = 1
    TOPICFILES = {
        'bipolar_features': 'bipolar_features.npy',
    }

    @dataclass
    class CONFIG(Datablock.CONFIG):
        from autopath.pancan.probes import BipolarFeatureBagProbe
        probe: BipolarFeatureBagProbe
        bag_index: int
        featurebag: FeatureBag

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __build__(self):
        assert self.cfg.featurebag.handle() == self.cfg.probe.cfg.featurebagclip.bag(self.cfg.bag_index).handle(), \
            f"Featurebag handle mismatch: {self.cfg.featurebag.handle()} != {self.cfg.probe.cfg.featurebagclip.bag(self.cfg.bag_index).handle()}"
        self.log.detailed(f" BipolarFeatureBag {self.cfg.bag_index}: build: BEGIN")
        all_features = self.cfg.probe.tile_bipolar_features
        lo = self.cfg.probe.bag_bounds[self.cfg.bag_index]
        hi = self.cfg.probe.bag_bounds[self.cfg.bag_index+1]
        my_features = all_features[lo:hi, :]
        write_npz(self.path('bipolar_features', ensure_dirpath=True), bipolar_features=my_features)
        self.log.detailed(f" BipolarFeatureBag {self.cfg.bag_index}: build:END")
        self._len = hi - lo
        return self

    def __read__(self, topic: str):
        if topic == 'bipolar_features':
            result = read_npz(self.path('bipolar_features'), 'bipolar_features')['bipolar_features']
        else:
            raise ValueError(f"Unknown {topic=}")
        return result
    
    @property
    def tensor(self):
        return self.bipolar_features

    @functools.cached_property
    def bipolar_features(self):
        return self.read('bipolar_features')

    @functools.cached_property
    def features(self):
        self.log.silent(f"Reading features from {self.cfg.featurebag}: BEGIN")
        result = self.cfg.featurebag.features
        self.log.silent(f"Reading features from {self.cfg.featurebag}: END")
        return result

    @property
    def name(self):
        return self.cfg.featurebag.name

    @functools.cached_property    
    def labels(self):
        self.log.silent(f"Reading labels from {self.cfg.featurebag}: BEGIN")
        result = self.cfg.featurebag.labels
        self.log.silent(f"Reading labels from {self.cfg.featurebag}: END")
        return result

    def __len__(self):
        if not hasattr(self, '_len'):
            if self.validtopic('bipolar_features'):
                # Read length from our own stored feature array — avoids loading
                # tile images via featurebag.labels -> tilebag.tiles, which is
                # the root cause of OOM when computing bag_lens for 2000+ bags.
                data = read_npz(self.path('bipolar_features'), 'bipolar_features')
                self._len = data['bipolar_features'].shape[0]
                del data
            else:
                self._len = len(self.cfg.featurebag)
        return self._len


class BipolarFeatureBagClip(Clip):
    VERSION = 3

    TOPICFILES = {'bag_lens': 'bag_lens.npy'}
    
    @dataclass
    class CONFIG(Datablock.CONFIG):
        from autopath.pancan.probes import BipolarFeatureBagProbe
        probe: BipolarFeatureBagProbe

    def __init__(self, 
                 *args, 
                 tag: str | None = None,
                 build_missing_only: bool = False, 
                 cpu_parallelization: str = None,
                 n_workers: int = 1, 
                 cpu_batch_size: int|None = None,
                 gpu_parallelization: str = 'multithreading',
                 bag_cpu_parallelization: str|None = None,
                 bag_n_workers: int = 1,
                 bag_cpu_batch_size: int|None = None,
                 **kwargs):
        super().__init__(*args, n_workers=n_workers, build_missing_only=build_missing_only, cpu_parallelization=cpu_parallelization, cpu_batch_size=cpu_batch_size, gpu_parallelization=gpu_parallelization, tag=tag, **kwargs)
        self.bag_n_workers = bag_n_workers
        self.bag_cpu_batch_size = bag_cpu_batch_size
        self.bag_cpu_parallelization = bag_cpu_parallelization
        self.executor_cls = get_executor_cls(cpu_parallelization)
        self.bag_executor_cls = get_executor_cls(bag_cpu_parallelization)
        self.builder_cls = {
            'ray': RayDatablocksBuilder,
            'multiprocessing': MultiprocessingDatablocksBuilder,
            'multithreading': MultithreadingDatablocksBuilder,
            'inline': InlineDatablocksBuilder,
        }[cpu_parallelization.lower() if cpu_parallelization is not None else 'inline']

    def __build__(self):
        self.log.verbose(f"__build__: BUILDING BipolarFeatureBags from {self.cfg.probe} using {self.n_workers} processes with {self.cpu_parallelization} parallelization: BEGIN")
        bags = self.bags
        missing_bags = [] 
        bag_lens = []
        if self.build_missing_only:
            if self.verbose:
                bagitor = tqdm.tqdm(bags, desc=f"{self.anchor}: LOOKING for missing BipolarFeatureBags")
            else:
                bagitor = bags
            for bag in bagitor:
                if not bag.valid():
                    missing_bags.append(bag)
                bag_lens.append(len(bag))
        else:
            missing_bags = bags

        self.log.verbose(f"__build__: BUILDING {len(missing_bags)} {'missing' if self.build_missing_only else 'all'} BipolarFeatureBags using {self.n_workers} processes with {self.cpu_parallelization} parallelization: BEGIN")
        if self.n_workers > 0:
            missing_blocks = self.builder_cls(n_workers=self.n_workers, log=self.log, tag=f"{self.anchor}: __build__: BUILDING BipolarFeatureBags").build_blocks(missing_bags)
        else:   
            if self.verbose:
                bagitor = tqdm.tqdm(missing_bags, desc=f"{self.anchor}: __build__: BUILDING BipolarFeatureBags")
            else:
                bagitor = missing_bags
            for bag in bagitor:
                bag.__build__()
        self.log.verbose(f"__build__: BUILDING {len(missing_bags)} {'missing' if self.build_missing_only else 'all'} BipolarFeatureBags using {self.n_workers} processes with {self.cpu_parallelization} parallelization: END")
        self.log.verbose(f"__build__: COMPUTING bag_lens: for {len(bags)} bags: BEGIN")
        if not self.build_missing_only:
            executables = [FeatureBagClip.FeatureBagLengthComputer(bag) for bag in bags]
            bag_lens = list(self.bag_executor_cls(n_workers=self.bag_n_workers, batch_size=self.bag_cpu_batch_size, log=self.log, tag='BipolarFeatureBagClip bag_lens').execute(executables))
        self.log.verbose(f"__build__: COMPUTING bag_lens: for {len(bags)} bags: END")
        write_npz(self.path('bag_lens', ensure_dirpath=True), bag_lens=bag_lens)
        return self

    def __read__(self, topic):
        return read_npz(self.path(topic), topic)[topic]

    @property
    def bags(self):
        if not hasattr(self, '_bags'):
            self.log.verbose(f"bags: FORMING BipolarFeatureBags: BEGIN")
            n_bags = self.cfg.probe.cfg.featurebagclip.n_bags
            executables = [BipolarFeatureBagMaker(self, idx) for idx in range(n_bags)]
            results = self.bag_executor_cls(n_workers=self.bag_n_workers, batch_size=self.bag_cpu_batch_size, log=self.log, tag='BipolarFeatureBagClip bag formation').execute(executables)
            self._bags = list(results)
            self.log.verbose(f"bags: FORMING BipolarFeatureBags: END")
        return self._bags

    def bag(self, idx: int):
        if hasattr(self, '_bags'):
            return self._bags[idx]
        bag = BipolarFeatureBag(
            root=self._root_,
            spec=dict(
                probe=self.cfg.probe,
                bag_index=idx,
                featurebag=self.cfg.probe.cfg.featurebagclip.bag(idx),
            ),
            tag=self.tag,
        )
        return bag

    def shard(self, idx: int):
        return self.bag(idx)
    
    @property
    def shards(self):
        return self.bags

    @functools.cached_property
    def bag_lens(self):
        return self.read('bag_lens')

    @property
    def n_bags(self):
        return len(self.bag_lens)

    @property
    def shard_lens(self):
        return self.bag_lens

    @property
    def n_shards(self):
        return self.n_bags

    def UNSAFE_clear_shards(self, OVERRIDE=False):
        if UNSAFE_allowed('UNSAFE_clear_shards', OVERRIDE=OVERRIDE):
            super().UNSAFE_clear_shards()
            if hasattr(self, '_bags'):
                del self._bags
            self.UNSAFE_clear(OVERRIDE=True)
        return self

    def UNSAFE_clear_bags(self, OVERRIDE=False):
        return self.UNSAFE_clear_shards(OVERRIDE=OVERRIDE)


class BipolarSingleFeatureBagClip(Clip):
    """Like BipolarFeatureBagClip, but contains exactly one bag at index `idx`
    from the given BipolarFeatureBagProbe."""

    from autopath.pancan.probes import BipolarFeatureBagProbe

    VERSION = 1

    TOPICFILES = {'bag_lens': 'bag_lens.npy'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        from autopath.pancan.probes import BipolarFeatureBagProbe
        probe: BipolarFeatureBagProbe
        idx: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __build__(self):
        idx = self.cfg.idx
        self.log.verbose(f"BUILDING single BipolarFeatureBag {repr(idx)} from {self.cfg.probe}: BEGIN")
        bag = self.bag(0)
        if not bag.valid():
            bag.__build__(probe=self.cfg.probe)
        bag_lens = [len(bag)]
        self.log.verbose(f"BUILDING single BipolarFeatureBag {repr(idx)} from {self.cfg.probe}: END")
        write_npz(self.path('bag_lens', ensure_dirpath=True), bag_lens=bag_lens)
        return self

    def __read__(self, topic):
        return read_npz(self.path(topic), topic)[topic]

    @property
    def bags(self):
        if not hasattr(self, '_bags'):
            self._bags = [self.bag(0)]
        return self._bags

    def bag(self, idx: int = 0):
        assert idx == 0, f"BipolarSingleFeatureBagClip contains exactly one bag, got {idx=}"
        if hasattr(self, '_bags'):
            return self._bags[0]
        real_idx = self.cfg.idx
        bag = BipolarFeatureBag(
            root=self._root_,
            spec=dict(
                probe=self.cfg.probe,
                bag_index=real_idx,
                featurebag=self.cfg.probe.cfg.featurebagclip.bag(real_idx),
            ),
            tag=self.tag,
        )
        return bag

    def shard(self, idx: int):
        return self.bag(idx)

    @property
    def shards(self):
        return self.bags

    @functools.cached_property
    def bag_lens(self):
        return self.read('bag_lens')

    @property
    def n_bags(self):
        return 1

    @property
    def shard_lens(self):
        return self.bag_lens

    @property
    def n_shards(self):
        return self.n_bags


class SphericalFeatureBagMaker:
    def __init__(self, clip, idx):
        self.clip = clip
        self.idx = idx
    def __call__(self):
        return dbx.eval(self.clip).bag(self.idx)
    def __repr__(self):
        return f"SphericalFeatureBagMaker({dbx.quote(self.clip)}, {self.idx})"


class SphericalFeatureBag(Bag):
    """A ``Bag`` that stores L2-normalised (unit-sphere) tile features.

    Each tile feature vector is divided by its L2 norm so that it lies
    on the unit hypersphere.  The raw (un-normalised) features remain
    accessible via the underlying ``FeatureBag``.

    Persisted topics:

    - ``features`` — L2-normalised tile feature matrix ``(n_tiles, d)``
    """

    TOPICFILES = {
        'features': 'features.npy',
    }

    @dataclass
    class CONFIG(Datablock.CONFIG):
        featurebag: FeatureBag

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __build__(self):
        self.log.detailed(f"SphericalFeatureBag: build: BEGIN")
        raw = self.cfg.featurebag.features           # (n_tiles, d)
        norms = raw.norm(dim=1, keepdim=True).clamp(min=1e-8)
        features = raw / norms
        dbx.write_tensor(features, self.path('features', ensure_dirpath=True))
        self._len = len(features)
        self.log.detailed(f"SphericalFeatureBag: build: END ({self._len} tiles)")
        return self

    def __read__(self, topic: str):
        if topic == 'features':
            result = dbx.read_tensor(self.path('features'))
        else:
            raise ValueError(f"Unknown {topic=}")
        return result

    @property
    def tensor(self):
        return self.features

    @functools.cached_property
    def features(self):
        return self.read('features')

    @property
    def name(self):
        return self.cfg.featurebag.name

    @property
    def label(self):
        return self.cfg.featurebag.label

    @functools.cached_property
    def labels(self):
        return self.cfg.featurebag.labels

    def __len__(self):
        if not hasattr(self, '_len'):
            if self.validtopic('features'):
                self._len = dbx.read_tensor(self.path('features')).shape[0]
            else:
                self._len = len(self.cfg.featurebag)
        return self._len


class SphericalFeatureBagClip(Clip):
    """A ``Clip`` of L2-normalised feature bags.

    Wraps a ``FeatureBagClip`` and materialises one ``SphericalFeatureBag``
    per source ``FeatureBag``.  Each bag independently L2-normalises its
    tile features during ``build()``.
    """

    VERSION = 1

    TOPICFILES = {'bag_lens': 'bag_lens.npy'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        featurebagclip: FeatureBagClip

    def __init__(self,
                 *args,
                 tag: str | None = None,
                 build_missing_only: bool = False,
                 cpu_parallelization: str | None = None,
                 n_workers: int = 1,
                 cpu_batch_size: int | None = None,
                 bag_cpu_parallelization: str | None = None,
                 bag_n_workers: int = 1,
                 bag_cpu_batch_size: int | None = None,
                 **kwargs):
        super().__init__(
            *args,
            n_workers=n_workers,
            build_missing_only=build_missing_only,
            cpu_parallelization=cpu_parallelization,
            cpu_batch_size=cpu_batch_size,
            tag=tag,
            **kwargs,
        )
        self.bag_n_workers = bag_n_workers
        self.bag_cpu_batch_size = bag_cpu_batch_size
        self.bag_cpu_parallelization = bag_cpu_parallelization
        self.executor_cls = get_executor_cls(cpu_parallelization)
        self.bag_executor_cls = get_executor_cls(bag_cpu_parallelization)
        self.builder_cls = {
            'ray': RayDatablocksBuilder,
            'multiprocessing': MultiprocessingDatablocksBuilder,
            'multithreading': MultithreadingDatablocksBuilder,
            'inline': InlineDatablocksBuilder,
        }[cpu_parallelization.lower() if cpu_parallelization is not None else 'inline']

    def __build__(self):
        self.log.verbose(f"__build__: BUILDING SphericalFeatureBags using "
                         f"{self.n_workers} {self.cpu_parallelization} workers: BEGIN")
        bags = self.bags
        missing_bags = []
        bag_lens = []
        if self.build_missing_only:
            if self.verbose:
                bagitor = tqdm.tqdm(bags, desc=f"{self.anchor}: LOOKING for missing SphericalFeatureBags")
            else:
                bagitor = bags
            for bag in bagitor:
                if not bag.valid():
                    missing_bags.append(bag)
                bag_lens.append(len(bag))
        else:
            missing_bags = bags

        self.log.verbose(f"__build__: BUILDING {len(missing_bags)} "
                         f"{'missing' if self.build_missing_only else 'all'} "
                         f"SphericalFeatureBags: BEGIN")
        if self.n_workers > 0:
            self.builder_cls(
                n_workers=self.n_workers,
                log=self.log,
                tag=f"{self.anchor}: __build__: BUILDING SphericalFeatureBags",
            ).build_blocks(missing_bags)
        else:
            if self.verbose:
                bagitor = tqdm.tqdm(missing_bags,
                                   desc=f"{self.anchor}: __build__: BUILDING SphericalFeatureBags")
            else:
                bagitor = missing_bags
            for bag in bagitor:
                bag.__build__()
        self.log.verbose(f"__build__: BUILDING SphericalFeatureBags: END")
        self.log.verbose(f"__build__: COMPUTING bag_lens for {len(bags)} bags: BEGIN")
        if not self.build_missing_only:
            executables = [FeatureBagClip.FeatureBagLengthComputer(bag) for bag in bags]
            bag_lens = list(self.bag_executor_cls(
                n_workers=self.bag_n_workers,
                batch_size=self.bag_cpu_batch_size,
                log=self.log,
                tag='SphericalFeatureBagClip bag_lens',
            ).execute(executables))
        self.log.verbose(f"__build__: COMPUTING bag_lens: END")
        write_npz(self.path('bag_lens', ensure_dirpath=True), bag_lens=bag_lens)
        return self

    def __read__(self, topic):
        return read_npz(self.path(topic), topic)[topic]

    @property
    def bags(self):
        if not hasattr(self, '_bags'):
            self.log.verbose("bags: FORMING SphericalFeatureBags: BEGIN")
            n_bags = self.cfg.featurebagclip.n_bags
            executables = [SphericalFeatureBagMaker(self, idx) for idx in range(n_bags)]
            results = self.bag_executor_cls(
                n_workers=self.bag_n_workers,
                batch_size=self.bag_cpu_batch_size,
                log=self.log,
                tag='SphericalFeatureBagClip bag formation',
            ).execute(executables)
            self._bags = list(results)
            self.log.verbose("bags: FORMING SphericalFeatureBags: END")
        return self._bags

    def bag(self, idx: int):
        if hasattr(self, '_bags'):
            return self._bags[idx]
        featurebag = self.cfg.featurebagclip.bag(idx)
        bag = SphericalFeatureBag(
            root=self._root_,
            spec=dict(featurebag=dbx.quote(featurebag)),
            tag=self.tag,
        )
        return bag

    def shard(self, idx: int):
        return self.bag(idx)

    @property
    def shards(self):
        return self.bags

    @functools.cached_property
    def bag_lens(self):
        return self.read('bag_lens')

    @property
    def n_bags(self):
        return self.cfg.featurebagclip.n_bags

    @property
    def shard_lens(self):
        return self.bag_lens

    @property
    def n_shards(self):
        return self.n_bags


class SpectralFeatureBagMaker:
    def __init__(self, clip, idx):
        self.clip = clip
        self.idx = idx
    def __call__(self):
        return dbx.eval(self.clip).bag(self.idx)
    def __repr__(self):
        return f"SpectralFeatureBagMaker({dbx.quote(self.clip)}, {self.idx})"


class SpectralFeatureBag(Bag):
    """A Bag that stores the Jacobian singular-value spectrum at probed layers.

    For each probed block, stores:
      - **cls mode**: ``singular_values`` (full d-dimensional spectrum)
      - **full mode**: ``top_singular_values`` and ``bottom_singular_values`` (k each)

    The extractor must be a ``SpectralBackboneEvaluator`` (or compatible).
    Topic files are named ``spectrum_block_{b}.npz`` for each probed block.
    """

    VERSION = 1

    @dataclass
    class CONFIG(Datablock.CONFIG):
        tilebag: TileBag
        extractor: Callable   # must be a SpectralBackboneEvaluator

    def __init__(self, *args, gpu_batch_size: int = 16, **kwargs):
        Datablock.__init__(self, *args, gpu_batch_size=gpu_batch_size, **kwargs)

    def __post_init__(self):
        self.TOPICFILES = {}
        for b in self.cfg.extractor.spectral_probe_blocks:
            self.TOPICFILES[f'spectrum_block_{b}'] = f'spectrum_block_{b}.npz'
        # Composed Jacobian topic (first → last probed block).
        # Always CLS-based, independent of per-block spectral_mode.
        if len(self.TOPICFILES) >= 2:
            self.TOPICFILES['spectrum_composed'] = 'spectrum_composed.npz'
        return self

    @property
    def spectral_mode(self):
        return self.cfg.extractor.spectral_mode

    @property
    def probe_blocks(self):
        return self.cfg.extractor.spectral_probe_blocks

    @property
    def name(self):
        return self.cfg.tilebag.name

    @property
    def label(self):
        return self.cfg.tilebag.label

    def __len__(self):
        return len(self.cfg.tilebag.tiles)

    def __build__(self, extractor=None):
        if extractor is None:
            extractor = self.cfg.extractor

        tilebag = self.cfg.tilebag
        n_tiles = len(tilebag.tiles)

        # Accumulate spectral results across GPU batches.
        # For each block we collect a list of per-batch result dicts.
        accumulated = {b: [] for b in extractor.spectral_probe_blocks}
        accumulated_composed = []

        for k in range(math.ceil(n_tiles / self.gpu_batch_size)):
            m = k * self.gpu_batch_size
            n = min((k + 1) * self.gpu_batch_size, n_tiles)
            batch = tilebag.tiles[m:n].to(self.device)
            self.log.verbose(f"Evaluating batch {k}: {m}:{n} out of {n_tiles} on device: {self.device}")

            # Forward pass — populates extractor.spectral_results
            _ = extractor(batch)

            spectral = extractor.spectral_results
            if spectral is not None:
                for b in extractor.spectral_probe_blocks:
                    if b in spectral:
                        accumulated[b].append(spectral[b])
                if 'composed' in spectral:
                    accumulated_composed.append(spectral['composed'])
            extractor.clear_spectral_results()
            extractor.clear_sideband()
            del batch
            gc.collect()
            torch.cuda.empty_cache()

        # Average the spectra across batches and write to disk
        for b in extractor.spectral_probe_blocks:
            entries = accumulated[b]
            if not entries:
                self.log.warning(f"No spectral results for block {b}")
                continue

            result = {}
            mode = extractor.spectral_mode

            if mode in ('cls', 'both'):
                sv_stack = np.stack([e['singular_values'] for e in entries])
                result['singular_values_mean'] = np.mean(sv_stack, axis=0)
                result['singular_values_std'] = np.std(sv_stack, axis=0)
                result['log_singular_values_mean'] = np.mean(
                    np.log(np.clip(sv_stack, 1e-12, None)), axis=0
                )
                result['n_batches'] = np.array(len(entries))

            if mode in ('full', 'both'):
                top_stack = np.stack([e['top_singular_values'] for e in entries])
                bot_stack = np.stack([e['bottom_singular_values'] for e in entries])
                result['top_singular_values_mean'] = np.mean(top_stack, axis=0)
                result['top_singular_values_std'] = np.std(top_stack, axis=0)
                result['bottom_singular_values_mean'] = np.mean(bot_stack, axis=0)
                result['bottom_singular_values_std'] = np.std(bot_stack, axis=0)
                result['n_batches'] = np.array(len(entries))

            write_npz(self.path(f'spectrum_block_{b}', ensure_dirpath=True), **result)

        # Write composed spectrum
        if accumulated_composed:
            sv_stack = np.stack([e['singular_values'] for e in accumulated_composed])
            composed_result = {
                'singular_values_mean': np.mean(sv_stack, axis=0),
                'singular_values_std': np.std(sv_stack, axis=0),
                'log_singular_values_mean': np.mean(
                    np.log(np.clip(sv_stack, 1e-12, None)), axis=0
                ),
                'n_batches': np.array(len(accumulated_composed)),
            }
            write_npz(self.path('spectrum_composed', ensure_dirpath=True), **composed_result)

        self._len = n_tiles
        return self

    def __read__(self, topic: str):
        keys = list(self.TOPICFILES.keys())
        if topic not in keys:
            raise ValueError(f"Unknown {topic=}, expected one of {keys}")
        path = self.path(topic)
        # read_npz requires explicit keys; discover them from the npz first
        import fsspec as _fsspec
        _fs, _ = _fsspec.url_to_fs(path)
        with _fs.open(path, 'rb') as f:
            npz_keys = list(np.load(f, allow_pickle=True).keys())
        return read_npz(path, *npz_keys)

    def spectrum(self, block_idx: int):
        """Read the stored spectrum for a given block index."""
        return self.read(f'spectrum_block_{block_idx}')

    @functools.cached_property
    def labels(self):
        return list(zip(self.cfg.tilebag.labels, self.cfg.tilebag.tiles))


class SpectralFeatureBagClip(Clip):
    """A Clip of SpectralFeatureBags — one per tile bag in a tile-bag clip.

    Builds each SpectralFeatureBag using GPU parallelization (same pattern
    as FeatureBagClip), storing the Jacobian singular-value spectra at
    probed layers for every bag in the clip.
    """

    VERSION = 1
    TOPICFILES = {"bag_lens": "bag_lens.npy"}

    @dataclass
    class CONFIG:
        tilebagclip: Clip
        extractor: Callable   # must be a SpectralBackboneEvaluator

    def __init__(self, *, tag: str | None = None, n_workers: int = 1,
                 devices: list[str] = ["cuda"],
                 gpu_batch_size: int = 16,
                 cpu_batch_size: int | None = None,
                 skip_unreadable: bool = True,
                 gpu_parallelization: str = 'multithreading',
                 cpu_parallelization: str | None = None,
                 bag_n_workers: int = 1,
                 bag_cpu_batch_size: int | None = None,
                 bag_cpu_parallelization: str | None = None,
                 **kwargs):
        super().__init__(
            n_workers=n_workers, devices=devices,
            gpu_batch_size=gpu_batch_size, cpu_batch_size=cpu_batch_size,
            skip_unreadable=skip_unreadable,
            gpu_parallelization=gpu_parallelization,
            cpu_parallelization=cpu_parallelization,
            tag=tag, **kwargs,
        )
        self.bag_n_workers = bag_n_workers
        self.bag_cpu_batch_size = bag_cpu_batch_size
        self.bag_cpu_parallelization = bag_cpu_parallelization
        self.executor_cls = get_executor_cls(cpu_parallelization)
        self.bag_executor_cls = get_executor_cls(bag_cpu_parallelization)
        self.builder_cls = {
            'multiprocessing': TorchMultiprocessingDatablocksBuilder,
            'multithreading': TorchMultithreadingDatablocksBuilder,
        }[gpu_parallelization.lower()]

    def __build__(self):
        bags = self.bags
        self.log.verbose(f"Formed {len(bags)} SpectralFeatureBags.  Looking for missing bags.")
        missing_bags = [bag for bag in bags if not bag.valid()]
        self.log.verbose(f"Found {len(missing_bags)} missing bags")
        self.log.verbose(f"Building {len(missing_bags)} missing spectral bags using devices {self.devices} and gpu_batch_size {self.gpu_batch_size}")
        built_bags = self.builder_cls(devices=self.devices, log=self.log).build_blocks(missing_bags, self.cfg.extractor)
        self.log.verbose(f"Built {len(built_bags)} missing spectral bags")
        self.log.verbose(f"Building {len(bags)} bag_lens: BEGIN")
        executor = self.executor_cls(n_workers=self.n_workers, batch_size=self.cpu_batch_size, log=self.log)
        executables = [FeatureBagClip.FeatureBagLengthComputer(bag) for bag in bags]
        bag_lens_list = executor.execute(executables)
        bag_lens = torch.tensor(bag_lens_list)
        self.log.verbose(f"Building {len(bags)} bag_lens: END")
        dbx.write_tensor(bag_lens, self.path("bag_lens", ensure_dirpath=True))
        return self

    def __read__(self, topic):
        if topic == "bag_lens":
            return dbx.read_tensor(self.path("bag_lens"))
        raise ValueError(f"Unknown {topic=}")

    @property
    def bags(self):
        if not hasattr(self, '_bags'):
            self.log.verbose(f"FORMING SpectralFeatureBags: BEGIN")
            n_bags = self.cfg.tilebagclip.n_bags
            executables = [SpectralFeatureBagMaker(self, idx) for idx in range(n_bags)]
            results = self.bag_executor_cls(
                n_workers=self.bag_n_workers, log=self.log,
                batch_size=self.bag_cpu_batch_size,
                tag='SpectralFeatureBagClip bag formation',
            ).execute(executables)
            self._bags = list(results)
            self.log.verbose(f"FORMING SpectralFeatureBags: END")
        return self._bags

    def bag(self, idx: int):
        if hasattr(self, '_bags'):
            return self._bags[idx]
        tilebag = self.cfg.tilebagclip.bag(idx)
        bag = SpectralFeatureBag(
            root=self._root_,
            spec=dict(tilebag=dbx.quote(tilebag), extractor=self.spec['extractor']),
            gpu_batch_size=self.gpu_batch_size,
            revision=self.revision,
            tag=self.tag,
        )
        return bag

    def shard(self, idx: int):
        return self.bag(idx)

    @property
    def shards(self):
        return self.bags

    @property
    def n_bags(self):
        return self.cfg.tilebagclip.n_bags

    @property
    def n_shards(self):
        return self.n_bags

    @functools.cached_property
    def bag_lens(self):
        return self.read("bag_lens")

    @property
    def shard_lens(self):
        return self.bag_lens


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2 — Deep Feature Bags & Clips
# ═══════════════════════════════════════════════════════════════════════

class DeepFeatureBagMaker:
    def __init__(self, clip, idx):
        self.clip = clip
        self.idx = idx
    def __call__(self):
        return dbx.eval(self.clip).bag(self.idx)
    def __repr__(self):
        return f"DeepFeatureBagMaker({dbx.quote(self.clip)}, {self.idx})"


class DeepFeatureBag(Bag):
    """A Bag that stores multi-layer activations from a DeepBackboneEvaluator.

    For each configured capture layer the evaluator produces, a separate
    tensor file ``features_{layer_name}.npy`` is written.  The bag also
    records the ordered list of layer names so consumers can discover
    which layers are available.

    Properties
    ----------
    tensor(layer) : Tensor
        Read a single layer's features (n_tiles, dim).
    tensors() : dict[str, Tensor]
        Read all layers.
    label : str
        Delegated to the source TileBag.
    """

    VERSION = 1

    @dataclass
    class CONFIG(Datablock.CONFIG):
        tilebag: TileBag
        evaluator_factory: object  # DeepBackboneEvaluatorFactory
        batch_size: int = 64

    def __init__(self, *args, gpu_batch_size: int = 64, **kwargs):
        Datablock.__init__(self, *args, gpu_batch_size=gpu_batch_size, **kwargs)

    def __post_init__(self):
        # Build topic files from factory's capture config.
        factory = self.cfg.evaluator_factory
        if hasattr(factory, 'evaluator'):
            layer_names = factory.evaluator(log=self.log).layer_names
        elif hasattr(factory, 'cfg') and hasattr(factory.cfg, 'capture_transformer_blocks'):
            # Reconstruct names from config without loading model.
            layer_names = []
            for b in factory.cfg.capture_transformer_blocks:
                for lyr in factory.cfg.capture_layers:
                    layer_names.append(f"B_{b}_{lyr}")
            if factory.cfg.capture_model_layers:
                layer_names.extend(factory.cfg.capture_model_layers)
        else:
            layer_names = []

        self._layer_names = layer_names
        self.TOPICFILES = {
            'layer_names': 'layer_names.npz',
        }
        for lyr in layer_names:
            safe_name = lyr.replace(".", "_")
            self.TOPICFILES[f'features_{safe_name}'] = f'features_{safe_name}.npy'
        return self

    @property
    def name(self):
        return self.cfg.tilebag.name

    @property
    def label(self):
        return self.cfg.tilebag.label

    @property
    def layer_names(self):
        return list(self._layer_names)

    def __len__(self):
        return len(self.cfg.tilebag.tiles)

    def __build__(self, evaluator=None):
        if evaluator is None:
            evaluator = self.cfg.evaluator_factory.evaluator(
                device=self.device, log=self.log
            )

        tilebag = self.cfg.tilebag
        n_tiles = len(tilebag.tiles)
        layer_names = evaluator.layer_names
        self._layer_names = layer_names

        # Accumulate per-layer feature lists.
        accumulated = {lyr: [] for lyr in layer_names}

        for k in range(math.ceil(n_tiles / self.gpu_batch_size)):
            m = k * self.gpu_batch_size
            n = min((k + 1) * self.gpu_batch_size, n_tiles)
            batch = tilebag.tiles[m:n].to(self.device)
            self.log.verbose(
                f"Evaluating batch {k}: {m}:{n} out of {n_tiles} "
                f"on device: {self.device}"
            )

            result = evaluator(batch)

            for lyr in layer_names:
                if lyr in result:
                    accumulated[lyr].append(result[lyr].cpu().detach())

            evaluator.clear()
            del batch, result
            gc.collect()
            torch.cuda.empty_cache()

        # Concatenate and write per-layer features.
        write_npz(
            self.path('layer_names', ensure_dirpath=True),
            layer_names=layer_names,
        )

        for lyr in layer_names:
            if accumulated[lyr]:
                features = torch.cat(accumulated[lyr], dim=0)
                safe_name = lyr.replace(".", "_")
                write_tensor(
                    features,
                    self.path(f'features_{safe_name}', ensure_dirpath=True),
                )
                self.log.verbose(
                    f"Wrote features for layer '{lyr}': {features.shape}"
                )
                del features
            else:
                self.log.warning(f"No features captured for layer '{lyr}'")
            gc.collect()
            torch.cuda.empty_cache()

        del accumulated
        gc.collect()
        self._len = n_tiles
        return self

    def __read__(self, topic):
        if topic == 'layer_names':
            return list(
                read_npz(self.path('layer_names'), 'layer_names')['layer_names']
            )
        # Feature topics: features_{safe_name}
        path = self.path(topic)
        return read_tensor(path)

    def tensor(self, layer: str = None):
        """Read features for a single layer.

        Parameters
        ----------
        layer : str or None
            Capture key (e.g. ``"B_0_norm1"``).  If ``None``, returns
            the backbone output (topic ``"output"``).
        """
        if layer is None:
            layer = "output"
        safe_name = layer.replace(".", "_")
        return self.read(f'features_{safe_name}')

    def tensors(self) -> dict:
        """Read all layer features as a dict."""
        return {lyr: self.tensor(lyr) for lyr in self.layer_names}

    @functools.cached_property
    def features(self):
        """Default features — the backbone output."""
        return self.tensor("output")

    @functools.cached_property
    def labels(self):
        return list(zip(self.cfg.tilebag.labels, self.cfg.tilebag.tiles))


class DeepFeatureClip(Datastack):
    """A clip of :class:`DeepFeatureBag` shards built in parallel via Datastack.

    Orchestrates building one :class:`DeepFeatureBag` per tile-bag in the
    source ``tilebagclip``, using the configured
    :class:`DeepBackboneEvaluatorFactory`.
    """

    VERSION = 1

    @dataclass
    class CONFIG(Datablock.CONFIG):
        tilebagclip: Clip
        evaluator_factory: object  # DeepBackboneEvaluatorFactory
        batch_size: int = 64

    def __init__(self, *args, gpu_batch_size: int = 64,
                 devices: list = None, **kwargs):
        self._devices = devices or ["cuda"]
        super().__init__(*args, gpu_batch_size=gpu_batch_size, **kwargs)

    @property
    def n_shards(self) -> int:
        return self.cfg.tilebagclip.n_shards

    def __shard__(self, idx: int):
        tilebag = self.cfg.tilebagclip.shard(idx)
        return DeepFeatureBag(
            root=self._root_,
            spec=dict(
                tilebag=dbx.quote(tilebag),
                evaluator_factory=self.spec['evaluator_factory'],
                batch_size=self.cfg.batch_size,
            ),
            gpu_batch_size=self.gpu_batch_size,
            revision=self.revision,
            tag=self.tag,
        )

    @property
    def bags(self):
        return self.shards()

    @property
    def n_bags(self):
        return self.n_shards

    def bag(self, idx: int):
        return self.shard(idx)


# ═══════════════════════════════════════════════════════════════════════
#  Phase 3 — Spherical & Corner Deep Feature Bags
# ═══════════════════════════════════════════════════════════════════════

class SphericalDeepFeatureBag(Bag):
    """L2-normalised view of a :class:`DeepFeatureBag`.

    Each layer's feature tensor is projected onto the unit hypersphere
    (L2-normalised along the feature dimension).  This is a pure
    runtime transformation — no separate build step is required.
    """

    VERSION = 1

    @dataclass
    class CONFIG(Datablock.CONFIG):
        deep_feature_bag: DeepFeatureBag

    def __init__(self, *args, **kwargs):
        Datablock.__init__(self, *args, **kwargs)

    def __post_init__(self):
        self.TOPICFILES = dict(self.cfg.deep_feature_bag.TOPICFILES)
        return self

    @property
    def name(self):
        return self.cfg.deep_feature_bag.name

    @property
    def label(self):
        return self.cfg.deep_feature_bag.label

    @property
    def layer_names(self):
        return self.cfg.deep_feature_bag.layer_names

    def __len__(self):
        return len(self.cfg.deep_feature_bag)

    def valid(self):
        return self.cfg.deep_feature_bag.valid()

    def tensor(self, layer: str = None):
        """L2-normalised features for a single layer."""
        t = self.cfg.deep_feature_bag.tensor(layer)
        return torch.nn.functional.normalize(t.float(), p=2, dim=-1)

    def tensors(self) -> dict:
        return {lyr: self.tensor(lyr) for lyr in self.layer_names}

    @functools.cached_property
    def features(self):
        return self.tensor("output")

    @functools.cached_property
    def labels(self):
        return self.cfg.deep_feature_bag.labels


class SphericalDeepFeatureClip:
    """Runtime wrapper that yields :class:`SphericalDeepFeatureBag` views
    over a :class:`DeepFeatureClip`.

    No separate build step — delegates storage to the underlying clip.
    """

    def __init__(self, deep_feature_clip: DeepFeatureClip):
        self._clip = deep_feature_clip

    @property
    def n_shards(self):
        return self._clip.n_shards

    @property
    def n_bags(self):
        return self.n_shards

    def bag(self, idx: int):
        dfb = self._clip.bag(idx)
        return SphericalDeepFeatureBag(
            root=dfb._root_,
            spec=dict(deep_feature_bag=dbx.quote(dfb)),
            tag=dfb.tag,
        )

    def shard(self, idx: int):
        return self.bag(idx)

    @property
    def bags(self):
        return [self.bag(i) for i in range(self.n_bags)]


class CornerDeepFeatureBag(Bag):
    """Bipolar (hypercube-vertex) encoding of a :class:`DeepFeatureBag`.

    For each layer, maps each tile's feature vector to the nearest vertex
    of the unit hypercube ``{-1, +1}^d`` using the sign of each
    coordinate.  This generalises the existing ``BipolarFeatureBag``
    concept to multi-layer deep features without requiring a median
    probe.

    Pure runtime transformation — no separate build step required.
    """

    VERSION = 1

    @dataclass
    class CONFIG(Datablock.CONFIG):
        deep_feature_bag: DeepFeatureBag

    def __init__(self, *args, **kwargs):
        Datablock.__init__(self, *args, **kwargs)

    def __post_init__(self):
        self.TOPICFILES = dict(self.cfg.deep_feature_bag.TOPICFILES)
        return self

    @property
    def name(self):
        return self.cfg.deep_feature_bag.name

    @property
    def label(self):
        return self.cfg.deep_feature_bag.label

    @property
    def layer_names(self):
        return self.cfg.deep_feature_bag.layer_names

    def __len__(self):
        return len(self.cfg.deep_feature_bag)

    def valid(self):
        return self.cfg.deep_feature_bag.valid()

    def tensor(self, layer: str = None):
        """Bipolar-encoded features for a single layer: sign(x) ∈ {-1, +1}^d."""
        t = self.cfg.deep_feature_bag.tensor(layer)
        return torch.sign(t)

    def corner_ids(self, layer: str = None, max_dims: int = 63):
        """Integer encoding of hypercube corners.

        Maps ``{-1, +1}^d → int`` by treating positive coordinates as
        binary 1-bits.  Truncated to *max_dims* dimensions to avoid
        integer overflow.

        Parameters
        ----------
        layer : str or None
            Capture key.
        max_dims : int
            Maximum number of dimensions to use (default 63, safe for
            int64).

        Returns
        -------
        Tensor
            Integer corner IDs, shape ``(n_tiles,)``.
        """
        signs = (self.cfg.deep_feature_bag.tensor(layer) > 0).long()
        d = min(signs.shape[-1], max_dims)
        powers = 2 ** torch.arange(d)
        return (signs[..., :d] * powers).sum(dim=-1)

    def tensors(self) -> dict:
        return {lyr: self.tensor(lyr) for lyr in self.layer_names}

    @functools.cached_property
    def features(self):
        return self.tensor("output")

    @functools.cached_property
    def labels(self):
        return self.cfg.deep_feature_bag.labels


class CornerDeepFeatureClip:
    """Runtime wrapper that yields :class:`CornerDeepFeatureBag` views
    over a :class:`DeepFeatureClip`.

    No separate build step — delegates storage to the underlying clip.
    """

    def __init__(self, deep_feature_clip: DeepFeatureClip):
        self._clip = deep_feature_clip

    @property
    def n_shards(self):
        return self._clip.n_shards

    @property
    def n_bags(self):
        return self.n_shards

    def bag(self, idx: int):
        dfb = self._clip.bag(idx)
        return CornerDeepFeatureBag(
            root=dfb._root_,
            spec=dict(deep_feature_bag=dbx.quote(dfb)),
            tag=dfb.tag,
        )

    def shard(self, idx: int):
        return self.bag(idx)

    @property
    def bags(self):
        return [self.bag(i) for i in range(self.n_bags)]

