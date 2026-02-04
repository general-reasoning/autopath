import copy
from dataclasses import dataclass
import functools
import gc
import itertools
import math
import traceback as tb
from typing import Callable

import tqdm
import numpy as np


import torch

import dbx
from dbx import (
    Datablock, 
    MultithreadingCallableExecutor,
    RemoteDatablocksBuilder,
    write_npz,
    read_npz,
)

from autopath.databits import Shard, Bag, Clip, ClipDataset
from .tiles import TileBag


def tensors_to_device(tensors, device, *, detach: bool = False):
    _tensors = {k: v.to(device) for k, v in tensors.items()}
    if detach:
        _tensors = {k: v.detach() for k, v in _tensors.items()} 
    return _tensors


def cat_tensor_dicts(tensor_dicts):
    tensors = {k: [] for k in tensor_dicts[0].keys()}
    for tensor_dict in tensor_dicts:
        for k, v in tensor_dict.items():
            tensors[k].append(v)
    _tensors = {k: torch.cat(v) for k, v in tensors.items()}
    return _tensors

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
        return self.cfg.tilebag.label, 

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
        self.log.selected(f"Reading features from {self.path('features')}: BEGIN")
        features = self.read('features')
        self.log.selected(f"Reading features from {self.path('features')}: END")
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
        self.log.selected(f"Assemblying labels from {self.cfg.tilebag}: BEGIN")
        labels = list(zip(self.cfg.tilebag.labels, self.cfg.tilebag.tiles))
        self.log.selected(f"Assemblying labels from {self.cfg.tilebag}: END")
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
    
    def __init__(self, devices: list[str] = ["cuda"], gpu_batch_size: int = 16, skip_unreadable: bool = True, **kwargs):
        super().__init__(devices=devices, gpu_batch_size=gpu_batch_size, skip_unreadable=skip_unreadable, **kwargs)
        self.log.debug(f"devices={self.devices}, gpu_batch_size={self.gpu_batch_size}, skip_unreadable={self.skip_unreadable}")

    def __build__(self):
        bags = self.bags
        self.log.verbose(f"Formed {len(bags)} FeatureBags.  Looking for missing bags.")
        missing_bags = [bag for bag in bags if not bag.valid()]
        self.log.verbose(f"Found {len(missing_bags)} missing bags")
        self.log.verbose(f"Building all missing features bags using devices {self.devices} and gpu_batch_size {self.gpu_batch_size}")
        built_bags = dbx.TorchMultithreadingDatablocksBuilder(devices=self.devices, log=self.log).build_blocks(missing_bags, self.cfg.extractor)
        self.log.verbose(f"Built all missing features shards: {len(built_bags)}")
        self.log.verbose(f"Building bag_lens: BEGIN")
        self.log.detailed(f"Building bag_lens for bags with hashe paths {[bag.hashpath() for bag in bags]}")
        executor = MultithreadingCallableExecutor(n_threads=len(self.devices))
        executables = [FeatureBagClip.FeatureBagLengthComputer(bag) for bag in bags]
        bag_lens_lists = executor.exec_callables(executables)
        bag_lens = torch.tensor(list(itertools.chain.from_iterable(bag_lens_lists)))
        self.log.debug(f"bag_lens: {bag_lens}")
        self.log.verbose(f"Building bag_lens: END")
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
    
    @functools.cached_property
    def bags(self):
        self.log.selected(f"bags: traceback:\n{''.join(tb.format_stack())}")
        self.log.verbose(f"FORMING FeatureBags from tilebagclip {self.cfg.tilebagclip} with revision {self.revision}: BEGIN ")
        if self.verbose:
            tilebagitor = tqdm.tqdm(self.cfg.tilebagclip.shards, desc=f"{self.anchor}: FORMING FeatureBags")
        else:
            tilebagitor = self.cfg.tilebagclip.shards
        bags = [
            FeatureBag(
                root=self._root_, 
                spec=dict(tilebag=dbx.quote(tilebag), extractor=self.spec['extractor'],), 
                gpu_batch_size=self.gpu_batch_size,
                revision=self.revision,
            )
            for tilebag in tilebagitor
        ]
        self.log.verbose(f"FORMING FeatureBags from tilebagclip {self.cfg.tilebagclip} with revision {self.revision}: END")
        return bags

    def bag(self, idx: int):
        tilebag = self.cfg.tilebagclip.bag(idx)
        bag = FeatureBag(
            root=self._root_, 
            spec=dict(tilebag=dbx.quote(tilebag), extractor=self.spec['extractor'],),
            gpu_batch_size=self.gpu_batch_size,
            revision=self.revision,
        )
        return bag

    def shard(self, idx: int):
        return self.bag(idx)
    
    @property
    def shards(self):
        return self.bags
    
    @property
    def n_bags(self):
        try:
            length = len(self.read('bag_lens'))
        except Exception as e:
            self.log.debug(f"Encountered exception {e} when reading 'bag_lens'. Instantiating bags to obtain their len.")
            length = len(self.bags)
        return length
    
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
     

def featurebagset(featurebagclip: FeatureBagClip, *, transform=None, bags_shuffle_seed: int = None):
    return ClipDataset(spec=dict(clip=featurebagclip, transform=transform, shuffle_seed=bags_shuffle_seed))


class FeatureShard(Shard):
    VERSION = 1
    TOPICFILES = {'features': 'features.npy', 'labels': 'labels.npy', 'tiles': 'tiles.npy'}
    
    @dataclass
    class CONFIG(Datablock.CONFIG):
        featureset: torch.utils.data.Dataset 
        shard_size: int
        index: int = 0
        seed: int = 42

    def __build__(self, featureset=None):
        if featureset is None:
            featureset = self.cfg.featureset
        dataset_len = len(self.cfg.featureset)
        rng = np.random.default_rng(self.cfg.seed)
        permutation = rng.permutation(dataset_len)
        offset = self.cfg.index * self.cfg.shard_size
        lo, hi = offset, offset+self.cfg.shard_size
        indices = permutation[lo:hi]
        self.log.debug(f"Building a shard of size {self.cfg.shard_size} with index {self.cfg.index} from a dataset of len {dataset_len}")
        self.log.detailed(f"Shard indices:{lo=}:{hi=}, {indices=}")
        _tensor_list, labels = zip(*[self.cfg.featureset[i] for i in indices])
        _labels_list, _tiles_list = zip(*labels)
        tensor = torch.stack(_tensor_list)
        labels = np.stack(_labels_list)
        tiles = np.stack(_tiles_list)
        dbx.write_tensor(tensor, self.path('features', ensure_dirpath=True))
        dbx.write_npz(self.path('labels', ensure_dirpath=True), labels=labels)
        dbx.write_npz(self.path('tiles', ensure_dirpath=True), tiles=tiles)
        self.log.debug(f"BUILT -------------------------> shard of size {self.cfg.shard_size} with index {self.cfg.index} from a dataset of len {dataset_len}")
        return self
    
    def __read__(self, topic):
        if topic == 'features':
            return dbx.read_tensor(self.path('features'))   
        elif topic == 'labels':
            return dbx.read_npz(self.path('labels'), 'labels')['labels']
        elif topic == 'tiles':
            return torch.tensor(dbx.read_npz(self.path('tiles'), 'tiles')['tiles'])
        else:
            raise ValueError(f"Unknown {topic=}")
    
    @property
    def tensor(self):
        return self.read('features')
    
    @property
    def features(self):
        return self.read('features')
    
    @property
    def labels(self):
        return self.read('labels')
    
    @property
    def tiles(self):
        return self.read('tiles')
    

class FeatureShardClip(Clip):
    VERSION = 1
    TOPICFILES = {'shard_lens': 'shard_lens.npy'}

    @dataclass
    class CONFIG(Datablock.CONFIG):
        featureset: torch.utils.data.Dataset 
        shard_size: int
        seed: int = 42

    class FeatureShardLengthComputer:
        def __init__(self, featureshard):
            self.featureshard = featureshard
        def __call__(self):
            return len(self.featureshard)
        def __repr__(self):
            return f"FeatureShardLengthComputer({self.featureshard})"

    def __init__(self, *args, n_threads: int = 1, **kwargs):
        super().__init__(*args, n_threads=n_threads, **kwargs)

    @functools.cached_property
    def shards(self):
        featureset = self.cfg.featureset
        featuresetlen = len(featureset)
        shard_size = self.cfg.shard_size
        seed = self.cfg.seed
        n_shards = math.ceil(featuresetlen / shard_size)
        self.log.detailed(f"Forming FeatureShards from featureset of len {featuresetlen} with shard_size {shard_size} and seed {seed}: {n_shards} shards")
        shards = [
            FeatureShard(
                spec=dict(
                    featureset=featureset,
                    shard_size=shard_size,
                    seed=seed,
                    index=i,
                )
            )
            for i in range(n_shards)
        ]
        return shards
    
    def __build__(self):
        shards = self.shards
        self.log.debug(f"Formed {len(shards)} FeatureShards.  Looking for missing shards.")
        missing_shards = [shard for shard in shards if not shard.valid()]
        self.log.debug(f"Found {len(missing_shards)} missing shards")
        self.log.debug(f"Building all missing features shards using  {self.n_threads} threads")
        built_shards = dbx.MultithreadingDatablocksBuilder(n_threads=self.n_threads, log=self.log).build_blocks(missing_shards, self.cfg.featureset)
        self.log.verbose(f"Built all missing features shards: {len(built_shards)}")
        self.log.verbose(f"Building shard_lens: BEGIN")
        self.log.detailed(f"Building shard_lens for shards with hashe paths {[shard.hashpath() for shard in shards]}")
        executor = MultithreadingCallableExecutor(n_threads=self.n_threads)
        executables = [FeatureShardClip.FeatureShardLengthComputer(shard) for shard in shards]
        _shard_lens = executor.exec_callables(executables)
        shard_lens = torch.tensor(_shard_lens)
        self.log.debug(f"shard_lens: {shard_lens}")
        self.log.verbose(f"Building shard_lens: END")
        dbx.write_tensor(shard_lens, self.path("shard_lens", ensure_dirpath=True))
        return self


def featureshardset(featureshardclip: FeatureShardClip, *, transform=None):
    return ClipDataset(spec=dict(clip=featureshardclip, transform=transform))


class BipolarFeatureBag(Bag):
    from autopath.pancan.probes import BipolarFeatureBagProbe

    VERSION = 1
    TOPICFILES = {
        'bipolar_features': 'bipolar_features.npy',
    }

    @dataclass
    class CONFIG(Datablock.CONFIG):
        probehandle: str
        bag_index: int
        featurebag: FeatureBag

    def __build__(self, *, probe: BipolarFeatureBagProbe):
        assert self.cfg.probehandle == probe.handle(), \
            f"Handle mismatch: {self.cfg.probehandle} != {probe.handle}"
        assert self.cfg.featurebag.handle() == probe.cfg.featurebagclip.bags[self.cfg.bag_index].handle(), \
            f"Featurebag handle mismatch: {self.cfg.featurebag.handle()} != {probe.cfg.featurebagclip.bags[self.cfg.bag_index].handle()}"
        self.log.detailed(f" BipolarFeatureBag {self.cfg.bag_index}: build: BEGIN")
        all_features = probe.tile_bipolar_features
        lo = probe.bag_bounds[self.cfg.bag_index]
        hi = probe.bag_bounds[self.cfg.bag_index+1]
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
        self.log.selected(f"Reading features from {self.cfg.featurebag}: BEGIN")
        result = self.cfg.featurebag.features
        self.log.selected(f"Reading features from {self.cfg.featurebag}: END")
        return result

    @property
    def name(self):
        return self.cfg.featurebag.name

    @functools.cached_property    
    def labels(self):
        self.log.selected(f"Reading labels from {self.cfg.featurebag}: BEGIN")
        result = self.cfg.featurebag.labels
        self.log.selected(f"Reading labels from {self.cfg.featurebag}: END")
        return result

    def __len__(self):
        if not hasattr(self, '_len'):
            self._len = len(self.cfg.featurebag)
        return self._len


class BipolarFeatureBagClip(Clip):
    VERSION = 2

    TOPICFILES = {'bag_lens': 'bag_lens.npy'}
    
    @dataclass
    class CONFIG(Datablock.CONFIG):
        from autopath.pancan.probes import BipolarFeatureBagProbe
        probe: BipolarFeatureBagProbe

    def __init__(self, *args, n_workers: int = 1, build_missing_only: bool = False, **kwargs):
        super().__init__(*args, n_workers=n_workers, build_missing_only=build_missing_only, **kwargs)

    def __build__(self):
        self.log.verbose(f"BUILDING BipolarFeatureBags from {self.cfg.probe} using {self.n_workers} processes: BEGIN")
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

        self.log.verbose(f"BUILDING {len(missing_bags)} {'missing' if self.build_missing_only else 'all'} BipolarFeatureBags using {self.n_workers} processes: BEGIN")
        if self.n_workers > 0:
            missing_blocks = RemoteDatablocksBuilder(log=self.log).build_blocks(missing_bags, probe=self.cfg.probe)
        else:   
            if self.verbose:
                bagitor = tqdm.tqdm(missing_bags, desc=f"{self.anchor}: BUILDING BipolarFeatureBags")
            else:
                bagitor = missing_bags
            for bag in bagitor:
                bag.__build__(probe=self.cfg.probe)
        self.log.verbose(f"BUILDING {len(missing_bags)} {'missing' if self.build_missing_only else 'all'} BipolarFeatureBags using {self.n_workers} processes: END")
        self.log.verbose(f"COMPUTING bag_lens: for {len(bags)} bags: BEGIN")
        if not self.build_missing_only:
            if self.verbose:
                    bagitor = tqdm.tqdm(bags, desc=f"{self.anchor}: COMPUTING bag_lens")
            else:
                bagitor = bags
            bag_lens = [len(bag) for bag in bagitor]
        self.log.verbose(f"COMPUTING bag_lens: for {len(bags)} bags: END")
        write_npz(self.path('bag_lens', ensure_dirpath=True), bag_lens=bag_lens)
        return self

    def __read__(self, topic):
        return read_npz(self.path(topic), topic)[topic]

    @functools.cached_property
    def bags(self):
        self.log.silent(f"traceback:\n{''.join(tb.format_stack())}")
        n_bags = self.cfg.probe.cfg.featurebagclip.n_bags
        if self.verbose:
            bagitor = tqdm.tqdm(range(n_bags), desc=f"{self.anchor}: FORMING BipolarFeatureBags")
        else:
            bagitor = range(n_bags)
        bags = [
            BipolarFeatureBag(
                root=self._root_,
                spec=dict(
                    probehandle=self.cfg.probe.handle(),
                    bag_index=i,
                    featurebag=self.cfg.probe.cfg.featurebagclip.bags[i],
                )
            )
            for i in bagitor
        ]
        return bags

    def bag(self, idx: int):
        bag = BipolarFeatureBag(
            root=self._root_,
            spec=dict(
                probehandle=self.cfg.probe.handle(),
                bag_index=idx,
                featurebag=self.cfg.probe.cfg.featurebagclip.bag(idx),
            )
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
    




        




        
