from dataclasses import dataclass
import functools
import math
import traceback as tb
from typing import Optional, Union

import tqdm

import numpy as np
import torch
import torchvision


import dbx
from dbx import Datablock


class Shard(Datablock):
    TOPICFILES = {'index': '', 'tensor': '', 'labels': None}
    @dataclass
    class CONFIG(Datablock.CONFIG):
        ...	

    def __read__(self, topic):
        if topic == 'index':
            result = np.load(self.path(topic))['arr_0']
        elif topic == 'tensor':
            result = self.tiles
        elif topic == 'labels':
            result = self.labels
        else:
            raise ValueError(f"Unknown topic: {topic}")
        return result
    
    def __len__(self):
        return len(self.tensor)
    
    def size(self):
        return len(self)

    @functools.cached_property
    def tensor(self):
        raise NotImplementedError

    @functools.cached_property
    def labels(self):
        raise NotImplementedError
    

class Bag(Shard):
    def __init__(self, *args, **kwargs):
        Shard.__init__(self, *args, **kwargs)
    
    @property
    def name(self):
        raise NotImplementedError

    @property
    def label(self):
        raise NotImplementedError

    @property
    @functools.cached_property
    def labels(self):
        return [self.label]*len(self)

    
class Clip(Datablock):
    TOPICFILE = "shard_lens.npz"
    def __len__(self):
        return len(self.shards)
    
    def __build__(self):
        self.log.verbose(f"Obtaining shard lens")
        if self.verbose:
            shardsitor = tqdm.tqdm(self.shards)
        else:
            shardsitor = self.shards
        shard_lens = [len(shard) for shard in shardsitor]
        dbx.write_npz(self.path(), shard_lens=shard_lens)
        return self
    
    def __read__(self):
        shard_lens = dbx.read_npz(self.path(), 'shard_lens')[0]
        return shard_lens

    def shard(self, idx: int):
        raise NotImplementedError

    @functools.cached_property
    def shards(self):
        raise NotImplementedError

    @functools.cached_property
    def shards_lens(self):
        return self.read()
    
    @property
    def n_shards(self):
        return len(self.shards_lens)
    
    def UNSAFE_clear_shards(self):
        for shard in self.shards:
            try:
                if shard.valid():
                    shard.UNSAFE_clear()
            except:
                pass
        return self

    def UNSAFE_copy_from(self, anchorpath: str, shard_anchorpath: str, *, overwrite: bool = False):
        super().UNSAFE_copy_from(anchorpath, overwrite=overwrite)
        self.UNSAFE_copy_shards_from(shard_anchorpath, overwrite=overwrite)
        return self

    def UNSAFE_copy_shards_from(self, shard_anchorpath: str, *, overwrite: bool = False):
        for shard in self.shards:
            shard.UNSAFE_copy_from(shard_anchorpath, overwrite=overwrite)
        return self
    

class Split(Datablock):
    TOPICFILES = {"train_shard_indices": "train_shard_indices.pt", 
                  "train_shard_lens":    "train_shard_lens.pt",
                  "test_shard_indices":  "test_shard_indices.pt",
                  "test_shard_lens":     "test_shard_lens.pt",
    }
    @dataclass
    class CONFIG:
        clip: Clip
        train_fraction: float = 0.8
        seed: int = 42

    def __build__(self):
        self.log.info(f"Building splits out of {len(self.cfg.clip.shards)} shards using train fraction {self.cfg.train_fraction}")
        N = len(self.cfg.clip.shards)
        K = int(math.ceil(N*self.cfg.train_fraction))
        np.random.seed(self.cfg.seed) #TODO: localize in a generator
        perm = np.random.permutation(N)
        train_shard_indices = torch.tensor(perm[:K])
        test_shard_indices = torch.tensor(perm[K:])
        self.log.verbose(f"Computing train shard lens")
        if self.verbose:
            train_shard_itor = tqdm.tqdm(train_shard_indices)
        else:
            train_shard_itor = train_shard_indices
        train_shard_lens = torch.tensor([len(self.cfg.clip.shards[i].dataset) for i in train_shard_itor])
        self.log.verbose(f"Computing test shard lens")
        if self.verbose:
            test_shard_itor = tqdm.tqdm(test_shard_indices)
        else:
            test_shard_itor = test_shard_indices
        test_shard_lens = torch.tensor([len(self.cfg.clip.shards[i].dataset) for i in test_shard_itor])
        dbx.write_tensor(train_shard_indices, self.path('train_shard_indices', ensure_dirpath=True),)
        dbx.write_tensor(train_shard_lens, self.path('train_shard_lens', ensure_dirpath=True),)
        dbx.write_tensor(test_shard_indices, self.path('test_shard_indices', ensure_dirpath=True),)
        dbx.write_tensor(test_shard_lens, self.path('test_shard_lens', ensure_dirpath=True),)
        return self
    
    def __read__(self, topic):
        tensor = dbx.read_tensor(self.path(topic))
        return tensor

    def shards(self, split):
        return [self.cfg.clip.shards[i] for i in self.shard_indices(split)]
    
    def shard_lens(self, split):
        return self.read(f"{split}_shard_lens")

    def shard_indices(self, split):
        return self.read(f"{split}_shard_indices")

    def shard(self, split, idx: int):
        return self.cfg.clip.shards[self.shard_indices(split)[idx]]


class Partition(Datablock):
    TOPICFILES = {"shard_indices": "shard_indices.npz", 
                  "shard_lens":    "shard_lens.npz",
    }
    @dataclass
    class CONFIG:
        clip: Clip
        fold_fractions: list[float]
        seed: int = 42

    def __post_init__(self):
        assert sum(self.cfg.fold_fractions) == 1.0, f"Fold fractions must sum to 1.0, got {self.cfg.fold_fractions}"
        return self

    def __build__(self):
        self.log.info(f"Building partition out of {len(self.cfg.clip.shards)} shards using fold fractions {self.cfg.fold_fractions}")
        N = len(self.cfg.clip.shards)
        np.random.seed(self.cfg.seed) #TODO: localize in a generator
        perm = np.random.permutation(N)

        shard_indices = {}
        shard_lens = {}
        Klo = 0
        self.log.verbose(f"Computing shard indices and lens for {len(self.cfg.fold_fractions)} folds: BEGIN")
        if self.verbose:
            fold_fraction_itor = tqdm.tqdm(self.cfg.fold_fractions)
        else:
            fold_fraction_itor = self.cfg.fold_fractions
        for fold, fraction in enumerate(fold_fraction_itor):
            k = int(math.ceil(N*fraction))
            Khi = min(Klo + k, N)
            fold_key = str(fold)
            shard_indices[fold_key] = perm[Klo:Khi]
            self.log.verbose(f"Computing shard lens for fold {fold}: BEGIN")
            if self.verbose:
                shard_itor = tqdm.tqdm(shard_indices[fold_key])
            else:
                shard_itor = shard_indices[fold_key]
            shard_lens[fold_key] = np.array([len(self.cfg.clip.shards[i].dataset) for i in shard_itor])
            self.log.verbose(f"Computing shard lens for fold {fold}: END")
            Klo = Khi
        self.log.verbose(f"Computing shard indices and lens for {len(self.cfg.fold_fractions)} folds: END")
        self.log.verbose(f"Writing shard indices and lens: BEGIN")
        dbx.write_npz(self.path('shard_indices', ensure_dirpath=True), **shard_indices)
        dbx.write_npz(self.path('shard_lens', ensure_dirpath=True), **shard_lens)
        self.log.verbose(f"Writing shard indices and lens: END")
        return self
    
    def __read__(self, topic):
        tensor = dbx.read_npz(self.path(topic), topic)
        return tensor

    def shards(self, fold):
        return [self.cfg.clip.shards[i] for i in self.shard_indices(fold)]
    
    def shard_lens(self, fold):
        return self.read("shard_lens")[fold]

    def shard_indices(self, fold):
        return self.read("shard_indices")[fold]

    def shard(self, fold, idx: int):
        return self.cfg.clip.shards[self.shard_indices(fold)[idx]]


class Fold(Clip):
    @dataclass
    class CONFIG:
        split: Union[Split, Partition]
        fold: Union[str, int]

    def __post_init__(self):
        return self
    
    def valid(self):
        return self.cfg.split.valid()

    @functools.cached_property
    def shards(self):
        return self.cfg.split.shards(self.cfg.fold)

    def shard(self, idx: int):
        return self.cfg.split.shard(self.cfg.fold, idx)

    @functools.cached_property
    def shard_lens(self):
        return self.cfg.split.shard_lens(self.cfg.fold)
    
            
class ClipDataset(Datablock, torch.utils.data.Dataset):
    @dataclass
    class CONFIG:
        clip: Clip
        transform: Optional[torchvision.transforms.Compose] = None
        target_transform: Optional[torchvision.transforms.Compose] = None
        shuffle_seed: Optional[int] = None

    def __post_init__(self):
        self.n_shards = self.cfg.clip.n_shards
        self.log.debug(f"INITIALIZING dataset using {self.n_shards} shards from clip {self.cfg.clip}: BEGIN")
        self.log.silent(f"traceback:\n{''.join(tb.format_stack())}")
        self._shard_indices = np.arange(self.n_shards)
        if self.cfg.shuffle_seed is not None:
            self.log.verbose(f"Shuffling shard indices with seed {self.cfg.shuffle_seed}")
            rng = np.random.default_rng(self.cfg.shuffle_seed)
            rng.shuffle(self._shard_indices)
        self.shard_lens = [self.cfg.clip.shard_lens[i] for i in self._shard_indices]
        self.log.debug(f"Computing shard_bounds...")
        self.shard_bounds = np.cumsum(self.shard_lens)
        self.log.debug(f"Computing shard_bounds... DONE")
        self.log.detailed(f"{self.n_shards=}, {self.shard_bounds=}")
        self._shard_idx = None
        self._shard = None
        self._shard_label = None
        self._shard_slide = None
        self.log.debug(f"INITIALIZING dataset using {self.n_shards} shards from clip {self.cfg.clip}: END")

    @functools.lru_cache(maxsize=3)
    def shard(self, shard_idx):
        if shard_idx != self._shard_idx:
            self._shard_idx = shard_idx
            self._shard = self.cfg.clip.shard(self._shard_indices[self._shard_idx])
        return self._shard

    def __len__(self):
        return self.shard_bounds[-1]

    def __getitem__(self, index):
        self.log.silent(f"GETTING item {index} from dataset")
        shard_idx = np.searchsorted(self.shard_bounds, index, side='right')
        shard_lo = self.shard_bounds[shard_idx-1] if shard_idx > 0 else 0
        shard_hi = self.shard_bounds[shard_idx]
        shard_len = self.shard_lens[shard_idx]
        idx = index - shard_lo
        self.log.silent(f"----------------------> {index=}, {shard_idx=}, {shard_lo=}, {shard_len=}, {shard_hi=}, {idx=}")
        tensor = self.shard(shard_idx).tensor
        sample = tensor[idx]
        if self.cfg.transform is not None:
            sample = self.cfg.transform(sample)
        labels = self.shard(shard_idx).labels
        label = labels[idx]
        self.log.silent(f"APPLYING target_transform")
        if self.cfg.target_transform is not None:
            label = self.cfg.target_transform(label)
        return sample, label
    

class ClipDataLoaderBuilder(Datablock):
    @dataclass
    class CONFIG:
        clip_dataset: ClipDataset
        batch_size: int
        shuffle: bool = False

    def __init__(self, spec: dict, *, dataloader_kwargs):
        Datablock.__init__(self, spec=spec, dataloader_kwargs=dataloader_kwargs)

    def __post_init__(self):
        self.dataloader_kwargs['shuffle'] = self.spec['shuffle']
        self.dataloader_kwargs['batch_size'] = self.spec['batch_size']

    def dataloader(self):
        self.log.debug(f"Initializing ClipDataLoaderBuilder dataloader with kwargs: {self.dataloader_kwargs}")
        return torch.utils.data.DataLoader(dataset=self.cfg.clip_dataset, **self.dataloader_kwargs)

