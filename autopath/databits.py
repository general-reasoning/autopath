
from dataclasses import dataclass
import functools
import math
import traceback as tb
from typing import Union

import tqdm

import numpy as np
import torch
import torchvision


import dbx
from dbx import (
    Datablock,
    MultithreadingDatablocksBuilder,
    MultiprocessingDatablocksBuilder,
    RemoteDatablocksBuilder,
)


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
    def shard_lens(self):
        return self.read()
    
    @property
    def n_shards(self):
        return len(self.shard_lens)
    
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
        keys = [f"{i}" for i in range(len(self.cfg.fold_fractions))]
        dict = dbx.read_npz(self.path(topic), *keys)
        return dict

    def shards(self, fold):
        return [self.cfg.clip.shards[i] for i in self.shard_indices(fold)]
    
    def shard_lens(self, fold):
        return self.read("shard_lens")[fold]

    def shard_indices(self, fold):
        return self.read("shard_indices")[fold]

    def shard(self, fold, idx: int):
        return self.cfg.clip.shards[self.shard_indices(fold)[idx]]

    def n_shards(self, fold):
        return len(self.shard_indices(fold))


class Fold(Clip):
    @dataclass
    class CONFIG:
        partition: Partition
        fold: Union[str, int]

    def __post_init__(self):
        return self
    
    def valid(self):
        return self.cfg.partition.valid()

    @functools.cached_property
    def shards(self):
        return self.cfg.partition.shards(self.cfg.fold)

    def shard(self, idx: int):
        return self.cfg.split.shard(self.cfg.fold, idx)

    @functools.cached_property
    def shard_lens(self):
        return self.cfg.partition.shard_lens(self.cfg.fold)
    
    def shard(self, idx: int):
        return self.cfg.partition.shard(self.cfg.fold, idx)

    @functools.cached_property
    def n_shards(self):
        return self.cfg.partition.n_shards(self.cfg.fold)



class ShuffledShard(Shard):
    """A ``Shard`` that holds a fixed subset of samples drawn from a ``Clip``.

    ``__build__`` instantiates ``ClipDatasetBuilder(cfg.clip).dataset()``,
    fetches the assigned sample indices, and persists the resulting tensors
    and labels as ``tensor.npy`` / ``labels.npy``.

    All ``Shard`` methods (``tensor``, ``labels``, ``__len__``) are served
    directly from those on-disk files after the first build.
    """

    TOPICFILES = {'tensor': 'tensor.npz', 'labels': 'labels.npz'}

    @dataclass
    class CONFIG(Shard.CONFIG):
        clip: Clip
        indices: list[int]      # global sample indices assigned to this shard

    # ------------------------------------------------------------------
    # Build / read
    # ------------------------------------------------------------------

    def __build__(self):
        self.log.verbose(f"fetching {len(self.cfg.indices)} samples: BEGIN")
        dataset = ClipDatasetBuilder(
            root=self._root_,
            spec=dict(clip=self.spec['clip']),
        ).dataset()
        samples = [dataset[i] for i in self.cfg.indices]
        self.log.verbose(f"fetching {len(self.cfg.indices)} samples: END")
        tensor = torch.stack([s[0] for s in samples])
        labels = np.array([s[1] for s in samples])
        self.log.verbose(f"writing tensor and labels: BEGIN")
        dbx.write_tensors(self.path('tensor', ensure_dirpath=True), tensor=tensor)
        dbx.write_npz(self.path('labels', ensure_dirpath=True), labels=labels)
        self.log.verbose(f"writing tensor and labels: END")
        return self

    def __read__(self, topic):
        if topic == 'tensor':
            return dbx.read_tensors(self.path('tensor'), 'tensor')['tensor']
        if topic == 'labels':
            return dbx.read_npz(self.path('labels'), 'labels')['labels']
        raise ValueError(f"Unknown topic: {topic!r}")

    # ------------------------------------------------------------------
    # Shard interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cfg.indices)

    @functools.cached_property
    def tensor(self):
        return self.read('tensor')

    @functools.cached_property
    def labels(self):
        return self.read('labels').tolist()


class ShuffledClip(Clip):
    """A ``Clip`` that draws a random subset of samples from an existing
    ``Clip`` and re-shards them into fixed-size ``ShuffledShard`` blocks.

    CONFIG parameters:

    - ``clip``         — source ``Clip`` to sample from.
    - ``fraction``     — fraction of the source dataset to keep (0 < fraction ≤ 1).
    - ``shuffle_seed`` — seed for the sample-index permutation.
    - ``shard_size``   — number of samples per output ``ShuffledShard``.

    Typical use::

        sc = ShuffledClip(
            spec=dict(
                clip=dbx.quote(my_clip),
                fraction=0.8,
                shuffle_seed=42,
                shard_size=512,
            ),
            n_workers=8,
        )
        sc.build()
    """

    @dataclass
    class CONFIG:
        clip: Clip
        fraction: float
        shard_size: int
        shuffle_seed: int = 42

    PARALLELIZERS = {
        'MultithreadingDatablocksBuilder':  MultithreadingDatablocksBuilder,
        'MultiprocessingDatablocksBuilder': MultiprocessingDatablocksBuilder,
        'RemoteDatablocksBuilder':          RemoteDatablocksBuilder,
    }

    def __init__(
        self,
        *args,
        n_workers: int = 1,
        parallelizer: str = 'MultithreadingDatablocksBuilder',
        **kwargs,
    ):
        if parallelizer not in self.PARALLELIZERS:
            raise ValueError(
                f"Unknown parallelizer {repr(parallelizer)}. "
                f"Choose one of: {list(self.PARALLELIZERS)}"
            )
        super().__init__(*args, n_workers=n_workers, **kwargs)
        self.parallelizer = parallelizer

    def __post_init__(self):
        assert 0 < self.cfg.fraction <= 1.0, (
            f"fraction must be in (0, 1], got {self.cfg.fraction}"
        )
        return self

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _shuffled_indices(self) -> list[int]:
        """Return the full list of sampled global indices (Python ints)."""
        dataset = ClipDatasetBuilder(
            root=self._root_,
            spec=dict(clip=self.spec['clip']),
        ).dataset()
        N = len(dataset)
        n_samples = int(math.floor(N * self.cfg.fraction))
        rng = np.random.default_rng(self.cfg.shuffle_seed)
        perm = rng.permutation(N)[:n_samples]
        return perm.tolist()   # plain list[int] — survives YAML serialisation

    def _make_shards(self, indices: list[int]) -> list[ShuffledShard]:
        """Partition *indices* into chunks and return the ``ShuffledShard`` list."""
        s = self.cfg.shard_size
        chunks = [indices[i:i + s] for i in range(0, len(indices), s)]
        return [
            ShuffledShard(
                root=self._root_,
                spec=dict(
                    clip=self.spec['clip'],
                    indices=chunk,
                ),
                revision=self.revision,
            )
            for idx, chunk in enumerate(chunks)
        ]

    # ------------------------------------------------------------------
    # Clip interface
    # ------------------------------------------------------------------

    @functools.cached_property
    def shards(self) -> list[ShuffledShard]:
        return self._make_shards(self._shuffled_indices())

    def shard(self, idx: int) -> ShuffledShard:
        return self.shards[idx]

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def __build__(self):
        all_shards = self.shards
        missing = [s for s in all_shards if not s.valid()]
        self.log.info(
            f"{len(all_shards)} shards total, "
            f"{len(missing)} need building, n_workers={self.n_workers}"
        )
        if missing:
            self.log.info(f"building {len(missing)} shards: BEGIN")
            if self.n_workers > 1:
                builder_cls = self.PARALLELIZERS[self.parallelizer]
                builder_cls(n_workers=self.n_workers, log=self.log).build_blocks(missing)
            else:
                for shard in missing:
                    shard.build()
            self.log.info(f"building {len(missing)} shards: END")
        # Write the shard_lens file expected by Clip
        super().__build__()
        return self


class ClipDatasetBuilder(Datablock):
    @dataclass
    class CONFIG:
        clip: Clip
        transform: torchvision.transforms.Compose | None = None
        target_transform: torchvision.transforms.Compose | None = None
        shuffle_seed: int | None = None

    def __post_init__(self):
        self.n_shards = self.cfg.clip.n_shards
        self.log.debug(f"INITIALIZING dataset using {self.n_shards} shards from clip {self.cfg.clip}: BEGIN")
        self.log.silent(f"traceback:\n{''.join(tb.format_stack())}")
        _shard_indices = np.arange(self.n_shards)
        if self.cfg.shuffle_seed is not None:
            self.log.verbose(f"Shuffling shard indices with seed {self.cfg.shuffle_seed}")
            rng = np.random.default_rng(self.cfg.shuffle_seed)
            rng.shuffle(_shard_indices)
        self._shard_indices = [int(i) for i in _shard_indices]
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

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, builder):
            self.dataset_builder = builder
        
        def __len__(self):
            return len(self.dataset_builder)
        
        def __getitem__(self, index):
            self.dataset_builder.log.silent(f"GETTING item {index} from dataset")
            shard_idx = np.searchsorted(self.dataset_builder.shard_bounds, index, side='right')
            shard_lo = self.dataset_builder.shard_bounds[shard_idx-1] if shard_idx > 0 else 0
            shard_hi = self.dataset_builder.shard_bounds[shard_idx]
            shard_len = self.dataset_builder.shard_lens[shard_idx]
            idx = index - shard_lo
            self.dataset_builder.log.silent(f"----------------------> {index=}, {shard_idx=}, {shard_lo=}, {shard_len=}, {shard_hi=}, {idx=}")
            tensor = self.dataset_builder.shard(shard_idx).tensor
            sample = tensor[idx]
            if self.dataset_builder.cfg.transform is not None:
                sample = self.dataset_builder.cfg.transform(sample)
            labels = self.dataset_builder.shard(shard_idx).labels
            label = labels[idx]
            self.dataset_builder.log.silent(f"APPLYING target_transform")
            if self.dataset_builder.cfg.target_transform is not None:
                label = self.dataset_builder.cfg.target_transform(label)
            return sample, label
    
    def dataset(self):
        return self.Dataset(self)
    

class ClipDataLoaderBuilder(Datablock):
    @dataclass
    class CONFIG:
        clip_dataset_builder: ClipDatasetBuilder
        batch_size: int
        shuffle: bool = False

    def __init__(self, spec: dict, *, dataloader_kwargs):
        Datablock.__init__(self, spec=spec, dataloader_kwargs=dataloader_kwargs)
        self.dataset = self.cfg.clip_dataset_builder.dataset()

    def __post_init__(self):
        self.dataloader_kwargs['shuffle'] = self.spec['shuffle']
        self.dataloader_kwargs['batch_size'] = self.spec['batch_size']

    def dataloader(self):
        self.log.debug(f"Initializing ClipDataLoaderBuilder dataloader with kwargs: {self.dataloader_kwargs}")
        return torch.utils.data.DataLoader(dataset=self.dataset, **self.dataloader_kwargs)

