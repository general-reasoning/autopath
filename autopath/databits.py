import collections
from dataclasses import dataclass
import functools
import gc
import math
import traceback as tb
from typing import Union

import tqdm

import numpy as np
import ray
import torch
import torchvision


import dbx
from dbx import (
    Datablock,
    MultithreadingCallableExecutor,
    MultithreadingDatablocksBuilder,
    MultiprocessingCallableExecutor,
    MultiprocessingDatablocksBuilder,
    RayCallableExecutor,
    RayDatablocksBuilder,
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
        return sum(self.shard_lens)
    
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



class ClipShardQueue:
    """
    A queue of shards from a Clip.
    """
    def __init__(self, queue_idx: int, clip_quote, clip_lens, fraction: float, n_shards: int, n_queues: int, shard_size: int, shuffle_seed: int, p_inv_ref, target_to_srcs, log=dbx.Logger()):
        self.log = log
        self.log.debug(f"[Queue {queue_idx}] Initializing actor...")
        self.queue_idx = queue_idx
        # Instantiate a fresh Clip object from the quote.
        # This keeps serialization lightweight while preserving the clip's configuration.
        self.clip = clip_quote()
        self.clip_lens = clip_lens
        self.fraction = fraction
        self.n_shards = n_shards
        self.n_queues = n_queues
        self.shard_size = shard_size
        self.shuffle_seed = shuffle_seed
        self.p_inv_ref = p_inv_ref
        self.target_to_srcs = target_to_srcs
        self._p_inv = None # Lazy fetch from Ray Object Store

        # Partition source shards among queues
        self.my_src_shards = list(range(self.queue_idx, len(self.clip_lens), self.n_queues))
        
        self.offsets = np.cumsum([0] + list(self.clip_lens))
        self.current_src_ptr = 0  # Index into self.my_src_shards
        self.buffers = collections.defaultdict(list)

    @property
    def p_inv(self):
        if self._p_inv is None:
            self.log.debug(f"[Queue {self.queue_idx}] Fetching p_inv from Ray Object Store...")
            self._p_inv = ray.get(self.p_inv_ref)
            self.log.debug(f"[Queue {self.queue_idx}] p_inv fetch complete.")
        return self._p_inv

    def _read_next(self):
        if self.current_src_ptr >= len(self.my_src_shards):
            return
        
        src_idx = self.my_src_shards[self.current_src_ptr]
        self.log.debug(f"[Queue {self.queue_idx}] Reading source shard {src_idx}...")
        shard = self.clip.shards[src_idx]
        tensor = shard.tensor
        labels = shard.labels
        
        offset = self.offsets[src_idx]
        _p_inv = self.p_inv # Fetch once from Object Store
        for i in range(len(tensor)):
            p = offset + i
            k = _p_inv[p]
            if k != -1:
                target_idx = k // self.shard_size
                # Store (data, label, k, global_p)
                self.buffers[target_idx].append((tensor[i], labels[i], k, p))
        
        self.current_src_ptr += 1

    def pull(self, target_shard_idx: int):
        needed = self.target_to_srcs.get(target_shard_idx, set())
        
        # Advance current_src_ptr if any needed shards are not yet read.
        # Since we read in order, we just need to check the maximum needed src_idx.
        if needed:
            max_needed = max(needed)
            while self.current_src_ptr < len(self.my_src_shards) and self.my_src_shards[self.current_src_ptr] <= max_needed:
                self._read_next()
        
        return self.buffers.pop(target_shard_idx, [])


class ShuffledShard(Shard):
    """A ``Shard`` that holds a fixed subset of samples drawn from a ``Clip``.

    ``__build__`` accepts pre-assembled tensors, labels, and indices,
    saving them as ``tensor.npz``, ``labels.npz``, and ``index.npz``.
    """

    TOPICFILES = {'tensor': 'tensor.npz', 'labels': 'labels.npz', 'index': 'index.npz'}

    @dataclass
    class CONFIG(Shard.CONFIG):
        clip: Clip
        fraction: float
        shard_size: int
        shuffle_seed: int
        shard_idx: int

    # ------------------------------------------------------------------
    # Build / read
    # ------------------------------------------------------------------

    def __build__(self, tensor=None, labels=None, indices=None):
        if tensor is None or labels is None or indices is None:
            raise ValueError("ShuffledShard.__build__ requires tensor, labels, and indices")

        self.log.verbose(f"writing shard {self.cfg.shard_idx} ({len(tensor)} samples): BEGIN")
        dbx.write_tensors(self.path('tensor', ensure_dirpath=True), tensor=tensor)
        dbx.write_npz(self.path('labels', ensure_dirpath=True), labels=labels)
        dbx.write_npz(self.path('index', ensure_dirpath=True), index=indices)
        self.log.verbose(f"writing shard {self.cfg.shard_idx}: END")
        return self

    def __read__(self, topic):
        if topic == 'tensor':
            return dbx.read_tensors(self.path('tensor'), 'tensor')['tensor']
        if topic == 'labels':
            return dbx.read_npz(self.path('labels'), 'labels')['labels']
        if topic == 'index':
            return dbx.read_npz(self.path('index'), 'index')['index']
        raise ValueError(f"Unknown topic: {topic!r}")

    # ------------------------------------------------------------------
    # Shard interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    @functools.cached_property
    def tensor(self):
        return self.read('tensor')

    @functools.cached_property
    def labels(self):
        return self.read('labels').tolist()

    @functools.cached_property
    def index(self):
        return self.read('index')


class ShuffledShardMaker:
    def __init__(self, root, *, clip, fraction, shard_size, shuffle_seed, shard_idx, queues, shuffled_shard_cls=ShuffledShard, verbose=False, build=True):
        self.root = root
        self.clip = clip
        self.fraction = fraction
        self.shard_size = shard_size
        self.shuffle_seed = shuffle_seed
        self.shard_idx = shard_idx
        self.queues = queues
        self.shuffled_shard_cls = shuffled_shard_cls
        self.verbose = verbose
        self.build = build

    def __call__(self):
        # Poll queues in random order
        n_queues = len(self.queues)
        q_indices = list(range(n_queues))
        rng = np.random.default_rng(self.shuffle_seed + self.shard_idx)
        rng.shuffle(q_indices)

        all_results = []
        if self.verbose:
            q_itor = tqdm.tqdm(q_indices, desc=f"Shard {self.shard_idx} assembly", leave=False)
        else:
            q_itor = q_indices

        for q_idx in q_itor:
            res = ray.get(self.queues[q_idx].pull.remote(self.shard_idx))
            all_results.extend(res)

        # Reassemble in deterministic order (by k)
        all_results.sort(key=lambda x: x[2])

        if not all_results:
            tensor = torch.empty(0)
            labels = np.empty(0)
            indices = np.empty(0, dtype=int)
        else:
            first_data = all_results[0][0]
            if isinstance(first_data, torch.Tensor):
                tensor = torch.stack([x[0] for x in all_results])
            else:
                tensor = np.array([x[0] for x in all_results])
            
            labels = np.array([x[1] for x in all_results], dtype=object)
            indices = np.array([x[3] for x in all_results], dtype=int)

        spec = dict(
            clip=self.clip,
            fraction=self.fraction,
            shard_size=self.shard_size,
            shuffle_seed=self.shuffle_seed,
            shard_idx=self.shard_idx
        )
        shard = self.shuffled_shard_cls(root=self.root, spec=spec)
        if self.build:
            shard.build(tensor=tensor, labels=labels, indices=indices)
            del shard
            gc.collect()
        else:
            return shard


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

    VERBOSE_CONFIG = True

    @dataclass
    class CONFIG:
        clip: Clip
        fraction: float
        shard_size: int
        shuffle_seed: int = 42
        n_queues: int = 1
        shuffled_shard_cls: type[ShuffledShard] = ShuffledShard

    PARALLELIZERS = {
        'Multithreading':  {'callable': MultithreadingCallableExecutor, 'datablock': MultithreadingDatablocksBuilder},
        'Multiprocessing': {'callable': MultiprocessingCallableExecutor, 'datablock': MultiprocessingDatablocksBuilder},
        'Ray':             {'callable': RayCallableExecutor, 'datablock': RayDatablocksBuilder},
    }

    def __init__(
        self,
        *args,
        n_workers: int = 1,
        parallelization: str | None = 'Ray',
        **kwargs,
    ):
        super().__init__(*args, n_workers=n_workers, parallelization=parallelization, **kwargs)

    def __post_init__(self):
        assert 0 < self.cfg.fraction <= 1.0, (
            f"fraction must be in (0, 1], got {self.cfg.fraction}"
        )
        assert self.parallelization == 'Ray', f"ShuffledClip currently only supports 'Ray' parallelization, got {repr(self.parallelization)}"
        return self
       
    # ------------------------------------------------------------------
    # Clip interface
    # ------------------------------------------------------------------

    @functools.cached_property
    def shards(self) -> list[ShuffledShard]:
        return self.__make_shards__()

    def shard(self, idx: int) -> ShuffledShard:
        return self.shards[idx]

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def __build__(self):
        self.log.verbose(f"Building ShuffledClip with {self.cfg.n_queues} queues: BEGIN")
        
        # Access shard_lens early to trigger any I/O
        shard_lens = self.cfg.clip.shard_lens
        N = len(self.cfg.clip)
        n_samples = int(math.floor(N * self.cfg.fraction))
        n_shards = int(math.ceil(n_samples / self.cfg.shard_size))

        # ------------------------------------------------------------------
        # Driver-side precomputation (Done BEFORE Ray setup to avoid contention)
        # ------------------------------------------------------------------
        self.log.info("Driver-side precomputation: BEGIN")
        offsets = np.cumsum([0] + list(shard_lens))
        
        self.log.info(f"Generating permutation for {N} samples: BEGIN")
        rng = np.random.default_rng(self.cfg.shuffle_seed)
        perm = rng.permutation(N)[:n_samples]
        self.log.info(f"Generating permutation for {N} samples: END")
        
        self.log.info("Generating inverse permutation map: BEGIN")
        p_inv = np.full(N, -1, dtype=np.int32)
        p_inv[perm] = np.arange(n_samples, dtype=np.int32)
        self.log.info("Generating inverse permutation map: END")
        
        self.log.info("Precomputing pull schedules for queues: BEGIN")
        queue_schedules = [collections.defaultdict(set) for _ in range(self.cfg.n_queues)]
        
        # Vectorized schedule calculation
        valid_indices = np.where(p_inv != -1)[0]
        if valid_indices.size > 0:
            v_dest_ks = p_inv[valid_indices]
            v_target_idxs = v_dest_ks // self.cfg.shard_size
            v_src_idxs = np.searchsorted(offsets, valid_indices, side='right') - 1
            
            combined = (v_src_idxs.astype(np.int64) << 32) | v_target_idxs.astype(np.int64)
            unique_combined = np.unique(combined)
            
            for val in unique_combined:
                s_idx = int(val >> 32)
                t_idx = int(val & 0xFFFFFFFF)
                q_idx = s_idx % self.cfg.n_queues
                queue_schedules[q_idx][t_idx].add(s_idx)
        self.log.info("Precomputing pull schedules for queues: END")
        self.log.info("Driver-side precomputation: END")

        # ------------------------------------------------------------------
        # Ray Setup
        # ------------------------------------------------------------------
        self.log.info(f"Setting up {self.cfg.n_queues} ClipShardQueues: BEGIN")
        if not ray.is_initialized():
            self.log.debug("Initializing Ray session...")
            ray.init(ignore_reinit_error=True)
            self.log.debug("Ray initialization returned.")

        import time
        time.sleep(1) # Give the system a second to stabilize

        self.log.debug("Broadcasting p_inv to Ray Object Store...")
        p_inv_ref = ray.put(p_inv)
        del p_inv
        gc.collect()
        self.log.debug("Broadcasting complete.")

        # Convert schedules to standard dicts with numpy arrays for faster serialization
        self.log.debug("Serializing schedules for actors...")
        serialized_schedules = []
        for sched in queue_schedules:
            s = {int(k): np.array(list(v), dtype=np.int32) for k, v in sched.items()}
            serialized_schedules.append(s)
        del queue_schedules
        self.log.debug("Serialization complete.")

        clip_quote = dbx.quote(self.cfg.clip)

        # ------------------------------------------------------------------
        # Actor Setup
        # ------------------------------------------------------------------
        
        q_itor = range(self.cfg.n_queues)
        if self.verbose:
            q_itor = tqdm.tqdm(q_itor, desc="Launching ClipShardQueues")
        
        RemoteQueue = ray.remote(ClipShardQueue)
        queues = []
        for i in q_itor:
            self.log.debug(f"Launching actor {i}...")
            queues.append(
                RemoteQueue.remote(
                    queue_idx=i,
                    clip_quote=clip_quote,
                    clip_lens=shard_lens,
                    fraction=self.cfg.fraction,
                    n_shards=n_shards,
                    n_queues=self.cfg.n_queues,
                    shard_size=self.cfg.shard_size,
                    shuffle_seed=self.cfg.shuffle_seed,
                    p_inv_ref=p_inv_ref,
                    target_to_srcs=serialized_schedules[i],
                    log=self.log
                )
            )
        self.log.debug("All actors launched.")
        self.log.info(f"Setting up {self.cfg.n_queues} ClipShardQueues: END")

        self.log.info(f"Setting up {n_shards} ShuffledShardMakers: BEGIN")
        shards_itor = range(n_shards)
        if self.verbose:
            shards_itor = tqdm.tqdm(shards_itor, desc="Setting up ShuffledShardMakers")
        
        # Consistent clip reference for makers
        clip_quoted = dbx.quote(self.cfg.clip)
        
        makers = [
            ShuffledShardMaker(
                root=self._root_,
                clip=clip_quoted,
                fraction=self.cfg.fraction,
                shard_size=self.cfg.shard_size,
                shuffle_seed=self.cfg.shuffle_seed,
                shard_idx=idx,
                queues=queues,
                shuffled_shard_cls=self.cfg.shuffled_shard_cls,
                verbose=self.verbose,
                build=True,
            )
            for idx in shards_itor
        ]
        self.log.info(f"Setting up {n_shards} ShuffledShardMakers: END")

        self.log.info(f"Building ShuffledClip shards: BEGIN")
        if self.parallelization is None or self.n_workers < 2:
            if self.verbose:
                makers_itor = tqdm.tqdm(makers, desc="Building ShuffledClip shards")
            else:
                makers_itor = makers
            [maker() for maker in makers_itor]
        else:
            self.PARALLELIZERS[self.parallelization]['callable'](n_workers=self.n_workers, log=self.log).execute(makers, verbose=self.verbose)

        super().__build__()
        self.log.verbose(f"Building ShuffledClip: END")
        return self

    def __make_shards__(self, build: bool = False):
        if build:
            return self.build().shards

        N = len(self.cfg.clip)
        n_samples = int(math.floor(N * self.cfg.fraction))
        n_shards = int(math.ceil(n_samples/self.cfg.shard_size))
        
        return [
            self.cfg.shuffled_shard_cls(
                root=self._root_,
                spec=dict(
                    clip=dbx.quote(self.cfg.clip),
                    fraction=self.cfg.fraction,
                    shard_size=self.cfg.shard_size,
                    shuffle_seed=self.cfg.shuffle_seed,
                    shard_idx=idx
                )
            )
            for idx in range(n_shards)
        ]


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
            self._shard = None
            self._shard_label = None
            self._shard_slide = None
            gc.collect()
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

