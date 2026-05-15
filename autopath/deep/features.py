"""Deep feature bags and clips for multi-layer activation capture.

Provides :class:`DeepFeatureBag` and :class:`DeepFeatureClip` for storing
multi-layer activations captured by a :class:`DeepBackboneEvaluator`,
plus spherical and corner (bipolar) encoding variants.

Storage uses `MDS shards <https://docs.mosaicml.com/projects/streaming>`_
for efficient streaming reads and row-level shuffling.
"""

import functools
import gc
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset

import dbx
from tqdm import tqdm
from dbx import (
    Datablock,
    write_npz,
    read_npz,
    write_tensor,
    read_tensor,
)

from streaming import MDSWriter, StreamingDataset

from autopath.databits import Bag, Clip, DeepBackboneEvaluatorFactory
from autopath.pancan.clips import TileBag


# ═══════════════════════════════════════════════════════════════════════
#  Dataset wrappers for label injection and tile pairing
# ═══════════════════════════════════════════════════════════════════════

class BagIndexDataset(Dataset):
    """Zips a dataset with its bags of origin:
    Wraps a base dataset and injects a ``bag_index`` field into the samples 
    for referring back to bags of origin for additional data."""

    def __init__(self, base, *, bag_index: int):
        self.base = base
        self.bag_index = bag_index

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        sample["bag_index"] = self.bag_index
        return sample


class LabeledDataset(Dataset):
    """Zips a dataset and its labels.
    Wraps a base dataset and injects a ``label`` field.

    Parameters
    ----------
    base : Dataset
        Underlying dataset.
    labels : array-like
        Flat label array, one per sample.
    """

    def __init__(self, base, labels):
        self.base = base
        self.labels = labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        sample["label"] = self.labels[idx]
        return sample


class TilePairingDataset(Dataset):
    """Wraps a feature dataset and lazily loads tiles from TileBags.

    Uses ``bag_index`` and ``tile_index`` fields in each sample to
    resolve the original tile from the correct :class:`TileBag`.

    Parameters
    ----------
    base : Dataset
        Underlying dataset (must contain ``bag_index`` and ``tile_index``).
    tilebags : list[TileBag]
        Ordered list of tile bags corresponding to bag indices.
    """

    def __init__(self, base, tilebags):
        self.base = base
        self.tilebags = tilebags

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = self.base[idx]
        bag_idx = sample["bag_index"]
        tile_idx = sample["tile_index"]
        sample["tile"] = self.tilebags[bag_idx].tiles[tile_idx]
        return sample


# ═══════════════════════════════════════════════════════════════════════
#  Deep Feature Bags & Clips
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

    All data is stored as MDS shards for efficient streaming reads.
    Each MDS sample contains feature columns for every captured layer,
    optionally raw tile images, and a ``tile_index`` for back-referencing
    the source :class:`TileBag`.

    Topics
    ------
    feature_names
        List of captured feature keys (e.g. ``['output']``,
        ``['block.0', 'block.38', 'output']``).
    dataset
        MDS shards directory.  ``read('dataset')`` returns a
        :class:`StreamingDataset`.
    """

    VERSION = 4

    TOPICFILES = {
        'feature_names': 'feature_names.npz',
        'dataset': 'shards',
    }

    @dataclass
    class CONFIG(Datablock.CONFIG):
        tilebag: TileBag
        evaluator_factory: DeepBackboneEvaluatorFactory
        shard_size: int = 1024    # samples per MDS shard
        capture_tiles: bool = True

    def __init__(self, *args, gpu_batch_size: int = 64, device: str = "cuda", **kwargs):
        Datablock.__init__(self, *args, gpu_batch_size=gpu_batch_size, device=device, **kwargs)

    def __post_init__(self):
        # Compute feature names from factory config (no model load needed).
        factory = self.cfg.evaluator_factory
        if hasattr(factory, 'evaluator'):
            self._feature_names = factory.evaluator(log=self.log).layer_names
        elif hasattr(factory, 'cfg'):
            names = []
            for block in getattr(factory.cfg, 'capture_blocks', []):
                names.append(f"block.{block}")
            for layer in getattr(factory.cfg, 'capture_layers', []):
                names.append(layer)
            if getattr(factory.cfg, 'capture_outputs', True):
                names.append('output')
            self._feature_names = names
        else:
            self._feature_names = []
        return self

    # ── Properties ──────────────────────────────────────────────────

    @property
    def tilebag(self):
        """The source TileBag."""
        return self.cfg.tilebag

    @property
    def feature_names(self):
        """Ordered list of captured feature keys."""
        return list(self._feature_names)

    @property
    def shards_path(self):
        """Directory containing MDS shards."""
        return self.path('dataset')

    def __len__(self):
        return len(self.cfg.tilebag.tiles)

    # ── Validity ────────────────────────────────────────────────────

    def validtopic(self, topic=None):
        if topic == 'dataset':
            # MDS directory is valid when index.json has been written.
            return self.fs.exists(
                os.path.join(self.path('dataset'), 'index.json')
            )
        return super().validtopic(topic)

    # ── Build ───────────────────────────────────────────────────────

    def __build__(self, evaluator=None, tilebag=None):
        if evaluator is None:
            evaluator = self.cfg.evaluator_factory.evaluator(
                device=self.device, log=self.log
            )

        if tilebag is None:
            tilebag = self.cfg.tilebag
        n_tiles = len(tilebag.tiles)
        feature_names = evaluator.layer_names
        self._feature_names = feature_names

        # Accumulate per-layer feature lists.
        accumulated = {lyr: [] for lyr in feature_names}
        tiles_list = [] if self.cfg.capture_tiles else None

        n_batches = math.ceil(n_tiles / self.gpu_batch_size)
        progress = tqdm(range(n_batches), desc=self.tag or 'tiles', unit='batch')
        for k in progress:
            m = k * self.gpu_batch_size
            n = min((k + 1) * self.gpu_batch_size, n_tiles)
            batch = tilebag.tiles[m:n].to(self.device)

            result = evaluator(batch)

            progress.set_postfix_str(
                f"VRAM {torch.cuda.memory_allocated(self.device)/1e9:.1f}/"
                f"{torch.cuda.get_device_properties(self.device).total_memory/1e9:.0f}GB "
                f"(peak {torch.cuda.max_memory_allocated(self.device)/1e9:.1f}GB)"
            )

            for lyr in feature_names:
                if lyr in result:
                    accumulated[lyr].append(result[lyr].cpu().detach())

            if tiles_list is not None:
                tiles_list.append(tilebag.tiles[m:n].cpu())

            evaluator.clear()
            del batch, result
            gc.collect()
            torch.cuda.empty_cache()

        # Write feature names.
        write_npz(
            self.path('feature_names', ensure_dirpath=True),
            feature_names=feature_names,
        )

        # Write MDS shards (features + tiles + tile_index).
        tiles_cat = None
        if tiles_list is not None:
            tiles_cat = torch.cat(tiles_list, dim=0).numpy()
            del tiles_list
        self._write_mds_shards(accumulated, feature_names, n_tiles, tiles=tiles_cat)

        del accumulated, tiles_cat
        gc.collect()
        self._len = n_tiles
        return self

    def _write_mds_shards(self, accumulated, feature_names, n_tiles, *, tiles=None):
        """Write accumulated features as MDS shards for streaming reads."""
        # Build column schema.
        columns = {}
        for lyr in feature_names:
            safe = lyr.replace(".", "_")
            columns[f"features_{safe}"] = "ndarray:float32"
        if tiles is not None:
            columns["tiles"] = "ndarray:uint8"
        columns["tile_index"] = "int32"

        # Concatenate once per layer for indexing.
        cat = {}
        for lyr in feature_names:
            if accumulated[lyr]:
                cat[lyr] = torch.cat(accumulated[lyr], dim=0).numpy()

        shards_dir = self.shards_path
        os.makedirs(shards_dir, exist_ok=True)

        size_limit = self.cfg.shard_size  # samples per shard

        # Estimate bytes per sample for size_limit conversion.
        # MDSWriter.size_limit is in bytes; we convert from sample count.
        sample_bytes = sum(
            cat[lyr][0].nbytes for lyr in feature_names if lyr in cat
        ) + 4  # +4 for tile_index int32
        if tiles is not None:
            sample_bytes += tiles[0].nbytes
        byte_limit = size_limit * sample_bytes

        with MDSWriter(
            out=shards_dir,
            columns=columns,
            size_limit=byte_limit,
        ) as writer:
            for i in range(n_tiles):
                sample = {}
                for lyr in feature_names:
                    safe = lyr.replace(".", "_")
                    if lyr in cat:
                        sample[f"features_{safe}"] = cat[lyr][i].astype(
                            np.float32
                        )
                    else:
                        # Layer missing — should not happen in practice.
                        pass
                if tiles is not None:
                    sample["tiles"] = tiles[i].astype(np.uint8)
                sample["tile_index"] = np.int32(i)
                writer.write(sample)

        n_shards = math.ceil(n_tiles / size_limit)
        self.log.verbose(
            f"Wrote MDS shards to {shards_dir}: "
            f"{n_shards} shards × ~{size_limit} samples/shard "
            f"({n_tiles} tiles total)"
        )
        del cat

    # ── Read ────────────────────────────────────────────────────────

    def __read__(self, topic):
        if topic == 'feature_names':
            return list(
                read_npz(self.path('feature_names'), 'feature_names')['feature_names']
            )
        if topic == 'dataset':
            return self.dataset()
        raise ValueError(f"Unknown topic: {topic!r}")

    def dataset(self, *, shuffle: bool = False) -> StreamingDataset:
        """Return a :class:`StreamingDataset` over this bag's MDS shards.

        Each sample is a dict with keys ``features_{name}`` (ndarray),
        optionally ``tiles`` (ndarray), and ``tile_index`` (int32).
        """
        return StreamingDataset(
            local=self.shards_path,
            shuffle=shuffle,
        )

    def layer_features(self, layer: str):
        """Read the full feature tensor for a single layer from MDS shards.

        Parameters
        ----------
        layer : str
            Feature key (e.g. ``"block.0"``, ``"output"``).

        Returns
        -------
        Tensor
            Shape ``(n_tiles, feature_dim)`` or ``(n_tiles, feature_dim)``
            depending on ``cls_token_only``.
        """
        safe = layer.replace(".", "_")
        col = f"features_{safe}"
        ds = self.dataset()
        arrays = [ds[i][col] for i in range(len(ds))]
        return torch.from_numpy(np.stack(arrays))

    @functools.cached_property
    def features(self) -> dict:
        """All layer features as ``{feature_name: Tensor}``."""
        return {name: self.layer_features(name) for name in self.feature_names}


class DeepFeatureClip(Clip):
    """A clip of :class:`DeepFeatureBag` shards built in parallel via Datastack.

    Orchestrates building one :class:`DeepFeatureBag` per tile-bag in the
    source ``tilebagclip``, using the configured
    :class:`DeepBackboneEvaluatorFactory`.
    """

    VERSION = 2

    @dataclass
    class CONFIG(Datablock.CONFIG):
        tilebagclip: Clip
        evaluator_factory: DeepBackboneEvaluatorFactory
        shard_size: int = 1024
        capture_tiles: bool = True

    def __init__(self, *args, gpu_batch_size: int = 64,
                 devices: list = None, **kwargs):
        self._devices = devices or ["cuda"]
        super().__init__(*args, gpu_batch_size=gpu_batch_size, **kwargs)

    @property
    def n_shards(self) -> int:
        return self.cfg.tilebagclip.n_shards

    class ShardMaker(Clip.ShardMaker):
        """Forward precomputed evaluator and tilebag to shard.build().

        Carries its assigned ``device`` so the assignment survives
        serialisation across process boundaries.
        """
        def __init__(self, idx: int, *, device: str = "cuda"):
            super().__init__(idx)
            self.device = device

        def __call__(self, stack, *, build=True):
            shard = stack.__shard__(self.idx, device=self.device)
            shard.keyby = stack.keyby
            if build:
                kwargs = {}
                if hasattr(stack, '_shared_evaluator'):
                    kwargs['evaluator'] = stack._shared_evaluator
                if hasattr(stack, '_precomputed_tilebags'):
                    kwargs['tilebag'] = stack._precomputed_tilebags[self.idx]
                shard.build(**kwargs)
            del shard
            gc.collect()

    def __shard__(self, idx: int, tilebagclip=None, device: str = "cuda"):
        if tilebagclip is None:
            tilebagclip = self.cfg.tilebagclip
        # Use precomputed tilebag if available to avoid re-forming the fold.
        if hasattr(self, '_precomputed_tilebags'):
            tilebag = self._precomputed_tilebags[idx]
        else:
            tilebag = tilebagclip.shard(idx)
        return DeepFeatureBag(
            url=self.url,
            spec=dict(
                tilebag=dbx.quote(tilebag),
                evaluator_factory=self.spec['evaluator_factory'],
                shard_size=self.cfg.shard_size,
                capture_tiles=self.cfg.capture_tiles,
            ),
            gpu_batch_size=self.gpu_batch_size,
            device=device,
            revision=self.revision,
            tag=tilebag.tag,
        )

    def __build__(self, *args, **kwargs):
        """Build with optional shared evaluator and precomputed tile-bags.

        In inline mode (single device), precomputes the evaluator once
        and shares it across all shards.  In multiprocessing mode, each
        worker creates its own evaluator on its assigned device.
        """
        devices = self._devices
        # Precompute flat tilebag list to avoid re-forming the fold.
        tilebagclip = self.cfg.tilebagclip
        self._precomputed_tilebags = [
            tilebagclip.shard(idx) for idx in range(self.n_shards)
        ]
        # Share a single evaluator only when running inline on one device.
        inline = (self.parallelization in (None, 'inline')
                  and len(devices) == 1)
        if inline:
            device = devices[0]
            self._shared_evaluator = self.cfg.evaluator_factory.evaluator(
                device=device, log=self.log,
            )
            self.log.info(
                f"Precomputed {self.n_shards} tile-bags and evaluator on {device}"
            )
        else:
            self.log.info(
                f"Precomputed {self.n_shards} tile-bags; "
                f"workers will create evaluators on {devices}"
            )
        # Override maker list with device-assigned ShardMakers.
        # The executor splits makers into contiguous chunks via
        # np.array_split — worker w gets chunk w.  Assign the device
        # for each shard based on which chunk (worker) it will land in.
        self.__split__()
        n_workers = len(devices)
        chunk_boundaries = np.array_split(range(self.n_shards), n_workers)
        shard_device = {}
        for worker_idx, chunk in enumerate(chunk_boundaries):
            dev = devices[worker_idx % len(devices)]
            for idx in chunk:
                shard_device[idx] = dev
        makers = [
            self.ShardMaker(idx, device=shard_device[idx])
            for idx in range(self.n_shards)
        ]
        executor_kwargs = dict(
            n_workers=self.n_workers,
            tag=f"BUILDING {len(makers)} shards [{self.__class__.__name__}, n_workers={self.n_workers}]",
        )
        if (hasattr(self, 'multiprocessing_start_method')
                and self.multiprocessing_start_method is not None):
            from dbx.dataparts import MultiprocessingCallableExecutor
            if issubclass(self.executor_cls, MultiprocessingCallableExecutor):
                executor_kwargs['start_method'] = self.multiprocessing_start_method
        executor = self.executor_cls(**executor_kwargs)
        self.log.info(
            f"Building {self.__class__.__name__}: {len(makers)} shards, "
            f"executor={executor.__class__.__name__}, n_workers={self.n_workers}"
        )
        try:
            executor.exec_callables(makers, self, build=True)
        finally:
            if hasattr(self, '_shared_evaluator'):
                del self._shared_evaluator
            del self._precomputed_tilebags
        self.log.info(f"Stacking {self.n_shards} shards of {self.__class__.__name__}")
        self.__stack__()
        self.log.info(f"Build complete: {self.__class__.__name__}")
        return self

    def dataset(
        self,
        *,
        shuffle: bool = False,
        include_tiles: bool = False,
    ) -> Dataset:
        """Return a unified, labeled dataset over all bags.

        Each sample dict contains:

        * ``features_{layer_name}`` — per-layer activation ndarray
        * ``tile_index`` — index into the originating TileBag
        * ``bag_index`` — index of the bag within this clip
        * ``label`` — label string from the source TileBag

        Parameters
        ----------
        shuffle : bool
            Whether to shuffle within each bag's streaming dataset.
        include_tiles : bool
            If ``True``, each sample also contains a ``tile`` field
            loaded lazily from the source TileBag (slower; suitable
            for visualization, not training).

        Returns
        -------
        Dataset
            A :class:`torch.utils.data.Dataset` over all bags.
        """
        bag_datasets = []
        tilebags = []
        bag_labels = []
        bag_lens = []

        for i, bag in enumerate(self.bags):
            ds = bag.dataset(shuffle=shuffle)
            bag_datasets.append(BagIndexDataset(ds, bag_index=i))
            tilebags.append(bag.tilebag)
            bag_labels.append(bag.tilebag.label)
            bag_lens.append(len(bag))

        # Concatenate all bag datasets.
        base = ConcatDataset(bag_datasets)

        # Build flat label array and wrap.
        labels = np.concatenate([
            np.full(n, label) for label, n in zip(bag_labels, bag_lens)
        ])
        result = LabeledDataset(base, labels)

        if include_tiles:
            result = TilePairingDataset(result, tilebags)

        return result


# ═══════════════════════════════════════════════════════════════════════
#  Spherical & Corner Deep Feature Bags
# ═══════════════════════════════════════════════════════════════════════

class SphericalDeepFeatureBag(Bag):
    """L2-normalised view of a :class:`DeepFeatureBag`.

    Each layer's feature tensor is projected onto the unit hypersphere
    (L2-normalised along the feature dimension).  This is a pure
    runtime transformation — no separate build step is required.
    """

    VERSION = 2

    @dataclass
    class CONFIG(Datablock.CONFIG):
        deep_feature_bag: DeepFeatureBag

    def __init__(self, *args, **kwargs):
        Datablock.__init__(self, *args, **kwargs)

    @property
    def tilebag(self):
        """The source TileBag (delegated through the underlying bag)."""
        return self.cfg.deep_feature_bag.tilebag

    @property
    def feature_names(self):
        return self.cfg.deep_feature_bag.feature_names

    def layer_features(self, layer: str):
        """L2-normalised features for a single layer."""
        t = self.cfg.deep_feature_bag.layer_features(layer)
        return torch.nn.functional.normalize(t.float(), p=2, dim=-1)

    @functools.cached_property
    def features(self) -> dict:
        """All layer features (L2-normalised) as ``{name: Tensor}``."""
        return {name: self.layer_features(name) for name in self.feature_names}

    def __len__(self):
        return len(self.cfg.deep_feature_bag)

    def valid(self):
        return self.cfg.deep_feature_bag.valid()


class SphericalDeepFeatureClip(Clip):
    """Clip of L2-normalised :class:`SphericalDeepFeatureBag` views
    over a :class:`DeepFeatureClip`.

    Pure runtime transformation — delegates storage to the underlying
    clip.  No separate build step is required.
    """

    @dataclass
    class CONFIG(Datablock.CONFIG):
        deep_feature_clip: DeepFeatureClip

    @property
    def n_shards(self):
        return self.cfg.deep_feature_clip.n_shards

    def __shard__(self, idx: int):
        dfb = self.cfg.deep_feature_clip.bag(idx)
        return SphericalDeepFeatureBag(
            url=self.url,
            spec=dict(deep_feature_bag=dbx.quote(dfb)),
            tag=dfb.tag,
        )

    def valid(self):
        return self.cfg.deep_feature_clip.valid()

    def __build__(self):
        """No build step — delegates to underlying clip. Persist bag_lens only."""
        self.__stack__()
        return self


class CornerDeepFeatureBag(Bag):
    """Bipolar (hypercube-vertex) encoding of a :class:`DeepFeatureBag`.

    For each layer, maps each tile's feature vector to the nearest vertex
    of the unit hypercube ``{-1, +1}^d`` using the sign of each
    coordinate.  This generalises the existing ``BipolarFeatureBag``
    concept to multi-layer deep features without requiring a median
    probe.

    Pure runtime transformation — no separate build step required.
    """

    VERSION = 2

    @dataclass
    class CONFIG(Datablock.CONFIG):
        deep_feature_bag: DeepFeatureBag

    def __init__(self, *args, **kwargs):
        Datablock.__init__(self, *args, **kwargs)

    @property
    def tilebag(self):
        """The source TileBag (delegated through the underlying bag)."""
        return self.cfg.deep_feature_bag.tilebag

    @property
    def feature_names(self):
        return self.cfg.deep_feature_bag.feature_names

    def layer_features(self, layer: str):
        """Bipolar-encoded features for a single layer: sign(x) ∈ {-1, +1}^d."""
        t = self.cfg.deep_feature_bag.layer_features(layer)
        return torch.sign(t)

    @functools.cached_property
    def features(self) -> dict:
        """All layer features (bipolar-encoded) as ``{name: Tensor}``."""
        return {name: self.layer_features(name) for name in self.feature_names}

    def __len__(self):
        return len(self.cfg.deep_feature_bag)

    def valid(self):
        return self.cfg.deep_feature_bag.valid()

    def corner_ids(self, layer: str, max_dims: int = 63):
        """Integer encoding of hypercube corners.

        Maps ``{-1, +1}^d → int`` by treating positive coordinates as
        binary 1-bits.  Truncated to *max_dims* dimensions to avoid
        integer overflow.

        Parameters
        ----------
        layer : str
            Capture key.
        max_dims : int
            Maximum number of dimensions to use (default 63, safe for
            int64).

        Returns
        -------
        Tensor
            Integer corner IDs, shape ``(n_tiles,)``.
        """
        signs = (self.cfg.deep_feature_bag.layer_features(layer) > 0).long()
        d = min(signs.shape[-1], max_dims)
        powers = 2 ** torch.arange(d)
        return (signs[..., :d] * powers).sum(dim=-1)


class CornerDeepFeatureClip(Clip):
    """Clip of bipolar-encoded :class:`CornerDeepFeatureBag` views
    over a :class:`DeepFeatureClip`.

    Pure runtime transformation — delegates storage to the underlying
    clip.  No separate build step is required.
    """

    @dataclass
    class CONFIG(Datablock.CONFIG):
        deep_feature_clip: DeepFeatureClip

    @property
    def n_shards(self):
        return self.cfg.deep_feature_clip.n_shards

    def __shard__(self, idx: int):
        dfb = self.cfg.deep_feature_clip.bag(idx)
        return CornerDeepFeatureBag(
            url=self.url,
            spec=dict(deep_feature_bag=dbx.quote(dfb)),
            tag=dfb.tag,
        )

    def valid(self):
        return self.cfg.deep_feature_clip.valid()

    def __build__(self):
        """No build step — delegates to underlying clip. Persist bag_lens only."""
        self.__stack__()
        return self
