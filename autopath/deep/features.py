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
)

from streaming import MDSWriter, Stream, StreamingDataset

from autopath.autobits import Bag, Clip, DeepBackboneEvaluatorFactory, TileBag


# ═══════════════════════════════════════════════════════════════════════
#  Dataset wrapper for tile–feature pairing
# ═══════════════════════════════════════════════════════════════════════

class TileFeatureDataset(Dataset):
    """Pairs deep-feature samples with their upstream tile-shard fields.

    Wraps a multi-stream :class:`StreamingDataset` of feature shards and
    enriches each sample with **all** fields from the corresponding
    tile shard (``tile``, ``bag_name``, ``annotations``, etc.).

    Bag membership is derived from cumulative bag lengths: a binary
    search maps the flat sample index to ``bag_index``, which selects
    the correct per-bag tile :class:`StreamingDataset`.  Within that
    dataset, ``tile_index`` (written by :class:`DeepFeatureBag`) is
    used to fetch the matching tile entry.

    Parameters
    ----------
    base : StreamingDataset
        Merged feature streaming dataset (one :class:`Stream` per bag).
    bag_lens : list[int]
        Length of each bag, in stream order.
    tile_datasets : list[StreamingDataset]
        Per-bag tile MDS datasets, in the same bag order.
    """

    def __init__(self, base, bag_lens, tile_datasets):
        self.base = base
        self.tile_datasets = tile_datasets
        # cumsum[i] = start index of bag i; cumsum[-1] = total length
        self._cumsum = np.cumsum([0] + list(bag_lens))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sample = dict(self.base[idx])
        bag_idx = int(np.searchsorted(self._cumsum[1:], idx, side='right'))
        sample['bag_index'] = bag_idx
        tile_idx = sample['tile_index']
        tile_sample = self.tile_datasets[bag_idx][tile_idx]
        # Inject all fields from the tile shard (tile, bag_name,
        # annotations, etc.) without overwriting feature columns.
        for k, v in tile_sample.items():
            if k not in sample:
                sample[k] = v
        return sample


# ═══════════════════════════════════════════════════════════════════════
#  Deep Feature Bags & Clips
# ═══════════════════════════════════════════════════════════════════════


class DeepFeatureBag(Bag):
    """A Bag that stores multi-layer activations from a DeepBackboneEvaluator.

    All data is stored as MDS shards for efficient streaming reads.
    Each MDS sample contains feature columns for every captured layer,
    optionally raw tile images, and a ``tile_index`` for back-referencing
    the source :class:`TileBag`.

    Topics
    ------
    shards
        MDS shards directory.  ``read('shards')`` returns an
        ``fs.ls`` listing of the shards directory.
    """

    VERSION = 7

    TOPICS = ['shards']

    @dataclass
    class CONFIG(Datablock.CONFIG):
        tilebag: TileBag
        evaluator_factory: DeepBackboneEvaluatorFactory
        shard_size: int = 1024    # samples per MDS shard

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
    def _is_local_fs(self):
        """True when this bag's storage is on a local filesystem."""
        protocol = self.fs.protocol if isinstance(self.fs.protocol, str) else self.fs.protocol[0]
        return protocol in ('file', 'local', '')

    def __len__(self):
        return len(self.cfg.tilebag)

    # ── Validity ────────────────────────────────────────────────────

    def validtopic(self, topic=None):
        if topic == 'shards':
            # MDS directory is valid when index.json has been written.
            return self.fs.exists(
                os.path.join(self.path('shards'), 'index.json')
            )
        return super().validtopic(topic)

    def valid(self, topic=None):
        if topic is not None:
            return self.validtopic(topic)
        return self.validtopics(reduce=True)

    # ── Build ───────────────────────────────────────────────────────

    def __build__(self, evaluator=None, tilebag=None):
        if evaluator is None:
            evaluator = self.cfg.evaluator_factory.evaluator(
                device=self.device, log=self.log
            )

        if tilebag is None:
            tilebag = self.cfg.tilebag
        n_tiles = len(tilebag)
        feature_names = evaluator.layer_names
        self._feature_names = feature_names

        # Stream tiles from the tilebag's MDS dataset to avoid loading
        # all tiles into RAM.  StreamingDataset allocates POSIX shared
        # memory, which can exhaust file descriptors when many bags are
        # built in sequence; fall back to in-memory tile loading.
        try:
            ds = tilebag.dataset()
        except (OSError, RuntimeError):
            ds = None
            self.log.info(
                "StreamingDataset unavailable (shared memory exhaustion); "
                "falling back to in-memory tile loading"
            )

        shards_dir = self.path('shards', ensure_dirpath=True)

        # Build MDS column schema.
        columns = {}
        for lyr in feature_names:
            safe = lyr.replace(".", "_")
            columns[f"features_{safe}"] = "ndarray:float32"
        columns["tile_index"] = "int32"

        n_batches = math.ceil(n_tiles / self.gpu_batch_size)
        progress = tqdm(range(n_batches), desc=self.tag or 'tiles', unit='batch')
        writer = None
        tile_cursor = 0

        try:
            for k in progress:
                m = k * self.gpu_batch_size
                n = min((k + 1) * self.gpu_batch_size, n_tiles)
                if ds is not None:
                    # Stream a batch of tiles from the dataset.
                    batch = torch.stack([
                        torch.as_tensor(ds[i]['tile']) for i in range(m, n)
                    ]).to(self.device)
                else:
                    # Fallback: slice from the full tile tensor.
                    batch = tilebag.tiles[m:n].to(self.device)

                result = evaluator(batch)

                # Lazily initialise the writer on the first batch once
                # per-sample byte sizes are known.
                if writer is None:
                    sample_bytes = sum(
                        result[lyr][0].numpy().nbytes
                        for lyr in feature_names if lyr in result
                    ) + 4  # +4 for tile_index int32
                    byte_limit = self.cfg.shard_size * sample_bytes
                    writer = MDSWriter(
                        out=shards_dir,
                        columns=columns,
                        size_limit=byte_limit,
                    )

                # Write features for each sample directly to MDS.
                batch_size = n - m
                for i in range(batch_size):
                    sample = {}
                    for lyr in feature_names:
                        safe = lyr.replace(".", "_")
                        if lyr in result:
                            sample[f"features_{safe}"] = result[lyr][i].numpy().astype(
                                np.float32
                            )
                    sample["tile_index"] = np.int32(tile_cursor + i)
                    writer.write(sample)

                tile_cursor += batch_size

                progress.set_postfix_str(
                    f"VRAM {torch.cuda.memory_allocated(self.device)/1e9:.1f}/"
                    f"{torch.cuda.get_device_properties(self.device).total_memory/1e9:.0f}GB "
                    f"(peak {torch.cuda.max_memory_allocated(self.device)/1e9:.1f}GB)"
                )

                evaluator.clear()
                del batch, result
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            if writer is not None:
                writer.finish()
            # Release StreamingDataset shared memory segments so they
            # don't accumulate across sequential bag builds.
            if ds is not None:
                del ds
            gc.collect()

        n_shards = math.ceil(n_tiles / self.cfg.shard_size)
        self.log.verbose(
            f"Wrote MDS shards to {shards_dir}: "
            f"{n_shards} shards × ~{self.cfg.shard_size} samples/shard "
            f"({n_tiles} tiles total)"
        )

        self._len = n_tiles
        return self

    # ── Read ────────────────────────────────────────────────────────

    def __read__(self, topic):
        if topic == 'shards':
            return self.fs.ls(self.path('shards'))
        raise ValueError(f"Unknown topic: {topic!r}")

    def dataset(self, *, shuffle: bool = False) -> StreamingDataset:
        """Return a :class:`StreamingDataset` over this bag's MDS shards.

        Each sample is a dict with keys ``features_{name}`` (ndarray),
        optionally ``tiles`` (ndarray), and ``tile_index`` (int32).
        """
        if self._is_local_fs:
            return StreamingDataset(local=self.path('shards'), shuffle=shuffle)
        else:
            return StreamingDataset(remote=self.path('shards'), shuffle=shuffle)

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

    Uses the v2 :class:`Datastack` API: :meth:`__split_v2__` returns
    device-assigned :class:`ShardMaker` callables and shared state as
    ``callable_kwargs``; :meth:`__stack_v2__` persists bag lengths.

    All shared state (evaluator, tilebags, evaluator factory) flows
    through ``callable_kwargs`` — never as side-effected ``self``
    attributes — so it survives serialisation across process boundaries.
    """

    v2 = True
    VERSION = 4

    TOPICS = ['bag_lens']
    _BAG_LENS_FILE = 'bag_lens.npz'

    @dataclass
    class CONFIG(Datablock.CONFIG):
        tilebagclip: Clip
        evaluator_factory: DeepBackboneEvaluatorFactory
        shard_size: int = 1024

    def __init__(self, *args, gpu_batch_size: int = 64,
                 devices: list = None, **kwargs):
        self._devices = devices or ["cuda"]
        super().__init__(*args, gpu_batch_size=gpu_batch_size, v2=True, **kwargs)

    @property
    def n_shards(self) -> int:
        return self.cfg.tilebagclip.n_shards

    class ShardMaker(Clip.ShardMaker):
        """Build a single shard using shared state from ``callable_kwargs``.

        Carries its assigned ``device`` so the assignment survives
        serialisation across process boundaries.

        The ``evaluator_factory`` is received via the executor's
        ``ctx_kwargs``.  The executor ``dbx.eval()``s all ctx kwargs,
        so a quoted factory spec is evaluated once per worker.  The
        factory caches its evaluator, so all ShardMakers within a
        worker share the same evaluator instance.
        """
        def __init__(self, idx: int, *, device: str = "cuda"):
            super().__init__(idx)
            self.device = device

        def __call__(self, stack, *, build=True,
                     tilebags, evaluator_factory):
            shard = stack.__shard__(self.idx, device=self.device,
                                    tilebag=tilebags[self.idx])
            shard.keyby = stack.keyby
            if build:
                shard.build(
                    evaluator=evaluator_factory.evaluator(
                        device=self.device, log=stack.log,
                    ),
                    tilebag=tilebags[self.idx],
                )
            del shard
            gc.collect()

    def __shard__(self, idx: int, tilebagclip=None, device: str = "cuda",
                  tilebag=None):
        if tilebag is None:
            if tilebagclip is None:
                tilebagclip = self.cfg.tilebagclip
            tilebag = tilebagclip.shard(idx)
        return DeepFeatureBag(
            url=self.url,
            spec=dict(
                tilebag=dbx.quote(tilebag),
                evaluator_factory=self.spec['evaluator_factory'],
                shard_size=self.cfg.shard_size,
            ),
            gpu_batch_size=self.gpu_batch_size,
            device=device,
            revision=self.revision,
            tag=tilebag.tag,
        )

    def __split__(self, *args, **kwargs):
        """Precompute tilebags; return device-assigned ShardMakers.

        All shared state is returned in ``callable_kwargs`` so it
        flows through the executor's ``ctx_kwargs`` mechanism — no
        ``self``-attribute side effects.

        The ``evaluator_factory`` is passed through ``callable_kwargs``.
        The executor ``dbx.eval()``s it once per worker, and the
        factory caches its evaluator, giving one evaluator per worker.

        Returns
        -------
        callables : list[ShardMaker]
            One ShardMaker per shard, each carrying its target device.
        callable_kwargs : dict
            Shared state forwarded to each ShardMaker.__call__:
            ``build``, ``tilebags``, ``evaluator_factory``.
        """
        devices = self._devices
        # Precompute flat tilebag list to avoid re-forming the fold.
        tilebagclip = self.cfg.tilebagclip
        tilebags = [tilebagclip.shard(idx) for idx in range(self.n_shards)]

        callable_kwargs = dict(
            build=True,
            tilebags=tilebags,
            evaluator_factory=self.cfg.evaluator_factory,
        )
        self.log.info(
            f"Precomputed {self.n_shards} tile-bags; "
            f"evaluator_factory passed to executor (devices={devices})"
        )
        # Assign devices to ShardMakers based on executor chunking.
        # The executor splits callables into contiguous chunks via
        # np.array_split — worker w gets chunk w.  Assign the device
        # for each shard based on which chunk (worker) it will land in.
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
        self.log.info(
            f"Split {self.__class__.__name__}: {len(makers)} shards across "
            f"{n_workers} workers on devices {devices}"
        )
        return makers, callable_kwargs

    def __read__(self, topic=None):
        if topic == 'bag_lens':
            path = os.path.join(self.path('bag_lens'), self._BAG_LENS_FILE)
            return dbx.read_npz(path, 'bag_lens')['bag_lens']
        raise ValueError(f"Unknown topic: {topic!r}")

    def __stack__(self, results=None):
        """Persist bag_lens after parallel build."""
        self.log.info(f"Stacking {self.n_shards} shards of {self.__class__.__name__}")
        bag_lens = [len(self.shard(i)) for i in tqdm.tqdm(
            range(self.n_shards), desc="Stacking bag lens"
        )]
        bag_lens_dir = self.path('bag_lens', ensure_dirpath=True)
        dbx.write_npz(os.path.join(bag_lens_dir, self._BAG_LENS_FILE), bag_lens=bag_lens)
        self.log.info(f"Build complete: {self.__class__.__name__}")
        return self

    @functools.cached_property
    def bag_lens(self):
        return self.read('bag_lens')

    def _ensure_precomputed_tilebags(self):
        """Precompute the tilebag list to avoid re-forming the source clip."""
        if not hasattr(self, '_precomputed_tilebags'):
            tilebagclip = self.cfg.tilebagclip
            self._precomputed_tilebags = [
                tilebagclip.shard(idx) for idx in range(self.n_shards)
            ]

    def shards(self):
        self._ensure_precomputed_tilebags()
        return super().shards()

    def dataset(
        self,
        *,
        shuffle: bool = False,
        include_tiles: bool = False,
        skip_invalid_bags: bool = False,
    ) -> Dataset:
        """Return a unified, labeled dataset over all bags.

        Uses the MDS multi-stream API to create a **single**
        :class:`StreamingDataset` backed by one :class:`Stream` per bag.
        This avoids opening hundreds of separate shared-memory segments.

        Each sample dict contains:

        * ``features_{layer_name}`` — per-layer activation ndarray
        * ``tile_index`` — index into the originating TileBag
        * ``bag_index`` — index of the bag within this clip
        * ``label`` — label string from the source TileBag

        Parameters
        ----------
        shuffle : bool
            Whether to shuffle within the streaming dataset.
        include_tiles : bool
            If ``True``, each sample also contains a ``tile`` field
            loaded lazily from the source TileBag (slower; suitable
            for visualization, not training).
        skip_invalid_bags : bool
            If ``True``, silently skip bags whose MDS shards have not
            been built yet instead of raising.  The number of skipped
            bags is reported at the end.

        Returns
        -------
        Dataset
            A :class:`torch.utils.data.Dataset` over all bags.
        """
        streams = []
        tile_datasets = []
        bag_lens_list = []
        n_skipped = 0

        for i, bag in enumerate(self.bags):
            if skip_invalid_bags and not bag.valid():
                n_skipped += 1
                continue
            if bag._is_local_fs:
                streams.append(Stream(local=bag.path('shards')))
            else:
                streams.append(Stream(remote=bag.path('shards')))
            if include_tiles:
                tile_datasets.append(bag.tilebag.dataset())
            bag_lens_list.append(len(bag))

        if n_skipped:
            self.log.info(
                f"Skipped {n_skipped}/{n_skipped + len(streams)} "
                f"unbuilt bags in {self.__class__.__name__}.dataset()"
            )

        # One StreamingDataset with N streams — one shared-memory segment.
        base = StreamingDataset(streams=streams, shuffle=shuffle)

        if include_tiles:
            return TileFeatureDataset(base, bag_lens_list, tile_datasets)

        return base


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

    v2 = True

    @dataclass
    class CONFIG(Datablock.CONFIG):
        deep_feature_clip: DeepFeatureClip

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def __split__(self, *args, **kwargs):
        """No parallel work — delegates to underlying clip."""
        return [], dict()

    def __stack__(self, results=None):
        """Persist bag_lens only."""
        return super().__stack__(results)


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

    v2 = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def __split__(self, *args, **kwargs):
        """No parallel work — delegates to underlying clip."""
        return [], dict()

    def __stack__(self, results=None):
        """Persist bag_lens only."""
        return super().__stack__(results)
