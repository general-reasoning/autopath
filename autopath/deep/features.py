"""Deep feature bags and clips for multi-layer activation capture.

Provides :class:`DeepFeatureBag` and :class:`DeepFeatureClip` for storing
multi-layer activations captured by a :class:`DeepBackboneEvaluator`,
plus spherical and corner (bipolar) encoding variants.

Storage uses `MDS shards <https://docs.mosaicml.com/projects/streaming>`_
for efficient streaming reads and row-level shuffling.
"""

import contextlib
import functools
import gc
import math
import os
from collections import OrderedDict
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
from autopath.tools import mds_readers, read_mds_samples


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

    Tile datasets are opened **lazily** to avoid exhausting file
    descriptors when a clip contains thousands of bags.  An LRU cache
    (default 32 entries) bounds the number of simultaneously open
    shared-memory segments.

    Parameters
    ----------
    base : StreamingDataset
        Merged feature streaming dataset (one :class:`Stream` per bag).
    bag_lens : list[int]
        Length of each bag, in stream order.
    bags : list[DeepFeatureBag]
        Per-bag bag objects, in the same stream order.  Their
        ``.tilebag.dataset()`` is called lazily on first access.
    cache_size : int
        Maximum number of tile datasets kept open simultaneously.
    """

    def __init__(self, base, bag_lens, bags, *, cache_size: int = 32):
        self.base = base
        self._bags = bags
        # cumsum[i] = start index of bag i; cumsum[-1] = total length
        self._cumsum = np.cumsum([0] + list(bag_lens))
        self._tile_ds_cache = OrderedDict()
        self._cache_size = cache_size

    def __len__(self):
        return len(self.base)

    def _get_tile_dataset(self, bag_idx):
        """Return the tile dataset for *bag_idx*, opening lazily."""
        if bag_idx in self._tile_ds_cache:
            # Move to end (most-recently used).
            self._tile_ds_cache.move_to_end(bag_idx)
            return self._tile_ds_cache[bag_idx][0]
        cm = self._bags[bag_idx].tilebag.dataset()
        ds = cm.__enter__()
        self._tile_ds_cache[bag_idx] = (ds, cm)
        # Evict oldest if over capacity.
        while len(self._tile_ds_cache) > self._cache_size:
            _, evicted_cm = self._tile_ds_cache.popitem(last=False)
            evicted_cm.__exit__(None, None, None)
        return ds

    def __getitem__(self, idx):
        sample = dict(self.base[idx])
        bag_idx = int(np.searchsorted(self._cumsum[1:], idx, side='right'))
        sample['bag_index'] = bag_idx
        tile_idx = sample['tile_index']
        tile_sample = self._get_tile_dataset(bag_idx)[tile_idx]
        # Inject all fields from the tile shard (tile, bag_name,
        # annotations, etc.) without overwriting feature columns.
        # Skip None values — default_collate cannot handle them.
        for k, v in tile_sample.items():
            if k not in sample and v is not None:
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

        # Load all tiles eagerly.  We avoid StreamingDataset here because
        # it allocates /dev/shm shared memory whose cleanup depends on
        # __del__ / GC — unreliable when building thousands of bags per
        # worker, leading to EMFILE.  tilebag.tiles reads from the source
        # TFRecord, which opens and closes a single file handle cleanly.
        tiles_tensor = tilebag.tiles
        n_tiles = len(tiles_tensor)

        shards_dir = self.path('shards', ensure_dirpath=True)
        # Clear stale files from a prior interrupted build.
        if self.fs.exists(shards_dir) and self.fs.ls(shards_dir):
            self.log.warning(f"Clearing stale shard directory: {shards_dir}")
            self.fs.rm(shards_dir, recursive=True)
            self.fs.mkdirs(shards_dir, exist_ok=True)

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
                batch = tiles_tensor[m:n].to(self.device)

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
                        exist_ok=True,
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
            # Release the eagerly-loaded tile tensor.
            del tiles_tensor
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

    def data(self):
        """Bulk-read all MDS samples as a list of dicts.

        Uses :class:`MDSReader` (plain file I/O) rather than
        :class:`StreamingDataset` to avoid shared-memory leaks
        when reading many bags.
        """
        return list(read_mds_samples(self.path('shards')))

    @contextlib.contextmanager
    def dataset(self, *, shuffle: bool = False):
        """Yield a :class:`StreamingDataset` over this bag's MDS shards.

        Use as a context manager to ensure shared-memory file
        descriptors are released promptly::

            with bag.dataset() as ds:
                x = ds[0]['features_output']

        Each sample is a dict with keys ``features_{name}`` (ndarray),
        optionally ``tiles`` (ndarray), and ``tile_index`` (int32).

        .. note::
            For bulk reads prefer :meth:`layer_features` or
            :meth:`annotations`, which use :class:`MDSReader`
            and avoid shared-memory overhead entirely.
        """
        if self._is_local_fs:
            ds = StreamingDataset(local=self.path('shards'), shuffle=shuffle)
        else:
            ds = StreamingDataset(remote=self.path('shards'), shuffle=shuffle)
        try:
            yield ds
        finally:
            del ds
            gc.collect()


    def layer_features(self, layer: str):
        """Read the full feature tensor for a single layer from MDS shards.

        Uses :class:`MDSReader` (plain file I/O) rather than
        :class:`StreamingDataset` to avoid shared-memory leaks
        when reading many bags.

        Parameters
        ----------
        layer : str
            Feature key (e.g. ``"block.0"``, ``"output"``).

        Returns
        -------
        Tensor
            Shape ``(n_tiles, feature_dim)``.
        """
        safe = layer.replace(".", "_")
        col = f"features_{safe}"
        arrays = list(read_mds_samples(self.path('shards'), column=col))
        return torch.from_numpy(np.stack(arrays))

    def annotations(self, i=0) -> dict | None:
        """Read the annotation dict from MDS sample *i*.

        Uses :class:`MDSReader` (plain file I/O) rather than
        :class:`StreamingDataset`.

        Parameters
        ----------
        i : int
            Sample index (default ``0``).

        Returns
        -------
        dict | None
            The annotations dict, or ``None`` when no annotations
            column exists or the value is ``None``.
        """
        try:
            for reader in mds_readers(self.path('shards')):
                if i < len(reader):
                    sample = reader.get_item(i)
                    return sample.get('annotations')
                i -= len(reader)
            return None
        except Exception:
            return None

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

    # Inherits TOPICFILES = {"bag_lens": "bag_lens.npz"} from Clip.
    # Do NOT also declare TOPICS — the presence of both causes path()
    # to take the TOPICFILES branch (appending the filename) while
    # callers assumed the TOPICS branch (directory-only), resulting in
    # a doubled "bag_lens.npz/bag_lens.npz" path.

    @dataclass
    class CONFIG(Datablock.CONFIG):
        tilebagclip: Clip
        evaluator_factory: DeepBackboneEvaluatorFactory
        shard_size: int = 1024

    def __init__(self, *args, gpu_batch_size: int = 64,
                 devices: list = None, **kwargs):
        self._devices = devices or ["cuda"]
        kwargs.pop('v2', None)
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
            return dbx.read_npz(self.path('bag_lens'), 'bag_lens')['bag_lens']
        raise ValueError(f"Unknown topic: {topic!r}")

    def __stack__(self, results=None):
        """Persist bag_lens after parallel build."""
        self.log.info(f"Stacking {self.n_shards} shards of {self.__class__.__name__}")
        bag_lens = [len(self.shard(i)) for i in tqdm(
            range(self.n_shards), desc="Stacking bag lens"
        )]
        dbx.write_npz(self.path('bag_lens', ensure_dirpath=True), bag_lens=bag_lens)
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
        batch_size: int | None = None,
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
        batch_size : int | None
            Passed to :class:`StreamingDataset` for deterministic
            resumption.  Should match the DataLoader batch size.

        Returns
        -------
        Dataset
            A :class:`torch.utils.data.Dataset` over all bags.
        """
        streams = []
        bag_lens_list = []
        n_skipped = 0

        valid_bags = []
        for i, bag in enumerate(self.bags):
            if skip_invalid_bags and not bag.valid():
                n_skipped += 1
                continue
            if bag._is_local_fs:
                streams.append(Stream(local=bag.path('shards')))
            else:
                streams.append(Stream(remote=bag.path('shards')))
            valid_bags.append(bag)
            bag_lens_list.append(len(bag))

        if n_skipped:
            self.log.info(
                f"Skipped {n_skipped}/{n_skipped + len(streams)} "
                f"unbuilt bags in {self.__class__.__name__}.dataset()"
            )

        # One StreamingDataset with N streams — one shared-memory segment.
        sd_kwargs = dict(streams=streams, shuffle=shuffle)
        if batch_size is not None:
            sd_kwargs['batch_size'] = batch_size
        base = StreamingDataset(**sd_kwargs)

        if include_tiles:
            return TileFeatureDataset(base, bag_lens_list, valid_bags)

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


# ═══════════════════════════════════════════════════════════════════════
#  Bipolar (median-thresholded) Deep Feature Bags & Clips
# ═══════════════════════════════════════════════════════════════════════


class BipolarDeepFeatureBag(Bag):
    """Median-thresholded bipolar encoding of a :class:`DeepFeatureBag`.

    For each tile, maps its feature vector to ``{-1, +1}^d`` via
    ``sign(features - median)`` where the median is obtained from a
    pre-built :class:`DeepFeatureStatsProbe`.

    Also computes a bag-level bipolar feature by averaging the
    tile-level bipolars and thresholding:

    - ``abs(mean) >= bag_aggregation_threshold`` → ``sign(mean)``
    - otherwise → ``0``

    This yields a ``{-1, 0, +1}^d`` bag-level signature.

    Topics
    ------
    shards
        MDS shards with per-tile ``tile_bipolar_features`` (int8).
    bag_bipolar_features
        Computed on-the-fly from tile bipolars via mean + threshold.
    """

    VERSION = 2

    TOPICS = ['shards']

    @dataclass
    class CONFIG(Datablock.CONFIG):
        deep_feature_bag: DeepFeatureBag
        layer: str = 'output'
        bag_aggregation_threshold: float = 0.5

    def __init__(self, *args, **kwargs):
        Datablock.__init__(self, *args, **kwargs)

    @property
    def tilebag(self):
        """The source TileBag (delegated through the underlying bag)."""
        return self.cfg.deep_feature_bag.tilebag

    def __len__(self):
        return len(self.cfg.deep_feature_bag)

    # ── Validity ────────────────────────────────────────────────────

    def validtopic(self, topic=None):
        if topic == 'shards':
            return self.fs.exists(
                os.path.join(self.path('shards'), 'index.json')
            )
        return super().validtopic(topic)

    def valid(self, topic=None):
        if topic is not None:
            return self.validtopic(topic)
        return self.validtopics(reduce=True)

    # ── Build ───────────────────────────────────────────────────────

    def __build__(self, median):
        """Build bipolar features for this bag.

        Parameters
        ----------
        median : ndarray
            Per-dimension median vector ``(d,)``.  Passed via
            ``callable_kwargs`` from the clip.
        """

        # 1. Read raw tile features for the target layer.
        features = self.cfg.deep_feature_bag.layer_features(
            self.cfg.layer
        ).numpy()  # (n_tiles, d)
        n_tiles = features.shape[0]

        # 2. Tile bipolar: sign(features - median) → {-1, +1}^d
        tile_bipolar = np.sign(features - median).astype(np.int8)
        # Ensure no zeros from exact-median ties: map 0 → +1
        tile_bipolar[tile_bipolar == 0] = 1

        # 3. Write tile-level bipolar features as MDS shards.
        shards_dir = self.path('shards', ensure_dirpath=True)
        if self.fs.exists(shards_dir) and self.fs.ls(shards_dir):
            self.fs.rm(shards_dir, recursive=True)
            self.fs.mkdirs(shards_dir, exist_ok=True)

        columns = {
            'tile_bipolar_features': 'ndarray:int8',
            'tile_index': 'int32',
        }
        writer = MDSWriter(
            out=shards_dir,
            columns=columns,
            size_limit=1 << 23,  # 8 MB shard limit
            exist_ok=True,
        )
        try:
            for i in range(n_tiles):
                writer.write({
                    'tile_bipolar_features': tile_bipolar[i],
                    'tile_index': np.int32(i),
                })
        finally:
            writer.finish()

        self.log.verbose(
            f"Built bipolar features: {n_tiles} tiles, d={tile_bipolar.shape[1]}"
        )
        return self

    # ── Read ────────────────────────────────────────────────────────

    def __read__(self, topic):
        if topic == 'shards':
            return self.fs.ls(self.path('shards'))
        raise ValueError(f"Unknown topic: {topic!r}")

    @functools.cached_property
    def tile_bipolar_features(self):
        """Tile-level bipolar features ``{-1, +1}^d``, shape ``(n_tiles, d)``."""
        arrays = list(read_mds_samples(
            self.path('shards'), column='tile_bipolar_features'
        ))
        return np.stack(arrays)

    @functools.cached_property
    def bag_bipolar_features(self):
        """Bag-level bipolar features ``{-1, 0, +1}^d``, shape ``(d,)``."""
        tiles = self.tile_bipolar_features.astype(np.float32)
        bag_mean = tiles.mean(axis=0)
        threshold = self.cfg.bag_aggregation_threshold
        return np.where(
            np.abs(bag_mean) >= threshold,
            np.sign(bag_mean),
            0.0,
        ).astype(np.int8)

    @contextlib.contextmanager
    def dataset(self, *, shuffle: bool = False):
        """Yield a :class:`StreamingDataset` over this bag's MDS shards.

        Use as a context manager to ensure shared-memory file
        descriptors are released promptly::

            with bag.dataset() as ds:
                x = ds[0]['tile_bipolar_features']
        """
        if self.is_local_fs:
            ds = StreamingDataset(local=self.path('shards'), shuffle=shuffle)
        else:
            ds = StreamingDataset(remote=self.path('shards'), shuffle=shuffle)
        try:
            yield ds
        finally:
            del ds
            gc.collect()

    def data(self):
        """Bulk-read all MDS samples as a list of dicts."""
        return list(read_mds_samples(self.path('shards')))


class BipolarDeepFeatureClip(Clip):
    """Clip of :class:`BipolarDeepFeatureBag` shards built in parallel.

    CONFIG has two main params:

    * ``clip`` — the :class:`DeepFeatureClip` whose bags are bipolarized.
    * ``stats_probe`` — a :class:`DeepFeatureStatsProbe` that provides
      the per-dimension median for thresholding.  This can (and often
      should) come from a *different* fold (e.g. CALIBRATE) to avoid
      data leakage.

    Each bag builds its own tile-level bipolar features independently,
    enabling parallel construction via Datastack.
    """

    v2 = True
    VERSION = 3

    @dataclass
    class CONFIG(Datablock.CONFIG):
        clip: object              # DeepFeatureClip
        stats_probe: object       # DeepFeatureStatsProbe
        layer: str = 'output'
        bag_aggregation_threshold: float = 0.5

    def __init__(self, *args, **kwargs):
        kwargs.pop('v2', None)
        super().__init__(*args, v2=True, **kwargs)

    @property
    def feature_clip(self):
        """The underlying :class:`DeepFeatureClip`."""
        return self.cfg.clip

    @property
    def n_shards(self) -> int:
        return self.feature_clip.n_shards

    def __shard__(self, idx: int, deep_feature_bag=None):
        if deep_feature_bag is None:
            deep_feature_bag = self.feature_clip.bag(idx)
        return BipolarDeepFeatureBag(
            url=self.url,
            spec=dict(
                deep_feature_bag=dbx.quote(deep_feature_bag),
                layer=self.cfg.layer,
                bag_aggregation_threshold=self.cfg.bag_aggregation_threshold,
            ),
            tag=deep_feature_bag.tag,
        )

    class ShardMaker(Clip.ShardMaker):
        """Build a single bipolar bag shard."""
        def __call__(self, stack, *, build=True, median):
            shard = stack.__shard__(self.idx)
            shard.keyby = stack.keyby
            if build:
                shard.build(median=median)
            del shard
            gc.collect()

    def __split__(self, *args, **kwargs):
        """Precompute median; return ShardMakers."""
        # Read the median from the stats probe.
        median = self.cfg.stats_probe.tile_feature_median

        callable_kwargs = dict(
            build=True,
            median=median,
        )
        self.log.info(
            f"Precomputed median for {self.n_shards} shards; "
            f"ready for parallel bipolar build"
        )
        makers = [self.ShardMaker(idx) for idx in range(self.n_shards)]
        return makers, callable_kwargs

    def __read__(self, topic=None):
        if topic == 'bag_lens':
            return dbx.read_npz(self.path('bag_lens'), 'bag_lens')['bag_lens']
        raise ValueError(f"Unknown topic: {topic!r}")

    def __stack__(self, results=None):
        """Persist bag_lens after parallel build."""
        self.log.info(f"Stacking {self.n_shards} shards of {self.__class__.__name__}")
        bag_lens = [len(self.shard(i)) for i in tqdm(
            range(self.n_shards), desc="Stacking bipolar bag lens"
        )]
        dbx.write_npz(self.path('bag_lens', ensure_dirpath=True), bag_lens=bag_lens)
        self.log.info(f"Build complete: {self.__class__.__name__}")
        return self

    def dataset(
        self,
        *,
        shuffle: bool = False,
        include_tiles: bool = False,
        skip_invalid_bags: bool = False,
        batch_size: int | None = None,
    ) -> Dataset:
        """Return a unified streaming dataset over all bipolar bags.

        Each sample dict contains ``tile_bipolar_features`` (int8 ndarray).

        Parameters
        ----------
        shuffle : bool
            Whether to shuffle within the streaming dataset.
        include_tiles : bool
            If ``True``, each sample also contains a ``tile`` field
            loaded lazily from the source TileBag.
        skip_invalid_bags : bool
            If ``True``, silently skip bags whose MDS shards have not
            been built yet instead of raising.
        batch_size : int | None
            Passed to :class:`StreamingDataset` for deterministic
            resumption.
        """
        streams = []
        bag_lens_list = []
        n_skipped = 0
        valid_bags = []

        for bag in self.bags:
            if skip_invalid_bags and not bag.valid():
                n_skipped += 1
                continue
            if bag.is_local_fs:
                streams.append(Stream(local=bag.path('shards')))
            else:
                streams.append(Stream(remote=bag.path('shards')))
            valid_bags.append(bag)
            bag_lens_list.append(len(bag))

        if n_skipped:
            self.log.info(
                f"Skipped {n_skipped}/{n_skipped + len(streams)} "
                f"unbuilt bags in {self.__class__.__name__}.dataset()"
            )

        sd_kwargs = dict(streams=streams, shuffle=shuffle)
        if batch_size is not None:
            sd_kwargs['batch_size'] = batch_size
        base = StreamingDataset(**sd_kwargs)

        if include_tiles:
            return TileFeatureDataset(base, bag_lens_list, valid_bags)

        return base

    def valid(self):
        return self.validtopics(reduce=True)
