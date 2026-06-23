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
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

import dbx
from tqdm import tqdm
from dbx import (
    Datablock,
)

from streaming import MDSWriter, Stream, StreamingDataset

from autopath.autobits import Bag, Clip, DeepBackboneEvaluatorFactory, TileBag
from autopath.tools import mds_readers, read_mds_samples



# ═══════════════════════════════════════════════════════════════════════
#  Deep Feature Bags & Clips
# ═══════════════════════════════════════════════════════════════════════


class DeepFeatureBag(Bag):
    """A Bag that stores multi-layer activations from a DeepBackboneEvaluator.

    All data is stored as MDS shards for efficient streaming reads.
    Each MDS sample contains:

    * ``features_{layer}`` — per-layer activation ndarray (float32)
    * ``tile_index`` — position of this tile within the source TileBag
    * ``bag_name`` — slide/bag name string for provenance

    Topics
    ------
    shards
        MDS shards directory.  ``read('shards')`` returns an
        ``fs.ls`` listing of the shards directory.
    """

    VERSION = 7  # bump to 8 when rebuilding with tile_index + bag_name

    TOPICS = ['shards']

    @dataclass
    class CONFIG(Datablock.CONFIG):
        tilebag: TileBag
        evaluator_factory: DeepBackboneEvaluatorFactory
        shard_size: int = 1024    # samples per MDS shard

    def __init__(self, *args, gpu_batch_size: int = 64, device: str = "cuda", **kwargs):
        Datablock.__init__(self, *args, gpu_batch_size=gpu_batch_size, device=device, **kwargs)

    def __post_init__(self):
        # Compute feature names without loading the model.
        # Priority:
        #   1. factory.layer_names  — cheap property, no model load (preferred)
        #   2. factory.cfg          — derive from config fields
        #   3. factory.evaluator()  — last resort: instantiates the model
        factory = self.cfg.evaluator_factory
        if hasattr(factory, 'layer_names'):
            self._feature_names = list(factory.layer_names)
        elif hasattr(factory, 'cfg'):
            names = []
            for block in getattr(factory.cfg, 'capture_blocks', []):
                names.append(f"block.{block}")
            for layer in getattr(factory.cfg, 'capture_layers', []):
                names.append(layer)
            if getattr(factory.cfg, 'capture_outputs', True):
                names.append('output')
            self._feature_names = names
        elif hasattr(factory, 'evaluator'):
            # Last resort — loads the model; avoid if factory has layer_names.
            self._feature_names = factory.evaluator(log=self.log).layer_names
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
        columns["bag_name"] = "str"

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
                bag_name = tilebag.name
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
                    sample["bag_name"] = bag_name
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

        Each sample is a dict with keys ``features_{name}`` (ndarray).

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
        """Read the annotation dict from the source tilebag's MDS sample *i*.

        Annotations are stored in the tile dataset, not the feature
        dataset.  This method delegates to the tilebag's MDS shards.

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
            for reader in mds_readers(self.tilebag.path('shards')):
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
    """A clip of :class:`DeepFeatureBag` blocks built in parallel via Datastack.

    Orchestrates building one :class:`DeepFeatureBag` per tile-bag in the
    source ``tilebagclip``, using the configured
    :class:`DeepBackboneEvaluatorFactory`.

    Uses the v2 :class:`Datastack` API: :meth:`__split_v2__` returns
    device-assigned :class:`BlockMaker` callables and shared state as
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
    def n_blocks(self) -> int:
        return self.cfg.tilebagclip.n_blocks

    class BlockMaker(Datastack.BlockMaker):
        """Build a single block using shared state from ``callable_kwargs``.

        Carries its assigned ``device`` so the assignment survives
        serialisation across process boundaries.

        The ``evaluator_factory`` is received via the executor's
        ``ctx_kwargs``.  The executor ``dbx.eval()``s all ctx kwargs,
        so a quoted factory spec is evaluated once per worker.  The
        factory caches its evaluator, so all BlockMakers within a
        worker share the same evaluator instance.
        """
        def __init__(self, idx: int, *, device: str = "cuda"):
            super().__init__(idx)
            self.device = device

        def __call__(self, stack, *, build=True,
                     tilebags, evaluator_factory):
            block = stack.__block__(self.idx, device=self.device,
                                    tilebag=tilebags[self.idx])
            block.keyby = stack.keyby
            if build:
                block.build(
                    evaluator=evaluator_factory.evaluator(
                        device=self.device, log=stack.log,
                    ),
                    tilebag=tilebags[self.idx],
                )
            del block
            gc.collect()

    def __block__(self, idx: int, tilebagclip=None, device: str = "cuda",
                  tilebag=None):
        if tilebag is None:
            if tilebagclip is None:
                tilebagclip = self.cfg.tilebagclip
            tilebag = tilebagclip.block(idx)
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
        """Precompute tilebags; return device-assigned BlockMakers.

        All shared state is returned in ``callable_kwargs`` so it
        flows through the executor's ``ctx_kwargs`` mechanism — no
        ``self``-attribute side effects.

        The ``evaluator_factory`` is passed through ``callable_kwargs``.
        The executor ``dbx.eval()``s it once per worker, and the
        factory caches its evaluator, giving one evaluator per worker.

        Returns
        -------
        callables : list[BlockMaker]
            One BlockMaker per block, each carrying its target device.
        callable_kwargs : dict
            Shared state forwarded to each BlockMaker.__call__:
            ``build``, ``tilebags``, ``evaluator_factory``.
        """
        devices = self._devices
        # Precompute flat tilebag list to avoid re-forming the fold.
        tilebagclip = self.cfg.tilebagclip
        tilebags = [tilebagclip.block(idx) for idx in range(self.n_blocks)]

        callable_kwargs = dict(
            build=True,
            tilebags=tilebags,
            evaluator_factory=self.cfg.evaluator_factory,
        )
        self.log.info(
            f"Precomputed {self.n_blocks} tile-bags; "
            f"evaluator_factory passed to executor (devices={devices})"
        )
        # Assign devices to BlockMakers based on executor chunking.
        # The executor splits callables into contiguous chunks via
        # np.array_split — worker w gets chunk w.  Assign the device
        # for each block based on which chunk (worker) it will land in.
        n_workers = len(devices)
        chunk_boundaries = np.array_split(range(self.n_blocks), n_workers)
        block_device = {}
        for worker_idx, chunk in enumerate(chunk_boundaries):
            dev = devices[worker_idx % len(devices)]
            for idx in chunk:
                block_device[idx] = dev
        makers = [
            self.BlockMaker(idx, device=block_device[idx])
            for idx in range(self.n_blocks)
        ]
        self.log.info(
            f"Split {self.__class__.__name__}: {len(makers)} blocks across "
            f"{n_workers} workers on devices {devices}"
        )
        return makers, callable_kwargs

    def __read__(self, topic=None):
        if topic == 'bag_lens':
            return dbx.read_npz(self.path('bag_lens'), 'bag_lens')['bag_lens']
        raise ValueError(f"Unknown topic: {topic!r}")

    def __stack__(self, results=None):
        """Persist bag_lens after parallel build."""
        self.log.info(f"Stacking {self.n_blocks} blocks of {self.__class__.__name__}")
        bag_lens = [len(self.block(i)) for i in tqdm(
            range(self.n_blocks), desc="Stacking bag lens"
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
                tilebagclip.block(idx) for idx in range(self.n_blocks)
            ]

    def blocks(self):
        self._ensure_precomputed_tilebags()
        return super().blocks()

    def dataset(
        self,
        *,
        shuffle: bool = False,
        skip_invalid_bags: bool = False,
        batch_size: int | None = None,
        **streaming_kwargs,
    ) -> Dataset:
        """Return a unified, labeled dataset over all bags.

        Uses the MDS multi-stream API to create a **single**
        :class:`StreamingDataset` backed by one :class:`Stream` per bag.
        This avoids opening hundreds of separate shared-memory segments.

        Each sample dict contains:

        * ``features_{layer_name}`` — per-layer activation ndarray

        Parameters
        ----------
        shuffle : bool
            Whether to shuffle within the streaming dataset.
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
        n_skipped = 0

        for i, bag in enumerate(self.bags):
            if skip_invalid_bags and not bag.valid():
                n_skipped += 1
                continue
            if bag._is_local_fs:
                streams.append(Stream(local=bag.path('shards')))
            else:
                streams.append(Stream(remote=bag.path('shards')))

        if n_skipped:
            self.log.info(
                f"Skipped {n_skipped}/{n_skipped + len(streams)} "
                f"unbuilt bags in {self.__class__.__name__}.dataset()"
            )

        # One StreamingDataset with N streams — one shared-memory segment.
        sd_kwargs = dict(streams=streams, shuffle=shuffle, **streaming_kwargs)
        if batch_size is not None:
            sd_kwargs['batch_size'] = batch_size
        return StreamingDataset(**sd_kwargs)

    def tile_feature_dataset(
        self,
        *,
        shuffle: bool = False,
        skip_invalid_bags: bool = False,
        batch_size: int | None = None,
        validate_zipping: bool = False,
        **streaming_kwargs,
    ):
        """Return a :class:`ZipStreamingDataset` aligning features with tiles.

        Iterates ``self.bags`` and for each valid bag creates both a
        feature stream and a tile stream (from ``bag.tilebag``) in the
        **same order**, guaranteeing that ``ds[i]`` pairs the correct
        feature with its source tile.  Invalid bags are skipped
        consistently from both sides.

        Parameters
        ----------
        shuffle : bool
            Whether to shuffle within both streaming datasets.
        skip_invalid_bags : bool
            If ``True``, silently skip bags whose feature blocks have
            not been built yet.
        batch_size : int | None
            Passed to both :class:`StreamingDataset` instances.
        validate_zipping : bool
            If ``True``, attach a per-sample validator that checks
            ``bag_name`` and ``tile_index`` consistency between the
            feature and tile sides.  Requires provenance columns in
            the feature shards (version ≥ 8).

        Returns
        -------
        ZipStreamingDataset
            Merged dataset with both ``features_*`` and ``tile`` columns.
        """
        from autopath.autobits import ZipStreamingDataset

        feature_streams = []
        tile_streams = []
        n_skipped = 0

        for bag in self.bags:
            if skip_invalid_bags and not bag.valid():
                n_skipped += 1
                continue
            # Feature stream
            if bag._is_local_fs:
                feature_streams.append(Stream(local=bag.path('shards')))
            else:
                feature_streams.append(Stream(remote=bag.path('shards')))
            # Corresponding tile stream — same bag order
            tilebag = bag.tilebag
            if tilebag._is_local_fs:
                tile_streams.append(Stream(local=tilebag.path('shards')))
            else:
                tile_streams.append(Stream(remote=tilebag.path('shards')))

        if n_skipped:
            self.log.info(
                f"Skipped {n_skipped}/{n_skipped + len(feature_streams)} "
                f"unbuilt bags in {self.__class__.__name__}.tile_feature_dataset()"
            )

        sd_kwargs = dict(shuffle=shuffle, **streaming_kwargs)
        if batch_size is not None:
            sd_kwargs['batch_size'] = batch_size

        feature_ds = StreamingDataset(streams=feature_streams, **sd_kwargs)
        tile_ds = StreamingDataset(streams=tile_streams, **sd_kwargs)

        # Build a zip validator that checks provenance consistency
        # between the feature and tile sides.  Only enabled when
        # validate_zipping=True and the feature shards contain
        # bag_name / tile_index (version >= 8).
        zip_validator = None
        if validate_zipping:
            first_sample = feature_ds[0]
            has_provenance = 'bag_name' in first_sample and 'tile_index' in first_sample
            if has_provenance:
                def _validate_tile_feature_zip(idx, feature_sample, tile_sample):
                    f_bag = feature_sample.get('bag_name')
                    t_bag = tile_sample.get('bag_name')
                    f_idx = feature_sample.get('tile_index')
                    t_idx = tile_sample.get('tile_index')
                    if f_bag != t_bag or f_idx != t_idx:
                        raise ValueError(
                            f"Tile-feature zip mismatch at flat index {idx}: "
                            f"feature side bag_name={f_bag!r} tile_index={f_idx}, "
                            f"tile side bag_name={t_bag!r} tile_index={t_idx}"
                        )
                zip_validator = _validate_tile_feature_zip

        return ZipStreamingDataset(feature_ds, tile_ds, zip_validator=zip_validator)

    def verify_batchsize_invariance(
        self,
        *,
        bag_index: int = 0,
        tile_index: int = 0,
        n_repeats: int = 3,
        build_batch_size: int = 64,
        device: str = 'cuda',
    ):
        """Compare evaluator output at BS=1 vs BS=``build_batch_size``.

        Re-evaluates the tile at (``bag_index``, ``tile_index``) and
        compares both batch-size variants against the stored feature
        at the corresponding flat dataset position.

        Parameters
        ----------
        bag_index : int
            Which bag (block) within the clip to probe.
        tile_index : int
            Which tile within that bag to probe.
        n_repeats : int
            Number of repeated forward passes per batch size.
        build_batch_size : int
            Batch size used during the original build (default 64).
        device : str
            CUDA device string.

        Returns a dict with ``bs1_vs_bsN``, ``stored_vs_bs1``,
        ``stored_vs_bsN`` max-absolute-error values.
        """
        evaluator = self.cfg.evaluator_factory.evaluator(device=device, log=self.log)

        tilebag = self.cfg.tilebagclip.block(bag_index)
        tiles = tilebag.tiles  # NCHW uint8
        tile = tiles[tile_index:tile_index + 1]
        self.log.info(
            f"Probing bag={bag_index} ({tilebag.name}) tile={tile_index}  "
            f"shape={tile.shape} dtype={tile.dtype}"
        )

        # --- BS=1 ---
        results_bs1 = []
        for _ in range(n_repeats):
            r = evaluator(tile)
            results_bs1.append(r['output'][0].numpy().copy())
            evaluator.clear()

        # --- BS=build_batch_size (same tile repeated) ---
        batch = tile.expand(build_batch_size, -1, -1, -1)
        results_bsN = []
        for _ in range(n_repeats):
            r = evaluator(batch)
            results_bsN.append(r['output'][0].numpy().copy())
            evaluator.clear()

        bs1 = np.stack(results_bs1)
        bsN = np.stack(results_bsN)
        bs1_mean = bs1.mean(axis=0)
        bsN_mean = bsN.mean(axis=0)

        self.log.info(
            f"BS1 repeat std:  max={bs1.std(0).max():.2e}  mean={bs1.std(0).mean():.2e}"
        )
        self.log.info(
            f"BS{build_batch_size} repeat std: "
            f"max={bsN.std(0).max():.2e}  mean={bsN.std(0).mean():.2e}"
        )

        diff = np.abs(bs1_mean - bsN_mean)
        denom = np.maximum(np.maximum(np.abs(bs1_mean), np.abs(bsN_mean)), 1.0)
        rel = diff / denom
        self.log.info(
            f"BS1 vs BS{build_batch_size}:  "
            f"max_abs={diff.max():.2e}  mean_abs={diff.mean():.2e}  "
            f"max_rel={rel.max():.2e}  mean_rel={rel.mean():.2e}"
        )

        # --- Compare against stored features ---
        # Compute the flat dataset offset: sum of bag lengths before
        # this bag, plus tile_index within the bag.
        flat_offset = tile_index
        for i in range(bag_index):
            flat_offset += len(self.bags[i])
        ds = self.dataset(shuffle=False, batch_size=1)
        sample = ds[flat_offset]
        stored = sample['features_output']
        if isinstance(stored, torch.Tensor):
            stored = stored.numpy()

        self.log.info(
            f"Stored feature at flat_offset={flat_offset} (bag={bag_index}, tile={tile_index})"
        )

        for label, fresh in [('BS1', bs1_mean), (f'BS{build_batch_size}', bsN_mean)]:
            d = np.abs(stored - fresh)
            den = np.maximum(np.maximum(np.abs(stored), np.abs(fresh)), 1.0)
            r = d / den
            self.log.info(
                f"Stored vs {label}:  "
                f"max_abs={d.max():.2e}  mean_abs={d.mean():.2e}  "
                f"max_rel={r.max():.2e}  mean_rel={r.mean():.2e}"
            )

        return {
            'bs1_vs_bsN': float(diff.max()),
            'stored_vs_bs1': float(np.abs(stored - bs1_mean).max()),
            'stored_vs_bsN': float(np.abs(stored - bsN_mean).max()),
        }


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
        MDS shards with per-tile ``bipolar_features_{layer}`` (int8)
        and per-bag ``bag_bipolar_features_{layer}`` (int8, identical
        across all tiles in the same bag).
    """

    VERSION = 5

    TOPICS = ['shards']

    @dataclass
    class CONFIG(Datablock.CONFIG):
        deep_feature_bag: DeepFeatureBag
        layer: str = 'output'
        bag_aggregation_threshold: float = 0.5
        ternarize_tiles: bool = False

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

        Reads tile-level features from the underlying
        :class:`DeepFeatureBag`, computes tile-level and bag-level
        bipolar encodings, and writes them into MDS shards.

        Columns written:

        * ``bipolar_features_{layer}``  — per-tile ``{-1, +1}^d`` (int8)
        * ``bag_bipolar_features_{layer}`` — per-bag ``{-1, 0, +1}^d``
          (int8, identical for every tile in the bag)

        Parameters
        ----------
        median : ndarray
            Per-dimension median vector ``(d,)``.  Passed via
            ``callable_kwargs`` from the clip.
        """
        layer = self.cfg.layer
        tile_col = f'bipolar_features_{layer}'
        bag_col = f'bag_bipolar_features_{layer}'

        # 1. Read raw tile features for the target layer.
        features = self.cfg.deep_feature_bag.layer_features(
            layer
        ).numpy()  # (n_tiles, d)
        n_tiles = features.shape[0]

        # 2. Tile bipolar: sign(features - median) → {-1, +1}^d
        tile_bipolar = np.sign(features - median).astype(np.int8)
        # Ensure no zeros from exact-median ties: map 0 → +1
        tile_bipolar[tile_bipolar == 0] = 1

        # 3. Optional ternarization: zero out tile dimensions where
        #    the bag's tiles disagree (bag mean rounds to 0).
        if self.cfg.ternarize_tiles:
            bag_mean = tile_bipolar.astype(np.float32).mean(axis=0)  # (d,)
            uncertain_mask = (np.round(bag_mean).astype(np.int8) == 0)  # (d,)
            n_uncertain = int(uncertain_mask.sum())
            tile_bipolar[:, uncertain_mask] = 0
            self.log.verbose(
                f"Ternarized tiles: zeroed {n_uncertain}/{tile_bipolar.shape[1]} "
                f"uncertain dimensions (bag mean ≈ 0)"
            )

        # 4. Bag-level bipolar: aggregate tile bipolars via mean + threshold.
        bag_mean = tile_bipolar.astype(np.float32).mean(axis=0)  # (d,)
        threshold = self.cfg.bag_aggregation_threshold
        bag_bipolar = np.where(
            np.abs(bag_mean) >= threshold,
            np.sign(bag_mean),
            0.0,
        ).astype(np.int8)  # (d,)

        # 5. Write tile-level + bag-level bipolar features as MDS shards.
        shards_dir = self.path('shards', ensure_dirpath=True)
        if self.fs.exists(shards_dir) and self.fs.ls(shards_dir):
            self.fs.rm(shards_dir, recursive=True)
            self.fs.mkdirs(shards_dir, exist_ok=True)

        columns = {
            tile_col: 'ndarray:int8',
            bag_col: 'ndarray:int8',
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
                    tile_col: tile_bipolar[i],
                    bag_col: bag_bipolar,
                })
        finally:
            writer.finish()

        self.log.verbose(
            f"Built bipolar features: {n_tiles} tiles, d={tile_bipolar.shape[1]}, "
            f"bag dims with |mean|>={threshold}: "
            f"{int((np.abs(bag_mean) >= threshold).sum())}/{tile_bipolar.shape[1]}"
        )
        return self

    # ── Read ────────────────────────────────────────────────────────

    def __read__(self, topic):
        if topic == 'shards':
            return self.fs.ls(self.path('shards'))
        raise ValueError(f"Unknown topic: {topic!r}")

    @functools.cached_property
    def bipolar_features(self):
        """Tile-level bipolar features ``{-1, +1}^d``, shape ``(n_tiles, d)``."""
        tile_col = f'bipolar_features_{self.cfg.layer}'
        arrays = list(read_mds_samples(
            self.path('shards'), column=tile_col
        ))
        return np.stack(arrays)

    @functools.cached_property
    def bag_bipolar_features(self):
        """Bag-level bipolar features ``{-1, 0, +1}^d``, shape ``(d,)``.

        Read from the first MDS sample (identical across all tiles).
        """
        bag_col = f'bag_bipolar_features_{self.cfg.layer}'
        samples = read_mds_samples(self.path('shards'), column=bag_col)
        return next(iter(samples))

    def layer_features(self, layer: str = None):
        """Read tile-level bipolar features as a float32 tensor.

        Implements the same API as :meth:`DeepFeatureBag.layer_features`
        so that bipolar bags can be used with
        :class:`DeepFeatureAffineLogisticProbe` and
        :class:`DeepFeatureStatsProbe`.

        The ``layer`` argument is accepted for API compatibility but
        ignored — bipolar bags contain only one feature set.

        Returns
        -------
        Tensor
            Shape ``(n_tiles, d)``, dtype ``float32``.
        """
        return torch.from_numpy(
            self.bipolar_features.astype(np.float32)
        )

    def annotations(self, i=0) -> dict | None:
        """Delegate annotations to the underlying :class:`DeepFeatureBag`."""
        return self.cfg.deep_feature_bag.annotations(i=i)

    @contextlib.contextmanager
    def dataset(self, *, shuffle: bool = False):
        """Yield a :class:`StreamingDataset` over this bag's MDS shards.

        Use as a context manager to ensure shared-memory file
        descriptors are released promptly::

            with bag.dataset() as ds:
                x = ds[0][f'bipolar_features_{layer}']
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
    """Clip of :class:`BipolarDeepFeatureBag` blocks built in parallel.

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
    VERSION = 5

    @dataclass
    class CONFIG(Datablock.CONFIG):
        clip: object              # DeepFeatureClip
        stats_probe: object       # DeepFeatureStatsProbe
        layer: str = 'output'
        bag_aggregation_threshold: float = 0.5
        ternarize_tiles: bool = False

    def __init__(self, *args, **kwargs):
        kwargs.pop('v2', None)
        super().__init__(*args, v2=True, **kwargs)

    @property
    def feature_clip(self):
        """The underlying :class:`DeepFeatureClip`."""
        return self.cfg.clip

    @property
    def n_blocks(self) -> int:
        return self.feature_clip.n_blocks

    def __block__(self, idx: int, deep_feature_bag=None):
        if deep_feature_bag is None:
            deep_feature_bag = self.feature_clip.bag(idx)
        return BipolarDeepFeatureBag(
            url=self.url,
            spec=dict(
                deep_feature_bag=dbx.quote(deep_feature_bag),
                layer=self.cfg.layer,
                bag_aggregation_threshold=self.cfg.bag_aggregation_threshold,
                ternarize_tiles=self.cfg.ternarize_tiles,
            ),
            tag=deep_feature_bag.tag,
        )

    class BlockMaker(Datastack.BlockMaker):
        """Build a single bipolar bag block."""
        def __call__(self, stack, *, build=True, median):
            block = stack.__block__(self.idx)
            block.keyby = stack.keyby
            if build:
                block.build(median=median)
            del block
            gc.collect()

    def __split__(self, *args, **kwargs):
        """Precompute median; return BlockMakers."""
        # Read the median from the stats probe.
        median = self.cfg.stats_probe.tile_feature_median

        callable_kwargs = dict(
            build=True,
            median=median,
        )
        self.log.info(
            f"Precomputed median for {self.n_blocks} blocks; "
            f"ready for parallel bipolar build"
        )
        makers = [self.BlockMaker(idx) for idx in range(self.n_blocks)]
        return makers, callable_kwargs

    def __read__(self, topic=None):
        if topic == 'bag_lens':
            return dbx.read_npz(self.path('bag_lens'), 'bag_lens')['bag_lens']
        raise ValueError(f"Unknown topic: {topic!r}")

    def __stack__(self, results=None):
        """Persist bag_lens after parallel build."""
        self.log.info(f"Stacking {self.n_blocks} blocks of {self.__class__.__name__}")
        bag_lens = [len(self.block(i)) for i in tqdm(
            range(self.n_blocks), desc="Stacking bipolar bag lens"
        )]
        dbx.write_npz(self.path('bag_lens', ensure_dirpath=True), bag_lens=bag_lens)
        self.log.info(f"Build complete: {self.__class__.__name__}")
        return self

    def dataset(
        self,
        *,
        shuffle: bool = False,
        skip_invalid_bags: bool = False,
        batch_size: int | None = None,
        **streaming_kwargs,
    ) -> Dataset:
        """Return a streaming dataset over all bipolar feature bags.

        Each sample dict contains ``bipolar_features_{layer}`` and
        ``bag_bipolar_features_{layer}`` (both int8 ndarray).

        To also include tile images, zip this dataset with the
        corresponding tile clip's dataset using
        :class:`ZipStreamingDataset`::

            bipolar_ds = bipolar_clip.dataset(...)
            tile_ds = bipolar_clip.cfg.clip.cfg.tilebagclip.dataset(...)
            ds = ZipStreamingDataset(bipolar_ds, tile_ds)

        Parameters
        ----------
        shuffle : bool
            Whether to shuffle within the streaming dataset.
        skip_invalid_bags : bool
            If ``True``, silently skip bags whose MDS shards have not
            been built yet instead of raising.
        batch_size : int | None
            Passed to :class:`StreamingDataset` for deterministic
            resumption.
        """
        streams = []
        n_skipped = 0

        for bag in self.bags:
            if skip_invalid_bags and not bag.valid():
                n_skipped += 1
                continue
            if bag.is_local_fs:
                streams.append(Stream(local=bag.path('shards')))
            else:
                streams.append(Stream(remote=bag.path('shards')))

        if n_skipped:
            self.log.info(
                f"Skipped {n_skipped}/{n_skipped + len(streams)} "
                f"unbuilt bags in {self.__class__.__name__}.dataset()"
            )

        sd_kwargs = dict(streams=streams, shuffle=shuffle, **streaming_kwargs)
        if batch_size is not None:
            sd_kwargs['batch_size'] = batch_size
        return StreamingDataset(**sd_kwargs)

    def valid(self):
        return self.validtopics(reduce=True)
