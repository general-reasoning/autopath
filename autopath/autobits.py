import collections
from dataclasses import dataclass, field
import functools
import gc
import math
import threading
import traceback as tb
from typing import Union

import tqdm

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset


import dbx
from dbx import (
    Datablock,
    Datastack,
    InlineCallableExecutor,
    InlineDatablocksBuilder,
    MultithreadingCallableExecutor,
    MultithreadingDatablocksBuilder,
    MultiprocessingCallableExecutor,
    MultiprocessingDatablocksBuilder,
    RayCallableExecutor,
    RayDatablocksBuilder,
    UNSAFE_allowed,
)



class Bag(Datablock):
    
    @property
    def name(self):
        raise NotImplementedError

    @property
    def label(self):
        raise NotImplementedError


class TileBag(Bag):
    """A bag of image tiles backed by a tensor.

    Combines the former ``TileBlock`` (tensor-backed tile storage) and
    ``TileBag`` (Bag subclass) into a single class.
    """

    @functools.cached_property
    def tiles(self):
        return self.tensor

    @property
    def labels(self):
        raise NotImplementedError()



class Clip(Datastack):
    """Abstract clip of bags, built in parallel via :class:`Datastack`.

    Subclasses must implement:

    * ``n_blocks``  — number of bags (property)
    * ``__block__(idx)`` — return the :class:`Bag` at *idx*

    After all bags are built in parallel, :meth:`__stack__` persists
    ``bag_lens.npz`` so that lengths can be read back without
    materializing bags.
    """

    v2 = True
    
    TOPICS = {"bag_lens": "bag_lens.npz"}

    def __len__(self):
        return sum(self.bag_lens)

    def __stack__(self, results=None):
        """Persist bag lengths after all bags have been built."""
        self.log.verbose(f"Stacking bag lens")
        bag_lens = [len(self.block(i)) for i in tqdm.tqdm(range(self.n_blocks), desc="Stacking bag lens")]
        dbx.write_npz(self.path('bag_lens', ensure_dirpath=True), bag_lens=bag_lens)
        return self

    def __read__(self):
        bag_lens = dbx.read_npz(self.path('bag_lens'), 'bag_lens')['bag_lens']
        return bag_lens

    @functools.cached_property
    def bag_lens(self):
        return self.read()

    # ── Bag-centric aliases ─────────────────────────────────────────
    @property
    def n_bags(self):
        return self.n_blocks

    def bag(self, idx: int):
        return self.block(idx)

    @property
    def bags(self):
        return self.blocks()

    @property
    def block_lens(self):
        return self.bag_lens

    def validbags(self):
        """Return a boolean array indicating which bags are valid."""
        return np.array([bag.valid() for bag in self.bags])


    def UNSAFE_clear_bags(self, *, OVERRIDE: bool = False):
        return self.UNSAFE_clear_blocks(OVERRIDE=OVERRIDE)

    def UNSAFE_copy_from(self, anchorpath: str, bag_anchorpath: str, *, overwrite: bool = False):
        super().UNSAFE_copy_from(anchorpath, overwrite=overwrite)
        self.UNSAFE_copy_bags_from(bag_anchorpath, overwrite=overwrite)
        return self

    def UNSAFE_copy_bags_from(self, bag_anchorpath: str, *, overwrite: bool = False):
        for bag in self.bags:
            bag.UNSAFE_copy_from(bag_anchorpath, overwrite=overwrite)
        return self
    

class Partition(Datablock):
    TOPICS = {"bag_indices": "bag_indices.npz", 
                  "bag_lens":    "bag_lens.npz",
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
        N = self.cfg.clip.n_bags
        self.log.info(f"Building partition out of {N} bags using fold fractions {self.cfg.fold_fractions}")
        bag_indices = self._compute_fold_indices()

        bag_lens = {}
        self.log.verbose(f"Computing bag lens for {len(self.cfg.fold_fractions)} folds: BEGIN")
        if self.verbose:
            fold_itor = tqdm.tqdm(enumerate(self.cfg.fold_fractions), total=len(self.cfg.fold_fractions))
        else:
            fold_itor = enumerate(self.cfg.fold_fractions)
        for fold, fraction in fold_itor:
            fold_key = str(fold)
            self.log.verbose(f"Computing bag lens for fold {fold}: BEGIN")
            if self.verbose:
                bag_itor = tqdm.tqdm(bag_indices[fold_key])
            else:
                bag_itor = bag_indices[fold_key]
            bag_lens[fold_key] = np.array([len(self.cfg.clip.bag(i)) for i in bag_itor])
            self.log.verbose(f"Computing bag lens for fold {fold}: END")
        self.log.verbose(f"Computing bag lens for {len(self.cfg.fold_fractions)} folds: END")
        self.log.verbose(f"Writing bag indices and lens: BEGIN")
        dbx.write_npz(self.path('bag_indices', ensure_dirpath=True), **bag_indices)
        dbx.write_npz(self.path('bag_lens', ensure_dirpath=True), **bag_lens)
        self.log.verbose(f"Writing bag indices and lens: END")
        return self
    
    def __read__(self, topic):
        keys = [f"{i}" for i in range(len(self.cfg.fold_fractions))]
        dict = dbx.read_npz(self.path(topic), *keys)
        return dict

    def _resolve_fold(self, fold):
        """Resolve fold key, supporting negative indices like Python lists."""
        n = len(self.cfg.fold_fractions)
        idx = int(fold)
        if idx < 0:
            idx = n + idx
        return str(idx)

    def bags(self, fold):
        return [self.cfg.clip.block(i) for i in self.bag_indices(fold)]
    
    def bag_lens(self, fold):
        return self.read("bag_lens")[self._resolve_fold(fold)]

    def bag_indices(self, fold):
        if self.validtopic("bag_indices"):
            return self.read("bag_indices")[self._resolve_fold(fold)]
        return self._compute_fold_indices()[self._resolve_fold(fold)]

    def bag(self, fold, idx: int):
        return self.cfg.clip.block(self.bag_indices(fold)[idx])

    def _compute_fold_indices(self):
        """Recompute per-fold bag index arrays from config (same logic as __build__).

        This is a pure function of ``clip.n_bags``, ``fold_fractions``,
        and ``seed`` — it does **not** require the partition to have been
        built or persisted.

        Returns
        -------
        dict[str, ndarray]
            Mapping from fold key (``"0"``, ``"1"``, …) to the array of
            bag indices belonging to that fold.
        """
        N = self.cfg.clip.n_bags
        rng = np.random.RandomState(self.cfg.seed)
        perm = rng.permutation(N)
        fold_indices = {}
        Klo = 0
        for fold, fraction in enumerate(self.cfg.fold_fractions):
            k = int(math.ceil(N * fraction))
            Khi = min(Klo + k, N)
            fold_indices[str(fold)] = perm[Klo:Khi]
            Klo = Khi
        return fold_indices

    def n_bags(self, fold):
        return len(self.bag_indices(fold))


class Fold(Clip):
    @dataclass
    class CONFIG:
        partition: Partition
        fold: Union[str, int]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, v2=True, **kwargs)

    def __post_init__(self):
        return self
    
    def valid(self):
        return self.cfg.partition.valid()

    @property
    def n_blocks(self):
        return self.cfg.partition.n_bags(self.cfg.fold)

    def __block__(self, idx: int):
        return self.cfg.partition.bag(self.cfg.fold, idx)

    def __split_v2__(self, *args, **kwargs):
        """Bags are managed by the Partition — no parallel work needed."""
        return [], dict()

    def __stack_v2__(self, results):
        """Persist bag_lens after (vacuous) parallel phase."""
        return super().__stack__(results)

    @functools.cached_property
    def bag_lens(self):
        return self.cfg.partition.bag_lens(self.cfg.fold)


# ═══════════════════════════════════════════════════════════════════════
#  Abstract Deep Backbone Evaluator & Factory
# ═══════════════════════════════════════════════════════════════════════

class DeepBackboneEvaluator:
    """Model-agnostic base for hook-based multi-layer activation capture.

    Subclasses must implement:

    * :meth:`layer_names` — ordered list of capture keys produced by
      ``__call__``.
    * :meth:`__call__` — run a forward pass and return a
      ``dict[str, Tensor]`` mapping capture keys to activation tensors,
      plus an ``"output"`` key for the backbone output.
    * :meth:`clear` — release captured tensors and free accelerator
      memory.

    Optionally override :meth:`__pre_call__` to lazily register hooks.

    Properties
    ----------
    layer_names : list[str]
        Ordered capture keys that ``__call__`` will produce.
    layer_features : dict[str, Tensor]
        Most recently captured activations (read-only snapshot).
    """

    def __init__(self, *, device: str = "cuda", log: dbx.Logger = dbx.Logger(stack_depth=3)):
        self.device = device
        self.log = log

    @property
    def layer_names(self):
        """Return the ordered list of capture keys that ``__call__`` produces.

        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def __pre_call__(self):
        """Hook called before each forward pass (e.g. to lazily register hooks)."""
        pass

    def __call__(self, x):
        """Run forward pass on batch *x* and return captured activations.

        Returns
        -------
        dict[str, Tensor]
            Mapping from capture-key to activation tensor, plus
            ``"output"`` for the backbone's own output.

        Must be overridden by subclasses.
        """
        raise NotImplementedError

    def clear(self):
        """Release captured tensors and free accelerator memory."""
        gc.collect()
        torch.cuda.empty_cache()
        return self

    @property
    def layer_features(self):
        """Most recently captured activations (read-only snapshot).

        Must be overridden by subclasses.
        """
        raise NotImplementedError


class DeepBackboneEvaluatorFactory(Datablock):
    """Abstract Datablock used for spec-based dependency tracking.

    This Datablock does **not** build or persist anything itself.
    It exists so that :class:`DeepFeatureBag` and :class:`DeepFeatureClip`
    can declare their evaluator configuration as a spec dependency,
    enabling deterministic hashing and lineage tracking.

    Concrete subclasses may extend ``CONFIG`` with model-specific fields
    and override :meth:`evaluator` to return a ready-to-use
    :class:`DeepBackboneEvaluator`.
    """

    @dataclass
    class CONFIG:
        capture_blocks: list = field(default_factory=list)  # list[int] — transformer block indices
        capture_layers: list = field(default_factory=list)  # list[str] — named layers
        capture_outputs: bool = True   # capture model output as 'features'
        cls_token_only: bool = False   # capture only CLS token activations

    def evaluator(self, *, device: str = None, log: dbx.Logger = None):
        """Create a live :class:`DeepBackboneEvaluator`.

        Must be overridden by subclasses.
        """
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════════════════
#  Zip wrapper for paired streaming datasets
# ═══════════════════════════════════════════════════════════════════════


class ZipStreamingDataset(Dataset):
    """Pairs multiple :class:`StreamingDataset` objects by index.

    All datasets must have the same length.  ``__getitem__`` merges
    the sample dicts from all datasets into a single dict.

    This avoids opening per-bag tile datasets inside DataLoader
    workers (which breaks DDP barriers) by creating multiple
    rank-coordinated ``StreamingDataset`` objects at the top level.

    Parameters
    ----------
    *datasets
        One or more ``StreamingDataset`` instances to zip.
    zip_validator : callable | None
        Optional callable ``(idx, *samples) → None`` that is invoked
        with the flat index and the individual sample dicts **before**
        they are merged.  It should raise on inconsistency (e.g. when
        ``bag_name`` or ``tile_index`` disagree across the datasets).
    """

    def __init__(self, *datasets, zip_validator=None):
        lengths = [len(d) for d in datasets]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"ZipStreamingDataset requires datasets of equal length, "
                f"got {lengths}"
            )
        self.datasets = datasets
        self.zip_validator = zip_validator

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        samples = [ds[idx] for ds in self.datasets]
        if self.zip_validator is not None:
            self.zip_validator(idx, *samples)
        merged = {}
        for sample in samples:
            for k, v in sample.items():
                if v is not None:
                    merged[k] = v
        return merged


class ValidateTileFeatureZipStreamingDataset:
    """Validate that a zipped tile+feature dataset is consistent.

    Takes a :class:`ZipStreamingDataset` (pairing tiles with stored
    features) and a :class:`DeepBackboneEvaluator`, re-runs the
    evaluator on the tiles, and compares the result against the stored
    feature columns.

    The feature columns in the dataset follow the naming convention
    ``features_{safe_layer}`` where ``safe_layer`` is the evaluator
    layer name with dots replaced by underscores (e.g. ``block.38``
    → ``features_block_38``).

    Parameters
    ----------
    dataset : ZipStreamingDataset
        Zipped dataset whose samples contain both ``tile`` and
        ``features_*`` columns.
    evaluator : DeepBackboneEvaluator
        A ready-to-use evaluator (e.g. from
        ``evaluator_factory.evaluator(device=...)``).
    n_samples : int | None
        Maximum number of samples to validate.  ``None`` validates all.
    tile_key : str
        Key for the tile column in the dataset.
    tile_source : str
        Where to read tiles for recomputation.

        * ``'dataset'`` (default) — from ``sample[tile_key]``.
        * ``'tfrecord'`` — from the source TFRecords via
          ``bag_name`` + ``tile_index`` metadata in each sample.
          Requires ``tilebagclip``.
    tilebagclip : Clip | None
        The source tile clip, required when ``tile_source='tfrecord'``.
        Bags are indexed by name and tiles loaded lazily.
    atol : float
        Absolute tolerance for ``np.allclose``.
    rtol : float
        Relative tolerance for ``np.allclose``.
    log : dbx.Logger | None
        Logger instance.

    Examples
    --------
    ::

        factory = GigapathDeepBackboneEvaluatorFactory(...)
        evaluator = factory.evaluator(device='cuda')
        ds = ZipStreamingDataset(feature_ds, tile_ds)
        result = ValidateTileFeatureZipStreamingDataset(
            ds, evaluator, n_samples=100,
        ).validate()
        print(result)  # ValidationResult(n_validated=100, n_mismatched=0, ...)
    """

    @dataclass
    class ValidationResult:
        """Summary of a validation run."""
        n_validated: int
        n_mismatched: int
        mismatched_indices: list
        max_abs_error: float
        mean_abs_error: float
        max_rel_error: float
        mean_rel_error: float
        layer_abs_errors: dict  # layer → max abs error across all samples
        layer_rel_errors: dict  # layer → max rel error across all samples
        # Repeat statistics (populated when n_repeats > 1)
        max_repeat_std: float   # max std across repeats (GPU jitter envelope)
        mean_repeat_std: float  # mean std across repeats

        @property
        def passed(self) -> bool:
            return self.n_mismatched == 0

        def __repr__(self):
            status = 'PASSED' if self.passed else 'FAILED'
            parts = [
                f"ValidationResult({status}: ",
                f"{self.n_validated} validated, ",
                f"{self.n_mismatched} mismatched, ",
                f"max_abs_error={self.max_abs_error:.2e}, ",
                f"mean_abs_error={self.mean_abs_error:.2e}, ",
                f"max_rel_error={self.max_rel_error:.2e}, ",
                f"mean_rel_error={self.mean_rel_error:.2e}",
            ]
            if self.max_repeat_std > 0:
                parts.append(
                    f", max_repeat_std={self.max_repeat_std:.2e}, "
                    f"mean_repeat_std={self.mean_repeat_std:.2e}"
                )
            parts.append(")")
            return "".join(parts)

    def __init__(
        self,
        dataset,
        evaluator,
        *,
        n_samples: int | None = None,
        tile_key: str = 'tile',
        tile_source: str = 'dataset',
        tilebagclip: 'Clip | None' = None,
        n_repeats: int = 5,
        gpu_batch_size: int = 64,
        # Tolerances are intentionally loose: fp32 non-determinism is
        # typically ~1e-6 relative, but mixed-precision (fp16/bf16)
        # builds or other sources of drift may be larger.  Inspect
        # max_rel_error in the result to assess actual divergence.
        atol: float = 1e-4,
        rtol: float = 1e-4,
        rel_floor: float = 1.0,
        log: dbx.Logger = None,
    ):
        self.dataset = dataset
        self.evaluator = evaluator
        self.n_samples = n_samples
        self.tile_key = tile_key
        self.tile_source = tile_source
        self.n_repeats = n_repeats
        self.gpu_batch_size = gpu_batch_size
        self.atol = atol
        self.rtol = rtol
        self.rel_floor = rel_floor
        self.log = log or dbx.Logger(name='ValidateTileFeatureZip')

        # When tile_source='tfrecord', index the tilebagclip's bags by
        # name so we can look up tiles via bag_name + tile_index.
        self._bag_index = None
        self._tile_cache = {}
        if tile_source == 'tfrecord':
            if tilebagclip is None:
                raise ValueError(
                    "tilebagclip is required when tile_source='tfrecord'"
                )
            self.log.info(
                "tile_source='tfrecord': will read tiles from source TFRecords"
            )
            self._bag_index = {}
            for idx in range(tilebagclip.n_blocks):
                bag = tilebagclip.block(idx)
                self._bag_index[bag.name] = bag
            self.log.info(
                f"Indexed {len(self._bag_index)} tile bags by name"
            )

    def _evaluate_tile(self, tile):
        """Run evaluator on a tile, return {layer: ndarray} for one sample.

        Replicates the tile to match ``gpu_batch_size`` so that
        batch-normalisation and other batch-size-sensitive ops produce
        the same result as during the original build.
        """
        if self.gpu_batch_size > 1 and tile.shape[0] == 1:
            tile = tile.expand(self.gpu_batch_size, -1, -1, -1)
        result = self.evaluator(tile)
        out = {}
        for layer in self.evaluator.layer_names:
            if layer in result:
                out[layer] = result[layer][0].numpy()
        self.evaluator.clear()
        return out

    def validate(self):
        """Run validation and return a :class:`ValidationResult`."""
        n_total = len(self.dataset)
        n_to_check = n_total if self.n_samples is None else min(self.n_samples, n_total)

        mismatched_indices = []
        max_abs_error = 0.0
        max_rel_error = 0.0
        sum_abs_error = 0.0
        sum_rel_error = 0.0
        max_repeat_std = 0.0
        sum_repeat_std = 0.0
        n_comparisons = 0
        layer_abs_errors = {}
        layer_rel_errors = {}

        layer_names = self.evaluator.layer_names
        # Build column mapping: evaluator layer → dataset column
        col_map = {}
        for layer in layer_names:
            safe = layer.replace('.', '_')
            col_map[layer] = f'features_{safe}'

        progress = tqdm.tqdm(range(n_to_check), desc='Validating', unit='sample')
        for i in progress:
            sample = self.dataset[i]

            # Extract tile — from TFRecord source or from the dataset
            if self.tile_source == 'tfrecord':
                bag_name = sample['bag_name']
                tile_idx = int(sample['tile_index'])
                if bag_name not in self._tile_cache:
                    self._tile_cache.clear()  # release previous bag
                    self._tile_cache[bag_name] = self._bag_index[bag_name].tiles
                tile = self._tile_cache[bag_name][tile_idx]
            else:
                tile = sample[self.tile_key]
            if isinstance(tile, np.ndarray):
                tile = torch.from_numpy(tile.copy())
            if tile.dim() == 3:
                tile = tile.unsqueeze(0)  # (H, W, C) → (1, H, W, C)

            # Run evaluator n_repeats times; use mean for comparison.
            runs = [self._evaluate_tile(tile) for _ in range(self.n_repeats)]

            # Compare each layer
            sample_ok = True
            for layer in layer_names:
                col = col_map[layer]
                if col not in sample:
                    continue
                stored = sample[col]
                if isinstance(stored, torch.Tensor):
                    stored = stored.numpy()

                # Stack repeats: (n_repeats, feature_dim)
                repeat_stack = np.stack([r[layer] for r in runs])
                computed = repeat_stack.mean(axis=0)

                # Repeat std: per-element std across runs, then take max
                if self.n_repeats > 1:
                    repeat_std = repeat_stack.std(axis=0)
                    sample_max_std = float(np.max(repeat_std))
                    sample_mean_std = float(np.mean(repeat_std))
                    max_repeat_std = max(max_repeat_std, sample_max_std)
                    sum_repeat_std += sample_mean_std

                diff = np.abs(stored - computed)
                abs_err = float(np.max(diff))
                mean_abs = float(np.mean(diff))
                # Floor the denominator at rel_floor so near-zero
                # features don't inflate relative error.  When
                # |feature| < rel_floor, this degrades gracefully
                # to absolute error (diff / 1.0).
                denom = np.maximum(
                    np.maximum(np.abs(stored), np.abs(computed)),
                    self.rel_floor,
                )
                rel = diff / denom
                rel_err = float(np.max(rel))
                mean_rel = float(np.mean(rel))

                max_abs_error = max(max_abs_error, abs_err)
                max_rel_error = max(max_rel_error, rel_err)
                sum_abs_error += mean_abs
                sum_rel_error += mean_rel
                n_comparisons += 1
                layer_abs_errors[layer] = max(layer_abs_errors.get(layer, 0.0), abs_err)
                layer_rel_errors[layer] = max(layer_rel_errors.get(layer, 0.0), rel_err)

                if not np.allclose(stored, computed, atol=self.atol, rtol=self.rtol):
                    sample_ok = False

            if not sample_ok:
                mismatched_indices.append(i)

            if (i + 1) % 50 == 0 or i == n_to_check - 1:
                _mean_abs = sum_abs_error / max(n_comparisons, 1)
                _mean_rel = sum_rel_error / max(n_comparisons, 1)
                msg = (
                    f"Validated {i + 1}/{n_to_check} samples, "
                    f"{len(mismatched_indices)} mismatches, "
                    f"max_abs={max_abs_error:.2e}, mean_abs={_mean_abs:.2e}, "
                    f"max_rel={max_rel_error:.2e}, mean_rel={_mean_rel:.2e}"
                )
                if self.n_repeats > 1:
                    _mean_std = sum_repeat_std / max(n_comparisons, 1)
                    msg += (
                        f", repeat_std: max={max_repeat_std:.2e}, "
                        f"mean={_mean_std:.2e}"
                    )
                self.log.info(msg)

        mean_abs_error = sum_abs_error / max(n_comparisons, 1)
        mean_rel_error = sum_rel_error / max(n_comparisons, 1)
        mean_repeat_std = sum_repeat_std / max(n_comparisons, 1)

        return self.ValidationResult(
            n_validated=n_to_check,
            n_mismatched=len(mismatched_indices),
            mismatched_indices=mismatched_indices,
            max_abs_error=max_abs_error,
            mean_abs_error=mean_abs_error,
            max_rel_error=max_rel_error,
            mean_rel_error=mean_rel_error,
            layer_abs_errors=layer_abs_errors,
            layer_rel_errors=layer_rel_errors,
            max_repeat_std=max_repeat_std,
            mean_repeat_std=mean_repeat_std,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Collation helpers
# ═══════════════════════════════════════════════════════════════════════

def _sanitize(obj):
    """Recursively replace ``None`` with ``{}`` in nested dicts/lists."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def sanitize_collate(batch):
    """Collate a batch of sample dicts, tolerating missing or ``None`` values.

    MDS JSON columns can deserialise to ``None`` (e.g. bags without a
    valid case-id have no annotations), or be absent entirely from some
    samples.  PyTorch's ``default_collate`` rejects both cases, so we:

    1. Union all keys across the batch (some shards may omit optional
       columns entirely).
    2. Fill missing keys with ``None``.
    3. Recursively replace ``None`` with ``{}`` via :func:`_sanitize`.
    4. Delegate to ``default_collate``.
    """
    from torch.utils.data._utils.collate import default_collate
    all_keys = set().union(*(s.keys() for s in batch))
    aligned = [{k: s.get(k, None) for k in all_keys} for s in batch]
    return default_collate([_sanitize(sample) for sample in aligned])

