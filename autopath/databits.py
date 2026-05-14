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

    
class Clip(Datastack):
    """Abstract clip of bags, built in parallel via :class:`Datastack`.

    Subclasses must implement:

    * ``n_shards``  — number of bags (property)
    * ``__shard__(idx)`` — return the :class:`Bag` at *idx*

    After all bags are built in parallel, :meth:`__stack__` persists
    ``bag_lens.npz`` so that lengths can be read back without
    materializing bags.
    """

    TOPICFILES = {"bag_lens": "bag_lens.npz"}

    def __len__(self):
        return sum(self.bag_lens)

    def __stack__(self):
        """Persist bag lengths after all bags have been built."""
        self.log.verbose(f"Stacking bag lens")
        bag_lens = [len(self.shard(i)) for i in tqdm.tqdm(range(self.n_shards), desc="Stacking bag lens")]
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
        return self.n_shards

    def bag(self, idx: int):
        return self.shard(idx)

    @property
    def bags(self):
        return self.shards()

    @property
    def shard_lens(self):
        return self.bag_lens
    # ───────────────────────────────────────────────────────────────

    def UNSAFE_clear_bags(self, *, OVERRIDE: bool = False):
        return self.UNSAFE_clear_shards(OVERRIDE=OVERRIDE)

    def UNSAFE_copy_from(self, anchorpath: str, bag_anchorpath: str, *, overwrite: bool = False):
        super().UNSAFE_copy_from(anchorpath, overwrite=overwrite)
        self.UNSAFE_copy_bags_from(bag_anchorpath, overwrite=overwrite)
        return self

    def UNSAFE_copy_bags_from(self, bag_anchorpath: str, *, overwrite: bool = False):
        for bag in self.bags:
            bag.UNSAFE_copy_from(bag_anchorpath, overwrite=overwrite)
        return self
    

class Partition(Datablock):
    TOPICFILES = {"bag_indices": "bag_indices.npz", 
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
        self.log.info(f"Building partition out of {len(self.cfg.clip.bags)} bags using fold fractions {self.cfg.fold_fractions}")
        N = len(self.cfg.clip.bags)
        np.random.seed(self.cfg.seed) #TODO: localize in a generator
        perm = np.random.permutation(N)

        bag_indices = {}
        bag_lens = {}
        Klo = 0
        self.log.verbose(f"Computing bag indices and lens for {len(self.cfg.fold_fractions)} folds: BEGIN")
        if self.verbose:
            fold_fraction_itor = tqdm.tqdm(self.cfg.fold_fractions)
        else:
            fold_fraction_itor = self.cfg.fold_fractions
        for fold, fraction in enumerate(fold_fraction_itor):
            k = int(math.ceil(N*fraction))
            Khi = min(Klo + k, N)
            fold_key = str(fold)
            bag_indices[fold_key] = perm[Klo:Khi]
            self.log.verbose(f"Computing bag lens for fold {fold}: BEGIN")
            if self.verbose:
                bag_itor = tqdm.tqdm(bag_indices[fold_key])
            else:
                bag_itor = bag_indices[fold_key]
            bag_lens[fold_key] = np.array([len(self.cfg.clip.bags[i]) for i in bag_itor])
            self.log.verbose(f"Computing bag lens for fold {fold}: END")
            Klo = Khi
        self.log.verbose(f"Computing bag indices and lens for {len(self.cfg.fold_fractions)} folds: END")
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
        return [self.cfg.clip.bags[i] for i in self.bag_indices(fold)]
    
    def bag_lens(self, fold):
        return self.read("bag_lens")[self._resolve_fold(fold)]

    def bag_indices(self, fold):
        return self.read("bag_indices")[self._resolve_fold(fold)]

    def bag(self, fold, idx: int):
        return self.cfg.clip.bags[self.bag_indices(fold)[idx]]

    def n_bags(self, fold):
        return len(self.bag_indices(fold))


class Fold(Clip):
    @dataclass
    class CONFIG:
        partition: Partition
        fold: Union[str, int]

    def __post_init__(self):
        return self
    
    def valid(self):
        return self.cfg.partition.valid()

    @property
    def n_shards(self):
        return self.cfg.partition.n_bags(self.cfg.fold)

    def __shard__(self, idx: int):
        return self.cfg.partition.bag(self.cfg.fold, idx)

    def __build__(self):
        """Bags are managed by the Partition — skip parallel shard building."""
        self.__stack__()
        return self

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
        capture_blocks: list           # list[int] — transformer block indices
        capture_layers: list = field(default_factory=list)  # list[str] — named layers
        cls_token_only: bool = False   # capture only CLS token activations

    def evaluator(self, *, device: str = None, log: dbx.Logger = None):
        """Create a live :class:`DeepBackboneEvaluator`.

        Must be overridden by subclasses.
        """
        raise NotImplementedError


class SpectralDeepBackboneEvaluator(DeepBackboneEvaluator):
    """Abstract :class:`DeepBackboneEvaluator` with spectral probing.

    Adds Jacobian singular-value analysis on top of the multi-layer
    activation capture provided by :class:`DeepBackboneEvaluator`.
    After each ``__call__``, a :class:`~autopath.deep.probes.SpectralProbe`
    is run on the captured activations.

    Subclasses **must** implement:

    * :attr:`backbone_blocks` — property returning a ``list[nn.Module]``
      of transformer blocks in forward-pass order.  The
      :class:`SpectralProbe` uses these to compute Jacobians via autograd.
    * All abstract methods inherited from :class:`DeepBackboneEvaluator`.

    Parameters
    ----------
    spectral_probe_blocks : list[int] | None
        Block indices to probe.  If ``None``, defaults to the
        evaluator's entire ``layer_names`` list filtered for integer
        block keys.
    spectral_mode : str
        ``'cls'``, ``'full'``, or ``'both'``.
    spectral_k : int
        Number of extreme singular values for full mode.
    device : str
        Target device.
    """

    def __init__(
        self,
        *,
        spectral_probe_blocks=None,
        spectral_mode: str = 'cls',
        spectral_k: int = 10,
        device: str = "cuda",
        log: dbx.Logger = dbx.Logger(stack_depth=3),
    ):
        super().__init__(device=device, log=log)
        self.spectral_probe_blocks = spectral_probe_blocks or []
        self.spectral_mode = spectral_mode
        self.spectral_k = spectral_k
        self._spectral_probe = None
        self._spectral_results = None

    @property
    def backbone_blocks(self):
        """Return the ordered list of ``nn.Module`` transformer blocks.

        Must be overridden by subclasses.
        """
        raise NotImplementedError

    @property
    def spectral_probe(self):
        """Lazily create the :class:`SpectralProbe`."""
        if self._spectral_probe is None and self.spectral_probe_blocks:
            from autopath.deep.probes import SpectralProbe
            self._spectral_probe = SpectralProbe(
                blocks=self.backbone_blocks,
                probe_blocks=self.spectral_probe_blocks,
                k=self.spectral_k,
                device=self.device,
            )
        return self._spectral_probe

    def _run_spectral_probe(self, layer_features):
        """Run spectral probing on captured activations.

        Called automatically after ``__call__``.  Populates
        ``spectral_results``.
        """
        if self.spectral_probe is None:
            return
        # Build activations dict from layer_features: keys like "block.0"
        activations = {}
        for b in self.spectral_probe_blocks:
            key = f"block.{b}"
            if key in layer_features and layer_features[key] is not None:
                activations[b] = layer_features[key].to(self.device)
        if activations:
            self._spectral_results = self.spectral_probe.probe(
                activations, mode=self.spectral_mode,
            )
            # Composed Jacobian (first → last probed block, CLS-only)
            if len(self.spectral_probe_blocks) >= 2:
                import numpy as np
                first_b = min(self.spectral_probe_blocks)
                last_b = max(self.spectral_probe_blocks)
                input_key = f"block.{first_b}_input"
                if input_key in layer_features and layer_features[input_key] is not None:
                    h_input = layer_features[input_key].to(self.device)
                    sv = self.spectral_probe._probe_composed_cls(
                        first_b, last_b, h_input,
                    )
                    self._spectral_results['composed'] = {
                        'singular_values': sv,
                        'log_singular_values': np.log(np.clip(sv, 1e-12, None)),
                        'condition_number_cls': float(sv[0] / (sv[-1] + 1e-12)),
                    }

    @property
    def spectral_results(self):
        """Spectral probe results from the last forward pass, keyed by block index."""
        return self._spectral_results

    def clear_spectral_results(self):
        self._spectral_results = None
        return self


