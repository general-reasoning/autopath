"""GigaPath-specific deep backbone evaluation infrastructure.

Provides :class:`GigapathDeepBackboneEvaluator` (runtime hook-based
activation capture for the GigaPath ViT backbone) and
:class:`GigapathDeepBackboneEvaluatorFactory` (a concrete
:class:`~autopath.autobits.DeepBackboneEvaluatorFactory` that persists
the resolved capture configuration so that downstream consumers know
the tensor shapes).

Inherits solely from :class:`~autopath.autobits.DeepBackboneEvaluator`;
the ``gigapath.dinov2`` package supplies only the raw model factory
(:func:`~autopath.gigapath.dinov2.backbone.gigapath_tile_backbone`) and
the tile transform
(:func:`~autopath.gigapath.dinov2.backbone.dino_tile_transform`).
"""

import copy
from dataclasses import dataclass
import gc
from typing import Dict, List, Optional, Union

import torch

import dbx
from dbx import (
    Logger,
    Datablock,
)

from autopath.autobits import (
    DeepBackboneEvaluator,
    DeepBackboneEvaluatorFactory,
)

from autopath.gigapath.dinov2.backbone import (
    backbone_blocks,
    dino_tile_transform,
)



# ── Available sub-layers inside each transformer block ──────────────
# This list mirrors the GigapathTensorBlock architecture; users pick a
# subset via ``capture_blocks`` and ``capture_layers``.
BLOCK_SUBLAYERS = [
    "norm1",
    "attn",           # full attention module
    "attn.qkv",
    "attn.proj",
    "ls1",
    "norm2",
    "mlp",            # full MLP module
    "mlp.fc1",
    "mlp.act",
    "mlp.fc2",
    "ls2",
]

# Top-level model layers that can be captured independently of any block.
MODEL_LAYERS = [
    "patch_embed",
    "norm",
    "head",
    "backbone",       # the model itself (final output)
]


def _resolve_sublayer(block, sublayer_path: str):
    """Traverse dotted attribute paths like ``'attn.qkv'`` on *block*."""
    obj = block
    for attr in sublayer_path.split("."):
        obj = getattr(obj, attr)
    return obj


class GigapathDeepBackboneEvaluator(DeepBackboneEvaluator):
    """GigaPath-specific configurable activation-capturing backbone evaluator.

    Concrete subclass of :class:`~autopath.autobits.DeepBackboneEvaluator`
    for the GigaPath ViT backbone.  Capture targets are specified via
    two parameters:

    * ``capture_blocks`` — a list of **int** transformer block indices
      (e.g. ``[0, 19, 38]`` or ``[-1]``).  Hooks capture each block's
      output activations (full sequence, shape ``(B, N, d)``).
    * ``capture_layers`` — a list of **str** top-level model attribute
      names (e.g. ``"patch_embed"``, ``"norm"``, ``"head"``) or
      ``"backbone"`` for the model-level output.

    The raw backbone model is lazy-loaded from the default HuggingFace
    cache via :func:`~autopath.gigapath.dinov2.backbone.gigapath_tile_backbone`
    on first access; ``dinov2`` supplies only the model and transform
    factories — no evaluator base class.

    Parameters
    ----------
    backbone
        A pre-loaded model, a lazy-eval string, or ``None`` (default)
        to auto-load via ``gigapath_tile_backbone()``.
    capture_blocks : list[int]
        Transformer block indices to capture.
    capture_layers : list[str]
        Named model layers to capture.
    cls_token_only : bool
        When ``True``, hooks capture only the CLS token activation
        (index 0 of the sequence dimension), reducing output from
        ``(B, N, d)`` to ``(B, d)``.  Layers that already produce
        2-D output are unaffected.
    transform
        Tile preprocessing transform.  Defaults to
        :func:`dino_tile_transform` (ImageNet normalisation).
    device : str
        Target device.
    log : Logger
        Logger instance.
    """

    def __init__(
        self,
        backbone=None,
        *,
        capture_blocks: List[int] = None,
        capture_layers: List[str] = None,
        capture_outputs: bool = True,
        cls_token_only: bool = False,
        transform=None, #defaults to dino_tile_transform (ImageNet normalisation)
        device: str = "cuda",
        log: Logger = Logger(),
    ):
        super().__init__(device=device, log=log)
        # Lazy backbone: None → default eval string, str → dbx.eval on first access
        self._backbone = backbone
        if self._backbone is None:
            self._backbone = "$autopath.gigapath.dinov2.backbone.gigapath_tile_backbone()"
        self.transform = transform
        if self.transform is None:
            self.transform = dino_tile_transform()
        self.capture_blocks = list(capture_blocks or [])
        self.capture_layers = list(capture_layers or [])
        self.capture_outputs = capture_outputs
        self.cls_token_only = cls_token_only
        self._captured: Dict[str, torch.Tensor] = {}
        self._hooks_registered = False

    @property
    def backbone(self):
        """Lazy-load the backbone model on first access."""
        if isinstance(self._backbone, str):
            self.log.verbose(f"Evaluating {self._backbone} on {self.device}")
            self._backbone = dbx.eval(self._backbone).to(self.device)
        return self._backbone

    # ── Hook management (private) ───────────────────────────────────

    def _make_capture_hook(self, name: str):
        """Return a forward-hook closure that stores output in ``_captured``.

        When ``cls_token_only`` is set, 3-D tensors ``(B, N, d)`` are
        sliced to ``(B, d)`` by taking the CLS token at index 0.
        """
        cls_only = self.cls_token_only

        def hook(module, input, output):
            t = output.cpu().detach()
            if cls_only and t.dim() == 3:
                t = t[:, 0]   # CLS token
            self._captured[name] = t
        return hook

    @staticmethod
    def _capture_key(layer) -> str:
        """Convert a capture entry to its string key."""
        return f"block.{layer}" if isinstance(layer, int) else layer

    def _register_capture_hooks(self):
        """Register forward hooks on the configured blocks and layers."""
        if self._hooks_registered:
            return

        blocks = backbone_blocks(self.backbone)

        # Block hooks (int indices, supports negative indexing)
        for idx in self.capture_blocks:
            if idx < 0:
                idx = len(blocks) + idx
            assert 0 <= idx < len(blocks), (
                f"Block index {idx} out of range [0, {len(blocks)})"
            )
            key = self._capture_key(idx)
            blocks[idx].register_forward_hook(
                self._make_capture_hook(key)
            )
            self.log.debug(f"Registered capture hook: {key}")

        # Named layer hooks (str names)
        for layer in self.capture_layers:
            key = self._capture_key(layer)
            if layer == "backbone":
                self.backbone.register_forward_hook(
                    self._make_capture_hook(key)
                )
            else:
                getattr(self.backbone, layer).register_forward_hook(
                    self._make_capture_hook(key)
                )
            self.log.debug(f"Registered capture hook: {key}")

        self._hooks_registered = True

    # ── Layer name introspection ────────────────────────────────────

    @property
    def layer_names(self) -> List[str]:
        """Return the ordered list of capture keys that ``__call__`` will produce."""
        names = [self._capture_key(b) for b in self.capture_blocks]
        names += [self._capture_key(l) for l in self.capture_layers]
        if self.capture_outputs:
            names.append('output')
        return names

    # ── Forward pass ────────────────────────────────────────────────

    def __pre_call__(self):
        """Ensure hooks are registered before the first forward pass."""
        self._register_capture_hooks()

    def __call__(self, x) -> Dict[str, torch.Tensor]:
        """Run forward pass and return captured activations.

        Parameters
        ----------
        x : Tensor
            Batch of tile images.

        Returns
        -------
        dict[str, Tensor]
            Mapping from capture-key to activation tensor, plus
            ``"output"`` for the backbone's own output.
        """
        self._captured.clear()
        self.__pre_call__()
        with torch.no_grad():
            y = self.transform(x.to(self.device))
            z = self.backbone(y).cpu().detach()
            del y

        result = dict(self._captured)
        if self.capture_outputs:
            out = z
            if self.cls_token_only and out.dim() == 3:
                out = out[:, 0]
            result['output'] = out
        return result

    @property
    def layer_features(self) -> Dict[str, torch.Tensor]:
        """Most recently captured activations (read-only snapshot)."""
        return dict(self._captured)

    def clear(self):
        """Release captured tensors and free GPU memory."""
        self._captured.clear()
        gc.collect()
        torch.cuda.empty_cache()
        return self


class GigapathDeepBackboneEvaluatorFactory(DeepBackboneEvaluatorFactory):
    """GigaPath-specific :class:`DeepBackboneEvaluatorFactory`.

    Exists for spec-based dependency tracking in :class:`DeepFeatureBag`
    and :class:`DeepFeatureClip`.  Call :meth:`evaluator` to obtain a
    ready-to-use :class:`GigapathDeepBackboneEvaluator`.
    """

    def evaluator(self, *, device: str = "cuda", log: Logger = None) -> GigapathDeepBackboneEvaluator:
        """Create a live :class:`GigapathDeepBackboneEvaluator`.

        The backbone is lazy-loaded on first call to the evaluator.
        """
        log = log or self.log
        return GigapathDeepBackboneEvaluator(
            backbone=None,  # lazy-loaded from default
            capture_blocks=self.cfg.capture_blocks,
            capture_layers=self.cfg.capture_layers,
            capture_outputs=self.cfg.capture_outputs,
            cls_token_only=self.cfg.cls_token_only,
            device=device,
            log=log,
        )
