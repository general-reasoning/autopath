"""GigaPath-specific deep backbone evaluation infrastructure.

Provides :class:`GigapathDeepBackboneEvaluator` (runtime hook-based
activation capture for the GigaPath ViT backbone) and
:class:`GigapathDeepBackboneEvaluatorFactory` (a concrete
:class:`~autopath.databits.DeepBackboneEvaluatorFactory` that persists
the resolved capture configuration so that downstream consumers know
the tensor shapes).

Modelled on
:class:`autopath.gigapath.dinov2.backbone.SidebandBackboneEvaluator` but
with two key improvements:

1.  Hook registration is driven entirely by a declarative
    ``capture_layers: list[str | int]`` config, not hardcoded.
2.  The evaluator *returns* captured activations from ``__call__``
    instead of exposing a mutable internal dict.
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

from autopath.databits import (
    DeepBackboneEvaluator,
    DeepBackboneEvaluatorFactory,
)

from autopath.gigapath.dinov2.backbone import (
    BackboneEvaluator,
    backbone_blocks,
    dino_tile_transform,
)



# ── Available sub-layers inside each transformer block ──────────────
# This list mirrors the GigapathTensorBlock architecture; users pick a
# subset via ``capture_layers``.
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


class GigapathDeepBackboneEvaluator(BackboneEvaluator, DeepBackboneEvaluator):
    """GigaPath-specific configurable activation-capturing backbone evaluator.

    Concrete subclass of :class:`~autopath.databits.DeepBackboneEvaluator`
    for the GigaPath ViT backbone.  Capture targets are specified via a
    unified ``capture_layers`` list:

    * **int** entries are transformer block indices — capture the block's
      output activations (full sequence, shape ``(B, N, d)``).
    * **str** entries are top-level model attribute names (e.g.
      ``"patch_embed"``, ``"norm"``, ``"head"``) or ``"backbone"`` for
      the model-level output.

    Parameters
    ----------
    backbone
        A pre-loaded model or a lazy-eval string (see
        :class:`BackboneEvaluator`).
    capture_layers : list[str | int]
        Capture targets.  Integers are transformer block indices;
        strings are top-level model layer names.
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
        capture_layers: List[Union[str, int]],
        cls_token_only: bool = False,
        transform=None, #defaults to dino_tile_transform (ImageNet normalisation)
        device: str = "cuda",
        log: Logger = Logger(),
    ):
        if transform is None:
            transform = dino_tile_transform()
        super().__init__(backbone, transform=transform, device=device, log=log)
        self.capture_layers = list(capture_layers)
        self.cls_token_only = cls_token_only
        self._captured: Dict[str, torch.Tensor] = {}
        self._hooks_registered = False

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
        """Convert a ``capture_layers`` entry to its string key."""
        return f"block.{layer}" if isinstance(layer, int) else layer

    def _register_capture_hooks(self):
        """Register forward hooks on the configured capture layers."""
        if self._hooks_registered:
            return

        blocks = backbone_blocks(self.backbone)
        for layer in self.capture_layers:
            key = self._capture_key(layer)
            if isinstance(layer, int):
                assert 0 <= layer < len(blocks), (
                    f"Block index {layer} out of range [0, {len(blocks)})"
                )
                blocks[layer].register_forward_hook(
                    self._make_capture_hook(key)
                )
            elif layer == "backbone":
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
        return [self._capture_key(l) for l in self.capture_layers]

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
        result["output"] = z
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
            capture_layers=self.cfg.capture_layers,
            cls_token_only=self.cfg.cls_token_only,
            device=device,
            log=log,
        )
