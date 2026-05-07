"""Deep backbone evaluation infrastructure for configurable activation capture.

Provides :class:`DeepBackboneEvaluator` (runtime hook-based activation
capture) and :class:`DeepBackboneEvaluatorFactory` (a :class:`Datablock`
that persists the resolved capture configuration so that downstream
consumers know the tensor shapes).

Modelled on
:class:`autopath.gigaq.dinov2.backbone.SidebandBackboneEvaluator` but
with two key improvements:

1.  Hook registration is driven entirely by a declarative config
    (``capture_transformer_blocks`` × ``capture_layers``), not hardcoded.
2.  The evaluator *returns* captured activations from ``__call__``
    instead of exposing a mutable ``sideband`` dict.
"""

import copy
from dataclasses import dataclass, asdict
import gc
from typing import Dict, List, Optional

import torch

import dbx
from dbx import (
    Logger,
    Datablock,
    write_npz,
    read_npz,
    write_pickle,
    read_pickle,
)

from autopath.gigaq.dinov2.backbone import (
    BackboneEvaluator,
    backbone_blocks,
    gigapath_tile_backbone,
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


class DeepBackboneEvaluator(BackboneEvaluator):
    """Configurable activation-capturing backbone evaluator.

    Unlike :class:`SidebandBackboneEvaluator`, capture targets are fully
    specified via ``capture_transformer_blocks`` and ``capture_layers``
    and the captured tensors are returned directly from ``__call__``.

    Parameters
    ----------
    backbone
        A pre-loaded model or a lazy-eval string (see
        :class:`BackboneEvaluator`).
    capture_transformer_blocks : list[int]
        Block indices to instrument (e.g. ``[0, 11, 23, 39]``).
    capture_layers : list[str]
        Sub-layer names within each captured block to hook
        (e.g. ``["norm1", "mlp.fc2"]``).  Names must be valid
        attribute paths on the block module.
    capture_model_layers : list[str] | None
        Optional top-level model layers to capture (``"patch_embed"``,
        ``"norm"``, ``"head"``, ``"backbone"``).
    transform
        Tile preprocessing transform.
    device : str
        Target device.
    log : Logger
        Logger instance.
    """

    def __init__(
        self,
        backbone=None,
        *,
        capture_transformer_blocks: List[int],
        capture_layers: List[str],
        capture_model_layers: Optional[List[str]] = None,
        transform=None,
        device: str = "cuda",
        log: Logger = Logger(stack_depth=3),
    ):
        super().__init__(backbone, transform=transform, device=device, log=log)
        self.capture_transformer_blocks = capture_transformer_blocks
        self.capture_layers = capture_layers
        self.capture_model_layers = capture_model_layers or []
        self._sideband: Dict[str, torch.Tensor] = {}
        self._hooks_registered = False

    # ── Hook management (private) ───────────────────────────────────

    def _capture_hook(self, name: str):
        """Return a forward-hook closure that stores output in ``_sideband``."""
        def hook(module, input, output):
            self._sideband[name] = output.cpu().detach()
        return hook

    def _register_hooks(self):
        """Register forward hooks on the configured blocks/layers."""
        if self._hooks_registered:
            return

        blocks = backbone_blocks(self.backbone)
        for b in self.capture_transformer_blocks:
            block = blocks[b]
            for layer in self.capture_layers:
                key = f"B_{b}_{layer}"
                target = _resolve_sublayer(block, layer)
                target.register_forward_hook(self._capture_hook(key))
                self.log.debug(f"Registered hook: {key}")

        for layer in self.capture_model_layers:
            if layer == "backbone":
                self.backbone.register_forward_hook(self._capture_hook(layer))
            else:
                getattr(self.backbone, layer).register_forward_hook(
                    self._capture_hook(layer)
                )
            self.log.debug(f"Registered hook: {layer}")

        self._hooks_registered = True

    # ── Layer name introspection ────────────────────────────────────

    @property
    def layer_names(self) -> List[str]:
        """Return the ordered list of capture keys that ``__call__`` will produce."""
        names = []
        for b in self.capture_transformer_blocks:
            for layer in self.capture_layers:
                names.append(f"B_{b}_{layer}")
        names.extend(self.capture_model_layers)
        return names

    # ── Forward pass ────────────────────────────────────────────────

    def __pre_call__(self):
        """Ensure hooks are registered before the first forward pass."""
        self._register_hooks()

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
        self._sideband.clear()
        self.__pre_call__()
        with torch.no_grad():
            y = (
                self.transform(x.to(self.device))
                if self.transform is not None
                else x.to(self.device)
            )
            z = self.backbone(y).cpu().detach()
            del y

        result = dict(self._sideband)
        result["output"] = z
        return result

    def clear(self):
        """Release captured tensors and free GPU memory."""
        self._sideband.clear()
        gc.collect()
        torch.cuda.empty_cache()
        return self


class DeepBackboneEvaluatorFactory(Datablock):
    """Datablock that materialises a :class:`DeepBackboneEvaluator` config.

    Building the factory resolves the backbone architecture (number of
    blocks, embedding dim) and persists the capture config so downstream
    consumers (e.g. :class:`DeepFeatureBag`) know the expected tensor
    shapes without loading the model.

    The factory itself does **not** hold a live model — call
    :meth:`evaluator` to obtain a ready-to-use evaluator.
    """

    TOPICFILES = {
        "config": "config.pkl",
        "layer_names": "layer_names.npz",
        "architecture": "architecture.pkl",
    }

    @dataclass
    class CONFIG:
        capture_transformer_blocks: list  # list[int]
        capture_layers: list              # list[str]
        capture_model_layers: list = None # list[str] | None
        model_type: str = "prov-gigapath"
        device: str = "cuda"

    def __post_init__(self):
        if self.cfg.capture_model_layers is None:
            self.cfg.capture_model_layers = []
        return self

    def __build__(self):
        # Temporarily load the backbone to resolve architecture details.
        self.log.verbose("Loading backbone to resolve architecture...")
        model = gigapath_tile_backbone(
            type=self.cfg.model_type,
            device=self.cfg.device,
        )
        blocks = backbone_blocks(model)
        n_blocks = len(blocks)
        embed_dim = model.embed_dim

        # Validate block indices.
        for b in self.cfg.capture_transformer_blocks:
            assert 0 <= b < n_blocks, (
                f"Block index {b} out of range [0, {n_blocks})"
            )

        # Build the evaluator transiently to get layer names.
        evaluator = DeepBackboneEvaluator(
            backbone=model,
            capture_transformer_blocks=self.cfg.capture_transformer_blocks,
            capture_layers=self.cfg.capture_layers,
            capture_model_layers=self.cfg.capture_model_layers,
            device=self.cfg.device,
            log=self.log,
        )
        layer_names = evaluator.layer_names

        # Persist.
        config = asdict(self.cfg)
        write_pickle(config, self.path("config", ensure_dirpath=True))
        write_npz(
            self.path("layer_names", ensure_dirpath=True),
            layer_names=layer_names,
        )
        architecture = {
            "n_blocks": n_blocks,
            "embed_dim": embed_dim,
            "layer_names": layer_names,
        }
        write_pickle(
            architecture, self.path("architecture", ensure_dirpath=True)
        )

        # Release the model.
        del model, evaluator
        gc.collect()
        torch.cuda.empty_cache()

        self.log.verbose(
            f"Factory built: {len(layer_names)} capture targets, "
            f"{n_blocks} blocks, embed_dim={embed_dim}"
        )
        return self

    def __read__(self, topic):
        if topic == "config":
            return read_pickle(self.path("config"))
        elif topic == "layer_names":
            return list(read_npz(self.path("layer_names"), "layer_names")["layer_names"])
        elif topic == "architecture":
            return read_pickle(self.path("architecture"))
        else:
            raise ValueError(f"Unknown topic: {topic}")

    def evaluator(self, *, device: str = None, log: Logger = None) -> DeepBackboneEvaluator:
        """Create a live :class:`DeepBackboneEvaluator` from persisted config.

        The backbone is lazy-loaded on first call to the evaluator.
        """
        device = device or self.cfg.device
        log = log or self.log
        return DeepBackboneEvaluator(
            backbone=None,  # lazy-loaded from default
            capture_transformer_blocks=self.cfg.capture_transformer_blocks,
            capture_layers=self.cfg.capture_layers,
            capture_model_layers=self.cfg.capture_model_layers,
            device=device,
            log=log,
        )
