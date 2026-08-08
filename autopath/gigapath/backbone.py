"""GigaPath-specific deep backbone evaluation infrastructure.

Provides `GigapathDeepBackboneEvaluator` (runtime hook-based
activation capture for the GigaPath ViT backbone) and
`GigapathDeepBackboneEvaluatorFactory` (a concrete
`DeepBackboneEvaluatorFactory` that persists
the resolved capture configuration so that downstream consumers know
the tensor shapes).

Inherits from `DataformerEvaluator`;
the `gigapath.dinov2` package supplies only the raw model factory
(`gigapath_tile_backbone()`) and
the tile transform
(`dino_tile_transform()`).
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
    DataformerEvaluator,
    DataformerEvaluatorFactory,
)

from autopath.autobits import (
    DeepBackboneEvaluator,
    DeepBackboneEvaluatorFactory,
)

from autopath.gigapath.dinov2.backbone import (
    GIGAPATH_BACKBONE_DEPTH,
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


def resolve_sublayer_name(sublayer: str) -> str:
    """Validate and normalize a block sub-layer name."""
    if sublayer not in BLOCK_SUBLAYERS:
        raise ValueError(
            f"Unknown block sub-layer {sublayer!r}. "
            f"Must be one of {BLOCK_SUBLAYERS}"
        )
    return sublayer


class GigapathDeepBackboneEvaluator(DataformerEvaluator):
    """GigaPath-specific configurable activation-capturing backbone evaluator.

    Concrete subclass of `DataformerEvaluator`
    for the GigaPath ViT backbone.
    """

    DEFAULT_MODEL = "$autopath.gigapath.dinov2.backbone.gigapath_tile_backbone()"

    def __init__(
        self,
        model=None,
        *,
        backbone=None,
        capture_blocks: List[int] = None,
        capture_layers: List[str] = None,
        capture_final: bool = True,
        cls_token_only: bool = False,
        transform=None,
        device: str = "cuda",
        log: Logger = Logger(),
    ):
        model_val = model if model is not None else backbone
        if transform is None:
            transform = dino_tile_transform()
        if capture_blocks == 'all':
            capture_blocks = list(range(GIGAPATH_BACKBONE_DEPTH))
        super().__init__(
            model=model_val,
            capture_blocks=capture_blocks,
            capture_layers=capture_layers,
            capture_final=capture_final,
            cls_token_only=cls_token_only,
            transform=transform,
            device=device,
            log=log,
        )

    @property
    def backbone(self):
        """Backwards compatible alias for model."""
        return self.model


class GigapathDeepBackboneEvaluatorFactory(DataformerEvaluatorFactory):
    """GigaPath-specific `DataformerEvaluatorFactory`.

    Exists for spec-based dependency tracking in `DeepFeatureBag`
    and `DeepFeatureClip`.  Call `evaluator()` to obtain a
    ready-to-use `GigapathDeepBackboneEvaluator`.
    """

    Evaluator = GigapathDeepBackboneEvaluator
