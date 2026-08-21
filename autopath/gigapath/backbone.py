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
from typing import Any, Dict, List, Optional, Union

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
    gigapath_tile_backbone,
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


class GigapathDeepBackboneEvaluatorFactory(DataformerEvaluatorFactory):
    """GigaPath-specific `DataformerEvaluatorFactory`.

    Exists for spec-based dependency tracking in `DeepFeatureBag`
    and `DeepFeatureClip`. Call `evaluator()` to obtain a
    ready-to-use `DataformerEvaluator`.
    """

    @property
    def model(self):
        return gigapath_tile_backbone()

    @property
    def transform(self):
        return dino_tile_transform()
