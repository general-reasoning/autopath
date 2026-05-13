"""Pipeline entrypoints for GigaPath deep feature extraction.

Provides declarative constructors for
:class:`~autopath.gigapath.backbone.GigapathDeepBackboneEvaluator`,
:class:`~autopath.deep.features.DeepFeatureBag`,
:class:`~autopath.deep.features.DeepFeatureClip`,
and a convenience dataloader sampler, all wired to the GigaPath ViT
backbone via
:class:`~autopath.gigapath.backbone.GigapathDeepBackboneEvaluatorFactory`.

Modelled on :mod:`autopath.gigapath.pipelines`.
"""

from __future__ import annotations

from typing import Literal

import tqdm
import torch

import dbx
from dbx import Logger, tagged

from autopath.gigapath.backbone import (
    GigapathDeepBackboneEvaluator,
    GigapathDeepBackboneEvaluatorFactory,
)
from autopath.deep.features import (
    DeepFeatureBag,
    DeepFeatureClip,
)
from autopath.pancan.pipelines import (
    pancan_tile_bag_clip,
    pancan_tile_fold,
)


log = Logger()


# ═══════════════════════════════════════════════════════════════════════
#  Evaluator
# ═══════════════════════════════════════════════════════════════════════

def gigapath_deep_backbone_evaluator(
    *,
    capture_layers: list | None = None,
    cls_token_only: bool = False,
    device: str = "cuda",
) -> GigapathDeepBackboneEvaluator:
    """Create a live :class:`GigapathDeepBackboneEvaluator`.

    Parameters
    ----------
    capture_layers : list[int | str] | None
        Layers to capture.  Defaults to all 39 transformer blocks
        ``[0..38]``.
    cls_token_only : bool
        When ``True``, hooks capture only the CLS token (index 0).
    device : str
        Target device.
    """
    if capture_layers is None:
        capture_layers = list(range(39))
    return GigapathDeepBackboneEvaluator(
        capture_layers=capture_layers,
        cls_token_only=cls_token_only,
        device=device,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Factory helper
# ═══════════════════════════════════════════════════════════════════════

def _evaluator_factory(
    capture_layers: list | None = None,
    cls_token_only: bool = False,
    *,
    root: str | None = None,
) -> GigapathDeepBackboneEvaluatorFactory:
    """Create a :class:`GigapathDeepBackboneEvaluatorFactory` spec-block."""
    if capture_layers is None:
        capture_layers = list(range(39))
    return GigapathDeepBackboneEvaluatorFactory(
        root=root,
        spec=dict(
            capture_layers=capture_layers,
            cls_token_only=cls_token_only,
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Single bag
# ═══════════════════════════════════════════════════════════════════════

def gigapath_deep_feature_bag(
    name: str,
    *,
    capture_layers: list | None = None,
    cls_token_only: bool = False,
    batch_size: int = 64,
    shard_size: int = 1024,
    gpu_batch_size: int = 64,
    root: str | None = None,
) -> DeepFeatureBag:
    """Create a single :class:`DeepFeatureBag`.

    Parameters
    ----------
    name : str
        Name identifying the source tile-bag.  Currently supported:
        ``"GIGAPATH_DEEP_CPTAC_SAMPLE"`` (first TileBag in the CPTAC clip).
    capture_layers, cls_token_only
        Forwarded to :func:`_evaluator_factory`.
    batch_size : int
        Number of tiles per evaluator forward pass.
    shard_size : int
        Number of samples per MDS shard.
    gpu_batch_size : int
        Forwarded as a non-hashing kwarg to ``DeepFeatureBag``.
    root : str | None
        Datablock root.
    """
    factory = _evaluator_factory(capture_layers, cls_token_only, root=root)
    if name == "GIGAPATH_DEEP_CPTAC_SAMPLE":
        from autopath.pancan.pipelines import pancan_tile_bag
        tilebag_quote = dbx.quote(pancan_tile_bag, "CPTAC_SAMPLE")
    else:
        raise ValueError(f"Unknown deep feature bag: {name!r}")
    return DeepFeatureBag(
        root=root,
        spec=dict(
            tilebag=tilebag_quote,
            evaluator_factory=dbx.quote(factory),
            batch_size=batch_size,
            shard_size=shard_size,
        ),
        gpu_batch_size=gpu_batch_size,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Clip
# ═══════════════════════════════════════════════════════════════════════

@tagged
def gigapath_deep_feature_clip(
    name: str = None,
    *,
    tag: str | None = None,
    capture_layers: list | None = None,
    cls_token_only: bool = False,
    batch_size: int = 64,
    shard_size: int = 1024,
    root: str | None = None,
    n_workers: int = 1,
    gpu_batch_size: int = 64,
    parallelization: str | None = None,
) -> DeepFeatureClip:
    """Create a :class:`DeepFeatureClip` over a named tile-bag clip.

    Parameters
    ----------
    name : str
        Named configuration.  Supported:
        ``"GIGAPATH_DEEP_CPTAC"`` — full CPTAC tile-bag clip.
        ``"GIGAPATH_DEEP_CPTAC_<partition_fold>"`` — a specific fold
        (e.g. ``"GIGAPATH_DEEP_CPTAC_8020_TRAIN"``).
    tag : str | None
        Human-readable pipeline tag.
    capture_layers, cls_token_only
        Forwarded to :func:`_evaluator_factory`.
    batch_size : int
        Number of tiles per evaluator forward pass.
    shard_size : int
        Number of samples per MDS shard.
    root : str | None
        Datablock root.
    n_workers : int
        Number of parallel workers for building bags.
    gpu_batch_size : int
        Forwarded to ``DeepFeatureClip``.
    parallelization : str | None
        Parallelization strategy (``'inline'``, ``'multithreading'``,
        ``'multiprocessing'``, ``'ray'``).
    """
    if name is None:
        return DeepFeatureClip
    factory = _evaluator_factory(capture_layers, cls_token_only, root=root)
    # Resolve the tile-bag clip
    if name == "GIGAPATH_DEEP_CPTAC":
        tilebagclip = dbx.quote(pancan_tile_bag_clip, "CPTAC")
    elif name.startswith("GIGAPATH_DEEP_CPTAC_"):
        fold_name = "CPTAC_" + name[len("GIGAPATH_DEEP_CPTAC_"):]
        tilebagclip = dbx.quote(pancan_tile_fold, fold_name)
    else:
        raise ValueError(f"Unknown deep feature clip: {name!r}")
    return DeepFeatureClip(
        root=root,
        spec=dict(
            tilebagclip=tilebagclip,
            evaluator_factory=dbx.quote(factory),
            batch_size=batch_size,
            shard_size=shard_size,
        ),
        gpu_batch_size=gpu_batch_size,
        n_workers=n_workers,
        parallelization=parallelization,
        tag=tag,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Dataloader sampler
# ═══════════════════════════════════════════════════════════════════════

def gigapath_deep_feature_clip_dataloader_samples(
    name: str,
    n: int,
    *,
    root: str | None = None,
    capture_layers: list | None = None,
    cls_token_only: bool = False,
    batch_size: int = 4,
    shuffle: bool = False,
    include_tiles: bool = False,
    return_last: bool = True,
    **dataloader_kwargs,
):
    """Iterate *n* samples from a :class:`DeepFeatureClip` dataset.

    Useful for smoke-testing the pipeline and inspecting sample dicts.

    Parameters
    ----------
    name : str
        Forwarded to :func:`gigapath_deep_feature_clip`.
    n : int
        Number of samples to iterate.
    batch_size : int
        DataLoader batch size.
    shuffle, include_tiles
        Forwarded to :meth:`DeepFeatureClip.dataset`.
    return_last : bool
        If ``True``, return the last batch.
    **dataloader_kwargs
        Extra keyword arguments forwarded to
        :class:`torch.utils.data.DataLoader`.
    """
    clip = gigapath_deep_feature_clip(
        name, root=root,
        capture_layers=capture_layers,
        cls_token_only=cls_token_only,
    )
    ds = clip.dataset(shuffle=shuffle, include_tiles=include_tiles)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, **dataloader_kwargs,
    )
    progress = tqdm.tqdm(total=n)
    last = None
    for i, batch in enumerate(loader):
        progress.update(batch_size)
        last = batch
        if (i + 1) * batch_size >= n:
            break
    if return_last:
        return last
