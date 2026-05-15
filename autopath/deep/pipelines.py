"""Pipeline entrypoints for GigaPath deep feature extraction.

Provides declarative constructors for
:class:`~autopath.gigapath.backbone.GigapathDeepBackboneEvaluator`,
:class:`~autopath.deep.features.DeepFeatureBag`,
:class:`~autopath.deep.features.DeepFeatureClip`,
:class:`~autopath.deep.features.SphericalDeepFeatureClip`,
:class:`~autopath.deep.features.CornerDeepFeatureClip`,
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

from autopath.pancan.pipelines import pancan_tile_bag

from autopath.gigapath.backbone import (
    GigapathDeepBackboneEvaluator,
    GigapathDeepBackboneEvaluatorFactory,
)
from autopath.deep.features import (
    DeepFeatureBag,
    DeepFeatureClip,
    SphericalDeepFeatureClip,
    CornerDeepFeatureClip,
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
    capture_blocks: list | None = None,
    capture_layers: list | None = None,
    capture_outputs: bool = True,
    cls_token_only: bool = False,
    device: str = "cuda",
) -> GigapathDeepBackboneEvaluator:
    """Create a live :class:`GigapathDeepBackboneEvaluator`.

    Parameters
    ----------
    capture_blocks : list[int] | None
        Transformer block indices to capture.  Defaults to ``[]``.
    capture_layers : list[str] | None
        Named model layers to capture (e.g. ``["norm"]``).
        Defaults to ``[]``.
    capture_outputs : bool
        When ``True`` (default), store the model's direct output
        under the key ``'output'``.
    cls_token_only : bool
        When ``True``, hooks capture only the CLS token (index 0).
    device : str
        Target device.
    """
    return GigapathDeepBackboneEvaluator(
        capture_blocks=capture_blocks or [],
        capture_layers=capture_layers or [],
        capture_outputs=capture_outputs,
        cls_token_only=cls_token_only,
        device=device,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Evaluator factory
# ═══════════════════════════════════════════════════════════════════════

"""
git commit -am 'deep: evaluator_factory: TEST' > /dev/null || true; dbx.print "autopath.deep.pipelines.gigapath_deep_backbone_evaluator_factory()"
git commit -am 'deep: evaluator_factory: TEST' > /dev/null || true; dbx.print "autopath.deep.pipelines.gigapath_deep_backbone_evaluator_factory(capture_blocks=[0,19,38], cls_token_only=True)"
"""
def gigapath_deep_backbone_evaluator_factory(
    capture_blocks: list | None = None,
    capture_layers: list | None = None,
    capture_outputs: bool = True,
    cls_token_only: bool = False,
    *,
    url: str | None = None,
) -> GigapathDeepBackboneEvaluatorFactory:
    """Create a :class:`GigapathDeepBackboneEvaluatorFactory` spec-block."""
    return GigapathDeepBackboneEvaluatorFactory(
        url=url,
        spec=dict(
            capture_blocks=capture_blocks or [],
            capture_layers=capture_layers or [],
            capture_outputs=capture_outputs,
            cls_token_only=cls_token_only,
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Single bag
# ═══════════════════════════════════════════════════════════════════════

"""
git commit -am 'deep: DeepFeatureBag: BUILD' > /dev/null || true; dbx.print "autopath.deep.pipelines.gigapath_deep_feature_bag('GIGAPATH_DEEP_CPTAC_SAMPLE', \
    cls_token_only=True, \
    shard_size=1024, \
    gpu_batch_size=1024 \
    ).build().valid()"
"""
def gigapath_deep_feature_bag(
    name: str,
    *,
    capture_blocks: list | None = None,
    capture_layers: list | None = None,
    capture_outputs: bool = True,
    cls_token_only: bool = False,
    capture_tiles: bool = True,
    shard_size: int = 1024,
    gpu_batch_size: int = 64,
    url: str | None = None,
) -> DeepFeatureBag:
    """Create a single :class:`DeepFeatureBag`.

    Parameters
    ----------
    name : str
        Name identifying the source tile-bag.  Currently supported:
        ``"GIGAPATH_DEEP_CPTAC_SAMPLE"`` (first TileBag in the CPTAC clip).
    capture_blocks, capture_layers, capture_outputs, cls_token_only
        Forwarded to :func:`gigapath_deep_backbone_evaluator_factory`.
    capture_tiles : bool
        When ``True`` (default), write raw tile images into MDS shards.
    shard_size : int
        Number of samples written per MDS shard file for streaming
        reads.
    gpu_batch_size : int
        Number of tiles sent to the GPU in a single forward pass.
        Controls VRAM usage.  Runtime parameter (not part of the
        config hash).
    url : str | None
        Datablock URL (symbolic specline for relocatability).
    """
    factory = gigapath_deep_backbone_evaluator_factory(
        capture_blocks, capture_layers, capture_outputs, cls_token_only, url=url,
    )
    if name == "GIGAPATH_DEEP_CPTAC_SAMPLE":
        
        tilebag_quote = dbx.quote(pancan_tile_bag, "CPTAC_SAMPLE")
    else:
        raise ValueError(f"Unknown deep feature bag: {name!r}")
    return DeepFeatureBag(
        url=url,
        spec=dict(
            tilebag=tilebag_quote,
            evaluator_factory=dbx.quote(factory),
            shard_size=shard_size,
            capture_tiles=capture_tiles,
        ),
        gpu_batch_size=gpu_batch_size,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Clip
# ═══════════════════════════════════════════════════════════════════════

"""
git commit -am 'deep: DeepFeatureClip: BUILD' > /dev/null || true; dbx.print "autopath.deep.pipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC', \
    cls_token_only=True, \
    shard_size=1024, \
    gpu_batch_size=1024, \
).build()"
git commit -am 'deep: DeepFeatureClip: BUILD fold' > /dev/null || true; dbx.print "autopath.deep.pipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_404020_TRAIN', \
    cls_token_only=True, \
    shard_size=1024, \
    gpu_batch_size=1024, \
).build()"
git commit -am 'deep: DeepFeatureClip: BUILD final layer only' > /dev/null || true; dbx.print "autopath.deep.pipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_404020_TRAIN', \
    capture_blocks=[-1], \
    cls_token_only=True, \
    gpu_batch_size=1024, \
).build()"
git commit -am 'deep: DeepFeatureClip: BUILD final layer only' > /dev/null || true; dbx.print "autopath.deep.pipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_200179_TRAIN', \
    cls_token_only=True, \
    gpu_batch_size=1024, \
    n_devices=3, \
).build().valid()"
"""
@tagged
def gigapath_deep_feature_clip(
    name: str = None,
    *,
    tag: str | None = None,
    capture_blocks: list | None = None,
    capture_layers: list | None = None,
    capture_outputs: bool = True,
    cls_token_only: bool = False,
    capture_tiles: bool = True,
    shard_size: int = 1024,
    url: str | None = None,
    n_devices: int | None = None,
    devices: list | None = None,
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
    capture_blocks, capture_layers, capture_outputs, cls_token_only
        Forwarded to :func:`gigapath_deep_backbone_evaluator_factory`.
    capture_tiles : bool
        When ``True`` (default), write raw tile images into MDS shards.
    shard_size : int
        Number of samples per MDS shard.
    url : str | None
        Datablock URL (symbolic specline for relocatability).
    n_devices : int | None
        Shorthand: use *n* CUDA devices (``['cuda:0', ..., 'cuda:{n-1}']``).
        Ignored if *devices* is explicitly provided.
    devices : list | None
        Explicit list of CUDA devices (e.g. ``["cuda:0", "cuda:1"]``).
        Overrides *n_devices*.  Defaults to ``["cuda"]``.
    gpu_batch_size : int
        Number of tiles sent to the GPU in a single forward pass.
    parallelization : str | None
        Parallelization strategy (``'inline'``, ``'multithreading'``,
        ``'multiprocessing'``, ``'ray'``).  Defaults to
        ``'multiprocessing'`` when multiple devices are used.
    """
    if name is None:
        return DeepFeatureClip
    # Resolve devices.
    if devices is not None:
        pass  # explicit devices wins
    elif n_devices is not None:
        devices = [f"cuda:{i}" for i in range(n_devices)]
    else:
        devices = ["cuda"]
    # Default to multiprocessing when multiple devices are used.
    if parallelization is None and len(devices) > 1:
        parallelization = "multiprocessing"
    factory = gigapath_deep_backbone_evaluator_factory(
        capture_blocks, capture_layers, capture_outputs, cls_token_only, url=url,
    )
    # Resolve the tile-bag clip
    if name == "GIGAPATH_DEEP_CPTAC":
        tilebagclip = dbx.quote(pancan_tile_bag_clip, "CPTAC")
    elif name.startswith("GIGAPATH_DEEP_CPTAC_"):
        fold_name = "CPTAC_" + name[len("GIGAPATH_DEEP_CPTAC_"):]
        tilebagclip = dbx.quote(pancan_tile_fold, fold_name)
    else:
        raise ValueError(f"Unknown deep feature clip: {name!r}")
    return DeepFeatureClip(
        url=url,
        spec=dict(
            tilebagclip=tilebagclip,
            evaluator_factory=dbx.quote(factory),
            shard_size=shard_size,
            capture_tiles=capture_tiles,
        ),
        gpu_batch_size=gpu_batch_size,
        devices=devices,
        n_workers=len(devices),
        parallelization=parallelization,
        tag=tag,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Dataloader sampler
# ═══════════════════════════════════════════════════════════════════════

"""
git commit -am 'deep: dataloader_samples: TEST' > /dev/null || true; dbx.print "autopath.deep.pipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_200179_TRAIN', \
    cls_token_only=True, \
    n=8,\
)"
git commit -am 'deep: dataloader_samples: TEST shuffle' > /dev/null || true; dbx.print "autopath.deep.pipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_404020_TRAIN', \
    capture_blocks=[0,19,38], \
    cls_token_only=True, \
    batch_size=8, \
    shuffle=True, \
    n=16,\
)"
git commit -am 'deep: dataloader_samples: TEST tiles' > /dev/null || true; dbx.print "autopath.deep.pipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC', \
    cls_token_only=True, \
    include_tiles=True, \
    n=4,\
)"
"""
def gigapath_deep_feature_clip_dataloader_samples(
    name: str,
    n: int,
    *,
    url: str | None = None,
    capture_blocks: list | None = None,
    capture_layers: list | None = None,
    cls_token_only: bool = False,
    batch_size: int = 4,
    shuffle: bool = False,
    include_tiles: bool = False,
    skip_unbuilt: bool = True,
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
    shuffle, include_tiles, skip_unbuilt
        Forwarded to :meth:`DeepFeatureClip.dataset`.
    return_last : bool
        If ``True``, return the last batch.
    **dataloader_kwargs
        Extra keyword arguments forwarded to
        :class:`torch.utils.data.DataLoader`.
    """
    clip = gigapath_deep_feature_clip(
        name, 
        url=url,
        capture_blocks=capture_blocks,
        capture_layers=capture_layers,
        cls_token_only=cls_token_only,
    )
    clip.log.verbose(f"dataloader_samples: clip.hash = {clip.hash}")
    if not clip.valid():
        clip.log.verbose(
            f"dataloader_samples: clip NOT valid (hash={clip.hash[:8]})\n"
            f"  validpaths = {clip.validpaths()}\n"
            f"  hashpath   = {clip.hashpath()}"
        )
    ds = clip.dataset(shuffle=shuffle, include_tiles=include_tiles, skip_unbuilt=skip_unbuilt)
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


# ═══════════════════════════════════════════════════════════════════════
#  Spherical (L2-normalised) deep feature clip
# ═══════════════════════════════════════════════════════════════════════

"""
git commit -am 'deep: SphericalDeepFeatureClip: BUILD' > /dev/null || true; dbx.print "autopath.deep.pipelines.gigapath_spherical_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC', \
    cls_token_only=True, \
    shard_size=1024, \
    gpu_batch_size=1024, \
).build()"
git commit -am 'deep: SphericalDeepFeatureClip: BUILD fold' > /dev/null || true; dbx.print "autopath.deep.pipelines.gigapath_spherical_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_404020_TRAIN', \
    cls_token_only=True, \
    shard_size=1024, \
    gpu_batch_size=1024, \
).build()"
"""
@tagged
def gigapath_spherical_deep_feature_clip(
    name: str = None,
    *,
    tag: str | None = None,
    capture_blocks: list | None = None,
    capture_layers: list | None = None,
    capture_outputs: bool = True,
    cls_token_only: bool = False,
    capture_tiles: bool = True,
    shard_size: int = 1024,
    url: str | None = None,
    n_devices: int | None = None,
    devices: list | None = None,
    gpu_batch_size: int = 64,
    parallelization: str | None = None,
) -> SphericalDeepFeatureClip:
    """Create a :class:`SphericalDeepFeatureClip` over a named deep feature clip.

    L2-normalises each layer's features onto the unit hypersphere.
    Pure runtime transformation — delegates storage to the underlying
    :class:`DeepFeatureClip`.

    Parameters
    ----------
    name : str
        Named configuration — same values accepted by
        :func:`gigapath_deep_feature_clip` (e.g.
        ``"GIGAPATH_DEEP_CPTAC"``,
        ``"GIGAPATH_DEEP_CPTAC_404020_TRAIN"``).
    tag : str | None
        Human-readable pipeline tag.
    capture_blocks, capture_layers, capture_outputs, cls_token_only
        Forwarded to :func:`gigapath_deep_backbone_evaluator_factory`.
    capture_tiles, shard_size
        Forwarded to the underlying :class:`DeepFeatureClip`.
    url : str | None
        Datablock URL (symbolic specline for relocatability).
    n_devices, devices, gpu_batch_size, parallelization
        Forwarded to :func:`gigapath_deep_feature_clip` for the
        underlying build.
    """
    if name is None:
        return SphericalDeepFeatureClip
    deep_clip = gigapath_deep_feature_clip(
        name,
        tag=tag,
        capture_blocks=capture_blocks,
        capture_layers=capture_layers,
        capture_outputs=capture_outputs,
        cls_token_only=cls_token_only,
        capture_tiles=capture_tiles,
        shard_size=shard_size,
        url=url,
        n_devices=n_devices,
        devices=devices,
        gpu_batch_size=gpu_batch_size,
        parallelization=parallelization,
    )
    return SphericalDeepFeatureClip(
        url=url,
        spec=dict(deep_feature_clip=dbx.quote(deep_clip)),
        tag=tag,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Corner (bipolar) deep feature clip
# ═══════════════════════════════════════════════════════════════════════

"""
git commit -am 'deep: CornerDeepFeatureClip: BUILD' > /dev/null || true; dbx.print "autopath.deep.pipelines.gigapath_corner_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC', \
    cls_token_only=True, \
    shard_size=1024, \
    gpu_batch_size=1024, \
).build()"
git commit -am 'deep: CornerDeepFeatureClip: BUILD fold' > /dev/null || true; dbx.print "autopath.deep.pipelines.gigapath_corner_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_404020_TRAIN', \
    cls_token_only=True, \
    shard_size=1024, \
    gpu_batch_size=1024, \
).build()"
"""
@tagged
def gigapath_corner_deep_feature_clip(
    name: str = None,
    *,
    tag: str | None = None,
    capture_blocks: list | None = None,
    capture_layers: list | None = None,
    capture_outputs: bool = True,
    cls_token_only: bool = False,
    capture_tiles: bool = True,
    shard_size: int = 1024,
    url: str | None = None,
    n_devices: int | None = None,
    devices: list | None = None,
    gpu_batch_size: int = 64,
    parallelization: str | None = None,
) -> CornerDeepFeatureClip:
    """Create a :class:`CornerDeepFeatureClip` over a named deep feature clip.

    Maps each layer's features to bipolar ``{-1, +1}^d`` hypercube
    vertices via ``sign(x)``.  Pure runtime transformation — delegates
    storage to the underlying :class:`DeepFeatureClip`.

    Parameters
    ----------
    name : str
        Named configuration — same values accepted by
        :func:`gigapath_deep_feature_clip` (e.g.
        ``"GIGAPATH_DEEP_CPTAC"``,
        ``"GIGAPATH_DEEP_CPTAC_404020_TRAIN"``).
    tag : str | None
        Human-readable pipeline tag.
    capture_blocks, capture_layers, capture_outputs, cls_token_only
        Forwarded to :func:`gigapath_deep_backbone_evaluator_factory`.
    capture_tiles, shard_size
        Forwarded to the underlying :class:`DeepFeatureClip`.
    url : str | None
        Datablock URL (symbolic specline for relocatability).
    n_devices, devices, gpu_batch_size, parallelization
        Forwarded to :func:`gigapath_deep_feature_clip` for the
        underlying build.
    """
    if name is None:
        return CornerDeepFeatureClip
    deep_clip = gigapath_deep_feature_clip(
        name,
        tag=tag,
        capture_blocks=capture_blocks,
        capture_layers=capture_layers,
        capture_outputs=capture_outputs,
        cls_token_only=cls_token_only,
        capture_tiles=capture_tiles,
        shard_size=shard_size,
        url=url,
        n_devices=n_devices,
        devices=devices,
        gpu_batch_size=gpu_batch_size,
        parallelization=parallelization,
    )
    return CornerDeepFeatureClip(
        url=url,
        spec=dict(deep_feature_clip=dbx.quote(deep_clip)),
        tag=tag,
    )
