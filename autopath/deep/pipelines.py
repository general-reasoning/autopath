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
from dbx import Logger

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
    pancan_tile_clip,
    pancan_tile_fold,
)
from autopath.autobits import sanitize_collate


log = Logger()


# ═══════════════════════════════════════════════════════════════════════
#  Evaluator
# ═══════════════════════════════════════════════════════════════════════

def gigapath_deep_backbone_evaluator(
    *,
    cfg_capture_blocks: list | None = None,
    cfg_capture_layers: list | None = None,
    cfg_capture_outputs: bool = True,
    cfg_cls_token_only: bool = False,
    device: str = "cuda",
) -> GigapathDeepBackboneEvaluator:
    """Create a live :class:`GigapathDeepBackboneEvaluator`.

    Parameters
    ----------
    cfg_capture_blocks : list[int] | None
        Transformer block indices to capture.  Defaults to ``[]``.
    cfg_capture_layers : list[str] | None
        Named model layers to capture (e.g. ``["norm"]``).
        Defaults to ``[]``.
    cfg_capture_outputs : bool
        When ``True`` (default), store the model's direct output
        under the key ``'output'``.
    cfg_cls_token_only : bool
        When ``True``, hooks capture only the CLS token (index 0).
    device : str
        Target device.
    """
    return GigapathDeepBackboneEvaluator(
        capture_blocks=cfg_capture_blocks or [],
        capture_layers=cfg_capture_layers or [],
        capture_outputs=cfg_capture_outputs,
        cls_token_only=cfg_cls_token_only,
        device=device,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Evaluator factory
# ═══════════════════════════════════════════════════════════════════════

def gigapath_deep_backbone_evaluator_factory(
    cfg_capture_blocks: list | None = None,
    cfg_capture_layers: list | None = None,
    cfg_capture_outputs: bool = True,
    cfg_cls_token_only: bool = False,
    *,
    url: str | None = None,
) -> GigapathDeepBackboneEvaluatorFactory:
    """Create a :class:`GigapathDeepBackboneEvaluatorFactory` spec-block."""
    return GigapathDeepBackboneEvaluatorFactory(
        url=url,
        spec=dict(
            capture_blocks=cfg_capture_blocks or [],
            capture_layers=cfg_capture_layers or [],
            capture_outputs=cfg_capture_outputs,
            cls_token_only=cfg_cls_token_only,
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Single bag
# ═══════════════════════════════════════════════════════════════════════

"""
### FINAL layer only
git commit -am 'deep: DeepFeatureBag: BUILD' > /dev/null || true; dbx.pprint "autopath.deep.pipelines.gigapath_deep_feature_bag('GIGAPATH_DEEP_CPTAC_SAMPLE', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    gpu_batch_size=1024 \
    ).build().valid()"
### DEEP layers
git commit -am 'deep: DeepFeatureBag: BUILD' > /dev/null || true; dbx.pprint "autopath.deep.pipelines.gigapath_deep_feature_bag('GIGAPATH_DEEP_CPTAC_SAMPLE', \
    cfg_cls_token_only=True, \
    cfg_capture_blocks='all', \
    cfg_shard_size=128, \
    gpu_batch_size=128 \
    ).build().valid()"
# [01:09<00:00, 34.52s/batch, VRAM 4.6/42GB (peak 23.6GB)]

git commit -am 'deep: DeepFeatureBag: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_deep_feature_bag('GIGAPATH_DEEP_CPTAC_SAMPLE', \
    cfg_cls_token_only=False, \
    cfg_capture_blocks='all', \
    cfg_shard_size=64, \
    gpu_batch_size=1024 \
    ).build().valid()\
    "
git commit -am 'deep: DeepFeatureBag: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_deep_feature_bag('GIGAPATH_DEEP_CPTAC_SAMPLE', \
    cfg_cls_token_only=False, \
    cfg_capture_blocks=[0, 7, 14, 21, 27, 33, 39], \
    cfg_shard_size=128, \
    gpu_batch_size=1024 \
    ).build().valid()\
    "
"""
def gigapath_deep_feature_bag(
    name: str,
    *,
    cfg_capture_blocks: list | None = None,
    cfg_capture_layers: list | None = None,
    cfg_capture_outputs: bool = True,
    cfg_cls_token_only: bool = False,
    cfg_shard_size: int = 1024,
    gpu_batch_size: int = 64,
    url: str | None = None,
) -> DeepFeatureBag:
    """Create a single :class:`DeepFeatureBag`.

    Parameters
    ----------
    name : str
        Name identifying the source tile-bag.  Currently supported:
        ``"GIGAPATH_DEEP_CPTAC_SAMPLE"`` (first TileBag in the CPTAC clip).
    cfg_capture_blocks, cfg_capture_layers, cfg_capture_outputs, cfg_cls_token_only
        Forwarded to :func:`gigapath_deep_backbone_evaluator_factory`.
    cfg_shard_size : int
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
        cfg_capture_blocks, cfg_capture_layers, cfg_capture_outputs, cfg_cls_token_only, url=url,
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
            shard_size=cfg_shard_size,
        ),
        gpu_batch_size=gpu_batch_size,
        keyby='tag_version_hash',
    )


# ═══════════════════════════════════════════════════════════════════════
#  Clip
# ═══════════════════════════════════════════════════════════════════════

"""
### CPTAC: OUTPUT-ONLY|CLS-ONLY
## 1 device
git commit -am 'deep: DeepFeatureClip: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    gpu_batch_size=512, \
    n_devices=1,\
).build_tree()"
## 3 devices
git commit -am 'deep: DeepFeatureClip: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    gpu_batch_size=512, \
    n_devices=3,\
    parallelization='multiprocessing',\
).build_tree()"
### CPTAC_602020_TRAIN|CALIBRATE|TEST: OUTPUT-ONLY|ALL_HEADS
## piggyback on CPTAC clip
git commit -am 'deep: DeepFeatureClip: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
).build_tree()"
git commit -am 'deep: DeepFeatureClip: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_CALIBRATE', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
).build_tree()"
git commit -am 'deep: DeepFeatureClip: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
).build_tree()"
"""
def gigapath_deep_feature_clip(
    name: str = None,
    *,
    cfg_capture_blocks: list | None = None,
    cfg_capture_layers: list | None = None,
    cfg_capture_outputs: bool = True,
    cfg_cls_token_only: bool = False,
    cfg_shard_size: int = 1024,
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
        ``"GIGAPATH_DEEP_CPTAC"`` -- full CPTAC tile-bag clip.
        ``"GIGAPATH_DEEP_CPTAC_<partition_fold>"`` -- a specific fold
        (e.g. ``"GIGAPATH_DEEP_CPTAC_8020_TRAIN"``).
    cfg_capture_blocks, cfg_capture_layers, cfg_capture_outputs, cfg_cls_token_only
        Forwarded to :func:`gigapath_deep_backbone_evaluator_factory`.
    cfg_shard_size : int
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
        cfg_capture_blocks, cfg_capture_layers, cfg_capture_outputs, cfg_cls_token_only, url=url,
    )
    # Resolve the tile-bag clip
    if name == "GIGAPATH_DEEP_CPTAC":
        tilebagclip = dbx.quote(pancan_tile_clip, "CPTAC")
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
            shard_size=cfg_shard_size,
        ),
        gpu_batch_size=gpu_batch_size,
        devices=devices,
        n_workers=len(devices),
        parallelization=parallelization,
        keyby='tag_version_hash',
    )


# ═══════════════════════════════════════════════════════════════════════
#  Dataloader sampler
# ═══════════════════════════════════════════════════════════════════════

"""
git commit -am 'deep: dataloader_samples: TEST' > /dev/null || true; dbx.pprint "autopath.deep.pipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64,\
    include_tiles=True,\
    n=8,\
)"
git commit -am 'deep: dataloader_samples: TEST' > /dev/null || true; dbx.pprint "autopath.deep.pipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_200179_TRAIN', \
    cfg_cls_token_only=True, \
    n=8,\
)"
git commit -am 'deep: dataloader_samples: TEST shuffle' > /dev/null || true; dbx.pprint "autopath.deep.pipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_404020_TRAIN', \
    cfg_capture_blocks=[0,19,38], \
    cfg_cls_token_only=True, \
    batch_size=8, \
    shuffle=True, \
    n=16,\
)"
git commit -am 'deep: dataloader_samples: TEST tiles' > /dev/null || true; dbx.pprint "autopath.deep.pipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC', \
    cfg_cls_token_only=True, \
    include_tiles=True, \
    n=4,\
)"
"""
def gigapath_deep_feature_clip_dataloader_samples(
    name: str,
    n: int,
    *,
    url: str | None = None,
    cfg_capture_blocks: list | None = None,
    cfg_capture_layers: list | None = None,
    cfg_cls_token_only: bool = False,
    cfg_shard_size: int = 1024,
    batch_size: int = 4,
    shuffle: bool = False,
    include_tiles: bool = False,
    skip_invalid_bags: bool = True,
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
    cfg_shard_size : int
        Forwarded to :func:`gigapath_deep_feature_clip` so the config
        hash matches the built clip.
    batch_size : int
        DataLoader batch size.
    shuffle, include_tiles, skip_invalid_bags
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
        cfg_capture_blocks=cfg_capture_blocks,
        cfg_capture_layers=cfg_capture_layers,
        cfg_cls_token_only=cfg_cls_token_only,
        cfg_shard_size=cfg_shard_size,
    )
    assert clip.valid(), (
        f"DeepFeatureClip is not valid (hash={clip.hash[:8]}). "
        f"Build it first with gigapath_deep_feature_clip(...).build()\n"
        f"  validpaths = {clip.validpaths()}\n"
        f"  anchorkeypath = {clip.anchorkeypath}"
    )
    ds = clip.dataset(shuffle=shuffle, include_tiles=include_tiles, skip_invalid_bags=skip_invalid_bags, batch_size=batch_size)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, collate_fn=sanitize_collate, **dataloader_kwargs,
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
git commit -am 'deep: SphericalDeepFeatureClip: BUILD' > /dev/null || true; dbx.pprint "autopath.deep.pipelines.gigapath_spherical_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC', \
    cfg_cls_token_only=True, \
    cfg_shard_size=1024, \
    gpu_batch_size=1024, \
).build()"
git commit -am 'deep: SphericalDeepFeatureClip: BUILD fold' > /dev/null || true; dbx.pprint "autopath.deep.pipelines.gigapath_spherical_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_404020_TRAIN', \
    cfg_cls_token_only=True, \
    cfg_shard_size=1024, \
    gpu_batch_size=1024, \
).build()"
"""
def gigapath_spherical_deep_feature_clip(
    name: str = None,
    *,
    cfg_capture_blocks: list | None = None,
    cfg_capture_layers: list | None = None,
    cfg_capture_outputs: bool = True,
    cfg_cls_token_only: bool = False,
    cfg_shard_size: int = 1024,
    url: str | None = None,
    n_devices: int | None = None,
    devices: list | None = None,
    gpu_batch_size: int = 64,
    parallelization: str | None = None,
) -> SphericalDeepFeatureClip:
    """Create a :class:`SphericalDeepFeatureClip` over a named deep feature clip.

    L2-normalises each layer's features onto the unit hypersphere.
    Pure runtime transformation -- delegates storage to the underlying
    :class:`DeepFeatureClip`.

    Parameters
    ----------
    name : str
        Named configuration -- same values accepted by
        :func:`gigapath_deep_feature_clip` (e.g.
        ``"GIGAPATH_DEEP_CPTAC"``,
        ``"GIGAPATH_DEEP_CPTAC_404020_TRAIN"``).
    cfg_capture_blocks, cfg_capture_layers, cfg_capture_outputs, cfg_cls_token_only
        Forwarded to :func:`gigapath_deep_backbone_evaluator_factory`.
    cfg_shard_size
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
        cfg_capture_blocks=cfg_capture_blocks,
        cfg_capture_layers=cfg_capture_layers,
        cfg_capture_outputs=cfg_capture_outputs,
        cfg_cls_token_only=cfg_cls_token_only,
        cfg_shard_size=cfg_shard_size,
        url=url,
        n_devices=n_devices,
        devices=devices,
        gpu_batch_size=gpu_batch_size,
        parallelization=parallelization,
    )
    return SphericalDeepFeatureClip(
        url=url,
        spec=dict(deep_feature_clip=dbx.quote(deep_clip)),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Corner (bipolar) deep feature clip
# ═══════════════════════════════════════════════════════════════════════

"""
git commit -am 'deep: CornerDeepFeatureClip: BUILD' > /dev/null || true; dbx.pprint "autopath.deep.pipelines.gigapath_corner_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC', \
    cfg_cls_token_only=True, \
    cfg_shard_size=1024, \
    gpu_batch_size=1024, \
).build()"
git commit -am 'deep: CornerDeepFeatureClip: BUILD fold' > /dev/null || true; dbx.pprint "autopath.deep.pipelines.gigapath_corner_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_404020_TRAIN', \
    cfg_cls_token_only=True, \
    cfg_shard_size=1024, \
    gpu_batch_size=1024, \
).build()"
"""
def gigapath_corner_deep_feature_clip(
    name: str = None,
    *,
    cfg_capture_blocks: list | None = None,
    cfg_capture_layers: list | None = None,
    cfg_capture_outputs: bool = True,
    cfg_cls_token_only: bool = False,
    cfg_shard_size: int = 1024,
    url: str | None = None,
    n_devices: int | None = None,
    devices: list | None = None,
    gpu_batch_size: int = 64,
    parallelization: str | None = None,
) -> CornerDeepFeatureClip:
    """Create a :class:`CornerDeepFeatureClip` over a named deep feature clip.

    Maps each layer's features to bipolar ``{-1, +1}^d`` hypercube
    vertices via ``sign(x)``.  Pure runtime transformation -- delegates
    storage to the underlying :class:`DeepFeatureClip`.

    Parameters
    ----------
    name : str
        Named configuration -- same values accepted by
        :func:`gigapath_deep_feature_clip` (e.g.
        ``"GIGAPATH_DEEP_CPTAC"``,
        ``"GIGAPATH_DEEP_CPTAC_404020_TRAIN"``).
    cfg_capture_blocks, cfg_capture_layers, cfg_capture_outputs, cfg_cls_token_only
        Forwarded to :func:`gigapath_deep_backbone_evaluator_factory`.
    cfg_shard_size
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
        cfg_capture_blocks=cfg_capture_blocks,
        cfg_capture_layers=cfg_capture_layers,
        cfg_capture_outputs=cfg_capture_outputs,
        cfg_cls_token_only=cfg_cls_token_only,
        cfg_shard_size=cfg_shard_size,
        url=url,
        n_devices=n_devices,
        devices=devices,
        gpu_batch_size=gpu_batch_size,
        parallelization=parallelization,
    )
    return CornerDeepFeatureClip(
        url=url,
        spec=dict(deep_feature_clip=dbx.quote(deep_clip)),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Affine logistic probe
# ═══════════════════════════════════════════════════════════════════════

"""
### CPTAC 60/20/20 — raw features (no normalization)
git commit -am 'deep: AffineLogisticProbe: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
).build_tree()"
git commit -am 'deep: AffineLogisticProbe: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_CALIBRATE', \
    layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
).build_tree()"
git commit -am 'deep: AffineLogisticProbe: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
).build_tree()"
### CPTAC 60/20/20 — L2-normalised features
git commit -am 'deep: AffineLogisticProbe(l2): BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    normalize='l2', \
).build_tree()"
git commit -am 'deep: AffineLogisticProbe(l2): BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_CALIBRATE', \
    layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    normalize='l2', \
).build_tree()"
git commit -am 'deep: AffineLogisticProbe(l2): BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    normalize='l2', \
).build_tree()"
### CPTAC 60/20/20 — corner-linfty features
git commit -am 'deep: AffineLogisticProbe(corner-linfty): BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    normalize='corner-linfty', \
).build_tree()"
git commit -am 'deep: AffineLogisticProbe(corner-linfty): BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_CALIBRATE', \
    layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    normalize='corner-linfty', \
).build_tree()"
git commit -am 'deep: AffineLogisticProbe(corner-linfty): BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    normalize='corner-linfty', \
).build_tree()"
"""
def deep_feature_affine_logistic_probe(
    name: str,
    *,
    layer: str = 'output',
    annotation_key: str | None = None,
    fit_intercept: bool = True,
    evaluation_fraction: float = 0.8,
    normalize: str | None = None,
    cfg_capture_blocks: list | None = None,
    cfg_capture_layers: list | None = None,
    cfg_capture_outputs: bool = True,
    cfg_cls_token_only: bool = False,
    cfg_shard_size: int = 64,
    url: str | None = None,
) -> 'DeepFeatureAffineLogisticProbe':
    """Create a :class:`DeepFeatureAffineLogisticProbe` on a named clip.

    Fits a logistic regression classifier on bag-level mean features
    for the specified ``layer``, persisting the fitted model's
    ``coef_``, ``intercept_``, and ``classes_`` arrays.

    Parameters
    ----------
    name : str
        Named configuration — same values accepted by
        :func:`gigapath_deep_feature_clip` (e.g.
        ``"GIGAPATH_DEEP_CPTAC_602020_TRAIN"``).
    layer : str
        Which capture key to probe (e.g. ``"output"``,
        ``"B_0_norm1"``).
    annotation_key : str | None
        Dotted path into the annotation dict for label extraction
        (e.g. ``"cohort"``).  ``None`` falls back to
        ``tilebag.label``.
    fit_intercept : bool
        Whether to fit an intercept term in the logistic regression.
    evaluation_fraction : float
        Fraction of bags used for training (rest for evaluation).
    normalize : str | None
        Feature normalization mode:
        ``None`` — raw features,
        ``'l2'`` — L2-normalise,
        ``'corner-l1'`` / ``'corner-l2'`` — snap to {-1,+1}^d,
        ``'corner-linfty'`` — axis-aligned vertex.
    cfg_capture_blocks, cfg_capture_layers, cfg_capture_outputs, cfg_cls_token_only
        Forwarded to :func:`gigapath_deep_feature_clip` to identify
        the underlying clip.
    cfg_shard_size : int
        Forwarded to :func:`gigapath_deep_feature_clip`.
    url : str | None
        Datablock URL.

    Examples
    --------
    Build on the CPTAC 60/20/20 train fold with no normalization::

        probe = deep_feature_affine_logistic_probe(
            'GIGAPATH_DEEP_CPTAC_602020_TRAIN',
            layer='output',
            cfg_cls_token_only=True,
            cfg_shard_size=64,
        )
        probe.build()

    With L2 normalization::

        probe = deep_feature_affine_logistic_probe(
            'GIGAPATH_DEEP_CPTAC_602020_TRAIN',
            layer='output',
            cfg_cls_token_only=True,
            cfg_shard_size=64,
            normalize='l2',
        )
        probe.build()

    With corner-linfty normalization::

        probe = deep_feature_affine_logistic_probe(
            'GIGAPATH_DEEP_CPTAC_602020_TRAIN',
            layer='output',
            cfg_cls_token_only=True,
            cfg_shard_size=64,
            normalize='corner-linfty',
        )
        probe.build()
    """
    from autopath.deep.probes import DeepFeatureAffineLogisticProbe

    clip = gigapath_deep_feature_clip(
        name,
        cfg_capture_blocks=cfg_capture_blocks,
        cfg_capture_layers=cfg_capture_layers,
        cfg_capture_outputs=cfg_capture_outputs,
        cfg_cls_token_only=cfg_cls_token_only,
        cfg_shard_size=cfg_shard_size,
        url=url,
    )
    return DeepFeatureAffineLogisticProbe(
        url=url,
        spec=dict(
            clip=dbx.quote(clip),
            layer=layer,
            annotation_key=annotation_key,
            fit_intercept=fit_intercept,
            evaluation_fraction=evaluation_fraction,
            normalize=normalize,
        ),
    )
