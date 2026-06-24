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

import os

from typing import Literal

import tqdm
import torch
import numpy as np

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
    BipolarDeepFeatureClip,
)
from autopath.deep.probes import (
    DeepFeatureAffineLogisticProbe,
    DeepFeatureStatsProbe,
)
from autopath.pancan.pipelines import (
    pancan_tile_clip,
    pancan_tile_fold,
)
from autopath.autobits import sanitize_collate, ValidateTileFeatureZipStreamingDataset, ZipStreamingDataset


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
## CLS-only
git commit -am 'deep: DeepFeatureBag: BUILD' > /dev/null || true; dbx.pprint "autopath.deep.pipelines.gigapath_deep_feature_bag('GIGAPATH_DEEP_CPTAC_SAMPLE', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    gpu_batch_size=1024 \
    ).build().valid(topic=None)"
### ALL DEEP layers
## CLS-only
git commit -am 'deep: DeepFeatureBag: BUILD' > /dev/null || true; dbx.pprint "autopath.deep.pipelines.gigapath_deep_feature_bag('GIGAPATH_DEEP_CPTAC_SAMPLE', \
    cfg_cls_token_only=True, \
    cfg_capture_blocks='all', \
    cfg_shard_size=128, \
    gpu_batch_size=128 \
    ).build().valid(topic=None)"
# [01:09<00:00, 34.52s/batch, VRAM 4.6/42GB (peak 23.6GB)]
## CLS + PATCH tokens
git commit -am 'deep: DeepFeatureBag: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_deep_feature_bag('GIGAPATH_DEEP_CPTAC_SAMPLE', \
    cfg_cls_token_only=False, \
    cfg_capture_blocks='all', \
    cfg_shard_size=64, \
    gpu_batch_size=1024 \
    ).build().valid(topic=None)\
    "
### SOME DEEP layers
## CLS-only
git commit -am 'deep: DeepFeatureBag: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_deep_feature_bag('GIGAPATH_DEEP_CPTAC_SAMPLE', \
    cfg_cls_token_only=True, \
    cfg_capture_blocks=[0, 7, 14, 21, 27, 33, 39], \
    cfg_shard_size=128, \
    gpu_batch_size=1024 \
    ).build().valid(topic=None)\
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
git commit -am 'gigapath_deep_feature_clip_dataloader_samples: TEST' > /dev/null || true; dbx.pprint "autopath.deep.pipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64,\
    n=8,\
)"
git commit -am 'gigapath_deep_feature_clip_dataloader_samples: TEST' > /dev/null || true; dbx.pprint "autopath.deep.pipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_200179_TRAIN', \
    cfg_cls_token_only=True, \
    n=8,\
)"
git commit -am 'gigapath_deep_feature_clip_dataloader_samples: TEST shuffle' > /dev/null || true; dbx.pprint "autopath.deep.pipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_404020_TRAIN', \
    cfg_capture_blocks=[0,19,38], \
    cfg_cls_token_only=True, \
    batch_size=8, \
    shuffle=True, \
    n=16,\
)"
git commit -am 'gigapath_deep_feature_clip_dataloader_samples: TEST tiles' > /dev/null || true; dbx.pprint "autopath.deep.pipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC', \
    cfg_cls_token_only=True, \
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
    skip_invalid_bags: bool = False,
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
    shuffle, skip_invalid_bags
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
    assert clip.valid(topic=None), (
        f"DeepFeatureClip is not valid (hash={clip.hash[:8]}). "
        f"Build it first with gigapath_deep_feature_clip(...).build()\n"
        f"  validpaths = {clip.validpaths()}\n"
        f"  anchorkeypath = {clip.anchorkeypath}"
    )
    ds = clip.dataset(shuffle=shuffle, skip_invalid_bags=skip_invalid_bags, batch_size=batch_size)
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
#  Affine logistic probe
# ═══════════════════════════════════════════════════════════════════════

"""
### CPTAC 60/20/20 — no normalization
git commit -am 'deep: AffineLogisticProbe: BUILD' > /dev/null || true; ç

### CPTAC 60/20/20 — L2-normalised features
git commit -am 'deep: AffineLogisticProbe(l2): BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    cfg_normalize='l2', \
).build_tree()"

### CPTAC 60/20/20 — corner-linfty features
git commit -am 'deep: AffineLogisticProbe(corner-linfty): BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    cfg_normalize='corner-linfty', \
).build_tree()"

### CPTAC 60/20/20 — corner-linfty features
git commit -am 'deep: AffineLogisticProbe(corner-linfty): BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    cfg_normalize='corner-l1', \
).build_tree()"
"""
def gigapath_deep_feature_affine_logistic_probe(
    name: str,
    *,
    cfg_layer: str = 'output',
    cfg_annotation_key: str | None = None,
    cfg_fit_intercept: bool = True,
    cfg_evaluation_fraction: float = 0.8,
    cfg_normalize: str | None = None,
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
    cfg_layer : str
        Which capture key to probe (e.g. ``"output"``,
        ``"B_0_norm1"``).
    cfg_annotation_key : str | None
        Dotted path into the annotation dict for label extraction
        (e.g. ``"cohort"``).  ``None`` falls back to
        ``tilebag.label``.
    cfg_fit_intercept : bool
        Whether to fit an intercept term in the logistic regression.
    cfg_evaluation_fraction : float
        Fraction of bags used for training (rest for evaluation).
    cfg_normalize : str | None
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

        probe = gigapath_deep_feature_affine_logistic_probe(
            'GIGAPATH_DEEP_CPTAC_602020_TRAIN',
            cfg_layer='output',
            cfg_cls_token_only=True,
            cfg_shard_size=64,
        )
        probe.build()

    With L2 normalization::

        probe = gigapath_deep_feature_affine_logistic_probe(
            'GIGAPATH_DEEP_CPTAC_602020_TRAIN',
            cfg_layer='output',
            cfg_cls_token_only=True,
            cfg_shard_size=64,
            cfg_normalize='l2',
        )
        probe.build()

    With corner-linfty normalization::

        probe = gigapath_deep_feature_affine_logistic_probe(
            'GIGAPATH_DEEP_CPTAC_602020_TRAIN',
            cfg_layer='output',
            cfg_cls_token_only=True,
            cfg_shard_size=64,
            cfg_normalize='corner-linfty',
        )
        probe.build()
    """


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
            layer=cfg_layer,
            annotation_key=cfg_annotation_key,
            fit_intercept=cfg_fit_intercept,
            evaluation_fraction=cfg_evaluation_fraction,
            normalize=cfg_normalize,
        ),
    )



# ═══════════════════════════════════════════════════════════════════════
#  Deep Feature Stats Probe pipeline
# ═══════════════════════════════════════════════════════════════════════

"""
### CPTAC 60/20/20 — raw features
git commit -am 'deep: DeepFeatureStatsProbe: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_deep_feature_stats_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_CALIBRATE', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
).build_tree()"
"""
def gigapath_deep_feature_stats_probe(
    name: str,
    *,
    cfg_layer: str = 'output',
    cfg_normalize: str | None = None,
    cfg_capture_blocks: list | None = None,
    cfg_capture_layers: list | None = None,
    cfg_capture_outputs: bool = True,
    cfg_cls_token_only: bool = False,
    cfg_shard_size: int = 64,
    url: str | None = None,
) -> 'DeepFeatureStatsProbe':
    """Create a :class:`DeepFeatureStatsProbe` on a named clip.

    Computes per-dimension tile-level and bag-level statistics
    (mean, std, median, min, max, L2 norms, distinct counts) for
    the specified capture ``layer``.

    Parameters
    ----------
    name : str
        Named configuration — same values accepted by
        :func:`gigapath_deep_feature_clip` (e.g.
        ``"GIGAPATH_DEEP_CPTAC_602020_CALIBRATE"``).
    cfg_layer : str
        Which capture key to probe (e.g. ``"output"``).
    cfg_normalize : str | None
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
    ::

        stats = gigapath_deep_feature_stats_probe(
            'GIGAPATH_DEEP_CPTAC_602020_CALIBRATE',
            cfg_layer='output',
            cfg_cls_token_only=True,
        )
        stats.build()
        print(stats.tile_feature_mean)
    """


    clip = gigapath_deep_feature_clip(
        name,
        cfg_capture_blocks=cfg_capture_blocks,
        cfg_capture_layers=cfg_capture_layers,
        cfg_capture_outputs=cfg_capture_outputs,
        cfg_cls_token_only=cfg_cls_token_only,
        cfg_shard_size=cfg_shard_size,
        url=url,
    )
    return DeepFeatureStatsProbe(
        url=url,
        spec=dict(
            clip=dbx.quote(clip),
            layer=cfg_layer,
            normalize=cfg_normalize,
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Bipolar deep feature clip
# ═══════════════════════════════════════════════════════════════════════

"""
### CPTAC 60/20/20 — bipolar features (TRAIN, stats from CALIBRATE)
git commit -am 'deep: BipolarDeepFeatureClip: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_bipolar_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    n_workers=8,\
).build_tree()"
### CPTAC 60/20/20 — bipolar features (TEST, stats from CALIBRATE)
git commit -am 'deep: BipolarDeepFeatureClip: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_bipolar_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    n_workers=8,\
).build_tree()"
### CPTAC 60/20/20 — bipolar features + ternarization (TRAIN, stats from CALIBRATE)
git commit -am 'deep: BipolarDeepFeatureClip: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_bipolar_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_ternarize_tiles=True, \
    cfg_shard_size=64, \
    n_workers=8,\
).build_tree()"
### CPTAC 60/20/20 — bipolar features + ternarization (TEST, stats from CALIBRATE)
git commit -am 'deep: BipolarDeepFeatureClip: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_bipolar_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_ternarize_tiles=True, \
    cfg_shard_size=64, \
    n_workers=8,\
).build_tree()"
"""
def gigapath_bipolar_deep_feature_clip(
    name: str,
    *,
    cfg_layer: str = 'output',
    cfg_bag_aggregation_threshold: float = 0.5,
    cfg_ternarize_tiles: bool = False,
    cfg_stats_probe_name: str | None = None,
    cfg_capture_blocks: list | None = None,
    cfg_capture_layers: list | None = None,
    cfg_capture_outputs: bool = True,
    cfg_cls_token_only: bool = False,
    cfg_shard_size: int = 64,
    url: str | None = None,
    n_workers: int = 1,
    parallelization: str | None = None,
) -> 'BipolarDeepFeatureClip':
    """Create a :class:`BipolarDeepFeatureClip` on a named clip.

    Builds median-thresholded bipolar features (``{-1, +1}^d`` per tile,
    ``{-1, 0, +1}^d`` per bag) for the specified capture ``layer``.

    The ``clip`` (which bags to bipolarize) comes from ``name``.
    The ``stats_probe`` (which provides the median threshold) defaults
    to the CALIBRATE fold of the same partition to avoid data leakage.
    Override with ``cfg_stats_probe_name``.

    Parameters
    ----------
    name : str
        Named configuration — same values accepted by
        :func:`gigapath_deep_feature_clip` (e.g.
        ``"GIGAPATH_DEEP_CPTAC_602020_TRAIN"``).
    cfg_layer : str
        Which capture key to bipolarize (e.g. ``"output"``).
    cfg_bag_aggregation_threshold : float
        Threshold for bag-level bipolar aggregation.
        ``abs(mean) >= threshold → sign(mean)``, else ``0``.
    cfg_stats_probe_name : str | None
        Name for the stats probe clip.  Defaults to the CALIBRATE
        fold of the same partition (e.g.
        ``"GIGAPATH_DEEP_CPTAC_602020_CALIBRATE"`` for a
        ``602020`` partition).
    cfg_capture_blocks, cfg_capture_layers, cfg_capture_outputs, cfg_cls_token_only
        Forwarded to :func:`gigapath_deep_feature_clip` to identify
        the underlying clip.
    cfg_shard_size : int
        Forwarded to :func:`gigapath_deep_feature_clip`.
    url : str | None
        Datablock URL.
    n_workers : int
        Number of parallel workers for building bipolar bags.
    parallelization : str | None
        Parallelization strategy (``'inline'``, ``'multithreading'``,
        ``'multiprocessing'``).  Defaults to ``'multiprocessing'``
        when ``n_workers > 1``.

    Examples
    --------
    ::

        bipolar = gigapath_bipolar_deep_feature_clip(
            'GIGAPATH_DEEP_CPTAC_602020_TRAIN',
            cfg_layer='output',
            cfg_cls_token_only=True,
        )
        bipolar.build_tree()
    """


    if parallelization is None and n_workers > 1:
        parallelization = 'multiprocessing'

    # The clip whose bags are bipolarized.
    clip = gigapath_deep_feature_clip(
        name,
        cfg_capture_blocks=cfg_capture_blocks,
        cfg_capture_layers=cfg_capture_layers,
        cfg_capture_outputs=cfg_capture_outputs,
        cfg_cls_token_only=cfg_cls_token_only,
        cfg_shard_size=cfg_shard_size,
        url=url,
    )

    # Stats probe — default to CALIBRATE fold.
    if cfg_stats_probe_name is None:
        # "GIGAPATH_DEEP_CPTAC_602020_TRAIN" → "GIGAPATH_DEEP_CPTAC_602020_CALIBRATE"
        parts = name.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in ('TRAIN', 'TEST', 'CALIBRATE'):
            cfg_stats_probe_name = parts[0] + '_CALIBRATE'
        else:
            cfg_stats_probe_name = name

    stats_probe = gigapath_deep_feature_stats_probe(
        cfg_stats_probe_name,
        cfg_layer=cfg_layer,
        cfg_capture_blocks=cfg_capture_blocks,
        cfg_capture_layers=cfg_capture_layers,
        cfg_capture_outputs=cfg_capture_outputs,
        cfg_cls_token_only=cfg_cls_token_only,
        cfg_shard_size=cfg_shard_size,
        url=url,
    )
    return BipolarDeepFeatureClip(
        url=url,
        spec=dict(
            clip=dbx.quote(clip),
            stats_probe=dbx.quote(stats_probe),
            layer=cfg_layer,
            bag_aggregation_threshold=cfg_bag_aggregation_threshold,
            ternarize_tiles=cfg_ternarize_tiles,
        ),
        n_workers=n_workers,
        parallelization=parallelization,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Bipolar deep feature affine logistic probe
# ═══════════════════════════════════════════════════════════════════════

"""
### CPTAC 60/20/20 — bipolar logistic probe
git commit -am 'deep: bipolar logistic probe: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_bipolar_deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
).build_tree()"
git commit -am 'deep: bipolar logistic probe: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_bipolar_deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    cfg_normalize='l2', \
).build_tree()"
git commit -am 'deep: bipolar logistic probe: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_bipolar_deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    cfg_ternarize_tiles=True, \
).build_tree()"
### CPTAC 60/20/20 — bipolar logistic probe
git commit -am 'deep: bipolar logistic probe: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_bipolar_deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
).build_tree()"
### CPTAC 60/20/20 — bipolar logistic probe
git commit -am 'deep: bipolar logistic probe: BUILD' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_bipolar_deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    cfg_ternarize_tiles=True, \
).build_tree()"
"""
def gigapath_bipolar_deep_feature_affine_logistic_probe(
    name: str,
    *,
    cfg_layer: str = 'output',
    cfg_annotation_key: str | None = None,
    cfg_fit_intercept: bool = True,
    cfg_evaluation_fraction: float = 0.8,
    cfg_normalize: str | None = None,
    cfg_bag_aggregation_threshold: float = 0.5,
    cfg_ternarize_tiles: bool = False,
    cfg_stats_probe_name: str | None = None,
    cfg_capture_blocks: list | None = None,
    cfg_capture_layers: list | None = None,
    cfg_capture_outputs: bool = True,
    cfg_cls_token_only: bool = False,
    cfg_shard_size: int = 64,
    url: str | None = None,
) -> 'DeepFeatureAffineLogisticProbe':
    """Create a :class:`DeepFeatureAffineLogisticProbe` on a bipolar clip.

    Fits a logistic regression classifier on bag-level mean bipolar
    features for the specified ``layer``.

    Parameters
    ----------
    name : str
        Named configuration (e.g.
        ``"GIGAPATH_DEEP_CPTAC_602020_TRAIN"``).
    cfg_layer : str
        Which capture key to bipolarize and probe.
    cfg_annotation_key : str | None
        Dotted path into annotations for label extraction.
    cfg_fit_intercept : bool
        Whether to fit an intercept in the logistic regression.
    cfg_evaluation_fraction : float
        Fraction of bags used for training.
    cfg_normalize : str | None
        Feature normalization mode.
    cfg_bag_aggregation_threshold : float
        Threshold for bag-level bipolar aggregation.
    cfg_stats_probe_name : str | None
        Name for the stats probe clip (defaults to CALIBRATE fold).
    cfg_capture_blocks, cfg_capture_layers, cfg_capture_outputs, cfg_cls_token_only
        Forwarded to :func:`gigapath_bipolar_deep_feature_clip`.
    cfg_shard_size : int
        Forwarded to :func:`gigapath_bipolar_deep_feature_clip`.
    url : str | None
        Datablock URL.

    Examples
    --------
    ::

        probe = gigapath_bipolar_deep_feature_affine_logistic_probe(
            'GIGAPATH_DEEP_CPTAC_602020_TRAIN',
            cfg_layer='output',
            cfg_cls_token_only=True,
        )
        probe.build_tree()
    """
    clip = gigapath_bipolar_deep_feature_clip(
        name,
        cfg_layer=cfg_layer,
        cfg_bag_aggregation_threshold=cfg_bag_aggregation_threshold,
        cfg_ternarize_tiles=cfg_ternarize_tiles,
        cfg_stats_probe_name=cfg_stats_probe_name,
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
            layer=cfg_layer,
            annotation_key=cfg_annotation_key,
            fit_intercept=cfg_fit_intercept,
            evaluation_fraction=cfg_evaluation_fraction,
            normalize=cfg_normalize,
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Zipped tile + feature datasets
# ═══════════════════════════════════════════════════════════════════════

"""
git commit -am 'deep: gigapath_tile_deep_feature_clip_dataloader_samples: TEST' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_tile_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    batch_size=4, \
    n=8, \
)"
"""
def gigapath_tile_deep_feature_clip_dataloader_samples(
    name: str,
    *,
    url: str | None = None,
    cfg_capture_blocks: list | None = None,
    cfg_capture_layers: list | None = None,
    cfg_cls_token_only: bool = False,
    cfg_shard_size: int = 1024,
    shuffle: bool = False,
    skip_invalid_bags: bool = True,
    batch_size: int | None = None,
    n: int | None = None,
    return_last: bool = True,
    **dataloader_kwargs,
):
    """Return a zipped dataset of tiles + deep features from a named clip.

    Creates a :class:`ZipStreamingDataset` pairing the
    :class:`DeepFeatureClip` dataset with its source tile clip's
    dataset.  Each sample dict contains both ``features_*`` columns
    and ``tile``, ``bag_name``, ``annotations`` from the tile shards.

    If *n* is given, iterate *n* samples via a DataLoader and return
    the last batch (for smoke-testing).  Otherwise return the dataset.

    Parameters
    ----------
    name : str
        Forwarded to :func:`gigapath_deep_feature_clip`.
    shuffle, skip_invalid_bags, batch_size
        Forwarded to both ``.dataset()`` calls.
    n : int | None
        If given, iterate this many samples and return the last batch.
    """
    clip = gigapath_deep_feature_clip(
        name,
        url=url,
        cfg_capture_blocks=cfg_capture_blocks,
        cfg_capture_layers=cfg_capture_layers,
        cfg_cls_token_only=cfg_cls_token_only,
        cfg_shard_size=cfg_shard_size,
    )
    assert clip.valid(topic=None), (
        f"DeepFeatureClip is not valid (hash={clip.hash[:8]}). "
        f"Build it first with gigapath_deep_feature_clip(...).build()\n"
        f"  validpaths = {clip.validpaths()}\n"
        f"  anchorkeypath = {clip.anchorkeypath}"
    )
    sd_kwargs = dict(shuffle=shuffle, skip_invalid_bags=skip_invalid_bags)
    if batch_size is not None:
        sd_kwargs['batch_size'] = batch_size
    feature_ds = clip.dataset(**sd_kwargs)
    tile_ds = clip.cfg.tilebagclip.dataset(**sd_kwargs)
    ds = ZipStreamingDataset(feature_ds, tile_ds)

    if n is None:
        return ds

    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size or 1, collate_fn=sanitize_collate, **dataloader_kwargs,
    )
    bs = batch_size or 1
    last = None
    for i, batch in enumerate(tqdm.tqdm(loader, total=(n + bs - 1) // bs)):
        last = batch
        if (i + 1) * bs >= n:
            break
    if return_last:
        return last


"""
git commit -am 'deep: gigapath_tile_bipolar_deep_feature_clip_dataloader_samples: TEST' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_tile_bipolar_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    batch_size=4, \
    n=8, \
)"
git commit -am 'deep: gigapath_tile_bipolar_deep_feature_clip_dataloader_samples: TEST' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_tile_bipolar_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    batch_size=4, \
    n=8, \
)"
git commit -am 'deep: gigapath_tile_bipolar_deep_feature_clip_dataloader_samples: TEST' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_tile_bipolar_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_ternarize_tiles=True, \
    cfg_shard_size=64, \
    batch_size=4, \
    n=8, \
)"
"""
def gigapath_tile_bipolar_deep_feature_clip_dataloader_samples(
    name: str,
    *,
    cfg_layer: str = 'output',
    cfg_bag_aggregation_threshold: float = 0.5,
    cfg_ternarize_tiles: bool = False,
    cfg_stats_probe_name: str | None = None,
    cfg_capture_blocks: list | None = None,
    cfg_capture_layers: list | None = None,
    cfg_capture_outputs: bool = True,
    cfg_cls_token_only: bool = False,
    cfg_shard_size: int = 64,
    url: str | None = None,
    shuffle: bool = False,
    skip_invalid_bags: bool = True,
    batch_size: int | None = None,
    n: int | None = None,
    return_last: bool = True,
    **dataloader_kwargs,
):
    """Return a zipped dataset of tiles + bipolar features from a named clip.

    Creates a :class:`ZipStreamingDataset` pairing the
    :class:`BipolarDeepFeatureClip` dataset with the source tile clip's
    dataset.  Each sample dict contains ``bipolar_features_{layer}``,
    ``bag_bipolar_features_{layer}``, ``tile``, ``bag_name``, and
    ``annotations``.

    If *n* is given, iterate *n* samples via a DataLoader and return
    the last batch (for smoke-testing).  Otherwise return the dataset.

    Parameters
    ----------
    name : str
        Forwarded to :func:`gigapath_bipolar_deep_feature_clip`.
    cfg_layer, cfg_bag_aggregation_threshold, cfg_ternarize_tiles, cfg_stats_probe_name
        Forwarded to :func:`gigapath_bipolar_deep_feature_clip`.
    shuffle, skip_invalid_bags, batch_size
        Forwarded to both ``.dataset()`` calls.
    n : int | None
        If given, iterate this many samples and return the last batch.
    """


    bipolar_clip = gigapath_bipolar_deep_feature_clip(
        name,
        cfg_layer=cfg_layer,
        cfg_bag_aggregation_threshold=cfg_bag_aggregation_threshold,
        cfg_ternarize_tiles=cfg_ternarize_tiles,
        cfg_stats_probe_name=cfg_stats_probe_name,
        cfg_capture_blocks=cfg_capture_blocks,
        cfg_capture_layers=cfg_capture_layers,
        cfg_capture_outputs=cfg_capture_outputs,
        cfg_cls_token_only=cfg_cls_token_only,
        cfg_shard_size=cfg_shard_size,
        url=url,
    )
    assert bipolar_clip.valid(topic=None), (
        f"BipolarDeepFeatureClip is not valid (hash={bipolar_clip.hash[:8]}). "
        f"Build it first with gigapath_bipolar_deep_feature_clip(...).build_tree()\n"
        f"  validpaths = {bipolar_clip.validpaths()}\n"
        f"  anchorkeypath = {bipolar_clip.anchorkeypath}"
    )
    sd_kwargs = dict(shuffle=shuffle, skip_invalid_bags=skip_invalid_bags)
    if batch_size is not None:
        sd_kwargs['batch_size'] = batch_size
    bipolar_ds = bipolar_clip.dataset(**sd_kwargs)
    tile_ds = bipolar_clip.cfg.clip.cfg.tilebagclip.dataset(**sd_kwargs)
    ds = ZipStreamingDataset(bipolar_ds, tile_ds)

    if n is None:
        return ds

    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size or 1, collate_fn=sanitize_collate, **dataloader_kwargs,
    )
    bs = batch_size or 1
    last = None
    for i, batch in enumerate(tqdm.tqdm(loader, total=(n + bs - 1) // bs)):
        last = batch
        if (i + 1) * bs >= n:
            break
    if return_last:
        return last


# ═══════════════════════════════════════════════════════════════════════
#  Validate tile ↔ feature consistency
# ═══════════════════════════════════════════════════════════════════════

"""
### CPTAC 60/20/20 — validate deep features (100 samples)
git commit -am 'deep: validate tile features' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_validate_tile_feature_zip_alignment( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    n_samples=100, \
)"
### CPTAC 60/20/20 — validate using TFRecord tiles (bypass MDS round-trip)
git commit -am 'deep: validate tile features' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_validate_tile_feature_zip_alignment( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    n_samples=100, \
    tile_source='tfrecord', \
)"
"""
def gigapath_validate_tile_feature_zip_alignment(
    name: str,
    *,
    url: str | None = None,
    cfg_capture_blocks: list | None = None,
    cfg_capture_layers: list | None = None,
    cfg_capture_outputs: bool = True,
    cfg_cls_token_only: bool = False,
    cfg_shard_size: int = 1024,
    n_samples: int | None = None,
    tile_key: str = 'tile',
    tile_source: str = 'dataset',
    n_repeats: int = 5,
    atol: float = 1e-4,
    rtol: float = 1e-4,
    device: str = 'cuda',
    shuffle: bool = False,
    shuffle_seed: int = 42,
    shuffle_block_size: int = 256,
    skip_invalid_bags: bool = True,
    batch_size: int = 1,
):
    """Validate stored deep features against fresh backbone evaluation.

    Builds a :class:`ZipStreamingDataset` pairing the
    :class:`DeepFeatureClip` dataset with its source tile clip, then
    uses :class:`ValidateTileFeatureZipStreamingDataset` to re-run the
    GigaPath backbone on each tile and compare.

    Parameters
    ----------
    name : str
        Forwarded to :func:`gigapath_deep_feature_clip`.
    n_samples : int | None
        Maximum number of samples to validate.  ``None`` validates all.
    tile_source : str
        Where to read tiles for recomputation.

        * ``'dataset'`` (default) — from the MDS tile
          :class:`StreamingDataset` (round-tripped through MDS).
        * ``'tfrecord'`` — directly from the source TFRecords, the
          same path the build uses.  Useful for isolating whether the
          MDS tile round-trip introduces numerical discrepancies.
    atol, rtol : float
        Tolerances for ``np.allclose``.
    device : str
        Device for the evaluator (e.g. ``'cuda'``, ``'cuda:1'``).
    shuffle : bool
        Whether to shuffle within the streaming datasets.  When
        ``True``, ``shuffle_seed`` and ``shuffle_block_size`` are
        applied to both datasets identically.
    shuffle_seed : int
        Seed for deterministic shuffle order.
    shuffle_block_size : int
        Block size for the shuffle algorithm.
    batch_size : int
        Passed to both datasets for deterministic resumption.

    Returns
    -------
    ValidateTileFeatureZipStreamingDataset.ValidationResult
    """
    clip = gigapath_deep_feature_clip(
        name,
        url=url,
        cfg_capture_blocks=cfg_capture_blocks,
        cfg_capture_layers=cfg_capture_layers,
        cfg_capture_outputs=cfg_capture_outputs,
        cfg_cls_token_only=cfg_cls_token_only,
        cfg_shard_size=cfg_shard_size,
    )
    assert clip.valid(topic=None), (
        f"DeepFeatureClip is not valid (hash={clip.hash[:8]}). "
        f"Build it first with gigapath_deep_feature_clip(...).build_tree()\n"
        f"  validpaths = {clip.validpaths()}\n"
        f"  anchorkeypath = {clip.anchorkeypath}"
    )

    sd_kwargs = dict(
        shuffle=shuffle,
        skip_invalid_bags=skip_invalid_bags,
        batch_size=batch_size,
    )
    if shuffle:
        sd_kwargs['shuffle_seed'] = shuffle_seed
        sd_kwargs['shuffle_block_size'] = shuffle_block_size

    ds = clip.tile_feature_dataset(**sd_kwargs)

    evaluator = clip.cfg.evaluator_factory.evaluator(device=device)

    validator = ValidateTileFeatureZipStreamingDataset(
        ds, evaluator,
        n_samples=n_samples,
        tile_key=tile_key,
        tile_source=tile_source,
        tilebagclip=clip.cfg.tilebagclip,
        n_repeats=n_repeats,
        gpu_batch_size=clip.gpu_batch_size,
        atol=atol,
        rtol=rtol,
    )
    return validator.validate()

"""
### Diagnostic: tile-feature zip index consistency
git commit -am 'deep: probe zip alignment' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_probe_tile_feature_zip_alignment( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    n_samples=20, \
)"
"""
def gigapath_probe_tile_feature_zip_alignment(
    name: str,
    *,
    url: str | None = None,
    cfg_cls_token_only: bool = False,
    cfg_shard_size: int = 1024,
    n_samples: int = 20,
):
    """Check that ``clip.dataset()[i]`` and ``clip.tile_feature_dataset()[i]``
    return identical stored features for each flat index *i*.

    The plain dataset streams only feature shards; the tile-feature
    dataset zips feature and tile streams via
    :class:`ZipStreamingDataset`.  If the two disagree at any index,
    the zip is reordering or misaligning samples.
    """
    clip = gigapath_deep_feature_clip(
        name, url=url,
        cfg_cls_token_only=cfg_cls_token_only,
        cfg_shard_size=cfg_shard_size,
    )

    ds_plain = clip.dataset(shuffle=False, batch_size=1)
    ds_zip = clip.tile_feature_dataset(shuffle=False, batch_size=1)

    clip.log.info(f"ds_plain length: {len(ds_plain)}")
    clip.log.info(f"ds_zip   length: {len(ds_zip)}")

    n_match = 0
    n_differ = 0
    max_diff = 0.0
    for i in range(min(n_samples, len(ds_plain))):
        feat_plain = ds_plain[i]['features_output']
        if isinstance(feat_plain, torch.Tensor):
            feat_plain = feat_plain.numpy()

        sample_zip = ds_zip[i]
        feat_zip = sample_zip.get('features_output')
        if feat_zip is None:
            clip.log.warning(f"[{i}] features_output MISSING in zip. keys={list(sample_zip.keys())}")
            n_differ += 1
            continue
        if isinstance(feat_zip, torch.Tensor):
            feat_zip = feat_zip.numpy()

        diff = np.abs(feat_plain - feat_zip).max()
        max_diff = max(max_diff, diff)
        zip_bag = sample_zip.get('bag_name', '<missing>')

        if diff == 0:
            n_match += 1
            tag = "MATCH"
        else:
            n_differ += 1
            tag = f"DIFFER max={diff:.2e}"

        clip.log.info(
            f"[{i:3d}] {tag}  zip_bag_name={zip_bag}  "
            f"plain[:3]=[{feat_plain.flat[0]:.6f}, {feat_plain.flat[1]:.6f}, {feat_plain.flat[2]:.6f}]"
        )

    status = 'PASSED' if n_differ == 0 else 'FAILED'
    clip.log.info(f"{status}: {n_match} match, {n_differ} differ, max_diff={max_diff:.2e}")
    return dict(n_match=n_match, n_differ=n_differ, max_diff=max_diff)


"""
### Diagnostic: batch-size invariance of DeepFeatureClip
git commit -am 'deep: verify batchsize' > /dev/null || true; dbx.pprint "\
autopath.deep.pipelines.gigapath_probe_deep_feature_clip_batchsize_effect( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    cfg_cls_token_only=True, \
    bag_index = 0, tile_indices = list(range(100)), \
    cfg_shard_size=64, \
)"
"""
def gigapath_probe_deep_feature_clip_batchsize_effect(
    name: str,
    *,
    url: str | None = None,
    cfg_capture_blocks: list | None = None,
    cfg_capture_layers: list | None = None,
    cfg_capture_outputs: bool = True,
    cfg_cls_token_only: bool = False,
    cfg_shard_size: int = 1024,
    bag_index: int = 0,
    tile_index: int | list[int] = 0,
    n_repeats: int = 3,
    device: str = 'cuda',
):
    """Verify that evaluator output is batch-size-invariant.

    Thin pipeline wrapper around
    :meth:`DeepFeatureClip.verify_batchsize_invariance`.

    Parameters
    ----------
    tile_index : int | list[int]
        A single tile index or a list of tile indices to probe.
        When a list is given, runs the test for each tile, prints
        per-tile line items, and returns a summary with averages.
    """
    clip = gigapath_deep_feature_clip(
        name,
        url=url,
        cfg_capture_blocks=cfg_capture_blocks,
        cfg_capture_layers=cfg_capture_layers,
        cfg_capture_outputs=cfg_capture_outputs,
        cfg_cls_token_only=cfg_cls_token_only,
        cfg_shard_size=cfg_shard_size,
    )

    if isinstance(tile_index, int):
        return clip.verify_batchsize_invariance(
            bag_index=bag_index,
            tile_index=tile_index,
            n_repeats=n_repeats,
            device=device,
        )

    # Multiple tiles
    results = []
    for ti in tile_index:
        r = clip.verify_batchsize_invariance(
            bag_index=bag_index,
            tile_index=ti,
            n_repeats=n_repeats,
            device=device,
        )
        results.append(r)
        clip.log.info(
            f"  tile {ti:3d}: stored_vs_bs1={r['stored_vs_bs1']:.2e}  "
            f"stored_vs_bsN={r['stored_vs_bsN']:.2e}  "
            f"bs1_vs_bsN={r['bs1_vs_bsN']:.2e}"
        )

    avg_stored_bs1 = np.mean([r['stored_vs_bs1'] for r in results])
    avg_stored_bsN = np.mean([r['stored_vs_bsN'] for r in results])
    avg_bs1_bsN = np.mean([r['bs1_vs_bsN'] for r in results])
    max_stored_bs1 = max(r['stored_vs_bs1'] for r in results)
    max_stored_bsN = max(r['stored_vs_bsN'] for r in results)
    max_bs1_bsN = max(r['bs1_vs_bsN'] for r in results)

    clip.log.info(
        f"Summary ({len(results)} tiles): "
        f"stored_vs_bs1: avg={avg_stored_bs1:.2e} max={max_stored_bs1:.2e}  "
        f"stored_vs_bsN: avg={avg_stored_bsN:.2e} max={max_stored_bsN:.2e}  "
        f"bs1_vs_bsN: avg={avg_bs1_bsN:.2e} max={max_bs1_bsN:.2e}"
    )
    return dict(
        results=results,
        avg_stored_vs_bs1=avg_stored_bs1,
        avg_stored_vs_bsN=avg_stored_bsN,
        avg_bs1_vs_bsN=avg_bs1_bsN,
        max_stored_vs_bs1=max_stored_bs1,
        max_stored_vs_bsN=max_stored_bsN,
        max_bs1_vs_bsN=max_bs1_bsN,
    )

