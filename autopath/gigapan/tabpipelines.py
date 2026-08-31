"""Pipeline entrypoints for GigaPath deep feature extraction using tabular dbx blocks (tabfeatures.py).

Provides declarative constructors for:
- :class:`~autopath.gigapan.tabfeatures.DeepFeatureBag`
- :class:`~autopath.gigapan.tabfeatures.DeepFeatureClip`
- :class:`~autopath.gigapan.tabfeatures.BipolarDeepFeatureClip`

Modelled on :mod:`autopath.gigapan.pipelines` using tabular blocks from :mod:`autopath.gigapan.tabfeatures`.
"""

from __future__ import annotations

import tqdm
import torch

import dbx
from dbx import Logger

from autopath.gigapath.backbone import (
    GigapathDeepBackboneEvaluatorFactory,
)
from autopath.gigapan.tabfeatures import (
    DeepFeatureBag,
    DeepFeatureClip,
    BipolarDeepFeatureClip,
    tile_collator,
)
from autopath.gigapan.tabprobes import (
    DeepFeatureStatsProbe,
    DeepFeatureAffineLogisticProbe,
)
from autopath.pancan.tabpipelines import (
    pancan_tile_bag,
    pancan_tile_clip,
    pancan_tile_fold,
)
from autopath.autobits import sanitize_collate, ZipStreamingDataset


log = Logger()


# ═══════════════════════════════════════════════════════════════════════
#  Evaluator & Factory
# ═══════════════════════════════════════════════════════════════════════

def gigapath_deep_backbone_evaluator(
    *,
    var_capture_blocks: list | None = None,
    var_capture_layers: list | None = None,
    var_capture_final: bool = True,
    var_cls_token_only: bool = False,
    device: str = "cuda",
):
    factory = gigapath_deep_backbone_evaluator_factory(
        var_capture_blocks=var_capture_blocks,
        var_capture_layers=var_capture_layers,
        var_capture_final=var_capture_final,
        var_cls_token_only=var_cls_token_only,
    )
    return factory.evaluator(device=device)



def gigapath_deep_backbone_evaluator_factory(
    var_capture_blocks: list | None = None,
    var_capture_layers: list | None = None,
    var_capture_final: bool = True,
    var_cls_token_only: bool = False,
    *,
    url: str | None = None,
) -> GigapathDeepBackboneEvaluatorFactory:
    return GigapathDeepBackboneEvaluatorFactory(
        url=url,
        spec=dict(
            capture_blocks=var_capture_blocks or [],
            capture_layers=var_capture_layers or [],
            capture_final=var_capture_final,
            cls_token_only=var_cls_token_only,
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Deep Feature Bag & Clip
# ═══════════════════════════════════════════════════════════════════════

"""
### FINAL-ONLY
## CLS-only
dbx.pprint \
"autopath.gigapan.tabpipelines.gigapath_deep_feature_bag('GIGAPATH_DEEP_CPTAC_SAMPLE', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    device_batch_size=1024 \
).build().valid()"
### ALL DEEP layers
## CLS-only
dbx.pprint \
"autopath.gigapan.tabpipelines.gigapath_deep_feature_bag('GIGAPATH_DEEP_CPTAC_SAMPLE', \
    var_cls_token_only=True, \
    var_capture_blocks='all', \
    var_shard_size=128, \
    device_batch_size=128 \
).build().valid()"
## CLS + PATCH tokens
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_bag('GIGAPATH_DEEP_CPTAC_SAMPLE', \
    var_cls_token_only=False, \
    var_capture_blocks='all', \
    var_shard_size=64, \
    device_batch_size=1024 \
    ).build().valid()\
    "
### SOME DEEP layers
## CLS-only
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_bag('GIGAPATH_DEEP_CPTAC_SAMPLE', \
    var_cls_token_only=True, \
    var_capture_blocks=[0, 7, 14, 21, 27, 33, 39], \
    var_shard_size=128, \
    device_batch_size=1024 \
    ).build().valid()\
    "
"""
def gigapath_deep_feature_bag(
    name: str,
    *,
    var_capture_blocks: list | None = None,
    var_capture_layers: list | None = None,
    var_capture_final: bool = True,
    var_cls_token_only: bool = False,
    var_shard_size: int = 1024,
    device_batch_size: int = 64,
    url: str | None = None,
    streaming: bool = False,
    dataloader_kwargs: dict | None = None,
) -> DeepFeatureBag:
    factory = gigapath_deep_backbone_evaluator_factory(
        var_capture_blocks, var_capture_layers, var_capture_final, var_cls_token_only, url=url,
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
            collator=dbx.quote(tile_collator),
            shard_size=var_shard_size,
        ),
        device_batch_size=device_batch_size,
        streaming=streaming,
        dataloader_kwargs=dataloader_kwargs,
    )


"""
### CPTAC: FINAL-ONLY|CLS-ONLY
## 1 device
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_devices=1, device_batch_size=512, \
).build_tree()"
## 1 device, streaming
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_devices=1, device_batch_size=256, streaming=True, dataloader_kwargs=dict(batch_size=512, num_workers=4, prefetch_factor=2, pin_memory=True), \
).build_tree()"
## 1 device x4
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    devices=['cuda', 'cuda', 'cuda', 'cuda'], device_batch_size=128, parallelization='multiprocessing', \
).build_tree()"
## 1 device x2, streaming
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    devices=['cuda', 'cuda'], device_batch_size=128, parallelization='multithreading', streaming=True, dataloader_kwargs=dict(batch_size=128, num_workers=4, prefetch_factor=2, persistent_workers=True), \
).build_tree()"
## 3 devices, streaming
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_devices=3, device_batch_size=32, parallelization='multiprocessing', work_stealing=True, streaming=True, dataloader_kwargs=dict(batch_size=32, num_workers=0),\
).build_tree()"
## 3 devices
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_devices=3, device_batch_size=128, parallelization='multiprocessing', work_stealing=True,\
).build_tree()"
### CPTAC_602020_TRAIN|CALIBRATE|TEST: FINAL-ONLY|CLS-ONLY
## 1 device
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_devices=1, device_batch_size=512, \
).build_tree()"
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip(\
    'GIGAPATH_DEEP_CPTAC_602020_CALIBRATE', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_devices=1, device_batch_size=512, \
).build_tree()"
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_devices=1, device_batch_size=512, \
).build_tree()"
## 1 device, streaming
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_devices=1, device_batch_size=256, streaming=True, dataloader_kwargs=dict(batch_size=512, num_workers=2, prefetch_factor=2, pin_memory=True), \
).build_tree()"
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_CALIBRATE', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_devices=1, device_batch_size=256, streaming=True, dataloader_kwargs=dict(batch_size=512, num_workers=2, prefetch_factor=2, pin_memory=True), \
).build_tree()"
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_devices=1, device_batch_size=256, streaming=True, dataloader_kwargs=dict(batch_size=512, num_workers=2, prefetch_factor=2, pin_memory=True), \
).build_tree()"
## 3 devices, streaming
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_CALIBRATE', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_devices=3, device_batch_size=32, parallelization='multiprocessing', work_stealing=True, streaming=True, dataloader_kwargs=dict(batch_size=32, num_workers=0),\
).build_tree()"
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_devices=3, device_batch_size=32, parallelization='multiprocessing', work_stealing=True, streaming=True, dataloader_kwargs=dict(batch_size=32, num_workers=0),\
).build_tree()"
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_devices=3, device_batch_size=32, parallelization='multiprocessing', work_stealing=True, streaming=True, dataloader_kwargs=dict(batch_size=32, num_workers=0),\
).build_tree()"
## 3 devices
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_devices=3, device_batch_size=32, parallelization='multiprocessing', work_stealing=True,).build_tree()"
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_CALIBRATE', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_devices=3, device_batch_size=32, parallelization='multiprocessing', work_stealing=True,).build_tree()"
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_devices=3, device_batch_size=32, parallelization='multiprocessing', work_stealing=True,).build_tree()"

"""
def gigapath_deep_feature_clip(
    name: str = None,
    *,
    var_capture_blocks: list | None = None,
    var_capture_layers: list | None = None,
    var_capture_final: bool = True,
    var_cls_token_only: bool = False,
    var_shard_size: int = 1024,
    url: str | None = None,
    n_devices: int | None = None,
    devices: list | None = None,
    device_batch_size: int = 64,
    parallelization: str | None = None,
    work_stealing: bool = False,
    streaming: bool = False,
    dataloader_kwargs: dict | None = None,
    filter_built_tabs: bool = False,
) -> DeepFeatureClip:
    if name is None:
        return DeepFeatureClip
    if devices is not None:
        pass
    elif n_devices is not None:
        devices = [f"cuda:{i}" for i in range(n_devices)]
    else:
        devices = ["cuda"]
    if parallelization is None and len(devices) > 1:
        parallelization = "multiprocessing"
    factory = gigapath_deep_backbone_evaluator_factory(
        var_capture_blocks, var_capture_layers, var_capture_final, var_cls_token_only, url=url,
    )
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
            datapoint_table=tilebagclip,
            evaluator_factory=dbx.quote(factory),
            collator=dbx.quote(tile_collator),
            shard_size=var_shard_size,
        ),
        device_batch_size=device_batch_size,
        devices=devices,
        n_workers=len(devices),
        parallelization=parallelization,
        work_stealing=work_stealing,
        streaming=streaming,
        dataloader_kwargs=dataloader_kwargs,
        filter_built_tabs=filter_built_tabs,
    )


"""
dbx.pprint "autopath.gigapan.tabpipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC', \
    var_cls_token_only=True, \
    var_shard_size=64,\
    n=8,\
)"
dbx.pprint "autopath.gigapan.tabpipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_200179_TRAIN', \
    var_cls_token_only=True, \
    n=8,\
)"
dbx.pprint "autopath.gigapan.tabpipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_404020_TRAIN', \
    var_capture_blocks=[0,19,38], \
    var_cls_token_only=True, \
    batch_size=8, \
    shuffle=True, \
    n=16,\
)"
dbx.pprint "autopath.gigapan.tabpipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC', \
    var_cls_token_only=True, \
    n=4,\
)"
dbx.pprint "autopath.gigapan.tabpipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n=4,\
)"
dbx.pprint "autopath.gigapan.tabpipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_602020_CALIBRATE', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n=4,\
)"
dbx.pprint "autopath.gigapan.tabpipelines.gigapath_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n=4,\
)"
"""
def gigapath_deep_feature_clip_dataloader_samples(
    name: str,
    n: int,
    *,
    url: str | None = None,
    var_capture_blocks: list | None = None,
    var_capture_layers: list | None = None,
    var_cls_token_only: bool = False,
    var_shard_size: int = 1024,
    batch_size: int = 4,
    shuffle: bool = False,
    return_last: bool = True,
    filter_built_tabs: bool = False,
    **dataloader_kwargs,
):
    clip = gigapath_deep_feature_clip(
        name,
        url=url,
        var_capture_blocks=var_capture_blocks,
        var_capture_layers=var_capture_layers,
        var_cls_token_only=var_cls_token_only,
        var_shard_size=var_shard_size,
        filter_built_tabs=filter_built_tabs,
    )
    assert clip.valid(), (
        f"DeepFeatureClip is not valid (hash={clip.hash[:8]}). "
        f"Build it first with gigapath_deep_feature_clip(...).build()\n"
        f"  validpaths = {clip.validpaths()}\n"
        f"  anchorkeypath = {clip.anchorkeypath}"
    )
    ds = clip.dataset(batch_size=batch_size, mode='map', shuffle=shuffle)
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
#  Probes & Bipolar Clip
# ═══════════════════════════════════════════════════════════════════════

"""
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_stats_probe( \
    'GIGAPATH_DEEP_CPTAC_9802_TEST', \
    var_layer='final', \
    var_cls_token_only=True, \
    var_shard_size=64, \
).build()"
"""
def gigapath_deep_feature_stats_probe(
    name: str,
    *,
    var_layer: str = 'final',
    var_normalize: str | None = None,
    var_capture_blocks: list | None = None,
    var_capture_layers: list | None = None,
    var_capture_final: bool = True,
    var_cls_token_only: bool = False,
    var_shard_size: int = 64,
    url: str | None = None,
    filter_built_tabs: bool = False,
) -> DeepFeatureStatsProbe:
    clip = gigapath_deep_feature_clip(
        name,
        var_capture_blocks=var_capture_blocks,
        var_capture_layers=var_capture_layers,
        var_capture_final=var_capture_final,
        var_cls_token_only=var_cls_token_only,
        var_shard_size=var_shard_size,
        url=url,
        filter_built_tabs=filter_built_tabs,
    )
    collator = getattr(clip.var, 'collator', None) or tile_collator()
    collator_spec = collator.quote() if hasattr(collator, 'quote') else dbx.quote(collator)
    return DeepFeatureStatsProbe(
        url=url,
        spec=dict(
            feature_table=dbx.quote(clip),
            collator=collator_spec,
            normalization=var_normalize,
        ),
    )


"""
### CPTAC 9802 — no normalization
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_9802_TEST', \
    var_layer='final', \
    var_cls_token_only=True, \
    var_shard_size=64, \
).build()"
### CPTAC 9802 — L2-normalised features
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_deep_feature_affine_logistic_probe( \
    'GIGAPATH_DEEP_CPTAC_9802_TEST', \
    var_layer='final', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    var_normalize='l2', \
).build()"
"""
def gigapath_deep_feature_affine_logistic_probe(
    name: str,
    *,
    var_layer: str = 'final',
    var_fit_intercept: bool = True,
    var_evaluation_fraction: float = 0.8,
    var_normalize: str | None = None,
    var_capture_blocks: list | None = None,
    var_capture_layers: list | None = None,
    var_capture_final: bool = True,
    var_cls_token_only: bool = False,
    var_shard_size: int = 64,
    url: str | None = None,
    filter_built_tabs: bool = False,
) -> DeepFeatureAffineLogisticProbe:
    """Create a `DeepFeatureAffineLogisticProbe` on a named clip.

    Fits a logistic regression classifier on sample-level features
    for the specified layer, persisting the fitted model's
    `coef_`, `intercept_`, and `classes_` arrays.

    Parameters
    ----------
    name : str
        Named configuration — same values accepted by `gigapath_deep_feature_clip`
        (e.g. `"GIGAPATH_DEEP_CPTAC_602020_TRAIN"`).
    var_layer : str
        Which capture key to probe (e.g. `"final"`, `"B_0_norm1"`).
    var_fit_intercept : bool
        Whether to fit an intercept term in the logistic regression.
    var_evaluation_fraction : float
        Fraction of samples used for training (rest for evaluation).
    var_normalize : str | None
        Feature normalization mode: `None` (raw), `'l2'`, `'corner-l1'`,
        `'corner-l2'`, or `'corner-linfty'`.
    var_capture_blocks, var_capture_layers, var_capture_final, var_cls_token_only
        Forwarded to `gigapath_deep_feature_clip` to identify the underlying clip.
    var_shard_size : int
        Forwarded to `gigapath_deep_feature_clip`.
    url : str | None
        Datablock URL.
    filter_built_tabs : bool
        Whether to filter built tabs.

    Examples
    --------
    Build on the CPTAC 60/20/20 train fold with no normalization:

        probe = gigapath_deep_feature_affine_logistic_probe(
            'GIGAPATH_DEEP_CPTAC_602020_TRAIN',
            var_layer='final',
            var_cls_token_only=True,
            var_shard_size=64,
        )
        probe.build()

    With L2 normalization:

        probe = gigapath_deep_feature_affine_logistic_probe(
            'GIGAPATH_DEEP_CPTAC_602020_TRAIN',
            var_layer='final',
            var_cls_token_only=True,
            var_shard_size=64,
            var_normalize='l2',
        )
        probe.build()
    """
    clip = gigapath_deep_feature_clip(
        name,
        var_capture_blocks=var_capture_blocks,
        var_capture_layers=var_capture_layers,
        var_capture_final=var_capture_final,
        var_cls_token_only=var_cls_token_only,
        var_shard_size=var_shard_size,
        url=url,
        filter_built_tabs=filter_built_tabs,
    )
    collator = getattr(clip.var, 'collator', None) or tile_collator()
    collator_spec = collator.quote() if hasattr(collator, 'quote') else dbx.quote(collator)
    return DeepFeatureAffineLogisticProbe(
        url=url,
        spec=dict(
            feature_table=dbx.quote(clip),
            collator=collator_spec,
            fit_intercept=var_fit_intercept,
            evaluation_fraction=var_evaluation_fraction,
            normalization=var_normalize,
        ),
    )



"""
### CPTAC 60/20/20 — bipolar features (TRAIN, stats from CALIBRATE)
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_bipolar_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    var_layer='final', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_workers=8,\
).build()"
### CPTAC 60/20/20 — bipolar features (TEST, stats from CALIBRATE)
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_bipolar_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    var_layer='final', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    n_workers=8,\
).build()"
### CPTAC 60/20/20 — bipolar features + ternarization (TRAIN, stats from CALIBRATE)
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_bipolar_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    var_layer='final', \
    var_cls_token_only=True, \
    var_ternarize_tiles=True, \
    var_shard_size=64, \
    n_workers=8,\
).build()"
### CPTAC 60/20/20 — bipolar features + ternarization (TEST, stats from CALIBRATE)
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_bipolar_deep_feature_clip( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    var_layer='final', \
    var_cls_token_only=True, \
    var_ternarize_tiles=True, \
    var_shard_size=64, \
    n_workers=8,\
).build()"
"""
def gigapath_bipolar_deep_feature_clip(
    name: str,
    *,
    var_layer: str = 'final',
    var_bag_aggregation_threshold: float = 0.5,
    var_ternarize_tiles: bool = False,
    var_stats_probe_name: str | None = None,
    var_capture_blocks: list | None = None,
    var_capture_layers: list | None = None,
    var_capture_final: bool = True,
    var_cls_token_only: bool = False,
    var_shard_size: int = 64,
    url: str | None = None,
    n_workers: int = 1,
    parallelization: str | None = None,
    streaming: bool = False,
    dataloader_kwargs: dict | None = None,
    filter_built_tabs: bool = False,
) -> BipolarDeepFeatureClip:
    if parallelization is None and n_workers > 1:
        parallelization = 'multiprocessing'

    clip = gigapath_deep_feature_clip(
        name,
        var_capture_blocks=var_capture_blocks,
        var_capture_layers=var_capture_layers,
        var_capture_final=var_capture_final,
        var_cls_token_only=var_cls_token_only,
        var_shard_size=var_shard_size,
        url=url,
        streaming=streaming,
        dataloader_kwargs=dataloader_kwargs,
        filter_built_tabs=filter_built_tabs,
    )

    if var_stats_probe_name is None:
        parts = name.rsplit('_', 1)
        if len(parts) == 2 and parts[1] in ('TRAIN', 'TEST', 'CALIBRATE'):
            var_stats_probe_name = parts[0] + '_CALIBRATE'
        else:
            var_stats_probe_name = name

    stats_probe = gigapath_deep_feature_stats_probe(
        var_stats_probe_name,
        var_layer=var_layer,
        var_capture_blocks=var_capture_blocks,
        var_capture_layers=var_capture_layers,
        var_capture_final=var_capture_final,
        var_cls_token_only=var_cls_token_only,
        var_shard_size=var_shard_size,
        url=url,
        filter_built_tabs=filter_built_tabs,
    )
    return BipolarDeepFeatureClip(
        url=url,
        spec=dict(
            featuretable=dbx.quote(clip),
            layer=var_layer,
            threshold=var_bag_aggregation_threshold,
            ternarize=var_ternarize_tiles,
        ),
        n_workers=n_workers,
        parallelization=parallelization,
        streaming=streaming,
        dataloader_kwargs=dataloader_kwargs,
        filter_built_tabs=filter_built_tabs,
    )


"""
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_tile_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    batch_size=4, \
    n=8, \
)"
"""
def gigapath_tile_deep_feature_clip_dataloader_samples(
    name: str,
    *,
    url: str | None = None,
    var_capture_blocks: list | None = None,
    var_capture_layers: list | None = None,
    var_cls_token_only: bool = False,
    var_shard_size: int = 1024,
    shuffle: bool = False,
    skip_invalid_bags: bool = True,
    batch_size: int | None = None,
    n: int | None = None,
    return_last: bool = True,
    **dataloader_kwargs,
):
    clip = gigapath_deep_feature_clip(
        name,
        url=url,
        var_capture_blocks=var_capture_blocks,
        var_capture_layers=var_capture_layers,
        var_cls_token_only=var_cls_token_only,
        var_shard_size=var_shard_size,
    )
    assert clip.valid(), (
        f"DeepFeatureClip is not valid (hash={clip.hash[:8]}). "
        f"Build it first with gigapath_deep_feature_clip(...).build()\n"
        f"  validpaths = {clip.validpaths()}\n"
        f"  anchorkeypath = {clip.anchorkeypath}"
    )
    sd_kwargs = dict(shuffle=shuffle, skip_invalid_bags=skip_invalid_bags)
    if batch_size is not None:
        sd_kwargs['batch_size'] = batch_size
    feature_ds = clip.dataset(**sd_kwargs)
    tilebagclip = getattr(clip.var, 'tilebagclip', None) or getattr(clip.var, 'datapoint_table', None)
    tile_ds = tilebagclip.dataset(**sd_kwargs)
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
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_tile_bipolar_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_602020_TRAIN', \
    var_layer='final', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    batch_size=4, \
    n=8, \
)"
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_tile_bipolar_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    var_layer='final', \
    var_cls_token_only=True, \
    var_shard_size=64, \
    batch_size=4, \
    n=8, \
)"
dbx.pprint "\
autopath.gigapan.tabpipelines.gigapath_tile_bipolar_deep_feature_clip_dataloader_samples( \
    'GIGAPATH_DEEP_CPTAC_602020_TEST', \
    var_layer='final', \
    var_cls_token_only=True, \
    var_ternarize_tiles=True, \
    var_shard_size=64, \
    batch_size=4, \
    n=8, \
)"
"""
def gigapath_tile_bipolar_deep_feature_clip_dataloader_samples(
    name: str,
    *,
    var_layer: str = 'final',
    var_bag_aggregation_threshold: float = 0.5,
    var_ternarize_tiles: bool = False,
    var_stats_probe_name: str | None = None,
    var_capture_blocks: list | None = None,
    var_capture_layers: list | None = None,
    var_capture_final: bool = True,
    var_cls_token_only: bool = False,
    var_shard_size: int = 64,
    url: str | None = None,
    shuffle: bool = False,
    skip_invalid_bags: bool = True,
    batch_size: int | None = None,
    n: int | None = None,
    return_last: bool = True,
    **dataloader_kwargs,
):
    bipolar_clip = gigapath_bipolar_deep_feature_clip(
        name,
        var_layer=var_layer,
        var_bag_aggregation_threshold=var_bag_aggregation_threshold,
        var_ternarize_tiles=var_ternarize_tiles,
        var_stats_probe_name=var_stats_probe_name,
        var_capture_blocks=var_capture_blocks,
        var_capture_layers=var_capture_layers,
        var_capture_final=var_capture_final,
        var_cls_token_only=var_cls_token_only,
        var_shard_size=var_shard_size,
        url=url,
    )
    assert bipolar_clip.valid(), (
        f"BipolarDeepFeatureClip is not valid (hash={bipolar_clip.hash[:8]}). "
        f"Build it first with gigapath_bipolar_deep_feature_clip(...).build()\n"
        f"  validpaths = {bipolar_clip.validpaths()}\n"
        f"  anchorkeypath = {bipolar_clip.anchorkeypath}"
    )
    sd_kwargs = dict(shuffle=shuffle, skip_invalid_bags=skip_invalid_bags)
    if batch_size is not None:
        sd_kwargs['batch_size'] = batch_size
    bipolar_ds = bipolar_clip.dataset(**sd_kwargs)
    featuretable = getattr(bipolar_clip.var, 'featuretable', None) or getattr(bipolar_clip.var, 'clip', None)
    tilebagclip = getattr(featuretable.var, 'tilebagclip', None) or getattr(featuretable.var, 'datapoint_table', None)
    tile_ds = tilebagclip.dataset(**sd_kwargs)
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
