"""Pipeline entrypoints for BitPath distillation stills.

Provides declarative constructors for
:class:`~autopath.stills.bitpath.BitPathConvStill`
and friends, wiring together bipolar deep feature clips,
dataloader builders, and Lightning module factories.
"""

from __future__ import annotations

import os

import dbx

from autopath.deep.features import DeepFeatureClip
from autopath.deep.pipelines import gigapath_bipolar_deep_feature_clip
from autopath.deep.probes import DeepFeatureAffineLogisticProbe
from autopath.pancan.pipelines import (
    pancan_tile_clip,
    pancan_tile_fold,
)
from autopath.stills.backbone import (
    BitConvDeepBackboneEvaluator,
    BitConvDeepBackboneEvaluatorFactory,
)
from autopath.stills.bitpath import (
    BitPathDataloaderBuilder,
    BitPathConvLightning,
    BitPathConvStill,
    BitPathConvStillProbe,
)


# Default root directory where TensorBoard discovers training logs.
# Each still writes events into /tmp and symlinks the run dir here so
# TensorBoard can be pointed at this single location for all stills.
DEFAULT_LOGSROOT = os.path.join(os.environ['HOME'], 'autopath', 'tensorboard')


# ═══════════════════════════════════════════════════════════════════════
#  BitPath conv still
# ═══════════════════════════════════════════════════════════════════════

"""
### CPTAC 60/20/20 — BitPath distillation (3 GPUs)
git commit -am 'stills: BitPathConvStill: BUILD' > /dev/null || true; dbx.pprint "\
autopath.stills.pipelines.bitpath_conv_still( \
    'GIGAPATH_DEEP_CPTAC_602020', \
    cfg_layer='output', \
    cfg_cls_token_only=True, \
    cfg_shard_size=64, \
    cfg_max_steps=10000, \
    cfg_max_epochs=100, \
    cfg_ckpt_every_n_steps=1000, \
    n_devices=3, \
).build_tree()"
"""
def bitpath_conv_still(
    name: str,
    *,
    # BitPathConvNet architecture
    cfg_n_blocks: int = 6,
    cfg_hidden_channels: int = 128,
    cfg_output_dim: int = 1536,
    cfg_n_classes: int = 3,
    cfg_activation_bits: int = 8,
    # Training
    cfg_learning_rate: float = 1e-3,
    cfg_weight_decay: float = 0.01,
    cfg_scheduler: str = 'cosine',
    # Logging / validation schedule
    cfg_log_every_n_steps: int = 20,
    cfg_val_every_n_steps: int = 100,
    cfg_limit_val_batches: int = 200,
    # Dataloader
    cfg_batch_size: int = 64,
    cfg_shuffle: bool = True,
    cfg_dataloader_seed: int | None = None,
    cfg_num_workers: int = 4,
    # Bipolar clip params
    cfg_layer: str = 'output',
    cfg_bag_aggregation_threshold: float = 0.5,
    cfg_ternarize_tiles: bool = False,
    cfg_stats_probe_name: str | None = None,
    cfg_capture_blocks: list | None = None,
    cfg_capture_layers: list | None = None,
    cfg_capture_outputs: bool = True,
    cfg_cls_token_only: bool = False,
    cfg_shard_size: int = 64,
    # Training-schedule params — go into BitPathConvStill.spec and AFFECT THE HASH.
    # Always include these explicitly when constructing a still for evaluation
    # so that the hash matches the training still and the right checkpoint is found.
    cfg_max_epochs: int = 1,
    cfg_max_steps: int = 10000,
    cfg_gradient_clip_val: float = 1.0,
    cfg_ckpt_every_n_steps: int | None = None,
    cfg_precision: str | None = None,
    # Build/execution params — do NOT go into spec and do NOT affect the hash.
    n_devices: int = 1,
    devices: list | None = None,
    n_workers: int = 1,
    parallelization: str | None = None,
    url: str | None = None,
    logsroot: str = DEFAULT_LOGSROOT,
) -> 'BitPathConvStill':
    """Create a :class:`BitPathConvStill` for BitNet1.58b distillation.

    Trains a ternary convolutional network to predict bipolar features
    directly from raw tiles, distilling the Gigapath backbone.

    .. note::
        **Hash-affecting parameters**: every argument that ends up in any
        Datablock ``spec=dict(...)`` (directly or transitively) changes the
        content-hash of the still and all its dependants.  When reusing a
        trained still for evaluation you must reproduce **all** hash-affecting
        arguments exactly; otherwise ``still.ckpt()`` will look in a
        different directory and find no checkpoint.

        * **All** ``cfg_*`` parameters affect the hash — including
          ``cfg_max_epochs``, ``cfg_max_steps``, ``cfg_gradient_clip_val``,
          ``cfg_ckpt_every_n_steps``, and ``cfg_precision``.
        * ``n_devices``, ``devices``, ``n_workers``, ``parallelization``,
          ``url``, and ``logsroot`` are **hash-neutral** — they control
          how the build is executed, not what it produces.

    Parameters
    ----------
    name : str
        Base clip name **without** a split suffix, e.g.
        ``"GIGAPATH_DEEP_CPTAC_602020"``.  Any trailing ``_TRAIN``,
        ``_TEST``, or ``_CALIBRATE`` suffix is stripped automatically.
        Training uses ``<name>_TRAIN``; validation uses ``<name>_TEST``.
        **Affects hash** (propagates through train/val clip hashes).

    Architecture (all affect hash)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cfg_n_blocks : int
        Number of residual BitBlocks (default 6).
    cfg_hidden_channels : int
        Base channel width of the conv network (default 128).
    cfg_output_dim : int
        Output dimension — Gigapath feature dim (default 1536).
    cfg_n_classes : int
        Number of classes per feature dim (default 3, for ternary).
    cfg_activation_bits : int
        Activation quantization precision (default 8).

    Training hyper-parameters (all affect hash)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cfg_learning_rate : float
        Learning rate for AdamW (default 1e-3).
    cfg_weight_decay : float
        Weight decay for AdamW (default 0.01).
    cfg_scheduler : str
        LR scheduler name — ``'cosine'`` or ``'onecyclelr'`` (default ``'cosine'``).

    Logging / validation schedule (all affect hash)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cfg_log_every_n_steps : int
        How often (in training steps) to write TensorBoard scalars and images
        (default 20).  Goes into **both** ``BitPathConvLightning.spec`` and
        ``BitPathConvStill.spec`` directly.
    cfg_val_every_n_steps : int
        Run a validation pass every this many training steps (default 100).
        Goes directly into ``BitPathConvStill.spec``.
    cfg_limit_val_batches : int
        Maximum number of validation batches per check (default 200).
        Goes directly into ``BitPathConvStill.spec``.

    Dataloader (all affect hash)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cfg_batch_size : int
        Batch size for the DataLoader (default 64).
    cfg_shuffle : bool
        Whether to shuffle the dataset (default ``True``).
    cfg_dataloader_seed : int | None
        Random seed for reproducibility (default ``None``).
    cfg_num_workers : int
        Number of DataLoader workers (default 4).

    Bipolar clip params (all affect hash)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cfg_layer : str
        Which Gigapath capture key to bipolarize (default ``'output'``).
    cfg_bag_aggregation_threshold : float
        Threshold for bag-level bipolar aggregation (default 0.5).
    cfg_ternarize_tiles : bool
        Zero out tile dimensions where the bag-level sign disagrees
        (default ``False``).
    cfg_stats_probe_name : str | None
        Name for the stats probe clip used for per-dimension median
        computation.  Defaults to the CALIBRATE fold of the same partition.
    cfg_capture_blocks : list | None
        Which residual blocks to capture intermediate activations from.
    cfg_capture_layers : list | None
        Sub-layer names within each block to capture.
    cfg_capture_outputs : bool
        Whether to capture block output activations (default ``True``).
    cfg_cls_token_only : bool
        Use only the CLS token from the Gigapath backbone (default ``False``).
    cfg_shard_size : int
        Samples per MDS shard for the bipolar clip (default 64).

    Training-schedule params — **AFFECT HASH** (go into BitPathConvStill.spec)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cfg_max_epochs : int
        Number of training epochs (default 1).
    cfg_max_steps : int
        Max training steps per epoch (default 10000).
    cfg_gradient_clip_val : float
        Gradient clipping value for Lightning (default 1.0).
    cfg_ckpt_every_n_steps : int | None
        Save a checkpoint every N steps (default ``None`` = end-of-epoch only).
    cfg_precision : str | None
        PyTorch Lightning precision setting (e.g. ``'16-mixed'``).

    Build/execution params — **hash-neutral** (not in any spec)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_devices : int
        Number of GPUs for distributed training (default 1).
    devices : list | None
        Specific GPU device IDs (e.g. ``[0, 1, 2]``).
    n_workers : int
        Parallel workers for bipolar clip building (default 1).
    parallelization : str | None
        Parallelization strategy for clip building
        (``'multiprocessing'``, ``'multithreading'``, ``'inline'``).
    url : str | None
        Datablock storage URL (determines where artifacts are written,
        not what they contain).
    logsroot : str | None
        Root directory for local TensorBoard log symlinks.

    Examples
    --------
    ::

        ### 3-GPU BitPath distillation on CPTAC 60/20/20
        still = bitpath_conv_still(
            'GIGAPATH_DEEP_CPTAC_602020',
            cfg_layer='output',
            cfg_cls_token_only=True,
            cfg_shard_size=64,
            cfg_max_epochs=100,          # AFFECTS HASH — include in eval call too
            cfg_max_steps=10000,         # AFFECTS HASH — include in eval call too
            cfg_ckpt_every_n_steps=1000, # AFFECTS HASH — include in eval call too
            n_devices=3,                 # hash-neutral
        )
        still.build_tree()
    """


    # Strip any known split suffix so the user can pass either the base
    # name or a fully-qualified split name.
    _SPLIT_SUFFIXES = ('_TRAIN', '_TEST', '_CALIBRATE', '_VAL')
    base_name = name
    for sfx in _SPLIT_SUFFIXES:
        if base_name.endswith(sfx):
            base_name = base_name[: -len(sfx)]
            break
    train_name = f'{base_name}_TRAIN'
    val_name = f'{base_name}_TEST'

    _clip_kwargs = dict(
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
        n_workers=n_workers,
        parallelization=parallelization,
    )

    # 1. Build train and val bipolar clips
    train_clip = gigapath_bipolar_deep_feature_clip(train_name, **_clip_kwargs)
    val_clip = gigapath_bipolar_deep_feature_clip(val_name, **_clip_kwargs)

    _dl_spec = dict(
        batch_size=cfg_batch_size,
        shuffle=cfg_shuffle,
        seed=cfg_dataloader_seed,
        num_workers=cfg_num_workers,
    )

    # 2. Dataloader builders
    train_dataloader_builder = BitPathDataloaderBuilder(
        url=url,
        spec=dict(clip=dbx.quote(train_clip), **_dl_spec),
    )
    val_dataloader_builder = BitPathDataloaderBuilder(
        url=url,
        spec=dict(clip=dbx.quote(val_clip), **_dl_spec),
    )

    # 3. Lightning module factory
    lightning = BitPathConvLightning(
        url=url,
        spec=dict(
            n_blocks=cfg_n_blocks,
            hidden_channels=cfg_hidden_channels,
            output_dim=cfg_output_dim,
            n_classes=cfg_n_classes,
            activation_bits=cfg_activation_bits,
            learning_rate=cfg_learning_rate,
            weight_decay=cfg_weight_decay,
            scheduler=cfg_scheduler,
            feature_layer=cfg_layer,
            log_every_n_steps=cfg_log_every_n_steps,
        ),
    )

    # 4. Assemble the still
    # Every key in spec= goes into the content-hash of the still.
    # Reproduce ALL of these exactly when constructing the still for evaluation.
    return BitPathConvStill(
        url=url,
        spec=dict(
            lightning=dbx.quote(lightning),            # AFFECTS HASH — carries all cfg_n_blocks/lr/scheduler/... via lightning hash
            dataloader=dbx.quote(train_dataloader_builder),   # AFFECTS HASH — carries cfg_batch_size/shuffle/seed/num_workers
            val_dataloader=dbx.quote(val_dataloader_builder), # AFFECTS HASH — same as dataloader but val clip
            max_epochs=cfg_max_epochs,                     # AFFECTS HASH
            max_steps=cfg_max_steps,                       # AFFECTS HASH
            log_every_n_steps=cfg_log_every_n_steps,       # AFFECTS HASH (also in lightning spec)
            val_every_n_steps=cfg_val_every_n_steps,       # AFFECTS HASH
            limit_val_batches=cfg_limit_val_batches,       # AFFECTS HASH
            gradient_clip_val=cfg_gradient_clip_val,       # AFFECTS HASH
            ckpt_every_n_steps=cfg_ckpt_every_n_steps,     # AFFECTS HASH
            precision=cfg_precision,                       # AFFECTS HASH
        ),
        n_devices=n_devices,   # hash-neutral
        devices=devices,       # hash-neutral
        logsroot=logsroot,     # hash-neutral
    )


# ═══════════════════════════════════════════════════════════════════════
#  BitConv evaluator factory
# ═══════════════════════════════════════════════════════════════════════


def bitconv_deep_backbone_evaluator_factory(
    still: 'BitPathConvStill',
    *,
    url: str | None = None,
) -> BitConvDeepBackboneEvaluatorFactory:
    """Create a :class:`~autopath.stills.bitpath.BitConvDeepBackboneEvaluatorFactory`.

    Parameters
    ----------
    still : BitPathConvStill
        Trained still whose latest checkpoint will be loaded at eval time.
    url : str | None
        Datablock URL for relocatability.
    """
    return BitConvDeepBackboneEvaluatorFactory(
        url=url,
        spec=dict(still=dbx.quote(still)),
    )


# ═══════════════════════════════════════════════════════════════════════
#  BitConv deep feature clip
# ═══════════════════════════════════════════════════════════════════════

"""
### CPTAC 60/20/20 — BitConv feature extraction (CPU, multiprocessing)
git commit -am 'stills: bitconv_deep_feature_clip: BUILD' > /dev/null || true; dbx.pprint "\
autopath.stills.pipelines.bitconv_deep_feature_clip( \
    'BITCONV_DEEP_CPTAC_602020_TEST', \
    still=autopath.stills.pipelines.bitpath_conv_still( \
        'GIGAPATH_DEEP_CPTAC_602020', \
        cfg_layer='output', \
        cfg_cls_token_only=True, \
        cfg_shard_size=64, \
        cfg_max_steps=10000, \
        cfg_max_epochs=100, \
        cfg_ckpt_every_n_steps=1000,\
    ), \
    cfg_shard_size=64, \
    n_devices=3, \
    batch_size=32,\
    parallelization='multiprocessing', \
).build_tree()"
"""
def bitconv_deep_feature_clip(
    name: str,
    still: 'BitPathConvStill',
    *,
    cfg_shard_size: int = 1024,
    url: str | None = None,
    n_devices: int | None = None,
    devices: list | None = None,
    batch_size: int = 64,
    parallelization: str | None = None,
) -> DeepFeatureClip:
    """Create a :class:`~autopath.deep.features.DeepFeatureClip` backed by a trained BitConv still.

    Evaluates the trained :class:`~autopath.stills.bitpath.BitPathConvNet`
    on each tile and stores the resulting ternary ``{-1, 0, +1}``
    feature vectors as ``features_output`` in MDS format, identically to
    how :func:`~autopath.deep.pipelines.gigapath_deep_feature_clip` stores
    Gigapath activations.

    Parameters
    ----------
    name : str
        Named tile-bag clip to evaluate.  Supported patterns:

        * ``"BITCONV_DEEP_CPTAC"``              → full CPTAC tile-bag clip
        * ``"BITCONV_DEEP_CPTAC_<fold>"``       → a fold of CPTAC
          (e.g. ``"BITCONV_DEEP_CPTAC_602020_TRAIN"``)

        The ``BITCONV_DEEP_`` prefix is stripped to resolve the
        underlying tile-bag clip name.
    still : BitPathConvStill
        Trained still to use as the evaluator.  Its hash is incorporated
        into the clip's hash so that a new still produces a new clip.
    cfg_shard_size : int
        Number of samples per MDS shard file (default 1024).
    url : str | None
        Datablock URL for relocatability.
    n_devices : int | None
        Shorthand: use *n* CUDA devices (``['cuda:0', ..., 'cuda:{n-1}']``).
        Ignored when *devices* is provided explicitly.
    devices : list | None
        Explicit device list (e.g. ``['cuda:0', 'cuda:1']`` or ``['cpu']``
        for single-CPU inference).  Overrides *n_devices*.  Defaults to
        ``['cpu']`` when neither is set.
    cpu_batch_size : int
        Number of tiles processed in a single forward pass (default 256).
        Larger values increase memory usage but reduce overhead.
    parallelization : str | None
        Parallelization strategy (``'inline'``, ``'multithreading'``,
        ``'multiprocessing'``, ``'ray'``).  Defaults to
        ``'multiprocessing'`` when multiple devices are used.
    """
    # Resolve devices — mirrors gigapath_deep_feature_clip:
    # explicit devices wins; n_devices → cuda:0..n-1; default → ['cpu'].
    if devices is not None:
        pass
    elif n_devices is not None:
        devices = [f'cuda:{i}' for i in range(n_devices)]
    else:
        devices = ['cpu']

    if parallelization is None and len(devices) > 1:
        parallelization = 'multiprocessing'

    evaluator_factory = bitconv_deep_backbone_evaluator_factory(still, url=url)

    # Resolve the underlying tile-bag clip, mirroring gigapath_deep_feature_clip.
    # Strip 'BITCONV_DEEP_CPTAC' and re-prepend 'CPTAC_' for fold names so that
    # 'BITCONV_DEEP_CPTAC_602020_TRAIN' → pancan_tile_fold('CPTAC_602020_TRAIN').
    if name == 'BITCONV_DEEP_CPTAC':
        tilebagclip = dbx.quote(pancan_tile_clip, 'CPTAC')
    elif name.startswith('BITCONV_DEEP_CPTAC_'):
        fold_name = 'CPTAC_' + name[len('BITCONV_DEEP_CPTAC_'):]
        tilebagclip = dbx.quote(pancan_tile_fold, fold_name)
    else:
        raise ValueError(
            f"bitconv_deep_feature_clip: unsupported name {name!r}. "
            f"Expected 'BITCONV_DEEP_CPTAC' or 'BITCONV_DEEP_CPTAC_<fold>'."
        )

    return DeepFeatureClip(
        url=url,
        spec=dict(
            tilebagclip=tilebagclip,
            evaluator_factory=dbx.quote(evaluator_factory),
            shard_size=cfg_shard_size,
        ),
        device_batch_size=batch_size,
        devices=devices,
        n_workers=len(devices),
        parallelization=parallelization,
    )


# ═══════════════════════════════════════════════════════════════════════
#  BitConv affine logistic probe
# ═══════════════════════════════════════════════════════════════════════

"""
### CPTAC 60/20/20 — BitConv logistic probe on train fold
git commit -am 'stills: bitconv_deep_feature_affine_logistic_probe: BUILD' > /dev/null || true; dbx.pprint "\
autopath.stills.pipelines.bitconv_deep_feature_affine_logistic_probe( \
    'BITCONV_DEEP_CPTAC_602020_TEST', \
    still=autopath.stills.pipelines.bitpath_conv_still( \
        'GIGAPATH_DEEP_CPTAC_602020', \
        cfg_layer='output', \
        cfg_cls_token_only=True, \
        cfg_shard_size=64, \
        cfg_max_steps=10000, \
        cfg_max_epochs=100, \
        cfg_ckpt_every_n_steps=1000, \
    ), \
    cfg_shard_size=64, \
).build_tree()"
"""
def bitconv_deep_feature_affine_logistic_probe(
    name: str,
    still: 'BitPathConvStill',
    *,
    cfg_annotation_key: str | None = None,
    cfg_fit_intercept: bool = True,
    cfg_evaluation_fraction: float = 0.8,
    cfg_normalize: str | None = None,
    cfg_shard_size: int = 64,
    batch_size: int = 32,
    n_devices: int | None = None,
    devices: list | None = None,
    parallelization: str | None = None,
    url: str | None = None,
) -> 'DeepFeatureAffineLogisticProbe':
    """Create a :class:`~autopath.deep.probes.DeepFeatureAffineLogisticProbe`
    on BitConv deep features.

    Fits a logistic regression classifier on bag-level mean features
    from the ``'output'`` layer of a trained
    :class:`~autopath.stills.bitpath.BitPathConvStill`, persisting the
    fitted model's ``coef_``, ``intercept_``, and ``classes_`` arrays.

    Parameters
    ----------
    name : str
        Named fold configuration accepted by
        :func:`bitconv_deep_feature_clip` (e.g.
        ``"BITCONV_DEEP_CPTAC_602020_TEST"``).
    still : BitPathConvStill
        Trained BitPath still whose checkpoint is used as the backbone.
        Its spec hash is propagated into the probe's lineage.
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
        ``'corner-l1'`` / ``'corner-l2'`` — snap to ``{-1,+1}^d``,
        ``'corner-linfty'`` — axis-aligned vertex.
    cfg_shard_size : int
        Forwarded to :func:`bitconv_deep_feature_clip`.
    batch_size : int
        GPU batch size forwarded to :func:`bitconv_deep_feature_clip`.
    n_devices, devices, parallelization
        Device/worker config forwarded to :func:`bitconv_deep_feature_clip`.
    url : str | None
        Datablock URL.

    Examples
    --------
    ::

        probe = bitconv_deep_feature_affine_logistic_probe(
            'BITCONV_DEEP_CPTAC_602020_TEST',
            still=bitpath_conv_still(
                'GIGAPATH_DEEP_CPTAC_602020',
                cfg_layer='output',
                cfg_cls_token_only=True,
                cfg_shard_size=64,
                cfg_max_steps=10000,
                cfg_max_epochs=100,
                cfg_ckpt_every_n_steps=1000,
            ),
            cfg_shard_size=64,
        )
        probe.build_tree()
    """
    clip = bitconv_deep_feature_clip(
        name,
        still=still,
        cfg_shard_size=cfg_shard_size,
        batch_size=batch_size,
        n_devices=n_devices,
        devices=devices,
        parallelization=parallelization,
        url=url,
    )
    return DeepFeatureAffineLogisticProbe(
        url=url,
        spec=dict(
            clip=dbx.quote(clip),
            layer='output',
            annotation_key=cfg_annotation_key,
            fit_intercept=cfg_fit_intercept,
            evaluation_fraction=cfg_evaluation_fraction,
            normalize=cfg_normalize,
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
#  BitConv still probe
# ═══════════════════════════════════════════════════════════════════════

"""
### Connectivity plot for a trained BitPathConvStill
git commit -am 'stills: bitpath_conv_still_probe: BUILD' > /dev/null || true; dbx.pprint "\
autopath.stills.pipelines.bitpath_conv_still_probe( \
    still=autopath.stills.pipelines.bitpath_conv_still( \
        'GIGAPATH_DEEP_CPTAC_602020', \
        cfg_layer='output', \
        cfg_cls_token_only=True, \
        cfg_shard_size=64, \
        cfg_max_steps=10000, \
        cfg_max_epochs=100, \
        cfg_ckpt_every_n_steps=1000, \
    ), \
).build()"
"""
def bitpath_conv_still_probe(
    still: 'BitPathConvStill',
    *,
    url: str | None = None,
) -> 'BitPathConvStillProbe':
    """Create a :class:`~autopath.stills.bitpath.BitPathConvStillProbe`.

    Loads the latest checkpoint from *still*, extracts the ternary
    (``{-1, 0, +1}``) weights from every ``BitConv2d158`` and
    ``BitLinear158`` layer, and writes a connectivity PNG to the
    Datablock's ``'connectivity_plot'`` topic file.

    Parameters
    ----------
    still : BitPathConvStill
        Trained BitPath still to visualise.  Its spec hash is included
        in the probe's lineage so any change to training config
        invalidates the cached plot.
    url : str | None
        Datablock storage URL.  Defaults to the dbx workspace default.

    Examples
    --------
    ::

        probe = bitpath_conv_still_probe(
            still=bitpath_conv_still(
                'GIGAPATH_DEEP_CPTAC_602020',
                cfg_layer='output',
                cfg_cls_token_only=True,
                cfg_shard_size=64,
                cfg_max_steps=10000,
                cfg_max_epochs=100,
                cfg_ckpt_every_n_steps=1000,
            ),
        )
        probe.build()                        # writes connectivity_plot.png
        print(probe.path('connectivity_plot'))  # full path to the PNG
    """
    return BitPathConvStillProbe(
        url=url,
        spec=dict(still=dbx.quote(still)),
    )
