"""Pipeline entrypoints for BitPath distillation stills.

Provides declarative constructors for
:class:`~autopath.stills.bitpath.BitPathConvStill`
and friends, wiring together bipolar deep feature clips,
dataloader builders, and Lightning module factories.
"""

from __future__ import annotations

import os

import dbx

from autopath.deep.pipelines import gigapath_bipolar_deep_feature_clip
from autopath.stills.bitpath import (
    BitPathDataloaderBuilder,
    BitPathConvLightning,
    BitPathConvStill,
)


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
    max_steps=10000, \
    max_epochs=100, \
    ckpt_every_n_steps=1000, \
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
    # Still params (non-CONFIG)
    max_epochs: int = 1,
    max_steps: int = 10000,
    gradient_clip_val: float = 1.0,
    ckpt_every_n_steps: int | None = None,
    precision: str | None = None,
    n_devices: int = 1,
    devices: list | None = None,
    n_workers: int = 1,
    parallelization: str | None = None,
    url: str | None = None,
    logsroot: str | None = None,
) -> 'BitPathConvStill':
    """Create a :class:`BitPathConvStill` for BitNet1.58b distillation.

    Trains a ternary convolutional network to predict bipolar features
    directly from raw tiles, distilling the Gigapath backbone.

    Parameters
    ----------
    name : str
        Base clip name **without** a split suffix, e.g.
        ``"GIGAPATH_DEEP_CPTAC_602020"``.  Any trailing ``_TRAIN``,
        ``_TEST``, or ``_CALIBRATE`` suffix is stripped automatically.
        Training uses ``<name>_TRAIN``; validation uses ``<name>_TEST``.
    cfg_log_every_n_steps : int
        How often (in training steps) to write TensorBoard scalars and
        images.  Stored in the Lightning module CONFIG (affects hash).
    cfg_val_every_n_steps : int
        Run a validation pass every this many training steps (default 200).
    cfg_limit_val_batches : int
        Maximum number of validation batches per check (default 200,
        = 200 * batch_size tiles).  Keeps each val pass quick even when
        the validation dataset is large.
    cfg_n_blocks : int
        Number of residual BitBlocks.
    cfg_hidden_channels : int
        Base channel width of the conv network.
    cfg_output_dim : int
        Output dimension (Gigapath feature dim, default 1536).
    cfg_n_classes : int
        Number of classes per feature dim (3 for ternary).
    cfg_activation_bits : int
        Activation quantization precision.
    cfg_learning_rate : float
        Learning rate for AdamW.
    cfg_weight_decay : float
        Weight decay for AdamW.
    cfg_scheduler : str
        LR scheduler name (``'cosine'`` or ``'onecyclelr'``).
    cfg_batch_size : int
        Batch size for the DataLoader.
    cfg_shuffle : bool
        Whether to shuffle the dataset.
    cfg_dataloader_seed : int | None
        Random seed for reproducibility.
    cfg_num_workers : int
        Number of DataLoader workers.
    cfg_layer : str
        Which capture key to bipolarize.
    cfg_bag_aggregation_threshold : float
        Threshold for bag-level bipolar aggregation.
    cfg_ternarize_tiles : bool
        Zero out tile dimensions where the bag disagrees.
    cfg_stats_probe_name : str | None
        Stats probe name (defaults to CALIBRATE fold).
    max_epochs : int
        Number of training epochs.
    max_steps : int
        Max batches per epoch.
    n_devices : int
        Number of GPUs.
    devices : list | None
        Specific GPU device IDs.
    n_workers : int
        Workers for parallel bipolar bag building.
    parallelization : str | None
        Parallelization strategy for bipolar clip building.
    url : str | None
        Datablock URL.
    logsroot : str | None
        Root directory for TensorBoard log symlinks.

    Examples
    --------
    ::

        ### 3-GPU BitPath distillation on CPTAC 60/20/20
        still = bitpath_conv_still(
            'GIGAPATH_DEEP_CPTAC_602020',
            cfg_layer='output',
            cfg_cls_token_only=True,
            cfg_shard_size=64,
            max_steps=10000,
            ckpt_every_n_steps=1000,
            n_devices=3,
        )
        still.build_tree()
    """

    if logsroot is None:
        logsroot = os.path.join(
            os.environ.get('HOME', '/tmp'), 'autopath', 'tensorboard',
        )

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
    return BitPathConvStill(
        url=url,
        spec=dict(
            lightning=dbx.quote(lightning),
            dataloader=dbx.quote(train_dataloader_builder),
            val_dataloader=dbx.quote(val_dataloader_builder),
            max_epochs=max_epochs,
            max_steps=max_steps,
            log_every_n_steps=cfg_log_every_n_steps,
            val_every_n_steps=cfg_val_every_n_steps,
            limit_val_batches=cfg_limit_val_batches,
            gradient_clip_val=gradient_clip_val,
            ckpt_every_n_steps=ckpt_every_n_steps,
            precision=precision,
        ),
        n_devices=n_devices,
        devices=devices,
        logsroot=logsroot,
    )


# ═══════════════════════════════════════════════════════════════════════
#  BitConv evaluator factory
# ═══════════════════════════════════════════════════════════════════════

from autopath.stills.backbone import (      # noqa: E402 (local import avoids circular dep at module load)
    BitConvDeepBackboneEvaluator,
    BitConvDeepBackboneEvaluatorFactory,
)
from autopath.deep.features import DeepFeatureClip  # noqa: E402
from autopath.pancan.pipelines import (             # noqa: E402
    pancan_tile_clip,
    pancan_tile_fold,
)


def bitconv_deep_backbone_evaluator_factory(
    still: 'BitPathConvStill',
    *,
    url: str | None = None,
) -> BitConvDeepBackboneEvaluatorFactory:
    """Create a :class:`~autopath.stills.backbone.BitConvDeepBackboneEvaluatorFactory`.

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
        ckpt_every_n_steps=1000,\
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
        gpu_batch_size=batch_size,
        devices=devices,
        n_workers=len(devices),
        parallelization=parallelization,
        keyby='tag_version_hash',
    )

