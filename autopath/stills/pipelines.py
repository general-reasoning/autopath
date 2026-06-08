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
    cfg_log_every_n_steps: int = 50,
    cfg_val_every_n_steps: int = 100,
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
        histograms.  Stored in the Lightning module CONFIG (affects hash).
    cfg_val_every_n_steps : int
        Run a validation pass every this many training steps.
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
            max_epochs=max_epochs,
            max_steps=max_steps,
            log_every_n_steps=cfg_log_every_n_steps,
            val_every_n_steps=cfg_val_every_n_steps,
            gradient_clip_val=gradient_clip_val,
            ckpt_every_n_steps=ckpt_every_n_steps,
            precision=precision,
        ),
        n_devices=n_devices,
        devices=devices,
        logsroot=logsroot,
    )
