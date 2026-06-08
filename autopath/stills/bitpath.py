"""BitPath: BitNet1.58b distillation of Gigapath backbone.

Trains a ternary convolutional network to predict bipolar features
``{-1, 0, +1}`` directly from raw pathology tiles, bypassing the
heavyweight Gigapath ViT backbone at inference time.

See ``~/autopath/BITPATH.md`` for architecture documentation.
"""
import atexit
from dataclasses import dataclass
from datetime import datetime
import functools
import glob
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
import lightning.pytorch.loggers

import dbx
from dbx import Datablock

from streaming import Stream, StreamingDataset

from autopath.autobits import ZipStreamingDataset, sanitize_collate


# ═══════════════════════════════════════════════════════════════════════
#  BitNet 1.58b primitives
# ═══════════════════════════════════════════════════════════════════════


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """Round with Straight-Through Estimator gradient."""
    return x + (x.round() - x).detach()


def _weight_quant_absmean(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Absmean ternary quantization of weights → {-1, 0, +1} × α.

    α = mean(|W|)
    W̃ = round(clip(W / α, -1, +1)) × α
    """
    alpha = w.abs().mean() + eps
    w_scaled = (w / alpha).clamp(-1, 1)
    w_quant = _ste_round(w_scaled) * alpha
    return w_quant


def _activation_quant_absmax(
    x: torch.Tensor, bits: int = 8
) -> torch.Tensor:
    """Absmax symmetric activation quantization.

    γ = max(|x|)
    Q_b = 2^(b−1)
    x̃ = clip(round(x × Q_b / γ), −Q_b+1, Q_b−1)
    """
    Qb = 2 ** (bits - 1)
    gamma = x.abs().max() + 1e-8
    x_scaled = x * Qb / gamma
    x_quant = x_scaled.round().clamp(-Qb + 1, Qb - 1)
    # STE: forward uses quantized, backward flows through
    return x + (x_quant * gamma / Qb - x).detach()


class RMSNorm2d(nn.Module):
    """RMSNorm for 4-D tensors (B, C, H, W)."""

    def __init__(self, channels: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — normalize over C dimension
        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale.view(1, -1, 1, 1)


class RMSNorm1d(nn.Module):
    """RMSNorm for 2-D tensors (B, D)."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale


class BitLinear158(nn.Module):
    """Linear layer with BitNet 1.58b ternary weight quantization.

    Weights are stored in full precision but quantized to ``{-1, 0, +1}``
    on every forward pass using *absmean* quantization.  Gradients flow
    through via the Straight-Through Estimator.

    Activations are optionally quantized to ``activation_bits``-bit
    symmetric range after RMSNorm.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_bits: int = 8,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_bits = activation_bits
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.norm = RMSNorm1d(in_features)
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize and quantize activations
        x = self.norm(x)
        x = _activation_quant_absmax(x, self.activation_bits)
        # Quantize weights
        w = _weight_quant_absmean(self.weight)
        return F.linear(x, w, self.bias)


class BitConv2d158(nn.Module):
    """Conv2d with BitNet 1.58b ternary weight quantization.

    A drop-in replacement for ``nn.Conv2d`` where weights are quantized
    to ``{-1, 0, +1}`` on every forward pass via absmean quantization
    with STE gradients.

    Activations are normalized with RMSNorm and quantized to
    ``activation_bits``-bit symmetric range before the convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
        activation_bits: int = 8,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.activation_bits = activation_bits
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.norm = RMSNorm2d(in_channels)
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize and quantize activations
        x = self.norm(x)
        x = _activation_quant_absmax(x, self.activation_bits)
        # Quantize weights
        w = _weight_quant_absmean(self.weight)
        return F.conv2d(x, w, self.bias, self.stride, self.padding)


# ═══════════════════════════════════════════════════════════════════════
#  BitBlock: residual block built from BitConv2d158
# ═══════════════════════════════════════════════════════════════════════


class BitBlock(nn.Module):
    """Residual block of two BitConv2d158 layers with batch norm + ReLU.

    When ``stride > 1``, the first conv downsamples spatially and
    a 1×1 projection adapts the shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation_bits: int = 8,
    ):
        super().__init__()
        self.conv1 = BitConv2d158(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1,
            activation_bits=activation_bits,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = BitConv2d158(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
            activation_bits=activation_bits,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut projection when dimensions change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


# ═══════════════════════════════════════════════════════════════════════
#  BitPathConvNet: full distillation network
# ═══════════════════════════════════════════════════════════════════════


class BitPathConvNet(nn.Module):
    """BitNet1.58b convolutional network for Gigapath distillation.

    Input: ``(B, 3, 256, 256)`` RGB tiles.
    Output: ``(B, n_classes, output_dim)`` logits — ``n_classes`` logits
    per feature dimension (default 3 × 1536).

    Architecture:
        Full-precision stem → N BitBlocks (ternary) → GAP → BitLinear158 head
    """

    def __init__(
        self,
        *,
        n_blocks: int = 6,
        hidden_channels: int = 128,
        output_dim: int = 1536,
        n_classes: int = 3,
        activation_bits: int = 8,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.n_classes = n_classes

        # Stem: full-precision conv to lift 3-channel RGB
        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
        )

        # BitBlocks with progressive channel doubling
        blocks = []
        in_ch = hidden_channels
        for i in range(n_blocks):
            # Double channels every 2 blocks, downsample at the same time
            if i > 0 and i % 2 == 0:
                out_ch = min(in_ch * 2, 1024)
                stride = 2
            else:
                out_ch = in_ch
                stride = 1
            blocks.append(BitBlock(
                in_ch, out_ch, stride=stride,
                activation_bits=activation_bits,
            ))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # Head: global average pool → ternary linear
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = BitLinear158(
            in_ch, n_classes * output_dim,
            activation_bits=activation_bits,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            ``(B, 3, H, W)`` RGB tile images, float32, [0, 255] range.

        Returns
        -------
        Tensor
            ``(B, n_classes, output_dim)`` raw logits.
        """
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x).flatten(1)       # (B, C_final)
        x = self.head(x)                  # (B, n_classes * output_dim)
        return x.view(x.size(0), self.n_classes, self.output_dim)


def _bitpath_collate(batch):
    """Collate a BitPath batch, dropping the un-batchable ``'annotations'`` column.

    ``annotations`` is a JSON dict column present in some tile shards but
    absent in others, and its nested structure varies per bag.  It is not
    used during BitPath training, so we drop it before delegating to
    :func:`sanitize_collate`.
    """
    drop = {'annotations'}
    filtered = [{k: v for k, v in s.items() if k not in drop} for s in batch]
    return sanitize_collate(filtered)



def _tb_bar(writer, tag, values_1d, step, *, ylim=(-1, 1)):
    """Plot a 1-D array as a red/blue bar chart figure in TensorBoard.

    Positive values → red, negative values → blue.
    Uses :func:`SummaryWriter.add_figure` so TensorBoard's *Images* tab
    shows one rendered figure per step with a step slider.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.asarray(values_1d, dtype=float)
    D = len(arr)
    fig, ax = plt.subplots(figsize=(20, 3))
    ax.bar(range(D), np.maximum(arr, 0), width=1.0, color='#dc2626', linewidth=0)
    ax.bar(range(D), np.minimum(arr, 0), width=1.0, color='#2563eb', linewidth=0)
    ax.axhline(0, color='k', linewidth=0.4, alpha=0.5)
    ax.set_xlim(0, D)
    ax.set_ylim(*ylim)
    ax.set_xlabel('Feature dimension')
    ax.set_ylabel('Value')
    ax.set_title(tag)
    fig.tight_layout()
    writer.add_figure(tag, fig, step)
    plt.close(fig)


def _tb_ternary_bar(writer, tag, vectors, step):
    """Batch-mean of ternary ``(B, D)`` vectors as a bar chart."""
    import numpy as np
    _tb_bar(writer, tag, vectors.float().mean(dim=0).numpy(), step, ylim=(-1, 1))





class BitPathDataloaderBuilder(Datablock):
    """Factory that builds a DataLoader from a BipolarDeepFeatureClip.

    CONFIG contains only parameters that affect the training output:
    batch_size, shuffle, seed, num_workers.
    """

    @dataclass
    class CONFIG:
        clip: object  # BipolarDeepFeatureClip (quoted)
        batch_size: int = 64
        shuffle: bool = True
        seed: int | None = None
        num_workers: int = 4

    @staticmethod
    def _mds_n_samples(path, fs):
        """Read the total sample count from an MDS ``index.json``."""
        import json as _json
        idx_path = os.path.join(path, 'index.json')
        with fs.open(idx_path, 'r') as fh:
            index = _json.load(fh)
        return sum(s['samples'] for s in index['shards'])

    def dataloader(self):
        """Build and return a DataLoader over zipped tile+feature datasets.

        Creates a :class:`ZipStreamingDataset` pairing the bipolar
        feature clip's dataset with the source tile clip's dataset,
        so each sample contains both ``bipolar_features_{layer}`` and
        ``bag_bipolar_features_{layer}`` alongside ``tile`` images.

        Bags are aligned **per-bag**: for each valid bipolar bag we
        locate the exact same tile bag and compare their MDS shard
        tile counts.  Bags where the counts differ (e.g. the tile bag
        was re-tiled after the bipolar bag was built) are skipped with
        a warning rather than causing a length-mismatch error.
        """
        bipolar_clip = self.cfg.clip
        bipolar_streams: list = []
        tile_streams: list = []
        n_skipped_invalid = 0
        n_skipped_mismatch = 0

        for bag in bipolar_clip.bags:
            if not bag.valid():
                n_skipped_invalid += 1
                continue

            tilebag = bag.tilebag

            # Per-bag count check — catches re-tiled bags
            n_bip = self._mds_n_samples(bag.path('shards'), bag.fs)
            n_tile = self._mds_n_samples(tilebag.path('shards'), tilebag.fs)
            if n_bip != n_tile:
                n_skipped_mismatch += 1
                bipolar_clip.log.warning(
                    f'Skipping bag {bag.tag!r}: bipolar shards have '
                    f'{n_bip} tiles but tile bag has {n_tile} tiles. '
                    f'Rebuild: bipolar bag path = {bag.path("shards")!r}'
                )
                continue

            if bag.is_local_fs:
                bipolar_streams.append(Stream(local=bag.path('shards')))
            else:
                bipolar_streams.append(Stream(remote=bag.path('shards')))

            if tilebag.is_local_fs:
                tile_streams.append(Stream(local=tilebag.path('shards')))
            else:
                tile_streams.append(Stream(remote=tilebag.path('shards')))

        n_total = len(bipolar_clip.bags)
        n_used = len(bipolar_streams)
        if n_skipped_invalid:
            bipolar_clip.log.info(
                f'Skipped {n_skipped_invalid}/{n_total} invalid bipolar bags'
            )
        if n_skipped_mismatch:
            bipolar_clip.log.warning(
                f'Skipped {n_skipped_mismatch}/{n_total} bags due to '
                f'bipolar/tile count mismatch — rebuild those bipolar bags '
                f'to restore them. Using {n_used}/{n_total} bags.'
            )

        sd_kwargs = dict(shuffle=self.cfg.shuffle, batch_size=self.cfg.batch_size)
        bipolar_ds = StreamingDataset(streams=bipolar_streams, **sd_kwargs)
        tile_ds = StreamingDataset(streams=tile_streams, **sd_kwargs)
        ds = ZipStreamingDataset(bipolar_ds, tile_ds)

        generator = None
        if self.cfg.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.cfg.seed)

        return torch.utils.data.DataLoader(
            dataset=ds,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            collate_fn=_bitpath_collate,
            pin_memory=True,
            generator=generator,
        )


# ═══════════════════════════════════════════════════════════════════════
#  BitPathConvLightning (Datablock)
# ═══════════════════════════════════════════════════════════════════════


class BitPathConvLightning(Datablock):
    """Lightning module factory for BitPath distillation.

    CONFIG contains only parameters that affect the training output:
    model architecture, optimizer, and scheduler parameters.
    """

    @dataclass
    class CONFIG:
        # Model architecture
        n_blocks: int = 6
        hidden_channels: int = 128
        output_dim: int = 1536
        n_classes: int = 3
        activation_bits: int = 8
        # Optimizer
        learning_rate: float = 1e-3
        weight_decay: float = 0.01
        # Scheduler
        scheduler: str = 'cosine'
        # Feature column
        feature_layer: str = 'output'
        # Logging
        log_every_n_steps: int = 50

    class Lightning(L.LightningModule):

        def __init__(
            self,
            model: nn.Module,
            feature_layer: str = 'output',
            learning_rate: float = 1e-3,
            weight_decay: float = 0.01,
            scheduler: str = 'cosine',
            log_every_n_steps: int = 50,
            log: dbx.Logger = None,
        ):
            super().__init__()
            self.model = model
            self.feature_layer = feature_layer
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.scheduler_name = scheduler
            self.log_every_n_steps = log_every_n_steps
            self.save_hyperparameters(ignore=['model'])
            self.log_ = log or dbx.Logger(name='BitPathConvLightning')

        def training_step(self, batch, batch_idx):
            # batch is a dict from StreamingDataset
            tiles = batch['tile']             # (B, H, W, 3) uint8 ndarray
            targets = batch[f'bipolar_features_{self.feature_layer}']  # (B, D) int8

            # Prepare tiles: (B, H, W, 3) → (B, 3, H, W) float32
            if tiles.ndim == 4 and tiles.shape[-1] == 3:
                tiles = tiles.permute(0, 3, 1, 2)
            tiles = tiles.float()

            # Prepare targets: {-1, 0, +1} → class indices {0, 1, 2}
            targets = (targets.long() + 1)  # -1→0, 0→1, +1→2

            # Forward: (B, n_classes, D) logits
            logits = self.model(tiles)

            # Cross-entropy: reshape to (B*D, n_classes) vs (B*D,)
            B, C, D = logits.shape
            loss = F.cross_entropy(
                logits.permute(0, 2, 1).reshape(-1, C),  # (B*D, C)
                targets.reshape(-1),                       # (B*D,)
            )

            if torch.isnan(loss) or torch.isinf(loss):
                self.log_.warning(f'[step {self.global_step}] Loss is {loss.item()}!')

            # Throttled TensorBoard logging (rank-0 only)
            if self.global_step % self.log_every_n_steps == 0 and self.trainer.is_global_zero:
                exp = self.logger.experiment
                step = self.global_step
                exp.add_scalar('Train/Loss', loss.item(), step)

                with torch.no_grad():
                    preds = logits.argmax(dim=1) - 1  # {-1, 0, +1}, shape (B, D)
                    actual = targets - 1
                    acc = (preds == actual).float().mean()
                    exp.add_scalar('Train/Accuracy', acc.item(), step)
                    # Batch-mean bar charts
                    _tb_ternary_bar(exp, 'Train/Output', preds.float().cpu(), step)
                    _tb_ternary_bar(exp, 'Train/Target', actual.float().cpu(), step)
                    # 0th-sample bar charts + diff
                    p0 = preds[0].float().cpu()
                    a0 = actual[0].float().cpu()
                    _tb_bar(exp, 'Train/Sample/Output', p0, step, ylim=(-1, 1))
                    _tb_bar(exp, 'Train/Sample/Target', a0, step, ylim=(-1, 1))
                    _tb_bar(exp, 'Train/Sample/Diff',   p0 - a0, step, ylim=(-2, 2))

                    scheduler = self.lr_schedulers()
                    if scheduler is not None:
                        exp.add_scalar('Train/LR', scheduler.get_last_lr()[0], step)

            return loss

        def validation_step(self, batch, batch_idx):
            tiles = batch['tile']
            targets = batch[f'bipolar_features_{self.feature_layer}']

            if tiles.ndim == 4 and tiles.shape[-1] == 3:
                tiles = tiles.permute(0, 3, 1, 2)
            tiles = tiles.float()
            targets = (targets.long() + 1)  # -1→0, 0→1, +1→2

            logits = self.model(tiles)
            B, C, D = logits.shape
            loss = F.cross_entropy(
                logits.permute(0, 2, 1).reshape(-1, C),
                targets.reshape(-1),
            )

            preds = logits.argmax(dim=1) - 1  # {-1, 0, +1}
            actual = targets - 1
            acc = (preds == actual).float().mean()

            # Aggregated across DDP ranks and val batches
            self.log('Val/Loss', loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log('Val/Accuracy', acc, on_step=False, on_epoch=True, sync_dist=True)

            # Histograms logged once per val run from rank 0
            if batch_idx == 0 and self.trainer.is_global_zero:
                step = self.global_step
                exp = self.logger.experiment
                # Loss preview: epoch-level log only fires after the full val
                # epoch, so also emit a step-level scalar immediately.
                exp.add_scalar('Val/Loss', loss.item(), step)
                _tb_ternary_bar(exp, 'Val/Output', preds.float().cpu(), step)
                _tb_ternary_bar(exp, 'Val/Target', actual.float().cpu(), step)
                p0 = preds[0].float().cpu()
                a0 = actual[0].float().cpu()
                _tb_bar(exp, 'Val/Diff', p0 - a0, step, ylim=(-2, 2))

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            stepping_batches = self.trainer.estimated_stepping_batches

            if self.scheduler_name == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=stepping_batches,
                    eta_min=1e-6,
                )
            elif self.scheduler_name == 'onecyclelr':
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.learning_rate,
                    total_steps=stepping_batches,
                )
            else:
                raise ValueError(f'Unknown scheduler: {self.scheduler_name}')

            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
            }

    @functools.cached_property
    def lightning_module(self):
        model = BitPathConvNet(
            n_blocks=self.cfg.n_blocks,
            hidden_channels=self.cfg.hidden_channels,
            output_dim=self.cfg.output_dim,
            n_classes=self.cfg.n_classes,
            activation_bits=self.cfg.activation_bits,
        )
        return self.Lightning(
            model=model,
            feature_layer=self.cfg.feature_layer,
            learning_rate=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
            scheduler=self.cfg.scheduler,
            log_every_n_steps=self.cfg.log_every_n_steps,
            log=self.log,
        )


# ═══════════════════════════════════════════════════════════════════════
#  BitPathConvStill (Datablock)
# ═══════════════════════════════════════════════════════════════════════


class BitPathDataModule(L.LightningDataModule):
    """LightningDataModule for BitPath training.

    Creates the ``StreamingDataset``-backed dataloader lazily in
    ``setup()`` — called **once per rank** after the DDP process group
    is already initialised — and caches the result so that repeated
    calls to ``train_dataloader()`` (e.g. from
    ``configure_optimizers → estimated_stepping_batches``) never
    re-create shared-memory segments.

    Lightning's ``SubprocessScriptLauncher`` sets ``LOCAL_RANK`` and
    ``WORLD_SIZE`` but omits ``RANK`` and ``LOCAL_WORLD_SIZE``.
    MosaicML streaming reads those from env vars, so without the
    patch every rank computes ``is_local_leader=True`` and they race
    to create the shared-memory segment (mosaicml/streaming#717).
    ``setup()`` fills in the missing vars from the trainer.
    """

    def __init__(self, dataloader_builder, val_dataloader_builder=None):
        super().__init__()
        self._builder = dataloader_builder
        self._builder_val = val_dataloader_builder
        self._train_dl = None
        self._val_dl = None

    def setup(self, stage=None):
        if stage in ('fit', None):
            # MosaicML streaming reads rank topology from env vars
            # (RANK, LOCAL_WORLD_SIZE) — not torch.distributed.  Lightning's
            # SubprocessScriptLauncher sets LOCAL_RANK and WORLD_SIZE but
            # omits RANK and LOCAL_WORLD_SIZE, so every rank computes
            # is_local_leader=True and they all race to create the shared
            # memory segment (mosaicml/streaming#717).  Patch env from the
            # trainer, which *does* know the correct topology.
            if self.trainer is not None:
                os.environ.setdefault('RANK', str(self.trainer.global_rank))
                os.environ.setdefault('LOCAL_WORLD_SIZE',
                                      str(self.trainer.num_devices))
            if self._train_dl is None:
                self._purge_shm()
                self._train_dl = self._builder.dataloader()
                atexit.register(self._purge_shm)
            if self._val_dl is None and self._builder_val is not None:
                self._val_dl = self._builder_val.dataloader()

    @staticmethod
    def _purge_shm():
        """Remove MosaicML streaming shared-memory files from /dev/shm.

        Best-effort cleanup of segments left by crashed runs.
        """
        for f in glob.glob('/dev/shm/[0-9][0-9][0-9][0-9][0-9][0-9]_*'):
            try:
                os.remove(f)
            except OSError:
                pass

    def train_dataloader(self):
        return self._train_dl

    def val_dataloader(self):
        # Return [] (not None) when no val data — Lightning requires an
        # iterable, and an empty list paired with limit_val_batches=0
        # cleanly skips validation.
        return self._val_dl if self._val_dl is not None else []


class BitPathConvStill(Datablock):
    """Full training pipeline for BitPath distillation.

    Uses Lightning for training with TensorBoard logging, checkpoint
    management, and log-directory symlinks for TensorBoard discovery.
    Follows the ``SteamrollerStill`` pattern from soundworld.

    Logs and checkpoints are written locally first, then synced to
    remote storage (if the Datablock URL is remote) at the end of
    training and via an ``atexit`` handler for crash resilience.
    """

    VERSION = 1
    TOPICS = ['ckpts', 'logs']

    @dataclass
    class CONFIG:
        lightning: BitPathConvLightning
        dataloader: BitPathDataloaderBuilder
        val_dataloader: object = None  # Optional BitPathDataloaderBuilder
        max_epochs: int = 1
        max_steps: int = 10000
        log_every_n_steps: int = 50
        val_every_n_steps: int = 100
        limit_val_batches: int = 200  # max batches per val check
        gradient_clip_val: float = 1.0
        gradient_clip_algorithm: str = 'norm'
        ckpt_every_n_steps: int | None = None
        precision: str | None = None

    def __init__(
        self,
        *args,
        n_devices: int = 1,
        devices=None,
        logsroot: str = None,
        save_remote_logs: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            n_devices=n_devices,
            devices=devices,
            logsroot=logsroot,
            save_remote_logs=save_remote_logs,
            **kwargs,
        )
        self.save_remote_logs = save_remote_logs

        if isinstance(self.cfg.max_steps, str):
            self.max_steps = int(self.cfg.max_steps.strip('%'))  # resolved in __build__
        else:
            self.max_steps = self.cfg.max_steps

    # -- local working directory -------------------------------------------

    @property
    def _local_workdir(self):
        """Local directory for training artifacts (logs, ckpts).

        Uses ``~/autopath/tensorboard/<anchorkey>`` so that TensorBoard
        can discover the logs locally even when the Datablock URL is
        remote.
        """
        base = os.path.join(
            os.environ.get('HOME', '/tmp'), 'autopath', 'tensorboard',
        )
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, self.anchorkey)

    @property
    def _local_logs_dir(self):
        return os.path.join(self._local_workdir, 'logs')

    @property
    def _local_ckpts_dir(self):
        return os.path.join(self._local_workdir, 'ckpts')

    # -- remote helpers ----------------------------------------------------

    def _is_remote(self):
        protocol = (
            self.fs.protocol
            if isinstance(self.fs.protocol, str)
            else self.fs.protocol[0]
        )
        return protocol not in ('file', 'local', '')

    def _sync_ckpts_to_remote(self, ckpts_dir):
        """Upload local checkpoints to remote Datablock storage."""
        if not self._is_remote():
            return
        remote_ckpts = self.dirpath('ckpts', ensure=True)
        self.log.info(f'Syncing ckpts {ckpts_dir} -> {remote_ckpts}')
        self.fs.put(ckpts_dir, remote_ckpts, recursive=True)

    def _sync_logs_to_remote(self, logs_dir):
        """Upload local logs to remote Datablock storage."""
        if not self._is_remote():
            return
        remote_logs = self.dirpath('logs', ensure=True)
        self.log.info(f'Syncing logs {logs_dir} -> {remote_logs}')
        self.fs.put(logs_dir, remote_logs, recursive=True)

    def upload(self, topic='logs'):
        """Upload a local-only topic to remote storage.

        ``'logs'`` and ``'ckpts'`` are written locally first and need
        explicit upload.  Other topics are a no-op.
        """
        if topic == 'logs':
            self._sync_logs_to_remote(self._local_logs_dir)
        elif topic == 'ckpts':
            self._sync_ckpts_to_remote(self._local_ckpts_dir)
        else:
            self.log.verbose("upload(%r): no-op (already remote)", topic)
        return self

    # -- log symlinking ----------------------------------------------------

    @property
    def _logslink(self):
        """Symlink target for TensorBoard discovery (or ``None``)."""
        if self.logsroot is None:
            return None
        return os.path.join(self.logsroot, self.tag)

    def linklogs(self):
        """Create a symlink so TensorBoard discovers the logs directory.

        If the Datablock storage is local, the symlink points directly
        to ``self.dirpath('logs')``.  If remote, it points to the local
        proxy directory.
        """
        logslink = self._logslink
        if logslink is None:
            return self

        if self._is_remote():
            logs_dir = self._local_logs_dir
            os.makedirs(logs_dir, exist_ok=True)
        else:
            logs_dir = self.dirpath('logs', ensure=True)

        os.makedirs(os.path.dirname(logslink), exist_ok=True)

        if os.path.lexists(logslink):
            if (
                os.path.islink(logslink)
                and os.readlink(logslink) == logs_dir
            ):
                return self
            try:
                os.remove(logslink)
            except Exception as e:
                self.log.warning(
                    f'Could not remove existing path at {logslink}: {e}'
                )
                return self
        try:
            os.symlink(logs_dir, logslink)
            self.log.verbose(f'Linked logs: {logs_dir} -> {logslink}')
        except Exception as e:
            self.log.warning(
                f'Failed to create symlink {logslink} -> {logs_dir}: {e}'
            )
        return self

    # -- validity ----------------------------------------------------------

    def valid(self):
        """Return True when a ``_COMPLETE`` marker is present."""
        local_marker = os.path.join(self._local_ckpts_dir, '_COMPLETE')
        if os.path.exists(local_marker):
            return True
        # Fall back to remote check
        try:
            remote_ckpts = self.dirpath('ckpts')
            return self.fs.exists(os.path.join(remote_ckpts, '_COMPLETE'))
        except Exception:
            return False

    def __pre_build__(self):
        super().__pre_build__()
        self.linklogs()
        return self

    # -- checkpoint helpers ------------------------------------------------

    @staticmethod
    def _ckpt_step(name):
        """Extract the numeric step from a checkpoint filename."""

        m = re.search(r'step=(?:step=)?(\d+)', name)
        return int(m.group(1)) if m else -1

    def ckpt(self):
        """Return the path to the most recent checkpoint, or None.

        Checks remote storage first (downloads to local if found),
        then scans local.
        """
        ckpts_dir = self._local_ckpts_dir
        os.makedirs(ckpts_dir, exist_ok=True)

        # 1) Try remote — download the latest ckpt if not already local
        try:
            remote_ckpts_dir = self.dirpath('ckpts')
            remote_files = [
                f for f in (self.fs.ls(remote_ckpts_dir, detail=False)
                            if self.fs.exists(remote_ckpts_dir) else [])
                if f.endswith('.ckpt')
            ]
            if remote_files:
                remote_files.sort(
                    key=lambda f: self._ckpt_step(os.path.basename(f))
                )
                remote_ckpt = remote_files[-1]
                name = os.path.basename(remote_ckpt)
                local_path = os.path.join(ckpts_dir, name)
                if not os.path.exists(local_path):
                    self.log.info(
                        'ckpt: downloading %s from remote...', name,
                    )
                    self.fs.get(remote_ckpt, local_path)
        except Exception as e:
            self.log.verbose(f'ckpt: remote check failed: {e}')

        # 2) Scan local directory for the latest valid checkpoint
        if not os.path.isdir(ckpts_dir):
            return None

        candidates = [
            os.path.join(ckpts_dir, f)
            for f in os.listdir(ckpts_dir)
            if f.endswith('.ckpt')
        ]
        candidates.sort(
            key=lambda p: self._ckpt_step(os.path.basename(p))
        )
        return candidates[-1] if candidates else None

    # -- build (training) --------------------------------------------------

    def __build__(self):
        """Run the training loop."""

        logs_dir = self._local_logs_dir
        ckpts_dir = self._local_ckpts_dir
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(ckpts_dir, exist_ok=True)

        # Ensure ckpts AND logs are synced even on interrupt/crash
        def _atexit_sync():
            self.log.info('atexit: syncing checkpoints to remote...')
            try:
                self._sync_ckpts_to_remote(ckpts_dir)
            except Exception as e:
                self.log.warning('atexit: ckpt sync failed: %s', e)
            if self.save_remote_logs:
                self.log.info('atexit: syncing logs to remote...')
                try:
                    self._sync_logs_to_remote(logs_dir)
                except Exception as e:
                    self.log.warning('atexit: log sync failed: %s', e)

        atexit.register(_atexit_sync)

        # Symlink for TensorBoard discovery
        self.linklogs()

        logger = L.pytorch.loggers.TensorBoardLogger(
            save_dir=logs_dir,
            default_hp_metric=False,
            name='',
            version=f"run_{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}",
        )
        logger.experiment.add_text(
            'BitPathConvStill: anchorkeypath',
            f'```python\n{self.anchorkeypath}\n```',
            global_step=0,
        )
        logger.experiment.add_text(
            'BitPathConvStill: dfn',
            f'```python\n{self.dfn}\n```',
            global_step=0,
        )
        logger.experiment.add_text(
            'path/ckpts', self.dirpath('ckpts'), global_step=0,
        )
        logger.experiment.add_text(
            'path/logs', self.dirpath('logs'), global_step=0,
        )

        self.log.verbose(f'Building trainer for {self.max_steps} steps')

        kwargs = {}
        if self.cfg.gradient_clip_val > 0.0:
            kwargs['gradient_clip_val'] = self.cfg.gradient_clip_val
            kwargs['gradient_clip_algorithm'] = self.cfg.gradient_clip_algorithm

        # Custom step checkpoint callback with upload-and-free
        callbacks = []
        _still = self
        _ckpts_dir = ckpts_dir

        def _upload_and_free(local_path, filename):
            """Upload a checkpoint to remote storage and delete local copy."""
            if not _still._is_remote():
                return
            try:
                remote_ckpts = _still.dirpath('ckpts', ensure=True)
                remote_path = remote_ckpts.rstrip('/') + '/' + filename
                _still.fs.put(local_path, remote_path)
                os.remove(local_path)
                _still.log.info('Uploaded %s and freed local copy', filename)
            except Exception as e:
                _still.log.warning(
                    'Ckpt upload failed (keeping local): %s', e,
                )

        if self.cfg.ckpt_every_n_steps is not None:
            _every_n_steps = self.cfg.ckpt_every_n_steps

            class _StepCheckpoint(L.pytorch.callbacks.Callback):
                def on_train_batch_end(
                    self, trainer, pl_module, outputs, batch, batch_idx,
                ):
                    step = trainer.global_step
                    if step == 0 or step % _every_n_steps != 0:
                        return
                    epoch = trainer.current_epoch
                    filename = f'epoch={epoch:03d}-step={step:07d}.ckpt'
                    local_path = os.path.join(_ckpts_dir, filename)
                    trainer.save_checkpoint(local_path)
                    _still.log.info('Saved step checkpoint: %s', filename)
                    _upload_and_free(local_path, filename)

            callbacks.append(_StepCheckpoint())

        trainer_kwargs = dict(
            default_root_dir=ckpts_dir,
            max_epochs=self.cfg.max_epochs,
            limit_train_batches=self.max_steps,
            log_every_n_steps=self.cfg.log_every_n_steps,
            callbacks=callbacks,
            logger=logger,
            num_sanity_val_steps=0,
            **kwargs,
        )
        if self.cfg.val_dataloader is not None:
            trainer_kwargs['val_check_interval'] = self.cfg.val_every_n_steps
            trainer_kwargs['limit_val_batches'] = self.cfg.limit_val_batches
        else:
            trainer_kwargs['limit_val_batches'] = 0

        # Multi-GPU support
        if hasattr(self, 'devices') and self.devices is not None:
            trainer_kwargs['devices'] = self.devices
        elif hasattr(self, 'n_devices') and self.n_devices > 1:
            trainer_kwargs['devices'] = self.n_devices
            trainer_kwargs['strategy'] = 'ddp'

        trainer = L.pytorch.Trainer(**trainer_kwargs)

        original_precision = torch.get_float32_matmul_precision()
        if self.cfg.precision is not None:
            self.log.info(f'Setting precision to {repr(self.cfg.precision)}')
            torch.set_float32_matmul_precision(self.cfg.precision)

        try:
            datamodule = BitPathDataModule(
                self.cfg.dataloader,
                val_dataloader_builder=self.cfg.val_dataloader,
            )

            model = self.cfg.lightning.lightning_module
            ckpt = self.ckpt()
            fit_kwargs = {}
            if ckpt is not None:
                self.log.info(f'Resuming from checkpoint {ckpt}')
                fit_kwargs['ckpt_path'] = ckpt

            trainer.fit(
                model=model,
                datamodule=datamodule,
                **fit_kwargs,
            )

            # Mark training as complete
            marker_path = os.path.join(ckpts_dir, '_COMPLETE')
            with open(marker_path, 'w') as f:
                f.write(datetime.now().isoformat() + '\n')

            # Sync to remote storage
            self._sync_ckpts_to_remote(ckpts_dir)
            if self.save_remote_logs:
                self._sync_logs_to_remote(logs_dir)
        finally:
            atexit.unregister(_atexit_sync)
            try:
                self._sync_ckpts_to_remote(ckpts_dir)
            except Exception as e:
                self.log.warning('finally: ckpt sync failed: %s', e)
            if self.save_remote_logs:
                try:
                    self._sync_logs_to_remote(logs_dir)
                except Exception as e:
                    self.log.warning('finally: log sync failed: %s', e)
            torch.set_float32_matmul_precision(original_precision)

        return self

