"""Hydro: Latent vector to image decoder.

Reconstructs NxN RGB images from K-length latent vectors using a 
deconvolutional architecture.
"""
from dataclasses import dataclass
import functools
import math
import os
from typing import Callable, List, Optional, Tuple

import fsspec

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
import lightning.pytorch.loggers

import dbx
from dbx import Datablock

from autopath.databits import ClipDataLoaderBuilder
from .layers import UpLayer, ConvBlock


class SpatialAttentionGate(nn.Module):
    """Simple spatial attention mechanism to focus on sharp transitions."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(x)


class FeatureModulation(nn.Module):
    """Latent-conditioned feature modulation (simple scaling)."""
    def __init__(self, latent_dim: int, out_channels: int):
        super().__init__()
        self.scale_fc = nn.Linear(latent_dim, out_channels)
        
    def forward(self, x: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        scale = self.scale_fc(latent).unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + torch.tanh(scale))


class Hydro(Datablock):
    """Datablock wrapper for a latent-to-image decoder.
    
    Reconstructs NxN RGB images from K-length latent vectors using a 
    deconvolutional (transposed convolution) architecture.
    """

    @dataclass
    class CONFIG:
        # Input/output dimensions
        latent_dim: int = 1536          # K: length of input latent vectors
        image_size: int = 256           # N: output image resolution (NxN)
        
        # Architecture parameters
        initial_size: int = 8           # Starting spatial size before upsampling
        hidden_channels: int = 256      # Base channel count
        channel_multipliers: Tuple[int, ...] = (8, 4, 2, 1, 1)  # Channel mult per layer
        kernel_size: int = 3            # Convolution kernel size
        use_batch_norm: bool = True     # Whether to use batch normalization
        use_bilinear_upsampling: bool = True  # Use bilinear upsampling instead of transposed conv
        
        # Architectural improvements (optional)
        use_residual_upsampling: bool = False
        use_spatial_attention_gates: bool = False
        use_feature_modulation: bool = False
        
        # Loss parameters
        loss_type: str = "mse"          # Loss function: "mse", "l1", or "l0"

        # Model selection
        model: int = 0                  # Model architecture identifier

    class Model_0(nn.Module):
        """Inner nn.Module implementing the deconvolutional decoder architecture.
        
        Architecture: latent (B, K) → Linear → reshape → UpLayer × N → Conv2d → RGB
        """

        def __init__(
            self,
            *,
            latent_dim: int = 1536,
            image_size: int = 256,
            initial_size: int = 8,
            hidden_channels: int = 256,
            channel_multipliers: Tuple[int, ...] = (8, 4, 2, 1, 1),
            kernel_size: int = 3,
            use_batch_norm: bool = True,
            use_bilinear_upsampling: bool = False,
            use_residual_upsampling: bool = False,
            use_spatial_attention_gates: bool = False,
            use_feature_modulation: bool = False,
            loss_type: str = "mse",
            log: dbx.Logger = None,
        ):
            super().__init__()
            self.latent_dim = latent_dim
            self.image_size = image_size
            self.initial_size = initial_size
            self.hidden_channels = hidden_channels
            self.channel_multipliers = channel_multipliers
            self.kernel_size = kernel_size
            self.use_batch_norm = use_batch_norm
            self.use_bilinear_upsampling = use_bilinear_upsampling
            self.use_residual_upsampling = use_residual_upsampling
            self.use_spatial_attention_gates = use_spatial_attention_gates
            self.use_feature_modulation = use_feature_modulation
            self.loss_type = loss_type
            self.log = log or dbx.Logger(self.__class__.__name__)

            # Compute number of upsampling layers needed
            self.n_upsample = int(math.log2(image_size // initial_size))
            assert 2 ** self.n_upsample * initial_size == image_size, \
                f"image_size ({image_size}) must be initial_size ({initial_size}) * power of 2"
            
            # Extend channel_multipliers if needed
            if len(channel_multipliers) < self.n_upsample:
                channel_multipliers = channel_multipliers + (1,) * (self.n_upsample - len(channel_multipliers))
            self.channel_multipliers = channel_multipliers[:self.n_upsample]

            # Initial projection: latent → spatial feature map
            initial_channels = hidden_channels * self.channel_multipliers[0]
            self.initial_proj = nn.Linear(
                latent_dim, 
                initial_channels * initial_size * initial_size
            )
            self.initial_channels = initial_channels

            # Build upsampling layers
            self.up_layers = nn.ModuleList()
            in_channels = initial_channels
            for i in range(self.n_upsample):
                out_channels = hidden_channels * self.channel_multipliers[i]
                layer = UpLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    use_batch_norm=use_batch_norm,
                    use_skip_connection=False,
                    use_bilinear_upsampling=self.use_bilinear_upsampling,
                )
                self.up_layers.append(layer)

                if self.use_residual_upsampling:
                    if not hasattr(self, 'residual_projs'):
                        self.residual_projs = nn.ModuleList()
                    if in_channels != out_channels:
                        proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
                    else:
                        proj = nn.Identity()
                    self.residual_projs.append(proj)
                
                if self.use_feature_modulation:
                    if not hasattr(self, 'modulators'):
                        self.modulators = nn.ModuleList()
                    self.modulators.append(FeatureModulation(latent_dim, out_channels))
                
                if self.use_spatial_attention_gates:
                    if not hasattr(self, 'attention_gates'):
                        self.attention_gates = nn.ModuleList()
                    self.attention_gates.append(SpatialAttentionGate(out_channels))

                in_channels = out_channels

            # Final conv to RGB
            self.final_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=3,
                kernel_size=1,
            )

            self.log.debug(f"Built Hydro.Model_0: {latent_dim}→{image_size}x{image_size} "
                          f"with {self.n_upsample} upsample layers")

        def forward(self, latent: torch.Tensor) -> torch.Tensor:
            """Decode latent vectors to RGB images.
            
            Args:
                latent: Tensor of shape (B, K) containing latent vectors
                
            Returns:
                Tensor of shape (B, 3, N, N) containing reconstructed RGB images
            """
            batch_size = latent.shape[0]
            
            # Project and reshape to spatial
            x = self.initial_proj(latent)
            x = x.view(batch_size, self.initial_channels, self.initial_size, self.initial_size)
            x = F.relu(x)
            
            self.log.detailed(f"After initial projection: {x.shape}")

            # Progressive upsampling
            for i, up_layer in enumerate(self.up_layers):
                if self.use_residual_upsampling:
                    # Identity path: upsample input to match resolution
                    identity = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                    identity = self.residual_projs[i](identity)
                    
                    x = up_layer(x, skip_features=None)
                    x = x + identity
                else:
                    x = up_layer(x, skip_features=None)

                if self.use_feature_modulation:
                    x = self.modulators[i](x, latent)
                
                if self.use_spatial_attention_gates:
                    x = self.attention_gates[i](x)

                self.log.detailed(f"After up_layer {i}: {x.shape}")

            # Final conv to RGB + sigmoid to [0, 255]
            x = self.final_conv(x)
            x = torch.sigmoid(x) * 255.0
            self.log.detailed(f"After final_conv: {x.shape}")

            return x

        def loss(self, latent: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Compute reconstruction loss between predicted and target images.
            
            Args:
                latent: Tensor of shape (B, K) containing latent vectors
                target: Tensor of shape (B, 3, N, N) containing target RGB images
                
            Returns:
                Scalar tensor containing the loss value
            """
            predicted = self.forward(latent)
            
            if self.loss_type == "mse":
                loss = F.mse_loss(predicted, target)
            elif self.loss_type == "l1":
                loss = F.l1_loss(predicted, target)
            elif self.loss_type == "l0":
                # Differentiable proxy for L0: log(1 + (x-y)^2 / epsilon)
                epsilon = 1e-3
                loss = torch.mean(torch.log(1 + (predicted - target)**2 / epsilon))
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            self.log.detailed(f"Loss ({self.loss_type}): {loss.item():.6f}")
            return loss

    Model = {0: Model_0}

    def model(self) -> nn.Module:
        """Create and return the decoder model for the configured model architecture."""
        ModelClass = self.Model[self.cfg.model]
        return ModelClass(
            latent_dim=self.cfg.latent_dim,
            image_size=self.cfg.image_size,
            initial_size=self.cfg.initial_size,
            hidden_channels=self.cfg.hidden_channels,
            channel_multipliers=self.cfg.channel_multipliers,
            kernel_size=self.cfg.kernel_size,
            use_batch_norm=self.cfg.use_batch_norm,
            use_bilinear_upsampling=self.cfg.use_bilinear_upsampling,
            use_residual_upsampling=self.cfg.use_residual_upsampling,
            use_spatial_attention_gates=self.cfg.use_spatial_attention_gates,
            use_feature_modulation=self.cfg.use_feature_modulation,
            loss_type=self.cfg.loss_type,
            log=self.log,
        )


class HydroLightning(Datablock):
    """Lightning wrapper for training Hydro."""

    @dataclass
    class CONFIG:
        hydro: Hydro
        learning_rate: float = 1e-3
        scheduler: str = "cosine"
        log_images: bool = False,
        log_images_interval: int = 100,
        
        # Architectural improvements (optional, hoisted from Hydro)
        use_residual_upsampling: bool = False,
        use_spatial_attention_gates: bool = False,
        use_feature_modulation: bool = False,

    class Lightning(L.LightningModule):
        def __init__(
            self, 
            decoder: nn.Module, 
            learning_rate: float = 1e-3, 
            scheduler: str = "cosine",
            log_images: bool = False,
            log_images_interval: int = 100,
            log: dbx.Logger = None,
        ):
            super().__init__()
            self.decoder = decoder
            self.learning_rate = learning_rate
            self.scheduler = scheduler
            self.log_images = log_images
            self.log_images_interval = log_images_interval
            self.save_hyperparameters(ignore=['decoder'])
            self.log_ = log or dbx.Logger(name="HydroLightning")

        def training_step(self, batch, batch_idx):
            latents, targets = batch
            loss = self.decoder.loss(latents, targets)
            
            self.logger.experiment.add_scalar("Loss", loss, self.global_step)
            
            # --- Diagnostics: NaN/Inf checks ---
            if torch.isnan(loss) or torch.isinf(loss):
                self.log_.warning(f"[step {self.global_step}] Loss is {loss.item()}!")
            
            # --- Diagnostics: output stats ---
            with torch.no_grad():
                predicted = self.decoder(latents)
                self.logger.experiment.add_scalar("Output/min", predicted.min(), self.global_step)
                self.logger.experiment.add_scalar("Output/max", predicted.max(), self.global_step)
                self.logger.experiment.add_scalar("Output/mean", predicted.mean(), self.global_step)
                self.logger.experiment.add_scalar("Target/min", targets.min(), self.global_step)
                self.logger.experiment.add_scalar("Target/max", targets.max(), self.global_step)
                self.logger.experiment.add_scalar("Target/mean", targets.mean(), self.global_step)

                if torch.isnan(predicted).any():
                    self.log_.warning(f"[step {self.global_step}] NaN in model output!")
            
            scheduler = self.lr_schedulers()
            lr = scheduler.get_last_lr()[0]
            self.logger.experiment.add_scalar("Learning Rate", lr, self.global_step)
            
            if self.log_images and self.global_step % self.log_images_interval == 0:
                with torch.no_grad():
                    predicted = self.decoder(latents[:1])
                    target = targets[:1]
                    self.logger.experiment.add_image("Predicted", predicted[0].clamp(0, 255).to(torch.uint8), self.global_step)
                    self.logger.experiment.add_image("Target", target[0].to(torch.uint8), self.global_step)
            
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.learning_rate)
            stepping_batches = self.trainer.estimated_stepping_batches

            if self.scheduler == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=stepping_batches,
                    eta_min=1e-6
                )
            elif self.scheduler == "onecyclelr":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, 
                    max_lr=self.learning_rate, 
                    total_steps=stepping_batches
                )
            else:
                raise ValueError(f"Unknown scheduler: {self.scheduler}")

            self.log_.verbose(f"Using learning rate scheduler: {self.scheduler}")
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

    def __post_init__(self):
        self.hydro = self.cfg.hydro

    @functools.cached_property
    def lightning_module(self):
        # Propagate hoisted flags to hydro config if ellos son True
        if self.cfg.use_residual_upsampling:
            self.hydro.cfg.use_residual_upsampling = True
        if self.cfg.use_spatial_attention_gates:
            self.hydro.cfg.use_spatial_attention_gates = True
        if self.cfg.use_feature_modulation:
            self.hydro.cfg.use_feature_modulation = True

        return self.Lightning(
            decoder=self.hydro.model(),
            learning_rate=self.cfg.learning_rate,
            scheduler=self.cfg.scheduler,
            log_images=self.cfg.log_images,
            log_images_interval=self.cfg.log_images_interval,
            log=self.log,
        )


class HydroStill(Datablock):
    """Full training pipeline for Hydro, similar to VariationalReEncoderDecoderStill."""
    
    VERSION = 1
    TOPICFILES = {
        'logs': None,
        'ckpts': None,
    }

    @dataclass
    class CONFIG:
        lightning: HydroLightning
        dataloader: ClipDataLoaderBuilder  # DataLoader or DataLoaderBuilder with .dataloader() method
        init_ckpt_path_or_anchor: str = None
        from_scratch: bool = False
        load_optimizer_state: bool = False
        max_epochs: int = 1
        max_steps: int = 1000
        log_interval: int = 10
        gradient_clip_val: float = 1.0
        gradient_clip_algorithm: str = "norm"
        ckpt_every_n_steps: int = None,
        precision: str = None,
        
        # Architectural improvements (optional, hoisted from Hydro)
        use_residual_upsampling: bool = False,
        use_spatial_attention_gates: bool = False,
        use_feature_modulation: bool = False,

    def __init__(self, *args, n_devices: int = 1, logsroot: str = None, **kwargs):
        super().__init__(*args, n_devices=n_devices, logsroot=logsroot, **kwargs)

    @property
    def logs(self):
        return self.dirpath('logs', ensure=True)

    @property
    def logslink(self):
        return os.path.join(self.logsroot, self.tag) if self.logsroot is not None else None

    def __pre_build__(self):
        super().__pre_build__()
        self.linklogs()
        return self

    def valid(self):
        #TODO: Implement validation
        return False

    def UNSAFE_clear_logs(self, *, OVERRIDE: bool = False):
        """Clear all TensorBoard logs from the logs directory."""
        if not OVERRIDE:
            response = input("ARE YOU SURE YOU WANT TO EXECUTE 'UNSAFE_clear_logs'? [y/N]")
            if response.lower() != 'y':
                return self
        import shutil
        if os.path.exists(self.logs):
            shutil.rmtree(self.logs)
            self.log.info(f"Cleared TensorBoard logs at {self.logs}")
        os.makedirs(self.logs, exist_ok=True)
        self.linklogs()
        return self

    def linklogs(self):
        # Link the logs directory to the provided location (e.g., for Tensorboard to pick up the logs)
        if self.logslink is not None:
            self.log.verbose(f"---------------------- Linking logs to {self.logslink}----------------------------")
            
            # Create a symlink only if self.logslink does not already point to target
            if os.path.lexists(self.logslink):
                if os.path.islink(self.logslink) and os.readlink(self.logslink) == self.logs:
                    return self # Correct link already exists
                
                # It exists but is NOT the correct link.
                # We remove it ONLY now to allow creation of the correct link.
                # This avoids the unconditional removal that was previously at the start of the function.
                try:
                    os.remove(self.logslink)
                except Exception as e:
                    self.log.warning(f"Could not remove existing path at {self.logslink}: {e}")
                    return self

            # Create the symlink
            try:
                # Ensure parent directory exists (needed if tag contains slashes)
                os.makedirs(os.path.dirname(self.logslink), exist_ok=True)
                os.symlink(self.logs, self.logslink)
                self.log.debug(f"os.symlink({self.logs}, {self.logslink})")
            except Exception as e:
                self.log.warning(f"Failed to create symlink {self.logslink} -> {self.logs}: {e}")
        return self

    def ckpt(self):
        """Find the latest checkpoint file."""
        dirpath = self.dirpath('ckpts')
        ckptfs, _ = fsspec.url_to_fs(dirpath)
        files = [] if not ckptfs.exists(dirpath) else ckptfs.ls(dirpath)
        ckpts = [f for f in files if f.endswith('.ckpt') and 'step=' in f]
        steps = []
        for ckpt in ckpts:
            _, basename = os.path.split(ckpt)
            name, _ = basename.split('.')
            _, stepstr = name.split('step=')
            # Handle versioned checkpoints (e.g., '100-v1') by taking the first part
            step = int(stepstr.split('-')[0])
            steps.append(step)
        if len(steps) == 0:
            ckpt = None
        else:
            i = np.argmax(np.array(steps))
            ckpt = ckpts[i]
        return ckpt

    def __build__(self):
        import os
        
        # Propagate hoisted flags to lightning config if ellos son True
        if self.cfg.use_residual_upsampling:
            self.cfg.lightning.cfg.use_residual_upsampling = True
        if self.cfg.use_spatial_attention_gates:
            self.cfg.lightning.cfg.use_spatial_attention_gates = True
        if self.cfg.use_feature_modulation:
            self.cfg.lightning.cfg.use_feature_modulation = True

        logger = L.pytorch.loggers.TensorBoardLogger(
            save_dir=self.logs, 
            default_hp_metric=False, 
            name=self.tag
        )
        # Log the datablock quote to TensorBoard for traceability
        logger.experiment.add_text("HydroStill: quote", f"```python\n{self.bid.quote}\n```", global_step=0)
        logger.experiment.add_text("HydroStill: handle", f"```python\n{self.bid.deslash('handle')}\n```", global_step=0)
        
        default_root_dir = self.dirpath('ckpts')
        
        self.log.debug(f"Building trainer for {self.cfg.max_steps} steps using {self.cfg.lightning}")
        
        kwargs = {}
        if self.cfg.gradient_clip_val > 0.0:
            kwargs['gradient_clip_val'] = self.cfg.gradient_clip_val
            kwargs['gradient_clip_algorithm'] = self.cfg.gradient_clip_algorithm
            self.log.info(f"Using gradient clipping with value {self.cfg.gradient_clip_val} and algorithm {self.cfg.gradient_clip_algorithm}")
            
        callbacks = []
        if self.cfg.ckpt_every_n_steps is not None:
            callbacks.append(
                L.pytorch.callbacks.ModelCheckpoint(
                    dirpath=self.dirpath('ckpts'), 
                    every_n_train_steps=self.cfg.ckpt_every_n_steps
                )
            )

        trainer = L.pytorch.Trainer(
            default_root_dir=default_root_dir,
            max_epochs=self.cfg.max_epochs,
            limit_train_batches=self.cfg.max_steps,
            log_every_n_steps=self.cfg.log_interval,
            callbacks=callbacks,
            devices=self.n_devices,
            logger=logger,
            **kwargs,
        )

        self.log.debug(f"Built {trainer=} for lightning {self.cfg.lightning}")
        self.log.debug(f"Launching training for {self.cfg.max_steps=}")

        original_precision = torch.get_float32_matmul_precision()
        if self.cfg.precision is not None:
            self.log.info(f"Setting precision to {repr(self.cfg.precision)}")
            torch.set_float32_matmul_precision(self.cfg.precision)
            
        try:
            model = self.cfg.lightning.lightning_module
            resume = not self.cfg.from_scratch
            ckpt = None
            if resume:
                ckpt = self.ckpt()
            if ckpt is None:
                if self.cfg.init_ckpt_path_or_anchor is not None:
                    ckpath = self.cfg.init_ckpt_path_or_anchor
                    if ckpath.startswith('/'):  # TODO: support for fsspec urls
                        ckpt = ckpath
                    else:
                        ckpt = os.path.join(self.root, ckpath)
            fit_kwargs = {}
            if ckpt is not None:
                self.log.info(f"Using checkpoint {ckpt}")
                if not self.cfg.load_optimizer_state:
                    self.log.info(f"Skipping optimizer state from {ckpt}")
                    checkpoint = torch.load(ckpt, weights_only=False)
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                else:
                    fit_kwargs['ckpt_path'] = ckpt
            
            # Get dataloader - support both DataLoader and builders with .dataloader() method
            dataloader = self.cfg.dataloader
            if hasattr(dataloader, 'dataloader'):
                dataloader = dataloader.dataloader()
                
            trainer.fit(model=model, train_dataloaders=dataloader, **fit_kwargs)
        finally:
            torch.set_float32_matmul_precision(original_precision)
            
        return self
