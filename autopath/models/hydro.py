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
        # Enhanced to 3x3 for better spatial context
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(x)



class FeatureModulation(nn.Module):
    """Latent-conditioned feature modulation (simple scaling)."""
    def __init__(self, latent_dim: int, out_channels: int):
        super().__init__()
        self.scale_fc = nn.Sequential(
            nn.Linear(latent_dim, out_channels),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        scale = self.scale_fc(z).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class PerceptualLoss(nn.Module):
    """LPIPS-like perceptual loss using pre-trained VGG16 features."""
    def __init__(self):
        super().__init__()
        try:
            from torchvision import models
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        except (ImportError, AttributeError):
            from torchvision import models
            vgg = models.vgg16(pretrained=True).features
            
        self.slices = nn.ModuleList([
            nn.Sequential(*[vgg[x] for x in range(4)]),    # conv1_2
            nn.Sequential(*[vgg[x] for x in range(4, 9)]),  # conv2_2
            nn.Sequential(*[vgg[x] for x in range(9, 16)]), # conv3_3
            nn.Sequential(*[vgg[x] for x in range(16, 23)]),# conv4_3
            nn.Sequential(*[vgg[x] for x in range(23, 30)]) # conv5_3
        ])
        
        for param in self.parameters():
            param.requires_grad = False
            
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.eval()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Expected input range: [0, 1] for VGG. 
        # Hydro outputs [0, 255], so we must scale.
        x = x / 255.0
        y = y / 255.0
        
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        loss = 0
        feat_x = x
        feat_y = y
        for slice in self.slices:
            feat_x = slice(feat_x)
            feat_y = slice(feat_y)
            loss += F.mse_loss(feat_x, feat_y)
        return loss


class GradientLoss(nn.Module):
    """Loss function penalizing differences in image gradients (edges)."""
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x.repeat(3, 1, 1, 1))
        self.register_buffer('kernel_y', kernel_y.repeat(3, 1, 1, 1))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        grad_x_real = F.conv2d(x, self.kernel_x, groups=3, padding=1)
        grad_y_real = F.conv2d(x, self.kernel_y, groups=3, padding=1)
        grad_x_target = F.conv2d(y, self.kernel_x, groups=3, padding=1)
        grad_y_target = F.conv2d(y, self.kernel_y, groups=3, padding=1)
        
        loss = F.l1_loss(grad_x_real, grad_x_target) + F.l1_loss(grad_y_real, grad_y_target)
        return loss


class SSIMLoss(nn.Module):
    """Structural Similarity Index Measure (SSIM) loss."""
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        
        # Create 1D Gaussian kernel
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        
        # Create 2D Gaussian kernel
        g2d = g.view(1, -1) * g.view(-1, 1)
        kernel = g2d.view(1, 1, window_size, window_size).repeat(3, 1, 1, 1)
        self.register_buffer('kernel', kernel)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Expected input range: [0, 255] for Hydro outputs.
        # Scale to [0, 1] for SSIM calculation.
        x = x / 255.0
        y = y / 255.0
        
        mu_x = F.conv2d(x, self.kernel, groups=3, padding=self.window_size//2)
        mu_y = F.conv2d(y, self.kernel, groups=3, padding=self.window_size//2)
        
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x * x, self.kernel, groups=3, padding=self.window_size//2) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, self.kernel, groups=3, padding=self.window_size//2) - mu_y_sq
        sigma_xy = F.conv2d(x * y, self.kernel, groups=3, padding=self.window_size//2) - mu_xy
        
        c1 = 0.01**2
        c2 = 0.03**2
        
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / \
                   ((mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2))
        
        return 1 - ssim_map.mean()



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
        
        # Model_CNN parameters
        cnn_initial_size: int = 8           # Starting spatial size
        cnn_hidden_channels: int = 256      # Base channel count
        cnn_channel_multipliers: Tuple[int, ...] = (8, 4, 2, 1, 1)
        cnn_kernel_size: int = 3
        cnn_use_batch_norm: bool = True
        cnn_use_bilinear_upsampling: bool = True
        cnn_use_residual_upsampling: bool = False
        cnn_use_pixel_shuffle: bool = False
        cnn_use_residual: bool = False
        cnn_use_spatial_attention_gates: bool = False
        cnn_use_feature_modulation: bool = False
        
        # Loss parameters
        loss_type: str = "mse"              # Loss function
        ssim_companion_weight: float = 0.8  # Weight for companion loss when using SSIM
        ssim_companion_loss: str = "lpips"  # Companion loss type ('l1', 'mse', or 'lpips')
        ssim_kernel_width: int = 11         # Kernel width for SSIM calculation

        # Model selection
        model: str = 'cnn'                  # Model identifier

        # Model_ViT parameters
        vit_patch_size: int = 16
        vit_hidden_channels: int = 256
        vit_n_layers: int = 12
        vit_n_heads: int = 8
        vit_dim_feedforward: int = 1024

    class Model_CNN(nn.Module):
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
            cnn_use_pixel_shuffle: bool = False,
            cnn_use_residual: bool = False,
            use_residual_upsampling: bool = False,
            use_spatial_attention_gates: bool = False,
            use_feature_modulation: bool = False,
            loss_type: str = "mse",
            ssim_companion_weight: float = 0.8,
            ssim_companion_loss: str = "lpips",
            ssim_kernel_width: int = 11,
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
            self.cnn_use_pixel_shuffle = cnn_use_pixel_shuffle
            self.cnn_use_residual = cnn_use_residual
            self.use_residual_upsampling = use_residual_upsampling
            self.use_spatial_attention_gates = use_spatial_attention_gates
            self.use_feature_modulation = use_feature_modulation

            self.loss_type = loss_type
            self.ssim_companion_weight = ssim_companion_weight
            self.ssim_companion_loss = ssim_companion_loss
            self.ssim_kernel_width = ssim_kernel_width
            self.log = log or dbx.Logger(self.__class__.__name__)

            # Pre-initialize loss modules to avoid 'Unexpected key' errors when resuming
            if self.loss_type == "ssim":
                self._ssim_loss = SSIMLoss(window_size=self.ssim_kernel_width)
            if self.loss_type == "grad":
                self._grad_loss = GradientLoss()
            if self.loss_type == "lpips" or (self.loss_type == "ssim" and self.ssim_companion_loss == "lpips" and self.ssim_companion_weight > 0):
                self._lpips = PerceptualLoss()

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
                    cnn_use_pixel_shuffle=self.cnn_use_pixel_shuffle,
                    cnn_use_residual=self.cnn_use_residual,
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

            self.log.debug(f"Built Hydro.Model_CNN: {latent_dim}→{image_size}x{image_size} "
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
                # The user's snippet uses `torch.norm(pred - target, p=0) / pred.numel()`
                # which is a direct L0 count, not a differentiable proxy.
                # I will use the user's provided L0 implementation.
                loss = torch.norm(predicted - target, p=0) / predicted.numel()
            elif self.loss_type == "lpips":
                loss = self._lpips(predicted, target)
            elif self.loss_type == "grad":
                loss = self._grad_loss(predicted, target)
            elif self.loss_type == "ssim":
                loss = self._ssim_loss(predicted, target)
                if self.ssim_companion_weight > 0:
                    if self.ssim_companion_loss == "l1":
                        comp_loss = F.l1_loss(predicted, target)
                    elif self.ssim_companion_loss == "mse":
                        comp_loss = F.mse_loss(predicted, target)
                    elif self.ssim_companion_loss == "lpips":
                        comp_loss = self._lpips(predicted, target)
                    else:
                        raise ValueError(f"Unknown companion loss: {self.ssim_companion_loss}")
                    
                    loss = (1 - self.ssim_companion_weight) * loss + self.ssim_companion_weight * comp_loss
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            self.log.detailed(f"Loss ({self.loss_type}): {loss.item():.6f}")
            return loss

    class Model_ViT(nn.Module):
        """Inner nn.Module implementing the Vision Transformer (ViT) decoder architecture.
        
        Architecture: latent (B, K) → MLP Projection → Tokens (B, M, D) + Pos. Embed. 
                      → Transformer × L → Linear Head → Patch-to-Pixel → RGB (B, 3, N, N)
        """

        def __init__(
            self,
            *,
            latent_dim: int = 1536,
            image_size: int = 256,
            hidden_channels: int = 256,
            patch_size: int = 16,
            n_layers: int = 12,
            n_heads: int = 8,
            dim_feedforward: int = 1024,
            loss_type: str = "mse",
            ssim_companion_weight: float = 0.8,
            ssim_companion_loss: str = "lpips",
            ssim_kernel_width: int = 11,
            log: dbx.Logger = None,
        ):
            super().__init__()
            self.latent_dim = latent_dim
            self.image_size = image_size
            self.hidden_channels = hidden_channels
            self.patch_size = patch_size
            self.loss_type = loss_type
            self.ssim_companion_weight = ssim_companion_weight
            self.ssim_companion_loss = ssim_companion_loss
            self.ssim_kernel_width = ssim_kernel_width
            self.log = log or dbx.Logger(self.__class__.__name__)

            # Pre-initialize loss modules to avoid 'Unexpected key' errors when resuming
            if self.loss_type == "ssim":
                self._ssim_loss = SSIMLoss(window_size=self.ssim_kernel_width)
            if self.loss_type == "grad":
                self._grad_loss = GradientLoss()
            if self.loss_type == "lpips" or (self.loss_type == "ssim" and self.ssim_companion_loss == "lpips" and self.ssim_companion_weight > 0):
                self._lpips = PerceptualLoss()

            self.n_patches = image_size // patch_size
            self.m_tokens = self.n_patches ** 2

            # 1. MLP Projection: latent -> sequence of tokens
            # We use a single linear layer to produce the sequence elements
            self.proj = nn.Linear(latent_dim, self.m_tokens * hidden_channels)

            # 2. Positional Embedding
            self.pos_embed = nn.Parameter(torch.zeros(1, self.m_tokens, hidden_channels))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

            # 3. Transformer Backbone
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_channels,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                activation='gelu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

            # 4. Reconstruction Head: token -> RGB patch
            self.head = nn.Linear(hidden_channels, 3 * patch_size * patch_size)

            self.log.debug(f"Built Hydro.Model_ViT: {latent_dim}→{image_size}x{image_size} "
                          f"ViT with {n_layers} layers and {patch_size}x{patch_size} patches")

        def forward(self, latent: torch.Tensor) -> torch.Tensor:
            """Decode latent vectors to RGB images.
            
            Args:
                latent: Tensor of shape (B, K) containing latent vectors
                
            Returns:
                Tensor of shape (B, 3, N, N) containing reconstructed RGB images
            """
            batch_size = latent.shape[0]
            
            # Project to token sequence: (B, K) -> (B, M, D)
            x = self.proj(latent)
            x = x.view(batch_size, self.m_tokens, self.hidden_channels)
            
            # Add positional embedding
            x = x + self.pos_embed
            
            # Transformer backbone
            x = self.transformer(x) # (B, M, D)
            
            # Head: (B, M, D) -> (B, M, 3*p*p)
            x = self.head(x)
            
            # Patch-to-Pixel reconstruction
            # (B, M, 3*p*p) -> (B, n_patches, n_patches, 3, p, p)
            p = self.patch_size
            n = self.n_patches
            x = x.view(batch_size, n, n, 3, p, p)
            
            # Permute to (B, 3, n*p, n*p) which is (B, 3, N, N)
            x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
            x = x.view(batch_size, 3, self.image_size, self.image_size)
            
            # Final sigmoid to [0, 255]
            x = torch.sigmoid(x) * 255.0
            
            return x

        def loss(self, latent: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            """Compute reconstruction loss."""
            predicted = self.forward(latent)
            
            if self.loss_type == "mse":
                loss = F.mse_loss(predicted, target)
            elif self.loss_type == "l1":
                loss = F.l1_loss(predicted, target)
            elif self.loss_type == "l0":
                loss = torch.norm(predicted - target, p=0) / predicted.numel()
            elif self.loss_type == "lpips":
                loss = self._lpips(predicted, target)
            elif self.loss_type == "ssim":
                loss = self._ssim_loss(predicted, target)
                if self.ssim_companion_weight > 0:
                    if self.ssim_companion_loss == "l1":
                        comp_loss = F.l1_loss(predicted, target)
                    elif self.ssim_companion_loss == "mse":
                        comp_loss = F.mse_loss(predicted, target)
                    elif self.ssim_companion_loss == "lpips":
                        comp_loss = self._lpips(predicted, target)
                    else:
                        raise ValueError(f"Unknown companion loss: {self.ssim_companion_loss}")
                    
                    loss = (1 - self.ssim_companion_weight) * loss + self.ssim_companion_weight * comp_loss
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            self.log.detailed(f"Loss ({self.loss_type}): {loss.item():.6f}")
            return loss

    Model = {'cnn': Model_CNN, 'vit': Model_ViT}

    def model(self) -> nn.Module:
        """Create and return the decoder model for the configured model architecture."""
        ModelClass = self.Model[self.cfg.model]
        if self.cfg.model == 'cnn':
            return ModelClass(
                latent_dim=self.cfg.latent_dim,
                image_size=self.cfg.image_size,
                initial_size=self.cfg.cnn_initial_size,
                hidden_channels=self.cfg.cnn_hidden_channels,
                channel_multipliers=self.cfg.cnn_channel_multipliers,
                kernel_size=self.cfg.cnn_kernel_size,
                use_batch_norm=self.cfg.cnn_use_batch_norm,
                use_bilinear_upsampling=self.cfg.cnn_use_bilinear_upsampling,
                cnn_use_pixel_shuffle=self.cfg.cnn_use_pixel_shuffle,
                cnn_use_residual=self.cfg.cnn_use_residual,
                use_residual_upsampling=self.cfg.cnn_use_residual_upsampling,
                use_spatial_attention_gates=self.cfg.cnn_use_spatial_attention_gates,
                use_feature_modulation=self.cfg.cnn_use_feature_modulation,
                loss_type=self.cfg.loss_type,
                ssim_companion_weight=self.cfg.ssim_companion_weight,
                ssim_companion_loss=self.cfg.ssim_companion_loss,
                ssim_kernel_width=self.cfg.ssim_kernel_width,
                log=self.log,
            )

        elif self.cfg.model == 'vit':
            return ModelClass(
                latent_dim=self.cfg.latent_dim,
                image_size=self.cfg.image_size,
                hidden_channels=self.cfg.vit_hidden_channels,
                patch_size=self.cfg.vit_patch_size,
                n_layers=self.cfg.vit_n_layers,
                n_heads=self.cfg.vit_n_heads,
                dim_feedforward=self.cfg.vit_dim_feedforward,
                loss_type=self.cfg.loss_type,
                ssim_companion_weight=self.cfg.ssim_companion_weight,
                ssim_companion_loss=self.cfg.ssim_companion_loss,
                ssim_kernel_width=self.cfg.ssim_kernel_width,
                log=self.log,
            )
        else:
            raise ValueError(f"Unknown model identifier: {self.cfg.model}")


class HydroLightning(Datablock):
    """Lightning wrapper for training Hydro."""

    @dataclass
    class CONFIG:
        hydro: Hydro
        learning_rate: float = 1e-3
        scheduler: str = "cosine"
        log_images: bool = False
        log_images_interval: int = 100
        
        # Architectural improvements (optional, hoisted from Hydro)
        cnn_use_residual_upsampling: bool = False
        cnn_use_pixel_shuffle: bool = False
        cnn_use_residual: bool = False
        cnn_use_spatial_attention_gates: bool = False
        cnn_use_feature_modulation: bool = False
        
        # SSIM parameters (optional, hoisted from Hydro)
        ssim_companion_weight: Optional[float] = None
        ssim_companion_loss: Optional[str] = None
        strict_loading: bool = True

    class Lightning(L.LightningModule):

        def __init__(
            self, 
            decoder: nn.Module, 
            learning_rate: float = 1e-3, 
            scheduler: str = "cosine",
            log_images: bool = False,
            log_images_interval: int = 100,
            strict_loading: bool = True,
            log: dbx.Logger = None,
        ):
            super().__init__()
            self.decoder = decoder
            self.learning_rate = learning_rate
            self.scheduler = scheduler
            self.log_images = log_images
            self.log_images_interval = log_images_interval
            self.strict_loading = strict_loading
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
        # Propagate hoisted flags to hydro config if ellos son True
        if self.cfg.cnn_use_residual_upsampling:
            self.hydro.cfg.cnn_use_residual_upsampling = True
        if self.cfg.cnn_use_pixel_shuffle:
            self.hydro.cfg.cnn_use_pixel_shuffle = True
        if self.cfg.cnn_use_residual:
            self.hydro.cfg.cnn_use_residual = True
        if self.cfg.cnn_use_spatial_attention_gates:
            self.hydro.cfg.cnn_use_spatial_attention_gates = True
        if self.cfg.cnn_use_feature_modulation:
            self.hydro.cfg.cnn_use_feature_modulation = True
        
        if self.cfg.ssim_companion_weight is not None:
            self.hydro.cfg.ssim_companion_weight = self.cfg.ssim_companion_weight
        if self.cfg.ssim_companion_loss is not None:
            self.hydro.cfg.ssim_companion_loss = self.cfg.ssim_companion_loss


    @functools.cached_property
    def lightning_module(self):
        return self.Lightning(
            decoder=self.hydro.model(),
            learning_rate=self.cfg.learning_rate,
            scheduler=self.cfg.scheduler,
            log_images=self.cfg.log_images,
            log_images_interval=self.cfg.log_images_interval,
            strict_loading=self.cfg.strict_loading,
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
        initial_ckpt: str = None
        from_scratch: bool = False
        reset_optimizer_state: bool = False
        max_epochs: int = 1
        max_steps: int = 1000
        log_interval: int = 10
        gradient_clip_val: float = 1.0
        gradient_clip_algorithm: str = "norm"
        ckpt_every_n_steps: int = None
        precision: str = None
        strict_loading: bool = False
        
        # Architectural improvements (optional, hoisted from Hydro)
        cnn_use_residual_upsampling: bool = False
        cnn_use_pixel_shuffle: bool = False
        cnn_use_residual: bool = False
        cnn_use_spatial_attention_gates: bool = False
        cnn_use_feature_modulation: bool = False
        
        # SSIM parameters (optional, hoisted from Hydro)
        ssim_companion_weight: Optional[float] = None
        ssim_companion_loss: Optional[str] = None


    def __init__(self, *args, n_devices: int = 1, logsroot: str = None, **kwargs):
        super().__init__(*args, n_devices=n_devices, logsroot=logsroot, **kwargs)
        if isinstance(self.cfg.max_steps, str):
            if self.cfg.max_steps.endswith('%'):
                total_n = len(self.cfg.dataloader)
                self.max_steps = int(total_n * float(self.cfg.max_steps.strip('%')) / 100)
                self.log.info(f"Computed max_steps={self.max_steps} from {self.cfg.max_steps} of {total_n}")
            else:
                self.max_steps = int(self.cfg.max_steps)
                self.log.info(f"Using max_steps={self.max_steps}")
        else:
            self.max_steps = self.cfg.max_steps
            self.log.info(f"Using max_steps={self.max_steps}")

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
        """Check whether the necessary number of checkpoints is present in the ckpts directory."""
        if self.max_steps is None or self.cfg.ckpt_every_n_steps is None or self.cfg.ckpt_every_n_steps <= 0:
            return False

        # Total expected number of checkpoints based on total training steps
        # total_steps = max_epochs * max_steps (since limit_train_batches is set to max_steps)
        
        expected_n = (self.cfg.max_epochs * self.max_steps) // self.cfg.ckpt_every_n_steps
        if expected_n <= 0:
            return False

        dirpath = self.dirpath('ckpts')
        ckptfs, _ = fsspec.url_to_fs(dirpath)
        files = [] if not ckptfs.exists(dirpath) else ckptfs.ls(dirpath)
        ckpts = [f for f in files if f.endswith('.ckpt') and 'step=' in f]
        
        steps = []
        for ckpt in ckpts:
            try:
                _, basename = os.path.split(ckpt)
                # Split by '.' and take the first part to get 'step=100-v1' from 'step=100-v1.ckpt'
                name = basename.split('.ckpt')[0]
                if 'step=' not in name:
                    continue
                _, stepstr = name.split('step=')
                # Handle versioned checkpoints (e.g., '100-v1') by taking the first part
                step = int(stepstr.split('-')[0])
                steps.append(step)
            except (ValueError, IndexError):
                continue
        
        unique_steps = set(steps)
        # Verify that we have at least the expected number of unique step checkpoints
        return len(unique_steps) >= expected_n

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
            self.log.verbose(f"---------------------- Linking logs to {self.logslink} ----------------------------")
            
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
        if self.cfg.cnn_use_residual_upsampling:
            self.cfg.lightning.cfg.cnn_use_residual_upsampling = True
        if self.cfg.cnn_use_pixel_shuffle:
            self.cfg.lightning.cfg.cnn_use_pixel_shuffle = True
        if self.cfg.cnn_use_residual:
            self.cfg.lightning.cfg.cnn_use_residual = True
        if self.cfg.cnn_use_spatial_attention_gates:
            self.cfg.lightning.cfg.cnn_use_spatial_attention_gates = True
        if self.cfg.cnn_use_feature_modulation:
            self.cfg.lightning.cfg.cnn_use_feature_modulation = True
            
        if self.cfg.ssim_companion_weight is not None:
            self.cfg.lightning.cfg.ssim_companion_weight = self.cfg.ssim_companion_weight
        if self.cfg.ssim_companion_loss is not None:
            self.cfg.lightning.cfg.ssim_companion_loss = self.cfg.ssim_companion_loss
            
        if self.cfg.strict_loading is not None:
            self.cfg.lightning.cfg.strict_loading = self.cfg.strict_loading


        logger = L.pytorch.loggers.TensorBoardLogger(
            save_dir=self.logs, 
            default_hp_metric=False, 
            name="",
        )

        # Log the datablock metadata to TensorBoard for traceability
        logger.experiment.add_text("HydroStill: anchorhashpath", f"```python\n{self.anchorhashpath}\n```", global_step=0)
        logger.experiment.add_text("HydroStill: dfn", f"```python\n{self.dfn}\n```", global_step=0)
        
        default_root_dir = self.dirpath('ckpts')
        
        self.log.verbose(f"Building trainer for {self.max_steps} steps using {self.cfg.lightning}")
        
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
            limit_train_batches=self.max_steps,
            log_every_n_steps=self.cfg.log_interval,
            callbacks=callbacks,
            devices=self.n_devices,
            logger=logger,
            **kwargs,
        )

        self.log.debug(f"Built {trainer=} for lightning {self.cfg.lightning}")
        self.log.debug(f"Launching training for {self.max_steps=}")

        original_precision = torch.get_float32_matmul_precision()
        if self.cfg.precision is not None:
            self.log.info(f"Setting precision to {repr(self.cfg.precision)}")
            torch.set_float32_matmul_precision(self.cfg.precision)
            
        try:
            model = self.cfg.lightning.lightning_module
            resume = not self.cfg.from_scratch
            ckpt = None
            if resume:
                self.log.info(f"Resuming training from {self.dirpath('ckpts')}")
                ckpt = self.ckpt()
            if ckpt is None:
                if self.cfg.initial_ckpt is not None:
                    self.log.info(f"Using initial checkpoint {self.cfg.initial_ckpt}")
                    ckpt = self.cfg.initial_ckpt
                    if ckpt.startswith('$'):
                        ckptblock = dbx.eval_term(ckpt)
                        ckpath = ckptblock.ckpt()
                    else:
                        ckpath = ckpt
                    if ckpath.startswith('/'):  # TODO: support for fsspec urls
                        ckpt = ckpath
                    else:
                        ckpt = os.path.join(self.root, ckpath)
            fit_kwargs = {}
            if ckpt is not None:
                self.log.info(f"Using checkpoint {ckpt}")
                if self.cfg.reset_optimizer_state:
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
