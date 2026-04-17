"""
    > Background

        Contains code applying prov-gigapath foundation histopathology model to TCGA pancancer WSI data.
        The (prov-)gigapath model is described in https://www.nature.com/articles/s41586-024-07441-w
        Xu, H., Usuyama, N., Bagga, J. et al. "A whole-slide foundation model for digital pathology from real-world data.",
         Nature 630, 181–188 (2024). https://doi.org/10.1038/s41586-024-07441-w
        Code is available at https://github.com/prov-gigapath/prov-gigapath, 
        pretrained model can be obtained from HuggingFace: https://huggingface.co/prov-gigapath/prov-gigapath. 
"""

import argparse
import copy
from dataclasses import dataclass, asdict
import datetime
from functools import partial
import gc
import json
import logging
import math
import os
import pdb
import pickle
import sys
import time
import traceback as tb
from typing import List, Dict, Optional, Union, Tuple, Callable

import fsspec

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import scipy as sp
from scipy.sparse.linalg import svds as scipy_svds
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator


import ray
import torch
from torchvision import transforms

import timm

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from dinov2.models.vision_transformer import DinoVisionTransformer
#from dinov2.layers.attention import Attention
#from dinov2.layers import Mlp, PatchEmbed, NestedTensorBlock, DropPath

import dbx

from .augmentations import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, dino_tile_transform


GIGAPATH_BACKBONE_DEPTH = 40


def gigapath_tensor_block_class():
    from dinov2.layers import NestedTensorBlock
    class GigapathTensorBlock(NestedTensorBlock):
        def __init__(
            self,
            *,
            dim: int = 1536,
            num_heads: int = 24,
            init_values=1.0,
            drop_path: float,
            act_layer: Callable[..., nn.Module] = nn.SiLU,
            norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
            attn_class: Callable[..., nn.Module] = None,
        ) -> None:
            from dinov2.layers.layer_scale import LayerScale

            if attn_class is None:
                from timm.models.vision_transformer import Attention
                attn_class = Attention
            nn.Module.__init__(self)
            # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
            qkv_bias = True
            proj_bias = True
            ffn_bias = True
            drop = 0.0
            attn_drop = 0.0

            self.norm1 = norm_layer(dim)
            self.attn = attn_class(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            from timm.layers.mlp import GluMlp
            from dinov2.layers import DropPath
            
            self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = 2*4096
            self.mlp = GluMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                bias=ffn_bias,
                gate_last=False,
            )
            self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
            self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

            self.sample_drop_ratio = drop_path
    return GigapathTensorBlock


class GigapathVisionTransformer(DinoVisionTransformer):
        def __init__(
            self, 
            *,
            drop_path_rate: float = 0.0,
            drop_path_uniform: bool = False,
            block_cls: Callable = None,
            attention_class: Callable[..., nn.Module] = None,
        ):
            from dinov2.models.vision_transformer import BlockChunk
            from dinov2.layers import PatchEmbed
            if block_cls is None:
                block_cls = gigapath_tensor_block_class()
            if attention_class is None:
                from timm.models.vision_transformer import Attention
                attention_class = Attention

            nn.Module.__init__(self)
            img_size = 224
            patch_size = 16
            in_chans = 3
            embed_dim = 1536
            depth = GIGAPATH_BACKBONE_DEPTH
            num_heads = 24
            drop_path_rate = drop_path_rate
            drop_path_uniform = drop_path_uniform
            init_values = 1.0  # for layerscale: None or 0 => no layerscale
            embed_layer = PatchEmbed
            act_layer = nn.SiLU
            block_chunks = 1
            num_register_tokens = 0
            interpolate_antialias = False
            interpolate_offset = 0.1

            norm_layer = partial(nn.LayerNorm, eps=1e-6)

            self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
            self.num_tokens = 1
            self.n_blocks = depth
            self.num_heads = num_heads
            self.patch_size = patch_size
            self.num_register_tokens = num_register_tokens
            self.interpolate_antialias = interpolate_antialias
            self.interpolate_offset = interpolate_offset

            self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            num_patches = self.patch_embed.num_patches

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
            assert num_register_tokens >= 0
            self.register_tokens = (
                nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
            )

            if drop_path_uniform is True:
                dpr = [drop_path_rate] * depth
            else:
                dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

            blocks_list = [
                block_cls(
                    dim=1536,
                    num_heads=24,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    attn_class=attention_class,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
            if block_chunks > 0:
                self.chunked_blocks = True
                chunked_blocks = []
                chunksize = depth // block_chunks
                for i in range(0, depth, chunksize):
                    # this is to keep the block index consistent if we chunk the block list
                    chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
                self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
            else:
                self.chunked_blocks = False
                self.blocks = nn.ModuleList(blocks_list)

            self.norm = norm_layer(embed_dim)
            self.head = nn.Identity()

            self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

            self.init_weights()

        def prepare_tokens_with_masks(self, x, masks=None):
            """
                For training we cannot use the timm.VisionTransformer code, 
                but must use the original dinov2 code.
            """
            '''
            #timm:
            x = self.patch_embed(x)
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed
            return x
            '''
            B, nc, w, h = x.shape
            x = self.patch_embed(x)
            if masks is not None:
                x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.interpolate_pos_encoding(x, w, h)

            if self.register_tokens is not None:
                x = torch.cat(
                    (
                        x[:, :1],
                        self.register_tokens.expand(x.shape[0], -1, -1),
                        x[:, 1:],
                    ),
                    dim=1,
                )

            return x

            
def gigapath_vision_transformer_class():
    return GigapathVisionTransformer


def gigapath_tile_backbone(
    *, 
    type: str = 'dinov2', # or 'prov-gigapath'
    weights: str = None,
    cache: str = None,
    tile_encoder_snapshot: str = "8d2b1d2e65832e16bf9ff100a081acf6170a44ca",
    hf_token: str = "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", 
    device: str = 'cuda',
    resize: int = 256,
    center_crop: int = 224,
    drop_path_rate: float = 0.0,
    drop_path_uniform: bool = False,
    attention_class: Callable[..., nn.Module] = None,
    **kwargs,
):
    if attention_class is None:
        from dinov2.layers.attention import Attention
        attention_class = Attention
    os.environ['HF_TOKEN'] = hf_token
    if weights is None:
        if cache is None:
            cache = os.path.join(os.environ['HOME'], ".cache")
        weights = f"{cache}/huggingface/hub/models--prov-gigapath--prov-gigapath/snapshots/{tile_encoder_snapshot}/pytorch_model.bin"
    
    if type == 'prov-gigapath':
        model = timm.create_model(
                "hf_hub:prov-gigapath/prov-gigapath", 
                pretrained=(weights is None)
            )
        if weights is not None:
            td = torch.load(weights, map_location=device)
            model.load_state_dict(td, strict=True)
    elif type == 'dinov2':
        layerscale = 1.0e-5
        GigapathVisionTransformer = gigapath_vision_transformer_class()
        model = GigapathVisionTransformer(
            drop_path_rate=drop_path_rate, 
            drop_path_uniform=drop_path_uniform,
            attention_class=attention_class,
        )
        state_dict = None
        if weights is not None:
            state_dict = torch.load(weights, map_location=device)
            def chkey(key):
                prefix = 'blocks.'
                if key.startswith(prefix): 
                    chkey = 'blocks.0.' + key[len(prefix):]
                else:
                    chkey = key
                return chkey
            state_dict_ = {chkey(key): val for key, val in state_dict.items()}
            state_dict_['mask_token'] = torch.zeros(1, model.embed_dim)
            model.load_state_dict(state_dict_, strict=True)
    else:
        raise ValueError(f"Uknown model type {type}")
    model.to(device)
    model.eval()
    return model


def backbone_blocks(model):
    if isinstance(model.blocks[0], torch.nn.modules.container.ModuleList):
        blocks = model.blocks[0]
    else:
        blocks = model.blocks
    return blocks


class BackboneEvaluator:
    def __init__(self, 
        backbone=None,
        *,
        transform=None,
        device: str = 'cuda',
        log: dbx.Logger = dbx.Logger(stack_depth=3),
    ):
        self._backbone = backbone
        if self._backbone is None:
            self._backbone = "@autopath.gigaq.dinov2.backbone.gigapath_tile_backbone()"
        self.transform = transform
        if self.transform is None:
            self.transform = dino_tile_transform()
        self.device = device
        self.log = log

    @property
    def backbone(self):
        if isinstance(self._backbone, str):
            self.log.verbose(f"Evaluating {self._backbone} on {self.device}")
            self._backbone = dbx.eval(self._backbone).to(self.device)
        return self._backbone

    def to(self, device):
        self.device = device
        bkn = copy.deepcopy(self)
        bkn.device = device
        bkn._backbone = bkn.backbone.to(device)
        return bkn

    def eval(self):
        self.backbone.eval()
        return self

    def __pre_call__(self):
        pass

    def __call__(self, x):
        self.__pre_call__()
        with torch.no_grad():
            y = self.transform(x.to(self.device)) if self.transform is not None else x.to(self.device)
            z = self.backbone(y).cpu().detach()
            del y
            return z


class SidebandBackboneEvaluator(BackboneEvaluator):
    def __init__(self, 
        backbone=None,
        *,
        transform=None,
        capture_blocks: Optional[List[int]] = None,
        capture_layers: Optional[List[str]] = None, # ['patch_embed', 'norm', 'head', 'norm', 'backbone',]
        device: str = 'cuda',
    ):
        super().__init__(backbone, transform=transform, device=device)
        self.capture_blocks = capture_blocks
        self.capture_layers = capture_layers if capture_layers is not None else []
        self._sideband = None

    @property
    def sideband_layers(self):
        return self.capture_layers + (
            [] if self.capture_blocks is None else 
            [f"block.{b}" for b in self.capture_blocks]
        )

    def __pre_call__(self):
        _ = self.sideband 

    @property
    def sideband(self):
        if self._sideband is None:
            self.log.debug(f"Setting up sideband layer captures for layers {self.capture_layers} and blocks {self.capture_blocks}")
            self._sideband = {}
            def capture_layer(name):
                def hook(model, input, output):
                    self._sideband[f"{name}"] = output.cpu().detach()
                return hook
            
            blocks = backbone_blocks(self.backbone)
            if self.capture_blocks is not None:
                """
                blocks[l].norm1.register_forward_hook(capture_layer(f'block.{l}_norm1'))
                blocks[l].attn.qkv.register_forward_hook(capture_layer(f'block.{l}_attn_qkv'))
                blocks[l].attn.proj.register_forward_hook(capture_layer(f'block.{l}_attn_proj'))
                blocks[l].ls1.register_forward_hook(capture_layer(f'block.{l}_ls1'))
                blocks[l].norm2.register_forward_hook(capture_layer(f'block.{l}_norm2'))
                blocks[l].mlp.fc1.register_forward_hook(capture_layer(f'block.{l}_mlp_fc1'))
                blocks[l].mlp.act.register_forward_hook(capture_layer(f'block.{l}_mlp_act'))
                blocks[l].mlp.fc2.register_forward_hook(capture_layer(f'block.{l}_mlp_fc2'))
                blocks[l].mlp.drop1.register_forward_hook(capture_layer(f'block.{l}_mlp_drop1'))
                blocks[l].mlp.register_forward_hook(capture_layer(f'block.{l}_mlp'))
                blocks[l].mlp.register_forward_hook(capture_layer(f'block.{l}_ls2'))
                """
                for b in self.capture_blocks:
                    blocks[b].register_forward_hook(capture_layer(f'block.{b}'))
            for layer in self.capture_layers:
                if layer == 'backbone':
                    self.backbone.register_forward_hook(capture_layer(layer))
                else:
                    getattr(self.backbone, layer).register_forward_hook(capture_layer(layer))
            self.log.debug(f"Done setting up layer captures")
        return self._sideband
    
    def clear_sideband(self):
        if self._sideband is not None:
            for k in self._sideband.keys():
                self._sideband[k] = None 
            gc.collect()
            torch.cuda.empty_cache()
        return self


class SpectralProbe:
    """Estimates singular values of layer-to-layer Jacobians dh_{l+1}/dh_l.

    Treats each transformer block as a map F_l : h_l -> h_{l+1} and estimates
    the singular value spectrum of its Jacobian.  Two modes are available:

        * **cls** — materialises the d×d Jacobian restricted to the CLS token
          and computes a full dense SVD.  Fast and gives the complete spectrum.

        * **full** — uses JVP/VJP to define a LinearOperator for the full
          (Nd)×(Nd) Jacobian and calls scipy.sparse.linalg.svds for the top-k
          and bottom-k singular values.  Memory-efficient but slower.

    Args:
        backbone:       The GigapathVisionTransformer model (already on device).
        probe_blocks:   Which block indices to probe (e.g. [0, 10, 20, 30, 39]).
        k:              Number of extreme singular values to estimate in full mode.
        device:         Torch device string.
    """

    def __init__(
        self,
        backbone,
        probe_blocks: List[int],
        k: int = 10,
        device: str = 'cuda',
        log: dbx.Logger = dbx.Logger(name='SpectralProbe', stack_depth=3),
    ):
        self.backbone = backbone
        self.probe_blocks = probe_blocks
        self.k = k
        self.device = device
        self.log = log
        self._blocks = backbone_blocks(backbone)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _block_fn(self, block_idx: int):
        """Return a pure function h -> block(h) for a single block."""
        block = self._blocks[block_idx]
        def fn(h):
            return block(h)
        return fn

    # ------------------------------------------------------------------
    #  Mode 1: CLS-token-only  (d × d, fully materialisable)
    # ------------------------------------------------------------------

    def _probe_cls(self, block_idx: int, h: torch.Tensor) -> np.ndarray:
        """Materialise the d×d CLS-token Jacobian and return all singular values.

        Args:
            block_idx:  Index of the block.
            h:          Activation tensor, shape (B, N, d): 
                - B is batch size
                - N is number of tokens
                - d is embedding dimension
                Only the CLS token (index 0) of sample 0 from the batch is used.

        Returns:
            Singular values as a 1-D numpy array in decreasing order, length d.
        """
        block = self._blocks[block_idx]
        d = h.shape[-1]
        h0 = h[0:1].detach().clone().requires_grad_(True)          # (1, N, d)

        def cls_fn(cls_vec):
            """Replace the CLS token, run the block, return the output CLS token."""
            h_in = h0.clone()
            h_in[0, 0, :] = cls_vec
            h_out = block(h_in)
            return h_out[0, 0, :]                                  # (d,)

        J = torch.autograd.functional.jacobian(cls_fn, h0[0, 0, :].detach())   # (d, d)
        sv = torch.linalg.svdvals(J.float()).cpu().numpy()
        return sv

    # ------------------------------------------------------------------
    #  Mode 2: Full-sequence matrix-free  (Nd × Nd, via JVP / VJP)
    # ------------------------------------------------------------------

    def _probe_full(self, block_idx: int, h: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate top-k and bottom-k singular values of the full Jacobian.

        Uses torch.func.jvp / vjp wrapped in a scipy LinearOperator fed to
        scipy.sparse.linalg.svds.

        Args:
            block_idx:  Index of the block.
            h:          Activation tensor, shape (B, N, d).  Only sample 0 is used.

        Returns:
            (s_top, s_bot) — each a 1-D numpy array of length k.
        """
        block = self._blocks[block_idx]
        B, N, d = h.shape
        flat_dim = N * d
        h0 = h[0:1].detach()                                      # (1, N, d)
        dtype = h.dtype
        k = min(self.k, flat_dim - 1)                              # svds requires k < min(m,n)

        def flat_fn(flat_h):
            return block(flat_h.view(1, N, d)).view(flat_dim)

        def matvec(v):
            """J @ v  via JVP."""
            v_t = torch.tensor(v, device=self.device, dtype=dtype).view(flat_dim)
            _, jvp_out = torch.func.jvp(flat_fn, (h0.view(flat_dim),), (v_t,))
            return jvp_out.detach().cpu().numpy().astype(np.float64)

        def rmatvec(u):
            """J^T @ u  via VJP."""
            u_t = torch.tensor(u, device=self.device, dtype=dtype).view(flat_dim)
            _, vjp_fn = torch.func.vjp(flat_fn, h0.view(flat_dim))
            v = vjp_fn(u_t)[0]
            return v.detach().cpu().numpy().astype(np.float64)

        J_op = ScipyLinearOperator(
            shape=(flat_dim, flat_dim),
            matvec=matvec,
            rmatvec=rmatvec,
            dtype=np.float64,
        )

        # Top-k singular values
        try:
            _, s_top, _ = scipy_svds(J_op, k=k, which='LM')
            s_top = np.sort(s_top)[::-1]
        except Exception as e:
            self.log.warning(f"scipy_svds (LM) failed for block {block_idx}: {e}")
            s_top = np.full(k, np.nan)

        # Bottom-k singular values
        try:
            _, s_bot, _ = scipy_svds(J_op, k=k, which='SM')
            s_bot = np.sort(s_bot)
        except Exception as e:
            self.log.warning(f"svds (SM) failed for block {block_idx}: {e}")
            s_bot = np.full(k, np.nan)

        return s_top, s_bot

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def probe(
        self,
        activations: Dict[int, torch.Tensor],
        mode: str = 'cls',
    ) -> Dict[int, dict]:
        """Run the spectral probe on pre-captured activations.

        Args:
            activations:  Mapping from block index to activation tensor (B, N, d),
                          as captured by SidebandBackboneEvaluator hooks.
            mode:         ``'cls'`` for CLS-token-only, ``'full'`` for matrix-free,
                          or ``'both'`` to run both.

        Returns:
            Dict mapping block_idx -> result dict with keys depending on mode:
                cls:  {'singular_values': ndarray of shape (d,)}
                full: {'top_singular_values': ndarray, 'bottom_singular_values': ndarray,
                       'condition_number': float}
        """
        results = {}
        for block_idx in self.probe_blocks:
            if block_idx not in activations:
                self.log.warning(f"No activation captured for block {block_idx}, skipping")
                continue

            h = activations[block_idx]
            self.log.debug(f"Probing block {block_idx}, h.shape={tuple(h.shape)}, mode={mode}")
            entry = {}

            if mode in ('cls', 'both'):
                sv = self._probe_cls(block_idx, h)
                entry['singular_values'] = sv
                entry['log_singular_values'] = np.log(np.clip(sv, 1e-12, None))
                entry['condition_number_cls'] = float(sv[0] / (sv[-1] + 1e-12))

            if mode in ('full', 'both'):
                s_top, s_bot = self._probe_full(block_idx, h)
                entry['top_singular_values'] = s_top
                entry['bottom_singular_values'] = s_bot
                entry['condition_number_full'] = float(s_top[0] / (s_bot[0] + 1e-12))

            results[block_idx] = entry
        return results


class SpectralBackboneEvaluator(SidebandBackboneEvaluator):
    """Backbone evaluator that also probes the Jacobian singular-value spectrum.

    Subclasses SidebandBackboneEvaluator, reusing its forward-hook
    infrastructure to capture intermediate activations at the probed blocks.
    After each forward pass, a SpectralProbe is run on the captured
    activations to estimate the Jacobian spectrum.

    The spectral results are available via the ``spectral_results`` property
    after calling the evaluator.

    Args:
        backbone:             Backbone model (or string to lazy-eval).
        transform:            Image preprocessing transform.
        capture_blocks:       Block indices for SidebandBackboneEvaluator hooks.
        spectral_probe_blocks: Block indices to probe for Jacobian spectrum.
                              Defaults to ``capture_blocks`` if not provided.
        spectral_mode:        ``'cls'``, ``'full'``, or ``'both'``.
        spectral_k:           Number of extreme singular values for full mode.
        device:               Torch device string.
    """

    def __init__(
        self,
        backbone=None,
        *,
        transform=None,
        capture_blocks: Optional[List[int]] = None,
        capture_layers: Optional[List[str]] = None,
        spectral_probe_blocks: Optional[List[int]] = None,
        spectral_mode: str = 'cls',
        spectral_k: int = 10,
        device: str = 'cuda',
    ):
        # Ensure the spectral probe blocks are also captured by the sideband hooks
        spectral_probe_blocks = spectral_probe_blocks or capture_blocks or []
        if capture_blocks is None:
            capture_blocks = spectral_probe_blocks
        else:
            # Merge: sideband must capture at least the spectral blocks
            capture_blocks = sorted(set(capture_blocks) | set(spectral_probe_blocks))

        super().__init__(
            backbone,
            transform=transform,
            capture_blocks=capture_blocks,
            capture_layers=capture_layers,
            device=device,
        )
        self.spectral_probe_blocks = spectral_probe_blocks
        self.spectral_mode = spectral_mode
        self.spectral_k = spectral_k
        self._spectral_probe = None
        self._spectral_results = None

    @property
    def spectral_probe(self) -> SpectralProbe:
        if self._spectral_probe is None and self.spectral_probe_blocks:
            self._spectral_probe = SpectralProbe(
                backbone=self.backbone,
                probe_blocks=self.spectral_probe_blocks,
                k=self.spectral_k,
                device=self.device,
            )
        return self._spectral_probe

    def __call__(self, x):
        # Normal forward pass (populates self.sideband via hooks)
        z = super().__call__(x)

        # Run spectral probe on the captured activations
        if self.spectral_probe is not None and self._sideband:
            # Build activations dict from sideband: hook keys are "block.{idx}"
            activations = {}
            for b in self.spectral_probe_blocks:
                key = f"block.{b}"
                if key in self._sideband and self._sideband[key] is not None:
                    activations[b] = self._sideband[key].to(self.device)
            if activations:
                self._spectral_results = self.spectral_probe.probe(
                    activations, mode=self.spectral_mode,
                )

        return z

    @property
    def spectral_results(self) -> Optional[Dict[int, dict]]:
        """Spectral probe results from the last forward pass, keyed by block index."""
        return self._spectral_results

    def clear_spectral_results(self):
        self._spectral_results = None
        return self


# DEPRECATED: internalize attention capture in a BackboneEvaluator and remove
def gigapath_tile_backbone_with_sideband_and_preprocessor(
    *, 
    type: str = 'prov-gigapath',
    weights: str = None,
    cache: str = None,
    tile_encoder_snapshot: str = "8d2b1d2e65832e16bf9ff100a081acf6170a44ca",
    hf_token: str = "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", 
    device: str = 'cuda',
    resize: int = 256,
    center_crop: int = 224,
    **kwargs,
):
    model = gigapath_tile_backbone(
        type=type,
        weights=weights,
        cache=cache,
        tile_encoder_snapshot=tile_encoder_snapshot,
        hf_token=hf_token, 
        device=device,
        resize=resize,
        center_crop=center_crop,
        **kwargs,
    )

    # ---------------------------------------------------------------------
    num_features = 1536
    # This preprocessing, with resizing to 256 followed by
    # center crop to 224, is the same as the original Gigapath
    all_transforms = []

    if resize:
        all_transforms += [
            transforms.Resize(
                256 if resize is True else resize,
                interpolation=transforms.InterpolationMode.BICUBIC),
        ]
    if center_crop:
        all_transforms += [
            transforms.CenterCrop(
                224 if center_crop is True else center_crop),
        ]
    all_transforms += [
        transforms.Lambda(lambda x: x / 255.),
        transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD),
    ]
    transform = transforms.Compose(all_transforms)
    
    sideband = {}
    def capture_layer(name):
        def hook(model, input, output):
            sideband[f"{name}_input"] = input[0].detach()
            sideband[f"{name}"] = output.detach()
        return hook

    blocks = backbone_blocks(model)
    L = len(blocks)
    for l in range(L):
        blocks[l].norm1.register_forward_hook(capture_layer(f'B_{l}_norm1'))
        blocks[l].attn.qkv.register_forward_hook(capture_layer(f'B_{l}_attn_qkv'))
        blocks[l].attn.proj.register_forward_hook(capture_layer(f'B_{l}_attn_proj'))
        blocks[l].ls1.register_forward_hook(capture_layer(f'B_{l}_ls1'))
        blocks[l].norm2.register_forward_hook(capture_layer(f'B_{l}_norm2'))
        blocks[l].mlp.fc1.register_forward_hook(capture_layer(f'B_{l}_mlp_fc1'))
        blocks[l].mlp.act.register_forward_hook(capture_layer(f'B_{l}_mlp_act'))
        blocks[l].mlp.fc2.register_forward_hook(capture_layer(f'B_{l}_mlp_fc2'))
        blocks[l].mlp.drop1.register_forward_hook(capture_layer(f'B_{l}_mlp_drop1'))
        blocks[l].mlp.register_forward_hook(capture_layer(f'B_{l}_mlp'))
        blocks[l].mlp.register_forward_hook(capture_layer(f'B_{l}_ls2'))
    blocks[L-1].attn.q_norm.register_forward_hook(capture_layer(f'Q_{L-1}'))
    blocks[L-1].attn.k_norm.register_forward_hook(capture_layer(f'K_{L-1}'))
    model.patch_embed.register_forward_hook(capture_layer('patch_embed'))
    model.norm.register_forward_hook(capture_layer('norm'))
    model.head.register_forward_hook(capture_layer('head'))
    model.register_forward_hook(capture_layer('model'))
    return model, sideband, transform


def apply(backbone, sideband, transform, image, *, output_root: str = None, scale: bool = True):
    image_size = image.shape[1]
    timage = transform(image).to('cuda')
    output = backbone.cuda()(timage[None].cuda()).cpu().detach()
    timage = timage.cpu()
    sb = sideband

    blocks = backbone_blocks(backbone)
    L = len(blocks)
    attn = (sb[f'Q_{L-1}'].cpu()) @ (sb[f'K_{L-1}'].cpu().transpose(-2, -1))
    if scale:
        attn *= blocks[L-1].attn.scale 
    b, num_heads, num_patches_1, _ = attn.shape 
    map_size = int(np.sqrt(num_patches_1))
    
    attention_maps = {}
    for attention_head in range(num_heads):
        attention_map = attn[:,attention_head, 0, 1:]
        attention_map = attention_map.view(1, 1, map_size, map_size)
        attention_map = torch.nn.Upsample(size=(image_size, image_size))(attention_map)
        attention_map = attention_map[0, 0, :, :]
        attention_maps[(L-1, attention_head)] = attention_map.detach().cpu().numpy()
    if output_root is not None:
        os.makedirs(output_root, exist_ok=True)
        image_path = os.path.join(output_root, f"input.npz")
        with open(image_path, 'wb') as f:
            np.savez(f, image=image)
        output_path = os.path.join(output_root, f"output.npz")
        with open(output_path, 'wb') as f:
            np.savez(f, output=output)
        sideband_path = os.path.join(output_root, f"sideband.npz")
        sideband_ = dict(
            **{f"attention_map_{i}_{j}": attention_map for (i, j), attention_map in attention_maps.items()},
            **{k: v.cpu().detach().numpy() for k, v in sideband.items()},
        )
        with open(sideband_path, 'wb') as f:
            np.savez(f, **sideband_)
    return output, sideband_

        