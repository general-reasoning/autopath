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
            self._backbone = "$autopath.gigapath.dinov2.backbone.gigapath_tile_backbone()"
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
    """GigaPath-specific spectral probe (backward-compat wrapper).

    Accepts a GigaPath backbone and unwraps its blocks via
    :func:`backbone_blocks`, then delegates to the model-agnostic
    :class:`autopath.gigaprobe.probes.SpectralProbe`.

    New code should use :class:`autopath.gigaprobe.probes.SpectralProbe`
    directly.
    """

    def __init__(
        self,
        backbone,
        probe_blocks: List[int],
        k: int = 10,
        device: str = 'cuda',
        log: dbx.Logger = dbx.Logger(name='SpectralProbe', stack_depth=3),
    ):
        from autopath.gigaprobe.probes import SpectralProbe as _GenericSpectralProbe
        blocks = list(backbone_blocks(backbone))
        self._delegate = _GenericSpectralProbe(
            blocks=blocks,
            probe_blocks=probe_blocks,
            k=k,
            device=device,
            log=log,
        )
        self.backbone = backbone
        self.probe_blocks = probe_blocks
        self.k = k
        self.device = device
        self.log = log

    def probe(self, activations, mode='cls'):
        return self._delegate.probe(activations, mode=mode)

    def _probe_cls(self, block_idx, h):
        return self._delegate._probe_cls(block_idx, h)

    def _probe_composed_cls(self, first_block, last_block, h_input):
        return self._delegate._probe_composed_cls(first_block, last_block, h_input)

    def _probe_full(self, block_idx, h):
        return self._delegate._probe_full(block_idx, h)


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
    def sideband(self):
        if self._sideband is None:
            # Let parent set up output-capture hooks
            _ = super().sideband
            # Also capture the *input* to the first probed block so we
            # can compute the composed Jacobian later.
            if self.spectral_probe_blocks:
                first_b = min(self.spectral_probe_blocks)
                blocks = backbone_blocks(self.backbone)
                def _capture_input(model, input, output):
                    self._sideband[f"block.{first_b}_input"] = input[0].cpu().detach()
                blocks[first_b].register_forward_hook(_capture_input)
        return self._sideband

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

            # Composed Jacobian (first → last probed block, CLS-only).
            # Computed directly via autograd through the chain of blocks;
            # does not depend on per-block spectral results.
            if len(self.spectral_probe_blocks) >= 2:
                first_b = min(self.spectral_probe_blocks)
                last_b = max(self.spectral_probe_blocks)
                input_key = f"block.{first_b}_input"
                if input_key in self._sideband and self._sideband[input_key] is not None:
                    h_input = self._sideband[input_key].to(self.device)
                    sv = self.spectral_probe._probe_composed_cls(
                        first_b, last_b, h_input,
                    )
                    self._spectral_results['composed'] = {
                        'singular_values': sv,
                        'log_singular_values': np.log(np.clip(sv, 1e-12, None)),
                        'condition_number_cls': float(sv[0] / (sv[-1] + 1e-12)),
                    }

        return z

    @property
    def spectral_results(self) -> Optional[Dict[int, dict]]:
        """Spectral probe results from the last forward pass, keyed by block index."""
        return self._spectral_results

    def clear_spectral_results(self):
        self._spectral_results = None
        return self



