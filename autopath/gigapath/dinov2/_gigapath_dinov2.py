from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from timm.layers.mlp import GluMlp
from timm.models.vision_transformer import Attention, Block
from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock
from dinov2.layers.layer_scale import LayerScale
from dinov2.models.vision_transformer import BlockChunk, DinoVisionTransformer


logger = logging.getLogger("gigapath_dinov2")


class GigapathBlock(NestedTensorBlock):
    def __init__(
        self,
        *,
        dim: int = 1536,
        num_heads: int = 24,
        init_values=1.0,
        drop_path: float,
        act_layer: Callable[..., nn.Module] = nn.SiLU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ) -> None:
        nn.Module.__init__(self)
        # print(f"biases: qkv: {qkv_bias}, proj: {proj_bias}, ffn: {ffn_bias}")
        qkv_bias = True
        proj_bias = True
        ffn_bias = True
        drop = 0.0
        attn_drop = 0.0

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
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


class GigapathVisionTransformer(DinoVisionTransformer):
    def __init__(
        self, 
        *,
        block_cls: Callable = GigapathBlock,
    ):
        nn.Module.__init__(self)
        img_size = 224
        patch_size = 16
        in_chans = 3
        embed_dim = 1536
        depth = 40
        num_heads = 24
        drop_path_rate = 0.0
        drop_path_uniform = False
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
            Make it match timm.VisionTransformer
        """
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        return x
        pos_embed2 = (pos_embed2 + fe2.pos_embed)
        
        B, nc, w, h = x.shape
        
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.pos_embed
        return x
