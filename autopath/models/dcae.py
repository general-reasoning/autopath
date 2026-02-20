

# This code is from dcae repo
# https://github.com/mit-han-lab/efficientvit/tree/master/applications/dc_ae


import functools
import os
from typing import Optional, Tuple, Union, Any, List
from dataclasses import dataclass
import itertools
import omegaconf


import torch
import torch.nn as nn
import torch.nn.functional as F


import numbers
from collections import OrderedDict
from dataclasses import fields, is_dataclass


import fsspec
import numpy as np

import lightning as L
import lightning.pytorch.loggers

import dbx
from dbx import Datablock

from autopath.databits import ClipDataLoaderBuilder

from einops import rearrange
from helm.third_party_tools.control_net.ldm.modules.distributions.distributions import (
   DiagonalGaussianDistribution,
)
from helm.conversion import onnx_safe_ops


# --- Stubs for HELM-specific types used by DCAutoencoder ---
# These are placeholders to allow the module to load without the full HELM dependency.
# Replace with actual imports when integrating with a HELM training pipeline.

def instantiate_from_config(config):
    """Stub: instantiate an object from an omegaconf config."""
    if hasattr(config, '_target_'):
        import importlib
        module_path, cls_name = config._target_.rsplit('.', 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, cls_name)
        params = {k: v for k, v in config.items() if k != '_target_'}
        return cls(**params)
    raise NotImplementedError(f"Cannot instantiate from config: {config}")


class _StubNamedTuple:
    """Minimal stub base for HELM data types."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def __repr__(self):
        attrs = ', '.join(f'{k}={v!r}' for k, v in self.__dict__.items())
        return f'{self.__class__.__name__}({attrs})'

class Field(_StubNamedTuple):
    def __init__(self, *, name: str = '', **kwargs):
        self.name = name
        super().__init__(**kwargs)

class RgbImageTensor(_StubNamedTuple): pass

class NetworkEvaluation(_StubNamedTuple):
    class NetworkField(_StubNamedTuple): pass
    def __init__(self, *, input_fields=None, output_fields=None, **kwargs):
        self.input_fields = input_fields or []
        self.output_fields = output_fields or []
        super().__init__(**kwargs)

class Transformation(_StubNamedTuple): pass
class TensorSpec(_StubNamedTuple): pass
class TensorSpecFields(_StubNamedTuple): pass

class EncoderOutput(_StubNamedTuple): pass
class DecoderOutput(_StubNamedTuple): pass




def is_tensor(x) -> bool:
   """
   Tests if `x` is a `torch.Tensor` or `np.ndarray`.
   """
   if isinstance(x, torch.Tensor):
       return True


   return isinstance(x, np.ndarray)




class BaseOutput(OrderedDict):
   """
   Base class for all model outputs as dataclass. Has a `__getitem__` that allows indexing by integer or slice (like a
   tuple) or strings (like a dictionary) that will ignore the `None` attributes. Otherwise behaves like a regular
   Python dictionary.


   <Tip warning={true}>


   You can't unpack a [`BaseOutput`] directly. Use the [`~utils.BaseOutput.to_tuple`] method to convert it to a tuple
   first.


   </Tip>
   """


   def __init_subclass__(cls) -> None:
       """Register subclasses as pytree nodes.


       This is necessary to synchronize gradients when using `torch.nn.parallel.DistributedDataParallel` with
       `static_graph=True` with modules that output `ModelOutput` subclasses.
       """
       import torch.utils._pytree


       torch.utils._pytree.register_pytree_node(
           cls,
           torch.utils._pytree._dict_flatten,
           lambda values, context: cls(**torch.utils._pytree._dict_unflatten(values, context)),
       )


       # torch.utils._pytree.register_pytree_node(
       #     cls,
       #     torch.utils._pytree._dict_flatten,
       #     lambda values, context: cls(**torch.utils._pytree._dict_unflatten(values, context)),
       # )


   def __post_init__(self) -> None:
       class_fields = fields(self)


       # Safety and consistency checks
       if not len(class_fields):
           raise ValueError(f"{self.__class__.__name__} has no fields.")


       first_field = getattr(self, class_fields[0].name)
       other_fields_are_none = all(getattr(self, field.name) is None for field in class_fields[1:])


       if other_fields_are_none and isinstance(first_field, dict):
           for key, value in first_field.items():
               self[key] = value
       else:
           for field in class_fields:
               v = getattr(self, field.name)
               if v is not None:
                   self[field.name] = v


   def __delitem__(self, *args, **kwargs):
       raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")


   def setdefault(self, *args, **kwargs):
       raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")


   def pop(self, *args, **kwargs):
       raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")


   def update(self, *args, **kwargs):
       raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")


   def __getitem__(self, k: Any) -> Any:
       if isinstance(k, str):
           inner_dict = dict(self.items())
           return inner_dict[k]
       else:
           return self.to_tuple()[k]


   def __setattr__(self, name: Any, value: Any) -> None:
       if name in self.keys() and value is not None:
           # Don't call self.__setitem__ to avoid recursion errors
           super().__setitem__(name, value)
       super().__setattr__(name, value)


   def __setitem__(self, key, value):
       # Will raise a KeyException if needed
       super().__setitem__(key, value)
       # Don't call self.__setattr__ to avoid recursion errors
       super().__setattr__(key, value)


   def __reduce__(self):
       if not is_dataclass(self):
           return super().__reduce__()
       callable, _args, *remaining = super().__reduce__()
       args = tuple(getattr(self, field.name) for field in fields(self))
       return callable, args, *remaining


   def to_tuple(self) -> Tuple[Any, ...]:
       """
       Convert self to a tuple containing all the attributes/keys that are not `None`.
       """
       return tuple(self[k] for k in self.keys())




class RMSNorm(nn.Module):
   def __init__(self, dim, eps: float, elementwise_affine: bool = True, bias: bool = False):
       super().__init__()


       self.eps = eps
       self.elementwise_affine = elementwise_affine


       if isinstance(dim, numbers.Integral):
           dim = (dim,)


       self.dim = torch.Size(dim)


       self.weight = None
       self.bias = None


       if elementwise_affine:
           self.weight = nn.Parameter(torch.ones(dim))
           if bias:
               self.bias = nn.Parameter(torch.zeros(dim))


   def forward(self, hidden_states):
       input_dtype = hidden_states.dtype
       variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
       hidden_states = hidden_states * torch.rsqrt(variance + self.eps)


       if self.weight is not None:
           # convert into half-precision if necessary
           if self.weight.dtype in [torch.float16, torch.bfloat16]:
               hidden_states = hidden_states.to(self.weight.dtype)
           hidden_states = hidden_states * self.weight
           if self.bias is not None:
               hidden_states = hidden_states + self.bias
       else:
           hidden_states = hidden_states.to(input_dtype)


       return hidden_states




def get_normalization(
   norm_type: str = "batch_norm",
   num_features: Optional[int] = None,
   eps: float = 1e-5,
   elementwise_affine: bool = True,
   bias: bool = True,
) -> nn.Module:
   if norm_type == "rms_norm":
       norm = RMSNorm(num_features, eps=eps, elementwise_affine=elementwise_affine, bias=bias)
   elif norm_type == "layer_norm":
       norm = nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine, bias=bias)
   elif norm_type == "batch_norm":
       norm = nn.BatchNorm2d(num_features, eps=eps, affine=elementwise_affine)
   else:
       raise ValueError(f"{norm_type=} is not supported.")
   return norm




class SanaMultiscaleAttnProcessor2_0:
   r"""
   Processor for implementing multiscale quadratic attention.
   """


   def __call__(self, attn, hidden_states: torch.Tensor) -> torch.Tensor:
       height, width = hidden_states.shape[-2:]
       if height * width > attn.attention_head_dim:
           use_linear_attention = True
       else:
           use_linear_attention = False


       residual = hidden_states


       batch_size, _, height, width = list(hidden_states.size())
       original_dtype = hidden_states.dtype


       hidden_states = onnx_safe_ops.movedim(hidden_states, 1, -1)
       query = attn.to_q(hidden_states)
       key = attn.to_k(hidden_states)
       value = attn.to_v(hidden_states)
       hidden_states = torch.cat([query, key, value], dim=3)
       hidden_states = onnx_safe_ops.movedim(hidden_states, -1, 1)


       multi_scale_qkv = [hidden_states]
       for block in attn.to_qkv_multiscale:
           multi_scale_qkv.append(block(hidden_states))


       hidden_states = torch.cat(multi_scale_qkv, dim=1)


       if use_linear_attention:
           # for linear attention upcast hidden_states to float32
           hidden_states = hidden_states.to(dtype=torch.float32)


       hidden_states = hidden_states.reshape(
           batch_size, -1, 3 * attn.attention_head_dim, height * width
       )


       query, key, value = hidden_states.chunk(3, dim=2)
       query = attn.nonlinearity(query)
       key = attn.nonlinearity(key)


       if use_linear_attention:
           hidden_states = attn.apply_linear_attention(query, key, value)
           hidden_states = hidden_states.to(dtype=original_dtype)
       else:
           hidden_states = attn.apply_quadratic_attention(query, key, value)


       hidden_states = torch.reshape(hidden_states, (batch_size, -1, height, width))
       hidden_states = onnx_safe_ops.movedim(
           attn.to_out(onnx_safe_ops.movedim(hidden_states, 1, -1)), -1, 1
       )


       if attn.norm_type == "rms_norm":
           hidden_states = onnx_safe_ops.movedim(
               attn.norm_out(onnx_safe_ops.movedim(hidden_states, 1, -1)), -1, 1
           )
       else:
           hidden_states = attn.norm_out(hidden_states)


       if attn.residual_connection:
           hidden_states = hidden_states + residual


       return hidden_states




class SanaMultiscaleAttentionProjection(nn.Module):
   def __init__(
       self,
       in_channels: int,
       num_attention_heads: int,
       kernel_size: int,
   ) -> None:
       super().__init__()


       channels = 3 * in_channels
       self.proj_in = nn.Conv2d(
           channels,
           channels,
           kernel_size,
           padding=kernel_size // 2,
           groups=channels,
           bias=False,
       )
       self.proj_out = nn.Conv2d(
           channels, channels, 1, 1, 0, groups=3 * num_attention_heads, bias=False
       )


   def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
       hidden_states = self.proj_in(hidden_states)
       hidden_states = self.proj_out(hidden_states)
       return hidden_states




class SanaMultiscaleLinearAttention(nn.Module):
   r"""Lightweight multi-scale linear attention"""


   def __init__(
       self,
       in_channels: int,
       out_channels: int,
       num_attention_heads: Optional[int] = None,
       attention_head_dim: int = 8,
       mult: float = 1.0,
       norm_type: str = "batch_norm",
       kernel_sizes: Tuple[int, ...] = (5,),
       eps: float = 1e-15,
       residual_connection: bool = False,
   ):
       super().__init__()


       self.eps = eps
       self.attention_head_dim = attention_head_dim
       self.norm_type = norm_type
       self.residual_connection = residual_connection


       num_attention_heads = (
           int(in_channels // attention_head_dim * mult)
           if num_attention_heads is None
           else num_attention_heads
       )
       inner_dim = num_attention_heads * attention_head_dim


       self.to_q = nn.Linear(in_channels, inner_dim, bias=False)
       self.to_k = nn.Linear(in_channels, inner_dim, bias=False)
       self.to_v = nn.Linear(in_channels, inner_dim, bias=False)


       self.to_qkv_multiscale = nn.ModuleList()
       for kernel_size in kernel_sizes:
           self.to_qkv_multiscale.append(
               SanaMultiscaleAttentionProjection(inner_dim, num_attention_heads, kernel_size)
           )


       self.nonlinearity = nn.ReLU()
       self.to_out = nn.Linear(inner_dim * (1 + len(kernel_sizes)), out_channels, bias=False)
       self.norm_out = get_normalization(norm_type, num_features=out_channels)


       self.processor = SanaMultiscaleAttnProcessor2_0()


   def apply_linear_attention(
       self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
   ) -> torch.Tensor:
       value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1)  # Adds padding
       scores = torch.matmul(value, onnx_safe_ops.transpose(key, -1, -2))
       hidden_states = torch.matmul(scores, query)


       hidden_states = hidden_states.to(dtype=torch.float32)
       hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + self.eps)
       return hidden_states


   def apply_quadratic_attention(
       self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
   ) -> torch.Tensor:
       scores = torch.matmul(onnx_safe_ops.transpose(key, -1, -2), query)
       scores = scores.to(dtype=torch.float32)
       scores = scores / (torch.sum(scores, dim=2, keepdim=True) + self.eps)
       hidden_states = torch.matmul(value, scores)
       return hidden_states


   def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
       return self.processor(self, hidden_states)




ACTIVATION_FUNCTIONS = {
   "swish": nn.SiLU(),
   "silu": nn.SiLU(),
   "mish": nn.Mish(),
   "gelu": nn.GELU(),
   "relu": nn.ReLU(),
}




def get_activation(act_fn: str) -> nn.Module:
   """Helper function to get activation function from string.


   Args:
       act_fn (str): Name of activation function.


   Returns:
       nn.Module: Activation function.
   """


   act_fn = act_fn.lower()
   if act_fn in ACTIVATION_FUNCTIONS:
       return ACTIVATION_FUNCTIONS[act_fn]
   else:
       raise ValueError(f"Unsupported activation function: {act_fn}")




class GLUMBConv(nn.Module):
   def __init__(
       self,
       in_channels: int,
       out_channels: int,
       expand_ratio: float = 4,
       norm_type: Optional[str] = None,
       residual_connection: bool = True,
   ) -> None:
       super().__init__()


       hidden_channels = int(expand_ratio * in_channels)
       self.norm_type = norm_type
       self.residual_connection = residual_connection


       self.nonlinearity = nn.SiLU()
       self.conv_inverted = nn.Conv2d(in_channels, hidden_channels * 2, 1, 1, 0)
       self.conv_depth = nn.Conv2d(
           hidden_channels * 2, hidden_channels * 2, 3, 1, 1, groups=hidden_channels * 2
       )
       self.conv_point = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False)


       self.norm = None
       if norm_type == "rms_norm":
           self.norm = RMSNorm(out_channels, eps=1e-5, elementwise_affine=True, bias=True)


   def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
       if self.residual_connection:
           residual = hidden_states


       hidden_states = self.conv_inverted(hidden_states)
       hidden_states = self.nonlinearity(hidden_states)


       hidden_states = self.conv_depth(hidden_states)
       hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
       hidden_states = hidden_states * self.nonlinearity(gate)


       hidden_states = self.conv_point(hidden_states)


       if self.norm_type == "rms_norm":
           # move channel to the last dimension so we apply RMSnorm across channel dimension
           hidden_states = onnx_safe_ops.movedim(
               self.norm(onnx_safe_ops.movedim(hidden_states, 1, -1)), -1, 1
           )


       if self.residual_connection:
           hidden_states = hidden_states + residual


       return hidden_states




@dataclass
class EncoderOutput(BaseOutput):
   r"""
   Output of encoding method.


   Args:
       latent (`torch.Tensor` of shape `(batch_size, num_channels, latent_height, latent_width)`):
           The encoded latent.
   """


   latent: torch.Tensor




@dataclass
class DecoderOutput(BaseOutput):
   r"""
   Output of decoding method.


   Args:
       sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`):
           The decoded output sample from the last layer of the model.
   """


   sample: torch.Tensor
   commit_loss: Optional[torch.FloatTensor] = None




class ResBlock(nn.Module):
   def __init__(
       self,
       in_channels: int,
       out_channels: int,
       norm_type: str = "batch_norm",
       act_fn: str = "relu6",
   ) -> None:
       super().__init__()


       self.norm_type = norm_type


       self.nonlinearity = get_activation(act_fn) if act_fn is not None else nn.Identity()
       self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
       self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
       self.norm = get_normalization(norm_type, out_channels)


   def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
       residual = hidden_states
       hidden_states = self.conv1(hidden_states)
       hidden_states = self.nonlinearity(hidden_states)
       hidden_states = self.conv2(hidden_states)


       if self.norm_type == "rms_norm":
           # move channel to the last dimension so we apply RMSnorm across channel dimension
           hidden_states = onnx_safe_ops.movedim(
               self.norm(onnx_safe_ops.movedim(hidden_states, 1, -1)), -1, 1
           )
       else:
           hidden_states = self.norm(hidden_states)


       return hidden_states + residual




class EfficientViTBlock(nn.Module):
   def __init__(
       self,
       in_channels: int,
       mult: float = 1.0,
       attention_head_dim: int = 32,
       qkv_multiscales: Tuple[int, ...] = (5,),
       norm_type: str = "batch_norm",
   ) -> None:
       super().__init__()


       self.attn = SanaMultiscaleLinearAttention(
           in_channels=in_channels,
           out_channels=in_channels,
           mult=mult,
           attention_head_dim=attention_head_dim,
           norm_type=norm_type,
           kernel_sizes=qkv_multiscales,
           residual_connection=True,
       )


       self.conv_out = GLUMBConv(
           in_channels=in_channels,
           out_channels=in_channels,
           norm_type="rms_norm",
       )


   def forward(self, x: torch.Tensor) -> torch.Tensor:
       x = self.attn(x)
       x = self.conv_out(x)
       return x




def get_block(
   block_type: str,
   in_channels: int,
   out_channels: int,
   attention_head_dim: int,
   norm_type: str,
   act_fn: str,
   qkv_mutliscales: Tuple[int] = (),
):
   if block_type == "ResBlock":
       block = ResBlock(in_channels, out_channels, norm_type, act_fn)


   elif block_type == "EfficientViTBlock":
       block = EfficientViTBlock(
           in_channels,
           attention_head_dim=attention_head_dim,
           norm_type=norm_type,
           qkv_multiscales=qkv_mutliscales,
       )


   else:
       raise ValueError(f"Block with {block_type=} is not supported.")


   return block




class DCDownBlock2d(nn.Module):
   def __init__(
       self, in_channels: int, out_channels: int, downsample: bool = False, shortcut: bool = True
   ) -> None:
       super().__init__()


       self.downsample = downsample
       self.factor = 2
       self.stride = 1 if downsample else 2
       self.group_size = in_channels * self.factor**2 // out_channels
       self.shortcut = shortcut


       out_ratio = self.factor**2
       if downsample:
           assert out_channels % out_ratio == 0
           out_channels = out_channels // out_ratio


       self.conv = nn.Conv2d(
           in_channels,
           out_channels,
           kernel_size=3,
           stride=self.stride,
           padding=1,
       )


   def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
       x = self.conv(hidden_states)
       # print(x.shape, hidden_states.shape, self.downsample)
       if self.downsample:
           x = F.pixel_unshuffle(x, self.factor)


       # print(x.shape)


       if self.shortcut:
           y = F.pixel_unshuffle(hidden_states, self.factor)
           y = y.unflatten(1, (-1, self.group_size))
           y = y.mean(dim=2)
           hidden_states = x + y
       else:
           hidden_states = x


       return hidden_states




class DCUpBlock2d(nn.Module):
   def __init__(
       self,
       in_channels: int,
       out_channels: int,
       interpolate: bool = False,
       shortcut: bool = True,
       interpolation_mode: str = "nearest",
   ) -> None:
       super().__init__()


       self.interpolate = interpolate
       self.interpolation_mode = interpolation_mode
       self.shortcut = shortcut
       self.factor = 2
       self.repeats = out_channels * self.factor**2 // in_channels


       out_ratio = self.factor**2


       if not interpolate:
           out_channels = out_channels * out_ratio


       self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)


   def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
       if self.interpolate:
           x = F.interpolate(hidden_states, scale_factor=self.factor, mode=self.interpolation_mode)
           x = self.conv(x)
       else:
           x = self.conv(hidden_states)
           x = F.pixel_shuffle(x, self.factor)


       if self.shortcut:
           y = hidden_states.repeat_interleave(self.repeats, dim=1)
           y = F.pixel_shuffle(y, self.factor)
           hidden_states = x + y
       else:
           hidden_states = x


       return hidden_states




class Encoder(nn.Module):
   def __init__(
       self,
       in_channels: int,
       latent_channels: int,
       attention_head_dim: int = 32,
       block_type: Union[str, Tuple[str]] = "ResBlock",
       block_out_channels: Tuple[int] = (128, 256, 512, 512, 1024, 1024),
       layers_per_block: Tuple[int] = (2, 2, 2, 2, 2, 2),
       qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,)),
       downsample_block_type: str = "pixel_unshuffle",
       out_shortcut: bool = True,
   ):
       super().__init__()


       num_blocks = len(block_out_channels)


       if isinstance(block_type, str):
           block_type = (block_type,) * num_blocks


       if layers_per_block[0] > 0:
           self.conv_in = nn.Conv2d(
               in_channels,
               block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1],
               kernel_size=3,
               stride=1,
               padding=1,
           )
       else:
           self.conv_in = DCDownBlock2d(
               in_channels=in_channels,
               out_channels=(
                   block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1]
               ),
               downsample=downsample_block_type == "pixel_unshuffle",
               shortcut=False,
           )


       down_blocks = []
       for i, (out_channel, num_layers) in enumerate(zip(block_out_channels, layers_per_block)):
           down_block_list = []


           for _ in range(num_layers):
               block = get_block(
                   block_type[i],
                   out_channel,
                   out_channel,
                   attention_head_dim=attention_head_dim,
                   norm_type="rms_norm",
                   act_fn="silu",
                   qkv_mutliscales=qkv_multiscales[i],
               )
               down_block_list.append(block)


           if i < num_blocks - 1 and num_layers > 0:
               downsample_block = DCDownBlock2d(
                   in_channels=out_channel,
                   out_channels=block_out_channels[i + 1],
                   downsample=downsample_block_type == "pixel_unshuffle",
                   shortcut=True,
               )
               down_block_list.append(downsample_block)


           down_blocks.append(nn.Sequential(*down_block_list))


       self.down_blocks = nn.ModuleList(down_blocks)


       self.conv_out = nn.Conv2d(block_out_channels[-1], latent_channels, 3, 1, 1)


       self.out_shortcut = out_shortcut
       if out_shortcut:
           self.out_shortcut_average_group_size = block_out_channels[-1] // latent_channels


   def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
       if hidden_states.shape[-1] == 3:
           hidden_states = rearrange(hidden_states, "B H W C -> B C H W")
       temporal = hidden_states.ndim == 5
       if temporal:
           T = hidden_states.shape[1]
           hidden_states = rearrange(hidden_states, "B T C H W -> (B T) C H W")
       hidden_states = self.conv_in(hidden_states)
       if temporal:
           hidden_states = rearrange(hidden_states, "(B T) C H W -> B T C H W", T=T)
       for idx, down_block in enumerate(self.down_blocks):
           hidden_states = down_block(hidden_states)


       if self.out_shortcut:
           x = hidden_states.unflatten(1, (-1, self.out_shortcut_average_group_size))
           x = x.mean(dim=2)
           hidden_states = self.conv_out(hidden_states) + x
       else:
           hidden_states = self.conv_out(hidden_states)


       return hidden_states




class Decoder(nn.Module):
   def __init__(
       self,
       in_channels: int,
       latent_channels: int,
       attention_head_dim: int = 32,
       block_type: Union[str, Tuple[str]] = "ResBlock",
       block_out_channels: Tuple[int] = (128, 256, 512, 512, 1024, 1024),
       layers_per_block: Tuple[int] = (2, 2, 2, 2, 2, 2),
       qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (), (5,), (5,), (5,)),
       norm_type: Union[str, Tuple[str]] = "rms_norm",
       act_fn: Union[str, Tuple[str]] = "silu",
       upsample_block_type: str = "pixel_shuffle",
       in_shortcut: bool = True,
   ):
       super().__init__()


       num_blocks = len(block_out_channels)


       if isinstance(block_type, str):
           block_type = (block_type,) * num_blocks
       if isinstance(norm_type, str):
           norm_type = (norm_type,) * num_blocks
       if isinstance(act_fn, str):
           act_fn = (act_fn,) * num_blocks


       self.conv_in = nn.Conv2d(latent_channels, block_out_channels[-1], 3, 1, 1)


       self.in_shortcut = in_shortcut
       if in_shortcut:
           self.in_shortcut_repeats = block_out_channels[-1] // latent_channels


       up_blocks = []
       for i, (out_channel, num_layers) in reversed(
           list(enumerate(zip(block_out_channels, layers_per_block)))
       ):
           up_block_list = []


           if i < num_blocks - 1 and num_layers > 0:
               upsample_block = DCUpBlock2d(
                   block_out_channels[i + 1],
                   out_channel,
                   interpolate=upsample_block_type == "interpolate",
                   shortcut=True,
               )
               up_block_list.append(upsample_block)


           for _ in range(num_layers):
               block = get_block(
                   block_type[i],
                   out_channel,
                   out_channel,
                   attention_head_dim=attention_head_dim,
                   norm_type=norm_type[i],
                   act_fn=act_fn[i],
                   qkv_mutliscales=qkv_multiscales[i],
               )
               up_block_list.append(block)


           up_blocks.insert(0, nn.Sequential(*up_block_list))


       self.up_blocks = nn.ModuleList(up_blocks)


       channels = block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1]


       self.norm_out = RMSNorm(channels, 1e-5, elementwise_affine=True, bias=True)
       self.conv_act = nn.ReLU()
       self.conv_out = None


       if layers_per_block[0] > 0:
           self.conv_out = nn.Conv2d(channels, in_channels, 3, 1, 1)
       else:
           self.conv_out = DCUpBlock2d(
               channels,
               in_channels,
               interpolate=upsample_block_type == "interpolate",
               shortcut=False,
           )


   def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
       if self.in_shortcut:
           x = hidden_states.repeat_interleave(self.in_shortcut_repeats, dim=1)
           hidden_states = self.conv_in(hidden_states) + x
       else:
           hidden_states = self.conv_in(hidden_states)


       for idx, up_block in enumerate(reversed(self.up_blocks)):
           hidden_states = up_block(hidden_states)


       hidden_states = onnx_safe_ops.movedim(
           self.norm_out(onnx_safe_ops.movedim(hidden_states, 1, -1)), -1, 1
       )
       hidden_states = self.conv_act(hidden_states)
       hidden_states = self.conv_out(hidden_states)
       return hidden_states


class DCAutoencoder(nn.Module):
   r"""
   An Autoencoder model introduced in [DCAE](https://arxiv.org/abs/2410.10733) and used in
   [SANA](https://arxiv.org/abs/2410.10629).
   """

   def __init__(
       self,
       in_channels: int = 3,
       out_channels: int = 3,
       latent_channels: int = 32,
       compression_factor: int = 32,
       init_latent_channels: int = 128,
       attention_head_dim: int = 32,
       encoder_block_types: Union[str, Tuple[str]] = "ResBlock",
       decoder_block_types: Union[str, Tuple[str]] = "ResBlock",
       upsample_block_type: str = "pixel_shuffle",
       downsample_block_type: str = "pixel_unshuffle",
       decoder_norm_types: Union[str, Tuple[str]] = "rms_norm",
       decoder_act_fns: Union[str, Tuple[str]] = "silu",
       lossconfig=None,
       resolution=None,
       loss=None,
       is_vae: bool = False,
       use_dcae_setup: bool = False,
       label_fields: Optional[List[Field]] = None,
       use_dino_matching: bool = False,
       pad_height: int = 0,
   ) -> None:
       def get_power_of_2(n):
           import math
           assert n >= 0, f"n should be positive, {n}"
           log_base_2 = math.log2(n)


           if log_base_2 == int(log_base_2):
               return int(log_base_2)
           raise ValueError(f"n is not a power of 2, {n}")


       nlayers = get_power_of_2(compression_factor) + 1
       if nlayers not in [5, 6, 7]:
           raise ValueError(f"Unsupported compression factor: {compression_factor}, {nlayers}")


       mulfactor = 8
       if nlayers >= 5:
           encoder_block_out_channels = [
               init_latent_channels,
               2 * init_latent_channels,
               4 * init_latent_channels,
               4 * init_latent_channels,
               mulfactor * init_latent_channels,
           ]
           decoder_block_out_channels = [
               init_latent_channels,
               2 * init_latent_channels,
               4 * init_latent_channels,
               4 * init_latent_channels,
               mulfactor * init_latent_channels,
           ]
           encoder_layers_per_block = [2, 2, 2, 3, 3]
           decoder_layers_per_block = [3, 3, 3, 3, 3]
           encoder_qkv_multiscales = [(), (), (), (5,), (5,)]
           decoder_qkv_multiscales = [(), (), (), (5,), (5,)]
       if nlayers >= 6:
           if use_dcae_setup:
               encoder_block_out_channels += [mulfactor * init_latent_channels]
               decoder_block_out_channels += [mulfactor * init_latent_channels]
           else:
               encoder_block_out_channels += [mulfactor * 2 * init_latent_channels]
               decoder_block_out_channels += [mulfactor * 2 * init_latent_channels]
           encoder_layers_per_block += [3]
           decoder_layers_per_block += [3]
           encoder_qkv_multiscales += [(5,)]
           decoder_qkv_multiscales += [(5,)]
       if nlayers >= 7:
           if use_dcae_setup:
               encoder_block_out_channels += [mulfactor * 2 * init_latent_channels]
               decoder_block_out_channels += [mulfactor * 2 * init_latent_channels]
           else:
               encoder_block_out_channels += [mulfactor * 4 * init_latent_channels]
               decoder_block_out_channels += [mulfactor * 4 * init_latent_channels]


           encoder_layers_per_block += [3]
           decoder_layers_per_block += [3]
           encoder_qkv_multiscales += [(5,)]
           decoder_qkv_multiscales += [(5,)]


       if loss is None and lossconfig is not None:
           loss = instantiate_from_config(lossconfig)

       self.resolution = resolution
       self.loss = loss
       self.latent_channels = latent_channels

       out_shortcut = True


       if latent_channels == 6:
           out_shortcut = False


       self.encoder = Encoder(
           in_channels=in_channels,
           latent_channels=latent_channels,
           attention_head_dim=attention_head_dim,
           block_type=encoder_block_types,
           block_out_channels=encoder_block_out_channels,
           layers_per_block=encoder_layers_per_block,
           qkv_multiscales=encoder_qkv_multiscales,
           downsample_block_type=downsample_block_type,
           out_shortcut=out_shortcut,
       )
       self.use_separate_for_seg = use_separate_for_seg
       if use_separate_for_seg and not trivial_seg:
           self.encoder_for_seg = Encoder(
               in_channels=in_channels,
               latent_channels=latent_channels,
               attention_head_dim=attention_head_dim,
               block_type=encoder_block_types,
               block_out_channels=encoder_block_out_channels,
               layers_per_block=encoder_layers_per_block,
               qkv_multiscales=encoder_qkv_multiscales,
               downsample_block_type=downsample_block_type,
               out_shortcut=out_shortcut,
           )
       decoder_latent_channels = 2 * latent_channels if use_separate_for_seg else latent_channels
       self.decoder = Decoder(
           in_channels=out_channels,
           latent_channels=decoder_latent_channels,
           attention_head_dim=attention_head_dim,
           block_type=decoder_block_types,
           block_out_channels=decoder_block_out_channels,
           layers_per_block=decoder_layers_per_block,
           qkv_multiscales=decoder_qkv_multiscales,
           norm_type=decoder_norm_types,
           act_fn=decoder_act_fns,
           upsample_block_type=upsample_block_type,
           in_shortcut=out_shortcut,
       )
       self.is_vae = is_vae
       if is_vae:
           if use_separate_for_seg:
               self.quant_conv = torch.nn.Conv2d(latent_channels, 2 * latent_channels, 1)
               self.quant_conv_seg = torch.nn.Conv2d(latent_channels, 2 * latent_channels, 1)
           else:
               self.quant_conv = torch.nn.Conv2d(latent_channels, 2 * latent_channels, 1)


       self.spatial_compression_ratio = 2 ** (len(encoder_block_out_channels) - 1)
       self.temporal_compression_ratio = 1


       # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
       # to perform decoding of a single video latent at a time.
       self.use_slicing = False


       # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
       # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
       # intermediate tiles together, the memory requirement can be lowered.
       self.use_tiling = False


       # The minimal tile height and width for spatial tiling to be used
       self.tile_sample_min_height = 512
       self.tile_sample_min_width = 512


       # The minimal distance between two spatial tiles
       self.tile_sample_stride_height = 448
       self.tile_sample_stride_width = 448


   def get_training_tensorspec_fields(
       self,
       include_input_fields=True,
       include_output_fields=True,
   ):
       """
       This is used as the tensor spec datapoint transform target in config-based training,
       specifying what fields to keep in the output features after the tensor spec datapoint transform.
       In addition to the input and output fields, we also add the additional_dataset_target_fields to it.
       This would make experiments on new fields not defined yet in the tensor spec easier to set up.
       """
       fields = []
       if include_input_fields:
           fields.extend(self.input_fields.values())
       if include_output_fields:
           fields.extend(self.output_fields.values())
       return TensorSpecFields(fields=fields)


   def _build_tensor_spec(
       self,
   ) -> TensorSpec:
       rgb_image = RgbImageTensor(height=self.height, width=self.width)
       input_field = Field(name=self.input_field_name, rgb_image=rgb_image)
       eval_input_field = NetworkEvaluation.NetworkField(
           field_name=input_field.name,
           network_field_name=input_field.name,
       )
       reconstruction_field = Field(name=self.reconstruction_field_name, rgb_image=rgb_image)
       eval_reconstruction_field = NetworkEvaluation.NetworkField(
           field_name=reconstruction_field.name,
           network_field_name=reconstruction_field.name,
       )


       im = RgbImageTensor(height=self.height, width=self.width)
       im_field = Field(name=self.image_field_name, rgb_image=im)
       eval_im_field = NetworkEvaluation.NetworkField(
           field_name=self.image_field_name,
           network_field_name=self.image_field_name,
       )


       label_fields = self.label_fields or []
       if isinstance(label_fields, omegaconf.dictconfig.DictConfig):
           label_fields = instantiate_from_config(label_fields)


       fields = [im_field, input_field, reconstruction_field] + label_fields
       network_eval = NetworkEvaluation(
           input_fields=[eval_input_field, eval_im_field]
           + [
               NetworkEvaluation.NetworkField(
                   field_name=label_field.name, network_field_name=label_field.name
               )
               for label_field in label_fields
           ],
           output_fields=[eval_reconstruction_field],
       )


       transformation = Transformation(network_eval=network_eval)
       tensor_spec = TensorSpec(
           name="autoencoder", fields=fields, transformations=[transformation]
       )


       return tensor_spec


   # TODO
   def downsample_rates(self) -> tuple[int, int, int]:
       return (2, 2, 2)


   def get_last_layer(self):
       return self.decoder.conv_out.weight


   def get_parameters(self):
       return list(
           itertools.chain(
               self.encoder.parameters(),
               self.decoder.parameters(),
           )
       )


   def enable_tiling(
       self,
       tile_sample_min_height: Optional[int] = None,
       tile_sample_min_width: Optional[int] = None,
       tile_sample_stride_height: Optional[float] = None,
       tile_sample_stride_width: Optional[float] = None,
   ) -> None:
       r"""
       Enable tiled AE decoding. When this option is enabled, the AE will split the input tensor into tiles to compute
       decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
       processing larger images.


       Args:
           tile_sample_min_height (`int`, *optional*):
               The minimum height required for a sample to be separated into tiles across the height dimension.
           tile_sample_min_width (`int`, *optional*):
               The minimum width required for a sample to be separated into tiles across the width dimension.
           tile_sample_stride_height (`int`, *optional*):
               The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
               no tiling artifacts produced across the height dimension.
           tile_sample_stride_width (`int`, *optional*):
               The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling
               artifacts produced across the width dimension.
       """
       self.use_tiling = True
       self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
       self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
       self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
       self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width


   def disable_tiling(self) -> None:
       r"""
       Disable tiled AE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
       decoding in one step.
       """
       self.use_tiling = False


   def enable_slicing(self) -> None:
       r"""
       Enable sliced AE decoding. When this option is enabled, the AE will split the input tensor in slices to compute
       decoding in several steps. This is useful to save some memory and allow larger batch sizes.
       """
       self.use_slicing = True


   def disable_slicing(self) -> None:
       r"""
       Disable sliced AE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
       decoding in one step.
       """
       self.use_slicing = False


   def _encode(self, x: torch.Tensor) -> torch.Tensor:
       if hasattr(self, "pad_height") and self.pad_height:
           x = F.pad(x, (0, 0, self.pad_height, self.pad_height), mode="replicate")
       height = x.shape[-2]
       width = x.shape[-1]


       if self.use_tiling and (
           width > self.tile_sample_min_width or height > self.tile_sample_min_height
       ):
           return self.tiled_encode(x, return_dict=False)[0]


       if not self.use_separate_for_seg:
           encoded = self.encoder(x)
           if self.is_vae:
               moments = self.quant_conv(encoded)
               posterior = DiagonalGaussianDistribution(moments)
               return posterior
           else:
               return encoded
       elif self.trivial_seg:
           return self.quant_conv(self.encoder(x))


       nc = x.shape[1] // 2
       encoded = self.encoder(x[:, :nc, ...])
       encoded_seg = self.encoder_for_seg(x[:, nc:, ...])
       if self.is_vae:
           moments_rgb = self.quant_conv(encoded)
           mean_rgb, logvar_rgb = torch.chunk(moments_rgb, 2, dim=1)
           moments_seg = self.quant_conv_seg(encoded_seg)
           mean_seg, logvar_seg = torch.chunk(moments_seg, 2, dim=1)
           posterior = torch.cat([mean_rgb, mean_seg, logvar_rgb, logvar_seg], dim=1)
           return DiagonalGaussianDistribution(posterior)
       else:
           return torch.cat([encoded, encoded_seg], dim=1)


   def encode(
       self, x: torch.Tensor, return_dict: bool = True
   ) -> Union[EncoderOutput, Tuple[torch.Tensor]]:
       r"""
       Encode a batch of images into latents.


       Args:
           x (`torch.Tensor`): Input batch of images.
           return_dict (`bool`, defaults to `True`):
               Whether to return a [`~models.vae.EncoderOutput`] instead of a plain tuple.


       Returns:
               The latent representations of the encoded videos. If `return_dict` is True, a
               [`~models.vae.EncoderOutput`] is returned, otherwise a plain `tuple` is returned.
       """
       if self.use_slicing and x.shape[0] > 1:
           encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
           encoded = torch.cat(encoded_slices)
       else:
           encoded = self._encode(x)


       return encoded


   def _decode(self, z: torch.Tensor) -> torch.Tensor:
       batch_size, num_channels, height, width = z.shape


       if self.use_tiling and (
           width > self.tile_latent_min_width or height > self.tile_latent_min_height
       ):
           return self.tiled_decode(z, return_dict=False)[0]


       decoded = self.decoder(z)


       if hasattr(self, "pad_height") and self.pad_height:
           decoded = decoded[..., self.pad_height : -self.pad_height, :]


       return decoded


   def decode(
       self, z: torch.Tensor, return_dict: bool = True
   ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
       r"""
       Decode a batch of images.


       Args:
           z (`torch.Tensor`): Input batch of latent vectors.
           return_dict (`bool`, defaults to `True`):
               Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.


       Returns:
           [`~models.vae.DecoderOutput`] or `tuple`:
               If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
               returned.
       """
       if self.use_slicing and z.size(0) > 1:
           decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
           decoded = torch.cat(decoded_slices)
       else:
           decoded = self._decode(z)


       return decoded


   def tiled_encode(self, x: torch.Tensor, return_dict: bool = True) -> torch.Tensor:
       raise NotImplementedError("`tiled_encode` has not been implemented for AutoencoderDC.")


   def tiled_decode(
       self, z: torch.Tensor, return_dict: bool = True
   ) -> Union[DecoderOutput, torch.Tensor]:
       raise NotImplementedError("`tiled_decode` has not been implemented for AutoencoderDC.")
