from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


import dbx


class Conv2dSame(nn.Module):
    """Manual convolution with same padding

    Although PyTorch >= 1.10.0 supports ``padding='same'`` as a keyword argument,
    this does not export to CoreML as of coremltools 5.1.0, so we need to
    implement the internal torch logic manually. Currently the ``RuntimeError`` is

    "PyTorch convert function for op '_convolution_mode' not implemented"

    Also same padding is not supported for strided convolutions at the moment
    https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L93

    Taken from: https://github.com/pytorch/pytorch/issues/3867#issuecomment-974159134
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, *, log = None, **kwargs):
        """Wrap base convolution layer

        See official PyTorch documentation for parameter details
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs,
        )

        # Setup internal representations
        #kernel_size_ = _pair(kernel_size)
        kernel_size_ = (kernel_size, kernel_size)
        #dilation_ = _pair(dilation)
        dilation_ = (dilation, dilation)
        self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size_)
        self.log = log or dbx.Logger(self.__class__.__name__)

        # Follow the logic from ``nn._ConvNd``
        # https://github.com/pytorch/pytorch/blob/v1.10.0/torch/nn/modules/conv.py#L116
        for d, k, i in zip(dilation_, kernel_size_, range(len(kernel_size_) - 1, -1, -1)):
            total_padding = d * (k - 1)
            left_pad = total_padding // 2
            self._reversed_padding_repeated_twice[2 * i] = left_pad
            self._reversed_padding_repeated_twice[2 * i + 1] = total_padding - left_pad

    def forward(self, imgs):
        """Setup padding so same spatial dimensions are returned

        All shapes (input/output) are ``(N, C, W, H)`` convention

        :param torch.Tensor imgs:
        :return torch.Tensor:
        """
        padded = F.pad(imgs, self._reversed_padding_repeated_twice)
        self.log.detailed(f"imgs.shape: {imgs.shape}")
        self.log.detailed(f"self._reversed_padding_repeated_twice: {self._reversed_padding_repeated_twice}")
        self.log.detailed(f"padded.shape: {padded.shape}")
        return self.conv(padded)


class BatchNorm2d(nn.BatchNorm2d):
    """Vanilla batch norm, but will put itself into eval mode if all of its
    weights are frozen, to avoid updating frozen model components in Hydra
    architectures.
    """

    def forward(self, *args, **kwargs):
        is_training = self.training
        if not (torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters())):
            self.train(False)
        result = super().forward(*args, **kwargs)
        self.train(is_training)
        return result


class ConvRelu(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        batch_norm_epsilon: float = 1e-5,
        batch_norm_training_override: bool = False,
        batch_norm_momentum: float = 0.9,
        batch_norm_affine: bool = True,
    ):
        super().__init__()
        self.conv2d = Conv2dSame(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=not use_batch_norm,
        )
        self.batch_normalization = None
        if use_batch_norm:
            # the current batchnorm momentum param might be too high for short finetuning runs
            self.batch_normalization = BatchNorm2d(
                out_channels,
                eps=batch_norm_epsilon,
                momentum=batch_norm_momentum,
                affine=batch_norm_affine,
                track_running_stats=not batch_norm_training_override,
            )

    def forward(self, input_t: torch.Tensor) -> torch.Tensor:
        out = self.conv2d(input_t)
        if self.batch_normalization is not None:
            out = self.batch_normalization(out)
        out = F.relu(out)
        return out


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        use_batch_norm: bool = True,
        batch_norm_training_override: bool = False,
        use_squeeze_and_excite: bool = False,
        cnn_use_residual: bool = False,
    ):
        super().__init__()
        self.cnn_use_residual = cnn_use_residual

        self.conv_relu_1 = ConvRelu(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            batch_norm_training_override=batch_norm_training_override,
        )
        if self.cnn_use_residual:
            if in_channels != out_channels:
                self.residual_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            else:
                self.residual_proj = nn.Identity()

        self.conv_relu_2 = ConvRelu(

            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            batch_norm_training_override=batch_norm_training_override,
        )
        self.use_squeeze_and_excite = use_squeeze_and_excite
        if self.use_squeeze_and_excite:
            self.avg_pool = nn.AvgPool2d(out_channels)
            squeezed_features = out_channels // 16
            self.dense = nn.Sequential(
                nn.Linear(in_features=out_channels, out_features=squeezed_features), nn.ReLU()
            )
            self.gate = nn.Sequential(
                nn.Linear(in_features=squeezed_features, out_features=out_channels), nn.Sigmoid()
            )

    def forward(self, input_t: torch.Tensor) -> torch.Tensor:
        out = self.conv_relu_1(input_t)
        if self.cnn_use_residual:
            out = self.conv_relu_2(out) + self.residual_proj(input_t)
        else:
            out = self.conv_relu_2(out)

        if self.use_squeeze_and_excite:
            F_sq = self.avg_pool(out)
            F_ex = self.dense(F_sq)
            channel_gate = self.gate(F_ex)
            out = channel_gate * out
        return out
    

class UpLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: Optional[int] = None,
        kernel_size: int = 3,
        channel_dim: int = 1,
        use_batch_norm: bool = True,
        batch_norm_training_override: bool = False,
        use_bilinear_upsampling: bool = False,
        use_squeeze_and_excite: bool = False,
        use_skip_connection: bool = True,
        add_skip: bool = False,
        cnn_use_pixel_shuffle: bool = False,
        cnn_use_residual: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cnn_use_pixel_shuffle = cnn_use_pixel_shuffle

        self.use_bilinear_upsampling = use_bilinear_upsampling
        if self.cnn_use_pixel_shuffle:
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.post_shuffle_block = nn.Sequential(
                Conv2dSame(
                    in_channels // 4,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.ReLU(),
            )
        elif self.use_bilinear_upsampling:

            self.post_bilinear_block = nn.Sequential(
                Conv2dSame(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    bias=False,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            )
        else:
            self.deconv_block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=(2, 2),
                    stride=(2, 2),
                    padding=0,
                    output_padding=0,
                    groups=1,
                    bias=False,
                    dilation=1,
                    padding_mode="zeros",
                ),
                nn.ReLU(),
            )

        self.use_skip_connection = use_skip_connection
        self.true_add_skip = add_skip
        assert (
            skip_channels is None or self.use_skip_connection
        ), "Cannot set skip_channels if not using skip connections."
        self.skip_channels = (skip_channels or out_channels) if self.use_skip_connection else 0
        self.conv2d_block = ConvBlock(
            in_channels=(
                self.skip_channels if (self.use_skip_connection and not self.true_add_skip) else 0
            )
            + out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            batch_norm_training_override=batch_norm_training_override,
            use_squeeze_and_excite=use_squeeze_and_excite,
            cnn_use_residual=cnn_use_residual,
        )

        self.channel_dim = channel_dim


    def forward(self, input_t: torch.Tensor, skip_features: torch.Tensor) -> torch.Tensor:
        if self.cnn_use_pixel_shuffle:
            # PixelShuffle expects in_channels to be divisible by 4 for 2x upscale
            curr_tensor = self.pixel_shuffle(input_t)
            curr_tensor = self.post_shuffle_block(curr_tensor)
        elif self.use_bilinear_upsampling:

            target_shape = tuple(2 * x for x in input_t.shape[-2:])
            curr_tensor = F.interpolate(input_t, size=target_shape, mode="bilinear")
            curr_tensor = self.post_bilinear_block(curr_tensor)
        else:
            curr_tensor = self.deconv_block(input_t)

        if self.use_skip_connection:
            assert skip_features is not None
            # added to load older nets
            if not hasattr(self, "true_add_skip"):
                self.true_add_skip = False

            # if the encoder or decoder have different sizes, the decoder uses the first <decoder_features> features
            if skip_features.shape[self.channel_dim] > self.skip_channels:
                skip_features = skip_features[:, : self.skip_channels, ...]

            if self.true_add_skip:  # literally add them
                curr_tensor = skip_features + curr_tensor
            else:
                curr_tensor = torch.concat(
                    tensors=[skip_features, curr_tensor],
                    dim=self.channel_dim,
                )

        return self.conv2d_block(curr_tensor)
