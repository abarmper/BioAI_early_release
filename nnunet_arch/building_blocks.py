"""Atomic building blocks, helper functions, regularisation, and weight init.

Consolidated from dynamic_network_architectures:
- building_blocks/helper.py
- building_blocks/simple_conv_blocks.py
- building_blocks/residual.py
- building_blocks/regularization.py
- initialization/weight_init.py
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Helper functions
# ======================================================================

def convert_conv_op_to_dim(conv_op: Type[nn.Module]) -> int:
    if conv_op == nn.Conv2d:
        return 2
    elif conv_op == nn.Conv3d:
        return 3
    elif conv_op == nn.Conv1d:
        return 1
    raise ValueError(f"Unknown conv op: {conv_op}")


def convert_dim_to_conv_op(dim: int) -> Type[nn.Module]:
    if dim == 2:
        return nn.Conv2d
    elif dim == 3:
        return nn.Conv3d
    elif dim == 1:
        return nn.Conv1d
    raise ValueError(f"Unknown dimension: {dim}")


def maybe_convert_scalar_to_list(conv_op: Type[nn.Module], scalar):
    """Convert a scalar kernel/stride to a list matching the conv op dims."""
    if not isinstance(scalar, (list, tuple)):
        dim = convert_conv_op_to_dim(conv_op)
        return [scalar] * dim
    return list(scalar)


def get_matching_convtransp(conv_op: Type[nn.Module]) -> Type[nn.Module]:
    if conv_op == nn.Conv2d:
        return nn.ConvTranspose2d
    elif conv_op == nn.Conv3d:
        return nn.ConvTranspose3d
    elif conv_op == nn.Conv1d:
        return nn.ConvTranspose1d
    raise ValueError(f"No matching ConvTranspose for {conv_op}")


def get_matching_instancenorm(conv_op: Type[nn.Module]) -> Type[nn.Module]:
    if conv_op == nn.Conv2d:
        return nn.InstanceNorm2d
    elif conv_op == nn.Conv3d:
        return nn.InstanceNorm3d
    elif conv_op == nn.Conv1d:
        return nn.InstanceNorm1d
    raise ValueError(f"No matching InstanceNorm for {conv_op}")


def get_matching_batchnorm(conv_op: Type[nn.Module]) -> Type[nn.Module]:
    if conv_op == nn.Conv2d:
        return nn.BatchNorm2d
    elif conv_op == nn.Conv3d:
        return nn.BatchNorm3d
    elif conv_op == nn.Conv1d:
        return nn.BatchNorm1d
    raise ValueError(f"No matching BatchNorm for {conv_op}")


def get_matching_pool_op(
    conv_op: Type[nn.Module],
    pool_type: str = "avg",
) -> Type[nn.Module]:
    dim = convert_conv_op_to_dim(conv_op)
    if pool_type == "avg":
        return {1: nn.AvgPool1d, 2: nn.AvgPool2d, 3: nn.AvgPool3d}[dim]
    elif pool_type == "max":
        return {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}[dim]
    raise ValueError(f"Unknown pool type: {pool_type}")


# ======================================================================
# Regularisation: DropPath, SqueezeExcite
# ======================================================================

def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Stochastic depth per sample (drop entire residual branch)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation block, works for 1D/2D/3D."""

    def __init__(
        self,
        channels: int,
        conv_op: Type[nn.Module],
        rd_ratio: float = 0.0625,
        rd_channels: Optional[int] = None,
        bias: bool = True,
    ):
        super().__init__()
        if rd_channels is None:
            rd_channels = make_divisible(channels * rd_ratio, 8)
        # Use 1x1 convolutions for channel reduction/expansion
        self.fc1 = conv_op(channels, rd_channels, kernel_size=1, bias=bias)
        self.fc2 = conv_op(rd_channels, channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 3D-safe global average pooling: mean over all spatial dims
        x_se = x.mean(dim=tuple(range(2, x.ndim)), keepdim=True)
        x_se = F.relu(self.fc1(x_se), inplace=True)
        x_se = torch.sigmoid(self.fc2(x_se))
        return x * x_se


# ======================================================================
# Weight initialisation
# ======================================================================

class InitWeights_He:
    """Kaiming (He) normal initialisation for conv / convtransp layers."""

    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module: nn.Module):
        if isinstance(
            module,
            (nn.Conv1d, nn.Conv2d, nn.Conv3d,
             nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d),
        ):
            nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def init_last_bn_before_add_to_0(module: nn.Module):
    """Zero the last BN gamma in BasicBlockD residual blocks."""
    if isinstance(module, BasicBlockD):
        # conv2 is the second ConvDropoutNormReLU; its norm is the BN/IN
        # whose weight should be zeroed so the residual branch starts as
        # identity.
        last_conv_block = module.conv2
        norm = last_conv_block.norm
        if norm is not None:
            weight = getattr(norm, 'weight', None)
            bias = getattr(norm, 'bias', None)
            if weight is not None:
                nn.init.constant_(weight, 0)
            if bias is not None:
                nn.init.constant_(bias, 0)


# ======================================================================
# ConvDropoutNormReLU
# ======================================================================

class ConvDropoutNormReLU(nn.Module):
    """Single conv block: conv → [dropout] → [norm] → [nonlin].

    Padding is always ``(kernel_size - 1) // 2`` per axis (same padding for
    odd kernels).
    """

    def __init__(
        self,
        conv_op: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[nn.Module]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        nonlin_first: bool = False,
    ):
        super().__init__()
        self.conv_op = conv_op
        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        stride = maybe_convert_scalar_to_list(conv_op, stride)
        padding = [(k - 1) // 2 for k in kernel_size]

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = conv_op(
            in_channels, out_channels, kernel_size, stride, padding,
            bias=conv_bias,
        )

        self.norm = None
        if norm_op is not None:
            self.norm = norm_op(out_channels, **(norm_op_kwargs or {}))

        self.dropout = None
        if dropout_op is not None:
            self.dropout = dropout_op(**(dropout_op_kwargs or {}))

        self.nonlin = None
        if nonlin is not None:
            self.nonlin = nonlin(**(nonlin_kwargs or {}))

        self.nonlin_first = nonlin_first

        # Build sequential order
        ops = [self.conv]
        if self.dropout is not None:
            ops.append(self.dropout)

        if nonlin_first:
            if self.nonlin is not None:
                ops.append(self.nonlin)
            if self.norm is not None:
                ops.append(self.norm)
        else:
            if self.norm is not None:
                ops.append(self.norm)
            if self.nonlin is not None:
                ops.append(self.nonlin)

        self.all_modules = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.all_modules(x)

    def compute_conv_feature_map_size(
        self, input_size: Union[List[int], Tuple[int, ...]],
    ) -> int:
        """Number of output feature map elements (not bytes)."""
        output_size = [
            (i + 2 * p - k) // s + 1
            for i, k, s, p in zip(input_size, self.kernel_size, self.stride, self.padding)
        ]
        return int(self.out_channels * np.prod(output_size))


# ======================================================================
# StackedConvBlocks
# ======================================================================

class StackedConvBlocks(nn.Module):
    """N sequential ConvDropoutNormReLU blocks.

    The first block uses ``initial_stride`` for down-sampling; the rest
    use stride 1.
    """

    def __init__(
        self,
        num_convs: int,
        conv_op: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        initial_stride: Union[int, List[int], Tuple[int, ...]] = 1,
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[nn.Module]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        nonlin_first: bool = False,
    ):
        super().__init__()
        blocks = []
        for i in range(num_convs):
            blocks.append(
                ConvDropoutNormReLU(
                    conv_op=conv_op,
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=initial_stride if i == 0 else 1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    nonlin_first=nonlin_first,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

    def compute_conv_feature_map_size(
        self, input_size: Union[List[int], Tuple[int, ...]],
    ) -> int:
        total = 0
        size = list(input_size)
        for block in self.blocks:
            blk = cast(ConvDropoutNormReLU, block)
            total += blk.compute_conv_feature_map_size(size)
            size = [
                (s + 2 * p - k) // st + 1
                for s, k, st, p in zip(size, blk.kernel_size, blk.stride, blk.padding)
            ]
        return total


# ======================================================================
# BasicBlockD (ResNet-D residual block)
# ======================================================================

class BasicBlockD(nn.Module):
    """ResNet-D style residual block with 2 convolutions.

    Skip connection uses AvgPool→1x1 when stride or channel mismatch.
    Optional DropPath (stochastic depth) and SqueezeExcite.
    """

    def __init__(
        self,
        conv_op: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]] = 1,
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[nn.Module]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        stochastic_depth_p: float = 0.0,
        squeeze_excitation: bool = False,
        squeeze_excitation_reduction_ratio: float = 0.0625,
    ):
        super().__init__()
        self.conv_op = conv_op
        self.in_channels = in_channels
        self.out_channels = out_channels

        stride_list = maybe_convert_scalar_to_list(conv_op, stride)
        kernel_size_list = maybe_convert_scalar_to_list(conv_op, kernel_size)

        # Conv1: strided conv (downsampling)
        self.conv1 = ConvDropoutNormReLU(
            conv_op=conv_op,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )

        # Conv2: stride-1 conv (no nonlin — added after residual)
        self.conv2 = ConvDropoutNormReLU(
            conv_op=conv_op,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=None,
            nonlin_kwargs=None,
        )

        # Skip connection: identity or AvgPool + 1x1 conv (ResNet-D)
        needs_skip = (in_channels != out_channels) or any(s > 1 for s in stride_list)
        if needs_skip:
            skip_layers = []
            if any(s > 1 for s in stride_list):
                pool_op = get_matching_pool_op(conv_op, "avg")
                skip_layers.append(
                    pool_op(kernel_size=stride_list, stride=stride_list, ceil_mode=True)
                )
            skip_layers.append(
                conv_op(in_channels, out_channels, kernel_size=1, stride=1, bias=conv_bias)
            )
            if norm_op is not None:
                skip_layers.append(norm_op(out_channels, **(norm_op_kwargs or {})))
            self.skip = nn.Sequential(*skip_layers)
        else:
            self.skip = nn.Identity()

        # Optional SE and stochastic depth
        self.se = (
            SqueezeExcite(out_channels, conv_op, rd_ratio=squeeze_excitation_reduction_ratio)
            if squeeze_excitation
            else None
        )
        self.drop_path = DropPath(stochastic_depth_p) if stochastic_depth_p > 0 else None

        # Final activation (applied after residual addition)
        if nonlin is not None:
            self.nonlin = nonlin(**(nonlin_kwargs or {}))
        else:
            self.nonlin = None

        self.stride = stride_list
        self.kernel_size = kernel_size_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        if self.drop_path is not None:
            out = self.drop_path(out)
        out = out + residual
        if self.nonlin is not None:
            out = self.nonlin(out)
        return out

    def compute_conv_feature_map_size(
        self, input_size: Union[List[int], Tuple[int, ...]],
    ) -> int:
        total = self.conv1.compute_conv_feature_map_size(input_size)
        output_size = [
            (i + 2 * p - k) // s + 1
            for i, k, s, p in zip(
                input_size, self.conv1.kernel_size, self.conv1.stride, self.conv1.padding
            )
        ]
        total += self.conv2.compute_conv_feature_map_size(output_size)
        return total


# ======================================================================
# StackedResidualBlocks
# ======================================================================

class StackedResidualBlocks(nn.Module):
    """N sequential BasicBlockD blocks.

    Only the first block uses ``initial_stride``; rest use stride 1.
    """

    def __init__(
        self,
        n_blocks: int,
        conv_op: Type[nn.Module],
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        initial_stride: Union[int, List[int], Tuple[int, ...]] = 1,
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[nn.Module]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        stochastic_depth_p: float = 0.0,
        squeeze_excitation: bool = False,
        squeeze_excitation_reduction_ratio: float = 0.0625,
    ):
        super().__init__()
        blocks = []
        for i in range(n_blocks):
            blocks.append(
                BasicBlockD(
                    conv_op=conv_op,
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=initial_stride if i == 0 else 1,
                    conv_bias=conv_bias,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    dropout_op=dropout_op,
                    dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    stochastic_depth_p=stochastic_depth_p,
                    squeeze_excitation=squeeze_excitation,
                    squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio,
                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_stride = maybe_convert_scalar_to_list(conv_op, initial_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

    def compute_conv_feature_map_size(
        self, input_size: Union[List[int], Tuple[int, ...]],
    ) -> int:
        total = 0
        size = list(input_size)
        for block in self.blocks:
            blk = cast(BasicBlockD, block)
            total += blk.compute_conv_feature_map_size(size)
            size = [
                (s + 2 * p - k) // st + 1
                for s, k, st, p in zip(
                    size, blk.conv1.kernel_size, blk.conv1.stride, blk.conv1.padding
                )
            ]
        return total
