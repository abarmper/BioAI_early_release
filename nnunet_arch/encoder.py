"""PlainConvEncoder and ResidualEncoder.

Ported from dynamic_network_architectures:
- building_blocks/plain_conv_encoder.py
- building_blocks/residual_encoders.py
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn

from models.nnunet_arch.building_blocks import (
    StackedConvBlocks,
    StackedResidualBlocks,
)


ConvOpType = Union[Type[nn.Conv1d], Type[nn.Conv2d], Type[nn.Conv3d]]
StageSpatialArg = Union[int, List[int]]


class PlainConvEncoder(nn.Module):
    """Multi-stage convolutional encoder with per-axis kernel/stride support.

    Each stage is a StackedConvBlocks. Down-sampling is via strided first
    convolution (default) or pooling.

    Parameters
    ----------
    input_channels : int
    n_stages : int
    features_per_stage : sequence of int
    conv_op : type
    kernel_sizes : sequence of (int or list-of-int)
    strides : sequence of (int or list-of-int)
        Length must equal n_stages.  Stage 0 typically has stride [1,1,1].
    n_conv_per_stage : int or sequence of int
    conv_bias : bool
    norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin,
    nonlin_kwargs : standard block kwargs
    return_skips : bool
        If True, forward returns list of all stage outputs.
    nonlin_first : bool
    pool : str
        'conv' (strided conv, default), 'avg', or 'max'.
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: ConvOpType,
        kernel_sizes: Union[int, List[Union[int, List[int]]]],
        strides: Union[int, List[Union[int, List[int]]]],
        n_conv_per_stage: Union[int, List[int]] = 2,
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[nn.Module]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        return_skips: bool = True,
        nonlin_first: bool = False,
        pool: str = "conv",
    ):
        super().__init__()

        feature_values = (
            [features_per_stage] * n_stages
            if isinstance(features_per_stage, int)
            else list(features_per_stage)
        )
        convs_per_stage = (
            [n_conv_per_stage] * n_stages
            if isinstance(n_conv_per_stage, int)
            else list(n_conv_per_stage)
        )
        kernel_size_values: List[StageSpatialArg] = (
            [kernel_sizes] * n_stages if isinstance(kernel_sizes, int) else list(kernel_sizes)
        )
        stride_values: List[StageSpatialArg] = (
            [strides] * n_stages if isinstance(strides, int) else list(strides)
        )

        assert len(feature_values) == n_stages
        assert len(convs_per_stage) == n_stages
        assert len(kernel_size_values) == n_stages
        assert len(stride_values) == n_stages

        # Store for decoder to read
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_size_values
        self.strides = stride_values
        self.n_stages = n_stages
        self.features_per_stage = feature_values
        self.n_conv_per_stage = convs_per_stage
        self.return_skips = return_skips
        self.nonlin_first = nonlin_first
        self.pool = pool
        self.output_channels = feature_values

        stages: List[StackedConvBlocks] = []
        for s in range(n_stages):
            in_ch = input_channels if s == 0 else feature_values[s - 1]
            stride_s = stride_values[s]

            if pool == "conv" or s == 0:
                initial_stride = stride_s
            else:
                initial_stride = 1

            stage = StackedConvBlocks(
                num_convs=convs_per_stage[s],
                conv_op=conv_op,
                in_channels=in_ch,
                out_channels=feature_values[s],
                kernel_size=kernel_size_values[s],
                initial_stride=initial_stride,
                conv_bias=conv_bias,
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                dropout_op=dropout_op,
                dropout_op_kwargs=dropout_op_kwargs,
                nonlin=nonlin,
                nonlin_kwargs=nonlin_kwargs,
                nonlin_first=nonlin_first,
            )
            stages.append(stage)

        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        if self.return_skips:
            return skips
        return x

    def forward_skips(self, x: torch.Tensor) -> List[torch.Tensor]:
        skips = self.forward(x)
        if not isinstance(skips, list):
            raise RuntimeError("PlainConvEncoder must be constructed with return_skips=True.")
        return skips

    def compute_conv_feature_map_size(
        self, input_size: Union[List[int], Tuple[int, ...]],
    ) -> int:
        total = 0
        size = list(input_size)
        for stage in self.stages:
            stg = cast(StackedConvBlocks, stage)
            total += stg.compute_conv_feature_map_size(size)
            stride = stg.initial_stride
            size = [s // st for s, st in zip(size, stride)]
        return total


class ResidualEncoder(nn.Module):
    """Multi-stage residual encoder with optional stem.

    Each stage uses StackedResidualBlocks (BasicBlockD).
    Has a stem (single StackedConvBlocks, stride 1) before stage 0
    unless ``disable_default_stem=True``.

    Parameters
    ----------
    input_channels : int
    n_stages : int
    features_per_stage : sequence of int
    conv_op : type
    kernel_sizes : sequence of (int or list-of-int)
    strides : sequence of (int or list-of-int)
    n_blocks_per_stage : int or sequence of int
    conv_bias : bool
    norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin,
    nonlin_kwargs : standard block kwargs
    return_skips : bool
    nonlin_first : bool
    stem_channels : int or None
        Output channels of the stem. Defaults to features_per_stage[0].
    stochastic_depth_p : float
    squeeze_excitation : bool
    squeeze_excitation_reduction_ratio : float
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: ConvOpType,
        kernel_sizes: Union[int, List[Union[int, List[int]]]],
        strides: Union[int, List[Union[int, List[int]]]],
        n_blocks_per_stage: Union[int, List[int]] = 2,
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[nn.Module]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        return_skips: bool = True,
        nonlin_first: bool = False,
        stem_channels: Optional[int] = None,
        stochastic_depth_p: float = 0.0,
        squeeze_excitation: bool = False,
        squeeze_excitation_reduction_ratio: float = 0.0625,
    ):
        super().__init__()

        feature_values = (
            [features_per_stage] * n_stages
            if isinstance(features_per_stage, int)
            else list(features_per_stage)
        )
        blocks_per_stage = (
            [n_blocks_per_stage] * n_stages
            if isinstance(n_blocks_per_stage, int)
            else list(n_blocks_per_stage)
        )
        kernel_size_values: List[StageSpatialArg] = (
            [kernel_sizes] * n_stages if isinstance(kernel_sizes, int) else list(kernel_sizes)
        )
        stride_values: List[StageSpatialArg] = (
            [strides] * n_stages if isinstance(strides, int) else list(strides)
        )

        assert len(feature_values) == n_stages
        assert len(blocks_per_stage) == n_stages
        assert len(kernel_size_values) == n_stages
        assert len(stride_values) == n_stages

        # Store for decoder
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_size_values
        self.strides = stride_values
        self.n_stages = n_stages
        self.features_per_stage = feature_values
        self.n_blocks_per_stage = blocks_per_stage
        self.return_skips = return_skips
        self.nonlin_first = nonlin_first

        # Stem: single conv, no stride, projects input_channels → stem_channels
        if stem_channels is None:
            stem_channels = feature_values[0]
        self.stem_channels = stem_channels

        self.stem = StackedConvBlocks(
            num_convs=1,
            conv_op=conv_op,
            in_channels=input_channels,
            out_channels=stem_channels,
            kernel_size=kernel_size_values[0],
            initial_stride=1,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            nonlin_first=nonlin_first,
        )

        # Residual stages
        stages: List[StackedResidualBlocks] = []
        for s in range(n_stages):
            in_ch = stem_channels if s == 0 else feature_values[s - 1]
            stages.append(
                StackedResidualBlocks(
                    n_blocks=blocks_per_stage[s],
                    conv_op=conv_op,
                    in_channels=in_ch,
                    out_channels=feature_values[s],
                    kernel_size=kernel_size_values[s],
                    initial_stride=stride_values[s],
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
        self.stages = nn.ModuleList(stages)
        self.output_channels = feature_values

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        x = self.stem(x)
        skips: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        if self.return_skips:
            return skips
        return x

    def forward_skips(self, x: torch.Tensor) -> List[torch.Tensor]:
        skips = self.forward(x)
        if not isinstance(skips, list):
            raise RuntimeError("ResidualEncoder must be constructed with return_skips=True.")
        return skips

    def compute_conv_feature_map_size(
        self, input_size: Union[List[int], Tuple[int, ...]],
    ) -> int:
        total = self.stem.compute_conv_feature_map_size(input_size)
        size = list(input_size)  # stem has stride 1, size unchanged
        for stage in self.stages:
            stg = cast(StackedResidualBlocks, stage)
            total += stg.compute_conv_feature_map_size(size)
            stride = stg.initial_stride
            size = [s // st for s, st in zip(size, stride)]
        return total
