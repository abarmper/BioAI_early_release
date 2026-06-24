"""UNetDecoder, PlainConvUNet, and ResidualEncoderUNet.

Ported from dynamic_network_architectures:
- building_blocks/unet_decoder.py
- architectures/unet.py
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Tuple, Type, Union, cast

import numpy as np
import torch
import torch.nn as nn

from models.nnunet_arch.building_blocks import (
    InitWeights_He,
    StackedConvBlocks,
    get_matching_convtransp,
    init_last_bn_before_add_to_0,
    maybe_convert_scalar_to_list,
)
from models.nnunet_arch.encoder import ConvOpType, PlainConvEncoder, ResidualEncoder


EncoderType = Union[PlainConvEncoder, ResidualEncoder]


class ConvTransposeFactory(Protocol):
    def __call__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]],
        bias: bool = True,
    ) -> nn.Module: ...


class UNetDecoder(nn.Module):
    """Standard UNet decoder with optional deep supervision.

    Reads operator types from the encoder unless explicitly overridden.
    Always builds segmentation heads at every level (for checkpoint
    compatibility between DS on/off).

    Parameters
    ----------
    encoder : PlainConvEncoder or ResidualEncoder
    num_classes : int
    n_conv_per_stage : int or list of int
        Number of conv blocks per decoder stage. Length = n_stages - 1.
    deep_supervision : bool
    nonlin_first : bool
    """

    def __init__(
        self,
        encoder: EncoderType,
        num_classes: int,
        n_conv_per_stage: Union[int, List[int], Tuple[int, ...]] = 2,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Read encoder attributes
        conv_op = encoder.conv_op
        norm_op = encoder.norm_op
        norm_op_kwargs = encoder.norm_op_kwargs
        dropout_op = encoder.dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs
        nonlin = encoder.nonlin
        nonlin_kwargs = encoder.nonlin_kwargs
        conv_bias = encoder.conv_bias
        features_per_stage = encoder.features_per_stage
        kernel_sizes = encoder.kernel_sizes
        strides = encoder.strides
        n_stages = encoder.n_stages

        n_decoder_stages = n_stages - 1
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_decoder_stages
        assert len(n_conv_per_stage) == n_decoder_stages

        convtransp_op = cast(ConvTransposeFactory, get_matching_convtransp(conv_op))

        # Build decoder stages (from bottleneck to full resolution)
        self.upsample_layers = nn.ModuleList()
        self.decode_stages = nn.ModuleList()
        self.seg_layers = nn.ModuleList()

        for s in range(n_decoder_stages):
            # s=0 is the first decoder stage (closest to bottleneck)
            # Encoder skip index: n_stages - 2 - s (going from deep to shallow)
            encoder_skip_idx = n_stages - 2 - s

            in_features_below = features_per_stage[encoder_skip_idx + 1]
            in_features_skip = features_per_stage[encoder_skip_idx]
            # Stride to upsample by: the stride of the encoder stage below
            upsample_stride = strides[encoder_skip_idx + 1]
            upsample_stride_list = maybe_convert_scalar_to_list(conv_op, upsample_stride)
            upsample_stride_tuple = tuple(upsample_stride_list)

            # ConvTranspose: upsample from below
            self.upsample_layers.append(
                convtransp_op(
                    in_features_below, in_features_skip,
                    kernel_size=upsample_stride_tuple,
                    stride=upsample_stride_tuple,
                    bias=conv_bias,
                )
            )

            # After concat: in_features_skip (upsampled) + in_features_skip (skip)
            self.decode_stages.append(
                StackedConvBlocks(
                    num_convs=n_conv_per_stage[s],
                    conv_op=conv_op,
                    in_channels=2 * in_features_skip,
                    out_channels=in_features_skip,
                    kernel_size=kernel_sizes[encoder_skip_idx],
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
            )

            # Seg head at this level (always built, for checkpoint compat)
            self.seg_layers.append(
                conv_op(in_features_skip, num_classes, kernel_size=1, stride=1, bias=True)
            )

        self.features_per_stage = features_per_stage
        self.strides = strides
        self.n_stages = n_stages

    def forward(
        self, encoder_skips: List[torch.Tensor],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Parameters
        ----------
        encoder_skips : list of tensors
            Output from encoder, one per stage (shallow → deep).

        Returns
        -------
        If deep_supervision: list of tensors [full_res, ds_1, ds_2, ...]
        Otherwise: single tensor at full resolution.
        """
        # Start from the bottleneck (last encoder output)
        x = encoder_skips[-1]
        seg_outputs: List[torch.Tensor] = []

        for s in range(len(self.decode_stages)):
            skip_idx = len(encoder_skips) - 2 - s
            x = self.upsample_layers[s](x)
            x = torch.cat([x, encoder_skips[skip_idx]], dim=1)
            x = self.decode_stages[s](x)
            seg_outputs.append(self.seg_layers[s](x))

        # seg_outputs: [deepest_decoder → shallowest_decoder (full res)]
        # Reverse so index 0 = full resolution
        seg_outputs = seg_outputs[::-1]

        if self.deep_supervision:
            return seg_outputs
        return seg_outputs[0]

    def compute_conv_feature_map_size(
        self, input_size: Union[List[int], Tuple[int, ...]],
    ) -> int:
        # Compute encoder output sizes at each stage
        sizes = [list(input_size)]
        for s in range(self.n_stages - 1):
            stride_value = self.strides[s + 1]
            stride_list = (
                stride_value if isinstance(stride_value, list) else [stride_value] * len(input_size)
            )
            new_size = [sz // st for sz, st in zip(sizes[-1], stride_list)]
            sizes.append(new_size)

        total = 0
        for s in range(len(self.decode_stages)):
            # Decoder stage s operates at the resolution of encoder skip (n_stages-2-s)
            skip_idx = self.n_stages - 2 - s
            decode_size = sizes[skip_idx]
            total += cast(StackedConvBlocks, self.decode_stages[s]).compute_conv_feature_map_size(decode_size)
            # seg layer: num_classes * spatial
            total += int(np.prod(decode_size))  # simplified: 1 channel per class ignored
        return total


# ======================================================================
# PlainConvUNet
# ======================================================================

class PlainConvUNet(nn.Module):
    """nnUNet-style plain convolutional U-Net.

    Supports per-axis kernel sizes and strides, native deep supervision
    toggling, and analytical VRAM estimation.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: ConvOpType,
        kernel_sizes: Union[int, List[Union[int, List[int]]]],
        strides: Union[int, List[Union[int, List[int]]]],
        n_conv_per_stage: Union[int, List[int]] = 2,
        n_conv_per_stage_decoder: Union[int, List[int]] = 2,
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[nn.Module]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
    ):
        super().__init__()

        self.encoder = PlainConvEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first,
        )

        self.decoder = UNetDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            nonlin_first=nonlin_first,
        )

        self.num_classes = num_classes

    @property
    def deep_supervision(self) -> bool:
        return self.decoder.deep_supervision

    @deep_supervision.setter
    def deep_supervision(self, value: bool):
        self.decoder.deep_supervision = value

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder.forward_skips(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(
        self, input_size: Union[List[int], Tuple[int, ...]],
    ) -> int:
        return (
            self.encoder.compute_conv_feature_map_size(input_size)
            + self.decoder.compute_conv_feature_map_size(input_size)
        )

    @staticmethod
    def initialize(module: nn.Module):
        InitWeights_He(1e-2)(module)


# ======================================================================
# ResidualEncoderUNet
# ======================================================================

class ResidualEncoderUNet(nn.Module):
    """nnUNet-style U-Net with residual encoder (BasicBlockD).

    Decoder uses plain conv blocks (not residual), matching nnUNet v2
    ``ResidualEncoderUNet`` behaviour.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: ConvOpType,
        kernel_sizes: Union[int, List[Union[int, List[int]]]],
        strides: Union[int, List[Union[int, List[int]]]],
        n_blocks_per_stage: Union[int, List[int]] = 2,
        n_conv_per_stage_decoder: Union[int, List[int]] = 1,
        conv_bias: bool = True,
        norm_op: Optional[Type[nn.Module]] = None,
        norm_op_kwargs: Optional[dict] = None,
        dropout_op: Optional[Type[nn.Module]] = None,
        dropout_op_kwargs: Optional[dict] = None,
        nonlin: Optional[Type[nn.Module]] = None,
        nonlin_kwargs: Optional[dict] = None,
        deep_supervision: bool = False,
        nonlin_first: bool = False,
        stem_channels: Optional[int] = None,
        stochastic_depth_p: float = 0.0,
        squeeze_excitation: bool = False,
        squeeze_excitation_reduction_ratio: float = 0.0625,
    ):
        super().__init__()

        self.encoder = ResidualEncoder(
            input_channels=input_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            return_skips=True,
            nonlin_first=nonlin_first,
            stem_channels=stem_channels,
            stochastic_depth_p=stochastic_depth_p,
            squeeze_excitation=squeeze_excitation,
            squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio,
        )

        self.decoder = UNetDecoder(
            encoder=self.encoder,
            num_classes=num_classes,
            n_conv_per_stage=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
            nonlin_first=nonlin_first,
        )

        self.num_classes = num_classes

    @property
    def deep_supervision(self) -> bool:
        return self.decoder.deep_supervision

    @deep_supervision.setter
    def deep_supervision(self, value: bool):
        self.decoder.deep_supervision = value

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]:
        skips = self.encoder.forward_skips(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(
        self, input_size: Union[List[int], Tuple[int, ...]],
    ) -> int:
        return (
            self.encoder.compute_conv_feature_map_size(input_size)
            + self.decoder.compute_conv_feature_map_size(input_size)
        )

    @staticmethod
    def initialize(module: nn.Module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)
