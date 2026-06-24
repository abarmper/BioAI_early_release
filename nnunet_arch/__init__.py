"""nnUNet dynamic network architectures ported for BioAI.

Provides PlainConvUNet and ResidualEncoderUNet with full per-axis kernel
support, native deep supervision toggling, and analytical VRAM estimation
via ``compute_conv_feature_map_size()``.

Ported from ``dynamic_network_architectures`` (MIC-DKFZ), consolidated
from 14 files into 3 modules.
"""
from models.nnunet_arch.unet import PlainConvUNet, ResidualEncoderUNet

__all__ = ["PlainConvUNet", "ResidualEncoderUNet"]
