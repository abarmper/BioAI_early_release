"""Model factory: builds models with topology-aware construction.

Each model receives the ``TopologySpec`` (pool/conv kernel sizes derived
from experiment plans) and maps it to its own constructor arguments.
Fixed-topology models (SwinUNETR, UNETR) use only ``img_size``.
"""
from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.nnunet_arch.building_blocks import InitWeights_He
from models.topology import (
    TopologySpec,
    topology_to_unet_args,
    topology_to_dynunet_args,
    topology_to_fixed_args,
    topology_to_nnunet_args,
)

logger = logging.getLogger(__name__)


def get_model(
    model_cfg: DictConfig,
    topology: TopologySpec,
    num_input_channels: int,
    num_output_channels: int,
    enable_deep_supervision: bool = True,
) -> nn.Module:
    """Build a segmentation model from config and topology.

    The instantiated architecture is determined dynamically from the
    provided ``TopologySpec`` together with the selected ``model_cfg``,
    rather than from a hard-coded network layout. ``TopologySpec`` is
    the basis for the architectural shape and is extracted or computed
    from the experiment plans, capturing patch size, voxel spacing,
    per-stage pooling, and convolution kernel sizes. Each model-specific
    builder then combines this topology information with the
    configuration in ``model_cfg`` to derive the constructor arguments
    and any model-dependent settings required by the underlying
    implementation.

    Parameters
    ----------
    model_cfg : DictConfig
        The model Hydra config group (e.g. ``configs/model/unet.yaml``).
    topology : TopologySpec
        Network topology derived from experiment plans.
    num_input_channels : int
        Number of image channels (from dataset.json).
    num_output_channels : int
        Number of segmentation classes (from labels dict).
    enable_deep_supervision : bool
        Whether deep supervision is enabled for this run.

    Returns
    -------
    nn.Module
        The instantiated model.
    """
    name = model_cfg.name.lower()

    builder = _MODEL_REGISTRY.get(name)
    if builder is None:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )

    model = builder(
        model_cfg=model_cfg,
        topology=topology,
        in_channels=num_input_channels,
        out_channels=num_output_channels,
        enable_deep_supervision=enable_deep_supervision,
    )

    logger.info(
        "Built model '%s' with %s parameters.",
        name,
        f"{sum(p.numel() for p in model.parameters()):,}",
    )
    return model


# ======================================================================
# Per-model builders
# ======================================================================

def _check_topology_mode(model_cfg: DictConfig, model_name: str, supported: set) -> None:
    """Raise ValueError if topology_mode in config is not in the supported set."""
    topology_mode = model_cfg.get("topology_mode", None)
    if topology_mode is not None and topology_mode not in supported:
        raise ValueError(
            f"Model '{model_name}' supports topology_mode in {sorted(supported)}, "
            f"but config specifies topology_mode='{topology_mode}'."
        )


def _build_unet(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "unet", {"dynamic"})
    from monai.networks.nets.unet import UNet

    args = topology_to_unet_args(
        topology,
        base_num_features=model_cfg.get("base_num_features", 32),
        max_num_features=model_cfg.get("max_num_features", 320),
        num_res_units=model_cfg.get("num_res_units", 2),
    )

    model = UNet(
        spatial_dims=args["spatial_dims"],
        in_channels=in_channels,
        out_channels=out_channels,
        channels=tuple(args["channels"]),
        strides=tuple(tuple(s) if isinstance(s, list) else s for s in args["strides"]), # type: ignore[list-item] # This is needed because there is an error in MONAI library. Actually, the monai can accept list of lists for strides to handle anisotropy.
        num_res_units=args["num_res_units"],
        kernel_size=tuple(args["kernel_size"]),
        up_kernel_size=model_cfg.get("up_kernel_size", 3),
        act=model_cfg.get("act", "prelu"),
        norm=model_cfg.get("norm", "instance"),
        dropout=model_cfg.get("dropout", 0.0),
        bias=model_cfg.get("bias", True),
    )

    if enable_deep_supervision:
        model = _wrap_deep_supervision_monai(
            model, topology, in_channels, out_channels, args["channels"],
        )

    return model


class _DynUNetDSAdapter(nn.Module):
    """Adapts DynUNet's stacked DS output to the list format expected by
    DeepSupervisionWrapper.

    MONAI DynUNet with ``deep_supervision=True`` returns a single tensor of
    shape ``(B, N, C, *spatial)`` where every head is interpolated to full
    resolution.  Our DS loss wrapper expects ``List[Tensor]`` with each
    entry at the *native decoder scale*.  Because DynUNet already upsamples,
    we unbind to a list and flag ``full_resolution_ds = True`` so the
    trainer provides full-resolution targets at every scale.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.deep_supervision = True
        self.full_resolution_ds = True

    def forward(self, x):
        out = self.model(x)
        if self.deep_supervision:
            if self.training:
                # (B, N, C, *spatial) → list of N tensors each (B, C, *spatial)
                return [out[:, i] for i in range(out.shape[1])]
            else:
                # Eval mode: DynUNet returns single full-res tensor.
                # Wrap in list for DS loss / validation_step compatibility.
                return [out]
        return out


def _build_dynunet(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "dynunet", {"dynamic"})
    from monai.networks.nets.dynunet import DynUNet

    args = topology_to_dynunet_args(topology)

    model = DynUNet(
        spatial_dims=args["spatial_dims"],
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=args["kernel_size"],
        strides=args["strides"],
        upsample_kernel_size=args["upsample_kernel_size"],
        deep_supervision=enable_deep_supervision,
        deep_supr_num=(
            model_cfg.get("deep_supervision_levels", None)
            if model_cfg.get("deep_supervision_levels", None) is not None
            else topology.n_stages - 2
        ) if enable_deep_supervision else 0,
        res_block=model_cfg.get("res_block", True),
    )

    if enable_deep_supervision:
        model = _DynUNetDSAdapter(model)

    return model


def _build_swinunetr(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "swinunetr", {"fixed"})
    from monai.networks.nets.swin_unetr import SwinUNETR

    args = topology_to_fixed_args(topology)

    model = SwinUNETR(
        spatial_dims=args["spatial_dims"],
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=model_cfg.get("feature_size", 48),
        depths=tuple(model_cfg.get("depths", [2, 2, 2, 2])),
        num_heads=tuple(model_cfg.get("num_heads", [3, 6, 12, 24])),
        norm_name=model_cfg.get("norm_name", "instance"),
        drop_rate=model_cfg.get("drop_rate", 0.0),
        attn_drop_rate=model_cfg.get("attn_drop_rate", 0.0),
        dropout_path_rate=model_cfg.get("dropout_path_rate", 0.0),
        use_checkpoint=model_cfg.get("use_checkpoint", False),
        use_v2=model_cfg.get("use_v2", False),
    )
    return model


def _build_unetr(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "unetr", {"fixed"})
    from monai.networks.nets.unetr import UNETR

    args = topology_to_fixed_args(topology)

    model = UNETR(
        img_size=args["img_size"],
        spatial_dims=args["spatial_dims"],
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=model_cfg.get("feature_size", 16),
        hidden_size=model_cfg.get("hidden_size", 768),
        num_heads=model_cfg.get("num_heads", 12),
        mlp_dim=model_cfg.get("mlp_dim", 3072),
    )
    return model


class _SegResNetDSAdapter(nn.Module):
    """Normalises SegResNetDS output for the DS loss / validator.

    SegResNetDS returns a list of multi-scale logits in train mode (deepest
    last) and a single full-resolution tensor in eval mode. The DS loss
    wrapper and full-volume validator expect a list in both modes (see
    ``_DynUNetDSAdapter``). Outputs are at native decoder scales, so we do
    *not* set ``full_resolution_ds`` — the trainer's standard scale list
    matches.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.deep_supervision = True

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return list(out)
        return [out]


def _build_segresnet(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "segresnet", {"fixed"})
    from monai.networks.nets.segresnet_ds import SegResNetDS

    dsdepth = (
        int(model_cfg.get("deep_supervision_levels", 4))
        if enable_deep_supervision
        else 1
    )
    blocks_up = model_cfg.get("blocks_up", None)
    spacing = getattr(topology, "spacing", None)
    resolution = tuple(spacing) if spacing is not None else None

    model = SegResNetDS(
        spatial_dims=topology.spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=model_cfg.get("init_filters", 32),
        blocks_down=tuple(model_cfg.get("blocks_down", [1, 2, 2, 4])),
        blocks_up=tuple(blocks_up) if blocks_up is not None else None,
        norm=model_cfg.get("norm", "batch"),
        act=model_cfg.get("act", "relu"),
        upsample_mode=model_cfg.get("upsample_mode", "deconv"),
        dsdepth=dsdepth,
        resolution=resolution,
    )

    if enable_deep_supervision:
        model = _SegResNetDSAdapter(model)

    return model


def _build_attention_unet(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "attention_unet", {"dynamic"})
    from monai.networks.nets.attentionunet import AttentionUnet

    args = topology_to_unet_args(topology)

    model = AttentionUnet(
        spatial_dims=args["spatial_dims"],
        in_channels=in_channels,
        out_channels=out_channels,
        channels=tuple(args["channels"]),
        strides=tuple(tuple(s) if isinstance(s, list) else s for s in args["strides"]), # type: ignore[list-item] # This is needed because there is an error in MONAI library. Actually, the monai can accept list of lists for strides to handle anisotropy.
        kernel_size=tuple(args["kernel_size"]),
        up_kernel_size=model_cfg.get("up_kernel_size", 3),
        dropout=model_cfg.get("dropout", 0.0), # type: ignore[list-item] # This is
        # needed because there is an error in MONAI library. Actually, the monai can accept list of lists for strides to handle anisotropy.
    )
    return model


# ======================================================================
# Op resolvers for nnUNet config strings
# ======================================================================

_OP_REGISTRY = {
    "InstanceNorm": {"2d": nn.InstanceNorm2d, "3d": nn.InstanceNorm3d},
    "BatchNorm": {"2d": nn.BatchNorm2d, "3d": nn.BatchNorm3d},
    "LeakyReLU": nn.LeakyReLU,
    "ReLU": nn.ReLU,
    "PReLU": nn.PReLU,
    "GELU": nn.GELU,
}


def _resolve_norm_op(name: Optional[str], spatial_dims: int) -> type | None:
    """Resolve norm name to class. Auto-selects 2D/3D variant."""
    if name is None:
        return nn.InstanceNorm3d if spatial_dims == 3 else nn.InstanceNorm2d
    entry = _OP_REGISTRY.get(name)
    if isinstance(entry, dict):
        return entry[f"{spatial_dims}d"]
    return entry


def _resolve_nonlin(name: Optional[str]) -> type:
    """Resolve nonlinearity name to class."""
    if name is None:
        return nn.LeakyReLU
    return _OP_REGISTRY.get(name, nn.LeakyReLU)


def _resolve_dropout_op(name: Optional[str], spatial_dims: int):
    """Resolve dropout name to class, or None."""
    if name is None:
        return None
    mapping = {
        "Dropout": {2: nn.Dropout2d, 3: nn.Dropout3d},
    }
    entry = mapping.get(name)
    if isinstance(entry, dict):
        return entry[spatial_dims]
    return None


# ======================================================================
# nnUNet builders
# ======================================================================

def _build_nnunet_plain(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "nnunet_plain", {"dynamic"})
    from models.nnunet_arch import PlainConvUNet

    args = topology_to_nnunet_args(
        topology,
        base_num_features=model_cfg.get("base_num_features", 32),
        max_num_features=model_cfg.get("max_num_features", 320),
        n_conv_per_stage_encoder=model_cfg.get("n_conv_per_stage_encoder", None),
        n_conv_per_stage_decoder=model_cfg.get("n_conv_per_stage_decoder", None),
    )

    norm_op = _resolve_norm_op(model_cfg.get("norm_op", None), topology.spatial_dims)
    nonlin = _resolve_nonlin(model_cfg.get("nonlin", None))
    dropout_op = _resolve_dropout_op(model_cfg.get("dropout_op", None), topology.spatial_dims)

    model = PlainConvUNet(
        input_channels=in_channels,
        num_classes=out_channels,
        n_stages=args["n_stages"],
        features_per_stage=args["features_per_stage"],
        conv_op=args["conv_op"],
        kernel_sizes=args["kernel_sizes"],
        strides=args["strides"],
        n_conv_per_stage=args["n_conv_per_stage_encoder"],
        n_conv_per_stage_decoder=args["n_conv_per_stage_decoder"],
        conv_bias=model_cfg.get("conv_bias", True),
        norm_op=norm_op,
        norm_op_kwargs=model_cfg.get("norm_op_kwargs", {"eps": 1e-5, "affine": True}),
        dropout_op=dropout_op,
        dropout_op_kwargs=model_cfg.get("dropout_op_kwargs", None),
        nonlin=nonlin,
        nonlin_kwargs=model_cfg.get("nonlin_kwargs", {"inplace": True}),
        deep_supervision=enable_deep_supervision,
    )

    model.apply(model.initialize)
    return model


def _build_nnunet_resenc(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "nnunet_resenc", {"dynamic"})
    from models.nnunet_arch import ResidualEncoderUNet

    args = topology_to_nnunet_args(
        topology,
        base_num_features=model_cfg.get("base_num_features", 32),
        max_num_features=model_cfg.get("max_num_features", 320),
    )

    n_stages = args["n_stages"]

    # Default n_blocks_per_stage: [1, 3, 4, 6, 6, 6, ...] up to n_stages
    n_blocks_default = [1, 3, 4, 6, 6, 6, 6, 6, 6, 6][:n_stages]
    n_blocks_per_stage = model_cfg.get("n_blocks_per_stage", None)
    if n_blocks_per_stage is None:
        n_blocks_per_stage = n_blocks_default

    n_conv_per_stage_decoder = model_cfg.get("n_conv_per_stage_decoder", None)
    if n_conv_per_stage_decoder is None:
        n_conv_per_stage_decoder = [1] * (n_stages - 1)

    norm_op = _resolve_norm_op(model_cfg.get("norm_op", None), topology.spatial_dims)
    nonlin = _resolve_nonlin(model_cfg.get("nonlin", None))
    dropout_op = _resolve_dropout_op(model_cfg.get("dropout_op", None), topology.spatial_dims)

    model = ResidualEncoderUNet(
        input_channels=in_channels,
        num_classes=out_channels,
        n_stages=n_stages,
        features_per_stage=args["features_per_stage"],
        conv_op=args["conv_op"],
        kernel_sizes=args["kernel_sizes"],
        strides=args["strides"],
        n_blocks_per_stage=n_blocks_per_stage,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=model_cfg.get("conv_bias", True),
        norm_op=norm_op,
        norm_op_kwargs=model_cfg.get("norm_op_kwargs", {"eps": 1e-5, "affine": True}),
        dropout_op=dropout_op,
        dropout_op_kwargs=model_cfg.get("dropout_op_kwargs", None),
        nonlin=nonlin,
        nonlin_kwargs=model_cfg.get("nonlin_kwargs", {"inplace": True}),
        deep_supervision=enable_deep_supervision,
        stem_channels=model_cfg.get("stem_channels", None),
        stochastic_depth_p=model_cfg.get("stochastic_depth_p", 0.0),
        squeeze_excitation=model_cfg.get("squeeze_excitation", False),
        squeeze_excitation_reduction_ratio=model_cfg.get(
            "squeeze_excitation_reduction_ratio", 0.0625
        ),
    )

    model.apply(model.initialize)
    return model


# ======================================================================
# nnFormer builder
# ======================================================================

def _build_nnformer(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "nnformer", {"fixed"})
    from models.nnformer_arch import nnFormer

    args = topology_to_fixed_args(topology)

    model = nnFormer(
        crop_size=list(args["img_size"]),
        embedding_dim=model_cfg.get("embedding_dim", 192),
        input_channels=in_channels,
        num_classes=out_channels,
        depths=list(model_cfg.get("depths", [2, 2, 2, 2])),
        num_heads=list(model_cfg.get("num_heads", [6, 12, 24, 48])),
        patch_size=list(model_cfg.get("patch_size", [2, 4, 4])),
        window_size=list(model_cfg.get("window_size", [4, 4, 8, 4])),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        drop_rate=model_cfg.get("drop_rate", 0.0),
        attn_drop_rate=model_cfg.get("attn_drop_rate", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.2),
        deep_supervision=enable_deep_supervision,
    )
    return model


# ======================================================================
# CSWin-UNet builder
# ======================================================================

def _build_cswinunet(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "cswinunet", {"fixed"})
    from models.cswinunet_arch import CSWinUNet

    args = topology_to_fixed_args(topology)

    embed_dim = model_cfg.get("embed_dim", None)
    if embed_dim is None:
        embed_dim = 96 if args["spatial_dims"] == 3 else 64

    num_heads = model_cfg.get("num_heads", None)
    if num_heads is None:
        num_heads = [3, 6, 12, 24] if args["spatial_dims"] == 3 else [2, 4, 8, 16]

    model = CSWinUNet(
        img_size=list(args["img_size"]),
        spatial_dims=args["spatial_dims"],
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=embed_dim,
        depths=list(model_cfg.get("depths", [1, 2, 9, 1])),
        num_heads=list(num_heads),
        split_sizes=list(model_cfg.get("split_sizes", [1, 2, 4, 4])),
        mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
        qkv_bias=model_cfg.get("qkv_bias", True),
        qk_scale=model_cfg.get("qk_scale", None),
        drop_rate=model_cfg.get("drop_rate", 0.0),
        attn_drop_rate=model_cfg.get("attn_drop_rate", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.2),
        use_chk=model_cfg.get("use_checkpoint", False),
        deep_supervision=enable_deep_supervision,
    )
    return model


# ======================================================================
# MambaHoME builder
# ======================================================================

def _build_mambahome(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "mambahome", {"fixed"})
    args = topology_to_fixed_args(topology)
    if args["spatial_dims"] != 3:
        raise ValueError("mambahome only supports 3D (spatial_dims=3).")
    if enable_deep_supervision:
        raise ValueError(
            "mambahome does not support deep supervision; "
            "set enable_deep_supervision=false in the model config."
        )

    # Lazy import: mamba_ssm is a CUDA extension and may not be installed
    # in environments that don't run mambahome. Failing here keeps unrelated
    # model builds working when the dep is missing.
    from models.mambahome_arch import MambaHoME

    return MambaHoME(
        in_chans=in_channels,
        out_chans=out_channels,
        img_size=list(args["img_size"]),
        spatial_dims=3,
        depths=list(model_cfg.get("depths", [2, 2, 2, 2])),
        feat_size=list(model_cfg.get("feat_size", [48, 96, 192, 384])),
        hidden_size=model_cfg.get("hidden_size", 768),
        norm_name=model_cfg.get("norm_name", "instance"),
        conv_block=model_cfg.get("conv_block", True),
        res_block=model_cfg.get("res_block", True),
        expert_mult=model_cfg.get("expert_mult", 2),
        moe_dropout=model_cfg.get("moe_dropout", 0.0),
        use_geglu=model_cfg.get("use_geglu", True),
        num_slots_per_expert_first=model_cfg.get("num_slots_per_expert_first", 4),
        experts_list=list(model_cfg.get("experts_list", [4, 8, 12, 16])),
        experts_list_second=list(model_cfg.get("experts_list_second", [8, 16, 24, 32])),
        group_list=list(model_cfg.get("group_list", [2048, 1024, 512, 256])),
    )


# ======================================================================
# U-Mamba builders (dynamic topology, anisotropic-aware)
# ======================================================================

def _build_umamba_common(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
    variant: str,
):
    """Shared topology→args mapping for both U-Mamba variants."""
    args = topology_to_nnunet_args(
        topology,
        base_num_features=model_cfg.get("base_num_features", 32),
        max_num_features=model_cfg.get("max_num_features", 320),
        n_conv_per_stage_encoder=model_cfg.get("n_conv_per_stage_encoder", None),
        n_conv_per_stage_decoder=model_cfg.get("n_conv_per_stage_decoder", None),
    )

    norm_op = _resolve_norm_op(model_cfg.get("norm_op", None), topology.spatial_dims)
    nonlin = _resolve_nonlin(model_cfg.get("nonlin", None))
    dropout_op = _resolve_dropout_op(model_cfg.get("dropout_op", None), topology.spatial_dims)

    common = dict(
        input_channels=in_channels,
        n_stages=args["n_stages"],
        features_per_stage=args["features_per_stage"],
        conv_op=args["conv_op"],
        kernel_sizes=args["kernel_sizes"],
        strides=args["strides"],
        n_conv_per_stage=args["n_conv_per_stage_encoder"],
        num_classes=out_channels,
        n_conv_per_stage_decoder=args["n_conv_per_stage_decoder"],
        conv_bias=model_cfg.get("conv_bias", True),
        norm_op=norm_op,
        norm_op_kwargs=model_cfg.get("norm_op_kwargs", {"eps": 1e-5, "affine": True}),
        dropout_op=dropout_op,
        dropout_op_kwargs=model_cfg.get("dropout_op_kwargs", None),
        nonlin=nonlin,
        nonlin_kwargs=model_cfg.get("nonlin_kwargs", {"inplace": True}),
        deep_supervision=enable_deep_supervision,
        stem_channels=model_cfg.get("stem_channels", None),
        trim_deep_stages=model_cfg.get("trim_deep_stages", True),
    )
    return common


def _build_umamba_bot(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "umamba_bot", {"dynamic"})
    # Lazy import: mamba_ssm is a CUDA extension and may be missing in
    # CPU-only envs. Importing here keeps unrelated model builds working.
    from models.umamba_arch import UMambaBot

    common = _build_umamba_common(
        model_cfg, topology, in_channels, out_channels, enable_deep_supervision,
        variant="bot",
    )
    model = UMambaBot(**common)
    model.apply(InitWeights_He(1e-2))
    return model


def _build_umamba_enc(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "umamba_enc", {"dynamic"})
    from models.umamba_arch import UMambaEnc

    common = _build_umamba_common(
        model_cfg, topology, in_channels, out_channels, enable_deep_supervision,
        variant="enc",
    )
    # UMambaEnc additionally needs the input patch size so per-stage
    # feature-map sizes (and patch- vs. channel-token decisions) match the
    # data loader.
    model = UMambaEnc(input_size=tuple(topology.patch_size), **common)
    model.apply(InitWeights_He(1e-2))
    return model


# ======================================================================
# MedNeXt builder
# ======================================================================

_MEDNEXT_PRESETS = {
    "S": {
        "exp_r": 2,
        "block_counts": [2, 2, 2, 2, 2, 2, 2, 2, 2],
        "checkpoint_style": None,
    },
    "B": {
        "exp_r": [2, 3, 4, 4, 4, 4, 4, 3, 2],
        "block_counts": [2, 2, 2, 2, 2, 2, 2, 2, 2],
        "checkpoint_style": None,
    },
    "M": {
        "exp_r": [2, 3, 4, 4, 4, 4, 4, 3, 2],
        "block_counts": [3, 4, 4, 4, 4, 4, 4, 4, 3],
        "checkpoint_style": "outside_block",
    },
    "L": {
        "exp_r": [3, 4, 8, 8, 8, 8, 8, 4, 3],
        "block_counts": [3, 4, 8, 8, 8, 8, 8, 4, 3],
        "checkpoint_style": "outside_block",
    },
}


def _build_mednext(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "mednext", {"fixed"})
    from models.mednext_arch import MedNeXt

    # Resolve size preset
    size = model_cfg.get("size", "B").upper()
    if size not in _MEDNEXT_PRESETS:
        raise ValueError(
            f"Unknown MedNeXt size '{size}'. Available: {list(_MEDNEXT_PRESETS.keys())}"
        )
    preset = _MEDNEXT_PRESETS[size]

    # Allow config overrides for preset fields
    exp_r = model_cfg.get("exp_r", None)
    if exp_r is None:
        exp_r = preset["exp_r"]
    else:
        exp_r = list(exp_r) if not isinstance(exp_r, int) else exp_r

    block_counts = model_cfg.get("block_counts", None)
    if block_counts is None:
        block_counts = preset["block_counts"]
    else:
        block_counts = list(block_counts)

    checkpoint_style = model_cfg.get("checkpoint_style", None)
    if checkpoint_style is None:
        if model_cfg.get("use_default_checkpoint_style", True):
            checkpoint_style = preset["checkpoint_style"]
        else:
            checkpoint_style = None

    # Convert spatial_dims (int) to dim string ('2d'/'3d')
    dim = f"{topology.spatial_dims}d"

    model = MedNeXt(
        in_channels=in_channels,
        n_channels=model_cfg.get("n_channels", 32),
        n_classes=out_channels,
        exp_r=exp_r,
        kernel_size=model_cfg.get("kernel_size", 3),
        enc_kernel_size=model_cfg.get("enc_kernel_size", None),
        dec_kernel_size=model_cfg.get("dec_kernel_size", None),
        deep_supervision=enable_deep_supervision,
        do_res=model_cfg.get("do_res", True),
        do_res_up_down=model_cfg.get("do_res_up_down", True),
        checkpoint_style=checkpoint_style,
        block_counts=block_counts,
        norm_type=model_cfg.get("norm_type", "group"),
        dim=dim,
        grn=model_cfg.get("grn", False),
        layer_scale_init_value=model_cfg.get("layer_scale_init_value", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
        skip_connection_mode=model_cfg.get("skip_connection_mode", "add"),
        num_stages=_resolve_mednext_num_stages(model_cfg),
    )

    return model


# ======================================================================
# nnMedNeXt builder (dynamic topology)
# ======================================================================

_N_MEDNEXT_STAGES = 5  # 4 encoder + 1 bottleneck


def _topology_to_nnmednext(
    topology: TopologySpec,
    kernel_size: int = 3,
    num_stages: int = _N_MEDNEXT_STAGES,
) -> Dict[str, Any]:
    """Map topology to nnMedNeXt per-stage kernel sizes and strides.

    Truncates topology to ``num_stages`` stages (MedNeXt's fixed depth, 5 by
    default; 6 for the optional deeper variant). Maps per-axis kernel values:
    1 stays 1 (axis has insufficient resolution), >1 becomes ``kernel_size``
    from config (the "large" kernel, e.g. 3, 5, or 7).
    """
    n_topo = topology.n_stages
    spatial_dims = topology.spatial_dims

    conv_kernels = list(topology.conv_kernel_sizes[:num_stages])
    pool_strides = list(topology.pool_op_kernel_sizes[:num_stages])

    # Pad if topology has fewer than num_stages stages (e.g. small datasets):
    # the extra deepest level defaults to isotropic 2x downsampling.
    while len(conv_kernels) < num_stages:
        conv_kernels.append([kernel_size] * spatial_dims)
    while len(pool_strides) < num_stages:
        pool_strides.append([2] * spatial_dims)

    # Map kernel values: 1 → 1, >1 → kernel_size
    mapped_kernels = []
    for stage_kernels in conv_kernels:
        mapped_kernels.append(
            tuple(1 if k == 1 else kernel_size for k in stage_kernels)
        )

    # Ensure strides are tuples
    mapped_strides = [tuple(s) for s in pool_strides]

    return {
        "conv_kernel_sizes": mapped_kernels,
        "strides": mapped_strides,
    }


def _build_nnmednext(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "nnmednext", {"dynamic"})
    from models.mednext_arch import nnMedNeXt

    # Resolve size preset (reuses same presets as fixed MedNeXt)
    size = model_cfg.get("size", "B").upper()
    if size not in _MEDNEXT_PRESETS:
        raise ValueError(
            f"Unknown nnMedNeXt size '{size}'. Available: {list(_MEDNEXT_PRESETS.keys())}"
        )
    preset = _MEDNEXT_PRESETS[size]

    # Allow config overrides for preset fields
    exp_r = model_cfg.get("exp_r", None)
    if exp_r is None:
        exp_r = preset["exp_r"]
    else:
        exp_r = list(exp_r) if not isinstance(exp_r, int) else exp_r

    block_counts = model_cfg.get("block_counts", None)
    if block_counts is None:
        block_counts = preset["block_counts"]
    else:
        block_counts = list(block_counts)

    checkpoint_style = model_cfg.get("checkpoint_style", None)
    if checkpoint_style is None:
        if model_cfg.get("use_default_checkpoint_style", True):
            checkpoint_style = preset["checkpoint_style"]
        else:
            checkpoint_style = None

    num_stages = _resolve_mednext_num_stages(model_cfg)

    # Map topology to per-stage kernels and strides
    topo_args = _topology_to_nnmednext(
        topology,
        kernel_size=model_cfg.get("kernel_size", 3),
        num_stages=num_stages,
    )

    # Convert spatial_dims (int) to dim string ('2d'/'3d')
    dim = f"{topology.spatial_dims}d"

    model = nnMedNeXt(
        in_channels=in_channels,
        n_channels=model_cfg.get("n_channels", 32),
        n_classes=out_channels,
        exp_r=exp_r,
        conv_kernel_sizes=topo_args["conv_kernel_sizes"],
        strides=topo_args["strides"],
        deep_supervision=enable_deep_supervision,
        do_res=model_cfg.get("do_res", True),
        do_res_up_down=model_cfg.get("do_res_up_down", True),
        checkpoint_style=checkpoint_style,
        block_counts=block_counts,
        norm_type=model_cfg.get("norm_type", "group"),
        dim=dim,
        grn=model_cfg.get("grn", False),
        layer_scale_init_value=model_cfg.get("layer_scale_init_value", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
        skip_connection_mode=model_cfg.get("skip_connection_mode", "add"),
        num_stages=num_stages,
    )

    return model


# ======================================================================
# MedMoENeXt / nnMedMoENeXt builders (Soft MoE in bottleneck)
# ======================================================================

def _resolve_mednext_num_stages(model_cfg):
    """Resolve and validate the number of resolution levels (5 or 6).

    5 (default) is the original MedNeXt depth (4 encoder + bottleneck); 6 adds
    one deeper level (5 encoder + bottleneck, bottleneck at 32n).
    """
    num_stages = int(model_cfg.get("num_stages", 5))
    if num_stages not in (5, 6):
        raise ValueError(f"model.num_stages must be 5 or 6, got {num_stages}.")
    return num_stages


def _resolve_mednext_preset(model_cfg):
    """Resolve size preset and config overrides for MedNeXt-family models."""
    size = model_cfg.get("size", "B").upper()
    if size not in _MEDNEXT_PRESETS:
        raise ValueError(
            f"Unknown MedNeXt size '{size}'. Available: {list(_MEDNEXT_PRESETS.keys())}"
        )
    preset = _MEDNEXT_PRESETS[size]

    exp_r = model_cfg.get("exp_r", None)
    if exp_r is None:
        exp_r = preset["exp_r"]
    else:
        exp_r = list(exp_r) if not isinstance(exp_r, int) else exp_r

    block_counts = model_cfg.get("block_counts", None)
    if block_counts is None:
        block_counts = preset["block_counts"]
    else:
        block_counts = list(block_counts)

    checkpoint_style = model_cfg.get("checkpoint_style", None)
    if checkpoint_style is None:
        if model_cfg.get("use_default_checkpoint_style", True):
            checkpoint_style = preset["checkpoint_style"]
        else:
            checkpoint_style = None

    return exp_r, block_counts, checkpoint_style


def _maybe_enable_tf32_for_routing(model_cfg: DictConfig) -> None:
    """Enable TF32 for fp32 matmul when ``routing_fp32`` is on.

    ``routing_fp32=True`` runs the MoE forward in fp32 outside autocast, which
    on Ampere+ GPUs would otherwise leave the einsum matmuls on the slower
    full-fp32 path and trigger PyTorch's "TensorFloat32 ... not enabled"
    warning. Setting the matmul precision to ``'high'`` lets those fp32
    matmuls use TF32 Tensor Cores — much faster, with accuracy loss small
    enough to be invisible for medical-image segmentation training.

    Skips the change if the user (or another part of the codebase) has
    already moved away from PyTorch's ``'highest'`` default, so an explicit
    reproducibility choice is preserved.
    """
    if not model_cfg.get("routing_fp32", True):
        return
    if torch.get_float32_matmul_precision() != "highest":
        return
    torch.set_float32_matmul_precision("high")
    logger.info(
        "Enabled TF32 for fp32 matmul (model.routing_fp32=True). "
        "Override globally with torch.set_float32_matmul_precision() if you "
        "need strict fp32."
    )


def _build_medmoenext(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "medmoenext", {"fixed"})
    from models.mednext_arch import MedMoENeXt

    exp_r, block_counts, checkpoint_style = _resolve_mednext_preset(model_cfg)
    dim = f"{topology.spatial_dims}d"

    model = MedMoENeXt(
        in_channels=in_channels,
        n_channels=model_cfg.get("n_channels", 32),
        n_classes=out_channels,
        exp_r=exp_r,
        kernel_size=model_cfg.get("kernel_size", 3),
        enc_kernel_size=model_cfg.get("enc_kernel_size", None),
        dec_kernel_size=model_cfg.get("dec_kernel_size", None),
        deep_supervision=enable_deep_supervision,
        do_res=model_cfg.get("do_res", True),
        do_res_up_down=model_cfg.get("do_res_up_down", True),
        checkpoint_style=checkpoint_style,
        block_counts=block_counts,
        norm_type=model_cfg.get("norm_type", "group"),
        dim=dim,
        grn=model_cfg.get("grn", False),
        layer_scale_init_value=model_cfg.get("layer_scale_init_value", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
        skip_connection_mode=model_cfg.get("skip_connection_mode", "add"),
        num_stages=_resolve_mednext_num_stages(model_cfg),
        # MoE parameters
        num_experts=model_cfg.get("num_experts", 4),
        expansion_factor_divisor=model_cfg.get("expansion_factor_divisor", 1),
        moe_mode=model_cfg.get("moe_mode", "conservative"),
        use_positional_embeddings=model_cfg.get("use_positional_embeddings", False),
        pos_emb_all_moe_blocks=model_cfg.get("pos_emb_all_moe_blocks", False),
        normalize_routing=model_cfg.get("normalize_routing", True),
        moe_norm_type=model_cfg.get("moe_norm_type", None),
        routing_fp32=model_cfg.get("routing_fp32", True),
        log_scale_clamp_max=model_cfg.get("log_scale_clamp_max", None),
        normalize_eps=model_cfg.get("normalize_eps", None),
    )

    _maybe_enable_tf32_for_routing(model_cfg)
    return model


def _build_nnmedmoenext(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "nnmedmoenext", {"dynamic"})
    from models.mednext_arch import nnMedMoENeXt

    exp_r, block_counts, checkpoint_style = _resolve_mednext_preset(model_cfg)
    num_stages = _resolve_mednext_num_stages(model_cfg)

    topo_args = _topology_to_nnmednext(
        topology,
        kernel_size=model_cfg.get("kernel_size", 3),
        num_stages=num_stages,
    )
    dim = f"{topology.spatial_dims}d"

    model = nnMedMoENeXt(
        in_channels=in_channels,
        n_channels=model_cfg.get("n_channels", 32),
        n_classes=out_channels,
        exp_r=exp_r,
        conv_kernel_sizes=topo_args["conv_kernel_sizes"],
        strides=topo_args["strides"],
        deep_supervision=enable_deep_supervision,
        do_res=model_cfg.get("do_res", True),
        do_res_up_down=model_cfg.get("do_res_up_down", True),
        checkpoint_style=checkpoint_style,
        block_counts=block_counts,
        norm_type=model_cfg.get("norm_type", "group"),
        dim=dim,
        grn=model_cfg.get("grn", False),
        layer_scale_init_value=model_cfg.get("layer_scale_init_value", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
        skip_connection_mode=model_cfg.get("skip_connection_mode", "add"),
        num_stages=num_stages,
        # MoE parameters
        num_experts=model_cfg.get("num_experts", 4),
        expansion_factor_divisor=model_cfg.get("expansion_factor_divisor", 1),
        moe_mode=model_cfg.get("moe_mode", "conservative"),
        use_positional_embeddings=model_cfg.get("use_positional_embeddings", False),
        pos_emb_all_moe_blocks=model_cfg.get("pos_emb_all_moe_blocks", False),
        normalize_routing=model_cfg.get("normalize_routing", True),
        moe_norm_type=model_cfg.get("moe_norm_type", None),
        routing_fp32=model_cfg.get("routing_fp32", True),
        log_scale_clamp_max=model_cfg.get("log_scale_clamp_max", None),
        normalize_eps=model_cfg.get("normalize_eps", None),
    )

    _maybe_enable_tf32_for_routing(model_cfg)
    return model


# ======================================================================
# MedMoEXNeXt / nnMedMoEXNeXt builders (Soft MoE in encoder + bottleneck or all stages)
# Ablation-only variants of MedMoENeXt / nnMedMoENeXt.
# ======================================================================

def _build_medmoexnext(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "medmoexnext", {"fixed"})
    from models.mednext_arch import MedMoEXNeXt

    exp_r, block_counts, checkpoint_style = _resolve_mednext_preset(model_cfg)
    dim = f"{topology.spatial_dims}d"

    model = MedMoEXNeXt(
        in_channels=in_channels,
        n_channels=model_cfg.get("n_channels", 32),
        n_classes=out_channels,
        exp_r=exp_r,
        kernel_size=model_cfg.get("kernel_size", 3),
        enc_kernel_size=model_cfg.get("enc_kernel_size", None),
        dec_kernel_size=model_cfg.get("dec_kernel_size", None),
        deep_supervision=enable_deep_supervision,
        do_res=model_cfg.get("do_res", True),
        do_res_up_down=model_cfg.get("do_res_up_down", True),
        checkpoint_style=checkpoint_style,
        block_counts=block_counts,
        norm_type=model_cfg.get("norm_type", "group"),
        dim=dim,
        grn=model_cfg.get("grn", False),
        layer_scale_init_value=model_cfg.get("layer_scale_init_value", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
        # MoE parameters
        num_experts=model_cfg.get("num_experts", 4),
        expansion_factor_divisor=model_cfg.get("expansion_factor_divisor", 1),
        moe_mode=model_cfg.get("moe_mode", "conservative"),
        use_positional_embeddings=model_cfg.get("use_positional_embeddings", False),
        pos_emb_all_moe_blocks=model_cfg.get("pos_emb_all_moe_blocks", False),
        normalize_routing=model_cfg.get("normalize_routing", True),
        routing_fp32=model_cfg.get("routing_fp32", True),
        log_scale_clamp_max=model_cfg.get("log_scale_clamp_max", None),
        normalize_eps=model_cfg.get("normalize_eps", None),
        # Extended-scope parameters
        moe_scope=model_cfg.get("moe_scope", "encoder_bottleneck"),
        pos_emb_max_len=model_cfg.get("pos_emb_max_len", 64),
    )

    _maybe_enable_tf32_for_routing(model_cfg)
    return model


def _build_nnmedmoexnext(
    model_cfg: DictConfig,
    topology: TopologySpec,
    in_channels: int,
    out_channels: int,
    enable_deep_supervision: bool,
) -> nn.Module:
    _check_topology_mode(model_cfg, "nnmedmoexnext", {"dynamic"})
    from models.mednext_arch import nnMedMoEXNeXt

    exp_r, block_counts, checkpoint_style = _resolve_mednext_preset(model_cfg)

    topo_args = _topology_to_nnmednext(
        topology,
        kernel_size=model_cfg.get("kernel_size", 3),
    )
    dim = f"{topology.spatial_dims}d"

    model = nnMedMoEXNeXt(
        in_channels=in_channels,
        n_channels=model_cfg.get("n_channels", 32),
        n_classes=out_channels,
        exp_r=exp_r,
        conv_kernel_sizes=topo_args["conv_kernel_sizes"],
        strides=topo_args["strides"],
        deep_supervision=enable_deep_supervision,
        do_res=model_cfg.get("do_res", True),
        do_res_up_down=model_cfg.get("do_res_up_down", True),
        checkpoint_style=checkpoint_style,
        block_counts=block_counts,
        norm_type=model_cfg.get("norm_type", "group"),
        dim=dim,
        grn=model_cfg.get("grn", False),
        layer_scale_init_value=model_cfg.get("layer_scale_init_value", 0.0),
        drop_path_rate=model_cfg.get("drop_path_rate", 0.0),
        # MoE parameters
        num_experts=model_cfg.get("num_experts", 4),
        expansion_factor_divisor=model_cfg.get("expansion_factor_divisor", 1),
        moe_mode=model_cfg.get("moe_mode", "conservative"),
        use_positional_embeddings=model_cfg.get("use_positional_embeddings", False),
        pos_emb_all_moe_blocks=model_cfg.get("pos_emb_all_moe_blocks", False),
        normalize_routing=model_cfg.get("normalize_routing", True),
        routing_fp32=model_cfg.get("routing_fp32", True),
        log_scale_clamp_max=model_cfg.get("log_scale_clamp_max", None),
        normalize_eps=model_cfg.get("normalize_eps", None),
        # Extended-scope parameters
        moe_scope=model_cfg.get("moe_scope", "encoder_bottleneck"),
        pos_emb_max_len=model_cfg.get("pos_emb_max_len", 64),
    )

    _maybe_enable_tf32_for_routing(model_cfg)
    return model


# ======================================================================
# Deep supervision wrapper for MONAI UNet
# ======================================================================

class MonaiUNetDeepSupervision(nn.Module):
    """Wraps a MONAI UNet to produce multi-scale outputs for deep supervision.

    Adds side-output convolutions at each decoder level and upsamples them
    to the full resolution.
    """

    def __init__(self, base_model, topology, in_channels, out_channels, channels):
        super().__init__()
        self.base_model = base_model
        self.deep_supervision = True
        # For MONAI UNet we cannot easily tap into decoder levels.
        # Deep supervision support is primarily via DynUNet (which has it built-in).
        # For plain UNet, we disable DS and log a warning.
        logger.warning(
            "Deep supervision for MONAI UNet is limited. "
            "Consider using DynUNet for full deep supervision support."
        )

    def forward(self, x):
        out = self.base_model(x)
        if self.deep_supervision:
            return [out]  # single-scale output as a list for compatibility
        return out


def _wrap_deep_supervision_monai(model, topology, in_channels, out_channels, channels):
    """Wrap a MONAI UNet model for deep supervision compatibility."""
    return MonaiUNetDeepSupervision(model, topology, in_channels, out_channels, channels)


# ======================================================================
# Registry
# ======================================================================

_MODEL_REGISTRY = {
    "unet": _build_unet,
    "dynunet": _build_dynunet,
    "swinunetr": _build_swinunetr,
    "unetr": _build_unetr,
    "segresnet": _build_segresnet,
    "attention_unet": _build_attention_unet,
    "nnunet_plain": _build_nnunet_plain,
    "nnunet_resenc": _build_nnunet_resenc,
    "nnformer": _build_nnformer,
    "cswinunet": _build_cswinunet,
    "mambahome": _build_mambahome,
    "umamba_bot": _build_umamba_bot,
    "umamba_enc": _build_umamba_enc,
    "mednext": _build_mednext,
    "nnmednext": _build_nnmednext,
    "medmoenext": _build_medmoenext,
    "nnmedmoenext": _build_nnmedmoenext,
    "medmoexnext": _build_medmoexnext,
    "nnmedmoexnext": _build_nnmedmoexnext,
}


# ======================================================================
# Per-model patch-size divisibility constraints
# ======================================================================
#
# Fixed-topology architectures impose their own input-shape divisibility
# requirements that the nnUNet-style planner in preprocessing/plan.py does
# NOT know about (the planner is purely data-driven, model-agnostic). The
# registered builders below expose each model's per-axis divisibility so
# `extract_topology_from_plans` can widen the planner's constraint before
# constructing both the network and the data loader.
#
# Convention: return a list of positive integers of length ``spatial_dims``
# (`[1, ...]` means "no constraint beyond the planner's").


def _no_constraint(model_cfg: DictConfig, spatial_dims: int) -> List[int]:
    return [1] * spatial_dims


def _constraint_swinunetr(model_cfg: DictConfig, spatial_dims: int) -> List[int]:
    # MONAI SwinUNETR hardcodes self.patch_size = 2 and checks input % 2**5.
    return [32] * spatial_dims


def _constraint_unetr(model_cfg: DictConfig, spatial_dims: int) -> List[int]:
    # MONAI UNETR hardcodes a patch embedding of 16 on every axis.
    return [16] * spatial_dims


def _mednext_n_down_from_block_counts(block_counts: List[int]) -> int:
    """MedNeXt block_counts has (n_enc + 1 bottleneck + n_dec) entries with
    n_enc == n_dec. The number of 2x downsamples equals n_enc."""
    n_blocks = len(block_counts)
    if n_blocks < 3 or n_blocks % 2 == 0:
        # Fall back to the default 5-stage layout (4 downs) if something
        # unexpected slips through. _resolve_mednext_preset ensures this is
        # a 9-entry list for all current presets.
        return 4
    return (n_blocks - 1) // 2


def _constraint_mednext(model_cfg: DictConfig, spatial_dims: int) -> List[int]:
    # Fixed-topology MedNeXt / MedMoENeXt downsamples by exactly 2x per stage.
    # num_stages == 6 adds one more downsample (5 downs => divisible by 32).
    num_stages = _resolve_mednext_num_stages(model_cfg)
    if num_stages == 6:
        n_down = 5
    else:
        size = str(model_cfg.get("size", "B")).upper()
        preset = _MEDNEXT_PRESETS.get(size, _MEDNEXT_PRESETS["B"])
        block_counts = model_cfg.get("block_counts", None) or preset["block_counts"]
        n_down = _mednext_n_down_from_block_counts(list(block_counts))
    return [2 ** n_down] * spatial_dims


def _constraint_nnformer(model_cfg: DictConfig, spatial_dims: int) -> List[int]:
    """nnFormer constraint: patch-embedding size × encoder-down factor per axis,
    LCM'd with each axis' largest window size so every stage's window attention
    divides the feature map.

    Constraint per axis (for 3D):
        patch_embed[i] * 2 ** (len(depths) - 1)
    LCM'd with max(window_size) * patch_embed[i]  (window size is applied at every
    stage after patch embedding).
    """
    depths = list(model_cfg.get("depths", [2, 2, 2, 2]))
    patch_embed = list(model_cfg.get("patch_size", [2, 4, 4]))
    window_size = list(model_cfg.get("window_size", [4, 4, 8, 4]))
    n_down = max(len(depths) - 1, 0)
    # Typical layout: 3 axes in the same order the planner uses.
    # Pad / truncate to the current spatial_dims.
    if len(patch_embed) < spatial_dims:
        patch_embed = [1] * (spatial_dims - len(patch_embed)) + patch_embed
    patch_embed = patch_embed[-spatial_dims:]

    max_window = max(window_size) if window_size else 1
    base = [pe * (2 ** n_down) for pe in patch_embed]
    with_window = [math.lcm(b, max_window * pe) for b, pe in zip(base, patch_embed)]
    return with_window


def _constraint_cswinunet(model_cfg: DictConfig, spatial_dims: int) -> List[int]:
    """CSWinUNet constraint: stem stride 4 + (n_stages - 1) merges of stride 2.

    Per non-last encoder stage ``i`` the feature resolution
    ``img / (4 * 2 ** i)`` must be divisible by ``split_sizes[i]`` so the
    cross-shaped windows tile cleanly. The deepest stage uses full-window
    attention and adds no extra factor, but the bottleneck resolution still
    needs to be a clean integer, which requires ``img`` divisible by
    ``4 * 2 ** (n_stages - 1)``.
    """
    split_sizes = list(model_cfg.get("split_sizes", [1, 2, 4, 4]))
    n_stages = len(split_sizes)
    # Bottleneck divisibility: img / (4 * 2^(n_stages-1)) must be a positive integer.
    constraint = 4 * (2 ** (n_stages - 1))
    # Non-last stages: feature_res must be divisible by split_sizes[i].
    for i in range(n_stages - 1):
        constraint = math.lcm(constraint, 4 * (2 ** i) * max(1, int(split_sizes[i])))
    return [constraint] * spatial_dims


def _constraint_mambahome(model_cfg: DictConfig, spatial_dims: int) -> List[int]:
    """MambaHoME constraint: stem stride 2 + three stride-2 stages = 16x downsampling on every axis."""
    return [16] * spatial_dims


_MODEL_CONSTRAINT_REGISTRY: Dict[str, Callable[[DictConfig, int], List[int]]] = {
    "swinunetr": _constraint_swinunetr,
    "unetr": _constraint_unetr,
    "mednext": _constraint_mednext,
    "medmoenext": _constraint_mednext,
    "medmoexnext": _constraint_mednext,
    "nnformer": _constraint_nnformer,
    "cswinunet": _constraint_cswinunet,
    "mambahome": _constraint_mambahome,
    # Dynamic-topology models and segresnet use _no_constraint by default.
}


def get_model_patch_constraint(model_cfg: DictConfig, spatial_dims: int) -> List[int]:
    """Return the per-axis patch-size divisibility a model requires.

    Used by :func:`models.topology.extract_topology_from_plans` to widen the
    planner's data-driven divisibility so fixed-topology models (SwinUNETR,
    UNETR, MedNeXt, nnFormer, ...) receive a patch their architecture accepts.

    Dynamic-topology models (``unet``, ``dynunet``, ``attention_unet``,
    ``nnunet_plain``, ``nnunet_resenc``, ``nnmednext``, ``nnmedmoenext``) and
    ``segresnet`` return ``[1, ...]`` — they adapt to whatever the planner
    produced.
    """
    name = str(model_cfg.get("name", "")).lower()
    fn = _MODEL_CONSTRAINT_REGISTRY.get(name, _no_constraint)
    return fn(model_cfg, spatial_dims)
