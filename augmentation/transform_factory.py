"""Augmentation pipeline factory using batchgeneratorsv2 transforms.

Builds training/validation transform pipelines from augmentation config
files (nnunet, fast, extended).
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
from omegaconf import DictConfig

from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform

logger = logging.getLogger(__name__)


def build_training_transforms(
    aug_cfg: DictConfig,
    patch_size: Union[List[int], Tuple[int, ...]],
    rotation_for_DA: Tuple[float, float],
    deep_supervision_scales: Optional[List[List[float]]],
    mirror_axes: Tuple[int, ...],
    do_dummy_2d_data_aug: bool,
    use_mask_for_norm: Optional[List[bool]] = None,
    is_cascaded: bool = False,
    foreground_labels: Optional[List[int]] = None,
) -> BasicTransform:
    """Build training augmentation pipeline from config.

    Parameters
    ----------
    aug_cfg : DictConfig
        Augmentation config (e.g. from ``configs/augmentation/nnunet.yaml``).
    patch_size : sequence of int
        Final patch size.
    rotation_for_DA : tuple of float
        Rotation range in radians ``(min, max)``.
    deep_supervision_scales : list of list of float or None
        Scales for deep supervision downsampling.
    mirror_axes : tuple of int
        Axes allowed for mirroring.
    do_dummy_2d_data_aug : bool
        Apply 2D augmentation on 3D data (for anisotropic volumes).
    use_mask_for_norm : list of bool or None
        Per-channel flag for mask-based normalisation.
    is_cascaded : bool
        Whether this is a cascade configuration.
    foreground_labels : list of int or None
        Foreground label indices (for cascade augmentation).
    """
    transforms = []

    # --- Dummy 2D handling ---
    if do_dummy_2d_data_aug:
        ignore_axes = (0,)
        transforms.append(Convert3DTo2DTransform())
        patch_size_spatial = tuple(patch_size[1:])
    else:
        patch_size_spatial = tuple(patch_size)
        ignore_axes = None

    # --- Spatial transform (rotation + scaling) ---
    spatial_cfg = aug_cfg.get("spatial", {})
    if spatial_cfg.get("enabled", True):
        scaling_range = tuple(spatial_cfg.get("scaling_range", [0.7, 1.4]))
        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=spatial_cfg.get("p_elastic", 0.0),
                elastic_deform_scale=tuple(spatial_cfg.get("elastic_deform_scale", (0, 0.2))),
                elastic_deform_magnitude=tuple(spatial_cfg.get("elastic_deform_magnitude", (0, 0.2))),
                p_rotation=spatial_cfg.get("p_rotation", 0.2),
                rotation=rotation_for_DA,
                p_scaling=spatial_cfg.get("p_scaling", 0.2),
                scaling=scaling_range,
                p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False,
            )
        )

    if do_dummy_2d_data_aug:
        transforms.append(Convert2DTo3DTransform())

    # --- Gaussian noise ---
    noise_cfg = aug_cfg.get("gaussian_noise", {})
    if noise_cfg.get("enabled", True):
        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=tuple(noise_cfg.get("variance", [0, 0.1])),
                p_per_channel=1,
                synchronize_channels=True,
            ),
            apply_probability=noise_cfg.get("probability", 0.1),
        ))

    # --- Gaussian blur ---
    blur_cfg = aug_cfg.get("gaussian_blur", {})
    if blur_cfg.get("enabled", True):
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=tuple(blur_cfg.get("sigma", [0.5, 1.0])),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5,
                benchmark=True,
            ),
            apply_probability=blur_cfg.get("probability", 0.2),
        ))

    # --- Brightness ---
    bright_cfg = aug_cfg.get("brightness", {})
    if bright_cfg.get("enabled", True):
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast(tuple(bright_cfg.get("range", [0.75, 1.25]))).sample_contrast,  # type: ignore[arg-type]
                synchronize_channels=False,
                p_per_channel=1,
            ),
            apply_probability=bright_cfg.get("probability", 0.15),
        ))

    # --- Contrast ---
    contrast_cfg = aug_cfg.get("contrast", {})
    if contrast_cfg.get("enabled", True):
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast(tuple(contrast_cfg.get("range", [0.75, 1.25]))),  # type: ignore[arg-type]
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1,
            ),
            apply_probability=contrast_cfg.get("probability", 0.15),
        ))

    # --- Low resolution simulation ---
    lowres_cfg = aug_cfg.get("low_resolution", {})
    if lowres_cfg.get("enabled", True):
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=tuple(lowres_cfg.get("scale", [0.5, 1])),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,  # type: ignore[arg-type]
                p_per_channel=0.5,
            ),
            apply_probability=lowres_cfg.get("probability", 0.25),
        ))

    # --- Gamma ---
    gamma_cfg = aug_cfg.get("gamma", {})
    if gamma_cfg.get("enabled", True):
        gamma_range = tuple(gamma_cfg.get("range", [0.7, 1.5]))
        # Inverted gamma
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast(gamma_range).sample_contrast,  # type: ignore[arg-type]
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1,
            ),
            apply_probability=gamma_cfg.get("p_inverted", 0.1),
        ))
        # Normal gamma
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast(gamma_range).sample_contrast,  # type: ignore[arg-type]
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1,
            ),
            apply_probability=gamma_cfg.get("p_normal", 0.3),
        ))

    # --- Mirror ---
    mirror_cfg = aug_cfg.get("mirror", {})
    if mirror_cfg.get("enabled", True) and mirror_axes and len(mirror_axes) > 0:
        transforms.append(MirrorTransform(allowed_axes=mirror_axes))

    # --- If Mask-based normalisation was used in preprocessing, set to zero the region where label is -1 (i.e. the background where image was zero before preprocessing) ---
    # I think this is not needed as in normalization.py the voxels where seg==-1 are set to zero after normalization, but I keep it here just in case (e.g. values changed during augmentations).
    if use_mask_for_norm is not None and any(use_mask_for_norm):
        transforms.append(MaskImageTransform(
            apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
            channel_idx_in_seg=0, # Segmentation maps are single-channeled (and not one-hot encoded here) so 0 is used here.
            set_outside_to=0, # The value to set in the place where the label is -1. Remember that -1 in segmentation maps is applied wherever the image was black.
        ))

    # --- Remove label -1 (outside non-black body marker from preprocessing) ---
    transforms.append(RemoveLabelTansform(-1, 0))

    # Deep-supervision downsampling is intentionally NOT performed here.
    # The trainer applies CutMix at full resolution first, then runs
    # downsample_seg_for_ds so every DS scale stays consistent with the mix.

    return ComposeTransforms(transforms)


def build_validation_transforms(
    deep_supervision_scales: Optional[List[List[float]]] = None,
) -> BasicTransform:
    """Build validation augmentation pipeline (minimal transforms)."""
    transforms = []
    transforms.append(RemoveLabelTansform(-1, 0))

    if deep_supervision_scales is not None: # Make sure you do not do deep supervision for the final metrics when performing full validation/testing.
        transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

    return ComposeTransforms(transforms)
