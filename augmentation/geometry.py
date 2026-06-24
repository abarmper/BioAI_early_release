"""Determine training geometry: rotation range, mirror axes, dummy-2D DA.

Based on the nnUNet approach in
``nnUNetTrainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from preprocessing.plan import _ANISO_THRESHOLD


@dataclass
class TrainingGeometry:
    """Training data-augmentation geometry parameters."""

    rotation_for_DA: Tuple[float, float]
    do_dummy_2d_data_aug: bool
    initial_patch_size: List[int]
    mirror_axes: Tuple[int, ...]


def determine_training_geometry(
    patch_size: List[int],
    spacing: List[float] | None = None,
    aniso_threshold: float = _ANISO_THRESHOLD,
) -> TrainingGeometry:
    """Compute augmentation geometry from patch size and spacing.

    Parameters
    ----------
    patch_size : list of int
        Final patch size (2D or 3D).
    spacing : list of float or None
        Voxel spacing.  Used only for 3D to detect anisotropy.
    aniso_threshold : float
        If ``max(patch_size) / patch_size[0] > threshold``, enable dummy 2D DA.

    Returns
    ----------
    TrainingGeometry
        Geometry parameters for training data augmentation (rotation range, mirror axes, dummy-2D flag, initial larger patch size required because it will shrink after augmentations).
    """
    dim = len(patch_size)

    if dim == 2:
        do_dummy_2d = False
        if max(patch_size) / min(patch_size) > 1.5:
            rotation = (-15.0 / 360 * 2 * math.pi, 15.0 / 360 * 2 * math.pi)
        else:
            rotation = (-180.0 / 360 * 2 * math.pi, 180.0 / 360 * 2 * math.pi)
        mirror_axes = (0, 1)

    elif dim == 3:
        do_dummy_2d = (max(patch_size) / patch_size[0]) > aniso_threshold
        if do_dummy_2d:
            rotation = (-180.0 / 360 * 2 * math.pi, 180.0 / 360 * 2 * math.pi)
        else:
            rotation = (-30.0 / 360 * 2 * math.pi, 30.0 / 360 * 2 * math.pi)
        mirror_axes = (0, 1, 2)
    else:
        raise ValueError(f"Unsupported dimensionality: {dim}")

    # Compute initial patch size (larger than final to allow spatial DA)
    initial_patch_size = _get_initial_patch_size(
        patch_size, rotation, do_dummy_2d,
    )

    return TrainingGeometry(
        rotation_for_DA=rotation,
        do_dummy_2d_data_aug=do_dummy_2d,
        initial_patch_size=initial_patch_size,
        mirror_axes=mirror_axes,
    )


def _get_initial_patch_size(
    patch_size: List[int],
    rotation: Tuple[float, float],
    do_dummy_2d: bool,
    scale_range: Tuple[float, float] = (0.85, 1.25),
) -> List[int]:
    """Compute initial (pre-augmentation) patch size.

    The initial patch must be large enough that after rotation + scaling
    the final patch is fully covered.  Ported from nnUNet's
    ``compute_initial_patch_size / get_patch_size``.
    """
    dim = len(patch_size)
    rot_max = max(abs(rotation[0]), abs(rotation[1]))
    scale_max = max(abs(scale_range[0]), abs(scale_range[1]))

    coords = np.array(patch_size, dtype=float)
    # compute a conservative rotation bounding box
    if dim == 3:
        # For each axis pair, compute the expanded size after rotation
        initial = np.copy(coords)
        for a in range(dim):
            for b in range(dim):
                if a == b:
                    continue
                # expansion factor from rotation in the (a, b) plane
                expand = abs(math.sin(rot_max)) * coords[b] + abs(math.cos(rot_max)) * coords[a]
                initial[a] = max(initial[a], expand)
        # scale expansion
        initial = np.ceil(initial / scale_range[0]).astype(int)
    else:
        # 2D
        initial = np.copy(coords)
        expand_0 = abs(math.sin(rot_max)) * coords[1] + abs(math.cos(rot_max)) * coords[0]
        expand_1 = abs(math.sin(rot_max)) * coords[0] + abs(math.cos(rot_max)) * coords[1]
        initial[0] = max(initial[0], expand_0)
        initial[1] = max(initial[1], expand_1)
        initial = np.ceil(initial / scale_range[0]).astype(int)

    initial = initial.tolist()

    if do_dummy_2d:
        # Through-plane axis should not be expanded
        initial[0] = patch_size[0]

    return initial
