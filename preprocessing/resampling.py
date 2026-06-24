"""Numpy/scipy-based resampling matching nnUNet v2's two-pass anisotropic strategy.

Reference
---------
``nnUNet/nnunetv2/preprocessing/resampling/default_resampling.py``
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from skimage.transform import resize


# ---------------------------------------------------------------------------
# Shape computation
# ---------------------------------------------------------------------------

def compute_new_shape(
    old_shape: Sequence[int] | np.ndarray,
    old_spacing: Sequence[float] | np.ndarray,
    new_spacing: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compute the new volume shape after resampling to *new_spacing*.

    Returns
    -------
    np.ndarray of int64
        New shape (same number of axes as *old_shape*).
    """
    old_shape = np.array(old_shape, dtype=np.float64)
    old_spacing = np.array(old_spacing, dtype=np.float64)
    new_spacing = np.array(new_spacing, dtype=np.float64)
    new_shape = np.round(old_shape * old_spacing / new_spacing).astype(np.int64)
    return new_shape


# ---------------------------------------------------------------------------
# Core resampling
# ---------------------------------------------------------------------------

def resample_data_or_seg(
    data: np.ndarray,
    new_shape: Sequence[int] | np.ndarray,
    current_spacing: Sequence[float] | np.ndarray,
    new_spacing: Sequence[float] | np.ndarray,
    is_seg: bool = False,
    order: int = 3,
    order_z: int = 0,
    force_separate_z: Optional[bool] = None,
    aniso_threshold: float = 3.0,
) -> np.ndarray:
    """Resample a multi-channel volume to *new_shape*. Arguments is_seg, order, order_z, force_separate_z, and aniso_threshold
    are set usually during the planning phase by plan.py.

    Parameters
    ----------
    data : np.ndarray
        Array of shape ``(C, *spatial_dims)`` where C is the channel
        dimension.  For segmentation, C is typically 1.
    new_shape : sequence of int
        Target spatial shape (excluding C).
    current_spacing, new_spacing : sequence of float
        Voxel spacing for each spatial axis.
    is_seg : bool
        ``True`` → nearest-neighbour resampling by default; ``False`` → spline.
    order : int
        Interpolation order for in-plane or isotropic resampling.
        Usually, order ``3`` for images (cubic), ``0`` for segmentations (nearest neighbor).
    order_z : int
        Interpolation order for the low-resolution (through-plane) axis.
        Only used when doing separate-z resampling.  Usually, ``0`` order for images
        across z, and if this is the case, then also ``0`` order for segmentation maps.
    force_separate_z : bool or None
        ``True``  → always do two-pass (in-plane then through-plane).
        ``False`` → always do single-pass isotropic resampling.
        ``None``  → auto-detect based on spacing anisotropy.
    aniso_threshold : float
        Spacing ratio above which the volume is considered anisotropic.

    Returns
    -------
    np.ndarray
        Resampled array of shape ``(C, *new_shape)``.
    """
    new_shape = np.array(new_shape, dtype=np.int64)
    current_spacing = np.array(current_spacing, dtype=np.float64)
    new_spacing = np.array(new_spacing, dtype=np.float64)
    n_channels = data.shape[0]
    spatial_ndim = len(new_shape) # Number of spatial dimensions: 2 or 3 for 2D or 3D volumes

    # Determine whether to use separate-z resampling
    if force_separate_z is None:
        do_separate_z = _needs_separate_z(current_spacing, aniso_threshold) or \
                        _needs_separate_z(new_spacing, aniso_threshold)
    else:
        do_separate_z = force_separate_z

    if np.array_equal(np.array(data.shape[1:]), new_shape):
        return data  # already correct shape

    result = np.zeros((n_channels, *new_shape), dtype=data.dtype)

    for c in range(n_channels):
        if do_separate_z and spatial_ndim == 3:
            result[c] = _resample_separate_z(
                data[c], new_shape, is_seg, order, order_z,
            )
        else:
            result[c] = _resample_isotropic(data[c], new_shape, is_seg, order)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _needs_separate_z(spacing: np.ndarray, threshold: float) -> bool:
    """Check if spacing is anisotropic enough to warrant separate-z."""
    if len(spacing) < 3:
        return False # 2D case, no need for resampling across z.
    max_sp = spacing.max()
    min_sp = spacing.min()
    if min_sp == 0:
        return False
    return (max_sp / min_sp) > threshold


def _resample_isotropic(
    data_c: np.ndarray,
    new_shape: np.ndarray,
    is_seg: bool,
    order: int,
) -> np.ndarray:
    """Single-pass resampling (isotropic or forced)."""
    if is_seg:
        # Nearest-neighbour for segmentation
        out = resize(
            data_c.astype(np.float32),
            output_shape=tuple(new_shape),
            order=0,
            preserve_range=True,
            anti_aliasing=False, # Consistent with nnUNet's use of anti_aliasing=False for segmentation resampling.
        )
        return np.round(out).astype(data_c.dtype)
    else:
        return resize(
            data_c.astype(np.float32),
            output_shape=tuple(new_shape),
            order=order,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32)


def _resample_separate_z(
    data_c: np.ndarray,
    new_shape: np.ndarray,
    is_seg: bool,
    order: int,
    order_z: int,
) -> np.ndarray:
    """Two-pass resampling: in-plane then through-plane (axis 0).

    nnUNet treats axis 0 of the transposed array as the low-resolution
    (through-plane) axis.
    """
    old_shape = np.array(data_c.shape, dtype=np.int64)

    # Step 1: resize in-plane (axes 1, 2) — keep axis 0 unchanged
    if not np.array_equal(old_shape[1:], new_shape[1:]):
        inplane_shape = (old_shape[0], *new_shape[1:])
        step1 = np.zeros(inplane_shape, dtype=np.float32)
        for z in range(old_shape[0]): # Loop through channels
            if is_seg:
                sl = resize(
                    data_c[z].astype(np.float32),
                    output_shape=tuple(new_shape[1:]),
                    order=0 if order_z == 0 and is_seg else order,
                    preserve_range=True,
                    anti_aliasing=False,
                )
                step1[z] = np.round(sl)
            else:
                step1[z] = resize(
                    data_c[z].astype(np.float32),
                    output_shape=tuple(new_shape[1:]),
                    order=order,
                    preserve_range=True,
                    anti_aliasing=False,
                )
    else:
        step1 = data_c.astype(np.float32)

    # Step 2: resize through-plane (axis 0)
    if old_shape[0] != new_shape[0]: # Check if axis-0-resampling is actually needed because in the case of 2D volumes, old shape and new shape are the same along axis 0.
        # Transpose to bring axis 0 last so we can iterate over (y, x) slices
        # shape: (new_y, new_x, old_z)
        step1_t = step1.transpose(1, 2, 0)
        result_t = np.zeros((*new_shape[1:], new_shape[0]), dtype=np.float32)

        # Reshape to 2D for vectorized resize: (new_y*new_x, old_z) → (new_y*new_x, new_z)
        n_pixels = int(np.prod(new_shape[1:]))
        flat_in = step1_t.reshape(n_pixels, old_shape[0])
        flat_out = np.zeros((n_pixels, new_shape[0]), dtype=np.float32)

        for p in range(n_pixels):
            row = flat_in[p].reshape(1, -1)
            resampled = resize(
                row,
                output_shape=(1, int(new_shape[0])),
                order=order_z,
                preserve_range=True,
                anti_aliasing=False,
            )
            flat_out[p] = resampled.squeeze()

        result_t = flat_out.reshape(*new_shape[1:], new_shape[0])
        result = result_t.transpose(2, 0, 1)  # back to (z, y, x)

        if is_seg:
            result = np.round(result)
    else:
        result = step1

    if is_seg:
        return result.astype(data_c.dtype)
    return result.astype(np.float32)
