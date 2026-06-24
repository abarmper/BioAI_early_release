"""Shared preprocessing utilities — I/O helpers and nnUNet-style cropping.

Imported by both plan.py and preprocessing4.py.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes

# nnUNet channel-suffix pattern: {case_id}_{4-digit-channel}
_CHANNEL_RE = re.compile(r"^(.+)_(\d{4})$")


# ---------------------------------------------------------------------------
# nnUNet-exact crop-to-nonzero  (with -1 labelling for outside-body voxels)
# ---------------------------------------------------------------------------

def crop_to_nonzero(
    data: np.ndarray,
    seg: Optional[np.ndarray] = None,
    nonzero_label: int = -1,
) -> Tuple[np.ndarray, Optional[np.ndarray], List[List[int]]]:
    """Crop data and segmentation to the bounding box of non-zero voxels.

    Mirrors nnUNet v2's ``crop_to_nonzero`` exactly:

    1. Union of ``channel != 0`` across all image channels.
    2. ``scipy.ndimage.binary_fill_holes`` to close enclosed zero regions.
    3. Bounding-box extraction and crop.
    4. Voxels in ``seg`` that were 0 *and* outside the non-zero mask are
       relabelled to ``nonzero_label`` (``-1``).  This lets the ZScore
       normaliser know which voxels are outside the body.
    5. For test cases (no ``seg``), a synthetic segmentation is created:
       ``0`` inside the body, ``-1`` outside.

    Parameters
    ----------
    data : np.ndarray
        Image array ``(C, *spatial)``.
    seg : np.ndarray or None
        Segmentation ``(1, *spatial)`` or ``None`` for test cases.
    nonzero_label : int
        Value assigned to outside-body voxels (default ``-1``).

    Returns
    -------
    data_cropped, seg_cropped, bbox
        ``bbox`` is a list of ``[lo, hi]`` per spatial axis (hi exclusive).
    """
    # 1. Union nonzero mask across channels
    nonzero_mask = np.zeros(data.shape[1:], dtype=bool)
    for c in range(data.shape[0]):
        nonzero_mask |= (data[c] != 0)

    # 2. Fill holes
    nonzero_mask = np.asarray(binary_fill_holes(nonzero_mask), dtype=bool)
    assert nonzero_mask is not None, "binary_fill_holes failed to produce a mask"
    coords = np.argwhere(nonzero_mask)
    if len(coords) == 0:
        bbox = [[0, s] for s in data.shape[1:]]
        if seg is None:
            seg_out = np.full((1,) + data.shape[1:], nonzero_label, dtype=np.int8)
        else:
            seg_out = seg.copy().astype(np.int8)
        return data, seg_out, bbox

    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    slicer = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
    bbox = [[int(l), int(h)] for l, h in zip(lo, hi)]

    # 3. Crop
    data_c = data[(slice(None),) + slicer]
    nonzero_mask_c = nonzero_mask[slicer]

    # 4/5. Build segmentation with outside-body labels
    if seg is None:
        seg_c = np.where(nonzero_mask_c[np.newaxis], 0, nonzero_label).astype(np.int8)
    else:
        seg_c = seg[(slice(None),) + slicer].copy()
        for ch in range(seg_c.shape[0]):
            outside = (~nonzero_mask_c) & (seg_c[ch] == 0)
            seg_c[ch][outside] = nonzero_label
        seg_c = seg_c.astype(np.int8)

    return data_c, seg_c, bbox


# ---------------------------------------------------------------------------
# SimpleITK I/O
# ---------------------------------------------------------------------------

def load_sitk_volume(file_path: str) -> Tuple[np.ndarray, Dict]:
    """Load a NIfTI/NRRD file via SimpleITK.

    Returns
    -------
    array : np.ndarray, float32, shape ``(Z, Y, X)``
        Volume in SimpleITK memory order.
    properties : dict
        ``spacing`` (x, y, z), ``origin``, ``direction``,
        ``shape_original`` (x, y, z voxels).
    """
    img = sitk.ReadImage(str(file_path))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    props: Dict = {
        "spacing": list(img.GetSpacing()),       # (x, y, z) mm
        "origin": list(img.GetOrigin()),
        "direction": list(img.GetDirection()),
        "shape_original": list(img.GetSize()),   # (x, y, z) voxels
    }
    return arr, props


# ---------------------------------------------------------------------------
# Case ID extraction
# ---------------------------------------------------------------------------

def case_id_from_path(path: str) -> str:
    """Extract the case ID from a NIfTI file path.

    Strips the ``_0000`` (or any ``_{4-digit}`` channel suffix) and the
    ``.nii.gz`` extension.

    Examples
    --------
    ``"liver_0001_0000.nii.gz"``  →  ``"liver_0001"``
    ``"case_042.nii.gz"``         →  ``"case_042"``
    """
    stem = Path(path).name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    m = _CHANNEL_RE.match(stem)
    return m.group(1) if m else stem
