"""Convert preprocessed .npz/.pkl pairs (or model output folders) back to NIfTI.

Reverses the nnUNet-style preprocessing pipeline:
    resample⁻¹ → uncrop → untranspose → axis-reorder → save .nii.gz

Usage
-----
# All cases in a preprocessed folder:
python -m preprocessing.convert_to_nifti \\
    --input  data/spleen/nnUNetPlans_3d_fullres \\
    --output data/spleen/converted_nifti

# A single case (by case_id, no extension):
python -m preprocessing.convert_to_nifti \\
    --input  data/spleen/nnUNetPlans_3d_fullres \\
    --output data/spleen/converted_nifti \\
    --case   spleen_003

# Model output folder (predictions already in preprocessed space):
python -m preprocessing.convert_to_nifti \\
    --input  data/spleen/results_test \\
    --output data/spleen/results_test_nifti \\
    --pkl-dir data/spleen/nnUNetPlans_3d_fullres_test

Notes
-----
- Each .npz contains ``data`` (C, *spatial) and optionally ``seg`` (1, *spatial).
- Both arrays are saved; ``data`` channels become separate _c{i}.nii.gz files.
- ``seg`` (if present) is saved as _seg.nii.gz.
- For model-output folders that contain only predictions (no matching .pkl),
  supply --pkl-dir pointing to the preprocessed folder with the .pkl files.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from preprocessing.resampling import resample_data_or_seg


# ---------------------------------------------------------------------------
# Denormalization helpers
# ---------------------------------------------------------------------------

def _denormalize_channel(arr: np.ndarray, params: Dict) -> np.ndarray:
    """Reverse the normalization for a single channel (in-place).

    Parameters
    ----------
    arr : np.ndarray
        Normalized channel array.
    params : dict
        Per-channel normalization parameters saved during preprocessing.
    """
    scheme = params.get("scheme", "NoNormalization")

    if scheme == "ZScore":
        mean = params["mean"]
        std = params["std"]
        use_mask = params.get("use_mask", False)
        if use_mask:
            bg_mask = arr == 0.0
            arr *= std
            arr += mean
            arr[bg_mask] = 0.0
        else:
            arr *= std
            arr += mean

    elif scheme == "CT":
        mean = params["mean"]
        std = params["std"]
        arr *= std
        arr += mean
        np.clip(arr, params["lower_clip"], params["upper_clip"], out=arr)

    # NoNormalization: pass-through
    return arr


# ---------------------------------------------------------------------------
# Core reverse pipeline
# ---------------------------------------------------------------------------

def _uncrop(
    data: np.ndarray,
    bbox: List[List[int]],
    full_shape: List[int],
    fill_value: float = 0.0,
) -> np.ndarray:
    """Pad *data* back into a volume of *full_shape* using *bbox*.

    Parameters
    ----------
    data : np.ndarray
        Cropped array ``(C, *cropped_spatial)``.
    bbox : list of [lo, hi]
        Per-axis crop bounds (from ``properties["crop_bbox"]``).
    full_shape : list of int
        Target spatial shape (``properties["shape_before_crop"]``).
    fill_value : float
        Value to use for padded voxels.
    """
    out = np.full((data.shape[0], *full_shape), fill_value, dtype=data.dtype)
    slices = tuple(slice(b[0], b[1]) for b in bbox)
    out[(slice(None),) + slices] = data
    return out


def revert_to_nifti(
    data: np.ndarray,
    properties: Dict,
    is_seg: bool = False,
    seg: Optional[np.ndarray] = None,
) -> Tuple[List[sitk.Image], Optional[sitk.Image]]:
    """Reverse the preprocessing pipeline and return SimpleITK images.

    Parameters
    ----------
    data : np.ndarray
        Preprocessed image array ``(C, *resampled_spatial)``.
    properties : dict
        Metadata dict loaded from the ``.pkl`` file.
    is_seg : bool
        If ``True``, treat *data* itself as a segmentation (nearest-neighbour
        resampling, integer dtype).  Useful when *data* is a model prediction.
    seg : np.ndarray or None
        Optional segmentation ``(1, *resampled_spatial)`` stored alongside
        image data in the .npz.

    Returns
    -------
    img_list : list of sitk.Image
        One SimpleITK image per channel in *data*.  Empty if ``is_seg=True``.
    seg_img : sitk.Image or None
        Reverted segmentation image, or ``None`` if no segmentation supplied.
    """
    tb: List[int] = properties["transpose_backward"]
    bbox: List[List[int]] = properties["crop_bbox"]
    shape_before_crop: List[int] = properties["shape_before_crop"]
    shape_after_crop: List[int] = properties["shape_after_crop"]
    spacing_after_transpose: List[float] = properties["spacing_after_transpose"]
    resampled_spacing: List[float] = properties["resampled_spacing"]
    original_spacing: List[float] = properties["original_spacing"]
    origin: List[float] = properties["origin"]
    direction: List[float] = properties["direction"]

    norm_params: Optional[List[Dict]] = properties.get("normalization_params")

    def _revert_array(arr: np.ndarray, nearest: bool) -> np.ndarray:
        # 1. Resample back to shape_after_crop — use the stored shape directly
        #    to avoid off-by-one from floating-point rounding in compute_new_shape.
        arr = resample_data_or_seg(
            arr,
            shape_after_crop,
            current_spacing=resampled_spacing,
            new_spacing=spacing_after_transpose,
            is_seg=nearest,
            order=0 if nearest else 3,
            order_z=0,
            force_separate_z=False,
        )

        # 2. Uncrop
        fill = float(arr.min())
        arr = _uncrop(arr, bbox, shape_before_crop, fill_value=fill)

        # 3. Reverse transpose (spatial axes only)
        ax = [0] + [i + 1 for i in tb]
        arr = np.ascontiguousarray(arr.transpose(ax))

        # 4. Convert (C, X, Y, Z) → (C, Z, Y, X)  (SimpleITK order)
        arr = np.ascontiguousarray(arr.transpose(0, 3, 2, 1))
        return arr

    def _to_sitk(arr_czyx: np.ndarray, spacing: List[float]) -> sitk.Image:
        # arr_czyx: (1, Z, Y, X) or (Z, Y, X)
        vol = arr_czyx[0] if arr_czyx.ndim == 4 else arr_czyx
        img = sitk.GetImageFromArray(vol)
        img.SetSpacing(spacing)
        img.SetOrigin(origin)
        img.SetDirection(direction)
        return img

    img_list: List[sitk.Image] = []
    seg_img: Optional[sitk.Image] = None

    if not is_seg:
        data_rev = _revert_array(data, nearest=False)
        # Denormalize per channel if normalization params are available
        if norm_params is not None:
            for c in range(data_rev.shape[0]):
                if c < len(norm_params):
                    data_rev[c] = _denormalize_channel(data_rev[c], norm_params[c])
        for c in range(data_rev.shape[0]):
            img_list.append(_to_sitk(data_rev[c:c+1], original_spacing))

    if seg is not None:
        if seg.dtype != np.int8:
            seg = seg.astype(np.int8)
        seg_rev = _revert_array(seg, nearest=True)
        seg_img = _to_sitk(seg_rev[0:1].astype(np.int16), original_spacing)
    elif is_seg:
        # data IS the segmentation
        seg_arr = data.astype(np.int8) if data.dtype != np.int8 else data
        seg_rev = _revert_array(seg_arr, nearest=True)
        seg_img = _to_sitk(seg_rev[0:1].astype(np.int16), original_spacing)

    return img_list, seg_img


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------

def convert_case(
    npz_path: Path,
    pkl_path: Path,
    output_dir: Path,
    is_seg: bool = False,
) -> None:
    """Load one .npz/.pkl pair and write NIfTI file(s) to *output_dir*."""
    with open(pkl_path, "rb") as fh:
        props = pickle.load(fh)

    npz = np.load(npz_path)
    data = npz["data"] if "data" in npz else npz[list(npz.keys())[0]]
    seg = npz["seg"] if "seg" in npz else None

    case_id: str = props.get("case_id", npz_path.stem)

    img_list, seg_img = revert_to_nifti(data, props, is_seg=is_seg, seg=seg)

    output_dir.mkdir(parents=True, exist_ok=True)

    for c, img in enumerate(img_list):
        suffix = f"_c{c}" if len(img_list) > 1 else ""
        out_path = output_dir / f"{case_id}{suffix}.nii.gz"
        sitk.WriteImage(img, str(out_path))
        print(f"  saved image  → {out_path}")

    if seg_img is not None:
        out_path = output_dir / f"{case_id}_seg.nii.gz"
        sitk.WriteImage(seg_img, str(out_path))
        print(f"  saved seg    → {out_path}")


def convert_folder(
    input_dir: Path,
    output_dir: Path,
    pkl_dir: Optional[Path] = None,
    case_id: Optional[str] = None,
    is_seg: bool = False,
) -> None:
    """Convert all (or one) .npz files in *input_dir*.

    Parameters
    ----------
    input_dir : Path
        Folder containing .npz files (and optionally matching .pkl).
    output_dir : Path
        Destination for .nii.gz files.
    pkl_dir : Path or None
        Alternative folder for .pkl files (e.g. when input_dir is a model
        output folder that has no .pkl files of its own).
    case_id : str or None
        If given, only convert this case.
    is_seg : bool
        Treat the ``data`` array as a segmentation (model predictions).
    """
    pkl_dir = pkl_dir or input_dir

    npz_files = sorted(input_dir.glob("*.npz"))
    if case_id is not None:
        npz_files = [f for f in npz_files if f.stem == case_id]
        if not npz_files:
            sys.exit(f"No .npz found for case '{case_id}' in {input_dir}")

    if not npz_files:
        sys.exit(f"No .npz files found in {input_dir}")

    print(f"Converting {len(npz_files)} case(s): {input_dir} → {output_dir}")
    for npz_path in npz_files:
        pkl_path = pkl_dir / f"{npz_path.stem}.pkl"
        if not pkl_path.exists():
            print(f"  WARNING: .pkl not found for {npz_path.name} — skipping.")
            continue
        print(f"  {npz_path.stem}")
        convert_case(npz_path, pkl_path, output_dir, is_seg=is_seg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert preprocessed .npz/.pkl pairs back to NIfTI (.nii.gz)."
    )
    p.add_argument("--input", "-i", required=True, type=Path,
                   help="Folder containing .npz files.")
    p.add_argument("--output", "-o", required=True, type=Path,
                   help="Output folder for .nii.gz files.")
    p.add_argument("--pkl-dir", type=Path, default=None,
                   help="Alternative folder for .pkl files (model-output use-case).")
    p.add_argument("--case", type=str, default=None,
                   help="Convert a single case by ID (no extension).")
    p.add_argument("--seg", action="store_true",
                   help="Treat 'data' array as segmentation (model predictions).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert_folder(
        input_dir=args.input,
        output_dir=args.output,
        pkl_dir=args.pkl_dir,
        case_id=args.case,
        is_seg=args.seg,
    )
