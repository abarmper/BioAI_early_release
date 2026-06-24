"""Dataset fingerprint extraction and experiment planning.

Two-phase pipeline (both triggered by ``command=plan``):

1. **``plan_dataset()``** — scans ``imagesTr/`` / ``labelsTr/``, extracts raw
   statistics (spacing, shape, intensity percentiles), saves
   ``dataset_fingerprint.json``.

2. **``plan_experiment()``** — reads the fingerprint + ``dataset.json``,
   decides target spacing / normalization / auto patch size / batch size,
   saves ``experiment_plans.json`` with per-configuration entries
   (``3d_fullres``, ``2d``, optionally ``3d_lowres``).

Dataset metadata (channel names, labels) must be declared in::

    {dataset_dir}/raw/dataset.json

Example (single-channel CT)::

    {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "liver": 1},
        "file_ending": ".nii.gz"
    }

Example (4-channel MRI — BraTS)::

    {
        "channel_names": {"0": "T1", "1": "T1CE", "2": "T2", "3": "FLAIR"},
        "labels": {"background": 0, "whole_tumor": 1},
        "file_ending": ".nii.gz"
    }

Normalization is selected per-channel from the channel name:
  ``"CT"`` / ``"ct"``         →  ``CTNormalization``  (clip + dataset z-score)
  ``"noNorm"`` / ``"none"``   →  ``NoNormalization``  (identity)
  anything else               →  ``ZScoreNormalization`` (per-image z-score)
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from preprocessing.compute_statistics import compute_statistics
from preprocessing.resampling import compute_new_shape

logger = logging.getLogger(__name__)

# nnUNet constant: spacing ratio above which an axis is considered low-resolution
_ANISO_THRESHOLD = 3

# nnUNet constant: crop ratio below which non-zero masking is applied for MRI
_NONZERO_MASK_THRESHOLD = 0.75

# nnUNet channel-suffix pattern
_CHANNEL_RE = re.compile(r"^(.+)_(\d{4})$")

# Map channel name → full normalizer class name (nnUNet convention)
_CHANNEL_NAME_TO_NORM: Dict[str, str] = {
    "ct": "CTNormalization",
    "CT": "CTNormalization",
    "nonorm": "NoNormalization",
    "noNorm": "NoNormalization",
    "NoNorm": "NoNormalization",
    "none": "NoNormalization",
    "None": "NoNormalization",
}
_DEFAULT_NORM = "ZScoreNormalization"

# Min bottleneck edge length for UNet topology (nnUNet default)
_MIN_FEATURE_MAP_SIZE = 4

# Lowres config is created when fullres patch covers < 25% of median volume
_LOWRES_CREATION_THRESHOLD = 0.25
_SPACING_INCREASE_FACTOR = 1.03

_MIN_BATCH_SIZE = 2


# ===========================================================================
# dataset.json reader
# ===========================================================================

def read_dataset_json(raw_dir: Path) -> dict:
    """Read and validate ``dataset.json`` from the raw data directory. \
    It searches for specific keys in the .json file that should be there:
    - "channel_names": a mapping of channel indices to human-readable names (e.g. {"0": "CT", "1": "T1", ...}).
    - "labels": a mapping of label names to integer IDs (e.g. {"background": 0, "liver": 1}).
    Supports legacy nnUNet "modality" key as an alias for "channel_names".

    Parameters
    ----------
    raw_dir : Path
        Directory containing ``dataset.json`` (the raw dataset folder).

    Returns
    -------
    dict
        Parsed ``dataset.json``.

    Raises
    ------
    FileNotFoundError
        If ``dataset.json`` does not exist.
    KeyError
        If required keys are missing.
    """
    path = Path(raw_dir) / "dataset.json"
    if not path.exists():
        raise FileNotFoundError(
            f"dataset.json not found at '{path}'.\n"
            "Create it with at least 'channel_names' and 'labels'.\n\n"
            "Single-channel CT example:\n"
            '  {"channel_names": {"0": "CT"}, '
            '"labels": {"background": 0, "liver": 1}}\n\n'
            "Multi-channel MRI example:\n"
            '  {"channel_names": {"0": "T1", "1": "T1CE", "2": "T2", "3": "FLAIR"}, '
            '"labels": {"background": 0, "whole_tumor": 1}}'
        )
    with open(path) as fh:
        dj = json.load(fh)

    # Support legacy nnUNet "modality" key
    if "channel_names" not in dj and "modality" in dj:
        dj["channel_names"] = dj["modality"]

    if "channel_names" not in dj:
        raise KeyError("dataset.json must contain 'channel_names'.")
    if "labels" not in dj:
        raise KeyError("dataset.json must contain 'labels'.")

    return dj


# ===========================================================================
# Case grouping
# ===========================================================================

def group_cases_nnunet(
    images_dir: str,
    num_channels: int,
    labels_dir: Optional[str] = None,
) -> Tuple[List[Tuple[str, ...]], Optional[List[str]]]:
    """Group raw NIfTI images into per-case channel tuples.

    Supports the nnUNet naming convention::

        {images_dir}/{case_id}_{channel:04d}.nii.gz
        {labels_dir}/{case_id}.nii.gz

    Single-channel images without the ``_0000`` suffix are also accepted
    (treated as channel 0).

    Parameters
    ----------
    images_dir : str
        Folder containing the images (e.g. ``raw/imagesTr``).
    num_channels : int
        Expected number of channels per case.
    labels_dir : str or None
        Segmentation folder (``labelsTr``, …).  ``None`` for test splits.

    Returns
    -------
    image_groups : list of tuple
        ``[(ch0_case1, ch1_case1, …), …]`` sorted by case ID.
    label_paths : list of str or None
        Aligned label paths, or ``None`` when ``labels_dir`` is ``None``.
    """
    images = sorted(glob(os.path.join(images_dir, "*.nii.gz")))
    if not images:
        raise FileNotFoundError(
            f"No NIfTI images found in '{images_dir}'."
        )

    case_channels: dict = defaultdict(list)
    for img_path in images:
        stem = Path(img_path).name # Get just the file name, not the full path.
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        if stem.startswith("."):
            # Skip hidden files.
            continue
        m = _CHANNEL_RE.match(stem)
        if m:
            case_id, ch_idx = m.group(1), int(m.group(2))
        else:
            case_id, ch_idx = stem, 0
        case_channels[case_id].append((ch_idx, img_path))

    case_ids = sorted(case_channels.keys())

    # Now we accumulate cases that don't have the expected number of channels, and raise an error if any are found.
    bad = [ 
        (cid, len(chans))
        for cid, chans in case_channels.items()
        if len(chans) != num_channels
    ]
    if bad:
        details = ", ".join(f"'{cid}': {n}" for cid, n in bad[:5])
        raise RuntimeError(
            f"Expected {num_channels} channel(s) per case but found mismatches: "
            f"{details}."
        )

    image_groups: List[Tuple[str, ...]] = [
        tuple(p for _, p in sorted(case_channels[cid], key=lambda x: x[0])) # Sort channels by index and extract paths per case_id.
        for cid in case_ids
    ]

    if labels_dir is None: #Finish if no labels_dir is provided (e.g. test split).
        return image_groups, None

    labels = sorted(glob(os.path.join(labels_dir, "*.nii.gz")))
    if not labels:
        raise FileNotFoundError(
            f"No NIfTI labels found in '{labels_dir}'."
        )

    label_dict: dict = {}
    for lbl_path in labels:
        stem = Path(lbl_path).name
        if stem.endswith(".nii.gz"):
            stem = stem[:-7]
        if stem.startswith("."):
            # Skip hidden files.
            continue
        label_dict[stem] = lbl_path
    
    if len(label_dict.keys()) > len(case_ids):
        logger.warning(
            "Found more label files (%d) than image cases (%d) . "
            "This may indicate a mismatch between image and label naming.",
            len(label_dict), len(case_ids)
        )

    matched: List[str] = []
    missing = []
    for cid in case_ids:
        lbl = label_dict.get(cid)
        if lbl is None:
            missing.append(cid)
        else:
            matched.append(lbl)

    if missing:
        raise RuntimeError(
            f"No label found for {len(missing)} case(s): "
            + ", ".join(f"'{c}'" for c in missing[:5])
            + (f" … (+{len(missing) - 5} more)" if len(missing) > 5 else "")
        )

    logger.info("Found %d case(s) with %d channel(s).", len(case_ids), num_channels)
    return image_groups, matched


# ===========================================================================
# Spacing / anisotropy
# ===========================================================================

def _compute_target_spacing(
    spacings: List[List[float]],
    shapes_after_crop: List[List[int]],
) -> Tuple[List[float], bool, Optional[int]]:
    """Median spacing with nnUNet anisotropy handling.

    Returns
    -------
    (target_spacing, is_anisotropic, anisotropic_axis)
    """
    sp = np.array(spacings, dtype=float)
    sz = np.array(shapes_after_crop, dtype=float)

    target = np.median(sp, axis=0)
    median_size = np.median(sz, axis=0)

    worst_axis = int(np.argmax(target))
    other_axes = [i for i in range(len(target)) if i != worst_axis]

    has_aniso_spacing = target[worst_axis] > (
        _ANISO_THRESHOLD * max(target[i] for i in other_axes)
    )
    has_aniso_voxels = (
        median_size[worst_axis] * _ANISO_THRESHOLD
        < min(median_size[i] for i in other_axes)
    )

    is_anisotropic = bool(has_aniso_spacing and has_aniso_voxels)
    anisotropic_axis: Optional[int] = None

    if is_anisotropic:
        anisotropic_axis = worst_axis
        p10 = float(np.percentile(sp[:, worst_axis], 10))
        min_allowed = float(max(target[i] for i in other_axes))
        target[worst_axis] = max(p10, min_allowed + 1e-5)

    return target.tolist(), is_anisotropic, anisotropic_axis


def _compute_iso_spacing(target_spacing_t: List[float], strategy: str) -> float:
    """Derive the isotropic target spacing from the anisotropy-corrected fullres
    target spacing (``target_spacing_t``).

    Parameters
    ----------
    target_spacing_t : list of float
        The fullres target spacing (transposed order) produced by
        :func:`_compute_target_spacing`. Already has nnUNet-style anisotropy
        correction applied (coarsest axis capped at the 10th percentile of
        its raw spacings).
    strategy : str
        One of:
          * ``"voxel_budget"``  — geometric mean of ``target_spacing_t``.
            Produces an iso volume with approximately the same voxel count as
            the non-iso fullres median. Recommended default.
          * ``"max_target"``   — ``max(target_spacing_t)``. Coarsest iso that
            never upsamples beyond the planner's non-iso choice.
          * ``"median_target"`` — ``median(target_spacing_t)``. Preserves
            in-plane detail; enlarges the coarse axis significantly.

    Raises
    ------
    ValueError
        If ``strategy`` is not one of the supported values.
    """
    t = np.array(target_spacing_t, dtype=float)
    if strategy == "voxel_budget":
        return float(np.prod(t) ** (1.0 / len(t)))
    if strategy == "max_target":
        return float(np.max(t))
    if strategy == "median_target":
        return float(np.median(t))
    raise ValueError(
        f"Unknown iso_spacing_strategy '{strategy}'. "
        f"Expected one of: voxel_budget, max_target, median_target."
    )


def _compute_transpose_axes(
    target_spacing: List[float],
) -> Tuple[List[int], List[int]]:
    """Transpose so the lowest-resolution (largest spacing) axis is first."""
    worst = int(np.argmax(target_spacing))
    remaining = [i for i in range(len(target_spacing)) if i != worst]
    forward = [worst] + remaining
    backward = [0] * len(forward)
    for new_idx, old_idx in enumerate(forward):
        backward[old_idx] = new_idx
    return forward, backward


# ===========================================================================
# Normalization planning
# ===========================================================================

def _compute_normalization_plan(
    channel_names: Dict[str, str],
    median_relative_size: float,
) -> Tuple[List[str], List[bool]]:
    """Decide per-channel normalization scheme and non-zero mask usage.

    Parameters
    ----------
    channel_names : dict
        Mapping ``{"0": "CT", "1": "T1", …}`` from ``dataset.json``.
    median_relative_size : float
        Median fraction of voxels inside the foreground bounding box.
        When below ``0.75``, the non-zero mask is applied for ZScore channels.

    Returns
    -------
    (normalization_schemes, use_mask_for_norm)
        Both lists have length ``len(channel_names)`` and contain the schemes and mask usage flags in channel index order (not transpose order).
    """
    apply_mask = median_relative_size < _NONZERO_MASK_THRESHOLD
    schemes: List[str] = []
    use_mask: List[bool] = []
    for idx in sorted(channel_names.keys(), key=int):
        name = channel_names[str(idx)]
        scheme = _CHANNEL_NAME_TO_NORM.get(name, _DEFAULT_NORM)
        schemes.append(scheme)
        if scheme == "CTNormalization":
            use_mask.append(False) # We never apply the non-zero mask for CTNormalization.
        elif scheme == "ZScoreNormalization":
            use_mask.append(apply_mask) # For ZScore normalization scheme, we apply the non-zero mask if the median relative size is below the threshold.
        else:
            # Default behaviour for unrecognized schemes: Do not apply non-zero mask for normalization of unrecognised schemes.
            use_mask.append(False)
    return schemes, use_mask


# ===========================================================================
# Network topology — ported from nnUNet's network_topology.py (MIT licence)
# ===========================================================================

def _pad_shape(shape, must_be_divisible_by) -> np.ndarray:
    """Pad shape up to the nearest multiple of must_be_divisible_by."""
    if not isinstance(must_be_divisible_by, (tuple, list, np.ndarray)):
        must_be_divisible_by = [must_be_divisible_by] * len(shape)
    new_shp = [
        shape[i] + must_be_divisible_by[i] - shape[i] % must_be_divisible_by[i]
        for i in range(len(shape))
    ]
    for i in range(len(shape)):
        if shape[i] % must_be_divisible_by[i] == 0:
            new_shp[i] -= must_be_divisible_by[i]
    return np.array(new_shp, dtype=int)


def _get_pool_and_conv_props(
    spacing: List[float],
    patch_size: List[int],
    min_feature_map_size: int,
    max_numpool: int,
) -> Tuple:
    """Derive nnUNet v2-style encoder topology for a given patch geometry.

    The function simulates the encoder stage by stage and returns the pooling
    schedule, convolution kernel schedule, and the divisibility constraints
    that the input patch must satisfy.

    The returned 5-tuple is::

        (
            num_pool_per_axis,
            pool_op_kernel_sizes,
            conv_kernel_sizes,
            padded_patch_size,
            shape_must_be_divisible_by,
        )

    where:

    - ``num_pool_per_axis`` counts how often each axis is downsampled.
    - ``pool_op_kernel_sizes`` lists the per-stage pooling kernels. The first
      entry is always ``[1, ..., 1]`` (the full-resolution stage), and each
      later entry uses ``2`` on axes that are pooled at that stage and ``1``
      on axes that are kept at the same resolution.
    - ``conv_kernel_sizes`` lists the per-stage convolution kernels. Kernels
      start as ``1`` per axis and switch to ``3`` once that axis is no longer
      strongly anisotropic relative to the currently finest spacing. The final
      appended entry is the bottleneck kernel ``[3, ..., 3]``.
    - ``padded_patch_size`` is the input ``patch_size`` padded so every axis is
      divisible by the total downsampling factor along that axis.
    - ``shape_must_be_divisible_by`` is that downsampling factor itself,
      namely ``2 ** num_pool_per_axis`` for each axis.

    The usually uneven pooling kernels (for example ``[1, 2, 2]``) and
    convolution kernels (for example ``[1, 3, 3]``) are intentional and follow
    the nnUNet v2 anisotropy heuristic:

    - An axis may only be pooled if its current feature-map size is still large
      enough and it has not exceeded ``max_numpool``.
    - Among those axes, nnUNet only pools axes whose current spacing is within
      a factor of 2 of the finest currently available spacing. Coarser axes are
      temporarily left untouched to avoid collapsing already low-resolution
      directions too early.
    - For the same reason, convolutions use kernel size ``1`` along such
      coarse axes until repeated pooling of the finer axes makes the effective
      spacings more similar; only then does that axis switch to kernel size
      ``3``.

    This is why anisotropic data often starts with in-plane pooling and
    in-plane ``3x3`` kernels while the thick-slice axis keeps pooling factor
    ``1`` and kernel size ``1``, and only later joins once the resolution gap
    has narrowed.
    """
    dim = len(spacing)
    current_spacing = deepcopy(list(float(s) for s in spacing))
    current_size = deepcopy(list(int(p) for p in patch_size))

    pool_op_kernel_sizes = [[1] * dim]
    conv_kernel_sizes: List[List[int]] = []
    num_pool_per_axis = [0] * dim
    kernel_size = [1] * dim

    while True:
        valid_axes = [
            i for i in range(dim)
            if current_size[i] >= 2 * min_feature_map_size
        ]
        if len(valid_axes) < 1:
            break

        spacings_of_valid = [current_spacing[i] for i in valid_axes]
        min_sp = min(spacings_of_valid)
        valid_axes = [
            i for i in valid_axes
            if current_spacing[i] / min_sp < 2 # Drop axes with more than twice the spacing of the minimum (nnUNet anisotropy handling)
        ]
        valid_axes = [
            i for i in valid_axes
            if num_pool_per_axis[i] < max_numpool # If an axis has already been pooled max_numpool times, drop it from the valid axes (nnUNet max pooling depth)
        ]

        if len(valid_axes) == 1:
            if current_size[valid_axes[0]] >= 3 * min_feature_map_size:
                pass
            else:
                break
        if len(valid_axes) < 1:
            break

        for d in range(dim):
            if kernel_size[d] != 3:
                if current_spacing[d] / min(current_spacing) < 2:
                    kernel_size[d] = 3

        other_axes = [i for i in range(dim) if i not in valid_axes]
        pool_kernels = [0] * dim
        for v in valid_axes:
            pool_kernels[v] = 2
            num_pool_per_axis[v] += 1
            current_spacing[v] *= 2
            current_size[v] = math.ceil(current_size[v] / 2)
        for nv in other_axes:
            pool_kernels[nv] = 1

        pool_op_kernel_sizes.append(pool_kernels)
        conv_kernel_sizes.append(deepcopy(kernel_size))

    must_be_divisible_by = 2 ** np.array(num_pool_per_axis)
    patch_size_arr = _pad_shape(patch_size, must_be_divisible_by)
    conv_kernel_sizes.append([3] * dim)

    return (
        num_pool_per_axis,
        pool_op_kernel_sizes,
        conv_kernel_sizes,
        patch_size_arr.tolist(),
        must_be_divisible_by.tolist(),
    )


# ===========================================================================
# Automatic patch-size computation
# ===========================================================================

def _compute_patch_size(
    target_spacing: List[float],
    median_size: List[int],
    patch_size_override: Optional[List[int]] = None,
    min_feature_map_size: int = _MIN_FEATURE_MAP_SIZE,
    max_numpool: int = 999999,
    initial_patch_ref_3d: int = 256,
    initial_patch_ref_2d: int = 2048,
) -> Tuple[List[int], List[int]]:
    """Auto-compute patch size using the nnUNet algorithm (stages 1–2).

    **Stage 1 — aspect-ratio initialisation:**
    The spacing aspect ratio ``1/spacing_i`` is normalised so the total
    volume equals ``initial_patch_ref_3d³`` voxels (3-D) or
    ``initial_patch_ref_2d²`` (2-D).  The result is clipped to the median
    image size (no point in having patches larger than the data).

    **Stage 2 — network topology divisibility:**
    ``_get_pool_and_conv_props`` determines how many times each axis can be
    halved (respecting spacing anisotropy and minimum bottleneck size).
    The patch is padded up to be divisible by ``2 ^ num_pool`` on each axis.

    Parameters
    ----------
    target_spacing : list of float
        Target voxel spacing (after transpose), low-res axis first.
    median_size : list of int
        Median image size in voxels (after transpose + crop).
    patch_size_override : list of int or None
        If given, skip Stage 1 and enforce this size directly (Stage 2
        divisibility is still applied).
    min_feature_map_size : int
        Minimum bottleneck edge length (default 4, nnUNet default).
    max_numpool : int
        Maximum pooling operations per axis (caps UNet depth).
    initial_patch_ref_3d : int
        Isotropic reference edge for 3D (total voxels = ref³).
    initial_patch_ref_2d : int
        Isotropic reference edge for 2D (total voxels = ref²).

    Returns
    -------
    (patch_size, shape_must_be_divisible_by)
        Both as lists of int.
    """
    ndim = len(target_spacing)

    if patch_size_override is None:
        sp = np.array(target_spacing, dtype=float)
        tmp = 1.0 / sp
        if ndim == 3:
            scale = (initial_patch_ref_3d ** 3 / np.prod(tmp)) ** (1.0 / 3)
        elif ndim == 2:
            scale = (initial_patch_ref_2d ** 2 / np.prod(tmp)) ** (1.0 / 2)
        else:
            scale = (initial_patch_ref_3d ** ndim / np.prod(tmp)) ** (1.0 / ndim)

        initial = np.round(tmp * scale).astype(int)
        # Cap to median size
        patch = np.minimum(initial, np.array(median_size[:ndim])).tolist()
    else:
        patch = list(patch_size_override)

    # Stage 2: enforce divisibility by 2^num_pool (may pad patch up slightly)
    _, _, _, patch_adj, div_by = _get_pool_and_conv_props(
        target_spacing, patch, min_feature_map_size, max_numpool
    )
    patch_adj = list(map(int, patch_adj))
    div_by = list(map(int, div_by))

    return patch_adj, div_by


# ===========================================================================
# Human-readable plan summary
# ===========================================================================

def _print_plan_summary(fp: Dict[str, Any], plans: Dict[str, Any]) -> None:
    """Print a human-readable summary of fingerprint + experiment plans."""
    n_cases = fp.get("num_cases", "?")
    med_sp = fp.get("median_spacing", [0, 0, 0])
    med_sz = fp.get("median_size_after_crop", [0, 0, 0])
    rel_size = fp.get("median_relative_size_after_cropping", 1.0)
    iprops = fp.get("foreground_intensity_properties_per_channel", {})

    tf = plans.get("transpose_forward", [0, 1, 2])
    configs = plans.get("configurations", {})

    print("\n" + "=" * 68)
    print("  BioAI Dataset Fingerprint & Experiment Plans")
    print("=" * 68)
    print(f"  Cases             : {n_cases}")
    ch_names = plans.get("channel_names", {})
    for idx, name in sorted(ch_names.items(), key=lambda x: int(x[0])):
        print(f"  Channel {idx}          : {name}")
    labels = plans.get("labels", {})
    print(f"  Labels            : {labels}")
    print()
    print(f"  Median spacing    : {[round(s, 3) for s in med_sp]} mm (x, y, z)")
    print(f"  Median size (crop): {[int(s) for s in med_sz]} vox")
    print(f"  Crop ratio        : {rel_size:.3f}"
          f"  → {'mask for ZScore norm' if rel_size < _NONZERO_MASK_THRESHOLD else 'no mask'}")
    print(f"  Transpose forward : {tf}")
    print()

    for cfg_name, cfg in configs.items():
        print(f"  [{cfg_name}]")
        print(f"    data_identifier : {cfg['data_identifier']}")
        print(f"    spacing         : {[round(s, 3) for s in cfg['spacing']]}")
        print(f"    patch_size      : {cfg['patch_size']}")
        print(f"    div_by          : {cfg.get('shape_must_be_divisible_by', '?')}")
        print(f"    batch_size      : {cfg.get('batch_size', '?')}")
        print(f"    norm schemes    : {cfg['normalization_schemes']}")
        print(f"    use mask norm   : {cfg['use_mask_for_norm']}")
        print()

    print("  Per-channel intensity properties:")
    for ch_str, props in sorted(iprops.items(), key=lambda x: int(x[0])):
        scheme = configs.get("3d_fullres", {}).get("normalization_schemes", [None])[int(ch_str)]
        if scheme == "CTNormalization":
            print(
                f"    ch{ch_str} [CT]  "
                f"clip=[{props['percentile_00_5']:.1f}, {props['percentile_99_5']:.1f}]  "
                f"mean={props['mean']:.1f}  std={props['std']:.1f}"
            )
        else:
            print(
                f"    ch{ch_str} [ZScore]  "
                f"mean={props['mean']:.3f}  std={props['std']:.3f}"
            )

    label_ratios = fp.get("label_class_ratios")
    if label_ratios:
        print()
        print("  Label class ratios:")
        for cls, ratio in sorted(label_ratios.items(), key=lambda x: int(x[0])):
            print(f"    class {cls}: {ratio:.4f}  ({ratio * 100:.2f} %)")

    print("=" * 68)
    print("  Next step: python main2.py command=preprocess data=<name>")
    print("=" * 68 + "\n")


# ===========================================================================
# plan_dataset  (Phase 1 — fingerprint extraction)
# ===========================================================================

def plan_dataset(
    raw_data_dir: Path,
    output_dir: Path,
    num_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract the dataset fingerprint from raw NIfTI training data.

    Reads ``dataset.json`` from *raw_data_dir* to discover the number of
    channels and labels.  Outputs ``dataset_fingerprint.json`` to *output_dir*.

    Parameters
    ----------
    raw_data_dir : Path
        Root directory containing ``imagesTr/``, ``labelsTr/``, and
        ``dataset.json``.
    output_dir : Path
        Where ``dataset_fingerprint.json`` is saved (the dataset root).
    num_workers : int or None
        Parallel workers for fingerprint extraction.

    Returns
    -------
    dict
        The fingerprint, also written to disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read dataset metadata
    dj = read_dataset_json(Path(raw_data_dir))
    channel_names: Dict[str, str] = dj["channel_names"]
    labels_dict: Dict[str, int] = dj["labels"]
    num_channels = len(channel_names)

    images_dir = os.path.join(str(raw_data_dir), "imagesTr")
    labels_dir = os.path.join(str(raw_data_dir), "labelsTr")

    logger.info("Scanning dataset at: %s  (%d channel(s))", raw_data_dir, num_channels)
    image_groups, label_paths = group_cases_nnunet(images_dir, num_channels, labels_dir)
    num_cases = len(image_groups)

    logger.info("Extracting fingerprint from %d cases…", num_cases)
    raw = compute_statistics(
        volume_paths=image_groups,
        mask_paths=label_paths,
        num_workers=num_workers,
    )

    fingerprint: Dict[str, Any] = {
        "num_cases": num_cases,
        # Per-case arrays (needed by planner for target-spacing computation)
        "spacings": raw["spacings"],
        "shapes_after_crop": raw["shapes_after_crop"],
        # Dataset-level aggregates
        "median_spacing": raw["median_spacing"],
        "10th_percentile_spacing": raw["10th_percentile_spacing"],
        "median_size_after_crop": raw["median_size_after_crop"],
        "median_relative_size_after_cropping": raw["median_relative_size_after_cropping"],
        # Per-channel intensity properties (nnUNet format)
        "foreground_intensity_properties_per_channel":
            raw["foreground_intensity_properties_per_channel"],
        # Label class frequencies (sampled)
        "label_class_ratios": raw.get("label_class_ratios"),
    }

    output_file = output_dir / "dataset_fingerprint.json"
    with open(output_file, "w") as fh:
        json.dump(fingerprint, fh, indent=2)

    logger.info("Fingerprint saved → %s", output_file)
    return fingerprint


# ===========================================================================
# plan_experiment  (Phase 2 — configuration planning)
# ===========================================================================

def plan_experiment(
    fingerprint_file: Path,
    output_dir: Path,
    patch_size_override: Optional[List[int]] = None,
    batch_size: int = _MIN_BATCH_SIZE,
    min_feature_map_size: Optional[int] = None,
    max_numpool: Optional[int] = None,
    initial_patch_ref_3d: int = 256,
    initial_patch_ref_2d: int = 2048,
    iso_spacing_strategy: str = "voxel_budget",
    add_size_variants: bool = False,
    s_ref_3d: int = 96,
    s_ref_2d: int = 1024,
    l_ref_3d: int = 176,
    l_ref_2d: int = 2880,
) -> dict:
    """Generate ``experiment_plans.json`` with per-configuration preprocessing plans.

    Reads ``dataset_fingerprint.json`` + ``{output_dir}/raw/dataset.json``
    and creates plans for ``3d_fullres``, ``2d``, ``3d_fullres_iso``,
    ``2d_iso``, and (optionally) ``3d_lowres`` and ``3d_lowres_iso``.

    Patch size is **auto-computed** from spacing + network topology unless
    *patch_size_override* is given.  Batch size is set directly (no VRAM
    auto-detection); adjust it manually based on what fits your GPU.

    Parameters
    ----------
    fingerprint_file : Path
        Path to ``dataset_fingerprint.json``.
    output_dir : Path
        Dataset root (``data/{dataset}/``).
        ``experiment_plans.json`` is written here.
    patch_size_override : list of int or None
        Forces a specific patch size for all configurations.
        For 2D configs the last 2 elements are used.
        ``None`` → auto-computed from spacing + topology.
    batch_size : int
        Batch size written to all configurations in ``experiment_plans.json``.
        Default 2.  Adjust based on VRAM.
    min_feature_map_size : int or None
        Minimum bottleneck edge length per axis for the UNet topology.
        ``None`` → use module default (``_MIN_FEATURE_MAP_SIZE = 4``).
    max_numpool : int or None
        Maximum pooling operations per axis (caps UNet depth).
        ``None`` → unlimited (nnUNet default).
    initial_patch_ref_3d : int
        Isotropic reference edge for 3D patch initialisation (total voxels = ref³).
    initial_patch_ref_2d : int
        Isotropic reference edge for 2D patch initialisation (total voxels = ref²).
    iso_spacing_strategy : str
        Selects how the isotropic target spacing for ``_iso`` configurations is
        derived from the anisotropy-corrected fullres target
        (``target_spacing_t``). One of ``"voxel_budget"`` (default),
        ``"max_target"``, or ``"median_target"``. See
        :func:`_compute_iso_spacing` for a description of each strategy.
    add_size_variants : bool
        If ``True``, every base configuration also gets ``_s`` (smaller patch)
        and ``_l`` (larger patch) sibling configurations using
        ``s_ref_*`` / ``l_ref_*`` as the patch-initialisation reference. The
        variants share the base config's ``data_identifier`` so the same
        preprocessed ``.npz`` / ``.pkl`` files are reused — no extra
        preprocessing required. Default ``False`` (preserves prior behavior).
    s_ref_3d, s_ref_2d : int
        Patch-init reference edge for the ``_s`` variants (3D / 2D).
    l_ref_3d, l_ref_2d : int
        Patch-init reference edge for the ``_l`` variants (3D / 2D).

    Returns
    -------
    dict
        The experiment plans, also written to disk.
    """
    fingerprint_file = Path(fingerprint_file)
    output_dir = Path(output_dir)

    # Resolve optional topology overrides
    min_fmap = int(min_feature_map_size) if min_feature_map_size is not None else _MIN_FEATURE_MAP_SIZE
    max_pool = int(max_numpool) if max_numpool is not None else 999999

    with open(fingerprint_file) as fh:
        fp = json.load(fh)

    # Load dataset.json for channel names and labels
    dj = read_dataset_json(output_dir / "raw")
    channel_names: Dict[str, str] = dj["channel_names"]
    labels_dict: Dict[str, int] = dj["labels"]
    num_channels = len(channel_names)

    # ---- Spacing & transpose --------------------------------------------------
    target_spacing, is_anisotropic, anisotropic_axis = _compute_target_spacing(
        fp["spacings"], fp["shapes_after_crop"]
    )
    transpose_forward, transpose_backward = _compute_transpose_axes(target_spacing) # e.g. [2, 0, 1] to move z to first axis for transpose_forward, and [1, 2, 0] to move it back for transpose_backward

    # Reorder to transposed axis order (low-res first)
    target_spacing_t = [target_spacing[i] for i in transpose_forward]
    median_size_xyz = fp["median_size_after_crop"]
    median_size_t = [int(round(median_size_xyz[i])) for i in transpose_forward]

    # ---- Normalization --------------------------------------------------------
    norm_schemes, use_mask = _compute_normalization_plan(
        channel_names,
        fp["median_relative_size_after_cropping"],
    )

    # ---- Resampling kwargs (same for all configs) ----------------------------
    resamp_data_kw = {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None, "aniso_threshold": _ANISO_THRESHOLD }
    resamp_seg_kw  = {"is_seg": True,  "order": 1, "order_z": 0, "force_separate_z": None, "aniso_threshold": _ANISO_THRESHOLD }

    num_cases = fp.get("num_cases", len(fp.get("spacings", [])))
    configurations: Dict[str, Any] = {}

    def _add_size_variants(
        base_name: str,
        base_cfg: Dict[str, Any],
        spacing,
        median,
        patch_override,
    ) -> None:
        # Emit `_s` (smaller) and `_l` (larger) patch-size variants that share
        # base_cfg's data_identifier (so preprocessed files are reused).
        if not add_size_variants:
            return
        for tag, ref_3d, ref_2d in (("_s", s_ref_3d, s_ref_2d), ("_l", l_ref_3d, l_ref_2d)):
            new_patch, new_div_by = _compute_patch_size(
                spacing, median,
                patch_size_override=patch_override,
                min_feature_map_size=min_fmap,
                max_numpool=max_pool,
                initial_patch_ref_3d=ref_3d,
                initial_patch_ref_2d=ref_2d,
            )
            variant = dict(base_cfg)
            variant["patch_size"] = new_patch
            variant["shape_must_be_divisible_by"] = new_div_by
            configurations[f"{base_name}{tag}"] = variant
            logger.info(
                "%s%s: patch=%s  div_by=%s  (shares data_identifier=%s)",
                base_name, tag, new_patch, new_div_by, base_cfg["data_identifier"],
            )

    # Note for the following code: We could easily run _get_pool_and_conv_props and store the topological properties for each config,
    # However, since we let the user change the patch size in the plan .json file, we choose to compute the properties dynamically at training time.

    # ---- 3d_fullres -----------------------------------------------------------
    patch_3d, div_by_3d = _compute_patch_size(
        target_spacing_t, median_size_t,
        patch_size_override=patch_size_override,
        min_feature_map_size=min_fmap,
        max_numpool=max_pool,
        initial_patch_ref_3d=initial_patch_ref_3d,
        initial_patch_ref_2d=initial_patch_ref_2d,
    )
    patch_volume_3d = float(np.prod(patch_3d))
    median_volume_3d = float(np.prod(median_size_t))

    configurations["3d_fullres"] = {
        "data_identifier": "BioAIPlans_3d_fullres",
        "spacing": target_spacing_t, # This is the transposed order (low-res first).
        "patch_size": patch_3d, # This applies to the transposed axis order (low-res first).
        "shape_must_be_divisible_by": div_by_3d, # Transposed axis order (low-res first). Applies to patch size as is.
        "batch_size": batch_size,
        "median_image_size_in_voxels": median_size_t, # Transposed axis order (low-res first) for median size.
        "normalization_schemes": norm_schemes,
        "use_mask_for_norm": use_mask,
        "resampling_fn_data_kwargs": resamp_data_kw,
        "resampling_fn_seg_kwargs": resamp_seg_kw,
    }
    logger.info("3d_fullres: patch=%s  div_by=%s  batch=%d",
                patch_3d, div_by_3d, batch_size)
    _add_size_variants("3d_fullres", configurations["3d_fullres"],
                       target_spacing_t, median_size_t, patch_size_override)

    # ---- 2d ------------------------------------------------------------------
    spacing_2d = target_spacing_t[1:]
    median_2d  = median_size_t[1:]
    # If the user gave a 3D override, take the last 2 axes for 2D configs;
    # if it's already 2D, use as-is.
    patch_override_2d = (
        patch_size_override[-2:] if patch_size_override is not None else None
    )
    patch_2d, div_by_2d = _compute_patch_size(
        spacing_2d, median_2d,
        patch_size_override=patch_override_2d,
        min_feature_map_size=min_fmap,
        max_numpool=max_pool,
        initial_patch_ref_3d=initial_patch_ref_3d,
        initial_patch_ref_2d=initial_patch_ref_2d,
    )

    configurations["2d"] = {
        "data_identifier": "BioAIPlans_2d",
        "spacing": spacing_2d,
        "patch_size": patch_2d,
        "shape_must_be_divisible_by": div_by_2d,
        "batch_size": batch_size,
        "median_image_size_in_voxels": median_2d,
        "through_plane_spacing": target_spacing_t[0],
        "normalization_schemes": norm_schemes,
        "use_mask_for_norm": use_mask,
        "resampling_fn_data_kwargs": resamp_data_kw,
        "resampling_fn_seg_kwargs": resamp_seg_kw,
    }
    logger.info("2d: patch=%s  div_by=%s  batch=%d",
                patch_2d, div_by_2d, batch_size)
    _add_size_variants("2d", configurations["2d"],
                       spacing_2d, median_2d, patch_override_2d)

    # ---- 3d_lowres (only if fullres patch covers < 25% of median volume) -----
    if median_volume_3d > 0 and (patch_volume_3d / median_volume_3d) < _LOWRES_CREATION_THRESHOLD:
        lowres_spacing = np.array(target_spacing_t, dtype=np.float64)
        lowres_median = np.array(median_size_t, dtype=np.float64)
        for _ in range(2000):  # safety cap
            lowres_median = compute_new_shape(
                median_size_t, target_spacing_t, lowres_spacing.tolist()
            ).astype(float)
            vol = float(np.prod(lowres_median))
            if vol <= 0 or (patch_volume_3d / vol) >= _LOWRES_CREATION_THRESHOLD:
                break
            lowres_spacing *= _SPACING_INCREASE_FACTOR

        lr_median = [int(round(s)) for s in lowres_median]
        lr_patch, lr_div_by = _compute_patch_size(
            lowres_spacing.tolist(), lr_median, patch_size_override=patch_size_override,
            min_feature_map_size=min_fmap,
            max_numpool=max_pool,
            initial_patch_ref_3d=initial_patch_ref_3d,
            initial_patch_ref_2d=initial_patch_ref_2d,
        )

        configurations["3d_lowres"] = {
            "data_identifier": "BioAIPlans_3d_lowres",
            "spacing": lowres_spacing.tolist(),
            "patch_size": lr_patch,
            "shape_must_be_divisible_by": lr_div_by,
            "batch_size": batch_size,
            "median_image_size_in_voxels": lr_median,
            "normalization_schemes": norm_schemes,
            "use_mask_for_norm": use_mask,
            "resampling_fn_data_kwargs": resamp_data_kw,
            "resampling_fn_seg_kwargs": resamp_seg_kw,
        }
        logger.info(
            "3d_lowres created (fullres patch covers %.1f%% < 25%% of median volume).  "
            "spacing=%s  patch=%s",
            100.0 * patch_volume_3d / median_volume_3d,
            [round(s, 3) for s in lowres_spacing.tolist()],
            lr_patch,
        )
        _add_size_variants("3d_lowres", configurations["3d_lowres"],
                           lowres_spacing.tolist(), lr_median, patch_size_override)
    else:
        logger.info(
            "3d_lowres NOT created (fullres patch covers %.1f%% of median volume).",
            100.0 * patch_volume_3d / max(median_volume_3d, 1),
        )

    # ---- iso spacing (used by all _iso configs) ---------------------------------
    # iso_spacing is derived from the anisotropy-corrected fullres target
    # (target_spacing_t, computed above via _compute_target_spacing). The
    # specific reduction is controlled by iso_spacing_strategy; see
    # _compute_iso_spacing for the three supported strategies.
    iso_spacing = _compute_iso_spacing(target_spacing_t, iso_spacing_strategy)
    iso_spacing_t = [iso_spacing] * len(target_spacing_t)
    logger.info(
        "iso_spacing strategy='%s': target_spacing_t=%s -> iso_spacing=%.4f",
        iso_spacing_strategy,
        [round(float(s), 4) for s in target_spacing_t],
        iso_spacing,
    )

    # ---- 3d_fullres_iso ---------------------------------------------------------
    iso_median_t = [
        int(round(s))
        for s in compute_new_shape(median_size_t, target_spacing_t, iso_spacing_t)
    ]
    patch_3d_iso, div_by_3d_iso = _compute_patch_size(
        iso_spacing_t, iso_median_t,
        patch_size_override=patch_size_override,
        min_feature_map_size=min_fmap,
        max_numpool=max_pool,
        initial_patch_ref_3d=initial_patch_ref_3d,
        initial_patch_ref_2d=initial_patch_ref_2d,
    )
    patch_volume_3d_iso = float(np.prod(patch_3d_iso))
    median_volume_3d_iso = float(np.prod(iso_median_t))

    configurations["3d_fullres_iso"] = {
        "data_identifier": "BioAIPlans_3d_fullres_iso",
        "spacing": iso_spacing_t,
        "patch_size": patch_3d_iso,
        "shape_must_be_divisible_by": div_by_3d_iso,
        "batch_size": batch_size,
        "median_image_size_in_voxels": iso_median_t,
        "normalization_schemes": norm_schemes,
        "use_mask_for_norm": use_mask,
        "resampling_fn_data_kwargs": resamp_data_kw,
        "resampling_fn_seg_kwargs": resamp_seg_kw,
    }
    logger.info("3d_fullres_iso: spacing=%.4f  patch=%s  div_by=%s  batch=%d",
                iso_spacing, patch_3d_iso, div_by_3d_iso, batch_size)
    _add_size_variants("3d_fullres_iso", configurations["3d_fullres_iso"],
                       iso_spacing_t, iso_median_t, patch_size_override)

    # ---- 2d_iso -----------------------------------------------------------------
    iso_spacing_2d = [iso_spacing, iso_spacing]
    iso_median_2d = iso_median_t[1:]
    patch_2d_iso, div_by_2d_iso = _compute_patch_size(
        iso_spacing_2d, iso_median_2d,
        patch_size_override=patch_override_2d,
        min_feature_map_size=min_fmap,
        max_numpool=max_pool,
        initial_patch_ref_3d=initial_patch_ref_3d,
        initial_patch_ref_2d=initial_patch_ref_2d,
    )
    configurations["2d_iso"] = {
        "data_identifier": "BioAIPlans_2d_iso",
        "spacing": iso_spacing_2d,
        "patch_size": patch_2d_iso,
        "shape_must_be_divisible_by": div_by_2d_iso,
        "batch_size": batch_size,
        "median_image_size_in_voxels": iso_median_2d,
        "through_plane_spacing": target_spacing_t[0],
        "normalization_schemes": norm_schemes,
        "use_mask_for_norm": use_mask,
        "resampling_fn_data_kwargs": resamp_data_kw,
        "resampling_fn_seg_kwargs": resamp_seg_kw,
    }
    logger.info("2d_iso: spacing=%.4f  patch=%s  div_by=%s  batch=%d",
                iso_spacing, patch_2d_iso, div_by_2d_iso, batch_size)
    _add_size_variants("2d_iso", configurations["2d_iso"],
                       iso_spacing_2d, iso_median_2d, patch_override_2d)

    # ---- 3d_lowres_iso (same threshold as 3d_lowres, spacing increases uniformly)
    if median_volume_3d_iso > 0 and (patch_volume_3d_iso / median_volume_3d_iso) < _LOWRES_CREATION_THRESHOLD:
        lowres_iso_spacing = np.array(iso_spacing_t, dtype=np.float64)
        for _ in range(2000):  # safety cap
            lowres_iso_median = compute_new_shape(
                iso_median_t, iso_spacing_t, lowres_iso_spacing.tolist()
            ).astype(float)
            vol_iso = float(np.prod(lowres_iso_median))
            if vol_iso <= 0 or (patch_volume_3d_iso / vol_iso) >= _LOWRES_CREATION_THRESHOLD:
                break
            lowres_iso_spacing *= _SPACING_INCREASE_FACTOR

        lr_iso_median = [int(round(s)) for s in lowres_iso_median]
        lr_iso_patch, lr_iso_div_by = _compute_patch_size(
            lowres_iso_spacing.tolist(), lr_iso_median, patch_size_override=patch_size_override,
            min_feature_map_size=min_fmap,
            max_numpool=max_pool,
            initial_patch_ref_3d=initial_patch_ref_3d,
            initial_patch_ref_2d=initial_patch_ref_2d,
        )
        configurations["3d_lowres_iso"] = {
            "data_identifier": "BioAIPlans_3d_lowres_iso",
            "spacing": lowres_iso_spacing.tolist(),
            "patch_size": lr_iso_patch,
            "shape_must_be_divisible_by": lr_iso_div_by,
            "batch_size": batch_size,
            "median_image_size_in_voxels": lr_iso_median,
            "normalization_schemes": norm_schemes,
            "use_mask_for_norm": use_mask,
            "resampling_fn_data_kwargs": resamp_data_kw,
            "resampling_fn_seg_kwargs": resamp_seg_kw,
        }
        logger.info(
            "3d_lowres_iso created (iso patch covers %.1f%% < 25%% of iso median volume).  "
            "spacing=%.4f  patch=%s",
            100.0 * patch_volume_3d_iso / median_volume_3d_iso,
            lowres_iso_spacing[0], lr_iso_patch,
        )
        _add_size_variants("3d_lowres_iso", configurations["3d_lowres_iso"],
                           lowres_iso_spacing.tolist(), lr_iso_median, patch_size_override)
    else:
        logger.info(
            "3d_lowres_iso NOT created (iso patch covers %.1f%% of iso median volume).",
            100.0 * patch_volume_3d_iso / max(median_volume_3d_iso, 1),
        )

    # ---- Assemble plans -------------------------------------------------------
    experiment_plans: Dict[str, Any] = {
        "dataset_name": output_dir.name,
        "plans_name": "BioAIPlans",
        "channel_names": channel_names,
        "labels": labels_dict,
        "transpose_forward": transpose_forward,
        "transpose_backward": transpose_backward,
        "foreground_intensity_properties_per_channel": fp["foreground_intensity_properties_per_channel"],
        "label_class_ratios": fp.get("label_class_ratios"),
        "configurations": configurations,
    }

    plans_file = output_dir / "experiment_plans.json"
    with open(plans_file, "w") as fh:
        json.dump(experiment_plans, fh, indent=2)

    _print_plan_summary(fp, experiment_plans)
    logger.info("Experiment plans saved → %s", plans_file)
    logger.info("Configurations: %s", list(configurations.keys()))
    return experiment_plans
