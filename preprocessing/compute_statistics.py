"""
Raw dataset fingerprint statistics — low-level per-case extraction.

This module collects per-case voxel spacing, cropped shape, and per-channel
foreground intensity samples.  The high-level planning logic (target spacing,
normalization scheme, anisotropy decisions) lives in ``plan.py``.

Output format mirrors nnUNet's ``DatasetFingerprintExtractor``:
  - per-case ``spacings`` and ``shapes_after_crop`` for target-spacing computation
  - ``foreground_intensity_properties_per_channel`` for normalization planning
  - ``median_relative_size_after_cropping`` for non-zero mask decision
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import logging
from scipy.ndimage import binary_fill_holes

logger = logging.getLogger(__name__)


# nnUNet-style: total foreground voxels to sample across the full dataset
_NUM_FOREGROUND_VOXELS_FOR_INTENSITY_STATS = int(10e7) # If we have 100 cases and this value set to 10e7, this means 1 million voxels per case (each case has about 200 million voxels).
_RANDOM_SEED = 1234

# ---------------------------------------------------------------------------
# Non-zero crop
# ---------------------------------------------------------------------------

def crop_to_nonzero(
    vol_arrays: List[np.ndarray], mask_array: np.ndarray
) -> Tuple[List[np.ndarray], np.ndarray, Tuple[int, ...]]:
    """
    Crop volumes and segmentation to the bounding box of non-zero voxels.

    The non-zero region is the union of the segmentation foreground
    (``mask > 0``) and any non-zero image voxels, mirroring nnUNet's
    ``crop_to_nonzero`` approach. It thus can not be used in testing 
    (unless dummy mask used).

    Arguments
    ---------
    vol_arrays:
        List of image volumes (one per channel) as numpy arrays. Assumed to be in z, y, x memory layout as produced by ``sitk.GetArrayFromImage()``.
    mask_array:
        Segmentation volume as a numpy array. Assumed to be in z, y, x memory layout as produced by ``sitk.GetArrayFromImage()``.

    Returns
    -------
    (cropped_volumes, cropped_mask, cropped_shape_zyx)
    """
    nonzero = mask_array > 0 # As you can observe, this process uses the segmentation mask so it can not be used for testing (unless dummy mask is used).
    for arr in vol_arrays:
        nonzero |= arr != 0

    coords = np.argwhere(nonzero) # Given 3D data, this will return an array of shape (N, 3). N is the number of non-zero voxels.
    if len(coords) == 0:
        return vol_arrays, mask_array, mask_array.shape

    lo = coords.min(axis=0) # Point with minimum coordinates among the non-zero voxels (inclusive)
    hi = coords.max(axis=0) + 1 # Point with maximum coordinates among the non-zero voxels (exclusive)
    slices = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi)) # Iterate over z, y, x dimensions to create a tuple of slice objects that can be used to index the arrays and extract the non-zero bounding box.

    cropped_vols = [arr[slices] for arr in vol_arrays] # Do the actual slicing.
    cropped_mask = mask_array[slices]
    return cropped_vols, cropped_mask, cropped_vols[0].shape


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------

class DatasetStatistics:
    """Accumulates per-case statistics across parallel workers."""

    def __init__(self) -> None:
        self.spacings: List[List[float]] = []           # per-case, SimpleITK (x,y,z)
        self.shapes_after_crop: List[List[int]] = []    # per-case, (x,y,z)
        self.relative_sizes_after_crop: List[float] = []
        # channel index → list of per-case float32 arrays of sampled foreground intensities.
        # Stored as a list of arrays (not a single concatenated array) so merge() is O(1)
        # per case; concatenation happens once in compute_fingerprint().
        self.intensities_per_channel: Dict[int, List[np.ndarray]] = {}
        self.mask_labels: set = set()
        self.label_sample_counts: Dict[int, int] = {}  # class_id → total voxels sampled across all cases
        self.label_total_samples: int = 0              # total label voxels sampled across all cases

    def merge(self, other: "DatasetStatistics") -> None:
        """Merge another DatasetStatistics into this one (in-place)."""
        self.spacings.extend(other.spacings)
        self.shapes_after_crop.extend(other.shapes_after_crop)
        self.relative_sizes_after_crop.extend(other.relative_sizes_after_crop)
        for ch, vals in other.intensities_per_channel.items():
            bucket = self.intensities_per_channel.setdefault(ch, [])
            if isinstance(vals, np.ndarray):
                bucket.append(vals)
            else:
                # Backwards-compat: tolerate legacy Python-list samples.
                bucket.append(np.asarray(vals, dtype=np.float32))
        self.mask_labels.update(other.mask_labels)
        for cls, cnt in other.label_sample_counts.items():
            self.label_sample_counts[cls] = self.label_sample_counts.get(cls, 0) + cnt
        self.label_total_samples += other.label_total_samples

    def compute_fingerprint(self) -> Dict[str, Any]:
        """
        Aggregate per-case data into the nnUNet-style dataset fingerprint dict.

        Returns
        -------
        dict with keys:
          ``num_cases``, ``num_classes``,
          ``spacings``, ``shapes_after_crop``,
          ``median_spacing``, ``10th_percentile_spacing``,
          ``median_size_after_crop``,
          ``median_relative_size_after_cropping``,
          ``foreground_intensity_properties_per_channel``
        """
        spacings_arr = np.array(self.spacings, dtype=float)      # (N, 3) x,y,z where N is the number of cases
        shapes_arr   = np.array(self.shapes_after_crop, dtype=float)  # (N, 3) x,y,z

        median_spacing  = np.median(spacings_arr, axis=0).tolist() # x,y,z median spacing across all cases
        p10_spacing     = np.percentile(spacings_arr, 10, axis=0).tolist()
        median_size_ac  = np.median(shapes_arr, axis=0).tolist()

        median_rel_size = (
            float(np.median(self.relative_sizes_after_crop))
            if self.relative_sizes_after_crop else 1.0
        )

        # ---- per-channel intensity statistics (nnUNet convention) -----------
        intensity_props: Dict[str, Dict[str, float]] = {}
        for ch_idx in sorted(self.intensities_per_channel.keys()):
            chunks = self.intensities_per_channel[ch_idx]
            if not chunks:
                continue
            ch_arr = np.concatenate(chunks).astype(np.float32, copy=False)
            if len(ch_arr) == 0:
                continue
            p005, median_i, p995 = np.percentile(ch_arr, [0.5, 50.0, 99.5])
            intensity_props[str(ch_idx)] = {
                "mean":            float(np.mean(ch_arr)),
                "std":             float(np.std(ch_arr)),
                "median":          float(median_i),
                "min":             float(np.min(ch_arr)),
                "max":             float(np.max(ch_arr)),
                "percentile_00_5": float(p005),
                "percentile_99_5": float(p995),
            }

        # ---- label class ratios (sampled, same budget as intensity stats) ---
        if self.label_total_samples > 0:
            label_class_ratios = {
                str(cls): cnt / self.label_total_samples
                for cls, cnt in sorted(self.label_sample_counts.items())
            }
        else:
            label_class_ratios = None  # no labels provided

        return {
            "num_cases":  len(self.spacings),
            "num_classes": int(len(self.mask_labels)),
            # per-case lists (needed by planner to compute target spacing)
            "spacings":          [list(s) for s in self.spacings],
            "shapes_after_crop": [list(s) for s in self.shapes_after_crop],
            # dataset-level aggregates
            "median_spacing":           median_spacing,
            "10th_percentile_spacing":  p10_spacing,
            "median_size_after_crop":   median_size_ac,
            "median_relative_size_after_cropping": median_rel_size,
            # per-channel intensity fingerprint (nnUNet format)
            "foreground_intensity_properties_per_channel": intensity_props,
            # average label class ratios estimated by random sampling
            "label_class_ratios": label_class_ratios,
        }


# ---------------------------------------------------------------------------
# Per-case worker
# ---------------------------------------------------------------------------

def process_volume(
    volume_paths: Tuple[str, ...],
    mask_path: str | None,
    num_samples: int = 10_000,
) -> DatasetStatistics:
    """
    Per-case worker.

    Loads all channel volumes and the segmentation, crops to the non-zero
    bounding box (nnUNet-style), then collects foreground intensity samples
    per channel using random sampling *with replacement* so that small
    foreground regions are not underrepresented.

    Parameters
    ----------
    volume_paths:
        Tuple of image file paths, one per channel.
    mask_path:
        Segmentation file path. Can be ``None`` if no segmentations are available (e.g. test split when no training dataset is available), but then the cropping will be based on non-zero image voxels only.
    num_samples:
        Maximum foreground voxels to sample per channel for this case.
    """
    stats = DatasetStatistics()
    try:
        volumes = [sitk.ReadImage(str(vp)) for vp in volume_paths]
        if mask_path is not None:
            mask = sitk.ReadImage(str(mask_path))
        else:
            # Create empty mask with zeros everywhere and the same shape as the volumes.
            mask = sitk.Image(volumes[0].GetSize(), sitk.sitkUInt8)

    except RuntimeError as exc:
        logger.warning(f"Cannot read {volume_paths}: {exc} — skipping this case.")
        return stats
    

    # --- metadata from reference (first) volume, assume all channels have same spacing and size ----------------------------
    ref          = volumes[0]
    spacing      = list(ref.GetSpacing())   # (x, y, z) in mm
    shape_before = list(ref.GetSize())      # (x, y, z) in voxels

    # --- numpy arrays (SimpleITK → z, y, x memory layout) -----------------
    vol_arrays = [sitk.GetArrayFromImage(v).astype(np.float32) for v in volumes]
    mask_array = sitk.GetArrayFromImage(mask)

    # --- crop to non-zero bounding box (nnUNet-style) ----------------------
    vol_arrays_c, mask_c, shape_after_zyx = crop_to_nonzero(vol_arrays, mask_array)

    # Convert cropped shape back to (x, y, z) to match SimpleITK convention
    shape_after_xyz = list(reversed(shape_after_zyx))

    stats.spacings.append(spacing)
    stats.shapes_after_crop.append(shape_after_xyz)

    total_before = float(np.prod(shape_before))
    total_after  = float(np.prod(shape_after_xyz))
    stats.relative_sizes_after_crop.append(
        total_after / total_before if total_before > 0 else 1.0
    )

    # --- per-channel foreground intensity sampling -------------------------
    fg_mask = mask_c > 0 if mask_path is not None else np.any([arr != 0 for arr in vol_arrays_c], axis=0) # Only sample from voxels that are labeled as foreground in the segmentation mask. If no mask, use any channel with nonzero value.
    rng     = np.random.RandomState(_RANDOM_SEED)   # fixed seed for reproducibility inside each worker.
    for ch_idx, arr in enumerate(vol_arrays_c):
        fg_pixels = arr[fg_mask] # Only foreground voxels are considered for candidates.
        if len(fg_pixels) == 0:
            # No foreground voxels for this channel in this case, skip sampling.
            continue
        n = min(num_samples, max(len(fg_pixels), 1))
        sampled = rng.choice(fg_pixels, size=n, replace=True) # If we set replace=False then we may under-represent the sparse foreground voxels.
        # Store as float32 (one chunk per case). This avoids the ~7x memory bloat of
        # Python lists of floats both in the worker and in the main-process aggregator,
        # and is what compute_fingerprint() ultimately reads as a numpy array anyway.
        stats.intensities_per_channel.setdefault(ch_idx, []).append(
            sampled.astype(np.float32, copy=False)
        )

    # --- label class ratio sampling (same rng and same num_samples budget) -
    # Sample num_samples voxels from the cropped label volume (including some
    # background voxels) to estimate the average class frequency across the dataset.
    # This process samples from the region of the mask that will eventually be fed into the network after preprocessing (cropping).
    if mask_path is not None:
        # Build the preprocessing-style bbox mask (image-only, hole-filled),
        # then sample labels only inside that bbox.
        nonzero_mask = np.zeros(vol_arrays[0].shape, dtype=bool)
        for arr in vol_arrays:
            nonzero_mask |= (arr != 0)
        nonzero_mask = binary_fill_holes(nonzero_mask)

        coords = np.argwhere(nonzero_mask) if nonzero_mask is not None else np.empty((0, 3), dtype=int) #  len(np.empty((0, 3), dtype=int)) = 0
        if len(coords) == 0:
            bbox_mask = np.ones_like(mask_array, dtype=bool)
        else:
            lo = coords.min(axis=0)
            hi = coords.max(axis=0) + 1
            slicer = tuple(slice(int(l), int(h)) for l, h in zip(lo, hi))
            bbox_mask = np.zeros_like(mask_array, dtype=bool)
            bbox_mask[slicer] = True

        label_flat = mask_array[bbox_mask].ravel()
        n = min(num_samples, len(label_flat))
        sampled_lbl = rng.choice(label_flat, size=n, replace=True)
        unique_cls, cls_counts = np.unique(sampled_lbl, return_counts=True)
        for cls, cnt in zip(unique_cls.tolist(), cls_counts.tolist()):
            stats.label_sample_counts[int(cls)] = (
                stats.label_sample_counts.get(int(cls), 0) + int(cnt)
            )
        stats.label_total_samples += n
    # else: no label provided — label_sample_counts stays empty, ratios → None

    # --- segmentation labels -----------------------------------------------
    stats.mask_labels.update(int(lb) for lb in np.unique(mask_array))

    return stats


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_statistics(
    volume_paths: List[Tuple[str, ...]],
    mask_paths:   Sequence[Optional[str]] | None,
    num_workers:  Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract raw dataset fingerprint statistics from raw NIfTI volumes.
    This will apply only to training cases, test data will use the statistics
    computed from the training set.

    Runs ``process_volume`` in parallel across all training cases and
    aggregates the results into the nnUNet-style fingerprint dict returned
    by ``DatasetStatistics.compute_fingerprint()``.

    Parameters
    ----------
    volume_paths:
        Per-case channel path tuples, e.g.
        ``[(case1_ch0, case1_ch1), (case2_ch0, ...), …]``.
    mask_paths:
        Corresponding segmentation paths (one per case). Can be ``None`` if no segmentations are available (e.g. test split), but then the cropping will be based on non-zero image voxels only, which may be less accurate.
    num_workers:
        Parallel worker processes.  ``None`` → all available CPUs.

    Returns
    -------
    dict
        Raw fingerprint as produced by
        ``DatasetStatistics.compute_fingerprint()``.
        Use ``plan.py::plan_dataset()`` to add planning decisions on top.
    """
    num_cases = len(volume_paths)
    if num_cases == 0:
        raise ValueError("volume_paths is empty — nothing to compute.")
    
    if mask_paths is None:
        logger.warning("mask_paths is None — will perform cropping only based on the image, not the segmentation mask.")
        mask_paths = [None] * num_cases


    # Distribute the total voxel budget evenly across cases (nnUNet strategy)
    samples_per_case = max(1, _NUM_FOREGROUND_VOXELS_FOR_INTENSITY_STATS // num_cases)

    agg = DatasetStatistics() # Aggregator

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_volume, v, m, samples_per_case): (v, m)
            for v, m in zip(volume_paths, mask_paths)
        }
        total = len(futures)
        for future in tqdm(
            as_completed(futures), total=total, desc="Fingerprint extraction"
        ):
            agg.merge(future.result())
            # Release the worker's DatasetStatistics as soon as it has been merged.
            # Without this, the futures dict (and each Future's cached _result) pin
            # every per-case result in main-process memory until the with-block exits,
            # making peak RAM grow as O(num_cases) instead of O(num_workers).
            futures.pop(future, None)
            future._result = None

    return agg.compute_fingerprint() # This should only be done after we have recieved the resluts from all workers.
