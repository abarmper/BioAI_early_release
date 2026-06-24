"""nnUNet-style preprocessing pipeline — pure numpy + SimpleITK + scipy.

Reads ``experiment_plans.json`` (produced by ``command=plan``) and
preprocesses raw NIfTI volumes into ``.npz`` + ``.pkl`` pairs.

Per-case pipeline::

    Load → (C, Z, Y, X) → (C, X, Y, Z)
    → transpose_forward
    → crop_to_nonzero  (binary_fill_holes; -1 outside body)
    → normalize per channel (before resampling!)
    → resample to target spacing
    → sample foreground locations  (class_locations for training oversampling)
    → save .npz + .pkl

Output directories::

    {dataset_dir}/{data_identifier}/           # training cases
    {dataset_dir}/{data_identifier}_test/      # test cases (no seg)
    {dataset_dir}/{data_identifier}_val/       # validation cases (rare)
    {dataset_dir}/gt_segmentations/            # raw label copies
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from preprocessing.normalization import get_normalizer
from preprocessing.plan import group_cases_nnunet, read_dataset_json
from preprocessing.resampling import compute_new_shape, resample_data_or_seg
from preprocessing.utils import case_id_from_path, crop_to_nonzero, load_sitk_volume

logger = logging.getLogger(__name__)


# ===========================================================================
# Distance map computation for boundary loss
# ===========================================================================

def compute_signed_distance_maps(
    seg: np.ndarray,
    num_classes: int,
    include_background: bool = False,
    spacing: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Compute signed distance maps from an integer segmentation.

    For each foreground class, computes the Euclidean distance transform
    outside the class mask (positive) minus the EDT inside (negative).

    Parameters
    ----------
    seg : np.ndarray
        Integer segmentation ``(1, *spatial)`` with labels 0..num_classes-1.
    num_classes : int
        Total number of classes including background.
    include_background : bool
        If True, compute SDF for background (class 0) as well.
    spacing : sequence of float or None
        Voxel spacing for anisotropic EDT.

    Returns
    -------
    np.ndarray
        ``(C_fg, *spatial)`` float32 signed distance maps clipped to [-10, 10].
        C_fg = num_classes if include_background, else num_classes - 1.
    """
    from scipy.ndimage import distance_transform_edt

    seg_squeezed = seg[0]  # (*spatial)
    spatial_shape = seg_squeezed.shape
    sp: Optional[Tuple[float, ...]] = tuple(float(s) for s in spacing) if spacing is not None else None

    ch_start = 0 if include_background else 1
    sdfs = []
    for c in range(ch_start, num_classes):
        mask = (seg_squeezed == c)
        # Handle degenerate cases: class fully absent or fully present
        d_out: npt.NDArray[np.float32]
        if (~mask).any():
            d_out = np.asarray(distance_transform_edt(~mask, sampling=sp), dtype=np.float32)
        else:
            d_out = np.zeros(spatial_shape, dtype=np.float32)
        d_in: npt.NDArray[np.float32]
        if mask.any():
            d_in = np.asarray(distance_transform_edt(mask, sampling=sp), dtype=np.float32)
        else:
            d_in = np.zeros(spatial_shape, dtype=np.float32)
        sdfs.append(d_out - d_in)

    if len(sdfs) == 0:
        return np.zeros((0,) + spatial_shape, dtype=np.float32)
    result = np.stack(sdfs, axis=0)
    np.clip(result, -10.0, 10.0, out=result)
    return result


# ===========================================================================
# Foreground location sampling  (ported from nnUNet DefaultPreprocessor)
# ===========================================================================

def _sample_foreground_locations(
    seg: np.ndarray,
    foreground_labels: List[int],
    seed: int = 1234,
    min_num_samples: int = 10_000,
    min_percent_coverage: float = 0.01,
) -> Dict[int, np.ndarray]:
    """Sample spatial locations for each foreground label.

    Used during training for probabilistic oversampling of small classes.
    Matches the nnUNet v2 ``_sample_foreground_locations`` algorithm.

    Parameters
    ----------
    seg : np.ndarray
        Segmentation array ``(1, *spatial)``.
    foreground_labels : list of int
        Labels to sample (background 0 is excluded).
    seed : int
        Random seed for reproducibility.
    min_num_samples : int
        Minimum samples per label.
    min_percent_coverage : float
        Minimum fraction of voxels to sample per label.

    Returns
    -------
    dict
        ``{label_id: np.ndarray of shape (N, ndim)}``.
        Empty array if the label is absent in this case.
    """
    rng = np.random.RandomState(seed)
    seg_flat = seg[0]  # (spatial...)
    class_locs: Dict[int, np.ndarray] = {}

    for label in foreground_labels:
        coords = np.argwhere(seg_flat == label)
        n_vox = len(coords)
        if n_vox == 0:
            class_locs[label] = np.empty((0, seg_flat.ndim), dtype=np.int64)
            continue
        target = max(min_num_samples, int(np.ceil(n_vox * min_percent_coverage)))
        target = min(target, n_vox)
        idx = rng.choice(n_vox, size=target, replace=False)
        class_locs[label] = coords[idx]

    return class_locs


# ===========================================================================
# DefaultPreprocessor
# ===========================================================================

class DefaultPreprocessor:
    """nnUNet-style per-case preprocessor driven by experiment plans.

    Parameters
    ----------
    plans : dict
        Full ``experiment_plans.json`` content.
    configuration_name : str
        Which configuration to use (``"3d_fullres"``, ``"2d"``, ``"3d_lowres"``).
    """

    def __init__(self, plans: dict, configuration_name: str) -> None:
        self.plans = plans
        self.configuration_name = configuration_name
        if configuration_name not in plans["configurations"]:
            raise ValueError(
                f"Configuration '{configuration_name}' not in plans. "
                f"Available: {list(plans['configurations'].keys())}"
            )
        self.config = plans["configurations"][configuration_name]
        self.transpose_forward: List[int] = plans["transpose_forward"]
        self.transpose_backward: List[int] = plans["transpose_backward"]
        self.iprops: Dict = plans.get("foreground_intensity_properties_per_channel", {})

    def run_case(
        self,
        image_files: Sequence[str],
        seg_file: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """Preprocess a single case end-to-end.

        Pipeline
        --------
        1. Load via SimpleITK → ``(C, Z, Y, X)`` float32
        2. Convert to ``(C, X, Y, Z)`` (match spacing axis order)
        3. Transpose axes (``transpose_forward``)
        4. Crop to nonzero (``binary_fill_holes``; ``-1`` for outside body)
        5. Normalize per channel **before** resampling
        6. Resample to target spacing
        7. Compute signed distance maps (for boundary loss)
        8. Sample foreground locations (``class_locations``)

        Parameters
        ----------
        image_files : sequence of str
            One path per channel.
        seg_file : str or None
            Segmentation path, or ``None`` for test cases.

        Returns
        -------
        (data, seg, distance_map, properties)
            ``data``         : float32 ``(C, *spatial)``
            ``seg``          : int8 ``(1, *spatial)`` or ``None``
            ``distance_map`` : float32 ``(C_fg, *spatial)`` or ``None``
            ``properties``   : dict with metadata for reconstruction
        """
        # 1. Load all channels
        channel_arrays: List[np.ndarray] = []
        props_ref: Optional[Dict] = None
        for fpath in image_files:
            arr, props = load_sitk_volume(str(fpath))
            channel_arrays.append(arr)
            if props_ref is None:
                props_ref = props

        # Stack → (C, Z, Y, X)
        data = np.stack(channel_arrays, axis=0)
        assert props_ref is not None

        # Load segmentation
        seg: Optional[np.ndarray] = None
        if seg_file is not None:
            seg_arr, _ = load_sitk_volume(str(seg_file))
            seg = seg_arr[np.newaxis]  # (1, Z, Y, X)

        # 2. SimpleITK is (Z, Y, X); spacing is (x, y, z).
        #    Convert to (C, X, Y, Z) so axis i ↔ spacing[i].
        data = np.ascontiguousarray(data.transpose(0, 3, 2, 1))  # (C, X, Y, Z)
        if seg is not None:
            seg = np.ascontiguousarray(seg.transpose(0, 3, 2, 1))

        original_spacing: List[float] = props_ref["spacing"]  # (x, y, z)

        # 3. Transpose axes (low-res axis first)
        tf = self.transpose_forward
        ax = [0] + [i + 1 for i in tf]
        data = np.ascontiguousarray(data.transpose(ax))
        if seg is not None:
            seg = np.ascontiguousarray(seg.transpose(ax))
        spacing_t: List[float] = [original_spacing[i] for i in tf]

        shape_before_crop = list(data.shape[1:])

        # 4. Crop to nonzero
        data, seg, bbox = crop_to_nonzero(data, seg)
        shape_after_crop = list(data.shape[1:])

        # 5. Normalize per channel (must happen BEFORE resampling)
        norm_schemes = self.config["normalization_schemes"]
        use_mask = self.config["use_mask_for_norm"]
        normalization_params: List[Dict[str, Any]] = []
        for c in range(data.shape[0]):  # Loop over channels
            scheme = norm_schemes[c] if c < len(norm_schemes) else norm_schemes[-1]
            mask_flag = use_mask[c] if c < len(use_mask) else use_mask[-1]
            ch_props = self.iprops.get(str(c), {})
            normalizer = get_normalizer(scheme, use_mask=mask_flag, intensity_properties=ch_props)
            data[c], norm_p = normalizer.run(data[c], seg[0] if seg is not None else None)
            normalization_params.append(norm_p)

        # 6. Resample to target spacing (target spacing is in transposed axis order, i.e. low-res axis first)
        target_spacing = self.config["spacing"]
        if "through_plane_spacing" in self.config:
            # Through-plane spacing kept as-is (covers "2d" and "2d_iso")
            target_spacing_full = [spacing_t[0]] + list(target_spacing)
        else:
            target_spacing_full = list(target_spacing)

        new_shape = compute_new_shape(data.shape[1:], spacing_t, target_spacing_full) # New shape without the channel dimension.

        resamp_data_kw = dict(self.config["resampling_fn_data_kwargs"])
        resamp_seg_kw  = dict(self.config["resampling_fn_seg_kwargs"])

        data = resample_data_or_seg(
            data, new_shape, spacing_t, target_spacing_full, **resamp_data_kw
        )

        if seg is not None:
            seg_f = resample_data_or_seg(
                seg.astype(np.float32), new_shape, spacing_t, target_spacing_full,
                **resamp_seg_kw
            )
            seg = np.round(seg_f).astype(np.int8)

        # 7. Compute signed distance maps (for boundary loss)
        distance_map: Optional[np.ndarray] = None
        labels_dict: Dict[str, int] = self.plans.get("labels", {})
        num_classes = len(labels_dict)
        if seg is not None and num_classes > 1:
            distance_map = compute_signed_distance_maps(
                seg, num_classes, include_background=False,
                spacing=target_spacing_full,
            )

        # 8. Sample foreground locations (for training oversampling)
        # labels_dict may be {"0": "name", "1": "name"} or {"name": 0, "name": 1}
        class_locations: Optional[Dict] = None
        if seg is not None:
            fg_labels = []
            for k, v in labels_dict.items():
                try:
                    label_idx = int(k)
                except (ValueError, TypeError):
                    label_idx = int(v)
                if label_idx != 0:
                    fg_labels.append(label_idx)
            if fg_labels:
                class_locations = _sample_foreground_locations(seg, fg_labels)

        properties: Dict[str, Any] = {
            "case_id": case_id_from_path(str(image_files[0])),
            "original_spacing": original_spacing,
            "original_shape": props_ref["shape_original"],
            "spacing_after_transpose": spacing_t,
            "shape_before_crop": shape_before_crop,
            "shape_after_crop": shape_after_crop,
            "crop_bbox": bbox,
            "resampled_spacing": target_spacing_full,
            "resampled_shape": list(data.shape[1:]),
            "transpose_forward": self.transpose_forward,
            "transpose_backward": self.transpose_backward,
            "origin": props_ref["origin"],
            "direction": props_ref["direction"],
            "configuration": self.configuration_name,
            "class_locations": class_locations,
            "normalization_params": normalization_params,
        }

        return data.astype(np.float32), seg, distance_map, properties

    def run(
        self,
        raw_data_dir: str,
        output_dir: str,
        num_channels: int,
        num_processes: int = 8,
        split: str = "train",
        convert_to_binary_mask: bool = False,
    ) -> int:
        """Discover all cases and process in parallel.

        Parameters
        ----------
        raw_data_dir : str
            Raw dataset root (``imagesTr/``, ``labelsTr/``, etc.).
        output_dir : str
            Destination for ``.npz`` + ``.pkl`` files.
        num_channels : int
            Number of image channels per case.
        num_processes : int
            Parallel workers.
        split : str
            ``"train"``, ``"val"``, or ``"test"``.
        convert_to_binary_mask : bool
            Convert multi-class labels to binary.

        Returns
        -------
        int
            Number of cases processed.
        """
        if split == "train":
            images_folder = os.path.join(raw_data_dir, "imagesTr")
            labels_folder: Optional[str] = os.path.join(raw_data_dir, "labelsTr")
        elif split == "val":
            images_folder = os.path.join(raw_data_dir, "imagesVal")
            labels_folder = os.path.join(raw_data_dir, "labelsVal")
        elif split == "test":
            images_folder = os.path.join(raw_data_dir, "imagesTs")
            labels_candidate = os.path.join(raw_data_dir, "labelsTs")
            labels_folder = labels_candidate if os.path.isdir(labels_candidate) else None
        else:
            raise ValueError(f"Unknown split: {split}")

        if not os.path.isdir(images_folder):
            logger.info("Folder not found: %s — skipping '%s'.", images_folder, split)
            return 0

        has_labels = labels_folder is not None and os.path.isdir(labels_folder)
        image_groups, label_paths = group_cases_nnunet(
            images_folder, num_channels, labels_folder if has_labels else None
        )
        if not image_groups:
            return 0

        os.makedirs(output_dir, exist_ok=True)
        logger.info(
            "Preprocessing %d %s case(s) → %s  [config=%s, workers=%d]",
            len(image_groups), split, output_dir, self.configuration_name, num_processes,
        )

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = {
                executor.submit(
                    _worker,
                    self.plans,
                    self.configuration_name,
                    output_dir,
                    img_group,
                    label_paths[i] if label_paths else None,
                    convert_to_binary_mask,
                ): img_group
                for i, img_group in enumerate(image_groups)
            }
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc=f"Preprocessing [{split}]", unit="case",
            ):
                future.result()  # re-raise worker exceptions

        return len(image_groups)


# ===========================================================================
# Worker (top-level so multiprocessing can pickle it)
# ===========================================================================

def _worker(
    plans: dict,
    configuration_name: str,
    output_dir: str,
    image_files: Sequence[str],
    seg_file: Optional[str],
    convert_to_binary_mask: bool,
) -> None:
    proc = DefaultPreprocessor(plans, configuration_name)
    data, seg, distance_map, properties = proc.run_case(image_files, seg_file)

    if seg is not None and convert_to_binary_mask:
        seg = (seg > 0).astype(np.int8)
        # Recompute distance maps for binary mask
        if distance_map is not None:
            distance_map = compute_signed_distance_maps(
                seg, num_classes=2, include_background=False,
                spacing=properties.get("resampled_spacing"),
            )

    case_id = properties["case_id"]
    os.makedirs(output_dir, exist_ok=True)

    npz_path = os.path.join(output_dir, f"{case_id}.npz")
    pkl_path = os.path.join(output_dir, f"{case_id}.pkl")

    if seg is not None:
        save_kwargs = {"data": data, "seg": seg}
        if distance_map is not None:
            save_kwargs["distance_map"] = distance_map
        np.savez_compressed(npz_path, **save_kwargs)
    else:
        np.savez_compressed(npz_path, data=data)

    with open(pkl_path, "wb") as fh:
        pickle.dump(properties, fh)


# ===========================================================================
# Public API
# ===========================================================================

def preprocess(
    raw_data_dir: Path,
    output_dir: Path,
    configuration: str = "3d_fullres",
    num_workers: int = 8,
    convert_to_binary_mask: bool = False,
    unpack: bool = False,
    delete_npz_after_unpack: bool = True,
) -> None:
    """Preprocess raw NIfTI volumes using an experiment plan configuration.

    Parameters
    ----------
    raw_data_dir : Path
        Raw dataset root (must contain ``imagesTr/``, ``labelsTr/``,
        ``dataset.json``).
    output_dir : Path
        Dataset root (``data/{dataset}/``).  Must contain
        ``experiment_plans.json``.
    configuration : str
        ``"3d_fullres"``, ``"2d"``, or ``"3d_lowres"``.
    num_workers : int
        Parallel worker processes.
    convert_to_binary_mask : bool
        Convert multi-class labels to binary.
    """
    output_dir = Path(output_dir)
    raw_data_dir = Path(raw_data_dir)

    plans_file = output_dir / "experiment_plans.json"
    if not plans_file.exists():
        raise FileNotFoundError(
            f"Experiment plans not found at '{plans_file}'. "
            "Run 'command=plan' first."
        )

    with open(plans_file) as fh:
        plans = json.load(fh)

    if configuration not in plans["configurations"]:
        raise ValueError(
            f"Configuration '{configuration}' not in plans. "
            f"Available: {list(plans['configurations'].keys())}"
        )

    # Number of channels from plans (read from dataset.json during planning)
    num_channels = len(plans["channel_names"])
    data_id = plans["configurations"][configuration]["data_identifier"]
    preprocessor = DefaultPreprocessor(plans, configuration)

    # Training cases
    n_tr = preprocessor.run(
        str(raw_data_dir), str(output_dir / data_id),
        num_channels, num_workers, split="train",
        convert_to_binary_mask=convert_to_binary_mask,
    )
    # Test cases
    n_ts = preprocessor.run(
        str(raw_data_dir), str(output_dir / f"{data_id}_test"),
        num_channels, num_workers, split="test",
        convert_to_binary_mask=False,
    )
    # Validation cases (rare)
    n_val = preprocessor.run(
        str(raw_data_dir), str(output_dir / f"{data_id}_val"),
        num_channels, num_workers, split="val",
        convert_to_binary_mask=convert_to_binary_mask,
    )

    # Copy raw ground-truth labels → gt_segmentations/ . This is optional as the ground truth labels are already included 
    # in the .npz/.pkl pairs for training/validation cases.
    gt_dir = output_dir / "gt_segmentations"
    labels_tr = raw_data_dir / "labelsTr"
    if labels_tr.is_dir():
        gt_dir.mkdir(parents=True, exist_ok=True)
        for lbl in sorted(labels_tr.glob("*.nii.gz")):
            dst = gt_dir / lbl.name
            if not dst.exists():
                shutil.copy2(str(lbl), str(dst))
        logger.info("Ground-truth labels copied → %s", gt_dir)

    # Optionally unpack .npz → .npy for faster mmap-based loading
    if unpack:
        from data_loading.dataset import BioAIDataset
        for subdir in [output_dir / data_id,
                       output_dir / f"{data_id}_test",
                       output_dir / f"{data_id}_val"]:
            if subdir.is_dir():
                BioAIDataset.unpack_dataset(
                    str(subdir),
                    num_processes=max(1, num_workers),
                    delete_npz=delete_npz_after_unpack,
                )
        if delete_npz_after_unpack:
            logger.info("Unpacked .npz → .npy for all splits (.npz removed).")
        else:
            logger.info("Unpacked .npz → .npy for all splits.")

    logger.info(
        "Done. train=%d  test=%d  val=%d | config=%s | dir=%s",
        n_tr, n_ts, n_val, configuration, data_id,
    )
