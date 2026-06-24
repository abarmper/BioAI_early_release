# Preprocessing

## Overview

The preprocessing pipeline converts raw NIfTI volumes into `.npz` + `.pkl` pairs ready for training. It is split into two commands:

- `command=plan` — extract the dataset fingerprint and generate `experiment_plans.json`
- `command=preprocess` — apply per-case preprocessing driven by those plans

Both follow the nnUNet v2 design closely.

---

## File Map

| File | Responsibility |
|---|---|
| `plan.py` | `plan_dataset` (fingerprint) + `plan_experiment` (topology + configs) |
| `preprocessing4.py` | `DefaultPreprocessor` — per-case pipeline + parallelism |
| `normalization.py` | `CTNormalization`, `ZScoreNormalization`, `NoNormalization` |
| `resampling.py` | `resample_data_or_seg` — anisotropic two-pass resampling |
| `compute_statistics.py` | Raw fingerprint stats (spacing, intensity, class ratios) |
| `convert_to_nifti.py` | `revert_to_nifti` — reverse pipeline for NIfTI export |
| `utils.py` | `crop_to_nonzero`, `load_sitk_volume`, `case_id_from_path` |

---

## Phase 1 — Dataset Fingerprint (`plan_dataset`)

Scans `{raw_data_dir}/imagesTr/` and `labelsTr/`, groups images into per-case channel tuples (nnUNet `{case_id}_{0000}.nii.gz` naming), and runs `compute_statistics` in parallel.

### `compute_statistics` / `process_volume`

Each case worker:

1. Loads all channels via SimpleITK.
2. Crops to the non-zero bounding box (union of segmentation foreground + any non-zero image voxel).
3. Records per-case spacing `(x, y, z)` and cropped shape.
4. Samples up to `N / num_cases` foreground voxels per channel for intensity statistics (with replacement to avoid under-representing small foreground regions). Budget: 10^8 total foreground voxels across the dataset.
5. Samples labels inside the bbox to estimate per-class frequency.

The `DatasetStatistics` accumulator merges results from all workers and computes:

| Fingerprint key | Description |
|---|---|
| `spacings` | Per-case spacing list `(x, y, z)` |
| `shapes_after_crop` | Per-case shape after non-zero crop `(x, y, z)` |
| `median_spacing` | Median spacing across all cases |
| `10th_percentile_spacing` | 10th-percentile spacing |
| `median_size_after_crop` | Median cropped volume size |
| `median_relative_size_after_cropping` | Median fraction of bbox / full volume |
| `foreground_intensity_properties_per_channel` | Per-channel mean, std, min, max, 0.5th/99.5th percentile |
| `label_class_ratios` | Sampled class frequency ratios (used for auto class weights) |

Output: `dataset_fingerprint.json`.

---

## Phase 2 — Experiment Planning (`plan_experiment`)

Reads `dataset_fingerprint.json` + `raw/dataset.json` and generates one configuration entry per plan type.

### Target Spacing

Computed as the **median spacing** across all cases. For anisotropic data (worst-axis spacing > 3× other axes AND worst-axis voxel count < 1/3 of in-plane), the through-plane target spacing is clamped to the 10th-percentile raw spacing to avoid excessive upsampling.

The lowest-resolution (highest spacing) axis is placed **first** via `transpose_forward` so the anisotropy-aware pooling logic always sees the low-res axis at index 0. `transpose_backward` reverses this for reconstruction.

### Normalization Planning

Per-channel normalization scheme is selected from the channel name in `dataset.json`:

| Channel name | Scheme |
|---|---|
| `CT`, `ct` | `CTNormalization` |
| `noNorm`, `none` | `NoNormalization` |
| anything else | `ZScoreNormalization` |

`use_mask_for_norm` is set to `True` for `ZScoreNormalization` channels when `median_relative_size_after_cropping < 0.75` (i.e. less than 75% of the volume is non-zero after crop — typical for brain MRI). CT channels never use the mask.

### Patch Size (`_compute_patch_size`)

Two-stage nnUNet algorithm:

**Stage 1 — aspect-ratio initialisation:** Compute an initial isotropic patch scaled so the total volume equals `256³` voxels (3D) or `2048²` (2D), then scale each axis by `1/spacing` to respect the aspect ratio. Clip to the median image size.

**Stage 2 — topology divisibility:** Simulate the encoder stage-by-stage with `_get_pool_and_conv_props` to find `num_pool_per_axis`, then pad the patch up to be divisible by `2^num_pool` per axis.

### Network Topology (`_get_pool_and_conv_props`)

Simulates the UNet encoder depth-first. At each stage:

- **Valid axes**: those with current feature map size ≥ `2 × min_feature_map_size` (default 4) and pooling count < `max_numpool`.
- **Anisotropy filter**: only pool axes whose current spacing is within a factor of 2 of the finest currently available spacing. Coarser axes are skipped until repeated in-plane pooling narrows the gap.
- **Conv kernels**: an axis switches from kernel size 1 to 3 once its spacing is within 2× of the minimum. This mirrors the nnUNet rule that 1×1 kernels are used across thick slices until the resolution gap is small enough.

This produces `pool_op_kernel_sizes` (one per encoder stage), `conv_kernel_sizes`, `shape_must_be_divisible_by`.

### Configurations Generated

| Config | Spacing | Notes |
|---|---|---|
| `3d_fullres` | Median spacing (transposed) | Main 3D config |
| `2d` | In-plane median spacing (last 2 transposed axes) | 2D slice-wise training; stores `through_plane_spacing` |
| `3d_fullres_iso` | `max(median spacing)` isotropic | Required by `mednext` / `medmoenext` (fixed-topology models) |
| `2d_iso` | Isotropic in-plane | 2D with isotropic in-plane spacing |
| `3d_lowres` | Coarsened until patch covers ≥25% of median volume | Created only when fullres patch < 25% of median volume |
| `3d_lowres_iso` | Same threshold, isotropic | Same condition applied to iso configs |

All configs share the same normalization schemes and resampling kwargs.

Output: `experiment_plans.json`.

---

## Per-Case Preprocessing (`DefaultPreprocessor.run_case`)

```
Load NIfTI → (C, Z, Y, X) → (C, X, Y, Z)
→ transpose_forward               (low-res axis first)
→ crop_to_nonzero                 (binary_fill_holes; -1 outside body)
→ normalize per channel           (BEFORE resampling)
→ resample to target spacing
→ compute signed distance maps    (for boundary loss, if seg present)
→ sample foreground locations     (for training oversampling)
→ save .npz + .pkl
```

### Step-by-step

**1. Load** — SimpleITK loads `(Z, Y, X)` arrays. Stacked channels become `(C, Z, Y, X)`.

**2. Axis reorder** — converted to `(C, X, Y, Z)` so `axis i ↔ spacing[i]` (SimpleITK spacing is `(x, y, z)`).

**3. Transpose** — `transpose_forward` permutes so the low-res axis is first. All spatial arrays (data + seg) are transposed together.

**4. Crop to nonzero** (`utils.crop_to_nonzero`) — union of all-channel non-zero masks, `binary_fill_holes`, then bounding-box crop. Voxels in `seg` that were 0 and outside the body mask are set to `-1`. For test cases (no `seg`), a synthetic segmentation of `0` inside / `-1` outside is created. The crop `bbox` is saved to properties for reconstruction.

**5. Normalize** — applied **before** resampling to avoid interpolating over intensity discontinuities. Each channel is normalized independently using the scheme from the plans (see Normalization section below). Normalization parameters are saved to properties.

**6. Resample** — `resample_data_or_seg` resamples to the target spacing. For 2D configs, the through-plane axis is kept at its original spacing. Uses the two-pass anisotropic strategy when spacing anisotropy > 3 (see Resampling below).

**7. Signed distance maps** — `compute_signed_distance_maps` computes per-class Euclidean signed distance transforms (EDT outside − EDT inside), clipped to `[-10, 10]`. Stored in the `.npz` as `distance_map (C_fg, *spatial)`. Only computed when a segmentation is present and needed by the loss (boundary loss).

**8. Foreground locations** — `_sample_foreground_locations` records up to `max(10000, 1% of voxels)` voxel coordinates per foreground class. Stored in `properties["class_locations"]` for the dataloader's foreground oversampling.

### Output files

Per case, two files are written to `{dataset_dir}/{data_identifier}/`:

- `{case_id}.npz` — compressed arrays: `data (C, *spatial)`, `seg (1, *spatial)`, optional `distance_map (C_fg, *spatial)`
- `{case_id}.pkl` — metadata dict

### `.pkl` properties

| Key | Description |
|---|---|
| `case_id` | Case identifier |
| `original_spacing` | Raw NIfTI spacing `(x, y, z)` |
| `original_shape` | Raw NIfTI shape `(x, y, z)` |
| `spacing_after_transpose` | Spacing in transposed axis order |
| `shape_before_crop` | Shape before nonzero crop |
| `shape_after_crop` | Shape after nonzero crop |
| `crop_bbox` | `[[lo, hi], ...]` per axis |
| `resampled_spacing` | Final target spacing |
| `resampled_shape` | Final spatial shape |
| `transpose_forward` / `transpose_backward` | Axis permutation |
| `origin`, `direction` | NIfTI geometry metadata for reconstruction |
| `class_locations` | `{label_id: (N, ndim) coords}` for oversampling |
| `normalization_params` | Per-channel normalization params for denormalization |

---

## Normalization (`normalization.py`)

All normalizers operate **in-place** on a single-channel float32 array.

### `CTNormalization`

1. Clip to dataset `[percentile_00_5, percentile_99_5]`.
2. Subtract dataset foreground mean, divide by dataset foreground std.

Uses dataset-level statistics from `foreground_intensity_properties_per_channel` in the plans. The mask is **never** applied (CT body regions are always well-defined from HU values).

### `ZScoreNormalization`

Per-image z-score. If `use_mask_for_norm=True`, mean and std are computed only from voxels where `seg >= 0` (inside the body after crop). Outside-body voxels (`seg == -1`) are set to zero after normalization. The outside-body mask is derived from the image itself during preprocessing (not from the GT labels), so there is no label leakage.

### `NoNormalization`

Identity pass — no transformation applied. Used for channels declared as `noNorm` / `none`.

---

## Resampling (`resampling.py`)

`resample_data_or_seg(data, new_shape, current_spacing, new_spacing, is_seg, order, order_z, force_separate_z)` processes each channel independently.

### Single-pass (isotropic)

`skimage.transform.resize` with spline interpolation (order 3 for images, order 0 / nearest for segmentations).

### Two-pass (anisotropic, `do_separate_z=True`)

Triggered when either the source or target spacing is anisotropic (max/min > 3):

1. **In-plane (axes 1, 2)**: resize each z-slice independently using `order` (3 for images, 0 for seg).
2. **Through-plane (axis 0)**: for each (y, x) pixel, resize the 1D z-profile using `order_z` (0 by default — nearest neighbour through-plane, which avoids interpolation artefacts across thick slices).

Default kwargs from the plans:
- Images: `order=3, order_z=0, force_separate_z=None` (auto-detect)
- Segmentations: `order=1, order_z=0, force_separate_z=None`

---

## NIfTI Reconstruction (`convert_to_nifti.py` / `revert_to_nifti`)

Reverses the preprocessing pipeline:

```
preprocessed (C, *resampled) 
→ resample⁻¹ to shape_after_crop
→ uncrop (pad back to shape_before_crop using bbox)
→ reverse transpose (transpose_backward)
→ (C, X, Y, Z) → (C, Z, Y, X)  (SimpleITK order)
→ restore origin + direction + original spacing
→ SimpleITK Image
```

Optionally denormalizes image channels using the saved `normalization_params`. Used by the testing pipeline (`export.py`) and the standalone CLI (`python -m preprocessing.convert_to_nifti`).

---

## Input Data Format

Expected raw dataset layout:

```
{raw_data_dir}/
  dataset.json            ← channel names + label definitions
  imagesTr/
    {case_id}_0000.nii.gz  ← channel 0
    {case_id}_0001.nii.gz  ← channel 1 (multi-channel)
    ...
  labelsTr/
    {case_id}.nii.gz
  imagesTs/               ← optional test images
  labelsTs/               ← optional test labels
  imagesVal/              ← optional val images
  labelsVal/              ← optional val labels
```

Single-channel images without the `_0000` suffix are also accepted (treated as channel 0).

### `dataset.json`

```json
{
  "channel_names": {"0": "CT"},
  "labels": {"background": 0, "liver": 1},
  "file_ending": ".nii.gz"
}
```

For multi-channel MRI:
```json
{
  "channel_names": {"0": "T1", "1": "T1CE", "2": "T2", "3": "FLAIR"},
  "labels": {"background": 0, "whole_tumor": 1}
}
```

The legacy nnUNet `"modality"` key is accepted as an alias for `"channel_names"`.
