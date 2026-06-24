# Augmentation

## Overview

The augmentation pipeline is built on [batchgeneratorsv2](https://github.com/MIC-DKFZ/batchgeneratorsv2) transforms (CPU path) or native PyTorch GPU ops (GPU path). Both paths apply transforms in the same order and are controlled by the same YAML configs under `configs/augmentation/`.

Validation uses a minimal pipeline: only label cleanup (`RemoveLabelTransform`) and optional deep-supervision downsampling.

---

## Augmentation Families

Select a family with `augmentation=<name>` on the CLI. All families share the same transform types; they differ only in which transforms are enabled and their probability/range settings.

| Family    | Use case                                           |
|-----------|----------------------------------------------------|
| `nnunet`  | Default. Matches nnUNet v2 settings.               |
| `fast`    | Lighter augmentation for faster iteration.         |
| `extended`| Heavier augmentation; adds elastic deformation.    |
| `cutmix`  | Same as `nnunet` but with CutMix enabled.          |

---

## Transform Descriptions

### Spatial Transform

Applies random **rotation** and **scaling** to the patch via an affine warp (`grid_sample`). Both image and segmentation are transformed with the same grid (bilinear for image, nearest for segmentation).

- **Rotation** — sampled uniformly from `rotation_for_DA`, which is computed by `geometry.py` based on patch dimensionality and anisotropy:
  - 3D isotropic: ±30°
  - 3D anisotropic (dummy-2D mode): ±180° in H,W plane only
  - 2D square patches: ±180°; 2D non-square: ±15°
- **Scaling** — uniform scale applied identically across all axes.
- **Elastic deformation** — optional (enabled in `extended` only). Gaussian-smoothed random displacement fields are added to the affine grid. Controlled by `p_elastic`, `elastic_deform_scale`, and `elastic_deform_magnitude`.
- **Dummy-2D DA** — for strongly anisotropic 3D volumes (thick-slice CT), spatial transforms are applied slice-by-slice in H,W only, leaving the depth axis unchanged.

The dataloader crops a larger-than-needed patch (`initial_patch_size`, computed in `geometry.py`) so that after rotation the final `patch_size` is fully covered.

| Config key | `nnunet` | `fast` | `extended` |
|---|---|---|---|
| `p_rotation` | 0.2 | 0.15 | 0.3 |
| `p_scaling` | 0.2 | 0.15 | 0.3 |
| `scaling_range` | [0.7, 1.4] | [0.8, 1.2] | [0.6, 1.5] |
| `p_elastic` | 0.0 | 0.0 | 0.4 |

---

### Gaussian Noise

Adds i.i.d. Gaussian noise to the entire image volume. Variance is sampled uniformly from `[v_min, v_max]` and the same noise level is applied to all channels (synchronised).

| Config key | `nnunet` | `fast` | `extended` |
|---|---|---|---|
| `probability` | 0.1 | 0.05 | 0.15 |
| `variance` | [0.0, 0.1] | [0.0, 0.05] | [0.0, 0.15] |

---

### Gaussian Blur

Convolves each channel independently with a Gaussian kernel. Sigma is shared across axes but sampled independently per channel (50% per-channel probability).

Disabled in the `fast` family.

| Config key | `nnunet` | `fast` | `extended` |
|---|---|---|---|
| `probability` | 0.2 | — | 0.3 |
| `sigma` | [0.5, 1.0] | — | [0.3, 1.2] |

---

### Multiplicative Brightness

Multiplies all voxel intensities by a per-channel scalar drawn uniformly from `[lo, hi]`.

| Config key | `nnunet` | `fast` | `extended` |
|---|---|---|---|
| `probability` | 0.15 | 0.1 | 0.2 |
| `range` | [0.75, 1.25] | [0.8, 1.2] | [0.7, 1.3] |

---

### Contrast

Scales the deviation from the spatial mean: `result = (x - mean) * factor + mean`, then clamps to the original per-channel intensity range (`preserve_range=True`). Factor is sampled per channel.

Disabled in the `fast` family.

| Config key | `nnunet` | `fast` | `extended` |
|---|---|---|---|
| `probability` | 0.15 | — | 0.2 |
| `range` | [0.75, 1.25] | — | [0.7, 1.3] |

---

### Simulate Low Resolution

Downsamples the image to a fraction of its size with nearest-neighbour interpolation, then upsamples back with bi/trilinear interpolation. Simulates MRI partial-volume effects. Axes are synchronised (same scale factor per axis); 50% per-channel probability. In dummy-2D mode the depth axis is left at full resolution.

Disabled in the `fast` family.

| Config key | `nnunet` | `fast` | `extended` |
|---|---|---|---|
| `probability` | 0.25 | — | 0.3 |
| `scale` | [0.5, 1.0] | — | [0.4, 1.0] |

---

### Gamma Correction

Applies a power-law transform `x^γ` to normalised intensities, then restores original mean and std (`p_retain_stats=1`). Applied in two independent passes:

- **Normal gamma** — standard `x^γ`, `γ ~ Uniform(range)`.
- **Inverted gamma** — image is negated before and after the power transform, simulating inverted contrast.

| Config key | `nnunet` | `fast` | `extended` |
|---|---|---|---|
| `p_inverted` | 0.1 | 0.05 | 0.15 |
| `p_normal` | 0.3 | 0.15 | 0.4 |
| `range` | [0.7, 1.5] | [0.8, 1.3] | [0.6, 1.6] |

---

### Mirror (Flipping)

Randomly flips the patch along each spatial axis independently (50% probability per axis). Axes are determined by `geometry.py`:
- 3D: axes (0, 1, 2)
- 2D: axes (0, 1)

Enabled in all families when `mirror.enabled: true` (default).

---

### Mask Image

Applied only when `use_mask_for_norm` is set during preprocessing (e.g. CT volumes where the background was zeroed). Re-zeros the masked channels at voxels where `seg == -1` (outside the body). Prevents augmentation from leaking artificial intensities into background regions.

---

### Remove Label -1

Converts all remaining `-1` label values to `0`. This cleans up the preprocessing "outside-body" marker so that the loss function sees only valid class indices.

---

### CutMix (`cutmix.py`)

Batch-level regularisation applied **after** the per-sample transforms and **before** deep-supervision downsampling. For each sample in the batch (with probability `p`), a random partner sample is selected and a random bounding-box region is pasted from the partner into the current sample. Both image and segmentation are pasted identically so labels remain consistent.

The bounding box size is derived from a mixing coefficient `λ ~ Beta(α, α)`: the cut region covers approximately `1 - λ` of the volume (per-axis ratio `√(1−λ)`). With `alpha=1.0` (default), `λ` is uniform over `[0, 1]`.

CutMix is applied in the trainer, not inside the transform pipeline, so that the full-resolution segmentation is mixed before DS downsampling produces the multi-scale targets.

Enabled via `augmentation=cutmix` or by setting `augmentation.cutmix.enabled=true` with any family.

| Config key | default |
|---|---|
| `probability` | 0.5 (per sample) |
| `alpha` | 1.0 |

---

## GPU Augmentation (`gpu_transforms.py`)

`GPUAugmenter` is a drop-in replacement for the CPU batchgeneratorsv2 pipeline. It implements the same transforms entirely in PyTorch on the training GPU:

- Spatial: batched affine via `torch.bmm` + `F.grid_sample`; elastic via Gaussian-smoothed random offsets
- Intensity: vectorised tensor ops (noise, separable blur, brightness, contrast, low-res, gamma)
- Mirror: `torch.flip`

The GPU path eliminates CPU-GPU transfer overhead for the augmented batch and is faster for small models where data loading is the bottleneck. It reads the same YAML config as the CPU path and produces numerically equivalent results.

---

## Geometry (`geometry.py`)

`determine_training_geometry` computes augmentation geometry from the patch size and voxel spacing:

- **`rotation_for_DA`** — rotation range in radians (see Spatial Transform above)
- **`do_dummy_2d_data_aug`** — `True` when the through-plane dimension is much smaller than in-plane (anisotropic 3D); triggers slice-by-slice 2D spatial augmentation
- **`mirror_axes`** — all spatial axes
- **`initial_patch_size`** — expanded patch size passed to the dataloader so that after rotation the final `patch_size` is fully covered (computed via the rotation bounding-box formula from nnUNet)

---

## Transform Order (Training)

```
[Convert3DTo2D]          ← if dummy_2d
SpatialTransform         ← rotation + scaling (+ elastic in extended)
[Convert2DTo3D]          ← if dummy_2d
GaussianNoise
GaussianBlur
MultiplicativeBrightness
Contrast
SimulateLowResolution
GammaTransform (inverted)
GammaTransform (normal)
MirrorTransform
[MaskImageTransform]     ← if use_mask_for_norm
RemoveLabelTransform     ← -1 → 0
--- trainer applies CutMix here (full resolution) ---
--- trainer applies DS downsampling here ---
```
