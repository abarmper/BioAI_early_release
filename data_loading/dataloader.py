"""BioAI DataLoader with nnUNet-style patch sampling and foreground oversampling.

Ported from ``nnunetv2/training/dataloading/data_loader.py``.
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from data_loading.dataset import BioAIDataset

logger = logging.getLogger(__name__)


def _crop_and_pad_nd(data: np.ndarray, bbox: list, pad_value=0):
    """Crop (and pad if needed) an ND array to the given bounding box.

    bbox : list of [lb, ub] pairs, one per spatial dim (excluding channel).
    """
    ndim = data.ndim - 1  # first dim is channels
    slicers_data = [slice(None)]  # channel dim
    pad_before = []
    pad_after = []

    for d in range(ndim):
        lb, ub = bbox[d]
        dim_size = data.shape[d + 1]

        # Clamp to valid range and compute padding
        actual_lb = max(0, lb)
        actual_ub = min(dim_size, ub)

        pad_b = max(0, -lb)
        pad_a = max(0, ub - dim_size)

        slicers_data.append(slice(actual_lb, actual_ub))
        pad_before.append(pad_b)
        pad_after.append(pad_a)

    cropped = data[tuple(slicers_data)]

    if any(p > 0 for p in pad_before) or any(p > 0 for p in pad_after):
        pad_widths = [(0, 0)]  # channel dim
        for pb, pa in zip(pad_before, pad_after):
            pad_widths.append((pb, pa))
        cropped = np.pad(cropped, pad_widths, mode="constant", constant_values=pad_value)

    return cropped


class BioAIDataLoader:
    """nnUNet-style DataLoader with patch sampling and foreground oversampling.

    This loader does not iterate over full images. Instead, it repeatedly samples
    fixed-size patches from randomly chosen dataset cases and assembles them into
    training batches. That is important for medical segmentation, where volumes
    can be too large to process end-to-end and where naive random sampling would
    often produce mostly background patches.

    The high-level sampling scheme is:

    1. Randomly select ``batch_size`` case identifiers from the dataset.
    2. For each case, decide whether the sample should be foreground-focused.
    3. Compute a bounding box for a patch of size ``patch_size``.
    4. Crop the corresponding region from image and segmentation and pad if the
       patch extends outside the case boundaries.
    5. Optionally apply transforms.
    6. Store the result in a preallocated batch tensor.

    Foreground oversampling follows the nnUNet strategy: a configurable fraction
    of samples is forced to contain foreground voxels whenever class location
    metadata is available. If multiple foreground classes are present, one class
    can be chosen uniformly or according to user-provided per-class weights.

    A key distinction in this loader is between ``patch_size`` and
    ``final_patch_size``. ``patch_size`` is the size of the region initially
    sampled from the case. It can be larger than the final network input to leave
    room for spatial augmentations such as rotations or other transforms that may
    otherwise remove valid context near the borders. ``final_patch_size`` is the
    size expected by the model and by the preallocated output tensors. In a
    typical nnUNet-style pipeline, transforms operate on the larger sampled patch
    and then crop or otherwise reduce it to the final size seen by the network.

    The loader also supports 2D training by internally representing 2D patches as
    pseudo-3D patches with depth 1 during sampling, then removing that dummy axis
    before returning the batch to the model.

    Implements infinite iteration: call ``next(loader)`` to get a batch.

    Parameters
    ----------
    dataset : BioAIDataset
        Dataset object that provides case identifiers and a ``load_case`` method.
        Each loaded case is expected to return image data, optional segmentation,
        and a ``properties`` dictionary that may contain ``class_locations`` for
        foreground-aware patch sampling.
    batch_size : int
        Number of independently sampled patches returned in each training batch.
        Each batch element may come from a different randomly chosen case.
    patch_size : tuple of int
        Spatial size of the patch extracted from a case before transforms are
        applied. This is the sampling window used by ``_get_bbox`` and
        ``_crop_and_pad_nd``. It may be chosen larger than the final model input
        size when augmentations need extra spatial context near the borders or
        when the transform pipeline performs a later crop back to the network
        input size.
    final_patch_size : tuple of int
        Spatial size expected by the downstream network and used to preallocate
        batch tensors. In 2D mode, this is the true 2D size; the loader
        temporarily expands it to pseudo-3D form internally and removes the dummy
        depth axis again before returning the batch. This is the size that should
        remain after transforms finish processing the sampled patch.
    oversample_foreground_percent : float
        Target fraction of batch elements that should be foreground-focused. For
        those samples, the loader tries to place the crop around a stored
        foreground voxel rather than sampling a purely random patch.
    probabilistic_oversampling : bool
        Controls how foreground oversampling is applied. If True, each sample is
        independently marked for foreground sampling with probability
        ``oversample_foreground_percent``. If False, the decision is deterministic
        within a batch and the last X% of samples are forced to be foreground.
    per_class_oversample : dict or None
        Optional class-specific sampling weights used when multiple foreground
        classes are available in a case. The keys are class labels and the values
        are relative probabilities for choosing which class to center a
        foreground-focused patch on. If ``None``, each eligible foreground
        *class* is equally likely to be chosen (equal probability per class,
        not per foreground voxel — a class with many voxels is not favoured).
        The ``"auto"`` config string is resolved upstream by the trainer; this
        loader only ever sees ``None`` or a ``Dict[int, float]``.
    transforms
        Optional ``batchgeneratorsv2`` transform pipeline applied after
        cropping/padding. It receives the sampled patch and segmentation target
        and can return either a single segmentation tensor or a list of targets,
        for example when deep supervision is enabled.
    """

    def __init__(
        self,
        dataset: BioAIDataset,
        batch_size: int,
        patch_size: Union[List[int], Tuple[int, ...]],
        final_patch_size: Union[List[int], Tuple[int, ...]],
        oversample_foreground_percent: float = 0.33,
        probabilistic_oversampling: bool = False,
        per_class_oversample: Optional[Dict[int, float]] = None,
        transforms=None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.transforms = transforms
        self.oversample_foreground_percent = oversample_foreground_percent
        self.probabilistic_oversampling = probabilistic_oversampling
        self.per_class_oversample = per_class_oversample

        # Handle 2D as pseudo-3D
        if len(patch_size) == 2:
            final_patch_size = (1, *final_patch_size)
            patch_size = (1, *patch_size)
            self.patch_size_was_2d = True
        else:
            self.patch_size_was_2d = False

        self.patch_size = list(patch_size)
        self.final_patch_size = list(final_patch_size)

        # How much we need to pad to cover borders
        self.need_to_pad = [
            int(p - f) for p, f in zip(self.patch_size, self.final_patch_size)
        ]

        # Pre-determine data shape from first case
        data, seg, _ = self.dataset.load_case(self.dataset.identifiers[0])
        spatial = self.final_patch_size[1:] if self.patch_size_was_2d else self.final_patch_size
        self.data_shape = (batch_size, data.shape[0], *spatial)
        seg_ch = seg.shape[0] if seg is not None else 1
        self.seg_shape = (batch_size, seg_ch, *spatial)

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()

    def set_thread_id(self, thread_id: int):
        """Called by NonDetMultiThreadedAugmenter per worker process."""
        self.thread_id = thread_id

    def _get_do_oversample(self, sample_idx: int) -> bool:
        """Whether sample_idx should be forced to contain foreground."""
        if self.probabilistic_oversampling:
            return np.random.uniform() < self.oversample_foreground_percent
        return not sample_idx < round(
            self.batch_size * (1 - self.oversample_foreground_percent) # so if oversample_foreground_percent is 0.33 then index 2 and above are oversampled
        )

    def _select_foreground_class(self, class_locations: dict) -> Optional[int]:
        """Select a foreground class for oversampling.

        If per_class_oversample is configured, classes are selected
        proportionally to their weights.  Otherwise uniform random.
        """
        eligible = [k for k in class_locations if k != -1 and len(class_locations[k]) > 0]
        if not eligible:
            return None

        if self.per_class_oversample is not None:
            weights = np.array([self.per_class_oversample.get(k, 1.0) for k in eligible])
            weights /= weights.sum()
            return eligible[np.random.choice(len(eligible), p=weights)]

        return eligible[np.random.choice(len(eligible))]

    def _get_bbox(
        self,
        data_shape: np.ndarray,
        force_fg: bool,
        class_locations: Optional[dict],
    ):
        """Compute a sampling bounding box for one training patch.

        This method returns the lower and upper coordinates of a patch with
        spatial size ``self.patch_size`` for a case of spatial shape
        ``data_shape``. The output follows the nnUNet patch-sampling logic and is
        designed to work together with ``_crop_and_pad_nd``:

        - The bounding box is allowed to extend outside the image boundaries.
        - Out-of-bounds regions are handled later by zero/ignore-label padding.
        - When requested, the patch is biased to contain foreground voxels.

        The method first adjusts ``self.need_to_pad`` for the current case. This
        matters when an image is smaller than the desired patch size in one or
        more dimensions. In that case, the valid sampling interval must be
        widened so that a full patch can still be requested and reconstructed via
        padding.

        After that, the function computes the valid range of lower bounds for the
        patch in each dimension:

        - ``lbs`` is the smallest allowed patch start. It can be negative, which
          means the crop begins before the image and must later be padded on the
          left side.
        - ``ubs`` is the largest allowed patch start. It can be positive even if
          the corresponding upper bound extends beyond the image, because the
          missing region will later be padded on the right side.

        Sampling then proceeds in one of two modes:

        - Random sampling: if ``force_fg`` is False or no class-location metadata
          is available, the lower bound is sampled uniformly between ``lbs`` and
          ``ubs`` in each dimension.
        - Foreground sampling: if ``force_fg`` is True and ``class_locations`` is
          available, one foreground class is selected, then one voxel belonging
          to that class is chosen, and the patch is positioned so that this voxel
          lies near the center of the sampled patch.

        Parameters
        ----------
        data_shape : np.ndarray
            Spatial shape of the current case, excluding the channel dimension.
        force_fg : bool
            Whether this sample should be biased toward foreground. This flag is
            typically determined by ``_get_do_oversample``.
        class_locations : dict or None
            Mapping from class label to an array/list of voxel coordinates for
            that class. If available, it is used to place foreground-focused
            patches around a selected voxel.

        Returns
        -------
        tuple[list[int], list[int]]
            Two lists containing the lower and upper bounds of the patch for each
            spatial dimension. The upper bounds are exclusive, so each dimension
            spans exactly ``self.patch_size[d]`` voxels.
        """
        need_to_pad = self.need_to_pad.copy()
        dim = len(data_shape)

        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.patch_size[d]: # If patch size is larger than data + safety padding for rotation transformations, we need to pad
                need_to_pad[d] = self.patch_size[d] - data_shape[d]

        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [
            data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.patch_size[i]
            for i in range(dim)
        ]

        if not force_fg or class_locations is None:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
        else:
            selected_class = self._select_foreground_class(class_locations)
            if selected_class is not None:
                voxels = class_locations[selected_class]
                selected_voxel = voxels[np.random.choice(len(voxels))]
                bbox_lbs = [
                    max(lbs[i], int(selected_voxel[i]) - self.patch_size[i] // 2)
                    for i in range(dim)
                ]
            else:
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + self.patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs

    def generate_train_batch(self) -> dict:
        """Generate one training batch of sampled image patches.

        This is the main entry point of the loader. Each call constructs a new
        batch by randomly drawing dataset cases, sampling one patch per selected
        case, and converting the results into tensors ready for the training
        step.

        The method follows this sequence:

        1. Randomly select ``batch_size`` case identifiers, with replacement.
        2. Preallocate output tensors using the shapes inferred at
           initialization time.
        3. For each selected case:
           - decide whether the sample should be foreground-focused,
           - load the image, segmentation, distance map, and metadata from the dataset,
           - compute a patch bounding box with ``_get_bbox``,
           - crop and pad the image patch, segmentation patch, and distance map
             patch with ``_crop_and_pad_nd``,
           - remove the dummy depth axis if the loader is operating in 2D
             pseudo-3D mode,
           - apply optional transforms to image, distance map (optional) and segmentation tensors,
           - place the result into the batch tensors.
        4. Return the assembled batch together with the sampled case keys.

        A few implementation details are important:

        - Sampling is done with replacement, so the same case may appear more
          than once in a batch.
        - Original dataset cases do not need to share the same spatial shape.
          The loader resolves that mismatch by sampling a fixed-size patch from
          each case and padding when necessary, so the returned batch still has a
          uniform tensor shape.
        - Image patches are padded with value ``0`` when a sampled region extends
          outside the image boundaries.
        - Segmentation patches are padded with value ``-1``, which is typically
          used as an ignore label for invalid/outside regions. Later transforms
          make this value to 0.
        - Distance maps are padded with zero value so
          that out-of-bounds regions do not contribute to boundary loss. Distance
          maps are also transformed as regression targets.
        - ``threadpool_limits(limits=1)`` is used to keep low-level numerical
          libraries from oversubscribing CPU threads while samples are being
          assembled.
        - If the transform pipeline returns a list of segmentation tensors, the
          loader interprets that as deep-supervision targets and builds a
          separate batch tensor for each scale.

        Returns
        -------
        dict
            Dictionary with the following entries:

            - ``"data"``: image batch tensor of shape ``(B, C, ...)``.
            - ``"target"``: segmentation batch tensor, or a list of tensors when
              deep supervision is enabled, or ``None`` if no segmentation is
              available.
            - ``"distance_map"``: distance-map batch tensor of shape
              ``(B, C_fg, ...)``, or ``None`` if distance maps are not
              available for the sampled cases.
            - ``"keys"``: list of sampled dataset identifiers used to build the
              batch.
        """
        # Random case selection
        selected_keys = [
            self.dataset.identifiers[np.random.randint(len(self.dataset.identifiers))]
            for _ in range(self.batch_size)
        ]

        data_all = torch.empty(self.data_shape, dtype=torch.float32)
        seg_all = None
        dist_all = None

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, case_id in enumerate(selected_keys):
                    force_fg = self._get_do_oversample(j)

                    data, seg, properties = self.dataset.load_case(case_id)
                    shape = np.asarray(data.shape[1:], dtype=np.int64)  # Because _get_bbox needs ndarray.

                    class_locs = properties.get("class_locations", None)
                    bbox_lbs, bbox_ubs = self._get_bbox(shape , force_fg, class_locs)
                    bbox = [[lb, ub] for lb, ub in zip(bbox_lbs, bbox_ubs)]

                    data_cropped = torch.from_numpy(
                        _crop_and_pad_nd(data, bbox, 0).copy()
                    ).float()
                    seg_cropped = torch.from_numpy(
                        _crop_and_pad_nd(seg, bbox, -1).copy()
                    ).to(torch.int16) if seg is not None else None

                    # Crop distance map with same bbox (pad with max value —
                    # outside-image regions are background; padding with 0
                    # would misrepresent them as boundary voxels)
                    dist_map = properties.get("distance_map", None)
                    if dist_map is not None:
                        dist_cropped = torch.from_numpy(
                            _crop_and_pad_nd(dist_map, bbox, 0).copy() # Zero values do not contribute to boundary loss.
                        ).float()
                    else:
                        dist_cropped = None

                    # Handle 2D pseudo-3D
                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        if seg_cropped is not None:
                            seg_cropped = seg_cropped[:, 0]
                        if dist_cropped is not None:
                            dist_cropped = dist_cropped[:, 0]

                    # Apply transforms (distance maps augmented via regression_target:
                    # bilinear interpolation for spatial transforms, skipped by intensity transforms)
                    if self.transforms is not None and seg_cropped is not None:
                        transform_dict = {"image": data_cropped, "segmentation": seg_cropped}
                        if dist_cropped is not None:
                            transform_dict["regression_target"] = dist_cropped
                        transformed = self.transforms(**transform_dict)
                        data_sample = transformed["image"]
                        seg_sample = transformed["segmentation"]
                        if dist_cropped is not None:
                            dist_cropped = transformed["regression_target"]
                    else:
                        data_sample = data_cropped
                        seg_sample = seg_cropped

                    data_all[j] = data_sample

                    if seg_sample is not None:
                        if isinstance(seg_sample, list):
                            # Deep supervision: list of tensors at different scales
                            if seg_all is None:
                                seg_all = [
                                    torch.empty((self.batch_size, *s.shape), dtype=s.dtype)
                                    for s in seg_sample
                                ]
                            for s_idx, s in enumerate(seg_sample):
                                seg_all[s_idx][j] = s
                        else:
                            if seg_all is None:
                                seg_all = torch.empty(
                                    (self.batch_size, *seg_sample.shape),
                                    dtype=seg_sample.dtype,
                                )
                            seg_all[j] = seg_sample

                    # Accumulate distance maps
                    if dist_cropped is not None:
                        if dist_all is None:
                            dist_all = torch.empty(
                                (self.batch_size, *dist_cropped.shape),
                                dtype=torch.float32,
                            )
                        dist_all[j] = dist_cropped

        return {
            "data": data_all,
            "target": seg_all,
            "distance_map": dist_all,
            "keys": selected_keys,
        }
