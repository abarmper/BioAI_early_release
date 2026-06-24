"""Batch-level CutMix augmentation for 3D/2D medical image segmentation.

Adapted from mosDAtoolkit (cli_cutmix.py) for online use within the BioAI
training pipeline. Operates on batch tensors rather than NIfTI files.

CutMix randomly replaces a spatial bounding-box region of one sample with
the corresponding region from another sample in the same batch. Both image
and segmentation tensors are cut-and-pasted identically so labels remain
consistent.

Reference:
    Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers
    with Localizable Features", ICCV 2019.
    mosDAtoolkit: https://github.com/MedICL-VU/mosDAtoolkit
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _random_bbox(
    spatial_shape: Tuple[int, ...],
    lam: float,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Compute a random bounding box whose volume ratio is ~ (1 - lam).

    Parameters
    ----------
    spatial_shape : tuple of int
        Spatial dimensions (e.g. ``(D, H, W)`` or ``(H, W)``).
    lam : float
        Sampled mixing coefficient in [0, 1]. The cut region covers
        approximately ``1 - lam`` of the total volume.

    Returns
    -------
    bb_lower, bb_upper : tuple of int
        Lower (inclusive) and upper (exclusive) bounds for each spatial axis.
    """
    cut_ratio = np.sqrt(1.0 - lam)  # per-axis ratio (cube root for 3D would
    # preserve volume but sqrt matches mosDAtoolkit and works well in practice)

    bb_lower = []
    bb_upper = []
    for size in spatial_shape:
        cut_len = int(size * cut_ratio)
        centre = np.random.randint(0, size)
        lb = int(np.clip(centre - cut_len // 2, 0, size))
        ub = int(np.clip(centre + cut_len // 2, 0, size))
        bb_lower.append(lb)
        bb_upper.append(ub)

    return tuple(bb_lower), tuple(bb_upper)


def apply_cutmix(
    data: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    probability: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply CutMix augmentation across samples in a batch.

    For each sample in the batch (with per-sample ``probability``), a random
    partner sample is selected and a random spatial bounding box is pasted
    from the partner into the current sample. Both ``data`` (image) and
    ``target`` (segmentation) are modified identically.

    Parameters
    ----------
    data : Tensor
        Image batch of shape ``(B, C, *spatial)``.
    target : Tensor
        Full-resolution segmentation batch of shape ``(B, 1, *spatial)``.
        CutMix must run before any deep-supervision downsampling so that the
        low-resolution targets stay consistent with the mixed full-res labels.
    alpha : float
        Parameter of the symmetric Beta distribution used to sample the
        mixing coefficient ``lam``. ``alpha=1.0`` gives a uniform
        distribution over [0, 1].
    probability : float
        Per-sample probability of applying CutMix.

    Returns
    -------
    data, target : Tensor
        The (possibly modified) batch tensors.
    """
    B = data.shape[0]
    if B < 2:
        return data, target

    spatial_shape = data.shape[2:]  # (D, H, W) or (H, W)

    for i in range(B):
        if np.random.rand() > probability:
            continue

        # Pick a random different sample from the batch
        j = np.random.randint(0, B - 1)
        if j >= i:
            j += 1

        # Sample mixing coefficient
        lam = np.random.beta(alpha, alpha)
        bb_lower, bb_upper = _random_bbox(spatial_shape, lam)

        # Build slice for the bounding box (batch-dim + channel-dim + spatial)
        spatial_slice = tuple(
            slice(lo, hi) for lo, hi in zip(bb_lower, bb_upper)
        )
        roi = (i, slice(None)) + spatial_slice  # sample i, all channels
        src = (j, slice(None)) + spatial_slice  # sample j, all channels

        # Paste region from sample j into sample i
        data[roi] = data[src].clone()
        target[roi] = target[src].clone()

    return data, target
