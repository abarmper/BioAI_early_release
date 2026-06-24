"""Numpy-based image normalization schemes matching nnUNet v2.

Each normalizer operates on a single 2-D or 3-D numpy array (one channel)
and modifies it **in-place** for efficiency.

Reference
---------
``nnUNet/nnunetv2/preprocessing/normalization/default_normalization_schemes.py``
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class ImageNormalization(ABC):
    """Base class for all normalization schemes.

    Parameters
    ----------
    use_mask_for_norm : bool
        Whether to restrict statistics to foreground voxels (``seg >= 0``).
    intensity_properties : dict or None
        Dataset-level intensity statistics for this channel
        (from ``dataset_fingerprint.json``).
    """

    def __init__(
        self,
        use_mask_for_norm: bool = False,
        intensity_properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.use_mask_for_norm = use_mask_for_norm
        self.intensity_properties = intensity_properties or {}

    @abstractmethod
    def run(
        self, image: np.ndarray, seg: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Normalize *image* in-place and return ``(image, params)``.

        Parameters
        ----------
        image : np.ndarray
            Single-channel volume (2-D or 3-D), dtype float32.
        seg : np.ndarray or None
            Corresponding segmentation.  Voxels with ``seg == -1`` are
            considered outside-body (nnUNet convention after crop).

        Returns
        -------
        image : np.ndarray
            The normalized image.
        params : dict
            Normalization parameters needed to reverse the transform.
        """
        ...


class CTNormalization(ImageNormalization):
    """nnUNet CT normalization: clip to dataset percentiles then z-score.

    1. Clip to ``[percentile_00_5, percentile_99_5]``.
    2. ``(x - mean) / std`` using dataset foreground statistics.

    ``use_mask_for_norm`` is ignored — the entire image is normalized.
    """

    def run(
        self, image: np.ndarray, seg: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        props = self.intensity_properties
        mean = props.get("mean", 0.0)
        std = max(props.get("std", 1.0), 1e-8)
        lower = props.get("percentile_00_5", image.min())
        upper = props.get("percentile_99_5", image.max())

        np.clip(image, lower, upper, out=image)
        image -= mean
        image /= std
        params = {
            "scheme": "CT",
            "mean": float(mean),
            "std": float(std),
            "lower_clip": float(lower),
            "upper_clip": float(upper),
        }
        return image, params


class ZScoreNormalization(ImageNormalization):
    """nnUNet MRI z-score normalization (per-image statistics).

    If ``use_mask_for_norm`` is ``True``, mean and std are computed only
    from voxels where ``seg >= 0`` (i.e. inside the body after nnUNet crop).
    Outside-body voxels (``seg == -1``) are left at zero after normalization.
    Note that outside-body voxels where computed based on the original image (not the 
    segmentation labels) so there is no leakage of label information into the normalization step.
    If seg is None, the entire image is used for statistics and no masking is applied.

    If ``use_mask_for_norm`` is ``False``, the entire image is z-scored.
    """

    def run(
        self, image: np.ndarray, seg: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        use_mask = self.use_mask_for_norm and seg is not None
        if use_mask and seg is not None:
            mask = seg >= 0 # True for inside-body voxels, False for outside-body (nnUNet convention after crop based on image != 0)
            if mask.any():
                mean = float(image[mask].mean())
                std = float(max(image[mask].std(), 1e-8))
                image -= mean
                image /= std
                image[~mask] = 0.0  # Set outside-body voxels to zero after normalization.
            else:
                # Mask is entirely False — image stays unchanged
                mean = 0.0
                std = 1.0
        else:
            mean = float(image.mean())
            std = float(max(image.std(), 1e-8))
            image -= mean
            image /= std
        params = {
            "scheme": "ZScore",
            "mean": mean,
            "std": std,
            "use_mask": use_mask,
        }
        return image, params


class NoNormalization(ImageNormalization):
    """Identity pass — no normalization applied."""

    def run(
        self, image: np.ndarray, seg: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return image, {"scheme": "NoNormalization"}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_NORMALIZERS = {
    "CTNormalization": CTNormalization,
    "CT": CTNormalization,
    "ZScoreNormalization": ZScoreNormalization,
    "ZScore": ZScoreNormalization,
    "NoNormalization": NoNormalization,
    "None": NoNormalization,
}


def get_normalizer(
    scheme_name: str,
    use_mask: bool = False,
    intensity_properties: Optional[Dict[str, Any]] = None,
) -> ImageNormalization:
    """Instantiate a normalizer by name.

    Parameters
    ----------
    scheme_name : str
        One of ``"CTNormalization"`` / ``"CT"``, ``"ZScoreNormalization"``
        / ``"ZScore"``, ``"NoNormalization"`` / ``"None"``.
    use_mask : bool
        Passed to the normalizer constructor.
    intensity_properties : dict or None
        Dataset-level intensity stats for this channel.

    Returns
    -------
    ImageNormalization
    """
    cls = _NORMALIZERS.get(scheme_name)
    if cls is None:
        raise ValueError(
            f"Unknown normalization scheme '{scheme_name}'. "
            f"Available: {list(_NORMALIZERS.keys())}"
        )
    return cls(use_mask_for_norm=use_mask, intensity_properties=intensity_properties)
