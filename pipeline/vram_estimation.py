"""VRAM estimation and batch/patch size recommendations."""
from __future__ import annotations

import logging
from typing import List, Optional, Protocol, Sequence, cast

import torch

logger = logging.getLogger(__name__)


class _FeatureMapSizeEstimator(Protocol):
    def compute_conv_feature_map_size(self, input_size: Sequence[int]) -> int:
        ...


def _next_smaller_patch(
    patch_size: List[int],
    div_by: List[int],
) -> Optional[List[int]]:
    """Return the next smaller patch divisible by div_by, preserving aspect ratio.

    Expresses each dimension as a multiple of div_by, reduces the smallest
    multiplier by 1, scales all others by the same ratio, and rounds back to the
    nearest valid multiple. Returns None if the patch is already at the minimum
    (every multiplier equals 1).
    """
    k = [p // d for p, d in zip(patch_size, div_by)]
    k_min = min(k)
    if k_min <= 1:
        return None
    scale = (k_min - 1) / k_min
    new_k = [max(1, round(ki * scale)) for ki in k]
    return [nk * d for nk, d in zip(new_k, div_by)]


def check_vram_and_adjust(
    model: torch.nn.Module,
    patch_size: list,
    batch_size: int,
    num_input_channels: int,
    device: torch.device,
    min_batch_size: int = 2,
    div_by: Optional[List[int]] = None,
) -> tuple:
    """Try a dummy forward+backward to verify the config fits in VRAM.

    Returns
    -------
    (fits: bool, recommended_batch_size: int, recommended_patch_size: List[int])
    """
    if device.type != "cuda":
        logger.info("VRAM check skipped (device=%s)", device)
        return True, batch_size, list(patch_size)

    # Analytical pre-check (nnUNet architectures only)
    if hasattr(model, "compute_conv_feature_map_size"):
        estimate_model = cast(_FeatureMapSizeEstimator, model)
        estimate_voxels = estimate_model.compute_conv_feature_map_size(patch_size)
        # Rough estimate: voxels * 4 bytes (float32) * 2 (activations + gradients) * batch_size
        estimated_gb = estimate_voxels * 4 * 2 * batch_size / (1024 ** 3)
        logger.info(
            "Analytical VRAM estimate: %.1f GB for bs=%d, patch=%s",
            estimated_gb, batch_size, patch_size,
        )

    model = model.to(device)
    original_training = model.training
    model.train()

    current_bs = batch_size
    while current_bs >= min_batch_size:
        try:
            torch.cuda.empty_cache()
            dummy_input = torch.randn(
                current_bs, num_input_channels, *patch_size,
                device=device, dtype=torch.float32,
            )
            with torch.autocast("cuda", enabled=True):
                output = model(dummy_input)
                loss: torch.Tensor
                if isinstance(output, (list, tuple)):
                    loss = torch.stack([o.sum() for o in output]).sum()
                else:
                    loss = output.sum()
                loss.backward()

            del dummy_input, output, loss
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)

            mem_used = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            mem_total = torch.cuda.get_device_properties(device).total_mem / (1024 ** 3)
            logger.info(
                "VRAM check passed: batch_size=%d, peak=%.1f GB / %.1f GB",
                current_bs, mem_used, mem_total,
            )
            model.train(original_training)
            return True, current_bs, list(patch_size)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                model.zero_grad(set_to_none=True)
                if current_bs > min_batch_size:
                    new_bs = max(min_batch_size, current_bs // 2)
                    logger.warning(
                        "Batch size %d does not fit in VRAM. Trying %d.",
                        current_bs, new_bs,
                    )
                    current_bs = new_bs
                else:
                    if div_by is not None:
                        suggested = _next_smaller_patch(patch_size, div_by)
                        if suggested is not None:
                            logger.error(
                                "Even batch_size=%d does not fit in VRAM with patch_size=%s. "
                                "Suggested next smaller patch: %s (re-run planning with a "
                                "smaller initial_patch_ref or adjust manually in plans).",
                                current_bs, patch_size, suggested,
                            )
                            model.train(original_training)
                            return False, current_bs, suggested
                    logger.error(
                        "Even batch_size=%d does not fit in VRAM with "
                        "patch_size=%s. Consider reducing total number of voxels and re-running planning phase.",
                        current_bs, patch_size,
                    )
                    model.train(original_training)
                    return False, 0, list(patch_size)
            else:
                raise

    model.train(original_training)
    return False, 0, list(patch_size)
