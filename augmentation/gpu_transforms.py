"""GPU-accelerated augmentation pipeline for faster small-model training.

Replaces CPU-bound batchgeneratorsv2 transforms with batched PyTorch GPU ops.
Workers do only fast I/O + crop; the GPU applies all transforms (spatial,
intensity, mirror, DS downsampling) using torch operations.

Transform order matches the CPU pipeline exactly:
  spatial → noise → blur → brightness → contrast → low_res → gamma →
  mirror → mask_image → remove_label → DS_downsample
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _gaussian_kernel_1d(sigma: float, device: torch.device) -> torch.Tensor:
    """Create a normalised 1-D Gaussian kernel on *device*."""
    ksize = max(int(2 * math.ceil(3 * sigma) + 1), 3)
    ksize += 1 - ksize % 2  # ensure odd
    x = torch.arange(ksize, device=device, dtype=torch.float32) - ksize // 2
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def downsample_seg_for_ds(
    target: torch.Tensor,
    ds_scales: List[List[float]],
) -> List[torch.Tensor]:
    """Build a multi-scale deep-supervision target list from a full-res target.

    Mirrors ``DownsampleSegForDSTransform``: the first scale (typically all-ones)
    yields the full-resolution tensor; subsequent scales are produced with
    nearest-neighbour ``F.interpolate``.

    Parameters
    ----------
    target : Tensor
        Full-resolution segmentation tensor ``(B, 1, *spatial)``.
    ds_scales : list of list of float
        One scale-factor tuple per DS level. ``ds_scales[0]`` is full resolution.

    Returns
    -------
    list of Tensor
        Per-scale targets, in the same order as ``ds_scales``.
    """
    result: List[torch.Tensor] = []
    for idx, scale in enumerate(ds_scales):
        if all(s == 1.0 for s in scale):
            # Full resolution: reuse tensor at index 0, clone otherwise so
            # downstream mutations don't leak across scales.
            result.append(target if idx == 0 else target.clone())
        else:
            spatial = target.shape[2:]
            new_shape = [int(round(sz * s)) for sz, s in zip(spatial, scale)]
            ds = F.interpolate(
                target.float(), size=new_shape, mode="nearest-exact",
            )
            result.append(ds.to(target.dtype))
    return result


def _separable_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur for a ``(N, 1, *spatial)`` tensor."""
    device = x.device
    kernel = _gaussian_kernel_1d(sigma, device)
    pad = len(kernel) // 2
    ndim = x.ndim - 2  # number of spatial dims
    for axis in range(ndim):
        shape = [1] * (ndim + 2)
        shape[axis + 2] = len(kernel)
        k = kernel.reshape(shape)
        padding = [0] * ndim
        padding[axis] = pad
        if ndim == 3:
            x = F.conv3d(x, k, padding=padding)
        else:
            x = F.conv2d(x, k, padding=padding)
    return x


# --------------------------------------------------------------------------- #
# GPUAugmenter
# --------------------------------------------------------------------------- #

class GPUAugmenter:
    """All-GPU augmentation pipeline matching the CPU batchgeneratorsv2 transforms.

    Runs under ``@torch.no_grad()`` in float32 (outside autocast) so that
    ``grid_sample`` and reductions keep full accuracy.

    Parameters
    ----------
    aug_cfg : DictConfig
        Augmentation config (nnunet / fast / extended YAML).
    patch_size : list of int
        Final network input size (output of spatial transform).
    input_patch_size : list of int
        Larger crop from dataloader (provides border context for rotation).
    rotation_for_DA : tuple of float
        ``(min_angle, max_angle)`` in radians.
    mirror_axes : tuple of int
        Axes allowed for mirroring (spatial axis indices).
    do_dummy_2d_data_aug : bool
        Rotate only in H,W plane (anisotropic 3D data).
    deep_supervision_scales : list of list of float, or None
        DS downsampling scales (first element is full-resolution ``[1,…]``).
    use_mask_for_norm : list of bool, or None
        Per-channel mask-based normalisation flags.
    device : torch.device
        GPU device.
    """

    def __init__(
        self,
        aug_cfg: DictConfig,
        patch_size: Union[List[int], Tuple[int, ...]],
        input_patch_size: Union[List[int], Tuple[int, ...]],
        rotation_for_DA: Tuple[float, float],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        deep_supervision_scales: Optional[List[List[float]]],
        use_mask_for_norm: Optional[List[bool]],
        device: torch.device,
    ):
        self.device = device
        self.patch_size = list(patch_size)
        self.input_patch_size = list(input_patch_size)
        self.rotation_range = rotation_for_DA
        self.mirror_axes = tuple(mirror_axes)
        self.do_dummy_2d = do_dummy_2d_data_aug
        self.ds_scales = deep_supervision_scales
        self.use_mask_for_norm = use_mask_for_norm
        self.dim = len(patch_size)

        self._parse_config(aug_cfg)
        self._build_identity_grid()

        logger.info(
            "GPUAugmenter: dim=%d patch=%s input_patch=%s dummy_2d=%s",
            self.dim, self.patch_size, self.input_patch_size, self.do_dummy_2d,
        )

    # ------------------------------------------------------------------ #
    # Config parsing
    # ------------------------------------------------------------------ #

    def _parse_config(self, cfg: DictConfig):
        sp = cfg.get("spatial", {})
        self.spatial_enabled = sp.get("enabled", True)
        self.p_rotation = sp.get("p_rotation", 0.2)
        self.p_scaling = sp.get("p_scaling", 0.2)
        self.scaling_range = tuple(sp.get("scaling_range", [0.7, 1.4]))
        self.p_elastic = sp.get("p_elastic", 0.0)
        self.elastic_scale_range = tuple(sp.get("elastic_deform_scale", [0.0, 0.2]))
        self.elastic_mag_range = tuple(sp.get("elastic_deform_magnitude", [0.0, 0.2]))

        n = cfg.get("gaussian_noise", {})
        self.noise_enabled = n.get("enabled", True)
        self.noise_p = n.get("probability", 0.1)
        self.noise_var = tuple(n.get("variance", [0.0, 0.1]))

        bl = cfg.get("gaussian_blur", {})
        self.blur_enabled = bl.get("enabled", True)
        self.blur_p = bl.get("probability", 0.2)
        self.blur_sigma_range = tuple(bl.get("sigma", [0.5, 1.0]))

        br = cfg.get("brightness", {})
        self.brightness_enabled = br.get("enabled", True)
        self.brightness_p = br.get("probability", 0.15)
        self.brightness_range = tuple(br.get("range", [0.75, 1.25]))

        co = cfg.get("contrast", {})
        self.contrast_enabled = co.get("enabled", True)
        self.contrast_p = co.get("probability", 0.15)
        self.contrast_range = tuple(co.get("range", [0.75, 1.25]))

        lr = cfg.get("low_resolution", {})
        self.lowres_enabled = lr.get("enabled", True)
        self.lowres_p = lr.get("probability", 0.25)
        self.lowres_scale_range = tuple(lr.get("scale", [0.5, 1.0]))

        ga = cfg.get("gamma", {})
        self.gamma_enabled = ga.get("enabled", True)
        self.gamma_p_inv = ga.get("p_inverted", 0.1)
        self.gamma_p_norm = ga.get("p_normal", 0.3)
        self.gamma_range = tuple(ga.get("range", [0.7, 1.5]))

        mi = cfg.get("mirror", {})
        self.mirror_enabled = mi.get("enabled", True) and len(self.mirror_axes) > 0

        self.mask_channels: List[int] = []
        if self.use_mask_for_norm is not None:
            self.mask_channels = [i for i, m in enumerate(self.use_mask_for_norm) if m]

    # ------------------------------------------------------------------ #
    # Identity grid (pre-computed once)
    # ------------------------------------------------------------------ #

    def _build_identity_grid(self):
        if self.do_dummy_2d:
            out_sz = self.patch_size[1:]       # H_out, W_out
            in_sz = self.input_patch_size[1:]  # H_in, W_in
            self._sdim = 2
        elif self.dim == 2:
            out_sz = self.patch_size
            in_sz = self.input_patch_size
            self._sdim = 2
        else:
            out_sz = self.patch_size
            in_sz = self.input_patch_size
            self._sdim = 3

        self._out_sz = list(out_sz)
        self._in_sz = list(in_sz)

        # Linspace centred at 0 per spatial axis
        coords = [
            torch.linspace(-(s - 1) / 2, (s - 1) / 2, s, device=self.device)
            for s in out_sz
        ]
        grids = torch.meshgrid(*coords, indexing="ij")
        # (*out_sz, sdim) — last dim holds coordinate per spatial axis
        self._id_grid = torch.stack(grids, dim=-1)

        # Normalisation factor for grid_sample (align_corners=True):
        #   grid_sample_coord = 2 * voxel_coord / (input_size - 1)
        in_m1 = torch.tensor(
            [max(s - 1, 1) for s in in_sz], dtype=torch.float32, device=self.device,
        )
        self._norm_factor = 2.0 / in_m1

    # ------------------------------------------------------------------ #
    # Main entry
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def __call__(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        distance_map: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        Union[torch.Tensor, List[torch.Tensor]],
        Optional[torch.Tensor],
    ]:
        """Apply the full augmentation pipeline on GPU.

        Parameters
        ----------
        data : Tensor, (B, C, *input_patch_size)
        target : Tensor, (B, 1, *input_patch_size)  — int16
        distance_map : Tensor or None, (B, C_fg, *input_patch_size)

        Returns
        -------
        data, target, distance_map
            ``target`` is always a single full-resolution tensor; DS
            downsampling is performed by the trainer after CutMix.
        """
        # 1. Spatial
        data, target, distance_map = self._spatial(data, target, distance_map)

        # 2-7. Intensity (data only)
        data = self._intensity_transforms(data)

        # 8. Mirror
        if self.mirror_enabled:
            data, target, distance_map = self._mirror(data, target, distance_map)

        # 9. Mask image (zero out channels where seg < 0)
        if self.mask_channels:
            outside = target[:, 0:1] < 0  # (B, 1, *spatial)
            for c in self.mask_channels:
                data[:, c:c + 1][outside] = 0

        # 10. Remove label -1 → 0
        target[target == -1] = 0

        # DS downsampling is intentionally NOT applied here — the trainer
        # runs CutMix at full resolution and performs DS downsampling after,
        # so that every scale stays consistent with the mixed target.

        return data, target, distance_map

    # ================================================================== #
    # Spatial transform
    # ================================================================== #

    def _spatial(self, data, target, distance_map):
        if not self.spatial_enabled:
            return self._crop_to_output(data, target, distance_map)
        B = data.shape[0]
        if self.do_dummy_2d:
            return self._spatial_dummy_2d(data, target, distance_map, B)
        if self._sdim == 3:
            M, is_id = self._affine_3d(B)
        else:
            M, is_id = self._affine_2d(B)
        if is_id.all() and self.p_elastic <= 0:
            return self._crop_to_output(data, target, distance_map)
        grid = self._make_grid(M, B)
        return self._grid_sample_all(data, target, distance_map, grid)

    def _spatial_dummy_2d(self, data, target, distance_map, B):
        """3-D volume but only rotate / scale in H,W plane."""
        C, D = data.shape[1], data.shape[2]
        M, is_id = self._affine_2d(B)
        if is_id.all() and self.p_elastic <= 0:
            return self._crop_to_output(data, target, distance_map)

        # Reshape 5-D → 4-D: merge B and D
        def _to4d(t):
            return t.reshape(t.shape[0] * D, t.shape[1], t.shape[3], t.shape[4])

        data_4d = _to4d(data)
        target_4d = _to4d(target)
        dm_4d = _to4d(distance_map) if distance_map is not None else None

        # Replicate affine per depth slice
        M_rep = M.unsqueeze(1).expand(B, D, 2, 2).reshape(B * D, 2, 2)
        grid = self._make_grid(M_rep, B * D)
        data_4d, target_4d, dm_4d = self._grid_sample_all(
            data_4d, target_4d, dm_4d, grid,
        )

        # Reshape back to 5-D
        Ho, Wo = self._out_sz
        tc = target.shape[1]
        data = data_4d.reshape(B, C, D, Ho, Wo)
        target = target_4d.reshape(B, tc, D, Ho, Wo)
        if dm_4d is not None:
            distance_map = dm_4d.reshape(B, distance_map.shape[1], D, Ho, Wo)

        return data, target, distance_map

    # -- grid construction ------------------------------------------------- #

    def _make_grid(self, M, batch_size):
        """Build the sampling grid from affine *M* and the identity grid."""
        sdim = self._sdim
        grid_flat = self._id_grid.reshape(-1, sdim)  # (N, sdim)

        # Batched matmul: (B, N, sdim) = (1, N, sdim) @ (B, sdim, sdim)^T
        grid = torch.bmm(
            grid_flat.unsqueeze(0).expand(batch_size, -1, -1),
            M.transpose(1, 2),
        )
        grid = grid.reshape(batch_size, *self._out_sz, sdim)

        # Optional elastic deformation (extended config only)
        if self.p_elastic > 0:
            do_el = torch.rand(batch_size, device=self.device) < self.p_elastic
            if do_el.any():
                grid = self._add_elastic(grid, do_el)

        # Normalise to grid_sample coords and reverse axis order (D,H,W → W,H,D)
        grid = grid * self._norm_factor
        grid = grid.flip(-1)
        return grid

    def _grid_sample_all(self, data, target, distance_map, grid):
        data = F.grid_sample(
            data, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        )
        target = F.grid_sample(
            target.float(), grid, mode="nearest", padding_mode="zeros", align_corners=True,
        ).to(target.dtype)
        if distance_map is not None:
            distance_map = F.grid_sample(
                distance_map, grid, mode="bilinear", padding_mode="zeros",
                align_corners=True,
            )
        return data, target, distance_map

    # -- centre-crop (identity fast path) ---------------------------------- #

    def _crop_to_output(self, data, target, distance_map):
        if self.do_dummy_2d:
            data = self._ccrop_hw(data)
            target = self._ccrop_hw(target)
            if distance_map is not None:
                distance_map = self._ccrop_hw(distance_map)
        else:
            data = self._ccrop(data, self._out_sz)
            target = self._ccrop(target, self._out_sz)
            if distance_map is not None:
                distance_map = self._ccrop(distance_map, self._out_sz)
        return data, target, distance_map

    @staticmethod
    def _ccrop(t: torch.Tensor, out_sz: list) -> torch.Tensor:
        """Centre-crop all spatial dims of t to *out_sz*."""
        slices: list = [slice(None), slice(None)]  # B, C
        for d, o in enumerate(out_sz):
            s = t.shape[d + 2]
            start = (s - o) // 2
            slices.append(slice(start, start + o))
        return t[tuple(slices)]

    def _ccrop_hw(self, t: torch.Tensor) -> torch.Tensor:
        """Centre-crop only H,W (last two dims) of a 5-D tensor."""
        Ho, Wo = self._out_sz
        h0 = (t.shape[3] - Ho) // 2
        w0 = (t.shape[4] - Wo) // 2
        return t[:, :, :, h0:h0 + Ho, w0:w0 + Wo]

    # ================================================================== #
    # Affine matrix builders
    # ================================================================== #

    def _affine_3d(self, B):
        """Per-sample 3-D affine: ``Rz @ Ry @ Rx @ S`` (synchronised scale).

        Returns ``(M (B,3,3), is_identity (B,))``.
        """
        dev = self.device
        do_rot = torch.rand(B, device=dev) < self.p_rotation
        ang = torch.zeros(B, 3, device=dev)
        nr = do_rot.sum().item()
        if nr:
            ang[do_rot] = torch.empty(nr, 3, device=dev).uniform_(
                *self.rotation_range,
            )

        do_sc = torch.rand(B, device=dev) < self.p_scaling
        sc = torch.ones(B, device=dev)
        ns = do_sc.sum().item()
        if ns:
            sc[do_sc] = torch.empty(ns, device=dev).uniform_(*self.scaling_range)

        is_id = ~do_rot & ~do_sc

        cx, sx = ang[:, 0].cos(), ang[:, 0].sin()
        cy, sy = ang[:, 1].cos(), ang[:, 1].sin()
        cz, sz = ang[:, 2].cos(), ang[:, 2].sin()

        # M = Rz @ Ry @ Rx, then multiply by scale
        M = torch.zeros(B, 3, 3, device=dev)
        M[:, 0, 0] = cz * cy * sc
        M[:, 0, 1] = (cz * sy * sx - sz * cx) * sc
        M[:, 0, 2] = (cz * sy * cx + sz * sx) * sc
        M[:, 1, 0] = sz * cy * sc
        M[:, 1, 1] = (sz * sy * sx + cz * cx) * sc
        M[:, 1, 2] = (sz * sy * cx - cz * sx) * sc
        M[:, 2, 0] = -sy * sc
        M[:, 2, 1] = cy * sx * sc
        M[:, 2, 2] = cy * cx * sc
        return M, is_id

    def _affine_2d(self, B):
        """Per-sample 2-D affine: ``R @ S`` (synchronised scale).

        Returns ``(M (B,2,2), is_identity (B,))``.
        """
        dev = self.device
        do_rot = torch.rand(B, device=dev) < self.p_rotation
        ang = torch.zeros(B, device=dev)
        nr = do_rot.sum().item()
        if nr:
            ang[do_rot] = torch.empty(nr, device=dev).uniform_(*self.rotation_range)

        do_sc = torch.rand(B, device=dev) < self.p_scaling
        sc = torch.ones(B, device=dev)
        ns = do_sc.sum().item()
        if ns:
            sc[do_sc] = torch.empty(ns, device=dev).uniform_(*self.scaling_range)

        is_id = ~do_rot & ~do_sc
        c, s = ang.cos(), ang.sin()

        M = torch.zeros(B, 2, 2, device=dev)
        M[:, 0, 0] = c * sc
        M[:, 0, 1] = -s * sc
        M[:, 1, 0] = s * sc
        M[:, 1, 1] = c * sc
        return M, is_id

    # ================================================================== #
    # Elastic deformation (extended config only, p_elastic > 0)
    # ================================================================== #

    def _add_elastic(self, grid, do_el):
        B, sdim = grid.shape[0], self._sdim
        max_sz = max(self._out_sz)

        el_scale = torch.empty(1, device=self.device).uniform_(
            *self.elastic_scale_range,
        ).item()
        el_mag = torch.empty(1, device=self.device).uniform_(
            *self.elastic_mag_range,
        ).item()
        if el_scale == 0 or el_mag == 0:
            return grid

        sigma = el_scale * max_sz
        magnitude = el_mag * max_sz

        # Random offsets → Gaussian-smooth → scale
        offsets = torch.randn(B, sdim, *self._out_sz, device=self.device)
        offsets = offsets.reshape(B * sdim, 1, *self._out_sz)
        offsets = _separable_blur(offsets, sigma)
        offsets = offsets.reshape(B, sdim, *self._out_sz)

        # Permute to (..., sdim)
        if sdim == 3:
            offsets = offsets.permute(0, 2, 3, 4, 1)
        else:
            offsets = offsets.permute(0, 2, 3, 1)
        offsets = offsets * magnitude

        # Only apply to selected samples
        mask = do_el.float().view(B, *([1] * (sdim + 1)))
        return grid + offsets * mask

    # ================================================================== #
    # Intensity transforms (data only)
    # ================================================================== #

    def _intensity_transforms(self, data):
        if self.noise_enabled:
            data = self._gaussian_noise(data)
        if self.blur_enabled:
            data = self._gaussian_blur(data)
        if self.brightness_enabled:
            data = self._brightness(data)
        if self.contrast_enabled:
            data = self._contrast(data)
        if self.lowres_enabled:
            data = self._low_resolution(data)
        if self.gamma_enabled:
            if self.gamma_p_inv > 0:
                data = self._gamma(data, self.gamma_p_inv, invert=True)
            if self.gamma_p_norm > 0:
                data = self._gamma(data, self.gamma_p_norm, invert=False)
        return data

    # -- Gaussian noise ---------------------------------------------------- #

    def _gaussian_noise(self, data):
        """synchronize_channels=True, p_per_channel=1."""
        B = data.shape[0]
        sel = torch.rand(B, device=self.device) < self.noise_p
        if not sel.any():
            return data
        idx = sel.nonzero(as_tuple=True)[0]
        var = torch.empty(len(idx), device=self.device).uniform_(*self.noise_var)
        std = var.sqrt().view(-1, *([1] * (data.ndim - 1)))
        data = data.clone()
        data[idx] = data[idx] + std * torch.randn_like(data[idx])
        return data

    # -- Gaussian blur ----------------------------------------------------- #

    def _gaussian_blur(self, data):
        """synchronize_channels=False, p_per_channel=0.5."""
        B, C = data.shape[:2]
        sel = torch.rand(B, device=self.device) < self.blur_p
        if not sel.any():
            return data
        idx = sel.nonzero(as_tuple=True)[0]
        K = len(idx)

        # Per-channel probability
        ch_mask = torch.rand(K, C, device=self.device) < 0.5
        if not ch_mask.any():
            return data

        sigma = torch.empty(1, device=self.device).uniform_(
            *self.blur_sigma_range,
        ).item()

        subset = data[idx]  # (K, C, *spatial)
        spatial = subset.shape[2:]
        blurred = _separable_blur(
            subset.reshape(K * C, 1, *spatial), sigma,
        ).reshape(K, C, *spatial)

        ch_mask = ch_mask.view(K, C, *([1] * len(spatial)))
        data = data.clone()
        data[idx] = torch.where(ch_mask, blurred, subset)
        return data

    # -- Brightness -------------------------------------------------------- #

    def _brightness(self, data):
        """synchronize_channels=False, p_per_channel=1."""
        B, C = data.shape[:2]
        sel = torch.rand(B, device=self.device) < self.brightness_p
        if not sel.any():
            return data
        idx = sel.nonzero(as_tuple=True)[0]
        factor = torch.empty(len(idx), C, device=self.device).uniform_(
            *self.brightness_range,
        ).view(len(idx), C, *([1] * (data.ndim - 2)))
        data = data.clone()
        data[idx] = data[idx] * factor
        return data

    # -- Contrast ---------------------------------------------------------- #

    def _contrast(self, data):
        """synchronize_channels=False, p_per_channel=1, preserve_range=True."""
        B, C = data.shape[:2]
        sel = torch.rand(B, device=self.device) < self.contrast_p
        if not sel.any():
            return data
        idx = sel.nonzero(as_tuple=True)[0]
        K = len(idx)
        subset = data[idx]
        spatial_dims = list(range(2, data.ndim))

        mean = subset.mean(dim=spatial_dims, keepdim=True)
        factor = torch.empty(K, C, device=self.device).uniform_(
            *self.contrast_range,
        ).view(K, C, *([1] * (data.ndim - 2)))

        result = (subset - mean) * factor + mean

        # preserve_range: clamp to original min/max per channel
        lo = subset.amin(dim=spatial_dims, keepdim=True)
        hi = subset.amax(dim=spatial_dims, keepdim=True)
        result = torch.clamp(result, lo, hi)

        data = data.clone()
        data[idx] = result
        return data

    # -- Simulate low resolution ------------------------------------------- #

    def _low_resolution(self, data):
        """synchronize_channels=False, synchronize_axes=True, p_per_channel=0.5."""
        B, C = data.shape[:2]
        sel = torch.rand(B, device=self.device) < self.lowres_p
        if not sel.any():
            return data
        idx = sel.nonzero(as_tuple=True)[0]
        K = len(idx)

        ch_mask = torch.rand(K, C, device=self.device) < 0.5
        if not ch_mask.any():
            return data

        scale = torch.empty(1, device=self.device).uniform_(
            *self.lowres_scale_range,
        ).item()
        if scale >= 1.0:
            return data

        spatial = list(data.shape[2:])
        if self.do_dummy_2d:
            # Don't downsample depth (axis 0 of spatial)
            down_sz = [spatial[0]] + [max(1, int(round(s * scale))) for s in spatial[1:]]
        else:
            down_sz = [max(1, int(round(s * scale))) for s in spatial]

        mode_up = "trilinear" if len(spatial) == 3 else "bilinear"

        subset = data[idx]
        down = F.interpolate(subset, size=down_sz, mode="nearest")
        up = F.interpolate(down, size=spatial, mode=mode_up, align_corners=False)

        ch_mask = ch_mask.view(K, C, *([1] * len(spatial)))
        data = data.clone()
        data[idx] = torch.where(ch_mask, up, subset)
        return data

    # -- Gamma correction -------------------------------------------------- #

    def _gamma(self, data, p_apply, invert):
        """synchronize_channels=False, p_per_channel=1, p_retain_stats=1."""
        B, C = data.shape[:2]
        sel = torch.rand(B, device=self.device) < p_apply
        if not sel.any():
            return data
        idx = sel.nonzero(as_tuple=True)[0]
        K = len(idx)
        subset = data[idx].clone()
        spatial_dims = list(range(2, data.ndim))

        if invert:
            subset = -subset

        orig_mean = subset.mean(dim=spatial_dims, keepdim=True)
        orig_std = subset.std(dim=spatial_dims, keepdim=True)

        lo = subset.amin(dim=spatial_dims, keepdim=True)
        hi = subset.amax(dim=spatial_dims, keepdim=True)
        rng = (hi - lo).clamp(min=1e-7)
        norm = ((subset - lo) / rng).clamp(0, 1)

        gamma = torch.empty(K, C, device=self.device).uniform_(*self.gamma_range)
        gamma = gamma.view(K, C, *([1] * (data.ndim - 2)))
        subset = norm.pow(gamma) * rng + lo

        # Retain stats
        new_mean = subset.mean(dim=spatial_dims, keepdim=True)
        new_std = subset.std(dim=spatial_dims, keepdim=True).clamp(min=1e-7)
        subset = (subset - new_mean) / new_std * orig_std + orig_mean

        if invert:
            subset = -subset

        data = data.clone()
        data[idx] = subset
        return data

    # ================================================================== #
    # Mirror
    # ================================================================== #

    def _mirror(self, data, target, distance_map):
        """Per-axis independent 50 % flip (equivalent to MirrorTransform)."""
        B = data.shape[0]
        for ax in self.mirror_axes:
            do_flip = torch.rand(B, device=self.device) < 0.5
            if not do_flip.any():
                continue
            dim = ax + 2  # spatial axis → tensor dim (+2 for B, C prefix)
            flip_idx = do_flip.nonzero(as_tuple=True)[0]
            data[flip_idx] = torch.flip(data[flip_idx], dims=[dim])
            target[flip_idx] = torch.flip(target[flip_idx], dims=[dim])
            if distance_map is not None:
                distance_map[flip_idx] = torch.flip(
                    distance_map[flip_idx], dims=[dim],
                )
        return data, target, distance_map

    # ================================================================== #
    # DS downsampling
    # ================================================================== #

    def _ds_downsample(self, target):
        return downsample_seg_for_ds(target, self.ds_scales)
