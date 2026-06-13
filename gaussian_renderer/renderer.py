"""Shared rendering utilities for 3D Gaussian Splatting.

Pure functions — no classes, no state. Used by both gaussian_model and
dino_gaussian_model to avoid code duplication without inheritance.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch

from utils.graphics_utils import BilateralGrid, slice


# ═══════════════════════════════════════════════════════════════════════════════
# image / rendering helpers
# ═══════════════════════════════════════════════════════════════════════════════


def get_viewmat(camera_to_world: torch.Tensor) -> torch.Tensor:
    """Convert c2w (OpenCV convention) to gsplat world-to-camera matrix.

    gsplat expects [R|T] where R maps from world→camera in OpenGL convention
    (x-right, y-up, z-backward). Nerfstudio uses OpenCV (x-right, y-down,
    z-forward), so we flip Y and Z axes.
    """
    R = camera_to_world[:, :3, :3]
    T = camera_to_world[:, :3, 3:4]
    # OpenCV → OpenGL: flip Y and Z
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


def resize_image(image: torch.Tensor, d: int) -> torch.Tensor:
    """Downscale image using area interpolation (same as OpenCV INTER_AREA).

    Args:
        image: shape [H, W, C]
        d: downscale factor (must be 2, 4, 8, …)
    Returns:
        downscaled image in shape [H//d, W//d, C]
    """
    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return torch.nn.functional.conv2d(
        image.permute(2, 0, 1)[:, None, ...], weight, stride=d
    ).squeeze(1).permute(1, 2, 0)


def get_downscale_factor(num_downscales: int, step: int, resolution_schedule: int, training: bool) -> int:
    """Compute training resolution downscale factor."""
    if not training:
        return 1
    return 2 ** max((num_downscales - step // resolution_schedule), 0)


def downscale_if_required(
    image: torch.Tensor,
    num_downscales: int,
    step: int,
    resolution_schedule: int,
    training: bool,
) -> torch.Tensor:
    """Apply downscale to image if the current training schedule requires it."""
    d = get_downscale_factor(num_downscales, step, resolution_schedule, training)
    if d > 1:
        return resize_image(image, d)
    return image


def get_background_color(
    background_color: str,
    training: bool,
    device: torch.device,
    cached_color: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return (background, updated_cached_color).

    If background_color == "random": returns a random color during training,
    otherwise the cached color.  Updates the cache on first eval call.

    Args:
        background_color: "random", "black", or "white"
        training: whether the model is in training mode
        device: target device
        cached_color: previously cached color (for eval consistency)
    Returns:
        (bg_tensor, updated_cache)
    """
    if background_color == "random":
        if training:
            return torch.rand(3, device=device), cached_color
        if cached_color is not None:
            return cached_color.to(device), cached_color
        # First evaluation call — pick a fixed color
        color = torch.tensor([0.1490, 0.1647, 0.2157], device=device)  # neutral grayish
        return color, color
    elif background_color == "white":
        return torch.ones(3, device=device), cached_color
    elif background_color == "black":
        return torch.zeros(3, device=device), cached_color
    else:
        raise ValueError(f"Unknown background color: {background_color}")


def get_empty_outputs(
    width: int, height: int, background: torch.Tensor
) -> Dict[str, Union[torch.Tensor, List]]:
    """Return a blank render result when no gaussians are visible."""
    rgb = background.repeat(height, width, 1)
    depth = background.new_ones(*rgb.shape[:2], 1) * 10
    accumulation = background.new_zeros(*rgb.shape[:2], 1)
    return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}


def composite_with_background(image: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    """If the image has an alpha channel, composite it over *background*."""
    if image.shape[2] == 4:
        alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
        return alpha * image[..., :3] + (1 - alpha) * background
    return image


def prepare_gt_image(
    image: torch.Tensor,
    num_downscales: int,
    step: int,
    resolution_schedule: int,
    training: bool,
    device: torch.device,
) -> torch.Tensor:
    """Convert uint8 GT image to float32 and downscale to match training resolution."""
    if image.dtype == torch.uint8:
        image = image.float() / 255.0
    return downscale_if_required(image, num_downscales, step, resolution_schedule, training).to(device)


def apply_bilateral_grid(
    bil_grids: BilateralGrid,
    rgb: torch.Tensor,
    cam_idx: int,
    H: int,
    W: int,
    device: torch.device,
) -> torch.Tensor:
    """Apply bilateral grid color correction at full resolution."""
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1.0, H, device=device),
        torch.linspace(0, 1.0, W, device=device),
        indexing="ij",
    )
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    out = slice(
        bil_grids=bil_grids,
        rgb=rgb,
        xy=grid_xy,
        grid_idx=torch.tensor(cam_idx, device=device, dtype=torch.long),
    )
    return out["rgb"]


# ═══════════════════════════════════════════════════════════════════════════════
# rasterization wrapper
# ═══════════════════════════════════════════════════════════════════════════════


try:
    from gsplat.rendering import rasterization  # noqa: F401
except ImportError:
    print("Please install gsplat>=1.0.0")
    rasterization = None


def render_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmat: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    sh_degree: Optional[int],
    render_mode: str = "RGB",
    absgrad: bool = False,
    rasterize_mode: str = "classic",
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Run gsplat rasterization — single call for both RGB and DINO features.

    Args:
        means: gaussian centers (N, 3)
        quats: gaussian rotations (N, 4)
        scales: gaussian scales in log-space (N, 3)
        opacities: sigmoid-activated opacities (N,) or (N, 1)
        colors: SH coefficients (N, num_sh, 3) or DINO features (N, D)
        viewmat: world-to-camera 4x4 matrix (1, 4, 4)
        K: intrinsics 3x3 matrix (1, 3, 3)
        width, height: output resolution
        sh_degree: SH degree (None for non-SH data like DINO features)
        render_mode: "RGB" or "RGB+ED"
        absgrad: use absolute gradient for densification
        rasterize_mode: "classic" or "antialiased"
    Returns:
        (render, alpha, info) — same as gsplat.rasterization
    """
    if rasterization is None:
        raise RuntimeError("gsplat>=1.0.0 is required for rendering")

    if opacities.dim() == 2:
        opacities = opacities.squeeze(-1)

    return rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=width,
        height=height,
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        render_mode=render_mode,
        sh_degree=sh_degree,
        sparse_grad=False,
        absgrad=absgrad,
        rasterize_mode=rasterize_mode,
    )
