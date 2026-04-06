from __future__ import annotations

import math
from typing import Literal, Tuple

import torch


def normalize(x: torch.Tensor) -> torch.Tensor:
    return x / torch.linalg.norm(x)


def get_distortion_params(
    k1: float = 0.0,
    k2: float = 0.0,
    k3: float = 0.0,
    k4: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> torch.Tensor:
    return torch.tensor([k1, k2, k3, k4, p1, p2], dtype=torch.float32)


def rotation_matrix_between(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.linalg.cross(a, b)
    eps = 1e-6
    if torch.sum(torch.abs(v)) < eps:
        x = torch.tensor([1.0, 0.0, 0.0], dtype=a.dtype, device=a.device) if abs(float(a[0])) < eps else torch.tensor(
            [0.0, 1.0, 0.0], dtype=a.dtype, device=a.device
        )
        v = torch.linalg.cross(a, x)
    v = v / torch.linalg.norm(v)
    skew = torch.tensor(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=a.dtype,
        device=a.device,
    )
    theta = torch.acos(torch.clip(torch.dot(a, b), -1.0, 1.0))
    return torch.eye(3, dtype=a.dtype, device=a.device) + torch.sin(theta) * skew + (1 - torch.cos(theta)) * (skew @ skew)


def focus_of_attention(poses: torch.Tensor, initial_focus: torch.Tensor) -> torch.Tensor:
    active_directions = -poses[:, :3, 2:3]
    active_origins = poses[:, :3, 3:4]
    focus_pt = initial_focus
    active = torch.sum(active_directions.squeeze(-1) * (focus_pt - active_origins.squeeze(-1)), dim=-1) > 0
    done = False
    while torch.sum(active.int()) > 1 and not done:
        dirs = active_directions[active]
        origins = active_origins[active]
        m = torch.eye(3, dtype=poses.dtype, device=poses.device) - dirs * torch.transpose(dirs, -2, -1)
        mt_m = torch.transpose(m, -2, -1) @ m
        focus_pt = torch.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
        new_active = torch.sum(dirs.squeeze(-1) * (focus_pt - origins.squeeze(-1)), dim=-1) > 0
        if new_active.all():
            done = True
        active = new_active
    return focus_pt


def auto_orient_and_center_poses(
    poses: torch.Tensor,
    method: Literal["pca", "up", "vertical", "none"] = "up",
    center_method: Literal["poses", "focus", "none"] = "poses",
) -> Tuple[torch.Tensor, torch.Tensor]:
    origins = poses[..., :3, 3]
    mean_origin = torch.mean(origins, dim=0)
    translation_diff = origins - mean_origin
    if center_method == "poses":
        translation = mean_origin
    elif center_method == "focus":
        translation = focus_of_attention(poses, mean_origin)
    elif center_method == "none":
        translation = torch.zeros_like(mean_origin)
    else:
        raise ValueError(f"Unknown center method: {center_method}")

    if method == "pca":
        _, eigvec = torch.linalg.eigh(translation_diff.T @ translation_diff)
        eigvec = torch.flip(eigvec, dims=(-1,))
        if torch.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]
        transform = torch.cat([eigvec, eigvec @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses
        if oriented_poses.mean(dim=0)[2, 1] < 0:
            oriented_poses[1:3, :] = -oriented_poses[1:3, :]
            transform[1:3, :] = -transform[1:3, :]
    elif method in ("up", "vertical"):
        up = torch.mean(poses[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)
        if method == "vertical":
            x_axis_matrix = poses[:, :3, 0]
            _, s, vh = torch.linalg.svd(x_axis_matrix, full_matrices=False)
            if s[1] > 0.17 * math.sqrt(poses.shape[0]):
                up_vertical = vh[2, :]
                up = up_vertical if torch.dot(up_vertical, up) > 0 else -up_vertical
            else:
                up = up - vh[0, :] * torch.dot(up, vh[0, :])
                up = up / torch.linalg.norm(up)
        rotation = rotation_matrix_between(up, torch.tensor([0.0, 0.0, 1.0], dtype=poses.dtype, device=poses.device))
        transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
        oriented_poses = transform @ poses
    elif method == "none":
        transform = torch.eye(4, dtype=poses.dtype, device=poses.device)
        transform[:3, 3] = -translation
        transform = transform[:3, :]
        oriented_poses = transform @ poses
    else:
        raise ValueError(f"Unknown orientation method: {method}")
    return oriented_poses, transform
