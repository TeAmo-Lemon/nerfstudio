from __future__ import annotations

import torch


def multiply(pose_a: torch.Tensor, pose_b: torch.Tensor) -> torch.Tensor:
    R1, t1 = pose_a[..., :3, :3], pose_a[..., :3, 3:]
    R2, t2 = pose_b[..., :3, :3], pose_b[..., :3, 3:]
    R = R1.matmul(R2)
    t = t1 + R1.matmul(t2)
    return torch.cat([R, t], dim=-1)
