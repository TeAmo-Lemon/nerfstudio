from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import torch


@dataclass
class SceneBox:
    aabb: torch.Tensor

    def within(self, pts: torch.Tensor) -> torch.Tensor:
        return torch.all(pts > self.aabb[0], dim=-1) & torch.all(pts < self.aabb[1], dim=-1)

    def get_center(self) -> torch.Tensor:
        return (self.aabb[0] + self.aabb[1]) / 2.0

    def get_centered_and_scaled_scene_box(self, scale_factor: Union[float, torch.Tensor] = 1.0) -> "SceneBox":
        return SceneBox(aabb=(self.aabb - self.get_center()) * scale_factor)


@dataclass
class OrientedBox:
    R: torch.Tensor
    T: torch.Tensor
    S: torch.Tensor

    def within(self, pts: torch.Tensor) -> torch.Tensor:
        R, T, S = self.R.to(pts), self.T.to(pts), self.S.to(pts)
        H = torch.eye(4, device=pts.device, dtype=pts.dtype)
        H[:3, :3] = R
        H[:3, 3] = T
        H_world2bbox = torch.inverse(H)
        pts_h = torch.cat([pts, torch.ones_like(pts[..., :1])], dim=-1)
        pts_local = torch.matmul(H_world2bbox, pts_h.T).T[..., :3]
        lower = -S / 2
        upper = S / 2
        return torch.all(torch.cat([pts_local > lower, pts_local < upper], dim=-1), dim=-1)
