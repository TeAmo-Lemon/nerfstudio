from __future__ import annotations

import math
from typing import Tuple

import torch


def random_quat_tensor(num_quats: int) -> torch.Tensor:
    u = torch.rand(num_quats)
    v = torch.rand(num_quats)
    w = torch.rand(num_quats)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def k_nearest_sklearn(x: torch.Tensor, k: int, metric: str = "euclidean") -> Tuple[torch.Tensor, torch.Tensor]:
    from sklearn.neighbors import NearestNeighbors

    x_np = x.detach().cpu().numpy()
    nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric=metric).fit(x_np)
    distances, indices = nn_model.kneighbors(x_np)
    return (
        torch.tensor(distances[:, 1:], dtype=torch.float32),
        torch.tensor(indices[:, 1:], dtype=torch.int64),
    )
