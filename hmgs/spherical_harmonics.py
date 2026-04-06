from __future__ import annotations

import torch

MAX_SH_DEGREE = 4
C0 = 0.28209479177387814


def num_sh_bases(degree: int) -> int:
    if degree > MAX_SH_DEGREE:
        raise ValueError(f"SH degree {degree} is not supported")
    return (degree + 1) ** 2


def RGB2SH(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / C0


def SH2RGB(sh: torch.Tensor) -> torch.Tensor:
    return sh * C0 + 0.5
