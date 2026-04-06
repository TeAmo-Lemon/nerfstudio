from __future__ import annotations

from typing import Union

import torch

COLORS_DICT = {
    "white": torch.tensor([1.0, 1.0, 1.0]),
    "black": torch.tensor([0.0, 0.0, 0.0]),
    "red": torch.tensor([1.0, 0.0, 0.0]),
    "green": torch.tensor([0.0, 1.0, 0.0]),
    "blue": torch.tensor([0.0, 0.0, 1.0]),
}


def get_color(color: Union[str, list]) -> torch.Tensor:
    if isinstance(color, str):
        key = color.lower()
        if key not in COLORS_DICT:
            raise ValueError(f"Unsupported color preset: {color}")
        return COLORS_DICT[key]
    if isinstance(color, list) and len(color) == 3:
        return torch.tensor(color, dtype=torch.float32)
    raise ValueError(f"Color should be a preset string or 3-element list, got {type(color)}")
