from __future__ import annotations

import platform
import warnings
from typing import Any, Dict, List, Optional, TypeVar, Union

import torch

T = TypeVar("T")


def get_dict_to_torch(stuff: T, device: Union[torch.device, str] = "cpu", exclude: Optional[List[str]] = None) -> T:
    if isinstance(stuff, dict):
        for key, value in stuff.items():
            if exclude and key in exclude:
                continue
            stuff[key] = get_dict_to_torch(value, device=device)
        return stuff
    if isinstance(stuff, torch.Tensor):
        return stuff.to(device)  # type: ignore[return-value]
    return stuff


def step_check(step: int, step_size: int, run_at_zero: bool = False) -> bool:
    if step_size == 0:
        return False
    return (run_at_zero or step != 0) and step % step_size == 0


def torch_compile(*args, **kwargs) -> Any:
    if not hasattr(torch, "compile"):
        warnings.warn("torch.compile is unavailable; falling back to eager mode.", RuntimeWarning)
        if args and callable(args[0]):
            return args[0]
        return lambda x: x
    if platform.system() == "Windows":
        warnings.warn("torch.compile is disabled on Windows; falling back to eager mode.", RuntimeWarning)
        if args and callable(args[0]):
            return args[0]
        return lambda x: x
    return torch.compile(*args, **kwargs)
