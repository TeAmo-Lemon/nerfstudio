from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Type

from nerfstudio.cameras.camera_optimizers import CameraOptimizer as NSCameraOptimizer

from hmgs.config_base import InstantiateConfig


@dataclass
class CameraOptimizerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: CameraOptimizer)
    mode: Literal["off", "SO3xR3", "SE3"] = "off"
    trans_l2_penalty: float = 1e-2
    rot_l2_penalty: float = 1e-3


class CameraOptimizer(NSCameraOptimizer):
    """HMGS wrapper around Nerfstudio camera optimizer.

    Subclassing keeps runtime behavior while satisfying viewer type checks.
    """
