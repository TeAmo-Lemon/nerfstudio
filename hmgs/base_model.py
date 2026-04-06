from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import nn
from torch.nn import Parameter

from hmgs.callbacks import TrainingCallback, TrainingCallbackAttributes
from hmgs.cameras import Cameras
from hmgs.config_base import InstantiateConfig
from hmgs.scene_box import OrientedBox, SceneBox


@dataclass
class ModelConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Model)
    eval_num_rays_per_chunk: int = 4096
    prompt: Optional[str] = None


class Model(nn.Module):
    config: ModelConfig

    def __init__(self, config: ModelConfig, scene_box: SceneBox, num_train_data: int, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.scene_box = scene_box
        self.num_train_data = num_train_data
        self.kwargs = kwargs
        self.populate_modules()
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        return self.device_indicator_param.device

    def populate_modules(self) -> None:
        return None

    def forward(self, camera: Cameras):
        return self.get_outputs(camera)

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        del training_callback_attributes
        return []

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        raise NotImplementedError

    def get_outputs(self, camera: Cameras):
        raise NotImplementedError

    def get_metrics_dict(self, outputs, batch):
        del outputs, batch
        return {}

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        del outputs, batch, metrics_dict
        raise NotImplementedError

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None):
        del obb_box
        return self.get_outputs(camera.to(self.device))

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def update_to_step(self, step: int) -> None:
        del step
