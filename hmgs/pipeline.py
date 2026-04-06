from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple, Type, cast

import torch
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn import Parameter
from torch.nn.parallel import DistributedDataParallel as DDP

from hmgs.base_model import Model, ModelConfig
from hmgs.callbacks import TrainingCallback, TrainingCallbackAttributes
from hmgs.config_base import InstantiateConfig
from hmgs.data import FullImageDatamanager, FullImageDatamanagerConfig


def module_wrapper(ddp_or_model: nn.Module) -> Model:
    if isinstance(ddp_or_model, DDP):
        return cast(Model, ddp_or_model.module)
    return cast(Model, ddp_or_model)


@dataclass
class VanillaPipelineConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: VanillaPipeline)
    datamanager: FullImageDatamanagerConfig = field(default_factory=FullImageDatamanagerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


class VanillaPipeline(nn.Module):
    datamanager: FullImageDatamanager
    _model: Model

    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: str = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.datamanager = config.datamanager.setup(
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
        )
        seed_points = None
        if "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata:
            seed_points = (
                self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"],
                self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"],
            )
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_points,
        )
        self._model.to(device)
        self.world_size = world_size
        if world_size > 1:
            self._model = cast(Model, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True))

    @property
    def model(self) -> Model:
        return module_wrapper(self._model)

    @property
    def device(self):
        return self.model.device

    def forward(self):
        raise NotImplementedError

    def get_train_loss_dict(self, step: int):
        camera, batch = self.datamanager.next_train(step)
        model_outputs = self._model(camera)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    def get_eval_loss_dict(self, step: int):
        self.eval()
        camera, batch = self.datamanager.next_eval(step)
        model_outputs = self.model(camera)
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        self.train()
        return model_outputs, loss_dict, metrics_dict

    def get_eval_image_metrics_and_images(self, step: int):
        self.eval()
        camera, batch = self.datamanager.next_eval_image(step)
        outputs = self.model.get_outputs_for_camera(camera)
        metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
        metrics_dict["num_rays"] = int((camera.height * camera.width * camera.size).item())
        self.train()
        return metrics_dict, images_dict

    def get_average_eval_image_metrics(
        self,
        step: Optional[int] = None,
        output_path: Optional[Path] = None,
        get_std: bool = False,
    ):
        del step, get_std
        self.eval()
        metrics_list = []
        if output_path is not None:
            import torchvision.utils as vutils

            output_path.mkdir(parents=True, exist_ok=True)
        for idx, (camera, batch) in enumerate(self.datamanager.fixed_indices_eval_dataloader):
            inner_start = time()
            outputs = self.model.get_outputs_for_camera(camera)
            height, width = camera.height, camera.width
            num_rays = height * width
            metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
            if output_path is not None:
                for key, image in images_dict.items():
                    vutils.save_image(image.permute(2, 0, 1).cpu(), output_path / f"eval_{key}_{idx:04d}.png")
            metrics_dict["num_rays_per_sec"] = float((num_rays / (time() - inner_start)).item())
            metrics_list.append(metrics_dict)
        self.train()
        if not metrics_list:
            return {}
        return {
            key: float(torch.mean(torch.tensor([metrics[key] for metrics in metrics_list])))
            for key in metrics_list[0].keys()
        }

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        return self.model.get_training_callbacks(training_callback_attributes)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return self.model.get_param_groups()
