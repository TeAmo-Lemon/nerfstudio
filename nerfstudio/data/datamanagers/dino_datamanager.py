from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.data.datasets.dino_dataset import DinoInputDataset


@dataclass
class DinoDatamanagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: DinoDatamanager)
    dino_features_dir: Optional[Path] = None
    dino_feature_dim: int = 16
    strict_dino_loading: bool = True


class DinoDatamanager(FullImageDatamanager[DinoInputDataset]):
    """Full-image datamanager that carries precomputed DINO feature maps in each batch."""

    def _resolve_dino_feature_dir(self) -> Path:
        if self.config.dino_features_dir is not None:
            return Path(self.config.dino_features_dir)
        assert self.config.dataparser.data is not None
        return Path(self.config.dataparser.data) / "dino_features"

    def create_train_dataset(self) -> DinoInputDataset:
        return DinoInputDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            cache_compressed_images=self.config.cache_compressed_images,
            dino_features_dir=self._resolve_dino_feature_dir(),
            dino_feature_dim=self.config.dino_feature_dim,
            strict_dino_loading=self.config.strict_dino_loading,
        )

    def create_eval_dataset(self) -> DinoInputDataset:
        return DinoInputDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
            cache_compressed_images=self.config.cache_compressed_images,
            dino_features_dir=self._resolve_dino_feature_dir(),
            dino_feature_dim=self.config.dino_feature_dim,
            strict_dino_loading=self.config.strict_dino_loading,
        )

    def _load_images(
        self, split: Literal["train", "eval"], cache_images_device: Literal["cpu", "gpu"]
    ) -> List[Dict[str, torch.Tensor]]:
        cached = super()._load_images(split=split, cache_images_device=cache_images_device)

        for cache in cached:
            if "dino_feature" not in cache:
                continue
            if cache_images_device == "gpu":
                cache["dino_feature"] = cache["dino_feature"].to(self.device)
            else:
                cache["dino_feature"] = cache["dino_feature"].pin_memory()

        return cached

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        camera, data = super().next_train(step)
        if "dino_feature" in data:
            data["dino_feature"] = data["dino_feature"].to(self.device)
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        camera, data = super().next_eval(step)
        if "dino_feature" in data:
            data["dino_feature"] = data["dino_feature"].to(self.device)
        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        camera, data = super().next_eval_image(step)
        if "dino_feature" in data:
            data["dino_feature"] = data["dino_feature"].to(self.device)
        return camera, data
