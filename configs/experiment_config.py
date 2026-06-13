"""Experiment configuration — separated from base_config to avoid circular imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import yaml

from configs.base_config import (
    InstantiateConfig,
    LoggingConfig,
    MachineConfig,
    ViewerConfig,
)
from configs.config_utils import to_immutable_dict
from engine.optimizers import OptimizerConfig
from engine.schedulers import SchedulerConfig
from engine.pipeline import VanillaPipelineConfig
from utils.system_utils import CONSOLE


@dataclass
class ExperimentConfig(InstantiateConfig):
    """Full config contents for running an experiment."""

    output_dir: Path = Path("outputs")
    method_name: Optional[str] = None
    experiment_name: Optional[str] = None
    project_name: Optional[str] = "nerfstudio-project"
    timestamp: str = "{timestamp}"
    machine: MachineConfig = field(default_factory=MachineConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    pipeline: VanillaPipelineConfig = field(default_factory=VanillaPipelineConfig)
    optimizers: Dict[str, Any] = to_immutable_dict(
        {"fields": {"optimizer": OptimizerConfig(), "scheduler": SchedulerConfig()}}
    )
    vis: Literal[
        "viewer", "wandb", "tensorboard", "comet", "viewer+wandb", "viewer+tensorboard", "viewer+comet"
    ] = "viewer"
    data: Optional[Path] = None
    prompt: Optional[str] = None
    relative_model_dir: Path = Path("nerfstudio_models/")
    load_scheduler: bool = True

    def is_viewer_enabled(self) -> bool:
        return self.vis in ("viewer", "viewer+wandb", "viewer+tensorboard", "viewer+comet")

    def is_wandb_enabled(self) -> bool:
        return ("wandb" == self.vis) | ("viewer+wandb" == self.vis)

    def is_tensorboard_enabled(self) -> bool:
        return ("tensorboard" == self.vis) | ("viewer+tensorboard" == self.vis)

    def is_comet_enabled(self) -> bool:
        return ("comet" == self.vis) | ("viewer+comet" == self.vis)

    def set_timestamp(self) -> None:
        if self.timestamp == "{timestamp}":
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def set_experiment_name(self) -> None:
        if self.experiment_name is None:
            datapath = self.pipeline.datamanager.data
            if datapath is not None:
                datapath = datapath.parent if datapath.is_file() else datapath
                self.experiment_name = str(datapath.stem)
            else:
                self.experiment_name = "unnamed"

    def get_base_dir(self) -> Path:
        assert self.method_name is not None, "Please set method name in config or via the cli"
        self.set_experiment_name()
        self.set_timestamp()
        return Path(f"{self.output_dir}/{self.method_name}/{self.experiment_name}/{self.timestamp}")

    def get_checkpoint_dir(self) -> Path:
        return Path(self.get_base_dir() / self.relative_model_dir)

    def print_to_terminal(self) -> None:
        CONSOLE.rule("Config")
        CONSOLE.print(self)
        CONSOLE.rule("")

    def save_config(self) -> None:
        base_dir = self.get_base_dir()
        assert base_dir is not None
        base_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = base_dir / "config.yml"
        CONSOLE.log(f"Saving config to: {config_yaml_path}")
        config_yaml_path.write_text(yaml.dump(self), "utf8")
