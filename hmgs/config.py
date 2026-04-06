from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from hmgs.camera_optimizers import CameraOptimizerConfig
from hmgs.data import ColmapDataParserConfig, FullImageDatamanagerConfig
from hmgs.model import SplatfactoModelConfig
from hmgs.optimizers import AdamOptimizerConfig
from hmgs.pipeline import VanillaPipelineConfig
from hmgs.schedulers import ExponentialDecaySchedulerConfig
from hmgs.configs.base_config import ViewerConfig


@dataclass
class MachineConfig:
    seed: int = 42
    num_devices: int = 1
    num_machines: int = 1
    machine_rank: int = 0
    dist_url: str = "auto"
    device_type: Literal["cpu", "cuda", "mps"] = "cuda"


@dataclass
class HMGSViewerConfig(ViewerConfig):
    enabled: bool = True
    open_browser: bool = True


def _base_pipeline_config(data: Optional[Path] = None, model_config: Optional[SplatfactoModelConfig] = None):
    return VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            data=data or Path(),
            dataparser=ColmapDataParserConfig(data=data or Path(), load_3D_points=True),
            cache_images_type="uint8",
        ),
        model=model_config or SplatfactoModelConfig(camera_optimizer=CameraOptimizerConfig(mode="off")),
    )


def _base_optimizers(include_dino: bool = False) -> Dict[str, Any]:
    optimizers: Dict[str, Any] = {
        "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6, max_steps=30000),
        },
        "features_dc": {"optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15), "scheduler": None},
        "features_rest": {"optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15), "scheduler": None},
        "opacities": {"optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15), "scheduler": None},
        "scales": {"optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15), "scheduler": None},
        "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
        "camera_opt": {
            "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
        "bilateral_grid": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
            ),
        },
    }
    if include_dino:
        optimizers["features_dino"] = {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        }
    return optimizers


@dataclass
class HMGSTrainConfig:
    output_dir: Path = Path("outputs/hmgs")
    method_name: str = "splatfacto"
    preset_name: str = "splatfacto"
    data: Optional[Path] = None
    machine: MachineConfig = field(default_factory=MachineConfig)
    viewer: HMGSViewerConfig = field(default_factory=HMGSViewerConfig)
    pipeline: VanillaPipelineConfig = field(default_factory=_base_pipeline_config)
    optimizers: Dict[str, Any] = field(default_factory=_base_optimizers)
    steps_per_save: int = 2000
    steps_per_eval_image: int = 1000
    max_num_iterations: int = 8000
    mixed_precision: bool = False
    use_grad_scaler: bool = False
    save_only_latest_checkpoint: bool = True
    load_scheduler: bool = True
    load_dir: Optional[Path] = None
    load_step: Optional[int] = None
    load_checkpoint: Optional[Path] = None
    logging_steps: int = 10
    gradient_accumulation_steps: Dict[str, int] = field(default_factory=dict)

    def finalize(self) -> "HMGSTrainConfig":
        if self.data is not None:
            self.pipeline.datamanager.data = self.data
            self.pipeline.datamanager.dataparser.data = self.data
        return self

    def get_base_dir(self) -> Path:
        return self.output_dir

    def get_checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_preset(cls, preset: str, data: Optional[Path] = None, output_dir: Optional[Path] = None) -> "HMGSTrainConfig":
        if preset == "splatfacto":
            config = cls(
                output_dir=output_dir or Path("outputs/hmgs"),
                method_name="splatfacto",
                preset_name="splatfacto",
                data=data,
                pipeline=_base_pipeline_config(data=data),
                optimizers=_base_optimizers(),
            )
        elif preset == "splatfacto-dino":
            config = cls(
                output_dir=output_dir or Path("outputs/hmgs"),
                method_name="splatfacto",
                preset_name="splatfacto-dino",
                data=data,
                pipeline=_base_pipeline_config(
                    data=data,
                    model_config=SplatfactoModelConfig(
                        enable_dino_distillation=True,
                        dino_feature_dim=16,
                        dino_loss_mult=0.1,
                        camera_optimizer=CameraOptimizerConfig(mode="off"),
                    ),
                ),
                optimizers=_base_optimizers(include_dino=True),
            )
        elif preset == "splatfacto-big":
            config = cls(
                output_dir=output_dir or Path("outputs/hmgs"),
                method_name="splatfacto",
                preset_name="splatfacto-big",
                data=data,
                pipeline=_base_pipeline_config(
                    data=data,
                    model_config=SplatfactoModelConfig(
                        cull_alpha_thresh=0.005,
                        densify_grad_thresh=0.0005,
                        camera_optimizer=CameraOptimizerConfig(mode="off"),
                    ),
                ),
                optimizers=_base_optimizers(),
            )
        elif preset == "splatfacto-mcmc":
            config = cls(
                output_dir=output_dir or Path("outputs/hmgs"),
                method_name="splatfacto",
                preset_name="splatfacto-mcmc",
                data=data,
                pipeline=_base_pipeline_config(
                    data=data,
                    model_config=SplatfactoModelConfig(
                        strategy="mcmc",
                        cull_alpha_thresh=0.005,
                        stop_split_at=25000,
                        camera_optimizer=CameraOptimizerConfig(mode="off"),
                    ),
                ),
                optimizers=_base_optimizers(),
            )
        else:
            raise ValueError(f"Unknown HMGS preset: {preset}")
        return config.finalize()
