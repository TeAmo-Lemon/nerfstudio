#!/usr/bin/env python
from __future__ import annotations

import argparse
import dataclasses
import functools
import random
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import yaml
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import AdamOptimizerConfig, Optimizers
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.available_devices import get_available_devices
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.viewer.viewer import Viewer

# Speed up static-shape CUDA kernels.
torch.backends.cudnn.benchmark = True  # type: ignore


@dataclass
class SessionState:
    training_state: Literal["training", "paused", "completed"] = "training"


class DirectPipeline(nn.Module):
    """Minimal pipeline wrapper used by the viewer and checkpoints."""

    datamanager: FullImageDatamanager
    _model: SplatfactoModel

    def __init__(self, datamanager: FullImageDatamanager, model: SplatfactoModel):
        super().__init__()
        self.datamanager = datamanager
        self._model = model

    @property
    def model(self) -> SplatfactoModel:
        return self._model

    def load_pipeline(self, loaded_state: Dict[str, torch.Tensor], step: int) -> None:
        state = {
            (key[len("module.") :] if key.startswith("module.") else key): value for key, value in loaded_state.items()
        }
        self.model.update_to_step(step)
        self.load_state_dict(state, strict=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Splatfacto training entrypoint.")
    parser.add_argument("-s", "--source-path", type=Path, required=True, help="COLMAP dataset root.")
    parser.add_argument("-m", "--model-path", type=Path, required=True, help="Output directory for config/checkpoints.")
    parser.add_argument("--max-steps", type=int, default=30000, help="Number of training iterations.")
    parser.add_argument("--save-every", type=int, default=2000, help="Checkpoint period.")
    parser.add_argument("--eval-image-every", type=int, default=100, help="Eval image period.")
    parser.add_argument("--eval-all-every", type=int, default=1000, help="Eval-all period.")
    parser.add_argument("--disable-viewer", action="store_true", help="Disable Viser during training.")
    parser.add_argument("--viewer-port", type=int, default=None, help="Viewer websocket port override.")
    parser.add_argument("--load-checkpoint", type=Path, default=None, help="Checkpoint to resume from.")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in the model dir.")
    parser.add_argument("--vis", choices=["viewer", "none"], default="viewer", help="Visualization mode.")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda", help="Training device.")
    parser.add_argument("--data-device", choices=["cpu", "gpu", "disk"], default="gpu", help="Image cache target.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--steps-per-log", type=int, default=10, help="Scalar logging period.")
    parser.add_argument("--mixed-precision", action="store_true", help="Enable autocast/GradScaler.")
    return parser.parse_args()


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_config(args: argparse.Namespace) -> TrainerConfig:
    viewer_config = ViewerConfig(num_rays_per_chunk=1 << 15, websocket_port=args.viewer_port)
    config = TrainerConfig(
        method_name="splatfacto",
        experiment_name=args.model_path.name,
        output_dir=args.model_path.parent,
        steps_per_eval_image=args.eval_image_every,
        steps_per_eval_batch=0,
        steps_per_save=args.save_every,
        steps_per_eval_all_images=args.eval_all_every,
        max_num_iterations=args.max_steps,
        mixed_precision=args.mixed_precision,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=ColmapDataParserConfig(data=args.source_path, load_3D_points=True),
                cache_images=args.data_device,
                cache_images_type="uint8",
            ),
            model=SplatfactoModelConfig(),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6, max_steps=args.max_steps),
            },
            "features_dc": {"optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15), "scheduler": None},
            "features_rest": {"optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15), "scheduler": None},
            "opacities": {"optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15), "scheduler": None},
            "scales": {"optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15), "scheduler": None},
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7,
                    max_steps=args.max_steps,
                    warmup_steps=1000,
                    lr_pre_warmup=0,
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4,
                    max_steps=args.max_steps,
                    warmup_steps=1000,
                    lr_pre_warmup=0,
                ),
            },
        },
        viewer=viewer_config,
        vis="none" if args.disable_viewer or args.vis == "none" else "viewer",
    )
    config.machine.seed = args.seed
    config.machine.device_type = args.device
    config.logging.steps_per_log = args.steps_per_log
    return config


def _save_config(config: TrainerConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yml"
    CONSOLE.log(f"Saving config to: {config_path}")
    config_path.write_text(yaml.dump(config), encoding="utf8")


def _get_seed_points(datamanager: FullImageDatamanager) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    metadata = getattr(datamanager.train_dataparser_outputs, "metadata", {})
    if "points3D_xyz" not in metadata:
        return None
    return metadata["points3D_xyz"], metadata["points3D_rgb"]


def _setup_pipeline(
    config: TrainerConfig, device: str, grad_scaler: GradScaler
) -> Tuple[DirectPipeline, Optimizers, List[TrainingCallback]]:
    datamanager = config.pipeline.datamanager.setup(device=device, test_mode="val", world_size=1, local_rank=0)
    assert isinstance(datamanager, FullImageDatamanager)
    assert datamanager.train_dataset is not None, "Missing train dataset."

    model = config.pipeline.model.setup(
        scene_box=datamanager.train_dataset.scene_box,
        num_train_data=len(datamanager.train_dataset),
        metadata=datamanager.train_dataset.metadata,
        device=device,
        grad_scaler=grad_scaler,
        seed_points=_get_seed_points(datamanager),
    )
    assert isinstance(model, SplatfactoModel)
    model.to(device)

    pipeline = DirectPipeline(datamanager=datamanager, model=model)
    param_groups = {**datamanager.get_param_groups(), **model.get_param_groups()}
    optimizers = Optimizers(config.optimizers.copy(), param_groups)
    callbacks = datamanager.get_training_callbacks(
        TrainingCallbackAttributes(optimizers=optimizers, grad_scaler=grad_scaler, pipeline=None, trainer=None)
    ) + model.get_training_callbacks(
        TrainingCallbackAttributes(optimizers=optimizers, grad_scaler=grad_scaler, pipeline=None, trainer=None)
    )
    return pipeline, optimizers, callbacks


def _setup_runtime_logging(config: TrainerConfig, output_dir: Path, banner_messages: Optional[List[str]]) -> None:
    writer_log_path = output_dir / config.logging.relative_log_dir
    writer.setup_event_writer(
        config.is_wandb_enabled(),
        config.is_tensorboard_enabled(),
        config.is_comet_enabled(),
        log_dir=writer_log_path,
        experiment_name=config.experiment_name,
        project_name=config.project_name,
    )
    writer.setup_local_writer(config.logging, max_iter=config.max_num_iterations, banner_messages=banner_messages)
    writer.put_config(name="config", config_dict=dataclasses.asdict(config), step=0)
    profiler.setup_profiler(config.logging, writer_log_path)


def _find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    checkpoints = sorted(checkpoint_dir.glob("step-*.ckpt"))
    return checkpoints[-1] if checkpoints else None


def _load_checkpoint(
    pipeline: DirectPipeline,
    optimizers: Optimizers,
    grad_scaler: GradScaler,
    checkpoint_path: Path,
) -> int:
    loaded_state = torch.load(checkpoint_path, map_location="cpu")
    start_step = loaded_state["step"] + 1
    pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
    optimizers.load_optimizers(loaded_state["optimizers"])
    if "schedulers" in loaded_state:
        optimizers.load_schedulers(loaded_state["schedulers"])
    grad_scaler.load_state_dict(loaded_state["scalers"])
    CONSOLE.print(f"Done loading checkpoint from {checkpoint_path}")
    return start_step


def _save_checkpoint(
    pipeline: DirectPipeline,
    optimizers: Optimizers,
    grad_scaler: GradScaler,
    checkpoint_dir: Path,
    step: int,
    save_only_latest: bool = True,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"step-{step:09d}.ckpt"
    torch.save(
        {
            "step": step,
            "pipeline": pipeline.state_dict(),
            "optimizers": {name: opt.state_dict() for name, opt in optimizers.optimizers.items()},
            "schedulers": {name: sch.state_dict() for name, sch in optimizers.schedulers.items()},
            "scalers": grad_scaler.state_dict(),
        },
        checkpoint_path,
    )
    if save_only_latest:
        for other_checkpoint in checkpoint_dir.glob("*.ckpt"):
            if other_checkpoint != checkpoint_path:
                other_checkpoint.unlink()


def _resolve_checkpoint_path(args: argparse.Namespace, checkpoint_dir: Path) -> Optional[Path]:
    if args.load_checkpoint is not None:
        return args.load_checkpoint
    if args.resume:
        return _find_latest_checkpoint(checkpoint_dir)
    return None


def _setup_viewer(
    config: TrainerConfig,
    output_dir: Path,
    source_path: Path,
    pipeline: DirectPipeline,
    session_state: SessionState,
    train_lock: Lock,
) -> Optional[Viewer]:
    if not config.is_viewer_enabled():
        return None
    viewer_log_path = output_dir / config.viewer.relative_log_filename
    viewer = Viewer(
        config.viewer,
        log_filename=viewer_log_path,
        datapath=source_path,
        pipeline=pipeline,
        trainer=session_state,
        train_lock=train_lock,
        share=config.viewer.make_share_url,
    )
    viewer.init_scene(
        train_dataset=pipeline.datamanager.train_dataset,
        train_state=session_state.training_state,
        eval_dataset=pipeline.datamanager.eval_dataset,
    )
    return viewer


def _run_train_loop(
    config: TrainerConfig,
    output_dir: Path,
    checkpoint_dir: Path,
    pipeline: DirectPipeline,
    optimizers: Optimizers,
    callbacks: List[TrainingCallback],
    grad_scaler: GradScaler,
    session_state: SessionState,
    viewer: Optional[Viewer],
    train_lock: Lock,
    start_step: int,
) -> None:
    datamanager = pipeline.datamanager
    model = pipeline.model
    if hasattr(datamanager, "train_dataparser_outputs"):
        datamanager.train_dataparser_outputs.save_dataparser_transform(output_dir / "dataparser_transforms.json")

    gradient_accumulation_steps = config.gradient_accumulation_steps or {}
    with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
        for step in range(start_step, config.max_num_iterations):
            while session_state.training_state == "paused":
                time.sleep(0.01)

            with train_lock:
                with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                    model.train()
                    for callback in callbacks:
                        callback.run_callback_at_location(step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)

                    needs_zero = [
                        group
                        for group in optimizers.parameters.keys()
                        if step % gradient_accumulation_steps.get(group, 1) == 0
                    ]
                    optimizers.zero_grad_some(needs_zero)

                    device_type = model.device.type
                    autocast_device = "cpu" if device_type == "mps" else device_type
                    with torch.autocast(device_type=autocast_device, enabled=config.mixed_precision):
                        camera, batch = datamanager.next_train(step)
                        outputs = model(camera)
                        metrics_dict = model.get_metrics_dict(outputs, batch)
                        loss_dict = model.get_loss_dict(outputs, batch, metrics_dict)
                        loss = functools.reduce(torch.add, loss_dict.values())

                    grad_scaler.scale(loss).backward()  # type: ignore[arg-type]

                    needs_step = [
                        group
                        for group in optimizers.parameters.keys()
                        if step % gradient_accumulation_steps.get(group, 1) == gradient_accumulation_steps.get(group, 1) - 1
                    ]
                    optimizers.optimizer_scaler_step_some(grad_scaler, needs_step)

                    scale = grad_scaler.get_scale()
                    grad_scaler.update()
                    if scale <= grad_scaler.get_scale():
                        optimizers.scheduler_step_all(step)

                    for callback in callbacks:
                        callback.run_callback_at_location(step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)

            if step > 1:
                writer.put_time(
                    name=EventName.TRAIN_RAYS_PER_SEC,
                    duration=datamanager.get_train_rays_per_batch() / max(0.001, train_t.duration),
                    step=step,
                    avg_over_steps=True,
                )

            if viewer is not None:
                viewer.update_scene(step, datamanager.get_train_rays_per_batch())

            if step_check(step, config.logging.steps_per_log, run_at_zero=True):
                writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                if torch.cuda.is_available():
                    writer.put_scalar(
                        name="GPU Memory (MB)",
                        scalar=torch.cuda.max_memory_allocated() / (1024**2),
                        step=step,
                    )

            if step_check(step, config.steps_per_save):
                _save_checkpoint(
                    pipeline=pipeline,
                    optimizers=optimizers,
                    grad_scaler=grad_scaler,
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    save_only_latest=config.save_only_latest_checkpoint,
                )

            writer.write_out_storage()

    session_state.training_state = "completed"
    _save_checkpoint(
        pipeline=pipeline,
        optimizers=optimizers,
        grad_scaler=grad_scaler,
        checkpoint_dir=checkpoint_dir,
        step=config.max_num_iterations - 1,
        save_only_latest=config.save_only_latest_checkpoint,
    )
    writer.write_out_storage()
    table = Table(title=None, show_header=False, box=box.MINIMAL, title_style=style.Style(bold=True))
    table.add_row("Config File", str(output_dir / "config.yml"))
    table.add_row("Checkpoint Directory", str(checkpoint_dir))
    CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))
    if viewer is not None:
        viewer.training_complete()
        if not config.viewer.quit_on_train_completion:
            CONSOLE.print("Use ctrl+c to quit", justify="center")
            while True:
                time.sleep(0.01)


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    available_devices = get_available_devices()
    if config.machine.device_type not in available_devices:
        raise RuntimeError(
            f"Specified device type '{config.machine.device_type}' is not available. "
            f"Available device types: {available_devices}."
        )

    device = config.machine.device_type
    if device == "cuda":
        device = "cuda:0"
    output_dir = args.model_path.resolve()
    checkpoint_dir = output_dir / config.relative_model_dir
    _set_random_seed(config.machine.seed)
    _save_config(config, output_dir)
    CONSOLE.log(f"Saving checkpoints to: {checkpoint_dir}")

    train_lock = Lock()
    session_state = SessionState(
        training_state="paused" if config.start_paused else "training",
    )
    grad_scaler = GradScaler(enabled=config.mixed_precision or config.use_grad_scaler)
    pipeline, optimizers, callbacks = _setup_pipeline(config=config, device=device, grad_scaler=grad_scaler)

    viewer = _setup_viewer(
        config=config,
        output_dir=output_dir,
        source_path=args.source_path.resolve(),
        pipeline=pipeline,
        session_state=session_state,
        train_lock=train_lock,
    )
    banner_messages = viewer.viewer_info if viewer is not None else None
    _setup_runtime_logging(config, output_dir, banner_messages)

    start_step = 0
    checkpoint_path = _resolve_checkpoint_path(args, checkpoint_dir)
    if checkpoint_path is not None:
        start_step = _load_checkpoint(
            pipeline=pipeline,
            optimizers=optimizers,
            grad_scaler=grad_scaler,
            checkpoint_path=checkpoint_path,
        )
    else:
        CONSOLE.print("No checkpoint to load, training from scratch.")

    try:
        _run_train_loop(
            config=config,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            pipeline=pipeline,
            optimizers=optimizers,
            callbacks=callbacks,
            grad_scaler=grad_scaler,
            session_state=session_state,
            viewer=viewer,
            train_lock=train_lock,
            start_step=start_step,
        )
    except KeyboardInterrupt:
        CONSOLE.print(traceback.format_exc())
    finally:
        if viewer is not None:
            viewer.viser_server.stop()
        profiler.flush_profiler(config.logging)


if __name__ == "__main__":
    main()
