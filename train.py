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
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import numpy as np
import torch
import yaml
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.dino_datamanager import DinoDatamanagerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.data.datasets.dino_dataset import DinoInputDataset
from nerfstudio.engine.optimizers import AdamOptimizerConfig, Optimizers
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.dino_splatfacto import DinoSplatfactoModelConfig
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


@dataclass
class DirectPipeline:
    """Minimal object shared with the viewer."""

    datamanager: FullImageDatamanager
    model: SplatfactoModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct Splatfacto training entrypoint.")
    parser.add_argument("-s", "--source-path", type=Path, required=True, help="COLMAP dataset root.")
    parser.add_argument("-m", "--model-path", type=Path, required=True, help="Output directory for config/checkpoints.")
    parser.add_argument(
        "--pipeline",
        choices=["splatfacto", "dino-splatfacto"],
        default="dino-splatfacto",
        help="Pipeline to train.",
    )
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
    parser.add_argument(
        "--dino-feature-dir",
        type=Path,
        default=None,
        help="Directory containing precomputed DINO .pt features (defaults to <model-path>/dino_features).",
    )
    parser.add_argument("--dino-feature-dim", type=int, default=16, help="DINO feature channel count after PCA.")
    parser.add_argument("--dino-loss-weight", type=float, default=0.05, help="Loss weight for DINO feature L1 loss.")
    parser.add_argument(
        "--dino-loss-start-step",
        type=int,
        default=3000,
        help="Enable DINO feature loss from this global training step.",
    )
    return parser.parse_args()


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_config(args: argparse.Namespace) -> TrainerConfig:
    viewer_config = ViewerConfig(num_rays_per_chunk=1 << 15, websocket_port=args.viewer_port)
    use_dino_pipeline = args.pipeline == "dino-splatfacto"

    if use_dino_pipeline:
        dino_feature_dir = args.dino_feature_dir if args.dino_feature_dir is not None else args.model_path / "dino_features"
        datamanager_config = DinoDatamanagerConfig(
            dataparser=ColmapDataParserConfig(data=args.source_path, load_3D_points=True),
            cache_images=args.data_device,
            cache_images_type="uint8",
            dino_features_dir=dino_feature_dir,
            dino_feature_dim=args.dino_feature_dim,
            strict_dino_loading=True,
        )
        model_config = DinoSplatfactoModelConfig(
            dino_feature_dim=args.dino_feature_dim,
            dino_loss_weight=args.dino_loss_weight,
            dino_loss_start_step=args.dino_loss_start_step,
        )
        method_name = "dino-splatfacto"
    else:
        datamanager_config = FullImageDatamanagerConfig(
            dataparser=ColmapDataParserConfig(data=args.source_path, load_3D_points=True),
            cache_images=args.data_device,
            cache_images_type="uint8",
        )
        model_config = SplatfactoModelConfig()
        method_name = "splatfacto"

    optimizers = {
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
    }
    if use_dino_pipeline:
        optimizers["dino_features"] = {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        }

    config = TrainerConfig(
        method_name=method_name,
        experiment_name=args.model_path.name,
        output_dir=args.model_path.parent,
        steps_per_eval_image=args.eval_image_every,
        steps_per_eval_batch=0,
        steps_per_save=args.save_every,
        steps_per_eval_all_images=args.eval_all_every,
        max_num_iterations=args.max_steps,
        mixed_precision=args.mixed_precision,
        pipeline=VanillaPipelineConfig(datamanager=datamanager_config, model=model_config),
        optimizers=optimizers,
        viewer=viewer_config,
        vis=cast(Any, "none" if args.disable_viewer or args.vis == "none" else "viewer"),
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


def _ensure_dino_features_ready(args: argparse.Namespace, pipeline: DirectPipeline) -> None:
    """Auto-extract missing DINO features before viewer/train accesses the dataset."""
    if args.pipeline != "dino-splatfacto":
        return

    train_dataset = pipeline.datamanager.train_dataset
    eval_dataset = pipeline.datamanager.eval_dataset
    if not isinstance(train_dataset, DinoInputDataset):
        raise TypeError("DINO pipeline expects DinoInputDataset for train split")

    datasets: List[DinoInputDataset] = [train_dataset]
    if isinstance(eval_dataset, DinoInputDataset):
        datasets.append(eval_dataset)

    extraction_jobs: Dict[Tuple[Path, Path], List[Path]] = {}
    for dataset in datasets:
        job_key = (dataset.image_root.resolve(), dataset.feature_file_path().resolve())
        extraction_jobs.setdefault(job_key, [])
        missing_paths = dataset.missing_feature_image_paths()
        extraction_jobs[job_key].extend(path.resolve() for path in missing_paths)

    total_missing = sum(len(paths) for paths in extraction_jobs.values())
    if total_missing == 0:
        CONSOLE.log("DINO features already exist. Skipping extraction.")
        return

    from scripts.extract_dino_features import extract_dino_features_for_images

    # Prefer the active training device so extraction can run on GPU when available.
    device_type = pipeline.model.device.type
    requested_device = device_type if device_type in {"cuda", "mps", "cpu"} else "cuda"

    total_saved = 0
    total_skipped = 0
    for (image_root, output_file), image_paths in extraction_jobs.items():
        unique_paths = sorted(set(image_paths))
        if len(unique_paths) == 0:
            continue
        output_file.parent.mkdir(parents=True, exist_ok=True)
        CONSOLE.log(
            f"Auto-extracting missing DINO features: missing={len(unique_paths)}, "
            f"image_root={image_root}, output_file={output_file}"
        )
        saved, skipped = extract_dino_features_for_images(
            image_paths=unique_paths,
            input_dir=image_root,
            output_dir=output_file,
            feature_dim=args.dino_feature_dim,
            device=requested_device,
            skip_existing=True,
        )
        total_saved += saved
        total_skipped += skipped

    for dataset in datasets:
        dataset.invalidate_feature_cache()

    still_missing: List[Path] = []
    for dataset in datasets:
        for missing_image_path in dataset.missing_feature_image_paths()[:5]:
            still_missing.append(missing_image_path)
        if len(still_missing) >= 5:
            break

    if len(still_missing) > 0:
        raise RuntimeError(
            "DINO feature auto-extraction completed but some feature files are still missing. "
            f"Example missing files: {still_missing}"
        )

    CONSOLE.log(f"DINO feature auto-extraction finished. saved={total_saved}, skipped={total_skipped}")


def _get_seed_points(datamanager: FullImageDatamanager) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    metadata = getattr(datamanager.train_dataparser_outputs, "metadata", {})
    if "points3D_xyz" not in metadata:
        return None
    return metadata["points3D_xyz"], metadata["points3D_rgb"]


def _setup_pipeline(
    config: TrainerConfig, device: str, grad_scaler: GradScaler
) -> Tuple[DirectPipeline, Optimizers]:
    datamanager = cast(
        FullImageDatamanager,
        config.pipeline.datamanager.setup(
            device=device,
            test_mode="val",
            world_size=1,
            local_rank=0,
        ),
    )
    assert datamanager.train_dataset is not None, "Missing train dataset."

    model = cast(
        SplatfactoModel,
        config.pipeline.model.setup(
            scene_box=datamanager.train_dataset.scene_box,
            num_train_data=len(datamanager.train_dataset),
            metadata=datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=_get_seed_points(datamanager),
        ),
    )
    model.to(device)

    pipeline = DirectPipeline(datamanager=datamanager, model=model)
    param_groups = {**datamanager.get_param_groups(), **model.get_param_groups()}
    optimizers = Optimizers(config.optimizers.copy(), param_groups)
    return pipeline, optimizers


def _pipeline_state_dict(model: SplatfactoModel) -> Dict[str, torch.Tensor]:
    return {f"model.{key}": value for key, value in model.state_dict().items()}


def _load_model_checkpoint(model: SplatfactoModel, loaded_state: Dict[str, torch.Tensor], step: int) -> None:
    state = {}
    for key, value in loaded_state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("_model."):
            key = key[len("_model.") :]
        elif key.startswith("model."):
            key = key[len("model.") :]
        state[key] = value
    model.update_to_step(step)
    model.load_state_dict(state, strict=False)


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
    _load_model_checkpoint(pipeline.model, loaded_state["pipeline"], loaded_state["step"])
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
            "pipeline": _pipeline_state_dict(pipeline.model),
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
        pipeline=cast(Any, pipeline),
        trainer=cast(Any, session_state),
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
    grad_scaler: GradScaler,
    session_state: SessionState,
    viewer: Optional[Viewer],
    train_lock: Lock,
    start_step: int,
) -> None:
    datamanager = pipeline.datamanager
    model = pipeline.model
    # 训练开始前先保存数据解析器的相机变换，方便后续导出和复现。
    if hasattr(datamanager, "train_dataparser_outputs"):
        datamanager.train_dataparser_outputs.save_dataparser_transform(output_dir / "dataparser_transforms.json")

    # 读取梯度累积配置；如果没有配置，就按每个参数组都不累积处理。
    gradient_accumulation_steps = config.gradient_accumulation_steps or {}
    with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
        # 主训练循环：从恢复步数开始，直到达到最大迭代次数。
        for step in range(start_step, config.max_num_iterations):
            # 如果训练被暂停，就在这里等待恢复。
            while session_state.training_state == "paused":
                time.sleep(0.01)

            # 用锁保护训练与 viewer 之间共享状态，避免并发读写冲突。
            with train_lock:
                with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                    # 切换到训练模式，并让模型根据当前步数更新内部状态。
                    model.train()
                    model.prepare_train_step(optimizers, step)

                    # 需要清零梯度的参数组：到达累积周期起点时清一次。
                    needs_zero = [
                        group
                        for group in optimizers.parameters.keys()
                        if step % gradient_accumulation_steps.get(group, 1) == 0
                    ]
                    optimizers.zero_grad_some(needs_zero)

                    # 前向与反向计算：根据设备类型选择 autocast 设备。
                    device_type = model.device.type
                    autocast_device = "cpu" if device_type == "mps" else device_type
                    with torch.autocast(device_type=autocast_device, enabled=config.mixed_precision):
                        camera, batch = datamanager.next_train(step)
                        outputs = model(camera)
                        metrics_dict = model.get_metrics_dict(outputs, batch)
                        loss_dict = model.get_loss_dict(outputs, batch, metrics_dict)
                        loss = sum(loss_dict.values())

                    # 反向传播，把损失梯度传回模型参数。
                    grad_scaler.scale(loss).backward()  # type: ignore[arg-type]

                    # 到达累积周期末尾时，执行优化器 step。
                    needs_step = [
                        group
                        for group in optimizers.parameters.keys()
                        if step % gradient_accumulation_steps.get(group, 1) == gradient_accumulation_steps.get(group, 1) - 1
                    ]
                    optimizers.optimizer_scaler_step_some(grad_scaler, needs_step)

                    # 更新 GradScaler；如果没有发生溢出，再推进学习率调度器。
                    scale = grad_scaler.get_scale()
                    grad_scaler.update()
                    if scale <= grad_scaler.get_scale():
                        optimizers.scheduler_step_all(step)

                    # 让模型完成这一轮训练后的收尾处理。
                    model.finish_train_step(step)

            # 统计训练速度，按帧/秒或射线/秒记录。
            if step > 1:
                writer.put_time(
                    name=EventName.TRAIN_RAYS_PER_SEC,
                    duration=datamanager.get_train_rays_per_batch() / max(0.001, train_t.duration),
                    step=step,
                    avg_over_steps=True,
                )

            # 如果开启 viewer，则刷新当前场景显示。
            if viewer is not None:
                viewer.update_scene(step, datamanager.get_train_rays_per_batch())

            # 按日志周期记录训练损失和指标。
            if step_check(step, config.logging.steps_per_log, run_at_zero=True):
                writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                if "psnr" in metrics_dict:
                    writer.put_scalar(name="PSNR", scalar=metrics_dict["psnr"], step=step)
                if "gaussian_count" in metrics_dict:
                    writer.put_scalar(name="Gaussian Count", scalar=metrics_dict["gaussian_count"], step=step)
                if torch.cuda.is_available():
                    writer.put_scalar(
                        name="GPU Memory (MB)",
                        scalar=torch.cuda.max_memory_allocated() / (1024**2),
                        step=step,
                    )

            # 按保存周期写出 checkpoint。
            if step_check(step, config.steps_per_save):
                _save_checkpoint(
                    pipeline=pipeline,
                    optimizers=optimizers,
                    grad_scaler=grad_scaler,
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    save_only_latest=config.save_only_latest_checkpoint,
                )

            # 将本轮写入的日志刷新到磁盘。
            writer.write_out_storage()

    # 训练结束后更新状态并保存最终 checkpoint。
    session_state.training_state = "completed"
    _save_checkpoint(
        pipeline=pipeline,
        optimizers=optimizers,
        grad_scaler=grad_scaler,
        checkpoint_dir=checkpoint_dir,
        step=config.max_num_iterations - 1,
        save_only_latest=config.save_only_latest_checkpoint,
    )
    # 输出最终训练总结信息。
    writer.write_out_storage()
    table = Table(title=None, show_header=False, box=box.MINIMAL, title_style=style.Style(bold=True))
    table.add_row("Config File", str(output_dir / "config.yml"))
    table.add_row("Checkpoint Directory", str(checkpoint_dir))
    CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))
    # 如果 viewer 还在运行，就通知训练已完成；必要时保持进程不退出。
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
    pipeline, optimizers = _setup_pipeline(config=config, device=device, grad_scaler=grad_scaler)
    _ensure_dino_features_ready(args=args, pipeline=pipeline)

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
