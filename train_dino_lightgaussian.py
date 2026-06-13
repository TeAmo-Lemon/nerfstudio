#!/usr/bin/env python
"""
3D Gaussian Splatting + DINO features + LightGaussian compression.

Combines DINO feature splatting with LightGaussian's importance-based pruning
and optional SH distillation for compressed, feature-aware 3DGS.

Usage:
    # DINO + LightGaussian training
    python train_dino_lightgaussian.py -s /path/to/colmap_scene

    # With SH distillation
    python train_dino_lightgaussian.py -s /path/to/colmap_scene --enable-sh-distillation

    # Custom DINO and prune schedule
    python train_dino_lightgaussian.py -s /path/to/colmap_scene \\
        --dino-feature-dim 32 --dino-loss-weight 0.5 \\
        --prune-iterations 15000 22000 --prune-percent 0.4
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from threading import Lock
from typing import Any, List, Literal, Optional, cast

import torch
from torch.cuda.amp.grad_scaler import GradScaler

from configs.base_config import ViewerConfig
from data.datamanager import DinoDatamanagerConfig, FullImageDatamanager
from data.dataparser import ColmapDataParserConfig
from engine.optimizers import AdamOptimizerConfig, Optimizers
from engine.schedulers import ExponentialDecaySchedulerConfig
from engine.trainer import TrainerConfig
from engine.pipeline import VanillaPipelineConfig
from scene.dino_lightgaussian_model import DinoLightGaussianModel, DinoLightGaussianModelConfig
from utils import profiler, writer
from utils.system_utils import CONSOLE, get_available_devices
from utils.writer import EventName, TimeWriter
from viewer.viewer import Viewer

# Shared training infrastructure — imported from train.py
from train import (
    SessionState,
    _setup_viewer,
    _set_random_seed,
    _save_config,
    _get_seed_points,
    _setup_model_and_datamanager,
    _pipeline_state_dict,
    _load_model_checkpoint,
    _setup_runtime_logging,
    _find_latest_checkpoint,
    _load_checkpoint,
    _save_checkpoint,
    _resolve_checkpoint_path,
    run_train_loop,
)

# Speed up static-shape CUDA kernels.
torch.backends.cudnn.benchmark = True  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════
# dino-lightgaussian pipeline — config and entry point
# ═══════════════════════════════════════════════════════════════════════════════


def _build_config(args: argparse.Namespace) -> TrainerConfig:
    scene_name = args.source_path.resolve().name
    method_name = "dino-lightgaussian"

    dino_feature_dir = (
        args.dino_feature_dir
        if args.dino_feature_dir is not None
        else args.model_path.resolve() / method_name / scene_name / "dino_features"
    )
    style_primitive_path = (
        args.model_path.resolve() / method_name / scene_name / "style_primitives.pt"
    )
    datamanager_config = DinoDatamanagerConfig(
        dataparser=ColmapDataParserConfig(data=args.source_path, load_3D_points=True),
        cache_images=args.data_device,
        cache_images_type="uint8",
        dino_features_dir=dino_feature_dir,
        dino_feature_dim=args.dino_feature_dim,
        strict_dino_loading=True,
        style_image_path=args.style_image,
        style_primitive_path=style_primitive_path,
        num_style_primitives=args.num_style_primitives,
    )
    model_config = DinoLightGaussianModelConfig(
        # DINO
        dino_feature_dim=args.dino_feature_dim,
        dino_loss_weight=args.dino_loss_weight,
        dino_loss_start_step=args.dino_loss_start_step,
        style_primitive_path=style_primitive_path,
        num_style_primitives=args.num_style_primitives,
        # LightGaussian
        prune_iterations=args.prune_iterations,
        prune_percent=args.prune_percent,
        prune_decay=args.prune_decay,
        v_pow=args.v_pow,
        prune_type=args.prune_type,
        num_importance_views=args.num_importance_views,
        enable_sh_distillation=args.enable_sh_distillation,
        distill_start_iter=args.distill_start_iter,
        distill_sh_degree=args.distill_sh_degree,
        freeze_covariance_during_distill=args.freeze_covariance_during_distill,
        distill_loss_weight=args.distill_loss_weight,
    )

    optimizers_dict = {
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
                lr_final=5e-7, max_steps=args.max_steps, warmup_steps=1000, lr_pre_warmup=0,
            ),
        },
        "bilateral_grid": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1e-4, max_steps=args.max_steps, warmup_steps=1000, lr_pre_warmup=0,
            ),
        },
        "dino_features": {
            "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
            "scheduler": None,
        },
    }

    viewer_config = ViewerConfig(num_rays_per_chunk=1 << 15, websocket_port=args.viewer_port)
    config = TrainerConfig(
        method_name=method_name,
        experiment_name=scene_name,
        output_dir=args.model_path.resolve(),
        steps_per_eval_image=args.eval_image_every,
        steps_per_eval_batch=0,
        steps_per_save=args.save_every,
        steps_per_eval_all_images=args.eval_all_every,
        max_num_iterations=args.max_steps,
        mixed_precision=args.mixed_precision,
        pipeline=VanillaPipelineConfig(datamanager=datamanager_config, model=model_config),
        optimizers=optimizers_dict,
        viewer=viewer_config,
        vis=cast(Any, "none" if args.disable_viewer or args.vis == "none" else "viewer"),
    )
    config.machine.seed = args.seed
    config.machine.device_type = args.device
    config.logging.steps_per_log = args.steps_per_log
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3DGS + DINO features + LightGaussian compression — dino-lightgaussian pipeline."
    )
    # ── standard args (same as train.py) ────────────────────────────
    parser.add_argument("-s", "--source-path", type=Path, required=True, help="COLMAP dataset root.")
    parser.add_argument(
        "-m", "--model-path", type=Path,
        default=Path("/mnt/data2/experiments/3dgs/nerfstudio/output"),
        help="Output directory for config/checkpoints.",
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

    # ── DINO-specific args (same as train_dino.py) ──────────────────
    parser.add_argument(
        "--dino-feature-dir", type=Path, default=None,
        help="Directory containing precomputed DINO .pt features.",
    )
    parser.add_argument("--dino-feature-dim", type=int, default=16, help="DINO feature channel count after PCA.")
    parser.add_argument("--dino-loss-weight", type=float, default=0.05, help="Loss weight for DINO feature L1 loss.")
    parser.add_argument(
        "--dino-loss-start-step", type=int, default=3000,
        help="Enable DINO feature loss from this global training step.",
    )
    parser.add_argument(
        "--style-image", type=Path, default="./datasets/style_images/14.jpg",
        help="Path to a 2D style image for texture primitive decomposition.",
    )
    parser.add_argument(
        "--style-primitive-path", type=Path, default=None,
        help="Optional path to precomputed style_primitives.pt.",
    )
    parser.add_argument(
        "--num-style-primitives", type=int, default=5,
        help="Number of texture primitives to extract from the style image (K in K-means).",
    )

    # ── LightGaussian-specific args ─────────────────────────────────
    parser.add_argument(
        "--prune-iterations", nargs="+", type=int, default=[15000, 22000],
        help="Steps at which to run importance-based pruning.",
    )
    parser.add_argument(
        "--prune-percent", type=float, default=0.5,
        help="Fraction of Gaussians to prune on the first prune round.",
    )
    parser.add_argument(
        "--prune-decay", type=float, default=0.8,
        help="Decay factor for subsequent prune rounds (prune_pct * decay^i).",
    )
    parser.add_argument(
        "--v-pow", type=float, default=0.1,
        help="Volume power exponent for v_importance score adjustment.",
    )
    parser.add_argument(
        "--prune-type", choices=["importance", "v_importance"], default="v_importance",
        help="'importance' uses raw multi-view score; 'v_importance' multiplies by volume factor.",
    )
    parser.add_argument(
        "--num-importance-views", type=int, default=0,
        help="Number of training views for importance scoring (0 = all).",
    )
    parser.add_argument(
        "--enable-sh-distillation", action="store_true",
        help="Run SH distillation phase after main training.",
    )
    parser.add_argument(
        "--distill-start-iter", type=int, default=25000,
        help="Step at which SH distillation begins.",
    )
    parser.add_argument(
        "--distill-sh-degree", type=int, default=2,
        help="Target SH degree after distillation.",
    )
    parser.add_argument(
        "--freeze-covariance-during-distill", action="store_true",
        help="Freeze scales and rotations during distillation.",
    )
    parser.add_argument(
        "--distill-loss-weight", type=float, default=1.0,
        help="Weight for the distillation L1+SSIM loss.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    output_dir = config.get_base_dir()
    checkpoint_dir = config.get_checkpoint_dir()
    available_devices = get_available_devices()
    if config.machine.device_type not in available_devices:
        raise RuntimeError(
            f"Specified device type '{config.machine.device_type}' is not available. "
            f"Available device types: {available_devices}."
        )

    device = config.machine.device_type
    if device == "cuda":
        device = "cuda:0"
    _set_random_seed(config.machine.seed)
    _save_config(config, output_dir)
    CONSOLE.log(f"Saving checkpoints to: {checkpoint_dir}")

    train_lock = Lock()
    session_state = SessionState(
        training_state="paused" if config.start_paused else "training",
    )
    grad_scaler = GradScaler(enabled=config.mixed_precision or config.use_grad_scaler)
    datamanager, model, optimizers = _setup_model_and_datamanager(config=config, device=device, grad_scaler=grad_scaler)

    # LightGaussian-specific: give model access to training cameras for importance scoring
    model.set_train_cameras(datamanager.train_dataset.cameras)

    checkpoint_path = _resolve_checkpoint_path(args, checkpoint_dir)

    _PipelineRef = type("_PipelineRef", (), {"datamanager": datamanager, "model": model})
    viewer = _setup_viewer(
        config=config,
        output_dir=output_dir,
        source_path=args.source_path.resolve(),
        pipeline_ref=_PipelineRef,
        session_state=session_state,
        train_lock=train_lock,
    )
    banner_messages = viewer.viewer_info if viewer is not None else None
    _setup_runtime_logging(config, output_dir, banner_messages)

    start_step = 0
    if checkpoint_path is not None:
        start_step = _load_checkpoint(
            model=model, optimizers=optimizers, grad_scaler=grad_scaler, checkpoint_path=checkpoint_path,
        )
        model.set_train_cameras(datamanager.train_dataset.cameras)
    else:
        CONSOLE.print("No checkpoint to load, training from scratch.")

    try:
        run_train_loop(
            config=config,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
            model=model,
            datamanager=datamanager,
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
        writer.close_local_writer()
        if viewer is not None:
            viewer.viser_server.stop()
        profiler.flush_profiler(config.logging)


if __name__ == "__main__":
    main()
