#!/usr/bin/env python
"""
SemGeo Splatfacto — Pipeline 2 entry point.

Loads a dino-splatfacto checkpoint, runs structural decomposition (clustering),
and visualizes results in Viser.

Usage:
    python train_semgeo.py -s /path/to/colmap_scene --load-checkpoint step-000029999.ckpt
"""

from __future__ import annotations

import argparse
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from rich import box, style
from rich.panel import Panel
from rich.table import Table
from torch.cuda.amp.grad_scaler import GradScaler

from configs.base_config import ViewerConfig
from data.datamanager import DinoDatamanagerConfig, FullImageDatamanager
from data.dataparser import ColmapDataParserConfig
from engine.optimizers import AdamOptimizerConfig
from engine.pipeline import VanillaPipelineConfig
from engine.schedulers import ExponentialDecaySchedulerConfig
from engine.trainer import TrainerConfig
from scene.semgeo_gaussian_model import SemGeoSplatfactoModel, SemGeoSplatfactoModelConfig
from utils import profiler, writer
from utils.system_utils import CONSOLE, get_available_devices
from viewer.viewer import Viewer

# Shared training infrastructure
from train import (
    SessionState,
    _setup_viewer,
    _set_random_seed,
    _save_config,
    _load_model_checkpoint,
    _setup_runtime_logging,
)

# Speed up static-shape CUDA kernels.
torch.backends.cudnn.benchmark = True  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════
# semgeo pipeline — config and entry point
# ═══════════════════════════════════════════════════════════════════════════════


def _build_config(args: argparse.Namespace) -> TrainerConfig:
    scene_name = args.source_path.resolve().name
    method_name = "semgeo-splatfacto"

    dino_feature_dir = (
        args.dino_feature_dir
        if args.dino_feature_dir is not None
        else args.model_path.resolve() / "dino-splatfacto" / scene_name / "dino_features"
    )
    datamanager_config = DinoDatamanagerConfig(
        dataparser=ColmapDataParserConfig(data=args.source_path, load_3D_points=True),
        cache_images=args.data_device,
        cache_images_type="uint8",
        dino_features_dir=dino_feature_dir,
        dino_feature_dim=args.dino_feature_dim,
        strict_dino_loading=False,  # Clustering can work with missing features (zeros)
    )
    model_config = SemGeoSplatfactoModelConfig(
        dino_feature_dim=args.dino_feature_dim,
        # Use minimal random init — checkpoint replaces params anyway.
        # Avoids allocating gaussians for all SfM seed points (100k+).
        random_init=True,
        num_random=100,
        geo_weight=args.geo_weight,
        dino_weight=args.dino_weight,
        k_base=args.k_base,
        alpha_radius=args.alpha_radius,
        k_max=args.k_max,
        semantic_gate=args.semantic_gate,
        tau_feat=args.tau_feat,
        tau_normal=args.tau_normal,
        gamma_spatial=args.gamma_spatial,
        min_cluster_size=args.min_cluster_size,
        cdist_chunk_size=2048,  # Lower peak memory during KNN
    )

    optimizers_dict = {
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
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-7, max_steps=30000),
        },
        "bilateral_grid": {
            "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=30000),
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
        steps_per_eval_image=0,
        steps_per_eval_batch=0,
        steps_per_save=0,
        steps_per_eval_all_images=0,
        max_num_iterations=1,  # Not training — just load + cluster
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(datamanager=datamanager_config, model=model_config),
        optimizers=optimizers_dict,
        viewer=viewer_config,
        vis=cast(Any, "none" if args.disable_viewer or args.vis == "none" else "viewer"),
    )
    config.machine.seed = args.seed
    config.machine.device_type = args.device
    config.logging.steps_per_log = 10
    return config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SemGeo Splatfacto — load dino checkpoint, cluster, visualize."
    )
    parser.add_argument("-s", "--source-path", type=Path, required=True, help="COLMAP dataset root.")
    parser.add_argument(
        "-m", "--model-path", type=Path,
        default=Path("/mnt/data2/experiments/3dgs/nerfstudio/output"),
        help="Output directory for config.",
    )
    parser.add_argument("--load-checkpoint", type=Path, required=True, help="Dino checkpoint to load.")
    parser.add_argument("--viewer-port", type=int, default=None, help="Viewer websocket port override.")
    parser.add_argument("--disable-viewer", action="store_true", help="Disable Viser viewer.")
    parser.add_argument("--vis", choices=["viewer", "none"], default="viewer", help="Visualization mode.")
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda", help="Device.")
    parser.add_argument("--data-device", choices=["cpu", "gpu", "disk"], default="gpu", help="Image cache target.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    # DINO feature args
    parser.add_argument("--dino-feature-dir", type=Path, default=None,
                        help="Directory containing dino_features.pt.")
    parser.add_argument("--dino-feature-dim", type=int, default=16, help="DINO feature dimensionality.")
    # Clustering hyper-params
    parser.add_argument("--geo-weight", type=float, default=1.0)
    parser.add_argument("--dino-weight", type=float, default=0.8)
    parser.add_argument("--k-base", type=int, default=8)
    parser.add_argument("--alpha-radius", type=float, default=2.0)
    parser.add_argument("--k-max", type=int, default=32)
    parser.add_argument("--semantic-gate", type=float, default=0.6)
    parser.add_argument("--tau-feat", type=float, default=0.75)
    parser.add_argument("--tau-normal", type=float, default=0.7)
    parser.add_argument("--gamma-spatial", type=float, default=1.2)
    parser.add_argument("--min-cluster-size", type=int, default=100)
    return parser.parse_args()


def _add_cluster_visualization(viewer: Viewer, model: SemGeoSplatfactoModel) -> None:
    """Add cluster-colored point clouds to the Viser viewer scene.

    Uses the same colormap as ``model._cluster_colormap`` so static point
    clouds match the ``cluster_rgb`` rendered output.
    """
    if model.cluster_labels is None:
        return

    labels = model.cluster_labels.cpu()
    means = model.means.detach().cpu().numpy()
    scale = 10.0  # VISER_NERFSTUDIO_SCALE_RATIO

    unique_labels = sorted(set(int(l.item()) for l in torch.unique(labels) if l >= 0))
    num_clusters = len(unique_labels)
    CONSOLE.log(f"Adding {num_clusters} cluster point clouds to viewer...")

    # Use model's colormap if available, otherwise generate HSV colors
    cmap = model._cluster_colormap
    if cmap is not None and cmap.shape[0] >= num_clusters:
        colors_rgb_float = cmap.cpu().numpy()
    else:
        import colorsys
        colors_hsv = [(i / max(num_clusters, 1), 0.85, 1.0) for i in range(num_clusters)]
        colors_rgb_float = np.array([colorsys.hsv_to_rgb(h, s, v) for h, s, v in colors_hsv])

    # Scale to 0-255 uint8 for viser
    colors_rgb = [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors_rgb_float]

    for i, label in enumerate(unique_labels):
        mask = (labels == label).numpy()
        pts = means[mask] * scale

        # Subsample large clusters for display performance
        max_display = 100000
        if pts.shape[0] > max_display:
            idx = np.random.RandomState(42).choice(pts.shape[0], max_display, replace=False)
            pts = pts[idx]

        viewer.viser_server.scene.add_point_cloud(
            f"/semgeo_clusters/cluster_{label:04d}",
            points=pts,
            colors=colors_rgb[i % len(colors_rgb)],
            point_size=0.03,
            point_shape="circle",
        )

    CONSOLE.log(
        "[bold green]Cluster visualization ready.[/bold green]\n"
        "  • Static point clouds: /semgeo_clusters/ in scene tree\n"
        "  • Rendered view: select 'cluster_rgb' in Control > Output Render\n"
        "  • Normal view: select 'rgb' in Control > Output Render"
    )


def _setup_semgeo_model_and_datamanager(
    config: TrainerConfig, device: str, grad_scaler: GradScaler
) -> Tuple[FullImageDatamanager, SemGeoSplatfactoModel]:
    """Create datamanager + model WITHOUT optimizers (inference-only pipeline)."""
    from engine.pipeline import VanillaPipeline
    pipeline = VanillaPipeline(
        config=config.pipeline,
        device=device,
        test_mode="val",
        world_size=1,
        local_rank=0,
        grad_scaler=grad_scaler,
    )
    datamanager = cast(FullImageDatamanager, pipeline.datamanager)
    model = cast(SemGeoSplatfactoModel, pipeline.model)
    model.to(device)
    return datamanager, model


def run_semgeo_loop(
    model: SemGeoSplatfactoModel,
    viewer: Optional[Viewer],
) -> None:
    """Pipeline 2 main loop: cluster → visualize → wait.

    This is the "run_train_loop" equivalent for semgeo — it runs the
    clustering step (管线二 阶段一) and then keeps the viewer open.
    """
    # ── Step 1: Show clustering status in Viser ───────────────────────────
    if viewer is not None:
        viewer.viser_server.gui.add_markdown(
            "**🔬 SemGeo clustering in progress...**  \n"
            "Check the terminal for progress. Clusters will appear here when done."
        )

    # ── Step 2: Clustering ────────────────────────────────────────────────
    CONSOLE.log("[bold cyan]Running structural decomposition...")
    model.run_clustering()

    # ── Step 3: Add cluster visualization to viewer ────────────────────────
    if viewer is not None:
        _add_cluster_visualization(viewer, model)
        # Force re-render so "cluster_rgb" appears in Output Type dropdown
        viewer._trigger_rerender()

    # ── Step 4: Keep viewer open ──────────────────────────────────────────
    CONSOLE.print(Panel(
        "[bold green]Clustering complete![/bold green]\n"
        "• Static point clouds: /semgeo_clusters/ in Viser scene tree\n"
        "• Rendered view: select 'cluster_rgb' in Control panel → Output Render\n"
        "• Normal view: select 'rgb' in Control panel → Output Render\n"
        "Press Ctrl+C to quit.",
        title="[bold]SemGeo Viewer[/bold]",
        expand=False,
    ))

    if viewer is not None:
        while True:
            time.sleep(0.01)
    else:
        CONSOLE.log("No viewer — exiting.")

    writer.close_local_writer()


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    output_dir = config.get_base_dir()
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
    CONSOLE.log(f"Output directory: {output_dir}")

    train_lock = Lock()
    session_state = SessionState(training_state="completed")  # Not training
    grad_scaler = GradScaler(enabled=False)

    CONSOLE.log("Setting up model and datamanager (inference-only, no optimizers)...")
    datamanager, model = _setup_semgeo_model_and_datamanager(
        config=config, device=device, grad_scaler=grad_scaler,
    )

    # ── Load checkpoint (required) ─────────────────────────────────────────
    checkpoint_path = args.load_checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    CONSOLE.log(f"Loading checkpoint: {checkpoint_path}")
    loaded_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _load_model_checkpoint(model, loaded_state["pipeline"], loaded_state["step"])
    model.eval()

    # Free training-only components — we only need gauss_params for clustering + rendering.
    model._strip_training_components()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    CONSOLE.log(f"Loaded checkpoint from step {loaded_state['step']}")
    N = model.means.shape[0]
    gpu_mem = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    CONSOLE.log(f"Gaussians: {N:,}  |  GPU memory: {gpu_mem:.2f} GB")

    # ── Setup viewer ───────────────────────────────────────────────────────
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

    try:
        run_semgeo_loop(model=model, viewer=viewer)
    except KeyboardInterrupt:
        CONSOLE.print(traceback.format_exc())
    finally:
        writer.close_local_writer()
        if viewer is not None:
            viewer.viser_server.stop()
        profiler.flush_profiler(config.logging)


if __name__ == "__main__":
    main()
