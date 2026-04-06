#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import socket
import sys
from datetime import timedelta
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Allow running via `python hmgs/train.py` in addition to `python -m hmgs.train`.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from hmgs.config import HMGSTrainConfig
from hmgs.console import CONSOLE
from hmgs.trainer import HMGSTrainer

DEFAULT_TIMEOUT = timedelta(minutes=30)
torch.backends.cudnn.benchmark = True  # type: ignore


def _find_free_port() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return str(port)


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_loop(local_rank: int, world_size: int, config: HMGSTrainConfig, global_rank: int = 0) -> None:
    _set_random_seed(config.machine.seed + global_rank)
    trainer = HMGSTrainer(config=config, local_rank=local_rank, world_size=world_size)
    trainer.save_config()
    trainer.setup()
    trainer.train()


def _distributed_worker(
    local_rank: int,
    main_func: Callable,
    world_size: int,
    num_devices_per_machine: int,
    machine_rank: int,
    dist_url: str,
    config: HMGSTrainConfig,
    timeout: timedelta = DEFAULT_TIMEOUT,
) -> None:
    assert torch.cuda.is_available(), "CUDA is not available."
    global_rank = machine_rank * num_devices_per_machine + local_rank
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=global_rank,
        timeout=timeout,
    )
    try:
        main_func(local_rank, world_size, config, global_rank)
    finally:
        dist.destroy_process_group()


def launch(main_func: Callable, config: HMGSTrainConfig) -> None:
    world_size = config.machine.num_machines * config.machine.num_devices
    if world_size == 1:
        main_func(local_rank=0, world_size=1, config=config)
        return
    if config.machine.device_type != "cuda":
        raise RuntimeError("Multi-process HMGS training currently requires CUDA.")
    dist_url = config.machine.dist_url
    if dist_url == "auto":
        if config.machine.num_machines != 1:
            raise ValueError("dist_url=auto is only supported for a single machine.")
        dist_url = f"tcp://127.0.0.1:{_find_free_port()}"
    process_context = mp.spawn(
        _distributed_worker,
        nprocs=config.machine.num_devices,
        join=False,
        args=(
            main_func,
            world_size,
            config.machine.num_devices,
            config.machine.machine_rank,
            dist_url,
            config,
            DEFAULT_TIMEOUT,
        ),
    )
    assert process_context is not None
    process_context.join()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone HMGS trainer")
    parser.add_argument("--data", "-d", type=Path, required=True, help="COLMAP project root containing images/ and sparse/")
    parser.add_argument("--output-dir", "-o", type=Path, required=True, help="Training output directory")
    parser.add_argument(
        "--preset",
        type=str,
        default="splatfacto",
        choices=["splatfacto", "splatfacto-dino", "splatfacto-big", "splatfacto-mcmc"],
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--steps-per-save", type=int, default=None)
    parser.add_argument("--steps-per-eval-image", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--num-devices", type=int, default=1)
    parser.add_argument("--eval-mode", type=str, default=None, choices=["fraction", "filename", "interval", "all"])
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--train-split-fraction", type=float, default=None)
    parser.add_argument("--cache-images", type=str, default=None, choices=["cpu", "gpu"])
    parser.add_argument("--load-dir", type=Path, default=None)
    parser.add_argument("--load-step", type=int, default=None)
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--viewer",
        dest="viewer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable viser viewer server (enabled by default).",
    )
    parser.add_argument("--viewer-host", type=str, default=None, help="Host for viser server.")
    parser.add_argument("--viewer-port", type=int, default=None, help="Port for viser server.")
    parser.add_argument(
        "--viewer-open-browser",
        dest="viewer_open_browser",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Open viewer URL in browser when training starts.",
    )
    return parser


def build_config(args: argparse.Namespace) -> HMGSTrainConfig:
    config = HMGSTrainConfig.from_preset(args.preset, data=args.data, output_dir=args.output_dir)
    config.machine.seed = args.seed
    config.machine.num_devices = args.num_devices
    if args.device is not None:
        config.machine.device_type = args.device  # type: ignore[assignment]
    elif torch.cuda.is_available():
        config.machine.device_type = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        config.machine.device_type = "mps"
    else:
        config.machine.device_type = "cpu"
    if args.max_iterations is not None:
        config.max_num_iterations = args.max_iterations
        for optimizer_group in config.optimizers.values():
            scheduler = optimizer_group["scheduler"]
            if scheduler is not None and hasattr(scheduler, "max_steps"):
                scheduler.max_steps = args.max_iterations
    if args.steps_per_save is not None:
        config.steps_per_save = args.steps_per_save
    if args.steps_per_eval_image is not None:
        config.steps_per_eval_image = args.steps_per_eval_image
    if args.logging_steps is not None:
        config.logging_steps = args.logging_steps
    if args.eval_mode is not None:
        config.pipeline.datamanager.dataparser.eval_mode = args.eval_mode
    if args.eval_interval is not None:
        config.pipeline.datamanager.dataparser.eval_interval = args.eval_interval
    if args.train_split_fraction is not None:
        config.pipeline.datamanager.dataparser.train_split_fraction = args.train_split_fraction
    if args.cache_images is not None:
        config.pipeline.datamanager.cache_images = args.cache_images
    if args.viewer is not None:
        config.viewer.enabled = args.viewer
    if args.viewer_host is not None:
        config.viewer.websocket_host = args.viewer_host
    if args.viewer_port is not None:
        config.viewer.websocket_port = args.viewer_port
    if args.viewer_open_browser is not None:
        config.viewer.open_browser = args.viewer_open_browser
    config.load_dir = args.load_dir
    config.load_step = args.load_step
    config.load_checkpoint = args.load_checkpoint
    return config.finalize()


def main(cli_args: Optional[list[str]] = None) -> None:
    parser = build_argparser()
    args = parser.parse_args(cli_args)
    config = build_config(args)
    CONSOLE.print(f"HMGS preset: {config.preset_name}")
    CONSOLE.print(f"Dataset root: {config.data}")
    CONSOLE.print(f"Output dir: {config.output_dir}")
    launch(train_loop, config)


if __name__ == "__main__":
    main()
