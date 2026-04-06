from __future__ import annotations

import functools
import json
import os
import re
import time
import webbrowser
from collections import defaultdict
from numbers import Integral
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, DefaultDict, Dict, Literal, Optional, Tuple, cast

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.text import Text

from hmgs.callbacks import TrainingCallbackAttributes, TrainingCallbackLocation
from hmgs.console import CONSOLE
from hmgs.misc import step_check
from hmgs.optimizers import Optimizers
from hmgs.pipeline import VanillaPipeline
from nerfstudio.utils.writer import GLOBAL_BUFFER
from nerfstudio.viewer.viewer import Viewer as ViewerState

if TYPE_CHECKING:
    from hmgs.config import HMGSTrainConfig

TRAIN_ITERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]


class LossColumn(ProgressColumn):
    """Renders the current loss value."""

    def render(self, task: Task) -> Text:
        loss = task.fields.get("loss", 0.0)
        return Text(f"loss={loss:.6f}", style="cyan")


class GaussiansColumn(ProgressColumn):
    """Renders the gaussian count."""

    def render(self, task: Task) -> Text:
        gaussians = task.fields.get("gaussians", 0)
        return Text(f"gaussians={gaussians}", style="green")


class RaysPerSecColumn(ProgressColumn):
    """Renders rays per second."""

    def render(self, task: Task) -> Text:
        rays_per_sec = task.fields.get("rays_per_sec", 0.0)
        return Text(f"rays/s={rays_per_sec:.0f}", style="yellow")


class _ViewerCamerasAdapter:
    """Adapter to match Nerfstudio viewer expectations for scalar camera indexing."""

    def __init__(self, cameras) -> None:
        self._cameras = cameras

    def __len__(self) -> int:
        return len(self._cameras)

    def __getitem__(self, item):
        camera = self._cameras[item]
        if isinstance(item, Integral) and getattr(camera.camera_to_worlds, "ndim", 0) == 3 and camera.camera_to_worlds.shape[0] == 1:
            camera.camera_to_worlds = camera.camera_to_worlds[0]
            camera.fx = camera.fx.reshape(-1)
            camera.fy = camera.fy.reshape(-1)
            camera.cx = camera.cx.reshape(-1)
            camera.cy = camera.cy.reshape(-1)
            camera.width = camera.width.reshape(-1)
            camera.height = camera.height.reshape(-1)
            camera.camera_type = camera.camera_type.reshape(-1)
            if camera.distortion_params is not None and camera.distortion_params.ndim > 1:
                camera.distortion_params = camera.distortion_params[0]
        return camera


class _ViewerDatasetAdapter:
    """Adapter that keeps HMGS dataset interface compatible with Nerfstudio viewer code."""

    def __init__(self, dataset) -> None:
        self._dataset = dataset
        self.cameras = _ViewerCamerasAdapter(dataset.cameras)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]


class HMGSTrainer:
    pipeline: VanillaPipeline
    optimizers: Optimizers

    def __init__(self, config: "HMGSTrainConfig", local_rank: int = 0, world_size: int = 1) -> None:
        self.train_lock = Lock()
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = config.machine.device_type
        if self.device == "cuda":
            self.device = f"cuda:{local_rank}"
        self.mixed_precision = config.mixed_precision and self.device != "cpu"
        self.use_grad_scaler = self.mixed_precision or config.use_grad_scaler
        self.gradient_accumulation_steps: DefaultDict[str, int] = defaultdict(lambda: 1)
        self.gradient_accumulation_steps.update(self.config.gradient_accumulation_steps)
        self.grad_scaler = GradScaler(enabled=self.use_grad_scaler)
        self.base_dir = config.get_base_dir()
        self.checkpoint_dir = config.get_checkpoint_dir()
        self._start_step = 0
        self.step = 0
        self.viewer_state: Optional[ViewerState] = None
        self.viewer_port: Optional[int] = None
        self.viewer_url: Optional[str] = None
        self.training_state: Literal["training", "paused", "completed"] = "training"
        self.stop_training = False
        GLOBAL_BUFFER.setdefault("max_iter", self.config.max_num_iterations)
        GLOBAL_BUFFER.setdefault("max_buffer_size", 20)
        GLOBAL_BUFFER.setdefault("steps_per_log", self.config.logging_steps)
        GLOBAL_BUFFER.setdefault("step", 0)
        if "events" not in GLOBAL_BUFFER:
            GLOBAL_BUFFER["events"] = {}

    @staticmethod
    def _parse_viewer_runtime_info(messages: list[str]) -> Tuple[Optional[str], Optional[int]]:
        for message in messages:
            url_match = re.search(r"(http://[^\s]+)", message)
            if url_match:
                url = url_match.group(1)
                port_match = re.search(r":(\d+)$", url)
                port = int(port_match.group(1)) if port_match else None
                return url, port
        return None, None

    def _setup_viewer(self) -> None:
        if self.local_rank != 0 or not self.config.viewer.enabled:
            return

        viewer_log_path = self.base_dir / self.config.viewer.relative_log_filename
        datapath = self.config.data if self.config.data is not None else self.base_dir
        try:
            self.viewer_state = ViewerState(
                self.config.viewer,
                log_filename=viewer_log_path,
                datapath=datapath,
                pipeline=self.pipeline,
                trainer=cast(Any, self),
                train_lock=self.train_lock,
                share=self.config.viewer.make_share_url,
            )
        except Exception as exc:
            self.viewer_state = None
            CONSOLE.print(f"Failed to initialize Nerfstudio viewer: {exc}")
            return

        for message in self.viewer_state.viewer_info:
            CONSOLE.print(message)
        self.viewer_url, self.viewer_port = self._parse_viewer_runtime_info(self.viewer_state.viewer_info)
        if self.viewer_port is not None:
            CONSOLE.print(f"Viser port: {self.viewer_port}")

        if self.config.viewer.open_browser and self.viewer_url is not None:
            try:
                webbrowser.open(self.viewer_url, new=2)
            except Exception as exc:
                CONSOLE.print(f"Could not auto-open browser for viewer URL: {exc}")

    def _init_viewer_state(self) -> None:
        if self.viewer_state is None:
            return
        assert self.pipeline.datamanager.train_dataset is not None
        train_dataset = _ViewerDatasetAdapter(self.pipeline.datamanager.train_dataset)
        eval_dataset = (
            _ViewerDatasetAdapter(self.pipeline.datamanager.eval_dataset)
            if self.pipeline.datamanager.eval_dataset is not None
            else None
        )
        self.viewer_state.init_scene(
            train_dataset=cast(Any, train_dataset),
            train_state=self.training_state,
            eval_dataset=cast(Any, eval_dataset),
        )

    def _update_viewer_state(self, step: int) -> None:
        if self.viewer_state is None:
            return
        num_rays_per_batch = self.pipeline.datamanager.get_train_rays_per_batch()
        try:
            self.viewer_state.update_scene(step, num_rays_per_batch)
        except RuntimeError:
            time.sleep(0.03)
            CONSOLE.log("Viewer failed. Continuing training.")

    def _train_complete_viewer(self) -> None:
        if self.viewer_state is None:
            return
        try:
            self.viewer_state.training_complete()
        except RuntimeError:
            time.sleep(0.03)
            CONSOLE.log("Viewer failed. Continuing training.")

    def setup(self) -> None:
        self.pipeline = self.config.pipeline.setup(
            device=self.device,
            test_mode="val",
            world_size=self.world_size,
            local_rank=self.local_rank,
            grad_scaler=self.grad_scaler,
        )
        self.optimizers = Optimizers(self.config.optimizers.copy(), self.pipeline.get_param_groups())
        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,
                grad_scaler=self.grad_scaler,
                pipeline=self.pipeline,
                trainer=self,
            )
        )
        self._load_checkpoint()
        self._setup_viewer()

    def save_config(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.base_dir / "config.json"
        config_path.write_text(json.dumps(self.config.as_dict(), indent=2, default=str), encoding="utf8")

    def train(self) -> None:
        outputs_path = self.base_dir
        outputs_path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.pipeline.datamanager, "train_dataparser_outputs"):
            self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
                self.base_dir / "dataparser_transforms.json"
            )

        self._init_viewer_state()

        if self.local_rank == 0 and self.viewer_port is not None:
            CONSOLE.print(f"Training started. Viser port: {self.viewer_port}")
            if self.viewer_url is not None:
                CONSOLE.print(f"Training started. Viser URL: {self.viewer_url}")

        # Create progress bar
        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(show_speed=True),
            LossColumn(),
            GaussiansColumn(),
            RaysPerSecColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
        )

        total_iterations = self.config.max_num_iterations - self._start_step

        start_time = time.perf_counter()

        with progress:
            task_id = progress.add_task(
                "Training",
                total=total_iterations,
                completed=self._start_step,
                loss=0.0,
                gaussians=0,
                rays_per_sec=0.0,
            )

            for iteration in range(self._start_step, self.config.max_num_iterations):
                self.step = iteration

                if self.stop_training:
                    break
                while self.training_state == "paused":
                    if self.stop_training:
                        break
                    time.sleep(0.01)
                if self.stop_training:
                    break

                with self.train_lock:
                    for callback in self.callbacks:
                        callback.run_callback_at_location(iteration, TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)
                    iter_start = time.perf_counter()
                    loss, _loss_dict, metrics_dict = self.train_iteration(iteration)
                    for callback in self.callbacks:
                        callback.run_callback_at_location(iteration, TrainingCallbackLocation.AFTER_TRAIN_ITERATION)

                self._update_viewer_state(iteration)

                current_step = iteration - self._start_step + 1

                if step_check(iteration, self.config.logging_steps, run_at_zero=True):
                    duration = max(1e-6, time.perf_counter() - iter_start)
                    rays_per_sec = self.world_size * self.pipeline.datamanager.get_train_rays_per_batch() / duration
                    gaussian_count = metrics_dict.get("gaussian_count", torch.tensor(0))
                    gaussian_count_int = int(gaussian_count.item()) if isinstance(gaussian_count, torch.Tensor) else int(gaussian_count)
                    progress.update(
                        task_id,
                        completed=current_step,
                        loss=float(loss),
                        gaussians=gaussian_count_int,
                        rays_per_sec=rays_per_sec,
                    )

                if step_check(iteration, self.config.steps_per_eval_image):
                    self.eval_iteration(iteration)
                if step_check(iteration, self.config.steps_per_save):
                    self.save_checkpoint(iteration)

        self._after_train(start_time)

    def _after_train(self, start_time: float) -> None:
        self.training_state = "completed"
        self.save_checkpoint(self.step)
        total_time = time.perf_counter() - start_time
        CONSOLE.print(f"Training finished in {total_time:.2f}s")
        CONSOLE.print(f"Config: {self.base_dir / 'config.json'}")
        CONSOLE.print(f"Checkpoints: {self.checkpoint_dir}")
        self._train_complete_viewer()
        if self.viewer_state is not None:
            self.viewer_state.viser_server.stop()
            CONSOLE.print("Viser server stopped.")
        for callback in self.callbacks:
            callback.run_callback_at_location(self.step, TrainingCallbackLocation.AFTER_TRAIN)

    def train_iteration(self, step: int) -> TRAIN_ITERATION_OUTPUT:
        needs_zero = [group for group in self.optimizers.parameters.keys() if step % self.gradient_accumulation_steps[group] == 0]
        self.optimizers.zero_grad_some(needs_zero)
        device_type = "cpu" if self.device.startswith("mps") else self.device.split(":")[0]
        with torch.autocast(device_type=device_type, enabled=self.mixed_precision):
            _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
            loss = functools.reduce(torch.add, loss_dict.values())
        self.grad_scaler.scale(loss).backward()
        needs_step = [
            group
            for group in self.optimizers.parameters.keys()
            if step % self.gradient_accumulation_steps[group] == self.gradient_accumulation_steps[group] - 1
        ]
        self.optimizers.optimizer_scaler_step_some(self.grad_scaler, needs_step)
        self.grad_scaler.update()
        self.optimizers.scheduler_step_all()
        return loss, loss_dict, metrics_dict

    def eval_iteration(self, step: int) -> None:
        if not self.pipeline.datamanager.eval_dataset:
            return
        metrics_dict, _ = self.pipeline.get_eval_image_metrics_and_images(step)
        scalar_metrics = ", ".join(
            f"{key}={value:.4f}" for key, value in metrics_dict.items() if isinstance(value, (int, float))
        )
        CONSOLE.print(f"[eval] step={step} {scalar_metrics}")

    def _load_checkpoint(self) -> None:
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                load_step = sorted(int(name[name.find("-") + 1 : name.find(".")]) for name in os.listdir(load_dir))[-1]
            load_checkpoint = load_dir / f"step-{load_step:09d}.ckpt"
        if load_checkpoint is None:
            return
        loaded_state = torch.load(load_checkpoint, map_location="cpu")
        self._start_step = loaded_state["step"] + 1
        self.pipeline.model.load_state_dict(loaded_state["model"], strict=False)
        self.optimizers.load_optimizers(loaded_state["optimizers"])
        if "schedulers" in loaded_state and self.config.load_scheduler:
            self.optimizers.load_schedulers(loaded_state["schedulers"])
        if "scaler" in loaded_state:
            self.grad_scaler.load_state_dict(loaded_state["scaler"])
        CONSOLE.print(f"Loaded checkpoint: {load_checkpoint}")

    def save_checkpoint(self, step: int) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "model": self.pipeline.model.state_dict(),
                "optimizers": {key: value.state_dict() for key, value in self.optimizers.optimizers.items()},
                "schedulers": {key: value.state_dict() for key, value in self.optimizers.schedulers.items()},
                "scaler": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        if self.config.save_only_latest_checkpoint:
            for file in self.checkpoint_dir.glob("*.ckpt"):
                if file != ckpt_path:
                    file.unlink()
