# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

"""
Datamanager.
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from itertools import islice
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
)

import fpsample
import numpy as np
import torch
from rich.progress import track
from torch.nn import Parameter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import TypeVar, assert_never

from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig
from nerfstudio.data.datasets import InputDataset, DinoInputDataset
from nerfstudio.data.pixel_samplers import PatchPixelSamplerConfig, PixelSampler, PixelSamplerConfig
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
    variable_res_collate,
    ImageBatchStream,
    _undistort_image,
)
from nerfstudio.data.utils.data_utils import identity_collate
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.misc import IterableWrapper, get_dict_to_torch, get_orig_class
from nerfstudio.utils.rich_utils import CONSOLE


# ── TDataset helper ───────────────────────────────────────────────────────────

TDataset = TypeVar("TDataset", bound=InputDataset, default=InputDataset)


# ═══════════════════════════════════════════════════════════════════════════════
# DataManager (abstract base)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DataManagerConfig(InstantiateConfig):
    """Configuration for data manager instantiation."""

    _target: Type = field(default_factory=lambda: DataManager)
    """Target class to instantiate."""
    data: Optional[Path] = None
    """Source of data, may not be used by all models."""
    masks_on_gpu: bool = False
    """Process masks on GPU for speed at the expense of memory, if True."""
    images_on_gpu: bool = False
    """Process images on GPU for speed at the expense of memory, if True."""


class DataManager:
    """Generic data manager's abstract class.

    Usage:
    To get data, use the next_train and next_eval functions.
    This data manager's next_train and next_eval methods will return 2 things:

    1. 'rays': This will contain the rays or camera we are sampling, with latents and
        conditionals attached (everything needed at inference)
    2. A "batch" of auxiliary information: This will contain the mask, the ground truth
        pixels, etc needed to actually train, score, etc the model

    Train Methods:
        setup_train: sets up for being used as train
        iter_train: will be called on __iter__() for the train iterator
        next_train: will be called on __next__() for the training iterator
        get_train_iterable: utility that gets a clean pythonic iterator for your training data

    Eval Methods:
        setup_eval: sets up for being used as eval
        iter_eval: will be called on __iter__() for the eval iterator
        next_eval: will be called on __next__() for the eval iterator
        get_eval_iterable: utility that gets a clean pythonic iterator for your eval data

    Attributes:
        train_count (int): the step number of our train iteration, needs to be incremented manually
        eval_count (int): the step number of our eval iteration, needs to be incremented manually
        train_dataset (Dataset): the dataset for the train dataset
        eval_dataset (Dataset): the dataset for the eval dataset
        includes_time (bool): whether the dataset includes time information
    """

    train_dataset: Optional[InputDataset] = None
    eval_dataset: Optional[InputDataset] = None
    train_sampler: Optional[DistributedSampler] = None
    eval_sampler: Optional[DistributedSampler] = None
    includes_time: bool = False
    test_mode: Literal["test", "val", "inference"] = "val"

    def __init__(self):
        super().__init__()
        self.train_count = 0
        self.eval_count = 0
        if self.train_dataset and self.test_mode != "inference":
            self.setup_train()
        if self.eval_dataset and self.test_mode != "inference":
            self.setup_eval()

    def forward(self):
        raise NotImplementedError

    def iter_train(self):
        self.train_count = 0

    def iter_eval(self):
        self.eval_count = 0

    def get_train_iterable(self, length=-1) -> IterableWrapper:
        return IterableWrapper(self.iter_train, self.next_train, length)

    def get_eval_iterable(self, length=-1) -> IterableWrapper:
        return IterableWrapper(self.iter_eval, self.next_eval, length)

    @abstractmethod
    def setup_train(self):
        """Sets up the data manager for training."""

    @abstractmethod
    def setup_eval(self):
        """Sets up the data manager for evaluation."""

    @abstractmethod
    def next_train(self, step: int) -> Tuple[Union[RayBundle, Cameras], Dict]:
        raise NotImplementedError

    @abstractmethod
    def next_eval(self, step: int) -> Tuple[Union[RayBundle, Cameras], Dict]:
        raise NotImplementedError

    @abstractmethod
    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        raise NotImplementedError

    @abstractmethod
    def get_train_rays_per_batch(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_eval_rays_per_batch(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_datapath(self) -> Path:
        """Returns the path to the data."""

    @abstractmethod
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# VanillaDataManager (ray-based, used by legacy scripts)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class VanillaDataManagerConfig(DataManagerConfig):
    """A basic data manager for a ray-based model."""

    _target: Type = field(default_factory=lambda: VanillaDataManager)
    dataparser: ColmapDataParserConfig = field(default_factory=ColmapDataParserConfig)
    cache_images_type: Literal["uint8", "float32"] = "float32"
    train_num_rays_per_batch: int = 1024
    train_num_images_to_sample_from: Union[int, float] = float("inf")
    train_num_times_to_repeat_images: Union[int, float] = float("inf")
    eval_num_rays_per_batch: int = 1024
    eval_num_images_to_sample_from: Union[int, float] = float("inf")
    eval_num_times_to_repeat_images: Union[int, float] = float("inf")
    eval_image_indices: Optional[Tuple[int, ...]] = (0,)
    collate_fn: Callable[[Any], Any] = cast(Any, staticmethod(nerfstudio_collate))
    camera_res_scale_factor: float = 1.0
    patch_size: int = 1
    camera_optimizer: Optional[CameraOptimizerConfig] = field(default=None)
    pixel_sampler: PixelSamplerConfig = field(default_factory=PixelSamplerConfig)

    def __post_init__(self):
        if self.camera_optimizer is not None:
            import warnings
            CONSOLE.print(
                "\nCameraOptimizerConfig has been moved from the DataManager to the Model.\n", style="bold yellow"
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)


class VanillaDataManager(DataManager, Generic[TDataset]):
    """Basic stored data manager implementation for ray-based models."""

    config: VanillaDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset
    train_dataparser_outputs: DataparserOutputs
    train_pixel_sampler: Optional[PixelSampler] = None
    eval_pixel_sampler: Optional[PixelSampler] = None

    def __init__(
        self,
        config: VanillaDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")

        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        self.exclude_batch_keys_from_device = self.train_dataset.exclude_batch_keys_from_device
        if self.config.masks_on_gpu is True and "mask" in self.exclude_batch_keys_from_device:
            self.exclude_batch_keys_from_device.remove("mask")
        if self.config.images_on_gpu is True and "image" in self.exclude_batch_keys_from_device:
            self.exclude_batch_keys_from_device.remove("image")

        if self.train_dataparser_outputs is not None:
            cameras = self.train_dataparser_outputs.cameras
            if len(cameras) > 1:
                for i in range(1, len(cameras)):
                    if cameras[0].width != cameras[i].width or cameras[0].height != cameras[i].height:
                        CONSOLE.print("Variable resolution, using variable_res_collate")
                        self.config.collate_fn = variable_res_collate
                        break
        super().__init__()

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[VanillaDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is VanillaDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is VanillaDataManager:
            return get_args(orig_class)[0]
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is VanillaDataManager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    def create_train_dataset(self) -> TDataset:
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
        )

    def create_eval_dataset(self) -> TDataset:
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
        )

    def _get_pixel_sampler(self, dataset: TDataset, num_rays_per_batch: int) -> PixelSampler:
        if self.config.patch_size > 1 and type(self.config.pixel_sampler) is PixelSamplerConfig:
            return PatchPixelSamplerConfig().setup(
                patch_size=self.config.patch_size, num_rays_per_batch=num_rays_per_batch
            )
        is_equirectangular = (dataset.cameras.camera_type == CameraType.EQUIRECTANGULAR.value).all()
        if is_equirectangular.any():
            CONSOLE.print("[bold yellow]Warning: Some cameras are equirectangular, but using default pixel sampler.")
        fisheye_crop_radius = None
        if dataset.cameras.metadata is not None:
            fisheye_crop_radius = dataset.cameras.metadata.get("fisheye_crop_radius")
        return self.config.pixel_sampler.setup(
            is_equirectangular=is_equirectangular,
            num_rays_per_batch=num_rays_per_batch,
            fisheye_crop_radius=fisheye_crop_radius,
        )

    def setup_train(self):
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))

    def setup_eval(self):
        assert self.eval_dataset is not None
        CONSOLE.print("Setting up evaluation dataset...")
        self.eval_image_dataloader = CacheDataloader(
            self.eval_dataset,
            num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
            exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
        )
        self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
        self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
        self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device))
        self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)
        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        self.eval_count += 1
        image_batch = next(self.iter_eval_image_dataloader)
        assert self.eval_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.eval_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.eval_ray_generator(ray_indices)
        return ray_bundle, batch

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        for camera, batch in self.eval_dataloader:
            assert camera.shape[0] == 1
            return camera, batch
        raise ValueError("No more eval images")

    def get_train_rays_per_batch(self) -> int:
        if self.train_pixel_sampler is not None:
            return self.train_pixel_sampler.num_rays_per_batch
        return self.config.train_num_rays_per_batch

    def get_eval_rays_per_batch(self) -> int:
        if self.eval_pixel_sampler is not None:
            return self.eval_pixel_sampler.num_rays_per_batch
        return self.config.eval_num_rays_per_batch

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# FullImageDatamanager
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FullImageDatamanagerConfig(DataManagerConfig):
    _target: Type = field(default_factory=lambda: FullImageDatamanager)
    dataparser: ColmapDataParserConfig = field(default_factory=ColmapDataParserConfig)
    camera_res_scale_factor: float = 1.0
    eval_num_images_to_sample_from: int = -1
    eval_num_times_to_repeat_images: int = -1
    cache_images: Literal["cpu", "gpu", "disk"] = "gpu"
    cache_images_type: Literal["uint8", "float32"] = "float32"
    max_thread_workers: Optional[int] = None
    train_cameras_sampling_strategy: Literal["random", "fps"] = "random"
    train_cameras_sampling_seed: int = 42
    fps_reset_every: int = 100
    dataloader_num_workers: int = 4
    prefetch_factor: Optional[int] = 4
    cache_compressed_images: bool = False


class FullImageDatamanager(DataManager, Generic[TDataset]):
    """A datamanager that outputs full images and cameras instead of raybundles."""

    config: FullImageDatamanagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset

    def __init__(
        self,
        config: FullImageDatamanagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        if config.cache_images == "disk":
            try:
                torch.multiprocessing.set_start_method("spawn")
            except RuntimeError:
                assert torch.multiprocessing.get_start_method() == "spawn", 'start method must be "spawn"'
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = "test" if test_mode in ["test", "inference"] else "val"
        self.dataparser_config = self.config.dataparser
        if self.config.data is not None:
            self.config.dataparser.data = Path(self.config.data)
        else:
            self.config.data = self.config.dataparser.data
        self.dataparser = self.dataparser_config.setup()
        if test_mode == "inference":
            self.dataparser.downscale_factor = 1
        self.includes_time = self.dataparser.includes_time
        self.train_dataparser_outputs: DataparserOutputs = self.dataparser.get_dataparser_outputs(split="train")
        self.train_dataset = self.create_train_dataset()
        self.eval_dataset = self.create_eval_dataset()
        if len(self.train_dataset) > 500 and self.config.cache_images == "gpu":
            CONSOLE.print(
                "Train dataset has over 500 images, overriding cache_images to cpu. "
                "If you still get OOM errors or segfault, please consider setting cache_images to 'disk'",
                style="bold yellow",
            )
            self.config.cache_images = "cpu"

        self.train_unseen_cameras = self.sample_train_cameras()
        self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        assert len(self.train_unseen_cameras) > 0, "No data found in dataset"
        super().__init__()

    def sample_train_cameras(self):
        num_train_cameras = len(self.train_dataset)
        if self.config.train_cameras_sampling_strategy == "random":
            if not hasattr(self, "random_generator"):
                self.random_generator = random.Random(self.config.train_cameras_sampling_seed)
            indices = list(range(num_train_cameras))
            self.random_generator.shuffle(indices)
            return indices
        elif self.config.train_cameras_sampling_strategy == "fps":
            if not hasattr(self, "train_unsampled_epoch_count"):
                np.random.seed(self.config.train_cameras_sampling_seed)
                self.train_unsampled_epoch_count = np.zeros(num_train_cameras)
            camera_origins = self.train_dataset.cameras.camera_to_worlds[..., 3].numpy()
            data = np.concatenate(
                (camera_origins, 0.1 * np.expand_dims(self.train_unsampled_epoch_count, axis=-1)), axis=-1
            )
            n = self.config.fps_reset_every
            if num_train_cameras < n:
                CONSOLE.log(
                    f"num_train_cameras={num_train_cameras} is smaller than fps_reset_ever={n}, "
                    "the behavior of camera sampler will be very similar to sampling random without replacement."
                )
                n = num_train_cameras
            kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(data, n, h=3)
            self.train_unsampled_epoch_count += 1
            self.train_unsampled_epoch_count[kdline_fps_samples_idx] = 0
            return kdline_fps_samples_idx.tolist()
        else:
            raise ValueError(f"Unknown train camera sampling strategy: {self.config.train_cameras_sampling_strategy}")

    @cached_property
    def cached_train(self) -> List[Dict[str, torch.Tensor]]:
        assert self.config.cache_images != "disk", "Can not call _load_images() with `disk` as input"
        return self._load_images("train", cache_images_device=self.config.cache_images)

    @cached_property
    def cached_eval(self) -> List[Dict[str, torch.Tensor]]:
        assert self.config.cache_images != "disk", "Can not call _load_images() with `disk` as input"
        return self._load_images("eval", cache_images_device=self.config.cache_images)

    def _load_images(
        self, split: Literal["train", "eval"], cache_images_device: Literal["cpu", "gpu"]
    ) -> List[Dict[str, torch.Tensor]]:
        undistorted_images: List[Dict[str, torch.Tensor]] = []
        if split == "train":
            dataset = self.train_dataset
        elif split == "eval":
            dataset = self.eval_dataset
        else:
            assert_never(split)

        def undistort_idx(idx: int) -> Dict[str, torch.Tensor]:
            data = dataset.get_data(idx, image_type=self.config.cache_images_type)
            camera = dataset.cameras[idx].reshape(())
            # If the loaded image resolution doesn't match the camera intrinsics, adjust the camera
            # intrinsics to match the actually loaded image instead of failing. This can happen when
            # images on disk have been externally downscaled / resaved compared to COLMAP camera params.
            img_h = int(data["image"].shape[0])
            img_w = int(data["image"].shape[1])
            cam_w = int(camera.width.item())
            cam_h = int(camera.height.item())
            if img_w != cam_w or img_h != cam_h:
                CONSOLE.print(
                    f"[bold yellow]Warning: Loaded image resolution ({img_w}, {img_h}) does not match "
                    f"camera resolution ({cam_w}, {cam_h}). Adjusting camera intrinsics to image size.[/bold yellow]"
                )
                # Scale intrinsics to the new resolution per-axis
                # Use float ratios to adjust fx/fy/cx/cy appropriately.
                try:
                    scale_w = img_w / cam_w if cam_w != 0 else 1.0
                    scale_h = img_h / cam_h if cam_h != 0 else 1.0
                    dataset.cameras.fx[idx] = float(dataset.cameras.fx[idx].item() * scale_w)
                    dataset.cameras.fy[idx] = float(dataset.cameras.fy[idx].item() * scale_h)
                    dataset.cameras.cx[idx] = float(dataset.cameras.cx[idx].item() * scale_w)
                    dataset.cameras.cy[idx] = float(dataset.cameras.cy[idx].item() * scale_h)
                    dataset.cameras.width[idx] = img_w
                    dataset.cameras.height[idx] = img_h
                    # Refresh local camera reference to reflect updates
                    camera = dataset.cameras[idx].reshape(())
                except Exception:
                    CONSOLE.print("[bold red]Failed to adjust camera intrinsics automatically.[/bold red]")
                    raise
            if camera.distortion_params is None or torch.all(camera.distortion_params == 0):
                return data
            K = camera.get_intrinsics_matrices().numpy()
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()
            K, image, mask = _undistort_image(camera, distortion_params, data, image, K)
            data["image"] = torch.from_numpy(image)
            if mask is not None:
                data["mask"] = mask
            dataset.cameras.fx[idx] = float(K[0, 0])
            dataset.cameras.fy[idx] = float(K[1, 1])
            dataset.cameras.cx[idx] = float(K[0, 2])
            dataset.cameras.cy[idx] = float(K[1, 2])
            dataset.cameras.width[idx] = image.shape[1]
            dataset.cameras.height[idx] = image.shape[0]
            return data

        CONSOLE.log(f"Caching / undistorting {split} images")
        with ThreadPoolExecutor(max_workers=2) as executor:
            undistorted_images = list(
                track(
                    executor.map(undistort_idx, range(len(dataset))),
                    description=f"Caching / undistorting {split} images",
                    transient=True,
                    total=len(dataset),
                )
            )
        if cache_images_device == "gpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
                if "depth" in cache:
                    cache["depth"] = cache["depth"].to(self.device)
                self.train_cameras = self.train_dataset.cameras.to(self.device)
        elif cache_images_device == "cpu":
            for cache in undistorted_images:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()
                self.train_cameras = self.train_dataset.cameras
        else:
            assert_never(cache_images_device)
        return undistorted_images

    def create_train_dataset(self) -> TDataset:
        return self.dataset_type(
            dataparser_outputs=self.train_dataparser_outputs,
            scale_factor=self.config.camera_res_scale_factor,
            cache_compressed_images=self.config.cache_compressed_images,
        )

    def create_eval_dataset(self) -> TDataset:
        return self.dataset_type(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=self.test_split),
            scale_factor=self.config.camera_res_scale_factor,
            cache_compressed_images=self.config.cache_compressed_images,
        )

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[FullImageDatamanager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is FullImageDatamanager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is FullImageDatamanager:
            return get_args(orig_class)[0]
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is FullImageDatamanager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    def get_datapath(self) -> Path:
        return self.config.dataparser.data

    def setup_train(self):
        if self.config.cache_images == "disk":
            self.train_imagebatch_stream = ImageBatchStream(
                input_dataset=self.train_dataset,
                sampling_seed=self.config.train_cameras_sampling_seed,
                cache_images_type=self.config.cache_images_type,
                device=self.device,
                custom_image_processor=self.custom_image_processor,
            )
            self.train_image_dataloader = DataLoader(
                self.train_imagebatch_stream,
                batch_size=1,
                num_workers=self.config.dataloader_num_workers,
                collate_fn=identity_collate,
            )
            self.iter_train_image_dataloader = iter(self.train_image_dataloader)

    def setup_eval(self):
        if self.config.cache_images == "disk":
            self.eval_imagebatch_stream = ImageBatchStream(
                input_dataset=self.eval_dataset,
                sampling_seed=self.config.train_cameras_sampling_seed,
                cache_images_type=self.config.cache_images_type,
                device=self.device,
                custom_image_processor=self.custom_image_processor,
            )
            self.eval_image_dataloader = DataLoader(
                self.eval_imagebatch_stream,
                batch_size=1,
                num_workers=0,
                collate_fn=identity_collate,
            )
            self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        if self.config.cache_images == "disk":
            dataloader = DataLoader(
                self.eval_imagebatch_stream,
                batch_size=1,
                num_workers=0,
                collate_fn=lambda x: x[0],
            )
            return list(islice(dataloader, len(self.eval_dataset)))
        image_indices = [i for i in range(len(self.eval_dataset))]
        data = [d.copy() for d in self.cached_eval]
        _cameras = deepcopy(self.eval_dataset.cameras).to(self.device)
        cameras = []
        for i in image_indices:
            data[i]["image"] = data[i]["image"].to(self.device)
            cameras.append(_cameras[i : i + 1])
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        return list(zip(cameras, data))

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        return {}

    def get_train_rays_per_batch(self) -> int:
        camera = self.train_dataset.cameras[0].reshape(())
        return int(camera.width[0].item() * camera.height[0].item())

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        self.train_count += 1
        if self.config.cache_images == "disk":
            camera, data = next(self.iter_train_image_dataloader)[0]
            camera = camera.to(self.device)
            data = get_dict_to_torch(data, self.device)
            return camera, data

        image_idx = self.train_unseen_cameras.pop(0)
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = self.sample_train_cameras()

        data = self.cached_train[image_idx]
        data = data.copy()
        data["image"] = data["image"].to(self.device)

        assert len(self.train_cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        self.eval_count += 1
        if self.config.cache_images == "disk":
            camera, data = next(self.iter_eval_image_dataloader)[0]
            camera = camera.to(self.device)
            data = get_dict_to_torch(data, self.device)
            return camera, data
        return self.next_eval_image(step=step)

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        if self.config.cache_images == "disk":
            camera, data = next(self.iter_eval_image_dataloader)[0]
            return camera, data
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]
        data = self.cached_eval[image_idx]
        data = data.copy()
        data["image"] = data["image"].to(self.device)
        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data

    def custom_image_processor(self, camera: Cameras, data: Dict) -> Tuple[Cameras, Dict]:
        return camera, data


# ═══════════════════════════════════════════════════════════════════════════════
# DINO feature extraction helper
# ═══════════════════════════════════════════════════════════════════════════════


def _ensure_dino_features(
    source_path: Path,
    dino_feature_dir: Path,
    feature_dim: int,
    device: str,
) -> None:
    """Auto-extract DINO features if the feature file doesn't exist or is incomplete."""
    from nerfstudio.data.dataparsers.colmap_dataparser import ColmapDataParserConfig

    # scripts/ isn't a package — load extract_dino_features.py by path
    _project_root = Path(__file__).resolve().parent.parent.parent  # repo root from nerfstudio/data/
    _extract_script = _project_root / "scripts" / "extract_dino_features.py"
    _spec = importlib.util.spec_from_file_location("_extract_dino_features", _extract_script)
    _extract_module = importlib.util.module_from_spec(_spec)
    sys.modules["_extract_dino_features"] = _extract_module
    _spec.loader.exec_module(_extract_module)

    # Minimal dataparser setup to discover image paths
    dataparser_cfg = ColmapDataParserConfig(data=source_path, load_3D_points=True)
    dataparser = dataparser_cfg.setup()
    train_outputs = dataparser.get_dataparser_outputs(split="train")
    eval_outputs = dataparser.get_dataparser_outputs(split="val")

    all_images = sorted(set(list(train_outputs.image_filenames) + list(eval_outputs.image_filenames)))
    if len(all_images) == 0:
        return

    # Compute image_root — same logic as DinoInputDataset
    image_parent_paths = [str(p.parent) for p in all_images]
    image_root = Path(os.path.commonpath(image_parent_paths))

    output_file = _extract_module._resolve_output_file(dino_feature_dir)

    needs_extraction = not output_file.exists()
    if not needs_extraction:
        try:
            existing, _ = _extract_module._load_existing_payload(output_file, feature_dim)
            for img_path in all_images:
                try:
                    key = img_path.relative_to(image_root).with_suffix(".pt").as_posix()
                except ValueError:
                    needs_extraction = True
                    break
                if key not in existing:
                    needs_extraction = True
                    break
        except Exception:
            needs_extraction = True

    if needs_extraction:
        CONSOLE.log(f"[bold yellow]DINO features missing or incomplete, auto-extracting to {output_file}...")
        saved, skipped = _extract_module.extract_dino_features_for_images(
            image_paths=all_images,
            input_dir=image_root,
            output_dir=dino_feature_dir,
            feature_dim=feature_dim,
            device=device,
            skip_existing=True,
        )
        CONSOLE.log(f"DINO feature extraction done: saved={saved}, skipped={skipped}")


def _ensure_style_primitives(
    style_image_path: Path,
    output_path: Path,
    num_primitives: int,
    device: str,
) -> None:
    """Extract style texture prototypes from a 2D style image via DINOv2 + K-Means.

    Follows the Phase-2 decomposition in the architecture doc:
    1. Extract DINOv2 patch-level features from the style image
    2. L2-normalize and K-Means cluster into *num_primitives* prototypes
    3. Save the L2-normalized prototypes to *output_path* as a .pt file

    The saved tensor ``style_primitives.pt`` contains a dict with key
    ``"prototypes"`` mapping to a ``(K, D)`` float32 tensor.
    """
    if output_path.exists():
        CONSOLE.log(f"Style primitives already exist at {output_path}, skipping extraction.")
        return

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms

    target_device = torch.device(device if torch.cuda.is_available() else "cpu")
    CONSOLE.log(f"[bold yellow]Extracting {num_primitives} style primitives from {style_image_path} ...")

    # ── 1. Load DINOv2 ──────────────────────────────────────────────────────
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(target_device)
    model.eval()

    # ── 2. Preprocess ───────────────────────────────────────────────────────
    patch_size = 14
    image_size = max(patch_size, (560 // patch_size) * patch_size)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_pil = Image.open(style_image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(target_device)

    # ── 3. Extract patch-level features ─────────────────────────────────────
    with torch.no_grad():
        features_dict = model.forward_features(img_tensor)
        features = features_dict["x_norm_patchtokens"]  # (1, N, D)
    features = features.squeeze(0)  # (N, D)
    features_norm = F.normalize(features, p=2, dim=-1)

    # ── 4. K-Means to get K prototypes ──────────────────────────────────────
    # Pure-PyTorch K-Means (same algorithm as dino_mask.py)
    N, D = features_norm.shape
    indices = torch.randperm(N)[:num_primitives]
    centroids = features_norm[indices].clone()

    for _ in range(50):
        dists = torch.cdist(features_norm, centroids, p=2)  # (N, K)
        labels = torch.argmin(dists, dim=1)
        new_centroids = []
        for i in range(num_primitives):
            mask = labels == i
            if mask.sum() > 0:
                new_centroids.append(features_norm[mask].mean(dim=0))
            else:
                new_centroids.append(features_norm[torch.randint(0, N, (1,))].squeeze(0))
        centroids = torch.stack(new_centroids)

    prototypes = F.normalize(centroids, p=2, dim=-1)

    # ── 5. Save ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prototypes": prototypes.cpu()}, output_path)
    CONSOLE.log(f"Style primitives saved to {output_path} (K={num_primitives}, D={D})")


# ═══════════════════════════════════════════════════════════════════════════════
# DinoDatamanager
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DinoDatamanagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory=lambda: DinoDatamanager)
    dino_features_dir: Optional[Path] = None
    dino_feature_dim: int = 16
    strict_dino_loading: bool = True
    style_image_path: Optional[Path] = None
    style_primitive_path: Optional[Path] = None
    num_style_primitives: int = 5


class DinoDatamanager(FullImageDatamanager[DinoInputDataset]):
    """Full-image datamanager that carries precomputed DINO feature maps in each batch."""

    def __init__(
        self,
        config: DinoDatamanagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        # Auto-extract DINO features BEFORE parent init creates datasets
        dino_feature_dir = (
            Path(config.dino_features_dir)
            if config.dino_features_dir is not None
            else Path(config.dataparser.data) / "dino_features"
        )
        _ensure_dino_features(
            source_path=Path(config.dataparser.data),
            dino_feature_dir=dino_feature_dir,
            feature_dim=config.dino_feature_dim,
            device=str(device),
        )
        # Auto-extract style texture primitives (if a style image was provided)
        if config.style_image_path is not None and config.style_primitive_path is not None:
            _ensure_style_primitives(
                style_image_path=config.style_image_path,
                output_path=config.style_primitive_path,
                num_primitives=config.num_style_primitives,
                device=str(device),
            )
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            **kwargs,
        )

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
