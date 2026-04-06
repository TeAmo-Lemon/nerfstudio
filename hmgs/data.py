from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Type

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from hmgs.camera_utils import auto_orient_and_center_poses, get_distortion_params
from hmgs.cameras import CAMERA_MODEL_TO_TYPE, CameraType, Cameras
from hmgs.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_cameras_text,
    read_images_binary,
    read_images_text,
    read_points3D_binary,
    read_points3D_text,
)
from hmgs.colmap_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
    parse_colmap_camera_params,
    resolve_colmap_project_path,
)
from hmgs.config_base import InstantiateConfig
from hmgs.console import CONSOLE
from hmgs.data_utils import get_image_mask_tensor_from_path, pil_to_numpy
from hmgs.scene_box import SceneBox


@dataclass
class DataparserOutputs:
    image_filenames: List[Path]
    cameras: Cameras
    scene_box: SceneBox
    mask_filenames: Optional[List[Path]] = None
    metadata: Dict = field(default_factory=dict)
    dataparser_transform: torch.Tensor = field(default_factory=lambda: torch.eye(4, dtype=torch.float32)[:3, :])
    dataparser_scale: float = 1.0

    def save_dataparser_transform(self, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "transform": self.dataparser_transform.tolist(),
                    "scale": float(self.dataparser_scale),
                },
                indent=4,
            ),
            encoding="utf8",
        )


@dataclass
class ColmapDataParserConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: ColmapDataParser)
    data: Path = Path()
    scale_factor: float = 1.0
    scene_scale: float = 1.0
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"
    center_method: Literal["poses", "focus", "none"] = "poses"
    auto_scale_poses: bool = True
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "interval"
    train_split_fraction: float = 0.9
    eval_interval: int = 8
    load_3D_points: bool = True
    images_path: Optional[Path] = None
    colmap_path: Optional[Path] = None


class ColmapDataParser:
    config: ColmapDataParserConfig

    def __init__(self, config: ColmapDataParserConfig) -> None:
        self.config = config
        self.project_root, self.images_dir, self.sparse_dir = resolve_colmap_project_path(config.data)
        if self.config.images_path is None:
            self.config.images_path = self.images_dir.relative_to(self.project_root)
        if self.config.colmap_path is None:
            self.config.colmap_path = self.sparse_dir.relative_to(self.project_root)

    def _get_all_images_and_cameras(self) -> Dict:
        recon_dir = self.sparse_dir
        if (recon_dir / "cameras.txt").exists():
            cam_id_to_camera = read_cameras_text(recon_dir / "cameras.txt")
            im_id_to_image = read_images_text(recon_dir / "images.txt")
        elif (recon_dir / "cameras.bin").exists():
            cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
            im_id_to_image = read_images_binary(recon_dir / "images.bin")
        else:
            raise ValueError(f"Could not find cameras.txt/cameras.bin in {recon_dir}")

        cameras = {}
        frames = []
        camera_model = None
        for cam_id, cam_data in cam_id_to_camera.items():
            cameras[cam_id] = parse_colmap_camera_params(cam_data)
        for im_id in sorted(im_id_to_image.keys()):
            im_data = im_id_to_image[im_id]
            rotation = qvec2rotmat(im_data.qvec)
            translation = im_data.tvec.reshape(3, 1)
            w2c = np.concatenate([rotation, translation], axis=1)
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], axis=0)
            c2w = np.linalg.inv(w2c)
            c2w[0:3, 1:3] *= -1
            c2w = c2w[np.array([0, 2, 1, 3]), :]
            c2w[2, :] *= -1
            frame = {
                "file_path": (self.images_dir / im_data.name).as_posix(),
                "transform_matrix": c2w,
                "colmap_im_id": im_id,
            }
            frame.update(cameras[im_data.camera_id])
            frames.append(frame)
            if camera_model is None:
                camera_model = frame["camera_model"]
        applied_transform = np.eye(4)[:3, :]
        applied_transform = applied_transform[np.array([0, 2, 1]), :]
        applied_transform[2, :] *= -1
        return {"frames": frames, "camera_model": camera_model, "applied_transform": applied_transform.tolist()}

    def _get_image_indices(self, image_filenames: List[Path], split: str) -> np.ndarray:
        if self.config.eval_mode == "fraction":
            i_train, i_eval = get_train_eval_split_fraction(image_filenames, self.config.train_split_fraction)
        elif self.config.eval_mode == "filename":
            i_train, i_eval = get_train_eval_split_filename(image_filenames)
        elif self.config.eval_mode == "interval":
            i_train, i_eval = get_train_eval_split_interval(image_filenames, self.config.eval_interval)
        elif self.config.eval_mode == "all":
            i_train, i_eval = get_train_eval_split_all(image_filenames)
        else:
            raise ValueError(f"Unknown eval mode: {self.config.eval_mode}")
        if split == "train":
            return i_train
        if split in ("val", "test"):
            return i_eval
        raise ValueError(f"Unknown split: {split}")

    def _load_3D_points(self, transform_matrix: torch.Tensor, scale_factor: float) -> Dict:
        if (self.sparse_dir / "points3D.bin").exists():
            colmap_points = read_points3D_binary(self.sparse_dir / "points3D.bin")
        elif (self.sparse_dir / "points3D.txt").exists():
            colmap_points = read_points3D_text(self.sparse_dir / "points3D.txt")
        else:
            raise ValueError(f"Could not find points3D.txt/points3D.bin in {self.sparse_dir}")
        points3D = torch.from_numpy(np.array([point.xyz for point in colmap_points.values()], dtype=np.float32))
        points3D = (torch.cat([points3D, torch.ones_like(points3D[..., :1])], dim=-1) @ transform_matrix.T) * scale_factor
        return {
            "points3D_xyz": points3D,
            "points3D_rgb": torch.from_numpy(np.array([point.rgb for point in colmap_points.values()], dtype=np.uint8)),
            "points3D_error": torch.from_numpy(np.array([point.error for point in colmap_points.values()], dtype=np.float32)),
            "points3D_num_points2D": torch.tensor([len(point.image_ids) for point in colmap_points.values()], dtype=torch.int64),
        }

    def get_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        meta = self._get_all_images_and_cameras()
        camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        image_filenames: List[Path] = []
        poses = []
        fx, fy, cx, cy, height, width = [], [], [], [], [], []
        distort = []

        for frame in meta["frames"]:
            image_filenames.append(Path(frame["file_path"]))
            poses.append(frame["transform_matrix"])
            fx.append(float(frame["fl_x"]))
            fy.append(float(frame["fl_y"]))
            cx.append(float(frame["cx"]))
            cy.append(float(frame["cy"]))
            height.append(int(frame["h"]))
            width.append(int(frame["w"]))
            distort.append(
                get_distortion_params(
                    k1=float(frame.get("k1", 0.0)),
                    k2=float(frame.get("k2", 0.0)),
                    k3=float(frame.get("k3", 0.0)),
                    k4=float(frame.get("k4", 0.0)),
                    p1=float(frame.get("p1", 0.0)),
                    p2=float(frame.get("p2", 0.0)),
                )
            )

        poses_tensor = torch.from_numpy(np.array(poses, dtype=np.float32))
        poses_tensor, transform_matrix = auto_orient_and_center_poses(
            poses_tensor,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses_tensor[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        poses_tensor[:, :3, 3] *= scale_factor

        indices = self._get_image_indices(image_filenames, split)
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        cameras = Cameras(
            camera_to_worlds=poses_tensor[idx_tensor, :3, :4],
            fx=torch.tensor(fx, dtype=torch.float32)[idx_tensor],
            fy=torch.tensor(fy, dtype=torch.float32)[idx_tensor],
            cx=torch.tensor(cx, dtype=torch.float32)[idx_tensor],
            cy=torch.tensor(cy, dtype=torch.float32)[idx_tensor],
            width=torch.tensor(width, dtype=torch.int64)[idx_tensor],
            height=torch.tensor(height, dtype=torch.int64)[idx_tensor],
            distortion_params=torch.stack(distort, dim=0)[idx_tensor],
            camera_type=int(camera_type),
        )

        scene_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [[-scene_scale, -scene_scale, -scene_scale], [scene_scale, scene_scale, scene_scale]],
                dtype=torch.float32,
            )
        )

        metadata = {}
        if self.config.load_3D_points:
            metadata.update(self._load_3D_points(transform_matrix, scale_factor))

        return DataparserOutputs(
            image_filenames=[image_filenames[i] for i in indices],
            cameras=cameras,
            scene_box=scene_box,
            metadata=metadata,
            dataparser_transform=transform_matrix,
            dataparser_scale=scale_factor,
        )


class InputDataset(Dataset):
    cameras: Cameras

    def __init__(self, dataparser_outputs: DataparserOutputs, cache_compressed_images: bool = False) -> None:
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cache_compressed_images = cache_compressed_images
        self.binary_images = None
        if cache_compressed_images:
            self.binary_images = []
            for image_filename in self._dataparser_outputs.image_filenames:
                with open(image_filename, "rb") as file:
                    self.binary_images.append(file.read())

    def __len__(self) -> int:
        return len(self._dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> np.ndarray:
        if self.binary_images is not None:
            import io

            pil_image = Image.open(io.BytesIO(self.binary_images[image_idx]))
        else:
            pil_image = Image.open(self._dataparser_outputs.image_filenames[image_idx])
        image = pil_to_numpy(pil_image)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        return image

    def get_image_float32(self, image_idx: int) -> torch.Tensor:
        image = self.get_numpy_image(image_idx).astype(np.float32) / 255.0
        return torch.from_numpy(image)

    def get_image_uint8(self, image_idx: int) -> torch.Tensor:
        return torch.from_numpy(self.get_numpy_image(image_idx))

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        image = self.get_image_uint8(image_idx) if image_type == "uint8" else self.get_image_float32(image_idx)
        data = {"image_idx": image_idx, "image": image}
        if self._dataparser_outputs.mask_filenames is not None:
            data["mask"] = get_image_mask_tensor_from_path(self._dataparser_outputs.mask_filenames[image_idx])
        return data

    def __getitem__(self, image_idx: int) -> Dict:
        return self.get_data(image_idx)


def _undistort_image(camera: Cameras, distortion_params: np.ndarray, data: Dict, image: np.ndarray, K: np.ndarray):
    mask = None
    camera_type = int(camera.camera_type.item())
    if camera_type == CameraType.PERSPECTIVE:
        distortion_params = np.array(
            [
                distortion_params[0],
                distortion_params[1],
                distortion_params[4],
                distortion_params[5],
                distortion_params[2],
                distortion_params[3],
                0,
                0,
            ]
        )
        K[0, 2] -= 0.5
        K[1, 2] -= 0.5
        if np.any(distortion_params):
            newK, roi = cv2.getOptimalNewCameraMatrix(K, distortion_params, (image.shape[1], image.shape[0]), 0)
            image = cv2.undistort(image, K, distortion_params, None, newK)
        else:
            newK = K
            roi = (0, 0, image.shape[1], image.shape[0])
        x, y, w, h = roi
        image = image[y : y + h, x : x + w]
        newK[0, 2] -= x
        newK[1, 2] -= y
        if "mask" in data:
            mask = data["mask"].numpy().astype(np.uint8) * 255
            if np.any(distortion_params):
                mask = cv2.undistort(mask, K, distortion_params, None, newK)
            mask = torch.from_numpy(mask[y : y + h, x : x + w]).bool()
            if len(mask.shape) == 2:
                mask = mask[:, :, None]
        newK[0, 2] += 0.5
        newK[1, 2] += 0.5
        return newK, image, mask

    if camera_type == CameraType.FISHEYE:
        K[0, 2] -= 0.5
        K[1, 2] -= 0.5
        distortion_params = np.array(
            [distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3]]
        )
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, distortion_params, (image.shape[1], image.shape[0]), np.eye(3), balance=0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, distortion_params, np.eye(3), newK, (image.shape[1], image.shape[0]), cv2.CV_32FC1
        )
        image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
        if "mask" in data:
            mask = data["mask"].numpy().astype(np.uint8) * 255
            mask = cv2.fisheye.undistortImage(mask, K, distortion_params, None, newK)
            mask = torch.from_numpy(mask).bool()
            if len(mask.shape) == 2:
                mask = mask[:, :, None]
        newK[0, 2] += 0.5
        newK[1, 2] += 0.5
        return newK, image, mask

    return K, image, mask


@dataclass
class FullImageDatamanagerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: FullImageDatamanager)
    data: Path = Path()
    dataparser: ColmapDataParserConfig = field(default_factory=ColmapDataParserConfig)
    cache_images: Literal["cpu", "gpu"] = "gpu"
    cache_images_type: Literal["uint8", "float32"] = "uint8"
    train_cameras_sampling_strategy: Literal["random", "fps"] = "random"
    train_cameras_sampling_seed: int = 42
    eval_num_images_to_sample_from: int = -1
    eval_num_times_to_repeat_images: int = -1
    cache_compressed_images: bool = False


class FullImageDatamanager:
    config: FullImageDatamanagerConfig

    def __init__(
        self,
        config: FullImageDatamanagerConfig,
        device: str = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ) -> None:
        del kwargs, world_size, local_rank
        self.config = config
        self.device = device
        self.test_mode = test_mode
        self.includes_time = False
        self.dataparser_config = deepcopy(config.dataparser)
        self.dataparser_config.data = config.data
        self.dataparser = self.dataparser_config.setup()
        self.train_dataparser_outputs = self.dataparser.get_dataparser_outputs(split="train")
        self.train_dataset = InputDataset(
            dataparser_outputs=self.train_dataparser_outputs,
            cache_compressed_images=self.config.cache_compressed_images,
        )
        eval_split = "test" if test_mode in ("test", "inference") else "val"
        self.eval_dataset = InputDataset(
            dataparser_outputs=self.dataparser.get_dataparser_outputs(split=eval_split),
            cache_compressed_images=self.config.cache_compressed_images,
        )
        if len(self.train_dataset) > 500 and self.config.cache_images == "gpu":
            CONSOLE.print("Dataset has over 500 images; forcing cache_images='cpu' to avoid GPU OOM.")
            self.config.cache_images = "cpu"
        self.train_cameras = self.train_dataset.cameras
        self.train_unseen_cameras = self.sample_train_cameras()
        self.eval_unseen_cameras = list(range(len(self.eval_dataset)))
        self.cached_train = self._load_images(self.train_dataset, split="train")
        self.cached_eval = self._load_images(self.eval_dataset, split="eval")

    def sample_train_cameras(self) -> List[int]:
        num_train_cameras = len(self.train_dataset)
        if self.config.train_cameras_sampling_strategy == "random":
            rng = random.Random(self.config.train_cameras_sampling_seed)
            indices = list(range(num_train_cameras))
            rng.shuffle(indices)
            return indices
        raise ValueError("HMGS standalone datamanager currently supports train_cameras_sampling_strategy='random' only.")

    def _load_images(self, dataset: InputDataset, split: str) -> List[Dict[str, torch.Tensor]]:
        def load_index(idx: int) -> Dict[str, torch.Tensor]:
            data = dataset.get_data(idx, image_type=self.config.cache_images_type)
            camera = dataset.cameras[idx].reshape(())
            if camera.distortion_params is not None and torch.any(camera.distortion_params != 0):
                K = camera.get_intrinsics_matrices()[0].numpy()
                distortion_params = camera.distortion_params[0].numpy()
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

        with ThreadPoolExecutor(max_workers=2) as executor:
            caches = list(executor.map(load_index, range(len(dataset))))
        for cache in caches:
            if self.config.cache_images == "gpu":
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
            else:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()
        if split == "train" and self.config.cache_images == "gpu":
            self.train_cameras = dataset.cameras.to(self.device)
        elif split == "train":
            self.train_cameras = dataset.cameras
        return caches

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        cameras = self.eval_dataset.cameras.to(self.device)
        batches = []
        for idx, cached in enumerate(self.cached_eval):
            data = cached.copy()
            data["image"] = data["image"].to(self.device)
            batches.append((cameras[idx : idx + 1], data))
        return batches

    def get_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        return {}

    def get_datapath(self) -> Path:
        return self.dataparser.project_root

    def get_train_rays_per_batch(self) -> int:
        camera = self.train_dataset.cameras[0].reshape(())
        return int(camera.width[0].item() * camera.height[0].item())

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        del step
        image_idx = self.train_unseen_cameras.pop(0)
        if not self.train_unseen_cameras:
            self.train_unseen_cameras = self.sample_train_cameras()
        data = self.cached_train[image_idx].copy()
        data["image"] = data["image"].to(self.device)
        camera = self.train_cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        return self.next_eval_image(step)

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        del step
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        if not self.eval_unseen_cameras:
            self.eval_unseen_cameras = list(range(len(self.eval_dataset)))
        data = self.cached_eval[image_idx].copy()
        data["image"] = data["image"].to(self.device)
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        return camera, data
