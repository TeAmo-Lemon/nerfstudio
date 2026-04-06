from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Dict, Optional, Union

import torch

TORCH_DEVICE = Union[torch.device, str]


class CameraType(IntEnum):
    PERSPECTIVE = auto()
    FISHEYE = auto()
    EQUIRECTANGULAR = auto()
    OMNIDIRECTIONALSTEREO_L = auto()
    OMNIDIRECTIONALSTEREO_R = auto()
    VR180_L = auto()
    VR180_R = auto()
    ORTHOPHOTO = auto()
    FISHEYE624 = auto()


CAMERA_MODEL_TO_TYPE = {
    "SIMPLE_PINHOLE": CameraType.PERSPECTIVE,
    "PINHOLE": CameraType.PERSPECTIVE,
    "SIMPLE_RADIAL": CameraType.PERSPECTIVE,
    "RADIAL": CameraType.PERSPECTIVE,
    "OPENCV": CameraType.PERSPECTIVE,
    "OPENCV_FISHEYE": CameraType.FISHEYE,
    "EQUIRECTANGULAR": CameraType.EQUIRECTANGULAR,
    "OMNIDIRECTIONALSTEREO_L": CameraType.OMNIDIRECTIONALSTEREO_L,
    "OMNIDIRECTIONALSTEREO_R": CameraType.OMNIDIRECTIONALSTEREO_R,
    "VR180_L": CameraType.VR180_L,
    "VR180_R": CameraType.VR180_R,
    "ORTHOPHOTO": CameraType.ORTHOPHOTO,
    "FISHEYE624": CameraType.FISHEYE624,
}


@dataclass(init=False)
class Cameras:
    camera_to_worlds: torch.Tensor
    fx: torch.Tensor
    fy: torch.Tensor
    cx: torch.Tensor
    cy: torch.Tensor
    width: torch.Tensor
    height: torch.Tensor
    distortion_params: Optional[torch.Tensor]
    camera_type: torch.Tensor
    metadata: Optional[Dict]

    def __init__(
        self,
        camera_to_worlds: torch.Tensor,
        fx: Union[torch.Tensor, float],
        fy: Union[torch.Tensor, float],
        cx: Union[torch.Tensor, float],
        cy: Union[torch.Tensor, float],
        width: Union[torch.Tensor, int],
        height: Union[torch.Tensor, int],
        distortion_params: Optional[torch.Tensor] = None,
        camera_type: Union[torch.Tensor, int, CameraType] = CameraType.PERSPECTIVE,
        metadata: Optional[Dict] = None,
    ) -> None:
        self.camera_to_worlds = self._ensure_tensor(camera_to_worlds, ndim=3, dtype=torch.float32)
        self.fx = self._ensure_tensor(fx, dtype=torch.float32)
        self.fy = self._ensure_tensor(fy, dtype=torch.float32)
        self.cx = self._ensure_tensor(cx, dtype=torch.float32)
        self.cy = self._ensure_tensor(cy, dtype=torch.float32)
        self.width = self._ensure_tensor(width, dtype=torch.int64)
        self.height = self._ensure_tensor(height, dtype=torch.int64)
        self.distortion_params = None if distortion_params is None else self._ensure_tensor(distortion_params, dtype=torch.float32, keep_last=False)
        if isinstance(camera_type, CameraType):
            camera_type = int(camera_type)
        self.camera_type = self._ensure_tensor(camera_type, dtype=torch.int64)

        # Match Nerfstudio behavior: scalar intrinsics/camera type should broadcast
        # across all cameras so per-index access stays valid.
        num_cameras = int(self.camera_to_worlds.shape[0])
        self.fx = self._broadcast_first_dim(self.fx, num_cameras)
        self.fy = self._broadcast_first_dim(self.fy, num_cameras)
        self.cx = self._broadcast_first_dim(self.cx, num_cameras)
        self.cy = self._broadcast_first_dim(self.cy, num_cameras)
        self.width = self._broadcast_first_dim(self.width, num_cameras)
        self.height = self._broadcast_first_dim(self.height, num_cameras)
        self.camera_type = self._broadcast_first_dim(self.camera_type, num_cameras)
        if self.distortion_params is not None:
            self.distortion_params = self._broadcast_first_dim(self.distortion_params, num_cameras)
        self.metadata = metadata

    @staticmethod
    def _ensure_tensor(value, dtype: torch.dtype, ndim: int = 1, keep_last: bool = True) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            tensor = value.to(dtype=dtype)
        else:
            tensor = torch.tensor(value, dtype=dtype)
        if ndim == 3 and tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if ndim == 1:
            if tensor.ndim == 0:
                tensor = tensor.view(1, 1)
            elif tensor.ndim == 1:
                if keep_last:
                    tensor = tensor.reshape(-1, 1)
                else:
                    tensor = tensor.unsqueeze(0)
            elif keep_last and tensor.shape[-1] != 1:
                tensor = tensor.unsqueeze(-1)
        return tensor

    @staticmethod
    def _broadcast_first_dim(tensor: torch.Tensor, target_size: int) -> torch.Tensor:
        if tensor.ndim == 0:
            tensor = tensor.reshape(1)
        current_size = int(tensor.shape[0])
        if current_size == target_size:
            return tensor
        if current_size != 1:
            raise ValueError(f"Cannot broadcast tensor with first dim {current_size} to {target_size}.")
        return tensor.expand(target_size, *tensor.shape[1:])

    @property
    def device(self) -> torch.device:
        return self.camera_to_worlds.device

    @property
    def shape(self):
        return self.camera_to_worlds.shape[:-2]

    @property
    def size(self) -> int:
        return int(self.camera_to_worlds.shape[0])

    def __len__(self) -> int:
        return self.camera_to_worlds.shape[0]

    def __getitem__(self, item) -> "Cameras":
        metadata = self.metadata.copy() if isinstance(self.metadata, dict) else self.metadata
        return Cameras(
            camera_to_worlds=self.camera_to_worlds[item],
            fx=self.fx[item],
            fy=self.fy[item],
            cx=self.cx[item],
            cy=self.cy[item],
            width=self.width[item],
            height=self.height[item],
            distortion_params=None if self.distortion_params is None else self.distortion_params[item],
            camera_type=self.camera_type[item],
            metadata=metadata,
        )

    def reshape(self, shape=()) -> "Cameras":
        if shape == () and self.camera_to_worlds.shape[0] == 1:
            return Cameras(
                camera_to_worlds=self.camera_to_worlds.reshape(1, 3, 4),
                fx=self.fx.reshape(1, 1),
                fy=self.fy.reshape(1, 1),
                cx=self.cx.reshape(1, 1),
                cy=self.cy.reshape(1, 1),
                width=self.width.reshape(1, 1),
                height=self.height.reshape(1, 1),
                distortion_params=None if self.distortion_params is None else self.distortion_params.reshape(1, -1),
                camera_type=self.camera_type.reshape(1, 1),
                metadata=self.metadata.copy() if isinstance(self.metadata, dict) else self.metadata,
            )
        raise NotImplementedError("Only scalar reshape is implemented for HMGS cameras.")

    def to(self, device: TORCH_DEVICE) -> "Cameras":
        return Cameras(
            camera_to_worlds=self.camera_to_worlds.to(device),
            fx=self.fx.to(device),
            fy=self.fy.to(device),
            cx=self.cx.to(device),
            cy=self.cy.to(device),
            width=self.width.to(device),
            height=self.height.to(device),
            distortion_params=None if self.distortion_params is None else self.distortion_params.to(device),
            camera_type=self.camera_type.to(device),
            metadata=self.metadata.copy() if isinstance(self.metadata, dict) else self.metadata,
        )

    def get_intrinsics_matrices(self) -> torch.Tensor:
        K = torch.zeros((len(self), 3, 3), dtype=self.fx.dtype, device=self.fx.device)
        K[:, 0, 0] = self.fx.squeeze(-1)
        K[:, 1, 1] = self.fy.squeeze(-1)
        K[:, 0, 2] = self.cx.squeeze(-1)
        K[:, 1, 2] = self.cy.squeeze(-1)
        K[:, 2, 2] = 1.0
        return K

    def rescale_output_resolution(self, scaling_factor: float) -> None:
        self.fx = self.fx * scaling_factor
        self.fy = self.fy * scaling_factor
        self.cx = self.cx * scaling_factor
        self.cy = self.cy * scaling_factor
        self.width = torch.clamp(torch.floor(self.width.float() * scaling_factor), min=1).to(torch.int64)
        self.height = torch.clamp(torch.floor(self.height.float() * scaling_factor), min=1).to(torch.int64)

    def update_tiling_intrinsics(self, tiling_factor: int) -> None:
        del tiling_factor
