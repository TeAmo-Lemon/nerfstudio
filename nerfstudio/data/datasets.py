# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataset classes.
"""

from __future__ import annotations

import io
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import numpy.typing as npt
import torch
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path, pil_to_numpy


class InputDataset(Dataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs
    """

    exclude_batch_keys_from_device: List[str] = ["image", "mask"]
    cameras: Cameras

    def __init__(
        self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0, cache_compressed_images: bool = False
    ):
        super().__init__()
        self._dataparser_outputs = dataparser_outputs
        self.scale_factor = scale_factor
        self.scene_box = deepcopy(dataparser_outputs.scene_box)
        self.metadata = deepcopy(dataparser_outputs.metadata)
        self.cameras = deepcopy(dataparser_outputs.cameras)
        self.cameras.rescale_output_resolution(scaling_factor=scale_factor)
        self.mask_color = dataparser_outputs.metadata.get("mask_color", None)
        self.cache_compressed_images = cache_compressed_images
        if cache_compressed_images:
            self.binary_images = []
            self.binary_masks = []
            for image_filename in self._dataparser_outputs.image_filenames:
                with open(image_filename, "rb") as f:
                    self.binary_images.append(io.BytesIO(f.read()))
            if self._dataparser_outputs.mask_filenames is not None:
                for mask_filename in self._dataparser_outputs.mask_filenames:
                    with open(mask_filename, "rb") as f:
                        self.binary_masks.append(io.BytesIO(f.read()))

    def __len__(self):
        return len(self._dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        if self.cache_compressed_images:
            pil_image = Image.open(self.binary_images[image_idx])
        else:
            pil_image = Image.open(image_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.Resampling.BILINEAR)
        image = pil_to_numpy(pil_image)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is incorrect."
        return image

    def get_image_float32(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        image = self.get_numpy_image(image_idx)
        image = image / np.float32(255)
        image = torch.from_numpy(image)
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        return image

    def get_image_uint8(self, image_idx: int) -> UInt8[Tensor, "image_height image_width num_channels"]:
        image = torch.from_numpy(self.get_numpy_image(image_idx))
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * (image[:, :, -1:] / 255.0) + 255.0 * self._dataparser_outputs.alpha_color * (
                1.0 - image[:, :, -1:] / 255.0
            )
            image = torch.clamp(image, min=0, max=255).to(torch.uint8)
        return image

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        if image_type == "float32":
            image = self.get_image_float32(image_idx)
        elif image_type == "uint8":
            image = self.get_image_uint8(image_idx)
        else:
            raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")

        data = {"image_idx": image_idx, "image": image}
        if self._dataparser_outputs.mask_filenames is not None:
            if self.cache_compressed_images:
                mask_filepath = self.binary_masks[image_idx]
            else:
                mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert data["mask"].shape[:2] == data["image"].shape[:2], (
                f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
            )
        if self.mask_color:
            data["image"] = torch.where(
                data["mask"] == 1.0, data["image"], torch.ones_like(data["image"]) * torch.tensor(self.mask_color)
            )
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def get_metadata(self, data: Dict) -> Dict:
        del data
        return {}

    def __getitem__(self, image_idx: int) -> Dict:
        data = self.get_data(image_idx)
        return data

    @property
    def image_filenames(self) -> List[Path]:
        return self._dataparser_outputs.image_filenames


# ── DinoInputDataset ──────────────────────────────────────────────────────────

MERGED_FEATURE_FILENAME = "dino_features.pt"


class DinoInputDataset(InputDataset):
    """InputDataset extension that also loads precomputed DINO feature maps."""

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["dino_feature"]

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
        cache_compressed_images: bool = False,
        dino_features_dir: Optional[Path] = None,
        dino_feature_dim: int = 16,
        strict_dino_loading: bool = True,
    ):
        super().__init__(
            dataparser_outputs=dataparser_outputs,
            scale_factor=scale_factor,
            cache_compressed_images=cache_compressed_images,
        )
        self.dino_feature_dim = dino_feature_dim
        self.strict_dino_loading = strict_dino_loading

        image_parent_paths = [str(path.parent) for path in self._dataparser_outputs.image_filenames]
        self.image_root = Path(os.path.commonpath(image_parent_paths))
        if dino_features_dir is None:
            self.dino_features_dir = self.image_root.parent / "dino_features"
        else:
            self.dino_features_dir = Path(dino_features_dir)
        self._cached_feature_file_mtime_ns: Optional[int] = None
        self._cached_features: Dict[str, torch.Tensor] = {}

    def image_path_for_image_idx(self, image_idx: int) -> Path:
        return Path(self._dataparser_outputs.image_filenames[image_idx])

    def feature_file_path(self) -> Path:
        return self.dino_features_dir / MERGED_FEATURE_FILENAME

    def feature_key_for_image_idx(self, image_idx: int) -> str:
        image_path = self._dataparser_outputs.image_filenames[image_idx]
        return image_path.relative_to(self.image_root).with_suffix(".pt").as_posix()

    def invalidate_feature_cache(self) -> None:
        self._cached_feature_file_mtime_ns = None
        self._cached_features = {}

    def _reload_features_if_needed(self) -> None:
        feature_file = self.feature_file_path()
        if not feature_file.exists():
            self.invalidate_feature_cache()
            return

        mtime_ns = feature_file.stat().st_mtime_ns
        if self._cached_feature_file_mtime_ns == mtime_ns:
            return

        payload = torch.load(feature_file, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Expected dict in {feature_file}, got {type(payload)}")
        if "features" not in payload:
            raise KeyError(f"Missing 'features' in {feature_file}")
        raw_features = payload["features"]
        if not isinstance(raw_features, dict):
            raise TypeError(f"Expected dict for 'features' in {feature_file}, got {type(raw_features)}")

        features: Dict[str, torch.Tensor] = {}
        for raw_key, raw_value in raw_features.items():
            if not isinstance(raw_value, torch.Tensor):
                raise TypeError(f"Feature entry '{raw_key}' in {feature_file} is not tensor: {type(raw_value)}")
            key = str(raw_key).replace("\\", "/")
            features[key] = self._normalize_feature_tensor(raw_value)

        self._cached_features = features
        self._cached_feature_file_mtime_ns = mtime_ns

    def missing_feature_image_paths(self) -> List[Path]:
        self._reload_features_if_needed()
        missing_paths: List[Path] = []
        for image_idx in range(len(self)):
            key = self.feature_key_for_image_idx(image_idx)
            if key not in self._cached_features:
                missing_paths.append(self.image_path_for_image_idx(image_idx))
        return missing_paths

    def _normalize_feature_tensor(self, feature: torch.Tensor) -> torch.Tensor:
        feature = feature.float()

        if feature.ndim == 4 and feature.shape[0] == 1:
            feature = feature.squeeze(0)

        if feature.ndim != 3:
            raise ValueError(f"DINO feature must be 3D tensor, got shape {tuple(feature.shape)}")

        if feature.shape[0] == self.dino_feature_dim and feature.shape[-1] != self.dino_feature_dim:
            feature = feature.permute(1, 2, 0).contiguous()

        if feature.shape[-1] != self.dino_feature_dim:
            raise ValueError(
                f"Expected last dim {self.dino_feature_dim} for DINO feature, got shape {tuple(feature.shape)}"
            )

        return feature

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        data = super().get_data(image_idx=image_idx, image_type=image_type)
        self._reload_features_if_needed()

        feature_key = self.feature_key_for_image_idx(image_idx)
        feature = self._cached_features.get(feature_key)

        if feature is None:
            if self.strict_dino_loading:
                raise FileNotFoundError(
                    f"Missing DINO feature for image index {image_idx} (key='{feature_key}') "
                    f"in merged file: {self.feature_file_path()}. "
                    "Run scripts/extract_dino_features.py first."
                )
            height, width = data["image"].shape[:2]
            data["dino_feature"] = torch.zeros((height, width, self.dino_feature_dim), dtype=torch.float32)
            return data

        data["dino_feature"] = feature
        return data
