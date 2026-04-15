from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

import torch

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset


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

        # Allow both [H, W, C] and [C, H, W] input formats.
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
