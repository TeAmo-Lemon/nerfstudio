from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union, cast

import torch

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.dino_splatfacto import DinoSplatfactoModel, DinoSplatfactoModelConfig, get_viewmat
from nerfstudio.utils.semgeo_utils import (
    build_stage1_scene_clusters,
    build_stage2_style_primitives,
    save_stage2_label_image,
)


@dataclass
class SemGeoSplatfactoModelConfig(DinoSplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: SemGeoSplatfactoModel)
    cluster_refresh_start_step: int = 3000
    cluster_refresh_every: int = 2000
    cluster_knn: int = 16
    cluster_opacity_threshold: float = 0.05
    cluster_feature_similarity_threshold: float = 0.6
    cluster_normal_similarity_threshold: float = 0.5
    cluster_radius_scale: float = 1.5
    cluster_min_size: int = 64
    cluster_max_sample_points: int = 8192
    cluster_max_metric_points: int = 32
    cluster_geometry_weight: float = 1.0
    cluster_dino_weight: float = 1.0
    cluster_max_viewer_points: int = 50000
    style_num_primitives: int = 8
    style_spatial_weight: float = 0.25
    style_kmeans_iters: int = 25


class SemGeoSplatfactoModel(DinoSplatfactoModel):
    """DINO splatfacto variant with Stage 1/2 SemGeo preprocessing assets."""

    def populate_modules(self):
        super().populate_modules()
        self._reset_semgeo_cache()
        self._phase1_asset_path: Optional[Path] = None
        self._phase2_asset_path: Optional[Path] = None
        self._bundle_asset_path: Optional[Path] = None
        self._last_cluster_refresh_step = -1

    def _semgeo_config(self) -> SemGeoSplatfactoModelConfig:
        return cast(SemGeoSplatfactoModelConfig, self.config)

    def _reset_semgeo_cache(self) -> None:
        self._cluster_labels_cpu: Optional[torch.Tensor] = None
        self._cluster_point_colors_cpu: Optional[torch.Tensor] = None
        self._cluster_viewer_points_cpu: Optional[torch.Tensor] = None
        self._cluster_viewer_colors_cpu: Optional[torch.Tensor] = None
        self._cluster_metric_spaces: Optional[list[dict[str, Any]]] = None
        self._cluster_visualization_version = 0
        self._cluster_render_colors_device: Optional[torch.Tensor] = None
        self._cluster_render_colors_version = -1
        self._latest_phase1_metadata: Dict[str, Any] = {}
        self._latest_phase2_metadata: Dict[str, Any] = {}

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        self._reset_semgeo_cache()
        super().load_state_dict(dict, **kwargs)

    def finish_train_step(self, step):
        super().finish_train_step(step)
        if self._cluster_labels_cpu is not None and self._cluster_labels_cpu.shape[0] != self.means.shape[0]:
            self._reset_semgeo_cache()

    def refresh_stage1_clusters(self, output_dir: Path) -> Path:
        cfg = self._semgeo_config()
        stage1 = build_stage1_scene_clusters(
            means=self.means,
            dino_features=self.dino_features,
            colors=self.colors,
            opacities=self.opacities,
            knn_k=cfg.cluster_knn,
            opacity_threshold=cfg.cluster_opacity_threshold,
            feature_similarity_threshold=cfg.cluster_feature_similarity_threshold,
            normal_similarity_threshold=cfg.cluster_normal_similarity_threshold,
            radius_scale=cfg.cluster_radius_scale,
            min_cluster_size=cfg.cluster_min_size,
            max_sample_points=cfg.cluster_max_sample_points,
            max_metric_points=cfg.cluster_max_metric_points,
            geometry_weight=cfg.cluster_geometry_weight,
            dino_weight=cfg.cluster_dino_weight,
            max_viewer_points=cfg.cluster_max_viewer_points,
        )

        self._cluster_labels_cpu = cast(torch.Tensor, stage1["labels"])
        self._cluster_point_colors_cpu = cast(torch.Tensor, stage1["point_colors"])
        self._cluster_viewer_points_cpu = cast(torch.Tensor, stage1["viewer_points"])
        self._cluster_viewer_colors_cpu = cast(torch.Tensor, stage1["viewer_colors"])
        self._cluster_metric_spaces = cast(list[dict[str, Any]], stage1["metric_spaces"])
        self._cluster_visualization_version += 1
        self._cluster_render_colors_device = None
        self._cluster_render_colors_version = -1
        self._latest_phase1_metadata = cast(Dict[str, Any], stage1["metadata"])

        semgeo_dir = output_dir / "semgeo"
        semgeo_dir.mkdir(parents=True, exist_ok=True)
        self._phase1_asset_path = semgeo_dir / "phase1_3d_clusters.pt"
        torch.save(
            {
                "step": self.step,
                "labels": self._cluster_labels_cpu,
                "point_colors": self._cluster_point_colors_cpu,
                "metric_spaces": self._cluster_metric_spaces,
                "metadata": self._latest_phase1_metadata,
            },
            self._phase1_asset_path,
        )
        self._last_cluster_refresh_step = self.step
        return self._phase1_asset_path

    def prepare_style_primitives(self, output_dir: Path, style_image_path: Path, dino_feature_file: Path) -> Path:
        cfg = self._semgeo_config()
        stage2 = build_stage2_style_primitives(
            style_image_path=style_image_path,
            dino_feature_file=dino_feature_file,
            feature_dim=cfg.dino_feature_dim,
            num_clusters=cfg.style_num_primitives,
            spatial_weight=cfg.style_spatial_weight,
            kmeans_iters=cfg.style_kmeans_iters,
            max_metric_points=cfg.cluster_max_metric_points,
        )

        semgeo_dir = output_dir / "semgeo"
        semgeo_dir.mkdir(parents=True, exist_ok=True)
        label_image_path = semgeo_dir / "phase2_style_labels.png"
        save_stage2_label_image(stage2["label_image"], label_image_path)

        self._phase2_asset_path = semgeo_dir / "phase2_style_primitives.pt"
        self._latest_phase2_metadata = cast(Dict[str, Any], stage2["metadata"])
        torch.save(
            {
                "style_image_path": stage2["style_image_path"],
                "original_hw": stage2["original_hw"],
                "patch_hw": stage2["patch_hw"],
                "patch_features": stage2["patch_features"],
                "patch_positions": stage2["patch_positions"],
                "labels": stage2["labels"],
                "palette": stage2["palette"],
                "label_image_path": str(label_image_path),
                "primitive_spaces": stage2["primitive_spaces"],
                "metadata": self._latest_phase2_metadata,
            },
            self._phase2_asset_path,
        )
        return self._phase2_asset_path

    def _write_stage12_bundle(self, output_dir: Path) -> None:
        semgeo_dir = output_dir / "semgeo"
        semgeo_dir.mkdir(parents=True, exist_ok=True)
        self._bundle_asset_path = semgeo_dir / "stage12_ready.pt"
        torch.save(
            {
                "phase1_3d_clusters": str(self._phase1_asset_path) if self._phase1_asset_path is not None else None,
                "phase2_style_primitives": str(self._phase2_asset_path) if self._phase2_asset_path is not None else None,
                "phase1_metadata": self._latest_phase1_metadata,
                "phase2_metadata": self._latest_phase2_metadata,
                "latest_training_step": self.step,
            },
            self._bundle_asset_path,
        )

    def _write_placeholder_record(self, output_dir: Path, file_name: str, payload: Dict[str, Any]) -> Path:
        semgeo_dir = output_dir / "semgeo"
        semgeo_dir.mkdir(parents=True, exist_ok=True)
        record_path = semgeo_dir / file_name
        record_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf8")
        return record_path

    def run_fgw_alignment_placeholder(self, output_dir: Path) -> Path:
        """Placeholder entrypoint for Stage 3 FGW alignment."""
        return self._write_placeholder_record(
            output_dir,
            "phase3_fgw_placeholder.json",
            {
                "implemented": False,
                "message": "FGW alignment is not implemented yet.",
                "step": self.step,
                "phase1_asset": str(self._phase1_asset_path) if self._phase1_asset_path is not None else None,
                "phase2_asset": str(self._phase2_asset_path) if self._phase2_asset_path is not None else None,
            },
        )

    def run_style_transfer_placeholder(self, output_dir: Path) -> Path:
        """Placeholder entrypoint for Stage 4 multi-view style transfer."""
        return self._write_placeholder_record(
            output_dir,
            "phase4_style_placeholder.json",
            {
                "implemented": False,
                "message": "Style transfer optimization is not implemented yet.",
                "step": self.step,
            },
        )

    def maybe_refresh_semgeo_assets(
        self,
        output_dir: Path,
        *,
        style_image_path: Optional[Path] = None,
        dino_feature_file: Optional[Path] = None,
        force: bool = False,
    ) -> bool:
        updated = False
        if style_image_path is not None and dino_feature_file is not None:
            if force or self._phase2_asset_path is None or not self._phase2_asset_path.exists():
                self.prepare_style_primitives(output_dir, style_image_path=style_image_path, dino_feature_file=dino_feature_file)
                updated = True

        cfg = self._semgeo_config()
        should_refresh_clusters = force or (
            self.step >= cfg.cluster_refresh_start_step
            and (self._last_cluster_refresh_step < 0 or self.step - self._last_cluster_refresh_step >= cfg.cluster_refresh_every)
        )
        if should_refresh_clusters:
            self.refresh_stage1_clusters(output_dir)
            updated = True

        if updated:
            self._write_stage12_bundle(output_dir)
        return updated

    def get_cluster_visualization_state(self) -> Optional[Dict[str, Any]]:
        if self._cluster_viewer_points_cpu is None or self._cluster_viewer_colors_cpu is None:
            return None
        return {
            "version": self._cluster_visualization_version,
            "points": self._cluster_viewer_points_cpu.numpy(),
            "colors": self._cluster_viewer_colors_cpu.numpy(),
            "point_size": 0.02,
            "num_clusters": self._latest_phase1_metadata.get("num_clusters", 0),
        }

    def _get_cluster_render_colors(self) -> Optional[torch.Tensor]:
        if self._cluster_point_colors_cpu is None or self._cluster_point_colors_cpu.shape[0] != self.means.shape[0]:
            return None
        if (
            self._cluster_render_colors_device is None
            or self._cluster_render_colors_version != self._cluster_visualization_version
        ):
            self._cluster_render_colors_device = self._cluster_point_colors_cpu.to(self.device).float() / 255.0
            self._cluster_render_colors_version = self._cluster_visualization_version
        return self._cluster_render_colors_device

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, list]]:
        outputs = super().get_outputs(camera)
        if not isinstance(camera, Cameras):
            return outputs

        cluster_colors = self._get_cluster_render_colors()
        if cluster_colors is None:
            return outputs

        if self.training:
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                width, height = int(camera.width.item()), int(camera.height.item())
                outputs["cluster_rgb"] = torch.zeros((height, width, 3), device=self.device)
                return outputs
        else:
            crop_ids = None

        means_crop, quats_crop, scales_crop, opacities_crop, _, _ = self._select_render_tensors(crop_ids)
        cluster_colors_crop = cluster_colors if crop_ids is None else cluster_colors[crop_ids]

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        intrinsics = camera.get_intrinsics_matrices().to(self.device)
        width, height = int(camera.width.item()), int(camera.height.item())
        camera.rescale_output_resolution(camera_scale_fac)

        rendered_cluster, _, _ = rasterization(  # type: ignore[reportPossiblyUnboundVariable]
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=cluster_colors_crop,
            viewmats=viewmat,
            Ks=intrinsics,
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=None,
            sparse_grad=False,
            absgrad=False,
            rasterize_mode=self.config.rasterize_mode,
        )
        outputs["cluster_rgb"] = rendered_cluster.squeeze(0).clamp(0.0, 1.0)
        return outputs
