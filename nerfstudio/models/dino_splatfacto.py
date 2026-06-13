from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import torch
import torch.nn.functional as F
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import duplicate, remove, reset_opa, split
from torch.nn import Parameter

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, get_viewmat
from nerfstudio.utils import writer as writer_module
from nerfstudio.utils.rich_utils import CONSOLE


# ═══════════════════════════════════════════════════════════════════════════════
# DinoDefaultStrategy — delegates densify/prune ops to the model so DINO
# features stay in sync with Gaussian params.
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DinoDefaultStrategy(DefaultStrategy):
    """DefaultStrategy variant that delegates densification ops to model methods."""

    model_ref: Optional["DinoSplatfactoModel"] = None
    _last_densify_stats: Optional[Dict[str, int]] = None

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        del optimizers
        if self.model_ref is None:
            raise RuntimeError("DinoDefaultStrategy.model_ref is not set")

        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        is_small = torch.exp(params["scales"]).max(dim=-1).values <= self.grow_scale3d * state["scene_scale"]
        is_dupli = is_grad_high & is_small
        n_dupli = int(is_dupli.sum().item())

        if n_dupli > 0:
            self.model_ref._clone_gaussians(mask=is_dupli, state=state)

        is_large = ~is_small
        is_split = is_grad_high & is_large
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d
        n_split = int(is_split.sum().item())

        if n_dupli > 0:
            is_split = torch.cat([is_split, torch.zeros(n_dupli, dtype=torch.bool, device=device)], dim=0)

        if n_split > 0:
            self.model_ref._split_gaussians(mask=is_split, state=state, revised_opacity=self.revised_opacity)

        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        del optimizers
        if self.model_ref is None:
            raise RuntimeError("DinoDefaultStrategy.model_ref is not set")

        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            is_too_big = torch.exp(params["scales"]).max(dim=-1).values > self.prune_scale3d * state["scene_scale"]
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d
            is_prune = is_prune | is_too_big

        n_prune = int(is_prune.sum().item())
        if n_prune > 0:
            self.model_ref._cull_gaussians(mask=is_prune, state=state)

        return n_prune

    @torch.no_grad()
    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ) -> None:
        """Override to suppress prints and capture densification stats."""
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
            n_prune = self._prune_gs(params, optimizers, state, step)

            # store stats instead of printing
            self._last_densify_stats = {
                "n_dupli": n_dupli,
                "n_split": n_split,
                "n_prune": n_prune,
                "total_gs": len(params["means"]),
            }

            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# DinoSplatfactoModel
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DinoSplatfactoModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory=lambda: DinoSplatfactoModel)
    dino_feature_dim: int = 16
    dino_loss_weight: float = 1.0
    dino_loss_start_step: int = 3000
    style_primitive_path: Optional[Path] = None
    num_style_primitives: int = 8


class DinoSplatfactoModel(SplatfactoModel):
    """Splatfacto model with DINO feature splatting and feature distillation loss."""

    # ── helpers ──────────────────────────────────────────────────────────

    def _dino_config(self) -> DinoSplatfactoModelConfig:
        return cast(DinoSplatfactoModelConfig, self.config)

    # ── setup ────────────────────────────────────────────────────────────

    def populate_modules(self):
        super().populate_modules()
        num_gaussians = self.means.shape[0]
        cfg = self._dino_config()
        self.gauss_params["dino_features"] = torch.nn.Parameter(
            0.01 * torch.randn(num_gaussians, cfg.dino_feature_dim, device=self.means.device)
        )
        self._override_default_strategy_with_dino_strategy()

        # ── Load style texture primitives ──────────────────────────────────
        if cfg.style_primitive_path is not None and cfg.style_primitive_path.exists():
            payload = torch.load(cfg.style_primitive_path, map_location="cpu", weights_only=True)
            primitives = payload.get("prototypes")
            if primitives is not None:
                self.register_buffer("style_primitives", primitives, persistent=False)
                CONSOLE.log(f"Loaded style primitives: {tuple(primitives.shape)}")
            else:
                CONSOLE.log("[yellow]style_primitives.pt found but missing 'prototypes' key[/yellow]")

    # ── core rendering ───────────────────────────────────────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """为给定相机渲染 RGB 图像 + DINO 语义特征图。

        流程:
        1. 调用父类 get_outputs → 获取 {"rgb", "depth", "accumulation", "background"}
        2. 相机类型守卫
        3. 训练时施加 camera optimizer 偏移量
        4. 评估时检查裁剪框，若框内无高斯则填充零特征并直接返回
        5. 选出高斯基元参数（含 DINO 特征）
        6. 缩放相机分辨率 → 计算视图矩阵 V 和内参 K
        7. 第二次 CUDA rasterization，用 DINO 特征作为 "color" 通道渲染
        8. 将渲染出的 DINO 特征图 + PCA 可视化结果写回 outputs
        """
        # 步骤 1: 先做 RGB 渲染（复用父类完整流程）
        outputs = super().get_outputs(camera)
        # 步骤 2: 非 Cameras 输入直接返回 RGB 结果
        if not isinstance(camera, Cameras):
            return outputs

        # 步骤 3: 训练时对相机外参施加可学习偏移
        if self.training:
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # 步骤 4: 评估时裁剪框过滤，框内无高斯则返回零特征图
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                width, height = int(camera.width.item()), int(camera.height.item())
                empty_features = torch.zeros((height, width, self._dino_config().dino_feature_dim), device=self.device)
                outputs["rendered_features"] = empty_features
                outputs["dino_semantic_rgb"] = self._semantic_rgb_from_features(empty_features)
                return outputs
        else:
            crop_ids = None

        # 步骤 5: 选出活跃高斯的参数 — 位置、旋转、缩放、不透明度、DINO 特征
        means_crop, quats_crop, scales_crop, opacities_crop, dino_features_crop, _ = self._select_render_tensors(crop_ids)

        # 步骤 6: 计算下采样后的视图矩阵、内参、输出分辨率
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().to(self.device)
        width, height = int(camera.width.item()), int(camera.height.item())
        camera.rescale_output_resolution(camera_scale_fac)

        # 步骤 7: 第二次光栅化 — 将 DINO 特征作为 "颜色" 做 alpha-blend，得到逐像素特征
        rendered_features, _, _ = rasterization(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=dino_features_crop,          # 关键区别：用的是 DINO 特征而非 RGB
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",                  # 纯 RGB 模式（特征值本身不需要深度通道）
            sh_degree=None,                     # 特征不做 SH 编码，直接插值
            sparse_grad=False,
            absgrad=False,
            rasterize_mode=self.config.rasterize_mode,
        )

        # 步骤 8: 写入输出字典 — 原始 DINO 特征 + PCA 降维到 3 通道的 RGB 可视化
        rendered_features = rendered_features.squeeze(0)
        outputs["rendered_features"] = rendered_features      # (H, W, dino_feature_dim)
        outputs["dino_semantic_rgb"] = self._semantic_rgb_from_features(rendered_features)  # (H, W, 3) PCA 可视化
        return outputs

    # ── training loss & metrics ──────────────────────────────────────────

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        cfg = self._dino_config()

        if self.step < cfg.dino_loss_start_step:
            return loss_dict
        if "rendered_features" not in outputs or "dino_feature" not in batch:
            return loss_dict

        rendered_features = outputs["rendered_features"]
        gt_feature = batch["dino_feature"]
        if not isinstance(rendered_features, torch.Tensor) or not isinstance(gt_feature, torch.Tensor):
            return loss_dict

        pred = rendered_features
        pred_hw = (int(pred.shape[0]), int(pred.shape[1]))
        gt = self._prepare_gt_dino_feature(gt_feature.to(self.device, dtype=pred.dtype), pred_hw=pred_hw)

        common_dim = min(pred.shape[-1], gt.shape[-1])
        if common_dim <= 0:
            return loss_dict

        dino_l1 = F.l1_loss(pred[..., :common_dim], gt[..., :common_dim])
        loss_dict["dino_feature_loss"] = cfg.dino_loss_weight * dino_l1
        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        metrics_dict = super().get_metrics_dict(outputs, batch)

        if "rendered_features" not in outputs or "dino_feature" not in batch:
            return metrics_dict

        rendered_features = outputs["rendered_features"]
        gt_feature = batch["dino_feature"]
        if not isinstance(rendered_features, torch.Tensor) or not isinstance(gt_feature, torch.Tensor):
            return metrics_dict

        pred = rendered_features
        pred_hw = (int(pred.shape[0]), int(pred.shape[1]))
        gt = self._prepare_gt_dino_feature(gt_feature.to(self.device, dtype=pred.dtype), pred_hw=pred_hw)
        common_dim = min(pred.shape[-1], gt.shape[-1])
        if common_dim > 0:
            metrics_dict["dino_l1"] = F.l1_loss(pred[..., :common_dim], gt[..., :common_dim]).detach()

        return metrics_dict

    # ── training lifecycle ───────────────────────────────────────────────

    def finish_train_step(self, step):
        super().finish_train_step(step)

        # Report current GS count (post-densification)
        writer_module.put_scalar("Total GSs", int(self.means.shape[0]), step)

        # Report densification stats if available
        s = self.strategy._last_densify_stats
        if s is not None:
            writer_module.put_scalar("GS Duplicated", s["n_dupli"], step)
            writer_module.put_scalar("GS Split", s["n_split"], step)
            writer_module.put_scalar("GS Pruned", s["n_prune"], step)
            self.strategy._last_densify_stats = None

        if self.dino_features.shape[0] != self.means.shape[0]:
            raise RuntimeError(
                "DINO feature count is out of sync with Gaussian count: "
                f"{self.dino_features.shape[0]} != {self.means.shape[0]}"
            )

    # ── optimizer param groups ───────────────────────────────────────────

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = super().get_gaussian_param_groups()
        groups["dino_features"] = [self.gauss_params["dino_features"]]
        return groups

    # ── properties ───────────────────────────────────────────────────────

    @property
    def dino_features(self):
        return self.gauss_params["dino_features"]

    # ── private helpers ──────────────────────────────────────────────────

    def _override_default_strategy_with_dino_strategy(self) -> None:
        if not isinstance(self.strategy, DefaultStrategy):
            return

        strategy = cast(DefaultStrategy, self.strategy)
        self.strategy = DinoDefaultStrategy(
            prune_opa=strategy.prune_opa,
            grow_grad2d=strategy.grow_grad2d,
            grow_scale3d=strategy.grow_scale3d,
            grow_scale2d=strategy.grow_scale2d,
            prune_scale3d=strategy.prune_scale3d,
            prune_scale2d=strategy.prune_scale2d,
            refine_scale2d_stop_iter=strategy.refine_scale2d_stop_iter,
            refine_start_iter=strategy.refine_start_iter,
            refine_stop_iter=strategy.refine_stop_iter,
            reset_every=strategy.reset_every,
            refine_every=strategy.refine_every,
            pause_refine_after_reset=strategy.pause_refine_after_reset,
            absgrad=strategy.absgrad,
            revised_opacity=strategy.revised_opacity,
            verbose=strategy.verbose,
            key_for_gradient=strategy.key_for_gradient,
            model_ref=self,
        )
        scene_scale = 1.0
        if isinstance(self.strategy_state, dict) and "scene_scale" in self.strategy_state:
            scene_scale = float(self.strategy_state["scene_scale"])
        self.strategy_state = self.strategy.initialize_state(scene_scale=scene_scale)

    @torch.no_grad()
    def _clone_gaussians(self, mask: torch.Tensor, state: Dict[str, Any]) -> None:
        duplicate(params=self.gauss_params, optimizers=self.optimizers, state=state, mask=mask)

    @torch.no_grad()
    def _split_gaussians(self, mask: torch.Tensor, state: Dict[str, Any], revised_opacity: bool = False) -> None:
        split(params=self.gauss_params, optimizers=self.optimizers, state=state, mask=mask, revised_opacity=revised_opacity)

    @torch.no_grad()
    def _cull_gaussians(self, mask: torch.Tensor, state: Dict[str, Any]) -> None:
        remove(params=self.gauss_params, optimizers=self.optimizers, state=state, mask=mask)

    def _select_render_tensors(
        self, crop_ids: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if crop_ids is None:
            return self.means, self.quats, self.scales, self.opacities, self.dino_features, self.features_dc
        return (
            self.means[crop_ids],
            self.quats[crop_ids],
            self.scales[crop_ids],
            self.opacities[crop_ids],
            self.dino_features[crop_ids],
            self.features_dc[crop_ids],
        )

    def _semantic_rgb_from_features(self, rendered_features: torch.Tensor) -> torch.Tensor:
        semantic = rendered_features[..., :3]
        if semantic.shape[-1] < 3:
            pad = torch.zeros((*semantic.shape[:2], 3 - semantic.shape[-1]), device=semantic.device, dtype=semantic.dtype)
            semantic = torch.cat([semantic, pad], dim=-1)
        min_vals = semantic.amin(dim=(0, 1), keepdim=True)
        max_vals = semantic.amax(dim=(0, 1), keepdim=True)
        semantic = (semantic - min_vals) / (max_vals - min_vals).clamp_min(1e-6)
        return semantic.clamp(0.0, 1.0)

    def _prepare_gt_dino_feature(self, gt_feature: torch.Tensor, pred_hw: Tuple[int, int]) -> torch.Tensor:
        gt = gt_feature
        cfg = self._dino_config()
        if gt.ndim == 4 and gt.shape[0] == 1:
            gt = gt.squeeze(0)
        if gt.ndim != 3:
            raise ValueError(f"Expected GT dino feature to be 3D tensor, got shape {tuple(gt.shape)}")
        if gt.shape[0] == cfg.dino_feature_dim and gt.shape[-1] != cfg.dino_feature_dim:
            gt = gt.permute(1, 2, 0).contiguous()
        if gt.shape[:2] != pred_hw:
            gt = F.interpolate(
                gt.permute(2, 0, 1).unsqueeze(0), size=pred_hw, mode="bilinear", align_corners=False
            ).squeeze(0)
            gt = gt.permute(1, 2, 0).contiguous()
        return gt
