# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ...

"""
Gaussian Splatting implementation that combines many recent advancements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from gsplat.strategy import DefaultStrategy, MCMCStrategy

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from pytorch_msssim import SSIM
from torch import nn
from torch.nn import Parameter

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.data.scene_box import OrientedBox, SceneBox
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.model_components.lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import k_nearest_sklearn, random_quat_tensor
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases


def resize_image(image: torch.Tensor, d: int):
    """Downscale images using the same 'area' method in opencv.

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)
    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


def get_viewmat(optimized_camera_to_world):
    """Convert c2w to gsplat world2camera matrix."""
    R = optimized_camera_to_world[:, :3, :3]
    T = optimized_camera_to_world[:, :3, 3:4]
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


# ═══════════════════════════════════════════════════════════════════════════════
# SplatfactoModel
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SplatfactoModelConfig(InstantiateConfig):
    """Splatfacto Model Config — Gaussian Splatting."""

    _target: Type = field(default_factory=lambda: SplatfactoModel)
    enable_collider: bool = True
    collider_params: Optional[Dict[str, float]] = field(default_factory=lambda: {"near_plane": 2.0, "far_plane": 6.0})
    loss_coefficients: Dict[str, float] = field(default_factory=lambda: {"rgb_loss_coarse": 1.0, "rgb_loss_fine": 1.0})
    eval_num_rays_per_chunk: int = 4096
    prompt: Optional[str] = None
    warmup_length: int = 500
    refine_every: int = 100
    resolution_schedule: int = 3000
    background_color: Literal["random", "black", "white"] = "random"
    num_downscales: int = 1
    cull_alpha_thresh: float = 0.005
    cull_scale_thresh: float = 0.1
    reset_alpha_every: int = 30
    densify_grad_thresh: float = 0.0002
    use_absgrad: bool = False
    densify_size_thresh: float = 0.01
    n_split_samples: int = 2
    sh_degree_interval: int = 1000
    cull_screen_size: float = 0.15
    split_screen_size: float = 0.05
    stop_screen_size_at: int = 0
    random_init: bool = False
    num_random: int = 50000
    random_scale: float = 10.0
    ssim_lambda: float = 0.2
    stop_split_at: int = 15000
    sh_degree: int = 3
    use_scale_regularization: bool = True
    max_gauss_ratio: float = 10.0
    output_depth_during_training: bool = False
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    use_bilateral_grid: bool = False
    grid_shape: Tuple[int, int, int] = (16, 16, 8)
    color_corrected_metrics: bool = False
    strategy: Literal["default", "mcmc"] = "default"
    max_gs_num: int = 1_000_000
    noise_lr: float = 5e5
    mcmc_opacity_reg: float = 0.01
    mcmc_scale_reg: float = 0.01


class SplatfactoModel(nn.Module):
    """Nerfstudio's implementation of 3D Gaussian Splatting."""

    config: SplatfactoModelConfig

    def __init__(
        self,
        config: SplatfactoModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.scene_box = scene_box
        self.render_aabb: Optional[SceneBox] = None
        self.num_train_data = num_train_data
        self.kwargs = kwargs
        self.seed_points = seed_points
        self.collider = None
        self.populate_modules()
        self.callbacks = None
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    # ── setup ────────────────────────────────────────────────────────────

    def populate_modules(self):
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
        distances, _ = k_nearest_sklearn(means.data, 3)
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        if (
            self.seed_points is not None
            and not self.config.random_init
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict({
            "means": means,
            "scales": scales,
            "quats": quats,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "opacities": opacities,
        })

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor([0.1490, 0.1647, 0.2157])
        else:
            self.background_color = get_color(self.config.background_color)

        if self.config.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                num=self.num_train_data,
                grid_X=self.config.grid_shape[0],
                grid_Y=self.config.grid_shape[1],
                grid_W=self.config.grid_shape[2],
            )

        if self.config.strategy == "default":
            self.strategy = DefaultStrategy(
                prune_opa=self.config.cull_alpha_thresh,
                grow_grad2d=self.config.densify_grad_thresh,
                grow_scale3d=self.config.densify_size_thresh,
                grow_scale2d=self.config.split_screen_size,
                prune_scale3d=self.config.cull_scale_thresh,
                prune_scale2d=self.config.cull_screen_size,
                refine_scale2d_stop_iter=self.config.stop_screen_size_at,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                reset_every=self.config.reset_alpha_every * self.config.refine_every,
                refine_every=self.config.refine_every,
                pause_refine_after_reset=self.num_train_data + self.config.refine_every,
                absgrad=self.config.use_absgrad,
                revised_opacity=False,
                verbose=True,
            )
            self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
        elif self.config.strategy == "mcmc":
            self.strategy = MCMCStrategy(
                cap_max=self.config.max_gs_num,
                noise_lr=self.config.noise_lr,
                refine_start_iter=self.config.warmup_length,
                refine_stop_iter=self.config.stop_split_at,
                refine_every=self.config.refine_every,
                min_opacity=self.config.cull_alpha_thresh,
                verbose=False,
            )
            self.strategy_state = self.strategy.initialize_state()
        else:
            raise ValueError(f"Splatfacto does not support strategy {self.config.strategy}")

    # ── core rendering ───────────────────────────────────────────────────

    def forward(self, ray_bundle: Union[RayBundle, Cameras]) -> Dict[str, Union[torch.Tensor, List]]:
        """Run forward: collider filter → get_outputs."""
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)
        return self.get_outputs(ray_bundle)

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """为给定的相机渲染高斯泼溅图像。

        完整流程:
        1. 相机类型检查
        2. 训练时对相机外参施加优化偏移量（camera optimizer）
        3. 评估时按裁剪框（crop_box）过滤高斯球，若为空则返回空白图
        4. 选出活跃的高斯基元参数（位置、缩放、旋转、颜色、不透明度）
        5. 将 SH 系数拼接为颜色特征（DC + 高阶带）
        6. 按下采样因子缩放相机分辨率 → 计算视图矩阵 V 和内参 K
        7. 选择渲染模式：训练时可仅输出 RGB（节省计算），否则输出 RGB+ED（额外深度通道）
        8. 随训练步数渐进提升 SH 阶数（sh_degree warm-up）
        9. 调用 CUDA rasterization 做前向泼溅渲染
        10. 训练时执行策略的 pre-backward 钩子（如 densify/prune 梯度统计）
        11. Alpha 合成：RGB = 渲染颜色 + (1 - alpha) * 背景色
        12. 可选应用双边网格（bilateral grid）微调颜色
        13. 提取深度图（若有 ED 通道），对无高斯的区域用最大深度填充
        14. 返回 {"rgb", "depth", "accumulation", "background"}
        """
        # 步骤 1: 确保输入是 Cameras 对象
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        # 步骤 2: 训练时对相机外参施加可学习的微小偏移（pose refinement）
        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # 步骤 3: 评估时按裁剪框过滤，减少不必要的渲染计算
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()  # 判断每个高斯球心是否在框内
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        # 步骤 4: 选出需要参与渲染的高斯参数（被 crop 的取子集，否则取全部）
        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        # 步骤 5: 拼接 SH 系数 — DC（第 0 阶）+ 高阶带 → 每个球有 (1 + sh_degree)² 个 RGB 分量
        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        # 步骤 6: 计算下采样后的视图矩阵 V、内参矩阵 K、输出分辨率 (W, H)
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)  # world→camera 的 4×4 外参矩阵
        K = camera.get_intrinsics_matrices().cuda()        # 3×3 内参矩阵
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # 恢复原始分辨率，避免副作用

        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        # 步骤 7: 选择渲染模式 — RGB+ED 额外输出期望深度（Expected Depth）通道，用于深度损失或可视化
        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        # 步骤 8: SH 阶数渐进式 warm-up — 随训练步数从 0 阶逐步提升到目标阶数，稳定训练初期
        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # 无 SH：直接 sigmoid 得到 RGB
            sh_degree_to_use = None

        # 步骤 9: CUDA 光栅化 — 将 3D 高斯投影到屏幕空间，按深度排序后 alpha-blend
        render, alpha, self.info = rasterization(  # type: ignore[reportPossiblyUnboundVariable]
            means=means_crop,                          # 球心位置 (N, 3)
            quats=quats_crop,                          # 旋转四元数 (N, 4)
            scales=torch.exp(scales_crop),             # 缩放因子，exp 保证正值
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),  # 不透明度，sigmoid 压到 [0,1]
            colors=colors_crop,                        # RGB 颜色（可能含 SH 系数）
            viewmats=viewmat,                          # 视图矩阵
            Ks=K,                                      # 内参矩阵
            width=W,                                   # 输出图像宽度
            height=H,                                  # 输出图像高度
            packed=False,                              # 单相机模式，非多相机打包
            near_plane=0.01,                           # 近裁剪面
            far_plane=1e10,                            # 远裁剪面
            render_mode=render_mode,                   # "RGB" 或 "RGB+ED"
            sh_degree=sh_degree_to_use,                # 当前激活的 SH 阶数
            sparse_grad=False,                         # 稠密梯度（对所有投影像素回传）
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode, # "classic" 或 "antialiased"
        )
        # 步骤 10: 训练时执行策略钩子 — 收集梯度统计，用于后续 densify/prune 决策
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )
        alpha = alpha[:, ...]  # 确保 alpha 维度一致

        # 步骤 11: Alpha 合成 — 渲染色与背景色按透射率混合
        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # 步骤 12: 可选应用双边网格 — 在高分辨率特征网格上做颜色微调，提升细节
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        # 步骤 13: 提取深度通道 — 若无高斯覆盖的区域，用场景最大深度填充（避免黑色空洞）
        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        # 评估时若背景是单色 (3,)，广播到图像尺寸以便后续拼接
        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        # 步骤 14: 返回渲染结果字典
        return {
            "rgb": rgb.squeeze(0),          # (H, W, 3) — 渲染 RGB
            "depth": depth_im,              # (H, W, 1) 或 None — 期望深度
            "accumulation": alpha.squeeze(0),  # (H, W, 1) — 透射率/累积 alpha
            "background": background,        # (3,) 或 (H, W, 3) — 使用的背景色
        }

    # ── training loss & metrics ──────────────────────────────────────────

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Compute L1 + SSIM + regularization losses."""
        # ─────────────────────────────────────────────────────
        # 1) 准备：合成带背景的 GT 图与预测图，并可选应用掩码
        # ─────────────────────────────────────────────────────
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        if "mask" in batch:
            mask = self._downscale_if_required(batch["mask"]).to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        # ─────────────────────────────────────────────────────
        # 2) 基本像素损失：L1 与 SSIM
        # ─────────────────────────────────────────────────────
        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])

        # ─────────────────────────────────────────────────────
        # 3) 尺度正则化（按步长节流，每 10 步计算一次）
        # ─────────────────────────────────────────────────────
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        # ─────────────────────────────────────────────────────
        # 4) 汇总损失字典（主损失 + 正则项）
        # ─────────────────────────────────────────────────────
        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
        }

        # ─────────────────────────────────────────────────────
        # 5) 策略相关的额外正则项（例如 MCMC 的不透明度/尺度正则）
        # ─────────────────────────────────────────────────────
        if self.config.strategy == "mcmc":
            if self.config.mcmc_opacity_reg > 0.0:
                loss_dict["mcmc_opacity_reg"] = (
                    self.config.mcmc_opacity_reg * torch.abs(torch.sigmoid(self.gauss_params["opacities"])).mean()
                )
            if self.config.mcmc_scale_reg > 0.0:
                loss_dict["mcmc_scale_reg"] = (
                    self.config.mcmc_scale_reg * torch.abs(torch.exp(self.gauss_params["scales"])).mean()
                )

        # ─────────────────────────────────────────────────────
        # 6) 训练时的额外损失：相机优化、双边网格的 TV 正则
        # ─────────────────────────────────────────────────────
        if self.training:
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        # ─────────────────────────────────────────────────────
        # 完成：返回损失字典
        # ─────────────────────────────────────────────────────
        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute PSNR and related metrics."""
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict: Dict[str, torch.Tensor] = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb)
        metrics_dict["gaussian_count"] = self.num_points
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    # ── evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """为给定相机渲染完整图像（仅评估模式）。

        对 get_outputs 的轻量封装，额外做了：
        1. 设置裁剪框（crop_box），用于交互式查看器的框选渲染
        2. 将相机移到模型所在设备
        3. 整个调用在 torch.no_grad() 下执行，不构建计算图
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)  # 设置可能为 None 的裁剪框（None 即取消裁剪）
        return self.get_outputs(camera.to(self.device))  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Compute PSNR/SSIM/LPIPS and produce a side-by-side comparison image."""
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]
        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        cc_rgb = None
        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]

        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}

        if self.config.color_corrected_metrics:
            assert cc_rgb is not None
            metrics_dict["cc_psnr"] = float(self.psnr(gt_rgb, cc_rgb).item())
            metrics_dict["cc_ssim"] = float(self.ssim(gt_rgb, cc_rgb))
            metrics_dict["cc_lpips"] = float(self.lpips(gt_rgb, cc_rgb))

        return metrics_dict, {"img": combined_rgb}

    # ── helpers ──────────────────────────────────────────────────────────

    def get_rgba_image(self, outputs: Dict[str, torch.Tensor], output_name: str = "rgb") -> torch.Tensor:
        """Composite RGBA from rendered RGB and accumulation mask."""
        accumulation_name = output_name.replace("rgb", "accumulation")
        if accumulation_name not in outputs:
            raise NotImplementedError(f"get_rgba_image is not implemented for model {self.__class__.__name__}")
        rgb = outputs[output_name]
        if self.renderer_rgb.background_color == "random":
            acc = outputs[accumulation_name]
            if acc.dim() < rgb.dim():
                acc = acc.unsqueeze(-1)
            return torch.cat((rgb / acc.clamp(min=1e-10), acc), dim=-1)
        return torch.cat((rgb, torch.ones_like(rgb[..., :1])), dim=-1)

    @property
    def device(self):
        return self.device_indicator_param.device

    def update_to_step(self, step: int) -> None:
        """Called when loading a checkpoint — update step-dependent state."""

    # ── training lifecycle ───────────────────────────────────────────────

    def prepare_train_step(self, optimizers: Optimizers, step):
        """Cache current step, optimizer, and scheduler handles."""
        self.step = step
        self.optimizers = optimizers.optimizers
        self.schedulers = optimizers.schedulers

    def finish_train_step(self, step):
        """Run post-backward strategy step (densify / prune)."""
        assert step == self.step
        if isinstance(self.strategy, DefaultStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=self.step,
                info=self.info,
                packed=False,
            )
        elif isinstance(self.strategy, MCMCStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=self.info,
                lr=self.schedulers["means"].get_last_lr()[0],
            )
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    # ── optimizer param groups ───────────────────────────────────────────

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        gps = self.get_gaussian_param_groups()
        if self.config.use_bilateral_grid:
            gps["bilateral_grid"] = list(self.bil_grids.parameters())
        self.camera_optimizer.get_param_groups(param_groups=gps)
        return gps

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        names = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        return {name: [self.gauss_params[name]] for name in names}

    # ── properties ───────────────────────────────────────────────────────

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        if self.config.sh_degree > 0:
            return self.features_dc
        return RGB2SH(torch.sigmoid(self.features_dc))

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.means.shape[0]

    @property
    def means(self):
        return self.gauss_params["means"]

    @property
    def scales(self):
        return self.gauss_params["scales"]

    @property
    def quats(self):
        return self.gauss_params["quats"]

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]

    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    # ── serialization ────────────────────────────────────────────────────

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        self.step = 30000
        if "means" in dict:
            legacy_params = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
            for p in legacy_params:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    # ── private helpers ──────────────────────────────────────────────────

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule), 0
            )
        return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                return torch.rand(3, device=self.device)
            return self.background_color.to(self.device)
        elif self.config.background_color == "white":
            return torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            return torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")

    def _apply_bilateral_grid(self, rgb: torch.Tensor, cam_idx: int, H: int, W: int) -> torch.Tensor:
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1.0, H, device=self.device),
            torch.linspace(0, 1.0, W, device=self.device),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        out = slice(
            bil_grids=self.bil_grids,
            rgb=rgb,
            xy=grid_xy,
            grid_idx=torch.tensor(cam_idx, device=self.device, dtype=torch.long),
        )
        return out["rgb"]

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def get_gt_img(self, image: torch.Tensor):
        """Downscale ground-truth image to match current training resolution."""
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        return self._downscale_if_required(image).to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """If the image has an alpha channel, composite it over *background*."""
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        return image

    # ── setters ──────────────────────────────────────────────────────────

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color
