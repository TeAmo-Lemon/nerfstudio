"""3D Gaussian Splatting + DINO features + LightGaussian compression.

Self-contained, no parent class. Read top-to-bottom.

Combines:
    - DINO feature splatting (from dino_gaussian_model.py)
    - LightGaussian compression: importance-based pruning + SH distillation
      (from lightgaussian_model.py)

Flow:
    init → setup gaussians + DINO features + DINO-aware strategy →
    [training loop]:
        prepare_train_step →
        get_outputs (RGB + DINO + optional teacher render) →
        get_loss_dict (L1 + SSIM + scale_reg + dino_feature_loss + distill_loss) →
        get_metrics_dict (PSNR + dino_l1) →
        finish_train_step (DINO-aware densify/prune + importance-prune +
                           distill mode entry)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union, cast

import torch
import torch.nn.functional as F
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import remove
from pytorch_msssim import SSIM
from torch import nn
from torch.nn import Parameter

from cameras.camera_optimizer import CameraOptimizer, CameraOptimizerConfig
from cameras.cameras import Cameras
from cameras.rays import RayBundle
from configs.base_config import InstantiateConfig
from data.scene_box import OrientedBox, SceneBox
from engine.optimizers import Optimizers
from gaussian_renderer.renderer import (
    apply_bilateral_grid,
    composite_with_background,
    downscale_if_required,
    get_background_color,
    get_downscale_factor,
    get_empty_outputs,
    get_viewmat,
    prepare_gt_image,
    render_gaussians,
)
from scene.dino_gaussian_model import DinoDefaultStrategy
from utils import writer as writer_module
from utils.colors import get_color
from utils.graphics_utils import BilateralGrid, color_correct, total_variation_loss
from utils.math_utils import k_nearest_sklearn, random_quat_tensor
from utils.sh_utils import RGB2SH, SH2RGB, num_sh_bases
from utils.system_utils import CONSOLE


# ═══════════════════════════════════════════════════════════════════════════════
# config
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class DinoLightGaussianModelConfig(InstantiateConfig):
    """Configuration for 3DGS + DINO features + LightGaussian compression."""

    _target: Type = field(default_factory=lambda: DinoLightGaussianModel)

    # ── base gaussian params (same as SplatfactoModelConfig) ─────────
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

    # ── DINO-specific ───────────────────────────────────────────────
    dino_feature_dim: int = 16
    dino_loss_weight: float = 1.0
    dino_loss_start_step: int = 3000
    style_primitive_path: Optional[Path] = None
    num_style_primitives: int = 8

    # ── LightGaussian-specific ───────────────────────────────────────
    prune_iterations: List[int] = field(default_factory=lambda: [15000, 22000])
    """Steps at which to run importance-based pruning."""
    prune_percent: float = 0.5
    """Fraction of Gaussians to prune on the first prune round."""
    prune_decay: float = 0.8
    """Decay factor for subsequent prune rounds (prune_pct * decay^i)."""
    v_pow: float = 0.1
    """Volume power exponent for v_importance score adjustment."""
    prune_type: Literal["importance", "v_importance"] = "v_importance"
    """'importance' uses raw multi-view score; 'v_importance' multiplies by volume factor."""
    num_importance_views: int = 0
    """Number of training views to use for importance scoring (0 = all)."""
    enable_sh_distillation: bool = False
    """If True, run SH distillation phase after main training."""
    distill_start_iter: int = 25000
    """Step at which SH distillation begins."""
    distill_sh_degree: int = 2
    """Target SH degree after distillation."""
    freeze_covariance_during_distill: bool = False
    """If True, freeze scales and rotations during distillation."""
    distill_loss_weight: float = 1.0
    """Weight for the distillation L1+SSIM loss."""


# ═══════════════════════════════════════════════════════════════════════════════
# model
# ═══════════════════════════════════════════════════════════════════════════════


class DinoLightGaussianModel(nn.Module):
    """3DGS + DINO features + LightGaussian compression — standalone implementation.

    Flow:
        init → setup gaussians + DINO features + DINO-aware strategy →
        [training loop]:
            prepare_train_step →
            get_outputs:
                ① RGB rasterization (gsplat, with SH)
                ② DINO feature rasterization (gsplat, no SH)
                ③ Optional teacher render (during distill phase)
            get_loss_dict:
                ① L1 + SSIM RGB loss
                ② Scale regularization
                ③ DINO feature L1 loss (enabled after dino_loss_start_step)
                ④ Distillation loss (during distill phase)
            finish_train_step:
                ① DINO-aware densify/prune (via DinoDefaultStrategy)
                ② Importance-based pruning at prune_iterations
                ③ Enter distillation mode at distill_start_iter
                ④ DINO feature count sync check
    """

    config: DinoLightGaussianModelConfig

    # ── init ──────────────────────────────────────────────────────────────

    def __init__(
        self,
        config: DinoLightGaussianModelConfig,
        scene_box: SceneBox,
        num_train_data: int,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.scene_box = scene_box
        self.num_train_data = num_train_data
        self.seed_points = seed_points

        # ── gaussian parameters ──────────────────────────────────────
        self.gauss_params = _init_gaussian_params(config, seed_points)

        # DINO features — one feature vector per gaussian
        num_gs = self.gauss_params["means"].shape[0]
        self.gauss_params["dino_features"] = nn.Parameter(
            0.01 * torch.randn(num_gs, config.dino_feature_dim, device=self.gauss_params["means"].device)
        )

        # ── camera optimizer ──────────────────────────────────────────
        self.camera_optimizer: CameraOptimizer = config.camera_optimizer.setup(
            num_cameras=num_train_data, device="cpu"
        )

        # ── metrics ───────────────────────────────────────────────────
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # ── state ─────────────────────────────────────────────────────
        self.step = 0
        self.crop_box: Optional[OrientedBox] = None
        self.collider = None
        self.render_aabb: Optional[SceneBox] = None
        self.callbacks = None
        self.info: Dict = {}
        self.background_color: Optional[torch.Tensor] = None

        # ── bilateral grid ────────────────────────────────────────────
        self.bil_grids: Optional[BilateralGrid] = None
        if config.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                num=num_train_data,
                grid_X=config.grid_shape[0],
                grid_Y=config.grid_shape[1],
                grid_W=config.grid_shape[2],
            )

        # ── densification strategy (DINO-aware) ───────────────────────
        base_strategy, base_state = _init_strategy(config, num_train_data)
        if isinstance(base_strategy, DefaultStrategy):
            self.strategy = DinoDefaultStrategy(
                prune_opa=base_strategy.prune_opa,
                grow_grad2d=base_strategy.grow_grad2d,
                grow_scale3d=base_strategy.grow_scale3d,
                grow_scale2d=base_strategy.grow_scale2d,
                prune_scale3d=base_strategy.prune_scale3d,
                prune_scale2d=base_strategy.prune_scale2d,
                refine_scale2d_stop_iter=base_strategy.refine_scale2d_stop_iter,
                refine_start_iter=base_strategy.refine_start_iter,
                refine_stop_iter=base_strategy.refine_stop_iter,
                reset_every=base_strategy.reset_every,
                refine_every=base_strategy.refine_every,
                pause_refine_after_reset=base_strategy.pause_refine_after_reset,
                absgrad=base_strategy.absgrad,
                revised_opacity=base_strategy.revised_opacity,
                verbose=base_strategy.verbose,
                key_for_gradient=base_strategy.key_for_gradient,
                model_ref=self,
            )
            scene_scale = 1.0
            if isinstance(base_state, dict) and "scene_scale" in base_state:
                scene_scale = float(base_state["scene_scale"])
            self.strategy_state = self.strategy.initialize_state(scene_scale=scene_scale)
        else:
            self.strategy = base_strategy
            self.strategy_state = base_state

        # ── style primitives (optional) ───────────────────────────────
        if config.style_primitive_path is not None and config.style_primitive_path.exists():
            payload = torch.load(config.style_primitive_path, map_location="cpu", weights_only=True)
            primitives = payload.get("prototypes")
            if primitives is not None:
                self.register_buffer("style_primitives", primitives, persistent=False)
                CONSOLE.log(f"Loaded style primitives: {tuple(primitives.shape)}")
            else:
                CONSOLE.log("[yellow]style_primitives.pt found but missing 'prototypes' key[/yellow]")

        # ── LightGaussian state ───────────────────────────────────────
        self.train_cameras: Optional[Cameras] = None
        self.teacher_params: Optional[nn.ParameterDict] = None
        self.in_distill_phase: bool = False
        self._prune_round: int = 0

        self.device_indicator_param = nn.Parameter(torch.empty(0))

    # ── training camera setup ────────────────────────────────────────────

    def set_train_cameras(self, cameras: Cameras) -> None:
        """Store training cameras reference for importance score computation."""
        self.train_cameras = cameras

    # ── core rendering ───────────────────────────────────────────────────

    def forward(self, ray_bundle: Union[RayBundle, Cameras]) -> Dict[str, Union[torch.Tensor, List]]:
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)
        return self.get_outputs(ray_bundle)

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Render RGB + depth + DINO feature map + optional teacher RGB.

        Three passes:
        1. RGB with SH (standard gsplat)
        2. DINO features — treated as flat colors, no SH
        3. Teacher RGB (during distill phase, no grad)
        """
        # ① camera guard
        if not isinstance(camera, Cameras):
            return {}

        # ② pose refinement
        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_c2w = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_c2w = camera.camera_to_worlds

        # ③ crop filter
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                bg, _ = get_background_color(self.config.background_color, False, self.device, self.background_color)
                return get_empty_outputs(int(camera.width.item()), int(camera.height.item()), bg)
        else:
            crop_ids = None

        # ④ select gaussians
        if crop_ids is None:
            means, quats, scales, opacities = (
                self.means, self.quats, self.scales, self.opacities,
            )
            features_dc, features_rest, dino_features = (
                self.features_dc, self.features_rest, self.dino_features,
            )
        else:
            means = self.means[crop_ids]
            quats = self.quats[crop_ids]
            scales = self.scales[crop_ids]
            opacities = self.opacities[crop_ids]
            features_dc = self.features_dc[crop_ids]
            features_rest = self.features_rest[crop_ids]
            dino_features = self.dino_features[crop_ids]

        # ⑤ prepare shared render params
        dsf = get_downscale_factor(self.config.num_downscales, self.step, self.config.resolution_schedule, self.training)
        camera.rescale_output_resolution(1 / dsf)
        viewmat = get_viewmat(optimized_c2w)
        K = camera.get_intrinsics_matrices().to(self.device)
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(dsf)

        # ── pass 1: RGB ──────────────────────────────────────────────────
        colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors = torch.sigmoid(colors).squeeze(1)
            sh_degree = None

        render, alpha, self.info = render_gaussians(
            means=means, quats=quats, scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities), colors=colors,
            viewmat=viewmat, K=K, width=W, height=H, sh_degree=sh_degree,
            render_mode=render_mode,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
        )

        if self.training:
            self.strategy.step_pre_backward(self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info)
        alpha = alpha[:, ...]

        # alpha composite
        background, self.background_color = get_background_color(
            self.config.background_color, self.training, self.device, self.background_color,
        )
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # bilateral grid
        if self.config.use_bilateral_grid and self.training and self.bil_grids is not None:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = apply_bilateral_grid(self.bil_grids, rgb, camera.metadata["cam_idx"], H, W, self.device)

        # depth
        if render_mode == "RGB+ED":
            depth = torch.where(alpha > 0, render[:, ..., 3:4], render[:, ..., 3:4].detach().max()).squeeze(0)
        else:
            depth = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        outputs: Dict[str, Union[torch.Tensor, List]] = {
            "rgb": rgb.squeeze(0),
            "depth": depth,
            "accumulation": alpha.squeeze(0),
            "background": background,
        }

        # ── pass 2: DINO features ─────────────────────────────────────────
        rendered_features, _, _ = render_gaussians(
            means=means, quats=quats, scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities), colors=dino_features,
            viewmat=viewmat, K=K, width=W, height=H,
            sh_degree=None,  # features don't use SH
            render_mode="RGB",
            absgrad=False,
            rasterize_mode=self.config.rasterize_mode,
        )
        rendered_features = rendered_features.squeeze(0)
        outputs["rendered_features"] = rendered_features
        outputs["dino_semantic_rgb"] = _semantic_rgb_from_features(rendered_features)

        # ── pass 3: teacher render (during distill phase) ──────────────────
        if self.in_distill_phase and self.teacher_params is not None:
            with torch.no_grad():
                teacher_rgb = self._render_teacher(
                    crop_ids=crop_ids, optimized_c2w=optimized_c2w,
                    viewmat=viewmat, K=K, W=W, H=H, background=background,
                )
            outputs["teacher_rgb"] = teacher_rgb

        return outputs

    def _render_teacher(
        self,
        crop_ids: Optional[torch.Tensor],
        optimized_c2w: torch.Tensor,
        viewmat: torch.Tensor,
        K: torch.Tensor,
        W: int,
        H: int,
        background: torch.Tensor,
    ) -> torch.Tensor:
        """Render the frozen teacher model from the same viewpoint."""
        tp = self.teacher_params
        if crop_ids is None:
            t_means, t_quats, t_scales, t_opacities = (
                tp["means"], tp["quats"], tp["scales"], tp["opacities"]
            )
            t_dc, t_rest = tp["features_dc"], tp["features_rest"]
        else:
            t_means = tp["means"][crop_ids]
            t_quats = tp["quats"][crop_ids]
            t_scales = tp["scales"][crop_ids]
            t_opacities = tp["opacities"][crop_ids]
            t_dc = tp["features_dc"][crop_ids]
            t_rest = tp["features_rest"][crop_ids]

        t_colors = torch.cat((t_dc[:, None, :], t_rest), dim=1)
        full_sh = self.config.sh_degree

        t_render, t_alpha, _ = render_gaussians(
            means=t_means,
            quats=t_quats,
            scales=torch.exp(t_scales),
            opacities=torch.sigmoid(t_opacities),
            colors=t_colors,
            viewmat=viewmat,
            K=K,
            width=W,
            height=H,
            sh_degree=full_sh,
            render_mode="RGB",
            rasterize_mode=self.config.rasterize_mode,
        )
        t_alpha = t_alpha[:, ...]
        t_rgb = t_render[:, ..., :3] + (1 - t_alpha) * background
        return torch.clamp(t_rgb, 0.0, 1.0).squeeze(0)

    # ── training ─────────────────────────────────────────────────────────

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """L1 + SSIM + scale_reg + DINO feature L1 + distill loss."""
        gt_img = composite_with_background(
            prepare_gt_image(
                batch["image"], self.config.num_downscales, self.step,
                self.config.resolution_schedule, True, self.device,
            ),
            outputs["background"],
        )
        pred_img = outputs["rgb"]

        if "mask" in batch:
            mask = downscale_if_required(
                batch["mask"], self.config.num_downscales, self.step,
                self.config.resolution_schedule, True,
            ).to(self.device)
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        # L1 + SSIM
        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])

        # scale regularization
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            ratio = scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1)
            scale_reg = 0.1 * torch.maximum(ratio, torch.tensor(self.config.max_gauss_ratio, device=self.device)).sub(self.config.max_gauss_ratio).mean()
        else:
            scale_reg = torch.tensor(0.0, device=self.device)

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
        }

        # MCMC regularizers
        if self.config.strategy == "mcmc":
            if self.config.mcmc_opacity_reg > 0.0:
                loss_dict["mcmc_opacity_reg"] = self.config.mcmc_opacity_reg * torch.abs(torch.sigmoid(self.opacities)).mean()
            if self.config.mcmc_scale_reg > 0.0:
                loss_dict["mcmc_scale_reg"] = self.config.mcmc_scale_reg * torch.abs(torch.exp(self.scales)).mean()

        # camera + bilateral TV
        if self.training:
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid and self.bil_grids is not None:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        # ── DINO feature loss (enabled after start step) ──────────────────
        if self.step >= self.config.dino_loss_start_step:
            if "rendered_features" in outputs and "dino_feature" in batch:
                rendered_feat = outputs["rendered_features"]
                gt_feat = batch["dino_feature"]
                if isinstance(rendered_feat, torch.Tensor) and isinstance(gt_feat, torch.Tensor):
                    pred_hw = (int(rendered_feat.shape[0]), int(rendered_feat.shape[1]))
                    gt = _prepare_gt_dino_feature(gt_feat.to(self.device, dtype=rendered_feat.dtype), pred_hw, self.config.dino_feature_dim)
                    common_dim = min(rendered_feat.shape[-1], gt.shape[-1])
                    if common_dim > 0:
                        loss_dict["dino_feature_loss"] = self.config.dino_loss_weight * F.l1_loss(
                            rendered_feat[..., :common_dim], gt[..., :common_dim]
                        )

        # ── distillation loss ─────────────────────────────────────────────
        if self.in_distill_phase and "teacher_rgb" in outputs:
            teacher_rgb = outputs["teacher_rgb"]
            if isinstance(teacher_rgb, torch.Tensor):
                distill_l1 = torch.abs(pred_img - teacher_rgb).mean()
                distill_ssim = 1 - self.ssim(
                    teacher_rgb.permute(2, 0, 1)[None, ...],
                    pred_img.permute(2, 0, 1)[None, ...],
                )
                loss_dict["distill_loss"] = self.config.distill_loss_weight * (
                    (1 - self.config.ssim_lambda) * distill_l1 + self.config.ssim_lambda * distill_ssim
                )

        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """PSNR + gaussian count + DINO L1."""
        gt_rgb = composite_with_background(
            prepare_gt_image(
                batch["image"], self.config.num_downscales, self.step,
                self.config.resolution_schedule, self.training, self.device,
            ),
            outputs["background"],
        )
        metrics_dict: Dict[str, torch.Tensor] = {}
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], gt_rgb)
        if self.config.color_corrected_metrics:
            cc = color_correct(outputs["rgb"], gt_rgb)
            metrics_dict["cc_psnr"] = self.psnr(cc, gt_rgb)
        metrics_dict["gaussian_count"] = self.num_points
        self.camera_optimizer.get_metrics_dict(metrics_dict)

        # DINO feature L1
        if "rendered_features" in outputs and "dino_feature" in batch:
            rendered_feat = outputs["rendered_features"]
            gt_feat = batch["dino_feature"]
            if isinstance(rendered_feat, torch.Tensor) and isinstance(gt_feat, torch.Tensor):
                pred_hw = (int(rendered_feat.shape[0]), int(rendered_feat.shape[1]))
                gt = _prepare_gt_dino_feature(gt_feat.to(self.device, dtype=rendered_feat.dtype), pred_hw, self.config.dino_feature_dim)
                common_dim = min(rendered_feat.shape[-1], gt.shape[-1])
                if common_dim > 0:
                    metrics_dict["dino_l1"] = F.l1_loss(rendered_feat[..., :common_dim], gt[..., :common_dim]).detach()

        return metrics_dict

    # ── evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        self.set_crop(obb_box)
        return self.get_outputs(camera.to(self.device))

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = composite_with_background(
            prepare_gt_image(
                batch["image"], self.config.num_downscales, self.step,
                self.config.resolution_schedule, self.training, self.device,
            ),
            outputs["background"],
        )
        pred_rgb = outputs["rgb"]
        combined = torch.cat([gt_rgb, pred_rgb], dim=1)

        gt = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        pr = torch.moveaxis(pred_rgb, -1, 0)[None, ...]

        metrics_dict = {
            "psnr": float(self.psnr(gt, pr).item()),
            "ssim": float(self.ssim(gt, pr)),
            "lpips": float(self.lpips(gt, pr)),
        }
        if self.config.color_corrected_metrics:
            cc = torch.moveaxis(color_correct(pred_rgb, gt_rgb), -1, 0)[None, ...]
            metrics_dict["cc_psnr"] = float(self.psnr(gt, cc).item())
            metrics_dict["cc_ssim"] = float(self.ssim(gt, cc))
            metrics_dict["cc_lpips"] = float(self.lpips(gt, cc))

        return metrics_dict, {"img": combined}

    # ── training lifecycle ───────────────────────────────────────────────

    def prepare_train_step(self, optimizers: Optimizers, step: int) -> None:
        self.step = step
        self.optimizers = optimizers.optimizers
        self.schedulers = optimizers.schedulers

    def finish_train_step(self, step: int) -> None:
        assert step == self.step

        # Standard DINO-aware densify/prune
        if isinstance(self.strategy, DefaultStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params, optimizers=self.optimizers,
                state=self.strategy_state, step=step, info=self.info, packed=False,
            )
        elif isinstance(self.strategy, MCMCStrategy):
            self.strategy.step_post_backward(
                params=self.gauss_params, optimizers=self.optimizers,
                state=self.strategy_state, step=step, info=self.info,
                lr=self.schedulers["means"].get_last_lr()[0],
            )
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

        # ── LightGaussian: importance-based pruning ────────────────────
        if step in self.config.prune_iterations and self.train_cameras is not None:
            self._importance_prune(step)

        # ── LightGaussian: enter SH distillation phase ─────────────────
        if self.config.enable_sh_distillation and step == self.config.distill_start_iter:
            self._enter_distill_mode()

        # ── DINO logging ───────────────────────────────────────────────
        writer_module.put_scalar("Total GSs", int(self.means.shape[0]), step)

        s = getattr(self.strategy, "_last_densify_stats", None)
        if s is not None:
            writer_module.put_scalar("GS Duplicated", s["n_dupli"], step)
            writer_module.put_scalar("GS Split", s["n_split"], step)
            writer_module.put_scalar("GS Pruned", s["n_prune"], step)
            self.strategy._last_densify_stats = None

        # Sanity check: DINO features must stay in sync with gaussians
        if self.dino_features.shape[0] != self.means.shape[0]:
            raise RuntimeError(
                f"DINO feature count mismatch: {self.dino_features.shape[0]} != {self.means.shape[0]}"
            )

        # Free gsplat info tensors from the last render (~12 MB for 1M GS)
        self.info.clear()

    # ── LightGaussian: importance scoring ───────────────────────────────

    @torch.no_grad()
    def compute_importance_scores(self) -> torch.Tensor:
        """Compute per-Gaussian importance by rendering multiple training views.

        Renders at half resolution to reduce gsplat tile-buffer memory (~4×).
        Uses the RGB pass only (not DINO features).
        For each view: importance += opacity × radii² (∝ screen coverage).

        Returns:
            importance: tensor of shape (num_points,) with accumulated scores.
        """
        if self.train_cameras is None:
            raise RuntimeError("train_cameras not set. Call set_train_cameras() first.")

        num_points = self.num_points
        importance = torch.zeros(num_points, device=self.device)
        opacity = torch.sigmoid(self.opacities).squeeze(-1)

        cameras = self.train_cameras
        n_cams = len(cameras)

        if self.config.num_importance_views > 0 and self.config.num_importance_views < n_cams:
            indices = random.sample(range(n_cams), self.config.num_importance_views)
        else:
            indices = list(range(n_cams))

        sh_deg = self.config.sh_degree if self.config.sh_degree > 0 else None

        for idx in indices:
            # Render at half resolution — radii scale proportionally but ranking
            # is preserved; gsplat internal tile buffers shrink by ~4×.
            cam = cameras[idx:idx + 1].to(self.device)
            cam.rescale_output_resolution(0.5)
            viewmat = get_viewmat(cam.camera_to_worlds)
            K = cam.get_intrinsics_matrices().to(self.device)
            W, H = int(cam.width.item()), int(cam.height.item())

            if sh_deg is not None:
                colors = torch.cat((self.features_dc[:, None, :], self.features_rest), dim=1)
                _, _, info = render_gaussians(
                    means=self.means, quats=self.quats,
                    scales=torch.exp(self.scales),
                    opacities=opacity,
                    colors=colors,
                    viewmat=viewmat, K=K, width=W, height=H,
                    sh_degree=sh_deg,
                    render_mode="RGB",
                    rasterize_mode=self.config.rasterize_mode,
                )
            else:
                colors = torch.sigmoid(self.features_dc)
                _, _, info = render_gaussians(
                    means=self.means, quats=self.quats,
                    scales=torch.exp(self.scales),
                    opacities=opacity,
                    colors=colors,
                    viewmat=viewmat, K=K, width=W, height=H,
                    sh_degree=None,
                    render_mode="RGB",
                    rasterize_mode=self.config.rasterize_mode,
                )

            # Accumulate importance: opacity × radii² (radii ∝ screen coverage)
            radii = info["radii"]  # [1, N] when packed=False
            visible = radii > 0.0
            if visible.any():
                gs_ids = torch.where(visible.squeeze(0))[0]
                visible_radii = radii[visible].float()
                importance.index_add_(0, gs_ids, opacity[gs_ids] * visible_radii ** 2)

            # Release gsplat info tensors and CUDA cache aggressively
            del info, radii, visible
            if idx % 5 == 4:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        return importance

    def _importance_prune(self, step: int) -> None:
        """Compute importance scores and prune the lowest-scoring Gaussians."""
        n_before = int(self.means.shape[0])
        CONSOLE.log(f"[bold yellow]LightGaussian prune at step {step}:[/] {n_before} GS before prune")

        importance = self.compute_importance_scores()

        if self.config.prune_type == "v_importance":
            volume = torch.prod(torch.exp(self.scales), dim=1)
            kth_idx = int(len(volume) * 0.9)
            sorted_volume, _ = torch.sort(volume, descending=True)
            kth_largest = sorted_volume[kth_idx].clamp_min(1e-10)
            v_factor = torch.pow(volume / kth_largest, self.config.v_pow)
            score = v_factor * importance
        else:
            score = importance

        prune_frac = self.config.prune_percent * (self.config.prune_decay ** self._prune_round)
        self._prune_round += 1

        k = max(1, int(len(score) * (1 - prune_frac)))
        _, top_indices = torch.topk(score, k)
        mask = torch.ones(len(score), dtype=torch.bool, device=self.device)
        mask[top_indices] = False

        n_prune = int(mask.sum().item())
        if n_prune > 0:
            remove(
                params=self.gauss_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                mask=mask,
            )
        torch.cuda.empty_cache()

        n_after = int(self.means.shape[0])
        CONSOLE.log(
            f"[bold yellow]LightGaussian prune done:[/] pruned {n_prune} GS, "
            f"{n_after} remaining ({100 * n_after / max(1, n_before):.1f}%)"
        )

    # ── LightGaussian: SH distillation ──────────────────────────────────

    def _enter_distill_mode(self) -> None:
        """Save frozen teacher copy and reduce student SH degree."""
        CONSOLE.log(f"[bold cyan]Entering SH distillation at step {self.step}[/]")
        CONSOLE.log(f"  Teacher SH degree: {self.config.sh_degree}")
        CONSOLE.log(f"  Student SH degree: {self.config.distill_sh_degree}")

        self.teacher_params = nn.ParameterDict({
            name: nn.Parameter(p.data.clone(), requires_grad=False)
            for name, p in self.gauss_params.items() if name in ("means", "quats", "scales", "opacities", "features_dc", "features_rest")
        })

        if self.config.distill_sh_degree < self.config.sh_degree:
            new_dim = num_sh_bases(self.config.distill_sh_degree)
            keep_rest = new_dim - 1
            current_rest = self.gauss_params["features_rest"]
            self.gauss_params["features_rest"] = nn.Parameter(
                current_rest.data[:, :keep_rest, :].contiguous().clone()
            )
            CONSOLE.log(f"  features_rest: {tuple(current_rest.shape)} → {tuple(self.gauss_params['features_rest'].shape)}")

        if self.config.freeze_covariance_during_distill:
            for key in ["scales", "quats"]:
                self.gauss_params[key].requires_grad = False
            CONSOLE.log("  Frozen scales and quats during distillation")

        self.in_distill_phase = True

    # ── param groups ─────────────────────────────────────────────────────

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        gps = {"means": [self.means], "scales": [self.scales], "quats": [self.quats],
               "features_dc": [self.features_dc], "features_rest": [self.features_rest],
               "opacities": [self.opacities], "dino_features": [self.dino_features]}
        if self.config.use_bilateral_grid and self.bil_grids is not None:
            gps["bilateral_grid"] = list(self.bil_grids.parameters())
        self.camera_optimizer.get_param_groups(param_groups=gps)
        return gps

    # ── properties ───────────────────────────────────────────────────────

    @property
    def device(self):
        return self.device_indicator_param.device

    @property
    def means(self):    return self.gauss_params["means"]
    @property
    def scales(self):   return self.gauss_params["scales"]
    @property
    def quats(self):    return self.gauss_params["quats"]
    @property
    def features_dc(self):  return self.gauss_params["features_dc"]
    @property
    def features_rest(self): return self.gauss_params["features_rest"]
    @property
    def opacities(self): return self.gauss_params["opacities"]
    @property
    def dino_features(self): return self.gauss_params["dino_features"]
    @property
    def num_points(self): return self.means.shape[0]

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

    # ── serialization ────────────────────────────────────────────────────

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        self.step = 30000
        if "means" in dict:
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities", "dino_features"]:
                if p in dict:
                    dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = nn.Parameter(torch.zeros(newp, *param.shape[1:], device=self.device))
        super().load_state_dict(dict, **kwargs)

    def update_to_step(self, step: int) -> None:
        """Called when loading a checkpoint."""

    # ── helpers ──────────────────────────────────────────────────────────

    def get_rgba_image(self, outputs: Dict[str, torch.Tensor], output_name: str = "rgb") -> torch.Tensor:
        acc_name = output_name.replace("rgb", "accumulation")
        if acc_name not in outputs:
            raise NotImplementedError(f"get_rgba_image not implemented for {self.__class__.__name__}")
        rgb = outputs[output_name]
        acc = outputs[acc_name]
        if acc.dim() < rgb.dim():
            acc = acc.unsqueeze(-1)
        return torch.cat((rgb / acc.clamp(min=1e-10), acc), dim=-1)

    # ── setters ──────────────────────────────────────────────────────────

    def set_crop(self, crop_box: Optional[OrientedBox]) -> None:
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor) -> None:
        assert background_color.shape == (3,)
        self.background_color = background_color


# ═══════════════════════════════════════════════════════════════════════════════
# internal helpers (module-level functions)
# ═══════════════════════════════════════════════════════════════════════════════


def _init_gaussian_params(
    config: DinoLightGaussianModelConfig,
    seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]],
) -> nn.ParameterDict:
    """Initialize gaussian parameters from SfM seed points or random."""
    if seed_points is not None and not config.random_init:
        means = nn.Parameter(seed_points[0])
    else:
        means = nn.Parameter((torch.rand((config.num_random, 3)) - 0.5) * config.random_scale)

    distances, _ = k_nearest_sklearn(means.data, 3)
    avg_dist = distances.mean(dim=-1, keepdim=True)
    scales = nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
    num_points = means.shape[0]
    quats = nn.Parameter(random_quat_tensor(num_points))
    dim_sh = num_sh_bases(config.sh_degree)

    if seed_points is not None and not config.random_init and seed_points[1].shape[0] > 0:
        shs = torch.zeros((seed_points[1].shape[0], dim_sh, 3)).float().cuda()
        if config.sh_degree > 0:
            shs[:, 0, :3] = RGB2SH(seed_points[1] / 255)
            shs[:, 1:, 3:] = 0.0
        else:
            CONSOLE.log("use color only optimization with sigmoid activation")
            shs[:, 0, :3] = torch.logit(seed_points[1] / 255, eps=1e-10)
        features_dc = nn.Parameter(shs[:, 0, :])
        features_rest = nn.Parameter(shs[:, 1:, :])
    else:
        features_dc = nn.Parameter(torch.rand(num_points, 3))
        features_rest = nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

    opacities = nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))

    return nn.ParameterDict({
        "means": means, "scales": scales, "quats": quats,
        "features_dc": features_dc, "features_rest": features_rest, "opacities": opacities,
    })


def _init_strategy(config: DinoLightGaussianModelConfig, num_train_data: int):
    """Create base densification strategy (wrapped by DinoDefaultStrategy later)."""
    if config.strategy == "default":
        strategy = DefaultStrategy(
            prune_opa=config.cull_alpha_thresh,
            grow_grad2d=config.densify_grad_thresh,
            grow_scale3d=config.densify_size_thresh,
            grow_scale2d=config.split_screen_size,
            prune_scale3d=config.cull_scale_thresh,
            prune_scale2d=config.cull_screen_size,
            refine_scale2d_stop_iter=config.stop_screen_size_at,
            refine_start_iter=config.warmup_length,
            refine_stop_iter=config.stop_split_at,
            reset_every=config.reset_alpha_every * config.refine_every,
            refine_every=config.refine_every,
            pause_refine_after_reset=num_train_data + config.refine_every,
            absgrad=config.use_absgrad,
            revised_opacity=False,
            verbose=True,
        )
        state = strategy.initialize_state(scene_scale=1.0)
    elif config.strategy == "mcmc":
        from gsplat.strategy import MCMCStrategy
        strategy = MCMCStrategy(
            cap_max=config.max_gs_num,
            noise_lr=config.noise_lr,
            refine_start_iter=config.warmup_length,
            refine_stop_iter=config.stop_split_at,
            refine_every=config.refine_every,
            min_opacity=config.cull_alpha_thresh,
            verbose=False,
        )
        state = strategy.initialize_state()
    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")
    return strategy, state


def _semantic_rgb_from_features(features: torch.Tensor) -> torch.Tensor:
    """PCA-style 3-channel visualization of DINO features (min-max norm)."""
    semantic = features[..., :3]
    if semantic.shape[-1] < 3:
        pad = torch.zeros((*semantic.shape[:2], 3 - semantic.shape[-1]), device=semantic.device, dtype=semantic.dtype)
        semantic = torch.cat([semantic, pad], dim=-1)
    min_vals = semantic.amin(dim=(0, 1), keepdim=True)
    max_vals = semantic.amax(dim=(0, 1), keepdim=True)
    semantic = (semantic - min_vals) / (max_vals - min_vals).clamp_min(1e-6)
    return semantic.clamp(0.0, 1.0)


def _prepare_gt_dino_feature(
    gt_feature: torch.Tensor, pred_hw: Tuple[int, int], expected_dim: int,
) -> torch.Tensor:
    """Reshape and resize ground-truth DINO features to match predicted size."""
    gt = gt_feature
    if gt.ndim == 4 and gt.shape[0] == 1:
        gt = gt.squeeze(0)
    if gt.ndim != 3:
        raise ValueError(f"Expected GT dino feature to be 3D, got {tuple(gt.shape)}")
    if gt.shape[0] == expected_dim and gt.shape[-1] != expected_dim:
        gt = gt.permute(1, 2, 0).contiguous()
    if gt.shape[:2] != pred_hw:
        gt = F.interpolate(
            gt.permute(2, 0, 1).unsqueeze(0), size=pred_hw, mode="bilinear", align_corners=False,
        ).squeeze(0).permute(1, 2, 0).contiguous()
    return gt
