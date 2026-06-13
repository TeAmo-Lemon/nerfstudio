"""3D Gaussian Splatting Model — splatfacto pipeline.

Self-contained, no parent class. Read top-to-bottom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from pytorch_msssim import SSIM
from torch import nn
from torch.nn import Parameter

from gsplat.strategy import DefaultStrategy, MCMCStrategy

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
from utils.colors import get_color
from utils.graphics_utils import BilateralGrid, color_correct, total_variation_loss
from utils.math_utils import k_nearest_sklearn, random_quat_tensor
from utils.sh_utils import RGB2SH, SH2RGB, num_sh_bases
from utils.system_utils import CONSOLE


# ═══════════════════════════════════════════════════════════════════════════════
# config
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SplatfactoModelConfig(InstantiateConfig):
    """Configuration for standard 3D Gaussian Splatting."""

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


# ═══════════════════════════════════════════════════════════════════════════════
# model
# ═══════════════════════════════════════════════════════════════════════════════


class SplatfactoModel(nn.Module):
    """3D Gaussian Splatting — standalone implementation.

    Flow:
        init → setup gaussians → [training loop]:
            prepare_train_step → forward/get_outputs → get_loss_dict →
            get_metrics_dict → finish_train_step (densify/prune)
    """

    config: SplatfactoModelConfig

    # ── init ──────────────────────────────────────────────────────────────

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
        self.num_train_data = num_train_data
        self.seed_points = seed_points
        self.kwargs = kwargs

        # ── gaussian parameters ──────────────────────────────────────
        self.gauss_params = _init_gaussian_params(config, seed_points)

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

        # ── densification strategy ────────────────────────────────────
        self.strategy, self.strategy_state = _init_strategy(config, num_train_data)

        self.device_indicator_param = nn.Parameter(torch.empty(0))

    # ── core rendering ───────────────────────────────────────────────────

    def forward(self, ray_bundle: Union[RayBundle, Cameras]) -> Dict[str, Union[torch.Tensor, List]]:
        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)
        return self.get_outputs(ray_bundle)

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Render RGB + depth from a camera viewpoint.

        Steps: ① camera check → ② pose refinement → ③ crop filter →
        ④ select active gaussians → ⑤ SH color → ⑥ viewmat/K → ⑦ pick render
        mode → ⑧ SH warm-up → ⑨ CUDA rasterization → ⑩ pre-backward hook →
        ⑪ alpha composite → ⑫ bilateral grid → ⑬ depth → ⑭ return
        """
        # ① camera guard
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        # ② pose refinement (train only)
        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_c2w = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_c2w = camera.camera_to_worlds

        # ③ crop box filter (eval only)
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                bg, _ = get_background_color(self.config.background_color, False, self.device, self.background_color)
                return get_empty_outputs(int(camera.width.item()), int(camera.height.item()), bg)
        else:
            crop_ids = None

        # ④ select active gaussians
        means, quats, scales, opacities, features_dc, features_rest = _select_gaussians(
            self.gauss_params, crop_ids
        )

        # ⑤ SH color
        colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)

        # ⑥ view matrix & intrinsics
        dsf = get_downscale_factor(self.config.num_downscales, self.step, self.config.resolution_schedule, self.training)
        camera.rescale_output_resolution(1 / dsf)
        viewmat = get_viewmat(optimized_c2w)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(dsf)

        # ⑦ render mode
        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        # ⑧ SH warm-up
        if self.config.sh_degree > 0:
            sh_degree = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors = torch.sigmoid(colors).squeeze(1)
            sh_degree = None

        # ⑨ CUDA rasterization
        render, alpha, self.info = render_gaussians(
            means=means,
            quats=quats,
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities),
            colors=colors,
            viewmat=viewmat,
            K=K,
            width=W,
            height=H,
            sh_degree=sh_degree,
            render_mode=render_mode,
            absgrad=self.strategy.absgrad if isinstance(self.strategy, DefaultStrategy) else False,
            rasterize_mode=self.config.rasterize_mode,
        )

        # ⑩ pre-backward hook (densify/prune gradients)
        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )
        alpha = alpha[:, ...]

        # ⑪ alpha composite
        background, self.background_color = get_background_color(
            self.config.background_color, self.training, self.device, self.background_color
        )
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # ⑫ bilateral grid
        if self.config.use_bilateral_grid and self.training and self.bil_grids is not None:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = apply_bilateral_grid(self.bil_grids, rgb, camera.metadata["cam_idx"], H, W, self.device)

        # ⑬ depth
        if render_mode == "RGB+ED":
            depth = torch.where(alpha > 0, render[:, ..., 3:4], render[:, ..., 3:4].detach().max()).squeeze(0)
        else:
            depth = None

        # ⑭ return
        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),
            "depth": depth,
            "accumulation": alpha.squeeze(0),
            "background": background,
        }

    # ── training ─────────────────────────────────────────────────────────

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """L1 + SSIM + scale regularization + camera / bilateral-grid losses."""
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

        # scale regularization (every 10 steps)
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

        # camera optimizer + bilateral TV loss
        if self.training:
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid and self.bil_grids is not None:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """PSNR, gaussian count, camera metrics."""
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
            cc_rgb = color_correct(outputs["rgb"], gt_rgb)
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb)
        metrics_dict["gaussian_count"] = self.num_points
        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    # ── evaluation ───────────────────────────────────────────────────────

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Render a full image for evaluation / viewer."""
        self.set_crop(obb_box)
        return self.get_outputs(camera.to(self.device))

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """PSNR / SSIM / LPIPS + side-by-side comparison image."""
        gt_rgb = composite_with_background(
            prepare_gt_image(
                batch["image"], self.config.num_downscales, self.step,
                self.config.resolution_schedule, False, self.device,
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

    def update_to_step(self, step: int) -> None:
        """Called when loading a checkpoint."""

    # ── param groups ─────────────────────────────────────────────────────

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        gps = self.get_gaussian_param_groups()
        if self.config.use_bilateral_grid and self.bil_grids is not None:
            gps["bilateral_grid"] = list(self.bil_grids.parameters())
        self.camera_optimizer.get_param_groups(param_groups=gps)
        return gps

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        return {name: [self.gauss_params[name]] for name in
                ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]}

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
        # handle legacy checkpoint format (pre-ParameterDict)
        if "means" in dict:
            for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = nn.Parameter(torch.zeros(newp, *param.shape[1:], device=self.device))
        super().load_state_dict(dict, **kwargs)

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
# internal helpers (module-level functions, not methods)
# ═══════════════════════════════════════════════════════════════════════════════


def _init_gaussian_params(
    config: SplatfactoModelConfig,
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


def _init_strategy(config: SplatfactoModelConfig, num_train_data: int):
    """Create densification strategy and initial state."""
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


def _select_gaussians(gauss_params: nn.ParameterDict, crop_ids: Optional[torch.Tensor]):
    """Return (means, quats, scales, opacities, features_dc, features_rest) for rendering."""
    if crop_ids is None:
        return (
            gauss_params["means"], gauss_params["quats"], gauss_params["scales"],
            gauss_params["opacities"], gauss_params["features_dc"], gauss_params["features_rest"],
        )
    return (
        gauss_params["means"][crop_ids], gauss_params["quats"][crop_ids],
        gauss_params["scales"][crop_ids], gauss_params["opacities"][crop_ids],
        gauss_params["features_dc"][crop_ids], gauss_params["features_rest"][crop_ids],
    )
