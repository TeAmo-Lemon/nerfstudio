"""SemGeo Splatfacto Model — Pipeline 2: structural decomposition + clustering.

Extends DinoSplatfactoModel. After loading a dino-splatfacto checkpoint,
run ``run_clustering()`` to execute Pipeline 2 Stage 1:
  1.1 Build enhanced hybrid feature space (geometry + DINO semantics)
  1.2 Coarse pre-clustering (skipped for < 50k points)
  1.3 Adaptive neighborhood graph construction
  1.4 BFS region growing
  1.5 Post-processing (label propagation + connected components)

Usage (via train_semgeo.py):
    python train_semgeo.py -s /path/to/scene --load-checkpoint step-*.ckpt
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.utils import normalized_quat_to_rotmat

from cameras.cameras import Cameras
from configs.base_config import InstantiateConfig
from gaussian_renderer.renderer import get_viewmat, render_gaussians
from scene.dino_gaussian_model import DinoSplatfactoModel, DinoSplatfactoModelConfig
from utils.sh_utils import SH2RGB
from utils.system_utils import CONSOLE


# ═══════════════════════════════════════════════════════════════════════════════
# config
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SemGeoSplatfactoModelConfig(DinoSplatfactoModelConfig):
    """Configuration for SemGeo — extends DinoSplatfacto with clustering params."""

    _target: Type = field(default_factory=lambda: SemGeoSplatfactoModel)

    # ── Clustering hyper-params (Pipeline 2 Stage 1) ──────────────────────
    geo_weight: float = 1.0
    """Weight for geometric features in hybrid space."""
    dino_weight: float = 0.8
    """Weight for semantic features in hybrid space."""
    k_base: int = 8
    """Number of neighbors for local scale estimation."""
    alpha_radius: float = 2.0
    """Adaptive radius multiplier."""
    k_max: int = 32
    """Max candidate neighbors for KNN search."""
    semantic_gate: float = 0.6
    """Cosine similarity threshold for semantic gating."""
    tau_feat: float = 0.75
    """Feature similarity threshold for region growing."""
    tau_normal: float = 0.7
    """Normal consistency threshold for region growing."""
    gamma_spatial: float = 1.2
    """Spatial constraint multiplier for region growing."""
    min_cluster_size: int = 100
    """Clusters smaller than this get merged via label propagation."""
    cdist_chunk_size: int = 4096
    """Chunk size for chunked cdist (to limit memory)."""


# ═══════════════════════════════════════════════════════════════════════════════
# model
# ═══════════════════════════════════════════════════════════════════════════════


class SemGeoSplatfactoModel(DinoSplatfactoModel):
    """3DGS + DINO features + structural clustering (Pipeline 2).

    Inherits DinoSplatfactoModel for rendering and dino feature handling.
    Adds ``run_clustering()`` that implements Pipeline 2 Stage 1.

    After clustering, access:
        - ``self.cluster_labels``: (N,) LongTensor of cluster assignments
        - ``self.cluster_stats``: dict with per-cluster metadata
    """

    config: SemGeoSplatfactoModelConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_labels: Optional[torch.Tensor] = None
        self.cluster_stats: Dict = {}
        self._cluster_colormap: Optional[torch.Tensor] = None  # (num_clusters, 3) on device

    # ── memory management ────────────────────────────────────────────────────

    def _strip_training_components(self) -> None:
        """Free all training-only components to reduce GPU memory.

        After loading a checkpoint for inference/clustering, we keep only:
          - gauss_params (means, scales, quats, features_dc, features_rest,
            opacities, dino_features)
          - background_color, last_size (for viewer rendering)
        Everything else is deleted.
        """
        freed = []

        # Camera optimizer
        if hasattr(self, "camera_optimizer") and self.camera_optimizer is not None:
            del self.camera_optimizer
            self.camera_optimizer = None  # type: ignore
            freed.append("camera_optimizer")

        # Densification strategy + state
        if hasattr(self, "strategy") and self.strategy is not None:
            del self.strategy
            self.strategy = None  # type: ignore
            freed.append("strategy")
        if hasattr(self, "strategy_state") and self.strategy_state is not None:
            del self.strategy_state
            self.strategy_state = None  # type: ignore
            freed.append("strategy_state")

        # Torchmetrics (SSIM, PSNR, LPIPS — use GPU buffers)
        for attr in ("psnr", "ssim", "lpips"):
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
                freed.append(attr)

        # Bilateral grid
        if hasattr(self, "bil_grids") and self.bil_grids is not None:
            del self.bil_grids
            self.bil_grids = None  # type: ignore
            freed.append("bil_grids")

        # Style primitives buffer (not needed for clustering)
        if hasattr(self, "style_primitives"):
            del self.style_primitives
            freed.append("style_primitives")

        if freed:
            CONSOLE.log(f"[dim]Freed training components: {', '.join(freed)}[/dim]")

    # ── cluster visualization helpers ────────────────────────────────────────

    def _build_cluster_colormap(self) -> torch.Tensor:
        """Build (num_clusters, 3) RGB colormap for current cluster labels."""
        if self.cluster_labels is None:
            return torch.empty(0, 3, device=self.device)

        unique = torch.unique(self.cluster_labels[self.cluster_labels >= 0])
        K = unique.numel()
        if K == 0:
            return torch.empty(0, 3, device=self.device)

        import colorsys
        colors = []
        for i in range(K):
            hue = i / max(K, 1)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
            colors.append([r, g, b])
        return torch.tensor(colors, dtype=torch.float32, device=self.device)

    def _cluster_rgb_for_gaussians(self, crop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return (N', 3) RGB colors — each gaussian colored by its cluster label.

        Labels are 0..K-1 after renumbering, so they directly index into the colormap.
        """
        if self.cluster_labels is None or self._cluster_colormap is None:
            N = self.means.shape[0] if crop_ids is None else int(crop_ids.sum().item())
            return torch.ones(N, 3, device=self.device) * 0.5  # gray fallback

        labels = self.cluster_labels if crop_ids is None else self.cluster_labels[crop_ids]
        labels_clamped = labels.clamp(min=0)  # -1 → 0
        # Direct index into colormap (labels are 0..K-1)
        rgb = self._cluster_colormap[labels_clamped % self._cluster_colormap.shape[0]]
        # Dark gray for unclustered (label == -1)
        unclustered = labels < 0
        if unclustered.any():
            rgb[unclustered] = torch.tensor([0.2, 0.2, 0.2], device=self.device)
        return rgb

    # ── get_outputs override (adds cluster_rgb rendering) ────────────────────

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Render RGB + depth + DINO + cluster_rgb.

        ``cluster_rgb`` is ALWAYS present in outputs (so it appears in Viser's
        Output Type dropdown from the first render). Before clustering, it
        falls back to the RGB output.
        """
        outputs = super().get_outputs(camera)

        if not isinstance(camera, Cameras):
            outputs["cluster_rgb"] = outputs.get("rgb",
                torch.zeros(1, 1, 3, device=self.device))
            return outputs

        H, W = self.last_size

        # ── fast path: no cluster labels yet → show RGB as cluster_rgb ──
        if self.cluster_labels is None or self._cluster_colormap is None:
            outputs["cluster_rgb"] = outputs.get("rgb",
                torch.zeros(H, W, 3, device=self.device))
            return outputs

        if self.training:
            optimized_c2w = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_c2w = camera.camera_to_worlds

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                outputs["cluster_rgb"] = outputs.get("background",
                    torch.zeros(H, W, 3, device=self.device))
                return outputs
        else:
            crop_ids = None

        viewmat = get_viewmat(optimized_c2w)
        K = camera.get_intrinsics_matrices().to(self.device)

        means = self.means if crop_ids is None else self.means[crop_ids]
        quats = self.quats if crop_ids is None else self.quats[crop_ids]
        scales = self.scales if crop_ids is None else self.scales[crop_ids]
        opacities = self.opacities if crop_ids is None else self.opacities[crop_ids]
        cluster_colors = self._cluster_rgb_for_gaussians(crop_ids)

        cluster_render, _, _ = render_gaussians(
            means=means, quats=quats, scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities), colors=cluster_colors,
            viewmat=viewmat, K=K, width=W, height=H,
            sh_degree=None,
            render_mode="RGB",
            absgrad=False,
            rasterize_mode=self.config.rasterize_mode,
        )

        alpha = outputs.get("accumulation", torch.ones(H, W, 1, device=self.device))
        if alpha.dim() == 2:
            alpha = alpha.unsqueeze(-1)
        bg = outputs.get("background", torch.zeros(H, W, 3, device=self.device))
        if bg.dim() == 2:
            bg = bg.unsqueeze(0).expand(H, W, 3)
        cluster_rgb = cluster_render.squeeze(0)[..., :3] + (1 - alpha) * bg
        cluster_rgb = torch.clamp(cluster_rgb, 0.0, 1.0)
        outputs["cluster_rgb"] = cluster_rgb

        return outputs

    # ── clustering API ───────────────────────────────────────────────────────

    @torch.no_grad()
    def run_clustering(self) -> torch.Tensor:
        """Execute Pipeline 2 Stage 1: structural decomposition.

        Returns:
            cluster_labels: (N,) LongTensor — cluster index for every gaussian
                (-1 = background / filtered out).
        """
        CONSOLE.log("[bold cyan]═════ Pipeline 2 Stage 1: Structural Decomposition ═════")
        cfg = self.config
        N = self.means.shape[0]
        CONSOLE.log(f"Total gaussians: {N:,}")

        # ── 1.1 Build enhanced hybrid feature space ──────────────────────
        V, valid_mask, all_indices = self._build_enhanced_features()
        M = int(valid_mask.sum().item())
        CONSOLE.log(f"1.1 Enhanced features: {M:,} valid (opacity >= 0.05)")

        if M == 0:
            CONSOLE.log("[bold yellow]No valid points (all opacity < 0.05). Returning all -1.")
            self.cluster_labels = torch.full((N,), -1, dtype=torch.long, device=self.means.device)
            self.cluster_stats = {"num_clusters": 0, "valid_points": 0, "total_points": N, "num_edges": 0}
            return self.cluster_labels

        # ── 1.3 Adaptive neighborhood graph ──────────────────────────────
        graph_edges = self._build_adaptive_graph(V, all_indices, M)
        num_edges = len(graph_edges)
        CONSOLE.log(f"1.3 Adaptive graph: {num_edges:,} edges ({num_edges / M:.1f} avg degree)")

        # ── 1.4 BFS region growing ───────────────────────────────────────
        raw_labels = self._bfs_region_growing(graph_edges, V, M)
        num_clusters = int(raw_labels.max().item()) + 1
        CONSOLE.log(f"1.4 BFS growing: {num_clusters} initial clusters")

        # ── 1.5 Post-processing ──────────────────────────────────────────
        refined_labels = self._postprocess(raw_labels, graph_edges, V, M)
        num_clusters = int(refined_labels.max().item()) + 1
        CONSOLE.log(f"1.5 Post-process: {num_clusters} final clusters")

        # ── Map back to full gaussian index space ────────────────────────
        full_labels = torch.full((N,), -1, dtype=torch.long, device=self.means.device)
        full_labels[valid_mask] = refined_labels
        self.cluster_labels = full_labels

        # ── Stats ────────────────────────────────────────────────────────
        labels_np = refined_labels.cpu()
        sizes = torch.bincount(labels_np)
        self.cluster_stats = {
            "num_clusters": num_clusters,
            "cluster_sizes": sizes,
            "valid_points": M,
            "total_points": N,
            "num_edges": num_edges,
        }
        CONSOLE.log(
            f"[bold green]Clustering complete: {num_clusters} clusters, "
            f"sizes min={sizes.min().item():,} max={sizes.max().item():,} "
            f"mean={sizes.float().mean().item():,.0f}"
        )

        # Build colormap for cluster_rgb rendering
        self._cluster_colormap = self._build_cluster_colormap()

        # Free clustering intermediates
        self._valid_normals = None  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return full_labels

    # ═══════════════════════════════════════════════════════════════════════
    # 1.1 Build enhanced hybrid feature space
    # ═══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _build_enhanced_features(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build V_enhanced = Whiten([w_g·Norm(g_i), w_d·Norm(f_dino,i)]).

        Returns:
            V: (M, D) enhanced feature vectors for valid points
            valid_mask: (N,) bool mask of valid (opaque enough) gaussians
            all_indices: (M,) indices mapping V rows → original gaussian indices
        """
        cfg = self.config

        # ── geometric component ──────────────────────────────────────────
        means = self.means  # (N, 3)

        # normal = direction of minimum scale (flattest axis of ellipsoid)
        quats = self.quats  # (N, 4) in (x,y,z,w) convention
        R = normalized_quat_to_rotmat(quats)  # (N, 3, 3)
        scales_exp = torch.exp(self.scales)  # (N, 3)
        min_scale_idx = torch.argmin(scales_exp, dim=1)  # (N,)
        normals = R[torch.arange(R.shape[0], device=R.device), :, min_scale_idx]  # (N, 3)

        # SH0 (base color) — features_dc: (N, 1, 3) for sh_degree>0, (N, 3) for degree 0
        if cfg.sh_degree > 0:
            sh0_rgb = SH2RGB(self.features_dc)  # (N, 3)
        else:
            sh0_rgb = torch.sigmoid(self.features_dc[:, :3])

        g_i = torch.cat([means, normals, sh0_rgb], dim=-1)  # (N, 9)

        # ── semantic component ───────────────────────────────────────────
        f_dino = self.dino_features  # (N, 16)

        # ── standardize ──────────────────────────────────────────────────
        g_mean = g_i.mean(dim=0, keepdim=True)
        g_std = g_i.std(dim=0, keepdim=True).clamp_min(1e-6)
        g_norm = (g_i - g_mean) / g_std

        f_mean = f_dino.mean(dim=0, keepdim=True)
        f_std = f_dino.std(dim=0, keepdim=True).clamp_min(1e-6)
        f_norm = (f_dino - f_mean) / f_std

        # ── weighted concatenation ───────────────────────────────────────
        V_raw = torch.cat([cfg.geo_weight * g_norm, cfg.dino_weight * f_norm], dim=-1)  # (N, 25)

        # ── whitening ────────────────────────────────────────────────────
        V_mean = V_raw.mean(dim=0, keepdim=True)
        V_centered = V_raw - V_mean
        cov = V_centered.T @ V_centered / (V_centered.shape[0] - 1)
        eps = 1e-6
        L, Q = torch.linalg.eigh(cov)
        # Clamp eigenvalues and invert
        L_clamped = L.clamp_min(eps)
        inv_sqrt_L = 1.0 / torch.sqrt(L_clamped)
        V_white = V_centered @ Q @ torch.diag(inv_sqrt_L)

        # ── filter by opacity ────────────────────────────────────────────
        opacity = torch.sigmoid(self.opacities).squeeze(-1)  # (N,)
        valid_mask = opacity >= 0.05
        all_indices = torch.where(valid_mask)[0]

        V_valid = V_white[valid_mask]
        valid_normals = normals[valid_mask]  # (M, 3) — cached for BFS normal check
        self._valid_normals = valid_normals

        return V_valid, valid_mask, all_indices

    # ═══════════════════════════════════════════════════════════════════════
    # 1.3 Adaptive neighborhood graph
    # ═══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _build_adaptive_graph(
        self, V: torch.Tensor, all_indices: torch.Tensor, M: int,
    ) -> List[Tuple[int, int]]:
        """Build adaptive semantic-gated neighborhood graph.

        Uses chunked cdist for memory efficiency.

        Returns:
            List of (src, dst) edge tuples (0-indexed within V).
        """
        cfg = self.config
        valid_means = self.means[all_indices]  # (M, 3)
        valid_dino = self.dino_features[all_indices]  # (M, 16)

        # ── KNN search (chunked) ─────────────────────────────────────────
        k_max = cfg.k_max
        k_base = cfg.k_base
        chunk_size = cfg.cdist_chunk_size

        # Accumulate top-k per row
        all_knn_dist = torch.full((M, k_max), float("inf"), device=valid_means.device)
        all_knn_idx = torch.full((M, k_max), -1, dtype=torch.long, device=valid_means.device)

        num_chunks = (M + chunk_size - 1) // chunk_size
        for chunk_i, start in enumerate(range(0, M, chunk_size)):
            end = min(start + chunk_size, M)
            CONSOLE.log(f"  KNN chunk {chunk_i+1}/{num_chunks} (points {start:,}–{end:,})")
            chunk_dists = torch.cdist(valid_means[start:end], valid_means)  # (chunk, M)
            for j in range(end - start):
                chunk_dists[j, start + j] = float("inf")  # self-distance
            chunk_topk = torch.topk(chunk_dists, k_max, dim=1, largest=False)

            # Merge with accumulated top-k
            existing_dist = all_knn_dist[start:end]
            existing_idx = all_knn_idx[start:end]
            merged_dist = torch.cat([existing_dist, chunk_topk.values], dim=1)
            merged_idx = torch.cat([existing_idx, chunk_topk.indices], dim=1)
            sorted_dist, sort_order = torch.sort(merged_dist, dim=1)
            sorted_idx = merged_idx.gather(1, sort_order)
            all_knn_dist[start:end] = sorted_dist[:, :k_max]
            all_knn_idx[start:end] = sorted_idx[:, :k_max]

        # ── local scale estimation ───────────────────────────────────────
        s_i = all_knn_dist[:, k_base - 1]  # (M,) — distance to k_base-th neighbor
        r_i = cfg.alpha_radius * s_i  # (M,)

        # ── filter edges: radius + semantic gate (VECTORIZED on GPU) ────
        CONSOLE.log("  Filtering edges (radius + semantic gate)...")
        dino_norm = F.normalize(valid_dino, p=2, dim=-1)  # (M, 16)

        # Gather neighbor DINO features: (M, k_max, 16)
        nbr_dino = dino_norm[all_knn_idx]  # (M, k_max, 16)
        # Cosine similarity between each point and its k_max neighbors
        cos_sim = (dino_norm.unsqueeze(1) * nbr_dino).sum(dim=-1)  # (M, k_max)

        # Combined mask: radius + semantic gate
        radius_mask = all_knn_dist < r_i.unsqueeze(-1)  # (M, k_max)
        sem_mask = cos_sim > cfg.semantic_gate
        combined = radius_mask & sem_mask

        # Extract edges from mask (GPU → CPU)
        edges_i, edges_j = torch.where(combined)
        edges_i = edges_i.cpu().tolist()
        edges_j = all_knn_idx[combined].cpu().tolist()
        edges = list(zip(edges_i, edges_j))
        CONSOLE.log(f"  Radius+semantic filter: {len(edges):,} directed edges")

        # ── graph symmetrization ─────────────────────────────────────────
        edge_set = set(edges)
        for i, j in edges:
            if (j, i) not in edge_set:
                edge_set.add((j, i))

        return list(edge_set)

    # ═══════════════════════════════════════════════════════════════════════
    # 1.4 BFS region growing
    # ═══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _bfs_region_growing(
        self, edges: List[Tuple[int, int]], V: torch.Tensor, M: int,
    ) -> torch.Tensor:
        """BFS region growing on the adaptive graph.

        Growing conditions (all must hold):
          - cos(h_i, h_j) > tau_feat     (feature similarity in enhanced space)
          - |n_i · n_j| > tau_normal     (normal consistency)
          - d(i,j) <= r_i * gamma        (spatial constraint, enforced in graph)

        Returns:
            labels: (M,) LongTensor of cluster assignments
        """
        cfg = self.config
        device = V.device

        # Build adjacency list
        adj: Dict[int, List[int]] = {i: [] for i in range(M)}
        for i, j in edges:
            adj[i].append(j)

        # Precompute L2-normalized V and normals
        V_norm = F.normalize(V, p=2, dim=-1)
        N_norm = F.normalize(self._valid_normals, p=2, dim=-1)  # (M, 3)

        labels = torch.full((M,), -1, dtype=torch.long, device=device)
        current_label = 0

        order = torch.randperm(M, device=device)

        for seed_idx, seed in enumerate(order):
            seed_i = int(seed.item())
            if labels[seed_i] != -1:
                continue

            labels[seed_i] = current_label
            queue = deque([seed_i])
            region_size = 0

            while queue:
                cur = queue.popleft()
                cur_feat = V_norm[cur]
                region_size += 1

                for nbr in adj.get(cur, []):
                    if labels[nbr] != -1:
                        continue

                    cos_feat = float((cur_feat @ V_norm[nbr]).item())
                    if cos_feat < cfg.tau_feat:
                        continue

                    cos_normal = float((N_norm[cur] @ N_norm[nbr]).abs().item())
                    if cos_normal < cfg.tau_normal:
                        continue

                    labels[nbr] = current_label
                    queue.append(nbr)

            current_label += 1

            # Log progress every 20 clusters
            if current_label % 20 == 0:
                visited = int((labels >= 0).sum().item())
                CONSOLE.log(f"  BFS: {current_label} clusters, {visited:,}/{M:,} points assigned")

        visited = int((labels >= 0).sum().item())
        CONSOLE.log(f"  BFS done: {current_label} clusters, {visited:,}/{M:,} points assigned")
        return labels

    # ═══════════════════════════════════════════════════════════════════════
    # 1.5 Post-processing
    # ═══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _postprocess(
        self, labels: torch.Tensor, edges: List[Tuple[int, int]], V: torch.Tensor, M: int,
    ) -> torch.Tensor:
        """Label propagation for small clusters + connected components split."""
        cfg = self.config

        # ── Build adjacency for KNN lookup ───────────────────────────────
        adj: Dict[int, List[int]] = {i: [] for i in range(M)}
        for i, j in edges:
            adj[i].append(j)

        # ── Label propagation for small clusters ─────────────────────────
        unique_labels = torch.unique(labels)
        refined = labels.clone()

        for label in unique_labels:
            mask = refined == label
            if mask.sum() < cfg.min_cluster_size:
                # For each point in this small cluster, reassign to majority neighbor label
                members = torch.where(mask)[0]
                for idx in members:
                    ii = int(idx.item())
                    nbr_labels = []
                    for nbr in adj.get(ii, [])[:10]:  # K=10 neighbors
                        if refined[nbr] != -1 and refined[nbr] != label:
                            nbr_labels.append(int(refined[nbr].item()))
                    if nbr_labels:
                        # Majority vote
                        from collections import Counter
                        most_common = Counter(nbr_labels).most_common(1)[0][0]
                        refined[ii] = most_common

        # ── Re-number labels sequentially ────────────────────────────────
        valid_mask = refined >= 0
        old_labels = refined[valid_mask]
        unique_old = torch.unique(old_labels)
        label_map = {int(old.item()): i for i, old in enumerate(unique_old)}
        new_labels = torch.full((M,), -1, dtype=torch.long, device=labels.device)
        for old, new in label_map.items():
            new_labels[refined == old] = new

        return new_labels
