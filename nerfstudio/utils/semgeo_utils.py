from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from scripts.extract_dino_features import _extract_patch_features, _load_and_preprocess_image, _resolve_device


def _standardize(features: torch.Tensor) -> torch.Tensor:
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (features - mean) / std


def _golden_ratio_palette(num_colors: int) -> torch.Tensor:
    if num_colors <= 0:
        return torch.zeros((0, 3), dtype=torch.uint8)

    colors: List[List[int]] = []
    for idx in range(num_colors):
        hue = (idx * 0.6180339887498948) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.65, 1.0)
        colors.append([int(channel * 255) for channel in rgb])
    return torch.tensor(colors, dtype=torch.uint8)


def _fps(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    count = points.shape[0]
    if count == 0:
        return torch.zeros((0,), dtype=torch.long)
    if num_samples >= count:
        return torch.arange(count, dtype=torch.long)

    selected = torch.zeros((num_samples,), dtype=torch.long)
    distances = torch.full((count,), float("inf"), dtype=torch.float32)
    farthest = 0
    for step in range(num_samples):
        selected[step] = farthest
        centroid = points[farthest].unsqueeze(0)
        current_distances = torch.cdist(points, centroid).squeeze(1)
        distances = torch.minimum(distances, current_distances)
        farthest = int(torch.argmax(distances).item())
    return selected


def _estimate_normals(points: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
    if points.shape[0] == 0:
        return torch.zeros((0, 3), dtype=torch.float32)
    neighbors = points[knn_idx]
    centered = neighbors - neighbors.mean(dim=1, keepdim=True)
    cov = torch.matmul(centered.transpose(1, 2), centered) / max(1, neighbors.shape[1] - 1)
    _, eigvecs = torch.linalg.eigh(cov)
    normals = eigvecs[..., 0]
    return F.normalize(normals, dim=-1, eps=1e-6)


def _relabel_dense(labels: torch.Tensor) -> torch.Tensor:
    unique = torch.unique(labels[labels >= 0], sorted=True)
    if unique.numel() == 0:
        return torch.zeros_like(labels)
    dense = labels.clone()
    for new_idx, old_idx in enumerate(unique.tolist()):
        dense[labels == old_idx] = new_idx
    dense[dense < 0] = 0
    return dense


def _label_propagation(labels: torch.Tensor, knn_idx: torch.Tensor, min_cluster_size: int) -> torch.Tensor:
    if labels.numel() == 0:
        return labels

    refined = labels.clone()
    cluster_ids, counts = torch.unique(refined[refined >= 0], return_counts=True)
    small_clusters = cluster_ids[counts < min_cluster_size]
    for cluster_id in small_clusters.tolist():
        refined[refined == cluster_id] = -1

    unresolved = torch.nonzero(refined < 0, as_tuple=False).flatten()
    for _ in range(4):
        if unresolved.numel() == 0:
            break
        updates = 0
        for point_idx in unresolved.tolist():
            neighbor_labels = refined[knn_idx[point_idx]]
            neighbor_labels = neighbor_labels[neighbor_labels >= 0]
            if neighbor_labels.numel() == 0:
                continue
            values, value_counts = torch.unique(neighbor_labels, return_counts=True)
            refined[point_idx] = values[torch.argmax(value_counts)]
            updates += 1
        if updates == 0:
            break
        unresolved = torch.nonzero(refined < 0, as_tuple=False).flatten()

    if torch.any(refined < 0):
        fallback_label = int(refined[refined >= 0][0].item()) if torch.any(refined >= 0) else 0
        refined[refined < 0] = fallback_label
    return _relabel_dense(refined)


def _project_features_with_pca_state(feature_map_1024: torch.Tensor, pca_state: Dict[str, torch.Tensor], target_dim: int) -> torch.Tensor:
    components = pca_state["components"].cpu().numpy().astype(np.float32)
    mean = pca_state["mean"].cpu().numpy().astype(np.float32)
    effective_dim = int(components.shape[0])

    flat = feature_map_1024.permute(1, 2, 0).reshape(-1, feature_map_1024.shape[0]).cpu().numpy().astype(np.float32)
    reduced = np.matmul(flat - mean, components.T)
    if effective_dim < target_dim:
        reduced = np.pad(reduced, ((0, 0), (0, target_dim - effective_dim)), mode="constant")
    elif effective_dim > target_dim:
        reduced = reduced[:, :target_dim]

    height, width = int(feature_map_1024.shape[1]), int(feature_map_1024.shape[2])
    return torch.from_numpy(reduced).view(height, width, target_dim)


def _run_torch_kmeans(features: torch.Tensor, num_clusters: int, num_iters: int) -> Tuple[torch.Tensor, torch.Tensor]:
    count = features.shape[0]
    if count == 0:
        return torch.zeros((0,), dtype=torch.long), torch.zeros((0, features.shape[-1]), dtype=features.dtype)

    actual_clusters = max(1, min(num_clusters, count))
    stride = max(1, count // actual_clusters)
    initial_idx = torch.arange(0, stride * actual_clusters, stride, dtype=torch.long)[:actual_clusters]
    centers = features[initial_idx].clone()
    labels = torch.zeros((count,), dtype=torch.long)

    for _ in range(num_iters):
        distances = torch.cdist(features, centers)
        new_labels = torch.argmin(distances, dim=1)
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_idx in range(actual_clusters):
            mask = labels == cluster_idx
            if torch.any(mask):
                centers[cluster_idx] = features[mask].mean(dim=0)
            else:
                farthest_idx = int(torch.argmax(torch.min(distances, dim=1).values).item())
                centers[cluster_idx] = features[farthest_idx]

    return labels, centers


def _smooth_patch_labels(label_grid: torch.Tensor, num_passes: int = 2) -> torch.Tensor:
    if label_grid.numel() == 0:
        return label_grid

    refined = label_grid.clone()
    height, width = refined.shape
    for _ in range(num_passes):
        updated = refined.clone()
        for row in range(height):
            for col in range(width):
                neighbors: List[int] = []
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        neighbors.append(int(refined[nr, nc].item()))
                if not neighbors:
                    continue
                values, counts = torch.unique(torch.tensor(neighbors, dtype=torch.long), return_counts=True)
                majority_idx = int(torch.argmax(counts).item())
                if int(counts[majority_idx].item()) >= 3:
                    updated[row, col] = values[majority_idx]
        refined = updated
    return refined


def build_stage1_scene_clusters(
    means: torch.Tensor,
    dino_features: torch.Tensor,
    colors: torch.Tensor,
    opacities: torch.Tensor,
    *,
    knn_k: int,
    opacity_threshold: float,
    feature_similarity_threshold: float,
    normal_similarity_threshold: float,
    radius_scale: float,
    min_cluster_size: int,
    max_sample_points: int,
    max_metric_points: int,
    geometry_weight: float,
    dino_weight: float,
    max_viewer_points: int,
) -> Dict[str, Any]:
    means_cpu = means.detach().float().cpu()
    dino_cpu = dino_features.detach().float().cpu()
    colors_cpu = colors.detach().float().cpu()
    opacities_cpu = torch.sigmoid(opacities.detach().float().cpu().squeeze(-1))

    active_mask = opacities_cpu >= opacity_threshold
    active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
    if active_indices.numel() == 0:
        active_indices = torch.arange(means_cpu.shape[0], dtype=torch.long)

    active_priorities = opacities_cpu[active_indices]
    if active_indices.numel() > max_sample_points:
        sort_idx = torch.argsort(active_priorities, descending=True)[:max_sample_points]
        sampled_active_indices = active_indices[sort_idx]
    else:
        sampled_active_indices = active_indices

    sampled_points = means_cpu[sampled_active_indices]
    sampled_dino = dino_cpu[sampled_active_indices]
    sampled_colors = colors_cpu[sampled_active_indices]

    if sampled_points.shape[0] <= 1:
        full_labels = torch.zeros((means_cpu.shape[0],), dtype=torch.long)
    else:
        point_count = sampled_points.shape[0]

        # Adaptive neighborhood defaults.
        k_base = max(1, min(8, point_count - 1))
        k_max = max(1, min(max(knn_k, 32), point_count - 1))
        adaptive_alpha = 2.0
        feature_gate_threshold = 0.6
        pairwise_chunk_size = 1024

        def _chunked_knn(points: torch.Tensor, k: int, chunk_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
            total = points.shape[0]
            out_dist = torch.empty((total, k), dtype=torch.float32)
            out_idx = torch.empty((total, k), dtype=torch.long)
            for start in range(0, total, chunk_size):
                stop = min(start + chunk_size, total)
                chunk = points[start:stop]
                dist_block = torch.cdist(chunk, points)
                row_idx = torch.arange(start, stop, dtype=torch.long)
                dist_block[torch.arange(stop - start, dtype=torch.long), row_idx] = float("inf")
                k_dist, k_idx = torch.topk(dist_block, k=k, largest=False)
                out_dist[start:stop] = k_dist
                out_idx[start:stop] = k_idx
            return out_dist, out_idx

        base_knn_dist, base_knn_idx = _chunked_knn(sampled_points, k=k_base, chunk_size=pairwise_chunk_size)
        max_knn_dist, max_knn_idx = _chunked_knn(sampled_points, k=k_max, chunk_size=pairwise_chunk_size)

        local_scale = base_knn_dist[:, k_base - 1]
        adaptive_radius = adaptive_alpha * local_scale

        sampled_dino_norm = F.normalize(sampled_dino, dim=-1, eps=1e-6)
        adaptive_neighbor_sets: List[set[int]] = [set() for _ in range(point_count)]

        # Build radius-constrained neighborhoods from capped KNN candidates.
        for idx in range(point_count):
            candidate_idx = max_knn_idx[idx]
            candidate_dist = max_knn_dist[idx]
            feature_sim = torch.sum(sampled_dino_norm[idx].unsqueeze(0) * sampled_dino_norm[candidate_idx], dim=-1)
            keep_mask = (candidate_dist < adaptive_radius[idx]) & (feature_sim > feature_gate_threshold)
            kept = candidate_idx[keep_mask]
            adaptive_neighbor_sets[idx].update(int(v) for v in kept.tolist() if int(v) != idx)

        # Symmetrize graph: keep edge if either direction exists.
        for idx in range(point_count):
            for nbr in list(adaptive_neighbor_sets[idx]):
                adaptive_neighbor_sets[nbr].add(idx)

        # Convert variable-sized neighbors to dense tensor for downstream ops.
        adaptive_max_degree = max(1, min(k_max, max(len(nbrs) for nbrs in adaptive_neighbor_sets)))
        adaptive_neighbors = torch.empty((point_count, adaptive_max_degree), dtype=torch.long)
        for idx in range(point_count):
            nbrs = sorted(adaptive_neighbor_sets[idx])
            if len(nbrs) == 0:
                adaptive_neighbors[idx].fill_(idx)
                continue
            if len(nbrs) >= adaptive_max_degree:
                nbr_tensor = torch.tensor(nbrs, dtype=torch.long)
                nbr_dist = torch.norm(sampled_points[nbr_tensor] - sampled_points[idx], dim=-1)
                top_local = torch.topk(nbr_dist, k=adaptive_max_degree, largest=False).indices
                nbr_tensor = nbr_tensor[top_local]
                adaptive_neighbors[idx] = nbr_tensor
            else:
                adaptive_neighbors[idx, : len(nbrs)] = torch.tensor(nbrs, dtype=torch.long)
                adaptive_neighbors[idx, len(nbrs) :] = idx

        normals = _estimate_normals(sampled_points, base_knn_idx)
        geometry = torch.cat([sampled_points, normals, sampled_colors], dim=-1)
        hybrid = torch.cat(
            [
                _standardize(geometry) * geometry_weight,
                _standardize(sampled_dino) * dino_weight,
            ],
            dim=-1,
        )
        hybrid = F.normalize(hybrid, dim=-1, eps=1e-6)

        labels = torch.full((sampled_points.shape[0],), -1, dtype=torch.long)
        local_radii = adaptive_radius * radius_scale
        seed_order = torch.argsort(opacities_cpu[sampled_active_indices], descending=True)
        cluster_id = 0

        for seed_idx in seed_order.tolist():
            if labels[seed_idx] >= 0:
                continue
            labels[seed_idx] = cluster_id
            queue = [seed_idx]
            while queue:
                current_idx = queue.pop()
                neighbors = adaptive_neighbors[current_idx]
                unlabeled_mask = labels[neighbors] < 0
                if not torch.any(unlabeled_mask):
                    continue
                candidate_idx = neighbors[unlabeled_mask]
                candidate_dist = torch.norm(sampled_points[candidate_idx] - sampled_points[current_idx], dim=-1)
                feature_sim = torch.sum(hybrid[current_idx].unsqueeze(0) * hybrid[candidate_idx], dim=-1)
                normal_sim = torch.abs(torch.sum(normals[current_idx].unsqueeze(0) * normals[candidate_idx], dim=-1))
                accept_mask = (
                    (feature_sim >= feature_similarity_threshold)
                    & (normal_sim >= normal_similarity_threshold)
                    & (candidate_dist <= local_radii[current_idx])
                )
                accepted = candidate_idx[accept_mask]
                if accepted.numel() == 0:
                    continue
                labels[accepted] = cluster_id
                queue.extend(accepted.tolist())
            cluster_id += 1

        labels = _label_propagation(labels, adaptive_neighbors, min_cluster_size)
        full_labels = torch.full((means_cpu.shape[0],), -1, dtype=torch.long)

        if sampled_active_indices.numel() == active_indices.numel():
            full_labels[active_indices] = labels
        else:
            full_points = means_cpu[active_indices]
            chunk_size = 2048
            for start in range(0, full_points.shape[0], chunk_size):
                stop = min(start + chunk_size, full_points.shape[0])
                chunk = full_points[start:stop]
                nearest = torch.argmin(torch.cdist(chunk, sampled_points), dim=1)
                full_labels[active_indices[start:stop]] = labels[nearest]
            if torch.any(full_labels < 0):
                fallback = int(labels[0].item()) if labels.numel() > 0 else 0
                full_labels[full_labels < 0] = fallback

        full_labels = _relabel_dense(full_labels)

    num_clusters = int(full_labels.max().item()) + 1 if full_labels.numel() > 0 else 0
    palette = _golden_ratio_palette(num_clusters)
    point_colors = torch.zeros((means_cpu.shape[0], 3), dtype=torch.uint8)
    if num_clusters > 0:
        point_colors = palette[full_labels]

    active_cluster_labels = full_labels[active_indices]
    metric_spaces: List[Dict[str, Any]] = []
    for cluster_idx in range(num_clusters):
        cluster_member_mask = active_cluster_labels == cluster_idx
        cluster_member_indices = active_indices[cluster_member_mask]
        if cluster_member_indices.numel() == 0:
            continue
        cluster_points = means_cpu[cluster_member_indices]
        cluster_features = dino_cpu[cluster_member_indices]
        representative_idx = _fps(cluster_points, min(max_metric_points, cluster_points.shape[0]))
        representatives = cluster_points[representative_idx]
        representative_features = cluster_features[representative_idx]
        metric_spaces.append(
            {
                "cluster_id": cluster_idx,
                "gaussian_indices": cluster_member_indices.clone(),
                "representative_points": representatives,
                "distance_matrix": torch.cdist(representatives, representatives),
                "measure": torch.full((representatives.shape[0],), 1.0 / max(1, representatives.shape[0])),
                "representative_features": representative_features,
                "member_count": int(cluster_member_indices.numel()),
            }
        )

    viewer_indices = active_indices
    if viewer_indices.numel() > max_viewer_points:
        top_view_idx = torch.argsort(opacities_cpu[viewer_indices], descending=True)[:max_viewer_points]
        viewer_indices = viewer_indices[top_view_idx]

    return {
        "labels": full_labels,
        "point_colors": point_colors,
        "palette": palette,
        "active_indices": active_indices,
        "viewer_points": means_cpu[viewer_indices],
        "viewer_colors": point_colors[viewer_indices],
        "metric_spaces": metric_spaces,
        "metadata": {
            "opacity_threshold": opacity_threshold,
            "active_gaussian_count": int(active_indices.numel()),
            "sampled_gaussian_count": int(sampled_active_indices.numel()),
            "num_clusters": num_clusters,
            "adaptive_k_base": k_base if sampled_points.shape[0] > 1 else 0,
            "adaptive_k_max": k_max if sampled_points.shape[0] > 1 else 0,
            "adaptive_alpha": adaptive_alpha if sampled_points.shape[0] > 1 else 0.0,
            "adaptive_feature_gate_threshold": feature_gate_threshold if sampled_points.shape[0] > 1 else 0.0,
        },
    }


def build_stage2_style_primitives(
    style_image_path: Path,
    dino_feature_file: Path,
    *,
    feature_dim: int,
    num_clusters: int,
    spatial_weight: float,
    kmeans_iters: int,
    max_metric_points: int,
) -> Dict[str, Any]:
    payload = torch.load(dino_feature_file, map_location="cpu")
    if "pca_state" not in payload or not isinstance(payload["pca_state"], dict):
        raise KeyError(f"Missing pca_state in {dino_feature_file}")
    pca_state = payload["pca_state"]

    device = _resolve_device("cuda")
    image_tensor, original_hw, patch_hw = _load_and_preprocess_image(style_image_path, patch_size=14, device=device)
    dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    dino_model = dino_model.to(device).eval()
    feature_map_1024 = _extract_patch_features(dino_model, image_tensor, patch_hw)
    reduced_patch_features = _project_features_with_pca_state(feature_map_1024, pca_state, feature_dim).float()

    patch_height, patch_width = int(reduced_patch_features.shape[0]), int(reduced_patch_features.shape[1])
    flat_features = reduced_patch_features.view(-1, feature_dim)

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0.0, 1.0, patch_height),
        torch.linspace(0.0, 1.0, patch_width),
        indexing="ij",
    )
    flat_positions = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)

    fused_features = torch.cat(
        [
            _standardize(flat_features),
            _standardize(flat_positions) * spatial_weight,
        ],
        dim=-1,
    )
    labels, centers = _run_torch_kmeans(fused_features, num_clusters=num_clusters, num_iters=kmeans_iters)
    label_grid = _smooth_patch_labels(labels.view(patch_height, patch_width))
    labels = _relabel_dense(label_grid.view(-1))

    actual_clusters = int(labels.max().item()) + 1 if labels.numel() > 0 else 0
    palette = _golden_ratio_palette(actual_clusters)
    label_rgb = palette[labels].view(patch_height, patch_width, 3).numpy()
    label_image = Image.fromarray(label_rgb, mode="RGB").resize((original_hw[1], original_hw[0]), resample=Image.NEAREST)

    primitive_spaces: List[Dict[str, Any]] = []
    for cluster_idx in range(actual_clusters):
        cluster_mask = labels == cluster_idx
        cluster_indices = torch.nonzero(cluster_mask, as_tuple=False).flatten()
        if cluster_indices.numel() == 0:
            continue
        primitive_points = flat_positions[cluster_indices]
        primitive_features = flat_features[cluster_indices]
        representative_idx = _fps(primitive_points, min(max_metric_points, primitive_points.shape[0]))
        representatives = primitive_points[representative_idx]
        representative_features = primitive_features[representative_idx]
        primitive_spaces.append(
            {
                "primitive_id": cluster_idx,
                "patch_indices": cluster_indices.clone(),
                "representative_points": representatives,
                "distance_matrix": torch.cdist(representatives, representatives),
                "measure": torch.full((representatives.shape[0],), 1.0 / max(1, representatives.shape[0])),
                "representative_features": representative_features,
                "member_count": int(cluster_indices.numel()),
            }
        )

    return {
        "style_image_path": str(style_image_path),
        "original_hw": original_hw,
        "patch_hw": (patch_height, patch_width),
        "patch_features": flat_features,
        "patch_positions": flat_positions,
        "labels": labels.view(patch_height, patch_width),
        "palette": palette,
        "label_image": label_image,
        "primitive_spaces": primitive_spaces,
        "metadata": {
            "num_primitives": actual_clusters,
            "feature_dim": feature_dim,
            "spatial_weight": spatial_weight,
            "kmeans_iters": kmeans_iters,
            "center_count": int(centers.shape[0]),
        },
    }


def save_stage2_label_image(label_image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_image.save(output_path)
