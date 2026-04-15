#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from rich.progress import track
from sklearn.decomposition import PCA


MERGED_FEATURE_FILENAME = "dino_features.pt"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract DINOv2 features and save 16D PCA maps to a single .pt file.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing input images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory (or .pt file path) for merged features.",
    )
    parser.add_argument("--feature-dim", type=int, default=16, help="Target PCA feature dimension.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
        help="Image file extensions to include.",
    )
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda", help="Inference device.")
    return parser.parse_args()


@dataclass
class ImageFeatureRecord:
    image_path: Path
    feature_key: str
    original_hw: Tuple[int, int]
    patch_hw: Tuple[int, int]
    features_1024: torch.Tensor


def _list_images(input_dir: Path, extensions: List[str]) -> List[Path]:
    ext_set = {e.lower() for e in extensions}
    image_paths = [p for p in sorted(input_dir.rglob("*")) if p.is_file() and p.suffix.lower() in ext_set]
    return image_paths


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ceil_to_patch_multiple(value: int, patch_size: int) -> int:
    return int(math.ceil(value / patch_size) * patch_size)


def _load_and_preprocess_image(path: Path, patch_size: int, device: torch.device) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    h, w = image_tensor.shape[1], image_tensor.shape[2]

    resized_h = _ceil_to_patch_multiple(h, patch_size)
    resized_w = _ceil_to_patch_multiple(w, patch_size)
    if resized_h != h or resized_w != w:
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.unsqueeze(0).to(device)

    return image_tensor, (h, w), (resized_h // patch_size, resized_w // patch_size)


def _extract_patch_features(
    model: torch.nn.Module, image_tensor: torch.Tensor, patch_hw: Tuple[int, int]
) -> torch.Tensor:
    with torch.no_grad():
        outputs = model.forward_features(image_tensor)
    patch_tokens = outputs["x_norm_patchtokens"]
    ph, pw = patch_hw
    feature_map = patch_tokens.reshape(1, ph, pw, -1).permute(0, 3, 1, 2).squeeze(0).contiguous()
    return feature_map


def _fit_pca(records: List[ImageFeatureRecord], feature_dim: int) -> Tuple[PCA, int]:
    flattened = [r.features_1024.permute(1, 2, 0).reshape(-1, r.features_1024.shape[0]).numpy() for r in records]
    all_features = np.concatenate(flattened, axis=0)
    effective_dim = min(feature_dim, all_features.shape[0], all_features.shape[1])
    pca = PCA(n_components=effective_dim, svd_solver="randomized", random_state=0)
    pca.fit(all_features)
    return pca, effective_dim


def _resolve_output_file(output_dir: Path) -> Path:
    if output_dir.suffix.lower() == ".pt":
        return output_dir
    return output_dir / MERGED_FEATURE_FILENAME


def _load_existing_payload(output_file: Path, target_dim: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if not output_file.exists():
        return {}, {}

    payload = torch.load(output_file, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict in {output_file}, got {type(payload)}")

    if "features" not in payload:
        raise KeyError(f"Missing 'features' key in {output_file}")
    if not isinstance(payload["features"], dict):
        raise TypeError(f"Expected dict under 'features' in {output_file}, got {type(payload['features'])}")
    raw_features: Dict[Any, Any] = payload["features"]

    features: Dict[str, torch.Tensor] = {}
    for raw_key, raw_value in raw_features.items():
        key = str(raw_key).replace("\\", "/")
        if not isinstance(raw_value, torch.Tensor):
            raise TypeError(f"Feature entry '{key}' is not a tensor: {type(raw_value)}")
        value = raw_value.float()
        if value.shape[-1] != target_dim:
            raise ValueError(
                f"Feature dim mismatch for key '{key}' in {output_file}: "
                f"expected last dim {target_dim}, got {tuple(value.shape)}"
            )
        features[key] = value

    if "pca_state" not in payload:
        raise KeyError(f"Missing 'pca_state' key in {output_file}")
    pca_state = payload["pca_state"]
    if not isinstance(pca_state, dict):
        raise TypeError(f"Expected dict pca_state in {output_file}, got {type(pca_state)}")
    if "components" not in pca_state or "mean" not in pca_state:
        raise KeyError(f"Invalid pca_state in {output_file}: expected 'components' and 'mean'")

    return features, cast(Dict[str, torch.Tensor], pca_state)


def _serialize_pca_state(pca: PCA) -> Dict[str, torch.Tensor]:
    return {
        "components": torch.from_numpy(np.asarray(pca.components_, dtype=np.float32)),
        "mean": torch.from_numpy(np.asarray(pca.mean_, dtype=np.float32)),
    }


def _project_with_pca(record: ImageFeatureRecord, pca: PCA, effective_dim: int, target_dim: int) -> torch.Tensor:
    flat = record.features_1024.permute(1, 2, 0).reshape(-1, record.features_1024.shape[0]).numpy()
    reduced = pca.transform(flat)
    if effective_dim < target_dim:
        reduced = np.pad(reduced, ((0, 0), (0, target_dim - effective_dim)), mode="constant")

    ph, pw = record.patch_hw
    reduced_tensor = torch.from_numpy(reduced.astype(np.float32)).view(ph, pw, target_dim).permute(2, 0, 1).unsqueeze(0)
    upsampled = F.interpolate(
        reduced_tensor,
        size=record.original_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return upsampled.permute(1, 2, 0).contiguous()


def _project_with_pca_state(record: ImageFeatureRecord, pca_state: Dict[str, torch.Tensor], target_dim: int) -> torch.Tensor:
    components = pca_state["components"].cpu().numpy().astype(np.float32)
    mean = pca_state["mean"].cpu().numpy().astype(np.float32)
    effective_dim = int(components.shape[0])

    flat = record.features_1024.permute(1, 2, 0).reshape(-1, record.features_1024.shape[0]).numpy()
    reduced = np.matmul(flat - mean, components.T)
    if effective_dim < target_dim:
        reduced = np.pad(reduced, ((0, 0), (0, target_dim - effective_dim)), mode="constant")
    elif effective_dim > target_dim:
        reduced = reduced[:, :target_dim]

    ph, pw = record.patch_hw
    reduced_tensor = torch.from_numpy(reduced.astype(np.float32)).view(ph, pw, target_dim).permute(2, 0, 1).unsqueeze(0)
    upsampled = F.interpolate(
        reduced_tensor,
        size=record.original_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return upsampled.permute(1, 2, 0).contiguous()


def _save_payload(
    output_file: Path,
    features: Dict[str, torch.Tensor],
    feature_dim: int,
    input_dir: Path,
    pca_state: Dict[str, torch.Tensor],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "version": 2,
        "feature_dim": feature_dim,
        "input_dir": str(input_dir),
        "features": features,
        "pca_state": pca_state,
    }
    torch.save(payload, output_file)


def extract_dino_features_for_images(
    image_paths: List[Path],
    input_dir: Path,
    output_dir: Path,
    feature_dim: int = 16,
    device: str = "cuda",
    skip_existing: bool = True,
) -> Tuple[int, int]:
    """Extract DINOv2 features for provided image paths and save PCA-reduced feature maps.

    Returns:
        Tuple[int, int]: (num_saved, num_skipped)
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if len(image_paths) == 0:
        return 0, 0

    requested = sorted({p.resolve() for p in image_paths})
    output_file = _resolve_output_file(output_dir.resolve())
    features_payload, pca_state = _load_existing_payload(output_file, feature_dim)

    pending_keys: List[str] = []
    requested_by_key: Dict[str, Path] = {}
    skipped = 0

    for image_path in requested:
        try:
            relative = image_path.relative_to(input_dir).with_suffix(".pt")
        except ValueError as exc:
            raise ValueError(
                f"Image path {image_path} is not under input directory {input_dir}. "
                "Set input_dir to the common image root."
            ) from exc

        key = relative.as_posix()
        requested_by_key[key] = image_path

        if skip_existing and key in features_payload:
            skipped += 1
            continue

        pending_keys.append(key)

    if len(pending_keys) == 0:
        return 0, skipped

    resolved_device = _resolve_device(device)
    print(f"[INFO] Using device: {resolved_device}")
    print("[INFO] Loading DINOv2 model (dinov2_vitl14)...")
    model = cast(torch.nn.Module, torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14"))
    model = model.to(resolved_device)
    model.eval()

    patch_size = 14
    records: List[ImageFeatureRecord] = []
    for feature_key in track(pending_keys, description="Extracting DINO patch features"):
        image_path = requested_by_key[feature_key]
        image_tensor, original_hw, patch_hw = _load_and_preprocess_image(image_path, patch_size, resolved_device)
        feature_map = _extract_patch_features(model, image_tensor, patch_hw).cpu()
        records.append(
            ImageFeatureRecord(
                image_path=image_path,
                feature_key=feature_key,
                original_hw=original_hw,
                patch_hw=patch_hw,
                features_1024=feature_map,
            )
        )

    if len(pca_state) == 0:
        print("[INFO] Fitting PCA...")
        pca, effective_dim = _fit_pca(records, feature_dim)
        if effective_dim < feature_dim:
            print(
                f"[WARN] Effective PCA dimension is {effective_dim} < requested {feature_dim}; "
                "remaining channels are padded with zeros."
            )

        for record in track(records, description="Projecting and upsampling features"):
            features_payload[record.feature_key] = _project_with_pca(record, pca, effective_dim, feature_dim)

        pca_state = _serialize_pca_state(pca)
    else:
        for record in track(records, description="Projecting and upsampling features"):
            features_payload[record.feature_key] = _project_with_pca_state(record, pca_state, feature_dim)

    _save_payload(
        output_file=output_file,
        features=features_payload,
        feature_dim=feature_dim,
        input_dir=input_dir,
        pca_state=pca_state,
    )

    return len(records), skipped


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    image_paths = _list_images(input_dir, args.extensions)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {input_dir}")

    saved, skipped = extract_dino_features_for_images(
        image_paths=image_paths,
        input_dir=input_dir,
        output_dir=output_dir,
        feature_dim=args.feature_dim,
        device=args.device,
        skip_existing=True,
    )
    if saved == 0:
        print("[INFO] All features already exist. Nothing to do.")
    else:
        output_file = _resolve_output_file(output_dir)
        print(f"[INFO] DINO feature extraction complete. saved={saved}, skipped={skipped}, file={output_file}")


if __name__ == "__main__":
    main()
