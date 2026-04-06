from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def resolve_colmap_project_path(path: Path) -> Tuple[Path, Path, Path]:
    path = path.expanduser().resolve()
    sparse_candidates = (Path("sparse/0"), Path("sparse"), Path("colmap/sparse/0"), Path("colmap/sparse"))
    roots = [path, *list(path.parents)[:4]]
    for root in roots:
        images_dir = root / "images"
        for sparse_rel in sparse_candidates:
            sparse_dir = root / sparse_rel
            if images_dir.is_dir() and sparse_dir.is_dir():
                if any((sparse_dir / name).exists() for name in ("cameras.bin", "cameras.txt")) and any(
                    (sparse_dir / name).exists() for name in ("images.bin", "images.txt")
                ):
                    return root, images_dir, sparse_dir
    raise FileNotFoundError(
        f"Could not resolve COLMAP project from {path}. Expected a root containing images/ and sparse[/0]/."
    )


def parse_colmap_camera_params(camera) -> Dict[str, Any]:
    out: Dict[str, Any] = {"w": camera.width, "h": camera.height}
    params = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        out.update(fl_x=float(params[0]), fl_y=float(params[0]), cx=float(params[1]), cy=float(params[2]), k1=0.0, k2=0.0, k3=0.0, k4=0.0, p1=0.0, p2=0.0)
        out["camera_model"] = "SIMPLE_PINHOLE"
    elif camera.model == "PINHOLE":
        out.update(fl_x=float(params[0]), fl_y=float(params[1]), cx=float(params[2]), cy=float(params[3]), k1=0.0, k2=0.0, k3=0.0, k4=0.0, p1=0.0, p2=0.0)
        out["camera_model"] = "PINHOLE"
    elif camera.model == "SIMPLE_RADIAL":
        out.update(fl_x=float(params[0]), fl_y=float(params[0]), cx=float(params[1]), cy=float(params[2]), k1=float(params[3]), k2=0.0, k3=0.0, k4=0.0, p1=0.0, p2=0.0)
        out["camera_model"] = "OPENCV"
    elif camera.model == "RADIAL":
        out.update(fl_x=float(params[0]), fl_y=float(params[0]), cx=float(params[1]), cy=float(params[2]), k1=float(params[3]), k2=float(params[4]), k3=0.0, k4=0.0, p1=0.0, p2=0.0)
        out["camera_model"] = "OPENCV"
    elif camera.model == "OPENCV":
        out.update(fl_x=float(params[0]), fl_y=float(params[1]), cx=float(params[2]), cy=float(params[3]), k1=float(params[4]), k2=float(params[5]), p1=float(params[6]), p2=float(params[7]), k3=0.0, k4=0.0)
        out["camera_model"] = "OPENCV"
    elif camera.model == "OPENCV_FISHEYE":
        out.update(fl_x=float(params[0]), fl_y=float(params[1]), cx=float(params[2]), cy=float(params[3]), k1=float(params[4]), k2=float(params[5]), k3=float(params[6]), k4=float(params[7]), p1=0.0, p2=0.0)
        out["camera_model"] = "OPENCV_FISHEYE"
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {camera.model}")
    return out


def get_train_eval_split_fraction(image_filenames: List[Path], train_split_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    num_images = len(image_filenames)
    num_train_images = math.ceil(num_images * train_split_fraction)
    all_indices = np.arange(num_images)
    train_indices = np.linspace(0, num_images - 1, num_train_images, dtype=int)
    eval_indices = np.setdiff1d(all_indices, train_indices)
    return train_indices, eval_indices


def get_train_eval_split_interval(image_filenames: List[Path], eval_interval: int) -> Tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(len(image_filenames))
    train_indices = all_indices[all_indices % eval_interval != 0]
    eval_indices = all_indices[all_indices % eval_interval == 0]
    return train_indices, eval_indices


def get_train_eval_split_filename(image_filenames: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    basenames = [path.name for path in image_filenames]
    all_indices = np.arange(len(image_filenames))
    train_indices = [idx for idx, name in zip(all_indices, basenames) if "train" in name]
    eval_indices = [idx for idx, name in zip(all_indices, basenames) if "eval" in name or "test" in name]
    if not train_indices or not eval_indices:
        raise ValueError("Filename split requires image names containing train/eval or train/test.")
    return np.array(train_indices), np.array(eval_indices)


def get_train_eval_split_all(image_filenames: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(len(image_filenames))
    return all_indices, all_indices
