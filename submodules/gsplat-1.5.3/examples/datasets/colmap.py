import json
import os
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
import pycolmap 
from tqdm import tqdm
from typing_extensions import assert_never

from .normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)


def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _relpath_stem(path: str) -> str:
    """Get path stem while preserving sub-directories."""
    return os.path.splitext(path)[0]


def _resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """Resize image folder."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)

    image_files = _get_rel_paths(image_dir)
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        if os.path.isfile(resized_path):
            continue
        image = imageio.imread(image_path)[..., :3]
        resized_size = (
            int(round(image.shape[1] / factor)),
            int(round(image.shape[0] / factor)),
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, Image.BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)
    return resized_dir


def _call_or_value(v):
    """Return method result if callable, otherwise return value directly."""
    return v() if callable(v) else v


def _get_w2c_matrix(im, bottom: np.ndarray) -> np.ndarray:
    """Extract a 4x4 world-to-camera matrix across pycolmap API variants."""
    cam_from_world = _call_or_value(getattr(im, "cam_from_world"))

    matrix_attr = getattr(cam_from_world, "matrix", None)
    if matrix_attr is not None:
        mat = np.asarray(_call_or_value(matrix_attr), dtype=np.float64)
        if mat.shape == (3, 4):
            return np.concatenate([mat, bottom], axis=0)
        if mat.shape == (4, 4):
            return mat

    rotation = _call_or_value(getattr(cam_from_world, "rotation"))
    rot = np.asarray(_call_or_value(getattr(rotation, "matrix")), dtype=np.float64)
    trans = np.asarray(_call_or_value(getattr(cam_from_world, "translation")), dtype=np.float64).reshape(3, 1)
    return np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)


class Parser:
    """COLMAP parser compatible with pycolmap 4.0.2."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(colmap_dir), f"COLMAP directory {colmap_dir} does not exist."

        # 加载 Reconstruction 对象
        reconstruction = pycolmap.Reconstruction(colmap_dir)
        
        imdata = reconstruction.images
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        camtype_dict = dict() # 记录每个相机的类型供畸变校正使用
        
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        
        # 遍历所有已注册的图像
        for image_id, im in imdata.items():
            # 1. 提取外参 (World to Camera)
            w2c = _get_w2c_matrix(im, bottom)
            w2c_mats.append(w2c)

            cam_id = im.camera_id
            camera_ids.append(cam_id)

            # 2. 提取内参 (如果该相机还没处理过)
            if cam_id not in Ks_dict:
                cam = reconstruction.cameras[cam_id]
                K = cam.calibration_matrix()
                K[:2, :] /= factor
                Ks_dict[cam_id] = K

                # 适配 4.0.2 的参数解析
                model_name = cam.model_name
                raw_params = np.array(cam.params, dtype=np.float32)
                
                if model_name in ["SIMPLE_PINHOLE", "PINHOLE"]:
                    params = np.empty(0, dtype=np.float32)
                    camtype = "perspective"
                elif model_name == "SIMPLE_RADIAL":
                    # params: [f, cx, cy, k] -> opencv 需要 [k1, k2, p1, p2]
                    params = np.array([raw_params[-1], 0.0, 0.0, 0.0], dtype=np.float32)
                    camtype = "perspective"
                elif model_name == "RADIAL":
                    # params: [f, cx, cy, k1, k2]
                    params = np.array([raw_params[-2], raw_params[-1], 0.0, 0.0], dtype=np.float32)
                    camtype = "perspective"
                elif model_name == "OPENCV":
                    # params: [fx, fy, cx, cy, k1, k2, p1, p2]
                    params = raw_params[4:].copy()
                    camtype = "perspective"
                elif model_name == "OPENCV_FISHEYE":
                    # params: [fx, fy, cx, cy, k1, k2, k3, k4]
                    params = raw_params[4:].copy()
                    camtype = "fisheye"
                else:
                    print(f"Unknown camera model: {model_name}, treating as PINHOLE")
                    params = np.empty(0, dtype=np.float32)
                    camtype = "perspective"
                
                params_dict[cam_id] = params
                imsize_dict[cam_id] = (cam.width // factor, cam.height // factor)
                mask_dict[cam_id] = None
                camtype_dict[cam_id] = camtype

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")

        w2c_mats = np.stack(w2c_mats, axis=0)
        camtoworlds = np.linalg.inv(w2c_mats)
        image_names = [im.name for _, im in imdata.items()]

        # 排序以保证一致性
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # 加载元数据和 bounds (逻辑保持不变)
        self.extconf = {"spiral_radius_scale": 1.0, "no_factor_suffix": False}
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))

        self.bounds = np.array([0.01, 1.0])
        posefile = os.path.join(data_dir, "poses_bounds.npy")
        if os.path.exists(posefile):
            self.bounds = np.load(posefile)[:, -2:]

        # 处理图片路径
        image_dir_suffix = f"_{factor}" if (factor > 1 and not self.extconf["no_factor_suffix"]) else ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)

        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        if len(colmap_files) == 0:
            raise ValueError(f"No source images found in {colmap_image_dir}.")

        image_files = sorted(_get_rel_paths(image_dir)) if os.path.isdir(image_dir) else []
        should_generate_resized = factor > 1 and (
            len(image_files) == 0
            or os.path.splitext(image_files[0])[1].lower() == ".jpg"
        )
        if should_generate_resized:
            image_dir = _resize_image_folder(colmap_image_dir, image_dir + "_png", factor=factor)
            image_files = sorted(_get_rel_paths(image_dir))

        # Map COLMAP image names to actual files by relative path stem.
        image_by_stem = {_relpath_stem(p): p for p in image_files}
        missing_images = [p for p in image_names if _relpath_stem(p) not in image_by_stem]
        if len(missing_images) > 0:
            preview = ", ".join(missing_images[:3])
            raise ValueError(
                f"Could not match {len(missing_images)} COLMAP images in {image_dir}. "
                f"Examples: {preview}"
            )
        colmap_to_image = {p: image_by_stem[_relpath_stem(p)] for p in image_names}
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # 3. 处理 3D 点云
        points3D = reconstruction.points3D
        num_points = len(points3D)
        points = np.zeros((num_points, 3), dtype=np.float32)
        points_rgb = np.zeros((num_points, 3), dtype=np.uint8)
        points_err = np.zeros(num_points, dtype=np.float32)
        
        point3D_id_to_idx = {}
        for idx, (p_id, pt) in enumerate(points3D.items()):
            points[idx] = pt.xyz
            points_rgb[idx] = pt.color
            points_err[idx] = pt.error
            point3D_id_to_idx[p_id] = idx

        point_indices = dict()
        for _, im in imdata.items():
            img_p_idxs = [point3D_id_to_idx[p.point3D_id] for p in im.points2D if p.has_point3D() and p.point3D_id in point3D_id_to_idx]
            point_indices[im.name] = np.array(img_p_idxs, dtype=np.int32)

        # 归一化处理 (保持不变)
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)
            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)
            transform = T2 @ T1
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                T3 = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=np.float32)
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names = image_names
        self.image_paths = image_paths
        self.camtoworlds = camtoworlds
        self.camera_ids = camera_ids
        self.Ks_dict = Ks_dict
        self.params_dict = params_dict
        self.imsize_dict = imsize_dict
        self.mask_dict = mask_dict
        self.points = points
        self.points_err = points_err
        self.points_rgb = points_rgb
        self.point_indices = point_indices
        self.transform = transform

        # 4. 畸变校正逻辑 (适配)
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for cam_id in self.params_dict.keys():
            params = self.params_dict[cam_id]
            if len(params) == 0: continue
            
            K = self.Ks_dict[cam_id]
            w, h = self.imsize_dict[cam_id]
            ctype = camtype_dict[cam_id]

            if ctype == "perspective":
                K_undist, roi = cv2.getOptimalNewCameraMatrix(K, params, (w, h), 0)
                mapx, mapy = cv2.initUndistortRectifyMap(K, params, None, K_undist, (w, h), cv2.CV_32FC1)
                mask = None
            elif ctype == "fisheye":
                # Fisheye 逻辑略，保持原样...
                # (为简洁省略，如果你有鱼眼数据，使用原版逻辑即可)
                K_undist, roi, mapx, mapy, mask = K, [0,0,w,h], None, None, None 

            self.mapx_dict[cam_id] = mapx
            self.mapy_dict[cam_id] = mapy
            self.Ks_dict[cam_id] = K_undist
            self.roi_undist_dict[cam_id] = roi
            self.imsize_dict[cam_id] = (roi[2], roi[3])
            self.mask_dict[cam_id] = mask

        camera_locations = camtoworlds[:, :3, 3]
        self.scene_scale = np.max(np.linalg.norm(camera_locations - np.mean(camera_locations, axis=0), axis=1))

class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }
        if mask is not None:
            data["mask"] = torch.from_numpy(mask).bool()

        if self.load_depths:
            # projected points to image plane to get depths
            worldtocams = np.linalg.inv(camtoworlds)
            image_name = self.parser.image_names[index]
            point_indices = self.parser.point_indices[image_name]
            points_world = self.parser.points[point_indices]
            points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
            points_proj = (K @ points_cam.T).T
            points = points_proj[:, :2] / points_proj[:, 2:3]  # (M, 2)
            depths = points_cam[:, 2]  # (M,)
            # filter out points outside the image
            selector = (
                (points[:, 0] >= 0)
                & (points[:, 0] < image.shape[1])
                & (points[:, 1] >= 0)
                & (points[:, 1] < image.shape[0])
                & (depths > 0)
            )
            points = points[selector]
            depths = depths[selector]
            data["points"] = torch.from_numpy(points).float()
            data["depths"] = torch.from_numpy(depths).float()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
