from __future__ import annotations

from pathlib import Path
from typing import IO, Union

import cv2
import numpy as np
import torch
from PIL import Image
from PIL.Image import Image as PILImage


def pil_to_numpy(im: PILImage) -> np.ndarray:
    im.load()
    encoder = Image._getencoder(im.mode, "raw", im.mode)
    encoder.setimage(im.im)
    shape, typestr = Image._conv_type_shape(im)
    data = np.empty(shape, dtype=np.dtype(typestr))
    mem = data.data.cast("B", (data.data.nbytes,))
    bufsize, s, offset = 65536, 0, 0
    while not s:
        _, s, d = encoder.encode(bufsize)
        mem[offset : offset + len(d)] = d
        offset += len(d)
    if s < 0:
        raise RuntimeError(f"encoder error {s} in tobytes")
    return data


def get_image_mask_tensor_from_path(filepath: Union[Path, IO[bytes]], scale_factor: float = 1.0) -> torch.Tensor:
    pil_mask = Image.open(filepath)
    if scale_factor != 1.0:
        width, height = pil_mask.size
        pil_mask = pil_mask.resize((int(width * scale_factor), int(height * scale_factor)), resample=Image.Resampling.NEAREST)
    mask_tensor = torch.from_numpy(pil_to_numpy(pil_mask)).unsqueeze(-1).bool()
    if len(mask_tensor.shape) != 3:
        raise ValueError("Mask image should have a single channel")
    return mask_tensor


def identity_collate(x):
    return x


def get_depth_image_from_path(filepath: Path, height: int, width: int, scale_factor: float) -> torch.Tensor:
    if filepath.suffix == ".npy":
        image = np.load(filepath).astype(np.float32) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    else:
        image = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH).astype(np.float32) * scale_factor
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(image[:, :, np.newaxis])
