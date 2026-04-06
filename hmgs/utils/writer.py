# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generic Writer class for HMGS
"""

from __future__ import annotations

import enum
import os
from abc import abstractmethod
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Union

import torch
from jaxtyping import Float
from torch import Tensor


EVENT_WRITERS = []
EVENT_STORAGE = []
GLOBAL_BUFFER = {}


class EventName(enum.Enum):
    """Names of possible events that can be logged via Local Writer for convenience."""

    ITER_TRAIN_TIME = "Train Iter (time)"
    TOTAL_TRAIN_TIME = "Train Total (time)"
    ETA = "ETA (time)"
    TRAIN_RAYS_PER_SEC = "Train Rays / Sec"
    TEST_RAYS_PER_SEC = "Test Rays / Sec"
    VIS_RAYS_PER_SEC = "Vis Rays / Sec"
    CURR_TEST_PSNR = "Test PSNR"


def to8b(x):
    """Converts a torch tensor to 8 bit"""
    return (255 * torch.clamp(x, min=0, max=1)).to(torch.uint8)


def is_initialized():
    """
    Returns True after setup was called
    """
    return "events" in GLOBAL_BUFFER


def put_scalar(name: str, scalar: Any, step: int):
    """Setter function to place scalars into the queue to be written out

    Args:
        name: name of scalar
        scalar: value
        step: step associated with scalar
    """
    if isinstance(name, EventName):
        name = name.value

    EVENT_STORAGE.append({"name": name, "event": scalar, "step": step})


def put_time(name: str, duration: float, step: int, avg_over_steps: bool = True, update_eta: bool = False):
    """Setter function to place a time element into the queue to be written out.

    Args:
        name: name of time item
        duration: value
        step: step associated with value
        avg_over_steps: if True, calculate and record a running average of the times
        update_eta: if True, update the ETA. should only be set for the training iterations/s
    """
    if isinstance(name, EventName):
        name = name.value

    if avg_over_steps:
        GLOBAL_BUFFER["step"] = step
        curr_event = GLOBAL_BUFFER["events"].get(name, {"buffer": [], "avg": 0})
        curr_buffer = curr_event["buffer"]
        if len(curr_buffer) >= GLOBAL_BUFFER["max_buffer_size"]:
            curr_buffer.pop(0)
        curr_buffer.append(duration)
        curr_avg = sum(curr_buffer) / len(curr_buffer)
        put_scalar(name, curr_avg, step)
        GLOBAL_BUFFER["events"][name] = {"buffer": curr_buffer, "avg": curr_avg}
    else:
        put_scalar(name, duration, step)


class TimeWriter:
    """Timer context manager that calculates duration around wrapped functions"""

    def __init__(self, writer, name, step=None, write=True):
        self.writer = writer
        self.name = name
        self.step = step
        self.write = write

        self.start: float = 0.0
        self.duration: float = 0.0

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, *args):
        self.duration = time() - self.start
        update_step = self.step is not None
        if self.write and is_initialized():
            self.writer.put_time(
                name=self.name,
                duration=self.duration,
                step=self.step if update_step else GLOBAL_BUFFER["max_iter"],
                avg_over_steps=update_step,
                update_eta=self.name == EventName.ITER_TRAIN_TIME,
            )