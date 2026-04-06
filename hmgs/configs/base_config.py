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

"""Base Configs for HMGS"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Type


# Pretty printing class
class PrintableConfig:
    """Printable Config defining str function"""

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


# Base instantiate configs
@dataclass
class InstantiateConfig(PrintableConfig):
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)


# Viewer related configs
@dataclass
class ViewerConfig(PrintableConfig):
    """Configuration for viewer instantiation"""

    relative_log_filename: str = "viewer_log_filename.txt"
    """Filename to use for the log file."""
    websocket_port: Optional[int] = None
    """The websocket port to connect to. If None, find an available port."""
    websocket_port_default: int = 7007
    """The default websocket port to connect to if websocket_port is not specified"""
    websocket_host: str = "0.0.0.0"
    """The host address to bind the websocket server to."""
    num_rays_per_chunk: int = 32768
    """number of rays per chunk to render with viewer"""
    max_num_display_images: int = 512
    """Maximum number of training images to display in the viewer, to avoid lag."""
    quit_on_train_completion: bool = False
    """Whether to kill the training job when it has completed."""
    image_format: Literal["jpeg", "png"] = "jpeg"
    """Image format viewer should use; jpeg is lossy compression, while png is lossless."""
    jpeg_quality: int = 75
    """Quality tradeoff to use for jpeg compression."""
    make_share_url: bool = False
    """Viewer beta feature: print a shareable URL."""
    camera_frustum_scale: float = 0.1
    """Scale for the camera frustums in the viewer."""
    default_composite_depth: bool = True
    """The default value for compositing depth."""