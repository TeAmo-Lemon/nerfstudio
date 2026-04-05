#!/usr/bin/env python
"""Train Splatfacto with DINOv2 feature distillation enabled by default."""

from __future__ import annotations

import sys

from nerfstudio.scripts import train


def entrypoint() -> None:
    """Entrypoint for use with pyproject scripts.

    This wrapper prepends the `splatfacto-dino` method subcommand so users can run
    a dedicated training command while keeping full parity with `ns-train` flags.
    """
    sys.argv = [sys.argv[0], "splatfacto-dino", *sys.argv[1:]]
    train.entrypoint()


if __name__ == "__main__":
    entrypoint()
