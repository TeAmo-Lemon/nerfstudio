#!/usr/bin/env python
"""Launch the interactive viewer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Nerfstudio viewer.")
    parser.add_argument("--config", type=Path, required=True, help="Path to config YAML file.")
    parser.add_argument("--port", type=int, default=None, help="Websocket port override.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from scripts.viewer.run_viewer import RunViewer
    sys.exit(RunViewer.main())


if __name__ == "__main__":
    main()
