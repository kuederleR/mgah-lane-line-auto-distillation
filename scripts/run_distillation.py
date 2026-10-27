#!/usr/bin/env python3
"""Distill a clip into camera-frame lane splines (JSON + NPZ)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_project_env, load_yaml  # noqa: E402
from src.teacher.pipeline import distill_clip  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_project_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/synthetic_clip"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/synthetic_clip"))
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.config is not None:
        config = load_yaml(args.config)
    elif Path("configs/teacher_config.yaml").is_file():
        config = load_yaml("configs/teacher_config.yaml")
    else:
        config = {}
    index = distill_clip(args.input, args.output, config)
    print(f"kept {index.frame_count()} frames in {args.output} ({index.notes})")


if __name__ == "__main__":
    main()
