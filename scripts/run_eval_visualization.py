#!/usr/bin/env python3
"""Draw teacher labels or a student checkpoint as a dual-view PNG plus a PLY."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.device import resolve_device  # noqa: E402
from src.common.types import DistillationDataset, Frame3DAnnotation  # noqa: E402
from src.edge.visualizer import render_dual_view, save_spline_ply  # noqa: E402
from src.student.dataset import preprocess_bgr  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    import cv2
    import torch

    from src.student.model import LaneSplineStudent, lanes_from_prediction

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/processed/synthetic_clip"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/viz"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--source", choices=("teacher", "student"), default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--existence-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)
    source = args.source
    if source is None:
        source = "student" if args.checkpoint else "teacher"
    if source == "student" and args.checkpoint is None:
        raise SystemExit("--source student needs --checkpoint")

    index = DistillationDataset.model_validate_json((args.labels / "index.json").read_text(encoding="utf-8"))
    model = None
    image_height, image_width = 288, 512
    if source == "student":
        payload = torch.load(args.checkpoint, map_location="cpu")
        cfg = payload.get("config") or {}
        image_height = int(cfg.get("image_height", image_height))
        image_width = int(cfg.get("image_width", image_width))
        model = LaneSplineStudent(
            backbone=payload.get("backbone") or cfg.get("backbone") or "resnet18",
            pretrained=False,
            distill_dim=int(cfg.get("distill_dim", 0)),
        )
        model.load_state_dict(payload["model"])
        model.to(resolve_device(args.device))
        model.eval()

    args.output.mkdir(parents=True, exist_ok=True)
    written = 0
    for name in index.frames:
        annotation = Frame3DAnnotation.model_validate_json((args.labels / name).read_text(encoding="utf-8"))
        image_path = (args.labels / index.image_root / annotation.image).resolve()
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(image_path)
        if source == "teacher":
            lanes = annotation.lanes
        else:
            assert model is not None
            chw = preprocess_bgr(frame, image_height, image_width)
            tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
            tensor = tensor.to(next(model.parameters()).device)
            with torch.no_grad():
                packed = model(tensor)["lanes"][0].detach().cpu().numpy()
            lanes = lanes_from_prediction(packed, threshold=args.existence_threshold)
        canvas = render_dual_view(
            frame,
            lanes,
            np.asarray(annotation.camera_k, dtype=np.float64),
            camera_height_m=annotation.camera_height_m,
            pitch_rad=annotation.pitch_rad,
        )
        stem = Path(name).stem
        cv2.imwrite(str(args.output / f"{stem}.png"), canvas)
        save_spline_ply(
            args.output / f"{stem}.ply",
            lanes,
            camera_height_m=annotation.camera_height_m,
            pitch_rad=annotation.pitch_rad,
        )
        written += 1
    print(f"wrote {written} dual-view frames to {args.output} from {source}")


if __name__ == "__main__":
    main()
