"""Export ``LaneSplineStudent`` to ONNX.

The graph input is ``images``: RGB float NCHW in ``[0, 1]``.
The graph output is ``lanes``: ``[B, 4, 31]``.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import torch

from src.common.config import load_yaml
from src.student.model import LaneSplineStudent, StudentOnnxWrapper

logger = logging.getLogger(__name__)


def export_onnx(
    output: str | Path,
    checkpoint: str | Path | None = None,
    backbone: str = "resnet18",
    image_height: int = 288,
    image_width: int = 512,
    distill_dim: int = 0,
    opset: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Write an ONNX file and return its path."""
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu")
        cfg = payload.get("config") or {}
        backbone = payload.get("backbone") or cfg.get("backbone") or backbone
        image_height = int(cfg.get("image_height", image_height))
        image_width = int(cfg.get("image_width", image_width))
        distill_dim = int(cfg.get("distill_dim", distill_dim))
        student = LaneSplineStudent(backbone=backbone, pretrained=False, distill_dim=distill_dim)
        student.load_state_dict(payload["model"])
    else:
        student = LaneSplineStudent(backbone=backbone, pretrained=False, distill_dim=distill_dim)
    student.eval()
    wrapper = StudentOnnxWrapper(student).eval()
    dummy = torch.zeros(1, 3, image_height, image_width)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    dynamic = {"images": {0: "batch"}, "lanes": {0: "batch"}} if dynamic_batch else None
    _torch_export(wrapper, dummy, path, dynamic, opset)
    logger.info("exported %s (%s, %dx%d)", path, student.backbone_name, image_height, image_width)
    return path


def _torch_export(
    model: torch.nn.Module,
    dummy: torch.Tensor,
    path: Path,
    dynamic: dict | None,
    opset: int,
) -> None:
    kwargs = dict(
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["lanes"],
        dynamic_axes=dynamic,
    )
    # Torch 2.9+ defaults to the dynamo exporter and warns that dynamo=False is
    # legacy. The scripted exporter is what the ORT parity test covers.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"torch\.onnx\..*")
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=".*legacy TorchScript-based ONNX export.*",
        )
        try:
            torch.onnx.export(model, dummy, str(path), dynamo=False, **kwargs)
        except TypeError:
            torch.onnx.export(model, dummy, str(path), **kwargs)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Export the lane student to ONNX")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/student.onnx"))
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--image-height", type=int, default=288)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic-batch", action="store_true", default=False)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.config is not None:
        cfg = load_yaml(args.config)
        args.backbone = cfg.get("backbone", args.backbone)
        args.image_height = int(cfg.get("image_height", args.image_height))
        args.image_width = int(cfg.get("image_width", args.image_width))
    export_onnx(
        output=args.output,
        checkpoint=args.checkpoint,
        backbone=args.backbone,
        image_height=args.image_height,
        image_width=args.image_width,
        opset=args.opset,
        dynamic_batch=args.dynamic_batch,
    )


if __name__ == "__main__":
    main()
