"""Train the lane student on Apple Silicon, CUDA, or CPU.

Device selection lives in ``src.common.device.resolve_device``:

    mps if torch.backends.mps.is_available() else cuda else cpu

Run from the repo root:

    python -m src.student.train --synthetic --epochs 1 --max-steps 2 --backbone tiny
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.config import load_project_env, load_yaml
from src.common.device import resolve_device
from src.student.dataset import LaneSplineDataset, SyntheticLaneDataset
from src.student.loss import LaneSplineLoss
from src.student.model import LaneSplineStudent

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    data: str | None = None
    synthetic: bool = False
    epochs: int = 10
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "auto"
    backbone: str = "resnet18"
    pretrained: bool = False
    image_height: int = 288
    image_width: int = 512
    num_workers: int = 0
    max_steps: int = 0
    output: str = "artifacts/student.pt"
    distill_dim: int = 0
    distill_weight: float = 0.0
    knot_weight: float = 1.0
    exist_weight: float = 1.0
    focal: bool = False
    seed: int = 0
    synthetic_length: int = 32


def run_training(config: TrainConfig) -> dict[str, float]:
    """Train and write a checkpoint. Returns the last epoch's mean losses."""
    set_seed(config.seed)
    device = resolve_device(config.device)
    logger.info("training on %s", device)
    if config.synthetic:
        dataset = SyntheticLaneDataset(
            length=config.synthetic_length,
            image_height=config.image_height,
            image_width=config.image_width,
            seed=config.seed,
        )
    elif config.data:
        dataset = LaneSplineDataset(config.data, config.image_height, config.image_width)
    else:
        raise ValueError("pass --data or --synthetic")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = LaneSplineStudent(
        backbone=config.backbone,
        pretrained=config.pretrained,
        distill_dim=config.distill_dim,
    ).to(device)
    criterion = LaneSplineLoss(
        knot_weight=config.knot_weight,
        exist_weight=config.exist_weight,
        distill_weight=config.distill_weight,
        focal=config.focal,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: dict[str, float] = {}
    best = float("inf")
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        model.train()
        totals = {"loss": 0.0, "knot": 0.0, "existence": 0.0}
        steps = 0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{config.epochs}", leave=False)
        for batch in progress:
            images = batch["image"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            losses = criterion(outputs, target)
            losses["loss"].backward()
            optimizer.step()
            for key in totals:
                totals[key] += float(losses[key].detach())
            steps += 1
            progress.set_postfix(loss=totals["loss"] / steps)
            if config.max_steps and steps >= config.max_steps:
                break
        if steps == 0:
            raise RuntimeError("data loader produced no batches")
        history = {key: value / steps for key, value in totals.items()}
        if not np.isfinite(history["loss"]):
            raise RuntimeError(f"non-finite training loss: {history}")
        logger.info(
            "epoch %d loss %.4f knot %.4f existence %.4f",
            epoch + 1,
            history["loss"],
            history["knot"],
            history["existence"],
        )
        _save_checkpoint(output.with_name(output.stem + "_last.pt"), model, config, epoch, history)
        if history["loss"] <= best:
            best = history["loss"]
            _save_checkpoint(output, model, config, epoch, history)
    return history


def _save_checkpoint(
    path: Path,
    model: LaneSplineStudent,
    config: TrainConfig,
    epoch: int,
    history: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "train_loss": history.get("loss"),
            "backbone": model.backbone_name,
            "config": asdict(config),
        },
        path,
    )
    logger.info("wrote %s", path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the monocular 3D lane student")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-height", type=int, default=288)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--output", type=str, default="artifacts/student.pt")
    parser.add_argument("--distill-dim", type=int, default=0)
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--knot-weight", type=float, default=1.0)
    parser.add_argument("--exist-weight", type=float, default=1.0)
    parser.add_argument("--focal", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--synthetic-length", type=int, default=32)
    return parser


def config_from_args(argv: list[str] | None = None) -> TrainConfig:
    """YAML from ``--config`` fills defaults; explicit flags override them."""
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path, default=None)
    known, _rest = preliminary.parse_known_args(argv)
    parser = build_parser()
    if known.config is not None:
        parser.set_defaults(**_yaml_defaults(load_yaml(known.config)))
    args = parser.parse_args(argv)
    return TrainConfig(
        data=args.data,
        synthetic=args.synthetic,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        backbone=args.backbone,
        pretrained=args.pretrained,
        image_height=args.image_height,
        image_width=args.image_width,
        num_workers=args.num_workers,
        max_steps=args.max_steps,
        output=args.output,
        distill_dim=args.distill_dim,
        distill_weight=args.distill_weight,
        knot_weight=args.knot_weight,
        exist_weight=args.exist_weight,
        focal=args.focal,
        seed=args.seed,
        synthetic_length=args.synthetic_length,
    )


def _yaml_defaults(cfg: dict) -> dict:
    loss = cfg.get("loss") or {}
    mapped = {
        "epochs": cfg.get("epochs"),
        "batch_size": cfg.get("batch_size"),
        "lr": cfg.get("lr"),
        "weight_decay": cfg.get("weight_decay"),
        "device": cfg.get("device"),
        "backbone": cfg.get("backbone"),
        "pretrained": cfg.get("pretrained"),
        "image_height": cfg.get("image_height"),
        "image_width": cfg.get("image_width"),
        "num_workers": cfg.get("num_workers"),
        "output": cfg.get("output"),
        "distill_dim": cfg.get("distill_dim"),
        "distill_weight": cfg.get("distill_weight"),
        "knot_weight": loss.get("knot_weight"),
        "exist_weight": loss.get("exist_weight"),
        "focal": loss.get("focal"),
    }
    return {key: value for key, value in mapped.items() if value is not None}


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    load_project_env()
    config = config_from_args(argv)
    run_training(config)


if __name__ == "__main__":
    main()
