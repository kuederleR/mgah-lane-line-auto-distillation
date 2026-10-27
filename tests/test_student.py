"""Student head, masked loss, and a CPU training step."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from src.common.device import mps_available, resolve_device
from src.common.geometry import LANE_OUTPUT_DIM, MAX_LANES
from src.student.dataset import SyntheticLaneDataset
from src.student.loss import LaneSplineLoss
from src.student.model import LaneSplineStudent
from src.student.train import TrainConfig, run_training

ROOT = Path(__file__).resolve().parents[1]


def test_device_order_accepts_cpu_and_rejects_a_missing_backend():
    assert resolve_device("cpu").type == "cpu"
    chosen = resolve_device("auto")
    assert chosen.type in {"cpu", "cuda", "mps"}
    if not mps_available():
        try:
            resolve_device("mps")
        except RuntimeError:
            return
        raise AssertionError("MPS was requested on a machine without it")


def test_student_heads_and_masked_loss():
    tiny = LaneSplineStudent(backbone="tiny", distill_dim=8)
    images = torch.rand(2, 3, 64, 96)
    output = tiny(images)
    assert output["lanes"].shape == (2, MAX_LANES, LANE_OUTPUT_DIM)
    assert output["distill_features"].shape[0] == 2
    assert output["distill_features"].shape[1] == 8

    resnet = LaneSplineStudent(backbone="resnet18")
    assert resnet(torch.rand(1, 3, 64, 96))["lanes"].shape == (1, MAX_LANES, LANE_OUTPUT_DIM)

    absent = torch.zeros(1, MAX_LANES, LANE_OUTPUT_DIM)
    pred = absent.clone()
    pred[..., 1:] = 5.0
    masked = LaneSplineLoss()({"lanes": pred}, absent)
    assert masked["knot"].item() == 0.0
    assert masked["existence"].item() > 0.0

    present = absent.clone()
    present[0, 0, 0] = 1.0
    opened = LaneSplineLoss()({"lanes": pred}, present)
    assert opened["knot"].item() > 0.0
    trained = LaneSplineLoss()(output, torch.zeros_like(output["lanes"]))
    trained["loss"].backward()
    assert tiny.head.weight.grad is not None


def test_training_step_on_cpu(tmp_path):
    history = run_training(
        TrainConfig(
            synthetic=True,
            epochs=1,
            batch_size=2,
            max_steps=2,
            backbone="tiny",
            device="cpu",
            image_height=64,
            image_width=96,
            output=str(tmp_path / "student.pt"),
            synthetic_length=4,
        )
    )
    assert history["loss"] > 0.0
    assert (tmp_path / "student.pt").is_file()
    sample = SyntheticLaneDataset(length=1, image_height=32, image_width=48)[0]
    assert sample["image"].shape == (3, 32, 48)
    assert sample["target"].shape == (MAX_LANES, LANE_OUTPUT_DIM)


def test_train_module_entrypoint(tmp_path):
    checkpoint = tmp_path / "student.pt"
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "src.student.train",
            "--synthetic",
            "--epochs",
            "1",
            "--max-steps",
            "1",
            "--batch-size",
            "2",
            "--backbone",
            "tiny",
            "--image-height",
            "64",
            "--image-width",
            "96",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--output",
            str(checkpoint),
        ],
        cwd=ROOT,
    )
    assert checkpoint.is_file()
