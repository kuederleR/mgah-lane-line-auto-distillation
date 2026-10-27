"""ONNX parity, the TensorRT command line, and the headless dual view."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.common.geometry import intrinsics_matrix
from src.common.types import Lane3D, SplineKnot3D
from src.edge.trt_runner import TensorRTRunner
from src.edge.visualizer import render_dual_view, save_spline_ply
from src.export.export_onnx import export_onnx
from src.student.model import LaneSplineStudent


def _lane() -> Lane3D:
    knots = [
        SplineKnot3D(x=-1.75, y=1.5, z=float(z))
        for z in np.linspace(8.0, 40.0, 10)
    ]
    return Lane3D(lane_id=0, lane_type="solid", confidence=1.0, knots=knots, role="ego_left")


def test_onnx_matches_pytorch_and_ort_runner(tmp_path: Path):
    student = LaneSplineStudent(backbone="tiny")
    torch.nn.init.normal_(student.head.weight, std=0.02)
    checkpoint = tmp_path / "student.pt"
    torch.save(
        {
            "model": student.state_dict(),
            "backbone": "tiny",
            "config": {
                "backbone": "tiny",
                "image_height": 64,
                "image_width": 96,
                "distill_dim": 0,
            },
        },
        checkpoint,
    )
    onnx_path = export_onnx(
        tmp_path / "student.onnx",
        checkpoint=checkpoint,
        dynamic_batch=True,
    )
    images = np.random.default_rng(0).random((2, 3, 64, 96), dtype=np.float32)
    with torch.no_grad():
        reference = student(torch.from_numpy(images))["lanes"].numpy()
    runner = TensorRTRunner(onnx_path=onnx_path)
    assert runner.load() == "onnxruntime"
    np.testing.assert_allclose(runner.infer(images), reference, atol=1e-4, rtol=1e-4)


def test_trtexec_fp16_command():
    command = TensorRTRunner.trtexec_command("artifacts/student.onnx", "artifacts/student_fp16.engine")
    assert command[0] == "trtexec"
    assert "--fp16" in command
    assert any(part.startswith("--onnx=") for part in command)
    assert any(part.startswith("--saveEngine=") for part in command)


def test_dual_view_is_headless_and_wider_than_the_camera(tmp_path: Path):
    image = np.full((180, 320, 3), 30, dtype=np.uint8)
    canvas = render_dual_view(image, [_lane()], intrinsics_matrix(400, 400, 160, 90), camera_height_m=1.5)
    assert canvas.shape[0] == image.shape[0]
    assert canvas.shape[1] > image.shape[1]
    assert not np.array_equal(canvas[:, : image.shape[1]], image)
    ply = tmp_path / "lanes.ply"
    save_spline_ply(ply, [_lane()], camera_height_m=1.5)
    text = ply.read_text(encoding="utf-8")
    assert text.startswith("ply")
    assert "element vertex 10" in text
