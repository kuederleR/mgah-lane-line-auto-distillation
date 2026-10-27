"""Teacher geometry, mocks, and the synthetic-clip distillation path."""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from src.common.geometry import ego_to_camera, ground_plane_depth, intrinsics_matrix, project_points, role_index
from src.common.types import Frame3DAnnotation, Lane3D, SplineKnot3D
from src.teacher.auditor import VLMGeometricAuditor, classify_marking, geometric_audit, parse_vlm_payload
from src.teacher.dinov3_depth import DINOv3DepthEstimator
from src.teacher.lifter import LaneLifter
from src.teacher.meta_client import MetaModelClient
from src.teacher.pipeline import distill_clip
from src.teacher.synthetic import LANE_LAYOUT, generate_synthetic_clip


def _straight_lane(role: str, x: float, spike: bool = False) -> Lane3D:
    knots = []
    for index, z in enumerate(np.linspace(6.0, 42.0, 10)):
        y = 8.0 if spike and index == 5 else 1.5
        knots.append(SplineKnot3D(x=x, y=y, z=float(z)))
    return Lane3D(
        lane_id=role_index(role),
        lane_type="solid",
        confidence=1.0,
        knots=knots,
        role=role,
    )


def test_lifter_recovers_a_known_ground_lane():
    k = intrinsics_matrix(700.0, 700.0, 320.0, 180.0)
    z = np.linspace(10.0, 40.0, 40)
    ego = np.stack([np.full_like(z, -1.75), z, np.zeros_like(z)], axis=1)
    camera = ego_to_camera(ego, camera_height_m=1.5, pitch_rad=0.0)
    pixels = np.round(project_points(camera, k)).astype(np.int32)
    mask = np.zeros((360, 640), dtype=np.uint8)
    import cv2

    cv2.polylines(mask, [pixels.reshape(-1, 1, 2)], False, 1, 3, cv2.LINE_8)
    depth = ground_plane_depth(360, 640, k, camera_height_m=1.5, pitch_rad=0.0)
    lane = LaneLifter().lift_mask(mask, depth, k, lane_id=0, lane_type="solid", role="ego_left")
    assert lane is not None
    knots = np.asarray(lane.knot_array())
    assert knots.shape == (10, 3)
    np.testing.assert_allclose(knots[:, 0], -1.75, atol=0.15)
    np.testing.assert_allclose(knots[:, 1], 1.5, atol=0.15)
    assert knots[0, 2] < knots[-1, 2]


def test_geometric_audit_rejects_a_z_spike_and_crossed_ego_lanes():
    clean = geometric_audit([_straight_lane("ego_left", -1.75), _straight_lane("ego_right", 1.75)])
    spiked = geometric_audit([_straight_lane("ego_left", -1.75, spike=True), _straight_lane("ego_right", 1.75)])
    crossed = geometric_audit([_straight_lane("ego_left", 2.0), _straight_lane("ego_right", -2.0)])
    assert clean[0] > 0.9
    assert spiked[0] < clean[0] - 0.2
    assert crossed[0] < 0.5
    assert any("ego_left is not left" in issue for issue in crossed[1])


def test_marking_classifier_and_vlm_parser():
    frame = np.zeros((40, 20, 3), dtype=np.uint8)
    mask = np.zeros((40, 20), dtype=np.uint8)
    mask[5:35, 8:12] = 1
    frame[mask > 0] = (255, 255, 255)
    assert classify_marking(frame, mask) == "solid"
    dashed = mask.copy()
    dashed[10:18, :] = 0
    dashed[22:28, :] = 0
    assert classify_marking(frame, dashed) == "dashed"
    yellow = frame.copy()
    yellow[mask > 0] = (40, 220, 240)
    assert classify_marking(yellow, mask) == "double_yellow"

    payload = {"choices": [{"message": {"content": '```json\n{"score": 0.82, "issues": []}\n```'}}]}
    parsed = parse_vlm_payload(payload)
    assert parsed is not None and parsed["score"] == 0.82


def test_mock_client_does_not_post():
    client = MetaModelClient(mock=True)

    def _boom(*_args, **_kwargs):
        raise AssertionError("HTTP should not run while META_API_MOCK is on")

    client._http_post = _boom  # type: ignore[method-assign]
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    result = client.segment_video([frame], [{"frame_idx": 0, "object_id": 0, "role": "ego_left", "points": [[8, 8]], "point_labels": [1]}])
    assert result["mock"] is True
    assert result["request"]["model"] == "sam-3.1-video"
    assert result["request"]["frames"][0]["image_jpeg_b64"] is None
    assert result["request"]["frames"][0]["sha256"]
    async_result = asyncio.run(client.segment_video_async([frame], []))
    assert async_result["mock"] is True


def test_auditor_mock_returns_a_unit_interval_score():
    lanes = [_straight_lane("ego_left", -1.75), _straight_lane("ego_right", 1.75)]
    annotation = Frame3DAnnotation(
        frame_idx=0,
        timestamp_us=0,
        lanes=lanes,
        camera_k=intrinsics_matrix(700, 700, 320, 180).tolist(),
        audit_score=0.0,
    )
    report = VLMGeometricAuditor(MetaModelClient(mock=True)).audit(annotation)
    assert 0.0 <= report.score <= 1.0
    assert report.source == "geometric-mock"
    assert "ego_left" in report.prompt
    assert '"score"' in report.prompt


def test_depth_fallback_and_cache(tmp_path):
    k = intrinsics_matrix(700.0, 700.0, 160.0, 90.0)
    estimator = DINOv3DepthEstimator(
        device="cpu",
        local_files_only=True,
        camera_k=k,
        camera_height_m=1.5,
        cache_dir=tmp_path,
    )
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    depth = estimator.predict(image).detach().cpu().numpy()
    assert depth.shape == (180, 320)
    assert estimator.last_backend in {"ground_plane", "dinov3-features+ground_plane"}
    assert depth[90, 160] == 0.0
    assert depth[140, 160] > 0.0
    assert estimator.extract_features(image) is None or estimator.features_available
    cached = np.full((180, 320), 7.0, dtype=np.float32)
    np.save(tmp_path / "000000.npy", cached)
    loaded = estimator.predict(image, frame_id="000000").detach().cpu().numpy()
    np.testing.assert_allclose(loaded, 7.0)
    assert estimator.last_backend == "cache"


def test_synthetic_distillation_recovers_ego_lanes(tmp_path):
    clip = generate_synthetic_clip(tmp_path / "clip", num_frames=2)
    index = distill_clip(clip, tmp_path / "labels", config={"mock_apis": True, "audit": {"min_score": 0.35}})
    assert index.frame_count() == 2
    assert index.mock_teacher is True
    annotation = Frame3DAnnotation.model_validate_json(
        (tmp_path / "labels" / index.frames[0]).read_text(encoding="utf-8")
    )
    by_role = {lane.role: lane for lane in annotation.lanes}
    assert set(by_role) >= {"ego_left", "ego_right"}
    expected = {lane["role"]: lane["x_m"] for lane in LANE_LAYOUT}
    for role in ("ego_left", "ego_right", "left_adjacent", "right_adjacent"):
        assert role in by_role, sorted(by_role)
        mean_x = float(np.mean(by_role[role].knot_array(), axis=0)[0])
        assert mean_x == pytest.approx(expected[role], abs=0.35), role
        assert by_role[role].lane_type == next(item["lane_type"] for item in LANE_LAYOUT if item["role"] == role)
    npz = np.load(tmp_path / "labels" / "frame_000000.npz")
    assert npz["knots"].shape == (4, 10, 3)
    assert npz["existence"].tolist() == [1.0, 1.0, 1.0, 1.0]
    prompts = json.loads((tmp_path / "labels" / "sam_prompts.json").read_text(encoding="utf-8"))
    assert prompts and prompts[0]["points"]

    from scripts.run_eval_visualization import main as visualize_main
    from src.student.dataset import LaneSplineDataset

    sample = LaneSplineDataset(tmp_path / "labels", image_height=64, image_width=96)[0]
    assert sample["image"].shape == (3, 64, 96)
    assert sample["target"].shape == (4, 31)
    assert float(sample["target"][0, 0]) == 1.0
    visualize_main(
        ["--labels", str(tmp_path / "labels"), "--output", str(tmp_path / "viz"), "--source", "teacher"]
    )
    assert (tmp_path / "viz" / "frame_000000.png").is_file()
    assert (tmp_path / "viz" / "frame_000000.ply").is_file()
