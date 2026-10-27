"""Run the offline teacher on a clip directory and write JSON + NPZ labels."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from src.common.config import dig
from src.common.geometry import LANE_ROLES, intrinsics_matrix, role_index
from src.common.types import DistillationDataset, Frame3DAnnotation, Lane3D
from src.teacher.auditor import VLMGeometricAuditor
from src.teacher.dinov3_depth import DINOv3DepthEstimator
from src.teacher.lifter import LaneLifter
from src.teacher.meta_client import MetaModelClient
from src.teacher.tracker import SamVideoTracker

logger = logging.getLogger(__name__)


def distill_clip(
    clip_dir: str | Path,
    output_dir: str | Path,
    config: dict | None = None,
) -> DistillationDataset:
    """Distill ``clip_dir`` into ``output_dir`` and return the written index."""
    import cv2

    clip = Path(clip_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    intrinsics = _load_intrinsics(clip, config)
    camera_k = intrinsics_matrix(
        intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    )
    type_by_role, id_by_role = _prompt_metadata(clip)
    frame_paths = _frame_paths(clip)
    frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in frame_paths]
    if any(frame is None for frame in frames):
        raise FileNotFoundError(f"failed to read one or more frames in {clip}")

    client = MetaModelClient(
        mock=bool(dig(config, "mock_apis", default=True)),
        sam_model=str(dig(config, "models", "sam", default="sam-3.1-video")),
        vlm_model=str(dig(config, "models", "vlm", default="llama-3.2-11b-vision-instruct")),
    )
    tracker = SamVideoTracker(client)
    tracked = tracker.track(frames)
    depth_estimator = DINOv3DepthEstimator(
        model_id=str(dig(config, "models", "dino", default="facebook/dinov3-vits16-pretrain-lvd1689m")),
        local_files_only=bool(dig(config, "local_files_only", default=True)),
        fallback=str(dig(config, "depth_fallback", default="ground_plane")),
        camera_k=camera_k,
        camera_height_m=intrinsics["camera_height_m"],
        pitch_rad=intrinsics["pitch_rad"],
        device="cpu",
    )
    lifter = LaneLifter(
        num_knots=int(dig(config, "spline", "knots", default=10)),
        z_min=float(dig(config, "spline", "z_min_m", default=2.0)),
        z_max=float(dig(config, "spline", "z_max_m", default=60.0)),
    )
    auditor = VLMGeometricAuditor(client)
    min_score = float(dig(config, "audit", "min_score", default=0.35))
    image_root = Path(os.path.relpath(clip.resolve(), output.resolve()))

    kept: list[str] = []
    for frame_idx, (path, frame, tracked_frame) in enumerate(zip(frame_paths, frames, tracked.frames)):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        depth = depth_estimator.predict(rgb, frame_id=f"{frame_idx:06d}").detach().cpu().numpy()
        lanes: list[Lane3D] = []
        for role, mask in tracked_frame.masks.items():
            slot = role_index(role)
            if slot is None:
                continue
            lane = lifter.lift_mask(
                mask,
                depth,
                camera_k,
                lane_id=int(id_by_role.get(role, slot)),
                lane_type=type_by_role.get(role, "unknown"),
                role=role,
            )
            if lane is None:
                logger.info("frame %d: no spline for %s", frame_idx, role)
                continue
            lanes.append(lane)
        lanes.sort(key=lambda lane: role_index(lane.role) or 0)
        annotation = Frame3DAnnotation(
            frame_idx=frame_idx,
            timestamp_us=frame_idx * 100_000,
            lanes=lanes,
            camera_k=camera_k.tolist(),
            audit_score=0.0,
            image=str(path.resolve().relative_to(clip.resolve())),
            camera_height_m=intrinsics["camera_height_m"],
            pitch_rad=intrinsics["pitch_rad"],
        )
        report = auditor.audit(annotation, image_bgr=frame, masks=tracked_frame.masks)
        for lane in annotation.lanes:
            if lane.lane_type == "unknown" and lane.role in report.lane_types:
                lane.lane_type = report.lane_types[lane.role]
        annotation.audit_score = report.clamped()
        annotation.audit_issues = report.issues
        stem = f"frame_{frame_idx:06d}"
        (output / f"{stem}.json").write_text(annotation.model_dump_json(indent=2), encoding="utf-8")
        _write_npz(output / f"{stem}.npz", annotation, camera_k)
        if lanes and annotation.audit_score >= min_score:
            kept.append(f"{stem}.json")
        else:
            logger.info(
                "frame %d kept out of the index (lanes=%d, audit=%.2f, min=%.2f)",
                frame_idx,
                len(lanes),
                annotation.audit_score,
                min_score,
            )

    (output / "sam_prompts.json").write_text(json.dumps(tracked.prompts, indent=2), encoding="utf-8")
    index = DistillationDataset(
        name=clip.name,
        image_root=str(image_root),
        frames=kept,
        mock_teacher=client.mock,
        notes=(
            f"depth={depth_estimator.last_backend}; tracker={tracked.source}; "
            f"audit_min={min_score}"
        ),
    )
    (output / "index.json").write_text(index.model_dump_json(indent=2), encoding="utf-8")
    logger.info("wrote %d / %d frames to %s", len(kept), len(frames), output)
    return index


def _write_npz(path: Path, annotation: Frame3DAnnotation, camera_k: np.ndarray) -> None:
    knots = np.zeros((len(LANE_ROLES), 10, 3), dtype=np.float32)
    existence = np.zeros((len(LANE_ROLES),), dtype=np.float32)
    for lane in annotation.lanes:
        slot = role_index(lane.role)
        if slot is None:
            slot = lane.lane_id
        existence[slot] = 1.0
        knots[slot] = np.asarray(lane.knot_array(), dtype=np.float32)
    np.savez_compressed(
        path,
        knots=knots,
        existence=existence,
        camera_k=np.asarray(camera_k, dtype=np.float32),
        roles=np.asarray(LANE_ROLES),
        audit_score=np.float32(annotation.audit_score),
    )


def _frame_paths(clip: Path) -> list[Path]:
    frame_dir = clip / "frames" if (clip / "frames").is_dir() else clip
    paths = sorted(frame_dir.glob("*.png")) + sorted(frame_dir.glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no frames in {frame_dir}")
    return paths


def _prompt_metadata(clip: Path) -> tuple[dict[str, str], dict[str, int]]:
    path = clip / "prompts.json"
    if not path.is_file():
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    types: dict[str, str] = {}
    ids: dict[str, int] = {}
    for lane in payload.get("lanes", []):
        role = str(lane["role"])
        if "lane_type" in lane:
            types[role] = str(lane["lane_type"])
        if "lane_id" in lane:
            ids[role] = int(lane["lane_id"])
    return types, ids


def _load_intrinsics(clip: Path, config: dict | None) -> dict:
    path = clip / "intrinsics.json"
    file_values = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return {
        "fx": float(file_values.get("fx", dig(config, "camera", "fx", default=700.0))),
        "fy": float(file_values.get("fy", dig(config, "camera", "fy", default=700.0))),
        "cx": float(file_values.get("cx", dig(config, "camera", "cx", default=320.0))),
        "cy": float(file_values.get("cy", dig(config, "camera", "cy", default=180.0))),
        "camera_height_m": float(
            file_values.get("camera_height_m", dig(config, "camera", "height_m", default=1.5))
        ),
        "pitch_rad": float(file_values.get("pitch_rad", dig(config, "camera", "pitch_rad", default=0.0))),
    }

