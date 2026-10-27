"""Synthetic front-camera clip with known lane geometry.

The paint is saturated white or yellow on a dark road so the offline color
tracker can lock instances without SAM. Ego-left is dashed; the other three
markings are solid. Intrinsics match ``configs/teacher_config.yaml``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.common.geometry import ego_to_camera, intrinsics_matrix, project_points

LANE_LAYOUT: tuple[dict, ...] = (
    {
        "role": "left_adjacent",
        "lane_id": 2,
        "lane_type": "double_yellow",
        "x_m": -5.25,
        "color_bgr": (40, 220, 240),
        "dashed": False,
    },
    {
        "role": "ego_left",
        "lane_id": 0,
        "lane_type": "dashed",
        "x_m": -1.75,
        "color_bgr": (255, 255, 255),
        "dashed": True,
    },
    {
        "role": "ego_right",
        "lane_id": 1,
        "lane_type": "solid",
        "x_m": 1.75,
        "color_bgr": (255, 255, 255),
        "dashed": False,
    },
    {
        "role": "right_adjacent",
        "lane_id": 3,
        "lane_type": "solid",
        "x_m": 5.25,
        "color_bgr": (255, 255, 255),
        "dashed": False,
    },
)


def default_intrinsics(width: int = 640, height: int = 360) -> dict:
    return {
        "fx": 700.0,
        "fy": 700.0,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "width": width,
        "height": height,
        "camera_height_m": 1.5,
        "pitch_rad": 0.0,
    }


def generate_synthetic_clip(
    output_dir: str | Path,
    num_frames: int = 4,
    width: int = 640,
    height: int = 360,
) -> Path:
    """Write ``frames/*.png``, ``intrinsics.json``, and ``prompts.json``."""
    import cv2

    output = Path(output_dir)
    frame_dir = output / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    intrinsics = default_intrinsics(width, height)
    camera_k = intrinsics_matrix(
        intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    )
    for index in range(num_frames):
        image = _render_frame(
            camera_k,
            width,
            height,
            camera_height_m=intrinsics["camera_height_m"],
            pitch_rad=intrinsics["pitch_rad"],
        )
        cv2.imwrite(str(frame_dir / f"{index:06d}.png"), image)
    (output / "intrinsics.json").write_text(json.dumps(intrinsics, indent=2), encoding="utf-8")
    prompts = {
        "lanes": [
            {
                "role": lane["role"],
                "lane_id": lane["lane_id"],
                "lane_type": lane["lane_type"],
                "x_m": lane["x_m"],
            }
            for lane in LANE_LAYOUT
        ]
    }
    (output / "prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    return output


def _render_frame(
    camera_k: np.ndarray,
    width: int,
    height: int,
    camera_height_m: float,
    pitch_rad: float,
) -> np.ndarray:
    import cv2

    image = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = int(round(camera_k[1, 2]))
    image[:horizon, :] = (180, 120, 70)
    image[horizon:, :] = (80, 80, 80)
    for lane in LANE_LAYOUT:
        polyline = _lane_pixels(
            lane["x_m"],
            camera_k,
            width,
            height,
            camera_height_m,
            pitch_rad,
        )
        if len(polyline) < 2:
            continue
        _paint(image, polyline, lane["color_bgr"], dashed=lane["dashed"])
    return image


def _lane_pixels(
    x_m: float,
    camera_k: np.ndarray,
    width: int,
    height: int,
    camera_height_m: float,
    pitch_rad: float,
) -> np.ndarray:
    z_ego = np.linspace(8.0, 45.0, 90)
    ego = np.stack(
        [np.full_like(z_ego, x_m), z_ego, np.zeros_like(z_ego)],
        axis=1,
    )
    camera = ego_to_camera(ego, camera_height_m=camera_height_m, pitch_rad=pitch_rad)
    pixels = project_points(camera, camera_k)
    inside = (
        (pixels[:, 0] >= 1)
        & (pixels[:, 0] < width - 1)
        & (pixels[:, 1] >= 1)
        & (pixels[:, 1] < height - 1)
    )
    return pixels[inside]


def _paint(image: np.ndarray, polyline: np.ndarray, color_bgr: tuple[int, int, int], dashed: bool) -> None:
    import cv2

    points = np.round(polyline).astype(np.int32)
    if not dashed:
        cv2.polylines(image, [points.reshape(-1, 1, 2)], False, color_bgr, 3, cv2.LINE_8)
        return
    period = 16.0
    duty = 8.0
    traveled = 0.0
    for start, end in zip(points[:-1], points[1:]):
        length = float(np.linalg.norm(end.astype(np.float64) - start.astype(np.float64)))
        steps = max(int(length), 1)
        for step in range(steps):
            alpha = step / steps
            point = start.astype(np.float64) * (1.0 - alpha) + end.astype(np.float64) * alpha
            if (traveled % period) < duty:
                cv2.circle(image, (int(round(point[0])), int(round(point[1]))), 2, color_bgr, -1, cv2.LINE_8)
            traveled += length / steps
