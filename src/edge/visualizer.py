"""Headless dual view: camera overlay plus an ego-frame bird's-eye grid.

The function returns a BGR image and never opens a window. The BEV panel is
the ego frame (+X right, +Y forward). Lane knots are stored in the camera
frame and converted with the mount height and pitch on the annotation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.common.geometry import LANE_ROLES, camera_to_ego, project_points
from src.common.types import Lane3D

ROLE_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "ego_left": (80, 220, 80),
    "ego_right": (80, 180, 255),
    "left_adjacent": (0, 220, 255),
    "right_adjacent": (255, 160, 60),
}


def render_dual_view(
    image_bgr: np.ndarray,
    lanes: list[Lane3D],
    camera_k: np.ndarray,
    camera_height_m: float = 1.45,
    pitch_rad: float = 0.0,
    forward_m: float = 50.0,
    lateral_m: float = 15.0,
) -> np.ndarray:
    """Return ``(H, W + H, 3)`` BGR: perspective overlay beside a square BEV."""
    overlay = draw_perspective(image_bgr, lanes, camera_k)
    bev = render_bev(
        lanes,
        size=overlay.shape[0],
        forward_m=forward_m,
        lateral_m=lateral_m,
        camera_height_m=camera_height_m,
        pitch_rad=pitch_rad,
    )
    return np.concatenate([overlay, bev], axis=1)


def draw_perspective(image_bgr: np.ndarray, lanes: list[Lane3D], camera_k: np.ndarray) -> np.ndarray:
    import cv2

    canvas = np.array(image_bgr, copy=True)
    k = np.asarray(camera_k, dtype=np.float64)
    for lane in lanes:
        knots = np.asarray(lane.knot_array(), dtype=np.float64)
        if knots.shape[0] < 2 or not np.isfinite(knots).all() or np.all(knots[:, 2] <= 0):
            continue
        pixels = project_points(knots, k)
        pts = np.round(pixels).astype(np.int32).reshape(-1, 1, 2)
        color = ROLE_COLORS_BGR.get(lane.role, (255, 255, 255))
        cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_8)
    return canvas


def render_bev(
    lanes: list[Lane3D],
    size: int = 360,
    forward_m: float = 50.0,
    lateral_m: float = 15.0,
    camera_height_m: float = 1.45,
    pitch_rad: float = 0.0,
) -> np.ndarray:
    """Square BEV in the ego frame. Forward is up. The grid step is 5 m."""
    import cv2

    canvas = np.full((size, size, 3), 24, dtype=np.uint8)
    for meters in range(0, int(forward_m) + 1, 5):
        x0, y0 = _bev_px(-lateral_m, meters, size, forward_m, lateral_m)
        x1, y1 = _bev_px(lateral_m, meters, size, forward_m, lateral_m)
        cv2.line(canvas, (x0, y0), (x1, y1), (48, 48, 48), 1, cv2.LINE_8)
    for meters in range(-int(lateral_m), int(lateral_m) + 1, 5):
        x0, y0 = _bev_px(meters, 0.0, size, forward_m, lateral_m)
        x1, y1 = _bev_px(meters, forward_m, size, forward_m, lateral_m)
        cv2.line(canvas, (x0, y0), (x1, y1), (48, 48, 48), 1, cv2.LINE_8)
    origin = _bev_px(0.0, 0.0, size, forward_m, lateral_m)
    cv2.circle(canvas, origin, 3, (220, 220, 220), -1, cv2.LINE_8)
    for lane in lanes:
        knots = np.asarray(lane.knot_array(), dtype=np.float64)
        if not np.isfinite(knots).all():
            continue
        ego = camera_to_ego(knots, camera_height_m=camera_height_m, pitch_rad=pitch_rad)
        pts = np.array(
            [_bev_px(float(p[0]), float(p[1]), size, forward_m, lateral_m) for p in ego],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        color = ROLE_COLORS_BGR.get(lane.role, (255, 255, 255))
        cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_8)
    cv2.putText(
        canvas,
        "BEV ego +X right +Y forward",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (200, 200, 200),
        1,
        cv2.LINE_8,
    )
    return canvas


def save_spline_ply(
    path: str | Path,
    lanes: list[Lane3D],
    camera_height_m: float = 1.45,
    pitch_rad: float = 0.0,
) -> None:
    """Write ego-frame knots as an ASCII PLY. No Open3D dependency."""
    points: list[tuple[float, float, float]] = []
    for lane in lanes:
        knots = np.asarray(lane.knot_array(), dtype=np.float64)
        if not np.isfinite(knots).all():
            continue
        ego = camera_to_ego(knots, camera_height_m=camera_height_m, pitch_rad=pitch_rad)
        points.extend((float(p[0]), float(p[1]), float(p[2])) for p in ego)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    body = "".join(f"{x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in points)
    Path(path).write_text(header + body, encoding="utf-8")


def _bev_px(
    x_ego: float,
    y_ego: float,
    size: int,
    forward_m: float,
    lateral_m: float,
) -> tuple[int, int]:
    px = size / 2.0 + (x_ego / lateral_m) * (size / 2.0)
    py = size * 0.96 - (y_ego / forward_m) * (size * 0.90)
    return int(round(px)), int(round(py))


__all__ = [
    "LANE_ROLES",
    "ROLE_COLORS_BGR",
    "draw_perspective",
    "render_bev",
    "render_dual_view",
    "save_spline_ply",
]
