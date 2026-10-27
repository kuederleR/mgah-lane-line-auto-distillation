"""Camera and ego geometry for monocular lane lifting.

This module is the coordinate-frame contract for the rest of the repo.

Camera optical frame (knots stored on labels and predicted by the student):
    +X right, +Y down, +Z forward along the optical axis.
    Origin at the camera center. Units are meters.

Ego / vehicle frame:
    +X right, +Y forward, +Z up.
    Origin on the ground plane directly under the camera.

The default mount places the camera ``camera_height_m`` above the ego origin.
``pitch_rad`` > 0 rotates the camera downward about its +X axis (right-hand
rule in the camera frame: positive pitch looks toward the ground).

Unprojection of a pixel ``(u, v)`` with metric depth ``Z`` and intrinsics
``fx, fy, cx, cy``:

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
"""

from __future__ import annotations

import numpy as np

MAX_LANES: int = 4
NUM_KNOTS: int = 10
LANE_OUTPUT_DIM: int = 1 + NUM_KNOTS * 3  # existence logit + XYZ knots = 31

LANE_ROLES: tuple[str, ...] = (
    "ego_left",
    "ego_right",
    "left_adjacent",
    "right_adjacent",
)
ROLE_TO_INDEX: dict[str, int] = {role: i for i, role in enumerate(LANE_ROLES)}

LANE_TYPES: tuple[str, ...] = (
    "dashed",
    "solid",
    "double_yellow",
    "road_edge",
    "unknown",
)


def role_index(role: str | None) -> int | None:
    """Return the fixed slot for a lane role, or None when the role is unknown."""
    if role is None:
        return None
    return ROLE_TO_INDEX.get(role)


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Build a 3×3 pinhole intrinsic matrix."""
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def unproject_pixels(uv: np.ndarray, z: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Unproject pixels to the camera optical frame.

    Args:
        uv: ``(N, 2)`` pixel coordinates ``(u, v)``.
        z: ``(N,)`` metric depth along the optical axis, in meters.
        k: ``(3, 3)`` intrinsics.

    Returns:
        ``(N, 3)`` points ``(X, Y, Z)`` in the camera frame.
    """
    uv = np.asarray(uv, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    k = np.asarray(k, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv must have shape (N, 2), got {uv.shape}")
    if uv.shape[0] != z.shape[0]:
        raise ValueError(f"uv and z length mismatch: {uv.shape[0]} vs {z.shape[0]}")
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    if fx == 0.0 or fy == 0.0:
        raise ValueError("focal length must be non-zero")
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    x = (uv[:, 0] - cx) * z / fx
    y = (uv[:, 1] - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def project_points(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Project camera-frame points to pixels. Z is clipped away from zero."""
    xyz = np.asarray(xyz, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    z = np.clip(xyz[:, 2], 1e-6, None)
    u = k[0, 0] * xyz[:, 0] / z + k[0, 2]
    v = k[1, 1] * xyz[:, 1] / z + k[1, 2]
    return np.stack([u, v], axis=-1)


def ego_to_camera(
    xyz_ego: np.ndarray,
    camera_height_m: float = 1.45,
    pitch_rad: float = 0.0,
) -> np.ndarray:
    """Map ego-frame points into the camera optical frame.

    See the module docstring for the mount convention.
    """
    xyz_ego = np.asarray(xyz_ego, dtype=np.float64)
    x = xyz_ego[:, 0]
    y_level = camera_height_m - xyz_ego[:, 2]
    z_level = xyz_ego[:, 1]
    cos_p = np.cos(pitch_rad)
    sin_p = np.sin(pitch_rad)
    # R_x(+pitch) takes the level camera (looking along ego +Y) into a
    # camera pitched down. A point on the level optical axis moves toward -Y
    # (up in the image) when the camera looks down.
    y_cam = cos_p * y_level - sin_p * z_level
    z_cam = sin_p * y_level + cos_p * z_level
    return np.stack([x, y_cam, z_cam], axis=-1)


def camera_to_ego(
    xyz_cam: np.ndarray,
    camera_height_m: float = 1.45,
    pitch_rad: float = 0.0,
) -> np.ndarray:
    """Map camera-frame points into the ego frame. Inverse of ``ego_to_camera``."""
    xyz_cam = np.asarray(xyz_cam, dtype=np.float64)
    x = xyz_cam[:, 0]
    y_cam = xyz_cam[:, 1]
    z_cam = xyz_cam[:, 2]
    cos_p = np.cos(pitch_rad)
    sin_p = np.sin(pitch_rad)
    y_level = cos_p * y_cam + sin_p * z_cam
    z_level = -sin_p * y_cam + cos_p * z_cam
    x_ego = x
    y_ego = z_level
    z_ego = camera_height_m - y_level
    return np.stack([x_ego, y_ego, z_ego], axis=-1)


def ground_plane_depth(
    height: int,
    width: int,
    k: np.ndarray,
    camera_height_m: float = 1.45,
    pitch_rad: float = 0.0,
    z_min: float = 0.5,
    z_max: float = 80.0,
) -> np.ndarray:
    """Per-pixel camera-frame Z where the ray hits the ground plane.

    Pixels that look above the ground, or that hit outside ``[z_min, z_max]``,
    are 0. Zero means "no metric depth", not "the camera origin".
    """
    k = np.asarray(k, dtype=np.float64)
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    us = np.arange(width, dtype=np.float64)
    vs = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    d_y = (vv - cy) / fy
    d_z = np.ones_like(d_y)
    cos_p = np.cos(pitch_rad)
    sin_p = np.sin(pitch_rad)
    # Ray direction transformed by R_x(-pitch) into the level camera.
    y_level_dir = cos_p * d_y + sin_p * d_z
    depth = np.zeros((height, width), dtype=np.float32)
    valid = y_level_dir > 1e-4
    z = np.zeros_like(y_level_dir)
    z[valid] = camera_height_m / y_level_dir[valid]
    ok = valid & (z >= z_min) & (z <= z_max)
    depth[ok] = z[ok].astype(np.float32)
    return depth


def resample_knots(
    xyz: np.ndarray,
    num_knots: int = NUM_KNOTS,
    z_min: float | None = None,
    z_max: float | None = None,
) -> np.ndarray:
    """Resample a 3D polyline to ``num_knots`` uniform in camera-frame Z.

    ``X`` and ``Y`` are linearly interpolated against sorted unique depths.
    Knots are NaN when the polyline has fewer than two distinct depths inside
    the requested window. NaNs are the signal the loss uses to drop a knot.
    """
    pts = np.asarray(xyz, dtype=np.float64)
    empty = np.full((num_knots, 3), np.nan, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        return empty
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) == 0:
        return empty
    pts = pts[np.argsort(pts[:, 2])]
    _, unique_idx = np.unique(np.round(pts[:, 2], decimals=4), return_index=True)
    pts = pts[np.sort(unique_idx)]
    if len(pts) < 2:
        return empty
    z_lo = float(pts[0, 2] if z_min is None else max(z_min, float(pts[0, 2])))
    z_hi = float(pts[-1, 2] if z_max is None else min(z_max, float(pts[-1, 2])))
    if z_hi <= z_lo:
        return empty
    # Interpolate inside the observed range only.
    inside = (pts[:, 2] >= z_lo - 1e-6) & (pts[:, 2] <= z_hi + 1e-6)
    pts = pts[inside]
    if len(pts) < 2:
        return empty
    z_query = np.linspace(z_lo, z_hi, num_knots)
    x_query = np.interp(z_query, pts[:, 2], pts[:, 0])
    y_query = np.interp(z_query, pts[:, 2], pts[:, 1])
    return np.stack([x_query, y_query, z_query], axis=-1)


def mask_centerline(mask: np.ndarray) -> np.ndarray:
    """Return an ``(N, 2)`` polyline of ``(u, v)`` along a binary mask.

    Lane markings in a front camera run toward the vanishing point, so the
    mask is usually taller than it is wide and the centerline is the per-row
    median column. Short, wide masks (a nearby stop bar, a bad crop) use the
    per-column median row instead.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be HxW, got {mask.shape}")
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 2:
        return np.zeros((0, 2), dtype=np.float64)
    height_span = int(ys.max() - ys.min())
    width_span = int(xs.max() - xs.min())
    points: list[tuple[float, float]] = []
    if height_span >= width_span:
        for v in range(int(ys.min()), int(ys.max()) + 1):
            cols = xs[ys == v]
            if len(cols) == 0:
                continue
            points.append((float(np.median(cols)), float(v)))
    else:
        for u in range(int(xs.min()), int(xs.max()) + 1):
            rows = ys[xs == u]
            if len(rows) == 0:
                continue
            points.append((float(u), float(np.median(rows))))
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def sample_depth(depth: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest-neighbor sample of an ``(H, W)`` depth map at ``(u, v)`` pixels."""
    depth = np.asarray(depth)
    uv = np.asarray(uv, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")
    height, width = depth.shape
    u = np.clip(np.rint(uv[:, 0]), 0, width - 1).astype(np.int32)
    v = np.clip(np.rint(uv[:, 1]), 0, height - 1).astype(np.int32)
    return depth[v, u].astype(np.float64)


def pack_lane_array(existence: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Pack existence ``(M,)`` and knots ``(M, K, 3)`` into ``(M, 31)``."""
    existence = np.asarray(existence, dtype=np.float32).reshape(-1)
    knots = np.asarray(knots, dtype=np.float32)
    if knots.ndim != 3 or knots.shape[0] != existence.shape[0] or knots.shape[2] != 3:
        raise ValueError(
            f"expected existence (M,) and knots (M, K, 3), got "
            f"{existence.shape} and {knots.shape}"
        )
    if knots.shape[1] != NUM_KNOTS:
        raise ValueError(f"expected {NUM_KNOTS} knots, got {knots.shape[1]}")
    packed = np.zeros((knots.shape[0], LANE_OUTPUT_DIM), dtype=np.float32)
    packed[:, 0] = existence
    packed[:, 1:] = knots.reshape(knots.shape[0], -1)
    return packed


def split_lane_array(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``(..., 31)`` into existence ``(...,)`` and knots ``(..., K, 3)``."""
    packed = np.asarray(packed, dtype=np.float32)
    if packed.shape[-1] != LANE_OUTPUT_DIM:
        raise ValueError(f"expected last dim {LANE_OUTPUT_DIM}, got {packed.shape[-1]}")
    existence = packed[..., 0]
    knots = packed[..., 1:].reshape(*packed.shape[:-1], NUM_KNOTS, 3)
    return existence, knots
