"""Lift a 2D lane mask and a depth map into K camera-frame knots.

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

The centerline is the mask skeleton (per-row median). Depth is sampled at
those pixels, unprojected, and resampled to ``K`` knots uniform in ``Z``.
"""

from __future__ import annotations

import numpy as np

from src.common.geometry import (
    NUM_KNOTS,
    mask_centerline,
    resample_knots,
    sample_depth,
    unproject_pixels,
)
from src.common.types import Lane3D, SplineKnot3D


class LaneLifter:
    """Combine a SAM mask, a depth map, and intrinsics."""

    def __init__(
        self,
        num_knots: int = NUM_KNOTS,
        z_min: float = 2.0,
        z_max: float = 60.0,
        min_points: int = 8,
        max_abs_x_m: float = 20.0,
        y_min_m: float = -1.0,
        y_max_m: float = 8.0,
    ) -> None:
        self.num_knots = num_knots
        self.z_min = z_min
        self.z_max = z_max
        self.min_points = min_points
        self.max_abs_x_m = max_abs_x_m
        self.y_min_m = y_min_m
        self.y_max_m = y_max_m

    def lift_mask(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        camera_k: np.ndarray,
        lane_id: int,
        lane_type: str = "unknown",
        role: str = "",
    ) -> Lane3D | None:
        """Return a lane, or None when the mask does not support K knots."""
        uv = mask_centerline(mask)
        if len(uv) < self.min_points:
            return None
        sampled = sample_depth(depth, uv)
        xyz = unproject_pixels(uv, sampled, camera_k)
        keep = (
            np.isfinite(xyz).all(axis=1)
            & (xyz[:, 2] >= self.z_min)
            & (xyz[:, 2] <= self.z_max)
            & (np.abs(xyz[:, 0]) <= self.max_abs_x_m)
            & (xyz[:, 1] >= self.y_min_m)
            & (xyz[:, 1] <= self.y_max_m)
        )
        xyz = xyz[keep]
        support = float(keep.mean()) if len(keep) else 0.0
        if len(xyz) < self.min_points:
            return None
        knots = resample_knots(xyz, num_knots=self.num_knots, z_min=self.z_min, z_max=self.z_max)
        if not np.isfinite(knots).all():
            return None
        return Lane3D(
            lane_id=lane_id,
            lane_type=lane_type if lane_type else "unknown",
            confidence=float(np.clip(support, 0.0, 1.0)),
            knots=[SplineKnot3D(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in knots],
            role=role,
        )
