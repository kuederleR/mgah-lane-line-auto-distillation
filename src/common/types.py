"""Pydantic schemas for distilled 3D lane labels.

Names requested by the scaffold (`LaneSpline3D`, `FrameAnnotation`) are
aliases of the concrete models below. Knot coordinates are meters in the
camera optical frame defined in ``src.common.geometry``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.common.geometry import LANE_TYPES, MAX_LANES, NUM_KNOTS


class SplineKnot3D(BaseModel):
    """One 3D control point in the camera optical frame, meters."""

    x: float
    y: float
    z: float

    def as_array(self) -> list[float]:
        return [self.x, self.y, self.z]


class Lane3D(BaseModel):
    """A single lane spline with a fixed knot count."""

    lane_id: int = Field(..., ge=0, lt=MAX_LANES)
    lane_type: str = "unknown"
    confidence: float = Field(..., ge=0.0, le=1.0)
    knots: list[SplineKnot3D]
    role: str = ""

    @field_validator("lane_type")
    @classmethod
    def _known_type(cls, value: str) -> str:
        if value not in LANE_TYPES:
            raise ValueError(f"lane_type must be one of {LANE_TYPES}, got {value!r}")
        return value

    @field_validator("knots")
    @classmethod
    def _knot_count(cls, value: list[SplineKnot3D]) -> list[SplineKnot3D]:
        if len(value) != NUM_KNOTS:
            raise ValueError(f"expected {NUM_KNOTS} knots, got {len(value)}")
        return value

    def knot_array(self) -> list[list[float]]:
        return [knot.as_array() for knot in self.knots]


class Frame3DAnnotation(BaseModel):
    """Teacher label for one front-camera frame."""

    frame_idx: int = Field(..., ge=0)
    timestamp_us: int = Field(..., ge=0)
    lanes: list[Lane3D]
    camera_k: list[list[float]]
    audit_score: float = Field(..., ge=0.0, le=1.0)
    image: str | None = None
    camera_height_m: float = 1.45
    pitch_rad: float = 0.0
    audit_issues: list[str] = Field(default_factory=list)

    @field_validator("camera_k")
    @classmethod
    def _intrinsics_shape(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 3 or any(len(row) != 3 for row in value):
            raise ValueError("camera_k must be a 3x3 matrix")
        return value


class DistillationDataset(BaseModel):
    """Index of distilled frames on disk.

    ``frames`` are JSON filenames relative to the directory that holds this
    index. ``image_root`` is where ``Frame3DAnnotation.image`` is resolved,
    relative to that same directory unless it is absolute.
    """

    name: str
    image_root: str
    frames: list[str]
    num_lanes: int = MAX_LANES
    num_knots: int = NUM_KNOTS
    coordinate_frame: str = "camera_optical"
    mock_teacher: bool = True
    notes: str = ""

    def frame_count(self) -> int:
        return len(self.frames)


# Names used in the original scaffold sketch.
LaneSpline3D = Lane3D
FrameAnnotation = Frame3DAnnotation


def knots_from_array(points: Any) -> list[SplineKnot3D]:
    """Build knots from an ``(K, 3)`` array. NaNs become a validation error upstream."""
    array = list(points)
    return [SplineKnot3D(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in array]
