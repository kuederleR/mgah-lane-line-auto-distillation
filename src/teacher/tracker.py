"""Prompt SAM on frame 0 and propagate lane masks through the clip.

The live path posts the payload from ``MetaModelClient`` and decodes
``mask_png_b64`` objects. The mock path never posts. It segments each frame
with a lane-color tracker and builds the frame-0 point prompts SAM would
have received, so the request structure stays exercised offline.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.common.geometry import LANE_ROLES, mask_centerline
from src.teacher.meta_client import MetaModelClient

logger = logging.getLogger(__name__)


@dataclass
class FrameMasks:
    """Role → binary ``uint8`` mask for one frame."""

    frame_idx: int
    masks: dict[str, np.ndarray]


@dataclass
class TrackingResult:
    frames: list[FrameMasks]
    prompts: list[dict[str, Any]]
    request: dict[str, Any]
    source: str
    notes: list[str] = field(default_factory=list)


class SamVideoTracker:
    """Temporal lane-instance tracker."""

    def __init__(self, client: MetaModelClient | None = None, min_area: int = 30) -> None:
        self.client = client if client is not None else MetaModelClient(mock=True)
        self.min_area = min_area

    def track(
        self,
        frames_bgr: list[np.ndarray],
        prompts: list[dict[str, Any]] | None = None,
    ) -> TrackingResult:
        if not frames_bgr:
            raise ValueError("track() needs at least one frame")
        if self.client.mock or prompts is None:
            return self._track_mock(frames_bgr, prompts or [])
        return self._track_live(frames_bgr, prompts)

    def _track_mock(
        self,
        frames_bgr: list[np.ndarray],
        prompts: list[dict[str, Any]],
    ) -> TrackingResult:
        per_frame = [segment_lane_masks(frame, min_area=self.min_area) for frame in frames_bgr]
        if not prompts:
            prompts = prompts_from_masks(0, per_frame[0])
        request = self.client.build_sam_request(frames_bgr, prompts, propagate=True, include_image_bytes=False)
        # Touch the mock transport so a regression that starts calling HTTP is visible.
        echo = self.client.segment_video(frames_bgr, prompts, propagate=True)
        notes = []
        if not per_frame[0]:
            notes.append("mock tracker found no lane-colored components on frame 0")
        logger.info("mock SAM propagation over %d frame(s), source=%s", len(frames_bgr), echo.get("model"))
        return TrackingResult(
            frames=[FrameMasks(frame_idx=i, masks=masks) for i, masks in enumerate(per_frame)],
            prompts=prompts,
            request=request,
            source="mock-color",
            notes=notes,
        )

    def _track_live(
        self,
        frames_bgr: list[np.ndarray],
        prompts: list[dict[str, Any]],
    ) -> TrackingResult:
        response = self.client.segment_video(frames_bgr, prompts, propagate=True)
        request = self.client.build_sam_request(frames_bgr, prompts, propagate=True, include_image_bytes=False)
        height, width = frames_bgr[0].shape[:2]
        decoded = decode_sam_response(response, height=height, width=width)
        return TrackingResult(
            frames=decoded,
            prompts=prompts,
            request=request,
            source="sam-3.1-video",
        )


def segment_lane_masks(frame_bgr: np.ndarray, min_area: int = 30) -> dict[str, np.ndarray]:
    """Link bright white and yellow paint into role-ordered masks.

    Scanning from the bottom of the frame upward keeps each marking on its
    own track: near the camera the lanes are far apart, and a per-row
    velocity bridges the gaps in a dashed line. Tracks are ordered by their
    nearest column, which is left-to-right in the image, then mapped to roles.
    """
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR frame, got {frame.shape}")
    rgb = frame[:, :, ::-1]
    white = (rgb[:, :, 0] > 210) & (rgb[:, :, 1] > 210) & (rgb[:, :, 2] > 210)
    yellow = (rgb[:, :, 0] > 200) & (rgb[:, :, 1] > 160) & (rgb[:, :, 2] < 140)
    binary = white | yellow
    height, width = binary.shape
    tracks: list[dict] = []
    for row in range(height - 1, int(height * 0.35), -1):
        runs = _row_runs(binary[row])
        if not runs:
            continue
        used: set[int] = set()
        for track in tracks:
            gap = track["last_v"] - row
            if gap <= 0 or gap > 30:
                continue
            predicted = track["last_u"] + track["velocity"] * gap
            best_index = None
            best_distance = 24.0
            for index, run in enumerate(runs):
                if index in used:
                    continue
                distance = abs(run[0] - predicted)
                if distance < best_distance:
                    best_index = index
                    best_distance = distance
            if best_index is None:
                continue
            center, start, end = runs[best_index]
            used.add(best_index)
            track["rows"].append((row, start, end, center))
            track["velocity"] = (center - track["last_u"]) / gap
            track["last_u"] = center
            track["last_v"] = row
        for index, (center, start, end) in enumerate(runs):
            if index in used:
                continue
            tracks.append(
                {
                    "rows": [(row, start, end, center)],
                    "last_u": center,
                    "last_v": row,
                    "velocity": 0.0,
                }
            )

    components: list[tuple[float, np.ndarray]] = []
    for track in tracks:
        if len(track["rows"]) < 12:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        for row, start, end, _center in track["rows"]:
            mask[row, start : end + 1] = 1
        if int(mask.sum()) < min_area:
            continue
        nearest = sorted(track["rows"], key=lambda item: item[0], reverse=True)[:12]
        order_u = float(np.median([item[3] for item in nearest]))
        components.append((order_u, mask))
    components.sort(key=lambda item: item[0])
    roles = _roles_for_count(len(components))
    return {role: mask for role, (_column, mask) in zip(roles, components)}


def _row_runs(row: np.ndarray) -> list[tuple[float, int, int]]:
    """Return ``(center, start, end)`` for contiguous True pixels, bridging 1 px gaps."""
    columns = np.flatnonzero(row)
    if len(columns) == 0:
        return []
    runs: list[tuple[float, int, int]] = []
    start = previous = int(columns[0])
    for column in columns[1:]:
        column = int(column)
        if column <= previous + 2:
            previous = column
            continue
        runs.append(((start + previous) / 2.0, start, previous))
        start = previous = column
    runs.append(((start + previous) / 2.0, start, previous))
    return runs


def prompts_from_masks(frame_idx: int, masks: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Three positive points (near, mid, far) per mask, in slot order."""
    prompts = []
    for object_id, role in enumerate(LANE_ROLES):
        mask = masks.get(role)
        if mask is None:
            continue
        center = mask_centerline(mask)
        if len(center) == 0:
            continue
        picks = _sample_rows(center, 3)
        prompts.append(
            {
                "frame_idx": frame_idx,
                "object_id": object_id,
                "role": role,
                "points": [[float(p[0]), float(p[1])] for p in picks],
                "point_labels": [1] * len(picks),
            }
        )
    return prompts


def decode_sam_response(response: dict[str, Any], height: int, width: int) -> list[FrameMasks]:
    """Decode ``mask_png_b64`` objects from a live SAM response."""
    import cv2

    frames_out: list[FrameMasks] = []
    for frame in response.get("frames", []):
        masks: dict[str, np.ndarray] = {}
        for obj in frame.get("objects", []):
            role = str(obj.get("role") or LANE_ROLES[int(obj.get("object_id", 0)) % len(LANE_ROLES)])
            encoded = obj.get("mask_png_b64")
            if not encoded:
                continue
            raw = base64.b64decode(encoded)
            decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if decoded is None:
                continue
            if decoded.shape != (height, width):
                decoded = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_NEAREST)
            masks[role] = (decoded > 0).astype(np.uint8)
        frames_out.append(FrameMasks(frame_idx=int(frame.get("frame_idx", len(frames_out))), masks=masks))
    return frames_out


# Image left → right. This is not the network slot order in ``LANE_ROLES``.
_SPATIAL_ROLES: tuple[str, ...] = (
    "left_adjacent",
    "ego_left",
    "ego_right",
    "right_adjacent",
)


def _roles_for_count(count: int) -> list[str]:
    """Map a left-to-right component list onto lane roles.

    Two components are the ego pair. Three keep the inner pair as ego and
    attach the outer component as the left adjacent lane. Four use spatial
    order, which is not the same as the ``[ego_left, ego_right, ...]`` slot
    order the student head uses.
    """
    if count <= 0:
        return []
    if count == 1:
        return ["ego_left"]
    if count == 2:
        return ["ego_left", "ego_right"]
    if count == 3:
        return ["left_adjacent", "ego_left", "ego_right"]
    return list(_SPATIAL_ROLES[: min(count, len(_SPATIAL_ROLES))])


def _sample_rows(centerline: np.ndarray, count: int) -> np.ndarray:
    if len(centerline) <= count:
        return centerline
    indices = np.linspace(0, len(centerline) - 1, count).astype(int)
    return centerline[indices]
