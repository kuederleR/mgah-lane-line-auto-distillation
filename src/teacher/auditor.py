"""Geometric quality gate and Llama 3.2 Vision auditor.

``VLMGeometricAuditor.audit`` always computes a deterministic score in
``[0, 1]`` from the splines (ground-plane residual, depth order, lateral
kinks, ego ordering). That score is the result while the Meta client is
mocked. A live client replaces it when the model returns a parseable JSON
score; a bad response falls back to the geometric score and records why.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

import numpy as np

from src.common.geometry import LANE_TYPES
from src.common.types import Frame3DAnnotation, Lane3D
from src.teacher.meta_client import MetaModelClient

logger = logging.getLogger(__name__)


@dataclass
class AuditReport:
    score: float
    issues: list[str] = field(default_factory=list)
    lane_types: dict[str, str] = field(default_factory=dict)
    prompt: str = ""
    source: str = "geometric"
    raw: dict = field(default_factory=dict)

    def clamped(self) -> float:
        return float(np.clip(self.score, 0.0, 1.0))


class VLMGeometricAuditor:
    """Grade reprojected 3D splines and optionally ask Llama 3.2 Vision."""

    def __init__(self, client: MetaModelClient | None = None) -> None:
        self.client = client if client is not None else MetaModelClient(mock=True)

    def audit(
        self,
        annotation: Frame3DAnnotation,
        image_bgr: np.ndarray | None = None,
        masks: dict[str, np.ndarray] | None = None,
    ) -> AuditReport:
        score, issues = geometric_audit(annotation.lanes)
        lane_types = {}
        if image_bgr is not None and masks:
            for role, mask in masks.items():
                lane_types[role] = classify_marking(image_bgr, mask)
        prompt = build_audit_prompt(annotation, issues)
        if self.client.mock:
            return AuditReport(
                score=score,
                issues=issues,
                lane_types=lane_types,
                prompt=prompt,
                source="geometric-mock",
                raw={"mock": True, "score": score, "issues": issues, "lane_types": lane_types},
            )
        try:
            payload = self.client.audit_vision(prompt, image_bgr)
        except Exception as exc:
            logger.warning("Llama vision audit failed (%s); using the geometric score", exc)
            issues = issues + [f"vlm_error: {exc}"]
            return AuditReport(
                score=score,
                issues=issues,
                lane_types=lane_types,
                prompt=prompt,
                source="geometric-fallback",
                raw={"error": str(exc)},
            )
        parsed = parse_vlm_payload(payload)
        if parsed is None or "score" not in parsed:
            issues = issues + ["vlm_unparsed"]
            return AuditReport(
                score=score,
                issues=issues,
                lane_types=lane_types,
                prompt=prompt,
                source="geometric-fallback",
                raw=payload,
            )
        vlm_score = float(np.clip(float(parsed["score"]), 0.0, 1.0))
        vlm_issues = [str(item) for item in parsed.get("issues", [])]
        parsed_types = {}
        for role, lane_type in dict(parsed.get("lane_types", {})).items():
            if lane_type in LANE_TYPES:
                parsed_types[str(role)] = str(lane_type)
        return AuditReport(
            score=vlm_score,
            issues=vlm_issues or issues,
            lane_types=parsed_types or lane_types,
            prompt=prompt,
            source="llama-3.2-vision",
            raw=payload,
        )


def geometric_audit(lanes: list[Lane3D]) -> tuple[float, list[str]]:
    """Score splines in ``[0, 1]``. Higher is more consistent with a road plane."""
    if not lanes:
        return 0.0, ["no_lanes"]
    issues: list[str] = []
    scores: list[float] = []
    for lane in lanes:
        points = np.asarray(lane.knot_array(), dtype=np.float64)
        score = 1.0
        label = lane.role or f"id{lane.lane_id}"
        if not np.isfinite(points).all():
            issues.append(f"{label} non-finite knots")
            scores.append(0.0)
            continue
        if np.any(points[:, 2] <= 0.0):
            score -= 0.4
            issues.append(f"{label} non-positive depth")
        dz = np.diff(points[:, 2])
        if len(dz) and np.any(dz <= 0.05):
            score -= 0.35
            issues.append(f"{label} depth is not increasing")
        residual = _line_rmse(points[:, 2], points[:, 1])
        if residual > 0.35:
            # A meter-scale departure from the ground plane should fail the frame,
            # not get averaged away by the other lanes.
            score -= min(0.95, residual)
            issues.append(f"{label} ground residual {residual:.2f}m")
        if len(points) >= 3:
            kink = float(np.max(np.abs(np.diff(points[:, 0], n=2))))
            if kink > 1.5:
                score -= 0.3
                issues.append(f"{label} lateral kink {kink:.2f}m")
        if len(dz) and float(np.median(np.abs(dz))) > 1e-6:
            if float(np.max(np.abs(dz))) > 5.0 * float(np.median(np.abs(dz))):
                score -= 0.25
                issues.append(f"{label} depth spike")
        scores.append(float(np.clip(score, 0.0, 1.0)))

    by_role = {lane.role: lane for lane in lanes if lane.role}
    order_cap = 1.0
    if "ego_left" in by_role and "ego_right" in by_role:
        left_x = float(np.mean(by_role["ego_left"].knot_array(), axis=0)[0])
        right_x = float(np.mean(by_role["ego_right"].knot_array(), axis=0)[0])
        if left_x >= right_x:
            issues.append("ego_left is not left of ego_right")
            order_cap = 0.2
    # The frame is only as trustworthy as its worst lane.
    return float(np.clip(min(scores) * order_cap, 0.0, 1.0)), issues


def classify_marking(frame_bgr: np.ndarray, mask: np.ndarray) -> str:
    """Offline stand-in for the VLM lane-type head.

    Yellow paint maps to ``double_yellow`` (a single yellow line is not
    separable by color; the live prompt asks Llama to make that call).
    Rows with no mask pixels inside the vertical span mark a dashed line.
    """
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 10:
        return "unknown"
    colors = frame_bgr[ys, xs]
    blue, green, red = colors[:, 0], colors[:, 1], colors[:, 2]
    yellow_frac = float(np.mean((red > 180) & (green > 150) & (blue < 140)))
    if yellow_frac > 0.4:
        return "double_yellow"
    rows = np.arange(int(ys.min()), int(ys.max()) + 1)
    present = np.array([np.any(ys == row) for row in rows])
    gap_frac = 1.0 - float(present.mean())
    if gap_frac > 0.15:
        return "dashed"
    return "solid"


def build_audit_prompt(annotation: Frame3DAnnotation, geometric_issues: list[str]) -> str:
    """Structured prompt. The model must return JSON and nothing else."""
    lanes = []
    for lane in annotation.lanes:
        lanes.append(
            {
                "role": lane.role,
                "lane_id": lane.lane_id,
                "lane_type": lane.lane_type,
                "knots_xyz_m": [[round(v, 3) for v in knot] for knot in lane.knot_array()],
            }
        )
    body = {
        "task": (
            "Grade these camera-frame lane splines reprojected onto the attached "
            "front-camera image. +X is right, +Y is down, +Z is forward, meters."
        ),
        "checks": [
            "ego_left mean X is less than ego_right mean X",
            "knots lie on a ground plane (Y is nearly linear in Z)",
            "no impossible Z spikes or folds",
            "lane paint matches dashed, solid, or double_yellow",
            "occluded or artifact fragments should lower the score",
        ],
        "lanes": lanes,
        "geometric_precheck": geometric_issues,
        "response_schema": {
            "score": "float in [0, 1]",
            "issues": ["short strings"],
            "lane_types": {"ego_left": "dashed|solid|double_yellow|road_edge|unknown"},
        },
    }
    return json.dumps(body, indent=2)


def parse_vlm_payload(payload: dict) -> dict | None:
    """Pull a JSON object out of an OpenAI-style chat response or a bare score."""
    if "score" in payload and not payload.get("choices"):
        return payload
    choices = payload.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        texts = [part.get("text", "") for part in content if isinstance(part, dict)]
        content = "\n".join(texts)
    if not isinstance(content, str):
        return None
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _line_rmse(z: np.ndarray, y: np.ndarray) -> float:
    if len(z) < 2 or float(np.ptp(z)) < 1e-6:
        return 0.0
    slope, intercept = np.polyfit(z, y, 1)
    residual = y - (slope * z + intercept)
    return float(np.sqrt(np.mean(residual**2)))
