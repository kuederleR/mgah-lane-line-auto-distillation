"""Masked Smooth L1 on knots plus existence BCE (or focal).

Targets are packed ``[B, 4, 31]`` with existence in ``{0, 1}`` on channel 0.
Predictions use a logit on that channel. Knots of absent lanes are ignored,
including knots that are NaN in the target.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.student.model import split_lane_tensor


class LaneSplineLoss(nn.Module):
    def __init__(
        self,
        knot_weight: float = 1.0,
        exist_weight: float = 1.0,
        distill_weight: float = 0.0,
        focal: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ) -> None:
        super().__init__()
        self.knot_weight = knot_weight
        self.exist_weight = exist_weight
        self.distill_weight = distill_weight
        self.focal = focal
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def forward(
        self,
        outputs: dict[str, torch.Tensor] | torch.Tensor,
        target: torch.Tensor,
        teacher_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        packed = outputs["lanes"] if isinstance(outputs, dict) else outputs
        pred_exist, pred_knots = split_lane_tensor(packed)
        tgt_exist, tgt_knots = split_lane_tensor(target)
        if self.focal:
            exist_loss = _focal_bce_with_logits(
                pred_exist, tgt_exist, self.focal_gamma, self.focal_alpha
            )
        else:
            exist_loss = F.binary_cross_entropy_with_logits(pred_exist, tgt_exist)

        valid = torch.isfinite(tgt_knots).all(dim=-1)
        safe_target = torch.nan_to_num(tgt_knots, nan=0.0)
        mask = tgt_exist.unsqueeze(-1) * valid.float()
        per_coord = F.smooth_l1_loss(pred_knots, safe_target, reduction="none")
        numer = (per_coord * mask.unsqueeze(-1)).sum()
        denom = (mask.sum() * pred_knots.shape[-1]).clamp(min=1.0)
        knot_loss = numer / denom

        total = self.knot_weight * knot_loss + self.exist_weight * exist_loss
        distill = packed.new_zeros(())
        student_features = outputs.get("distill_features") if isinstance(outputs, dict) else None
        if self.distill_weight > 0.0 and student_features is not None and teacher_features is not None:
            teacher = teacher_features
            if teacher.shape[-2:] != student_features.shape[-2:]:
                teacher = F.interpolate(
                    teacher,
                    size=student_features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if teacher.shape[1] != student_features.shape[1]:
                raise ValueError(
                    f"distill channels differ: student {student_features.shape[1]} "
                    f"vs teacher {teacher.shape[1]}"
                )
            distill = F.mse_loss(student_features, teacher)
            total = total + self.distill_weight * distill
        return {
            "loss": total,
            "knot": knot_loss,
            "existence": exist_loss,
            "distill": distill,
        }


def _focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()
