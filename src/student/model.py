"""Lightweight monocular 3D lane student.

Output tensor ``lanes`` has shape ``[B, 4, 31]``:
existence logit, then 10 camera-frame knots ``(X, Y, Z)``.

Input is RGB float ``NCHW`` in ``[0, 1]``. ImageNet normalization is applied
inside ``forward`` so ONNX and TensorRT consumers share one contract.

``distill_features`` is a 1×1 projection of the last backbone map, used to
match DINOv3 patch tokens when ``distill_dim > 0``.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from src.common.geometry import LANE_OUTPUT_DIM, LANE_ROLES, MAX_LANES, NUM_KNOTS
from src.common.types import Lane3D, SplineKnot3D

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class _TinyEncoder(nn.Module):
    """Three strided convolutions for tests and CPU smoke runs."""

    def __init__(self) -> None:
        super().__init__()
        self.out_channels = 32
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class _TimmEncoder(nn.Module):
    def __init__(self, model: nn.Module, channels: int) -> None:
        super().__init__()
        self.model = model
        self.out_channels = channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        if isinstance(features, (list, tuple)):
            return features[-1]
        return features


class LaneSplineStudent(nn.Module):
    """ResNet18 / MobileNetV4 / tiny backbone plus a spline head."""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,
        num_lanes: int = MAX_LANES,
        num_knots: int = NUM_KNOTS,
        distill_dim: int = 0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if num_lanes != MAX_LANES or num_knots != NUM_KNOTS:
            raise ValueError(
                f"this scaffold is fixed at {MAX_LANES} lanes and {NUM_KNOTS} knots"
            )
        encoder, channels, resolved = _build_encoder(backbone, pretrained)
        self.encoder = encoder
        self.backbone_name = resolved
        self.feature_channels = channels
        self.num_lanes = num_lanes
        self.num_knots = num_knots
        self.normalize = normalize
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, num_lanes * LANE_OUTPUT_DIM)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.distill_head = nn.Conv2d(channels, distill_dim, kernel_size=1) if distill_dim > 0 else None
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean)
        self.register_buffer("imagenet_std", std)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        # No Python-side shape branch: ONNX tracing would bake it in.
        x = images
        if self.normalize:
            x = (x - self.imagenet_mean) / self.imagenet_std
        features = self.encoder(x)
        pooled = self.pool(features).flatten(1)
        lanes = self.head(pooled).view(images.shape[0], self.num_lanes, LANE_OUTPUT_DIM)
        output: dict[str, torch.Tensor] = {"lanes": lanes}
        if self.distill_head is not None:
            output["distill_features"] = self.distill_head(features)
        return output


class StudentOnnxWrapper(nn.Module):
    """Single-tensor forward for ONNX / TensorRT."""

    def __init__(self, student: LaneSplineStudent) -> None:
        super().__init__()
        self.student = student

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.student(images)["lanes"]


def split_lane_tensor(packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``[..., 31]`` into existence logits and knots ``[..., K, 3]``."""
    if packed.shape[-1] != LANE_OUTPUT_DIM:
        raise ValueError(f"expected last dim {LANE_OUTPUT_DIM}, got {packed.shape[-1]}")
    existence = packed[..., 0]
    knots = packed[..., 1:].reshape(*packed.shape[:-1], NUM_KNOTS, 3)
    return existence, knots


def lanes_from_prediction(packed: np.ndarray, threshold: float = 0.5) -> list[Lane3D]:
    """Decode one ``[4, 31]`` prediction. Channel 0 is an existence logit."""
    array = np.asarray(packed, dtype=np.float32)
    if array.shape != (MAX_LANES, LANE_OUTPUT_DIM):
        raise ValueError(f"expected {(MAX_LANES, LANE_OUTPUT_DIM)}, got {array.shape}")
    logits = array[:, 0]
    probability = 1.0 / (1.0 + np.exp(-logits))
    knots = array[:, 1:].reshape(MAX_LANES, NUM_KNOTS, 3)
    lanes: list[Lane3D] = []
    for index, role in enumerate(LANE_ROLES):
        if float(probability[index]) < threshold:
            continue
        if not np.isfinite(knots[index]).all():
            continue
        lanes.append(
            Lane3D(
                lane_id=index,
                lane_type="unknown",
                confidence=float(np.clip(probability[index], 0.0, 1.0)),
                knots=[
                    SplineKnot3D(x=float(point[0]), y=float(point[1]), z=float(point[2]))
                    for point in knots[index]
                ],
                role=role,
            )
        )
    return lanes


def _build_encoder(backbone: str, pretrained: bool) -> tuple[nn.Module, int, str]:
    name = backbone.lower().strip()
    if name in {"tiny", "debug"}:
        encoder = _TinyEncoder()
        return encoder, encoder.out_channels, "tiny"
    if "mobile" in name:
        built = _try_timm(name, pretrained)
        if built is not None:
            return built
        logger.warning("timm could not build %s; falling back to resnet18", backbone)
    encoder, channels = _resnet18(pretrained)
    return encoder, channels, "resnet18"


def _resnet18(pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models import resnet18

    weights = None
    if pretrained:
        try:
            from torchvision.models import ResNet18_Weights

            weights = ResNet18_Weights.DEFAULT
        except Exception as exc:
            logger.warning("pretrained ResNet18 weights unavailable (%s)", exc)
            weights = None
    network = resnet18(weights=weights)
    encoder = nn.Sequential(*list(network.children())[:-2])
    return encoder, 512


def _try_timm(name: str, pretrained: bool) -> tuple[nn.Module, int, str] | None:
    try:
        import timm
    except ImportError:
        logger.info("timm is not installed")
        return None
    if name in {"mobilenetv4", "mobilenet_v4"}:
        candidates = [
            "mobilenetv4_conv_small",
            "mobilenetv4_conv_small.e2400_r224_in1k",
            "mobilenetv3_small_100",
        ]
    else:
        candidates = [name]
    for candidate in candidates:
        try:
            model = timm.create_model(
                candidate,
                pretrained=pretrained,
                features_only=True,
                out_indices=(-1,),
            )
        except Exception as exc:
            logger.info("timm model %s is unavailable (%s)", candidate, exc)
            continue
        channels = int(model.feature_info.channels()[-1])
        return _TimmEncoder(model, channels), channels, candidate
    return None
