"""Datasets that yield RGB in ``[0, 1]`` and packed ``[4, 31]`` targets.

The student module applies ImageNet normalization internally. Do not normalize
again in the dataset or the ONNX graph will see it twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.common.geometry import MAX_LANES, NUM_KNOTS, pack_lane_array, role_index
from src.common.types import DistillationDataset, Frame3DAnnotation


def preprocess_bgr(image_bgr: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a BGR uint8 image to RGB CHW float in ``[0, 1]``."""
    import cv2

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    chw = resized.astype(np.float32) / 255.0
    return np.transpose(chw, (2, 0, 1))


class LaneSplineDataset(Dataset):
    """Load ``index.json`` produced by ``src.teacher.pipeline.distill_clip``."""

    def __init__(self, root: str | Path, image_height: int = 288, image_width: int = 512) -> None:
        self.root = Path(root)
        index_path = self.root / "index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing distillation index: {index_path}")
        self.index = DistillationDataset.model_validate_json(index_path.read_text(encoding="utf-8"))
        if not self.index.frames:
            raise FileNotFoundError(f"distillation index {index_path} has no frames")
        self.image_height = image_height
        self.image_width = image_width

    def __len__(self) -> int:
        return len(self.index.frames)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        import cv2

        name = self.index.frames[item]
        annotation = Frame3DAnnotation.model_validate_json((self.root / name).read_text(encoding="utf-8"))
        image_path = _resolve_image(self.root, self.index.image_root, annotation.image)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"failed to read {image_path}")
        image = preprocess_bgr(bgr, self.image_height, self.image_width)
        target = _pack_annotation(annotation)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "target": torch.from_numpy(target),
        }


class SyntheticLaneDataset(Dataset):
    """In-memory straight lanes for a training-loop smoke test."""

    def __init__(
        self,
        length: int = 32,
        image_height: int = 128,
        image_width: int = 224,
        seed: int = 0,
    ) -> None:
        self.length = length
        self.image_height = image_height
        self.image_width = image_width
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed + item)
        image = rng.random((3, self.image_height, self.image_width), dtype=np.float32)
        existence = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        knots = np.zeros((MAX_LANES, NUM_KNOTS, 3), dtype=np.float32)
        z = np.linspace(5.0, 40.0, NUM_KNOTS, dtype=np.float32)
        knots[0, :, 0] = -1.75
        knots[0, :, 1] = 1.5
        knots[0, :, 2] = z
        knots[1, :, 0] = 1.75
        knots[1, :, 1] = 1.5
        knots[1, :, 2] = z
        target = pack_lane_array(existence, knots)
        return {
            "image": torch.from_numpy(image),
            "target": torch.from_numpy(target),
        }


def _pack_annotation(annotation: Frame3DAnnotation) -> np.ndarray:
    existence = np.zeros((MAX_LANES,), dtype=np.float32)
    knots = np.zeros((MAX_LANES, NUM_KNOTS, 3), dtype=np.float32)
    for lane in annotation.lanes:
        slot = role_index(lane.role)
        if slot is None and 0 <= lane.lane_id < MAX_LANES:
            slot = lane.lane_id
        if slot is None:
            continue
        existence[slot] = 1.0
        knots[slot] = np.asarray(lane.knot_array(), dtype=np.float32)
    return pack_lane_array(existence, knots)


def _resolve_image(index_dir: Path, image_root: str, image: str | None) -> Path:
    if not image:
        raise FileNotFoundError("annotation has no image path")
    path = Path(image)
    if path.is_absolute():
        return path
    return (index_dir / image_root / path).resolve()
