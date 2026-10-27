"""DINOv3 dense features and metric depth.

``DINOv3DepthEstimator.predict`` always returns a dense depth map
``Z`` in meters with shape ``(H, W)`` for a single frame (or ``(B, H, W)``
for a batch). The source is, in order:

1. A cached ``.npy`` when ``frame_id`` is given and the file exists.
2. DINOv3 patch tokens passed through ``PatchDepthHead``, only when a
   calibrated ``depth_head_path`` was loaded. An untrained head is not metric
   and is not used for the lifter.
3. The ground-plane prior from ``src.common.geometry.ground_plane_depth``.

Weight loading defaults to ``local_files_only=True`` and sets
``HF_HUB_OFFLINE`` for that attempt. Nothing is downloaded during scaffold
setup. ``extract_features`` returns ``None`` when the backbone is absent.
"""

from __future__ import annotations

import contextlib
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.device import resolve_device
from src.common.geometry import ground_plane_depth

logger = logging.getLogger(__name__)

DEFAULT_DINO_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"


class PatchDepthHead(nn.Module):
    """1×1 head from DINOv3 patch tokens ``(B, C, h, w)`` to positive depth."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        hidden = max(in_dim // 2, 32)
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.proj(tokens))


class DINOv3DepthEstimator:
    """Dense depth for an RGB frame ``[3, H, W]`` or ``[H, W, 3]``."""

    def __init__(
        self,
        model_id: str = DEFAULT_DINO_ID,
        device: str | torch.device | None = None,
        cache_dir: str | Path | None = None,
        fallback: str = "ground_plane",
        local_files_only: bool = True,
        camera_k: np.ndarray | None = None,
        camera_height_m: float = 1.45,
        pitch_rad: float = 0.0,
        depth_head_path: str | Path | None = None,
        z_min: float = 0.5,
        z_max: float = 80.0,
    ) -> None:
        self.model_id = model_id
        if isinstance(device, torch.device):
            self.device = device
        elif device is None:
            self.device = resolve_device("auto")
        else:
            self.device = resolve_device(str(device))
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.fallback = fallback
        self.local_files_only = local_files_only
        self.camera_k = None if camera_k is None else np.asarray(camera_k, dtype=np.float64)
        self.camera_height_m = float(camera_height_m)
        self.pitch_rad = float(pitch_rad)
        self.depth_head_path = Path(depth_head_path) if depth_head_path else None
        self.z_min = z_min
        self.z_max = z_max
        self._model: nn.Module | None = None
        self._head: PatchDepthHead | None = None
        self._head_ready = False
        self.last_backend = fallback
        self._try_load()

    @property
    def features_available(self) -> bool:
        return self._model is not None

    def _try_load(self) -> None:
        try:
            from transformers import AutoModel
        except ImportError:
            logger.info("transformers is not installed; DINOv3 features stay disabled")
            return
        try:
            with _offline_if(self.local_files_only):
                model = AutoModel.from_pretrained(
                    self.model_id,
                    local_files_only=self.local_files_only,
                )
            model.eval()
            model.to(self.device)
            for param in model.parameters():
                param.requires_grad_(False)
            hidden = int(getattr(model.config, "hidden_size"))
            self._model = model
            self._head = PatchDepthHead(hidden).to(self.device)
            if self.depth_head_path is not None and self.depth_head_path.is_file():
                state = torch.load(self.depth_head_path, map_location=self.device)
                self._head.load_state_dict(state)
                self._head.eval()
                self._head_ready = True
                logger.info("loaded DINOv3 depth head from %s", self.depth_head_path)
            else:
                logger.info(
                    "DINOv3 backbone ready; metric depth head is not calibrated, "
                    "predict() will use the %s prior",
                    self.fallback,
                )
        except Exception as exc:
            logger.warning(
                "DINOv3 weights are not available locally (%s). Depth backend: %s",
                exc,
                self.fallback,
            )
            self._model = None
            self._head = None

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor | np.ndarray,
        frame_id: str | None = None,
    ) -> torch.Tensor:
        """Return metric depth in meters.

        A single ``[3, H, W]`` / ``[H, W, 3]`` image returns ``[H, W]``.
        A batch ``[B, 3, H, W]`` returns ``[B, H, W]``. Values are 0 where the
        ground-plane ray misses the ground.
        """
        if frame_id and self.cache_dir is not None:
            cache_path = self.cache_dir / f"{frame_id}.npy"
            if cache_path.is_file():
                cached = np.load(cache_path)
                self.last_backend = "cache"
                return torch.from_numpy(np.asarray(cached, dtype=np.float32))

        batch, single = _as_batch(image, self.device)
        if self._model is not None and self._head_ready and self._head is not None:
            depth = self._dino_metric(batch)
            self.last_backend = "dinov3"
        elif self.fallback == "ground_plane":
            depth = self._plane_batch(batch.shape[0], batch.shape[2], batch.shape[3])
            self.last_backend = "dinov3-features+ground_plane" if self._model is not None else "ground_plane"
        else:
            raise RuntimeError(f"unknown depth fallback {self.fallback!r}")
        return depth[0] if single else depth

    @torch.no_grad()
    def extract_features(self, image: torch.Tensor | np.ndarray) -> torch.Tensor | None:
        """Return DINOv3 patch tokens ``(B, C, h, w)``, or None without weights."""
        if self._model is None:
            return None
        batch, _single = _as_batch(image, self.device)
        tokens = self._patch_tokens(batch)
        return tokens

    def _dino_metric(self, batch: torch.Tensor) -> torch.Tensor:
        assert self._head is not None
        tokens = self._patch_tokens(batch)
        relative = self._head(tokens)
        depth = F.interpolate(relative, size=batch.shape[-2:], mode="bilinear", align_corners=False)
        return depth[:, 0].clamp(self.z_min, self.z_max)

    def _patch_tokens(self, batch: torch.Tensor) -> torch.Tensor:
        assert self._model is not None
        # ImageNet normalization matches the student contract's internal norm,
        # but DINOv3 checkpoints often share that preprocessing. Callers pass
        # RGB in [0, 1].
        mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        normalized = (batch - mean) / std
        try:
            hidden = self._model(pixel_values=normalized)
        except TypeError:
            hidden = self._model(normalized)
        tokens = hidden.last_hidden_state
        patch = int(getattr(self._model.config, "patch_size", 16) or 16)
        if isinstance(patch, (tuple, list)):
            patch = int(patch[0])
        height, width = batch.shape[-2:]
        grid_h = height // patch
        grid_w = width // patch
        # Drop the class token when the sequence is one longer than the grid.
        expected = grid_h * grid_w
        if tokens.shape[1] == expected + 1:
            tokens = tokens[:, 1:, :]
        elif tokens.shape[1] != expected:
            # Some DINOv3 registers add extra prefix tokens.
            tokens = tokens[:, -expected:, :]
        channels = tokens.shape[-1]
        return tokens.transpose(1, 2).reshape(batch.shape[0], channels, grid_h, grid_w)

    def _plane_batch(self, batch: int, height: int, width: int) -> torch.Tensor:
        if self.camera_k is None:
            logger.warning("ground-plane depth requested without intrinsics; returning zeros")
            plane = np.zeros((height, width), dtype=np.float32)
        else:
            plane = ground_plane_depth(
                height,
                width,
                self.camera_k,
                camera_height_m=self.camera_height_m,
                pitch_rad=self.pitch_rad,
                z_min=self.z_min,
                z_max=self.z_max,
            )
        tensor = torch.from_numpy(plane).to(self.device)
        return tensor.unsqueeze(0).expand(batch, -1, -1).contiguous()


def _as_batch(image: torch.Tensor | np.ndarray, device: torch.device) -> tuple[torch.Tensor, bool]:
    """Return ``(batch CHW float in [0, 1], was_single_image)``."""
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(image))
    elif isinstance(image, torch.Tensor):
        tensor = image.detach()
    else:
        raise TypeError("image must be a numpy array or a torch tensor")
    tensor = tensor.float()
    single = tensor.ndim == 3
    if tensor.ndim == 3:
        if tensor.shape[0] == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.shape[-1] == 3:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"expected [3,H,W] or [H,W,3], got {tuple(tensor.shape)}")
    elif tensor.ndim == 4:
        if tensor.shape[1] != 3 and tensor.shape[-1] == 3:
            tensor = tensor.permute(0, 3, 1, 2)
        if tensor.shape[1] != 3:
            raise ValueError(f"expected 3 channels, got {tuple(tensor.shape)}")
    else:
        raise ValueError(f"expected 3D or 4D image, got {tuple(tensor.shape)}")
    if torch.isfinite(tensor).any() and float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    return tensor.to(device), single


@contextlib.contextmanager
def _offline_if(enabled: bool):
    if not enabled:
        yield
        return
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous
