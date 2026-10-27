"""Meta model API wrapper for SAM 3.1 Video and Llama 3.2 Vision.

The default client is a mock. It builds the same JSON the live endpoints
expect and never opens a socket. Set ``mock=False`` or ``META_API_MOCK=0``
together with ``META_AI_API_KEY`` to POST.

The base URL is a placeholder (``META_API_BASE_URL``). Point it at the
account you were issued for the hackathon; the payload shapes below are the
contract this repo parses.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
from typing import Any

import numpy as np

from src.common.config import env_flag, load_project_env

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
SAM_MODEL = "sam-3.1-video"
VLM_MODEL = "llama-3.2-11b-vision-instruct"


class MetaModelClient:
    """Sync client. Async entry points share the mock short-circuit."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        mock: bool | None = None,
        sam_model: str = SAM_MODEL,
        vlm_model: str = VLM_MODEL,
        timeout_s: float = 60.0,
    ) -> None:
        load_project_env()
        self.api_key = api_key if api_key is not None else os.environ.get("META_AI_API_KEY", "")
        self.base_url = (
            base_url if base_url is not None else os.environ.get("META_API_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")
        # Default mock stays on even if a key is present, so importing the
        # client during scaffold setup cannot spend credits.
        if mock is None:
            self.mock = env_flag("META_API_MOCK", default=True)
        else:
            self.mock = bool(mock)
        self.sam_model = sam_model
        self.vlm_model = vlm_model
        self.timeout_s = timeout_s

    def build_sam_request(
        self,
        frames: list[np.ndarray],
        prompts: list[dict[str, Any]],
        propagate: bool = True,
        include_image_bytes: bool = False,
    ) -> dict[str, Any]:
        """JSON body for ``POST /sam/video/segment``.

        Each prompt is ``{frame_idx, object_id, role, points: [[u, v], ...],
        point_labels: [1, ...]}``. Image bytes are JPEG base64 and are omitted
        unless ``include_image_bytes`` is set, which is what the live call uses.
        """
        payload_frames = []
        for index, frame in enumerate(frames):
            jpeg = _encode_jpeg(frame) if include_image_bytes else b""
            payload_frames.append(
                {
                    "frame_idx": index,
                    "height": int(frame.shape[0]),
                    "width": int(frame.shape[1]),
                    "sha256": hashlib.sha256(_encode_jpeg(frame)).hexdigest(),
                    "image_jpeg_b64": base64.b64encode(jpeg).decode("ascii") if jpeg else None,
                }
            )
        return {
            "model": self.sam_model,
            "propagate": propagate,
            "output": "mask_png_b64",
            "frames": payload_frames,
            "prompts": prompts,
        }

    def build_vlm_request(
        self,
        prompt: str,
        image_bgr: np.ndarray | None,
        include_image_bytes: bool = False,
    ) -> dict[str, Any]:
        """JSON body for ``POST /chat/completions`` (Llama 3.2 Vision)."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if image_bgr is not None and include_image_bytes:
            jpeg = _encode_jpeg(image_bgr)
            url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": url}})
        elif image_bgr is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": None, "sha256": hashlib.sha256(_encode_jpeg(image_bgr)).hexdigest()},
                }
            )
        return {
            "model": self.vlm_model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You audit monocular 3D lane splines for a driving perception "
                        "stack. Reply with a JSON object only."
                    ),
                },
                {"role": "user", "content": content},
            ],
        }

    def segment_video(
        self,
        frames: list[np.ndarray],
        prompts: list[dict[str, Any]],
        propagate: bool = True,
    ) -> dict[str, Any]:
        """Run SAM 3.1 Video, or return the request echo when mocked.

        The mock does not invent masks. ``src.teacher.tracker`` owns the
        offline color tracker so the HTTP client stays a transport.
        """
        payload = self.build_sam_request(
            frames,
            prompts,
            propagate=propagate,
            include_image_bytes=not self.mock,
        )
        if self.mock:
            logger.info("SAM 3.1 Video mock: %d frame(s), %d prompt(s)", len(frames), len(prompts))
            return {"mock": True, "model": self.sam_model, "request": _redact_request(payload), "frames": []}
        return self._http_post("/sam/video/segment", payload)

    def audit_vision(self, prompt: str, image_bgr: np.ndarray | None) -> dict[str, Any]:
        """Call Llama 3.2 Vision, or return a mock envelope the auditor fills."""
        payload = self.build_vlm_request(prompt, image_bgr, include_image_bytes=not self.mock)
        if self.mock:
            logger.info("Llama 3.2 Vision mock (%d prompt chars)", len(prompt))
            return {"mock": True, "model": self.vlm_model, "request": _redact_request(payload)}
        return self._http_post("/chat/completions", payload)

    async def segment_video_async(
        self,
        frames: list[np.ndarray],
        prompts: list[dict[str, Any]],
        propagate: bool = True,
    ) -> dict[str, Any]:
        """Async SAM call. The mock path returns before any aiohttp session."""
        payload = self.build_sam_request(
            frames,
            prompts,
            propagate=propagate,
            include_image_bytes=not self.mock,
        )
        if self.mock:
            return {"mock": True, "model": self.sam_model, "request": _redact_request(payload), "frames": []}
        return await self._http_post_async("/sam/video/segment", payload)

    def _http_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import requests

        if not self.api_key:
            raise RuntimeError("META_AI_API_KEY is empty and META_API_MOCK is off")
        response = requests.post(
            self.base_url + path,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError(f"expected a JSON object from {path}")
        return body

    async def _http_post_async(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        import aiohttp

        if not self.api_key:
            raise RuntimeError("META_AI_API_KEY is empty and META_API_MOCK is off")
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.base_url + path,
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            ) as response:
                response.raise_for_status()
                body = await response.json()
        if not isinstance(body, dict):
            raise RuntimeError(f"expected a JSON object from {path}")
        return body


def _encode_jpeg(frame: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 image as JPEG. Imported locally to keep cv2 optional here."""
    import cv2

    image = np.asarray(frame)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("failed to JPEG-encode a frame for the Meta API payload")
    return buf.tobytes()


def _redact_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop bulky image bytes before a mock response is logged or saved."""
    import copy

    redacted = copy.deepcopy(payload)
    for frame in redacted.get("frames", []):
        if isinstance(frame, dict) and frame.get("image_jpeg_b64"):
            frame["image_jpeg_b64"] = f"<{len(frame['image_jpeg_b64'])} chars>"
    for message in redacted.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            url = (part.get("image_url") or {}).get("url")
            if isinstance(url, str) and url.startswith("data:"):
                part["image_url"]["url"] = f"<data-url {len(url)} chars>"
    return redacted
