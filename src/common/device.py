"""Torch device selection.

Order: Apple Silicon MPS, then CUDA, then CPU. This is the project-wide rule.
``resolve_device("cpu")`` is what unit tests should pass so a GPU host does
not accidentally allocate.
"""

from __future__ import annotations

import torch


def mps_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def resolve_device(preference: str | None = None) -> torch.device:
    """Resolve ``auto``, ``mps``, ``cuda``, or ``cpu``.

    ``auto`` uses MPS when ``torch.backends.mps.is_available()``, else CUDA,
    else CPU. Requesting a backend that is missing raises ``RuntimeError``
    instead of silently moving the run.
    """
    pref = (preference or "auto").strip().lower()
    mps = mps_available()
    cuda = torch.cuda.is_available()
    if pref == "mps":
        if not mps:
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    if pref == "cuda":
        if not cuda:
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if pref == "cpu":
        return torch.device("cpu")
    if pref not in {"auto", ""}:
        raise ValueError(f"unknown device preference {preference!r}")
    if mps:
        return torch.device("mps")
    if cuda:
        return torch.device("cuda")
    return torch.device("cpu")
