"""Shared geometry, label schemas, and runtime helpers.

Import submodules directly (`src.common.geometry`, `src.common.types`) so that
loading a schema does not import Torch.
"""

__all__ = ["geometry", "types", "device", "config"]
