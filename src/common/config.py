"""YAML and .env loading. Missing files return empty defaults."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def load_project_env(env_file: Path | None = None) -> None:
    """Load ``.env`` from the working directory when python-dotenv is installed.

    Existing environment variables win. A missing dotenv package is ignored so
    geometry tests can import the teacher without the extra dependency.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    path = env_file if env_file is not None else Path.cwd() / ".env"
    if path.is_file():
        load_dotenv(path, override=False)


def load_yaml(path: str | Path | None) -> dict[str, Any]:
    """Load a YAML mapping. ``None`` and missing files yield ``{}``."""
    if path is None:
        return {}
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"config not found: {file_path}")
    import yaml

    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {file_path} must be a mapping")
    return data


def dig(config: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    """Walk nested dictionaries, returning ``default`` at the first miss."""
    node: Any = config or {}
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
