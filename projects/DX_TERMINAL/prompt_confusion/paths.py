from __future__ import annotations

import os
from pathlib import Path


def prompt_confusion_root(start: str | Path | None = None) -> Path:
    current = Path(start).resolve() if start is not None else Path(__file__).resolve()
    candidates = [current, *current.parents]
    for candidate in candidates:
        if candidate.name == "prompt_confusion" and candidate.parent.name == "DX_TERMINAL":
            return candidate
    raise ValueError(f"Could not resolve prompt_confusion root from {current}")


def dx_terminal_root(start: str | Path | None = None) -> Path:
    return prompt_confusion_root(start).parent


def phase_root(phase_name: str, start: str | Path | None = None) -> Path:
    return prompt_confusion_root(start) / phase_name


def phase_outputs_dir(phase_name: str, start: str | Path | None = None) -> Path:
    return phase_root(phase_name, start) / "outputs"


def dataset_exports_root(start: str | Path | None = None) -> Path:
    return dx_terminal_root(start) / "dataset_exports"


def pipelines_catalog_root() -> Path:
    configured = os.environ.get("XENON_PIPELINES_CATALOG_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".xenon" / "pipelines_v2" / "catalog").resolve()


def pipelines_cache_root() -> Path:
    configured = os.environ.get("XENON_PIPELINES_CACHE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".xenon" / "pipelines_v2" / "cache").resolve()
