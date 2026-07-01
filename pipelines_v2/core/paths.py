"""Workspace and local-state path helpers."""

from __future__ import annotations

import os
from pathlib import Path


def find_workspace_root(start: Path | None = None) -> Path:
    """Best-effort repository/workspace root for the current checkout.

    ``XENON_WORKSPACE_ROOT`` overrides detection entirely. Set it to the Xenon
    checkout when running workflows from a sibling repo (non-editable install),
    where module-path/cwd detection would otherwise land on the wrong repo and
    Modal source mounts like ``<root>/pipelines_v2`` would not exist.
    """

    override = os.environ.get("XENON_WORKSPACE_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    candidates: list[Path] = []
    if start is not None:
        resolved = Path(start).resolve()
        if resolved.is_file():
            resolved = resolved.parent
        candidates.extend([resolved, *resolved.parents])

    module_path = Path(__file__).resolve()
    candidates.extend([module_path.parent, *module_path.parents])

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate

    return module_path.parents[2]


def resolve_workspace_path(path: str | Path, *, workspace_root: Path | None = None) -> Path:
    """Resolve a possibly-relative path against the detected workspace root."""

    resolved = Path(path)
    if resolved.is_absolute():
        return resolved.resolve()
    root = workspace_root or find_workspace_root()
    return (root / resolved).resolve()


def xenon_home() -> Path:
    """Return the default local Xenon home directory."""

    override = os.environ.get("XENON_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.home() / ".xenon").resolve()


def pipelines_v2_state_root() -> Path:
    """Return the default local pipelines_v2 state root."""

    return xenon_home() / "pipelines_v2"


def pipelines_v2_catalog_root() -> Path:
    """Return the default local catalog mirror root."""

    return pipelines_v2_state_root() / "catalog"
