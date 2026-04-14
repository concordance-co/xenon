"""Workspace path helpers used by spec and runtime serialization."""

from __future__ import annotations

from pathlib import Path


def find_workspace_root(start: Path | None = None) -> Path:
    """Best-effort repository/workspace root for the current checkout."""

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
