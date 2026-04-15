"""Environment loading helpers for CLI entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def load_dotenv_if_present(*, search_roots: Iterable[Path] | None = None) -> Path | None:
    """Load a nearby ``.env`` file without overriding existing environment vars."""
    candidates = list(search_roots) if search_roots is not None else list(_default_search_roots())
    seen: set[Path] = set()
    for root in candidates:
        root_path = Path(root).resolve()
        for candidate in (root_path, *root_path.parents):
            env_path = candidate / ".env"
            if env_path in seen:
                continue
            seen.add(env_path)
            if not env_path.is_file():
                continue
            _load_dotenv_file(env_path)
            return env_path
    return None


def _default_search_roots() -> tuple[Path, ...]:
    return (
        Path.cwd(),
        Path(__file__).resolve().parent.parent.parent,
    )


def _load_dotenv_file(path: Path) -> None:
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key not in os.environ:
                os.environ[key] = value
