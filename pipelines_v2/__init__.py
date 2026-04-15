"""V2 pipelines package.

This package is intentionally separate from `pipelines` while the ARCH2 API is
being built out.

Keep the package root light: importing a submodule such as
``pipelines_v2.runtime.remote_executor`` should not eagerly pull in the full
public API surface, because that can drag local-only dependencies into remote
execution environments.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    """Lazily resolve public API symbols from ``pipelines_v2.api``."""

    api = import_module("pipelines_v2.api")
    try:
        return getattr(api, name)
    except AttributeError as exc:  # pragma: no cover - mirrors normal module lookup
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:
    api = import_module("pipelines_v2.api")
    return sorted(set(globals()) | set(getattr(api, "__all__", ())))
