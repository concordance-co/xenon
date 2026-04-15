"""Workflow orchestration layer.

Keep the package import light so callers that only need record types do not
eagerly import the orchestrator and runner stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_MODULES = (
    "pipelines_v2.workflow.records",
    "pipelines_v2.workflow.specs",
    "pipelines_v2.workflow.orchestrator",
)


def __getattr__(name: str) -> Any:
    for module_name in _MODULES:
        module = import_module(module_name)
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals())
    for module_name in _MODULES:
        module = import_module(module_name)
        names.update(getattr(module, "__all__", ()))
        names.update(name for name in vars(module) if not name.startswith("_"))
    return sorted(names)
