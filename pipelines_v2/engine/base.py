"""Engine contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pipelines_v2.core.types import EngineCapability, RuntimeSecret
from pipelines_v2.operations.specs import CaptureSpec


class RuntimeSpec(Protocol):
    """Runner-agnostic runtime requirements declared by an engine or spec."""

    kind: str
    secrets: tuple[RuntimeSecret, ...]


@dataclass(frozen=True, slots=True)
class PythonRuntimeSpec:
    """Concrete runtime description for Python-based execution environments."""
    kind: str = "python"
    python_version: str = "3.13"
    pip_packages: tuple[str, ...] = field(default_factory=tuple)
    env: dict[str, str] = field(default_factory=dict)
    secrets: tuple[RuntimeSecret, ...] = field(default_factory=tuple)
    local_python_sources: tuple[str, ...] = field(default_factory=tuple)


class Engine(Protocol):
    """Model-backed execution surface for model-bound operation specs."""

    def identity(self) -> dict[str, Any]:
        """Return a serializable engine descriptor."""
        ...

    def semantic_identity(self) -> dict[str, Any]:
        """Return the semantic engine descriptor used for reuse and invalidation."""
        ...

    def capabilities(self) -> set[EngineCapability]:
        """Return the feature set this engine can satisfy."""
        ...

    def runtime_spec(self) -> RuntimeSpec:
        """Return runtime requirements needed to execute this engine."""
        ...

    def planning_errors(self, spec: CaptureSpec) -> tuple[str, ...]:
        """Return engine-specific planning errors for this capture spec."""
        ...

    def capture(self, spec: CaptureSpec) -> "EngineCaptureResult":
        """Execute one capture spec and return features/generations."""
        ...


@dataclass(frozen=True, slots=True)
class EngineCaptureResult:
    """Raw engine output before the runner persists it into an artifact."""
    features: dict[str, dict[str, Any]]
    generations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
