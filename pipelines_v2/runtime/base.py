"""Runner contracts and execution plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from pipelines_v2.core.types import CapabilityError, EngineCapability, OperationSpec, SpecValidationError


class Runner(Protocol):
    """Execution environment for one operation spec."""

    def plan(self, spec: OperationSpec) -> "ExecutionPlan":
        """Preflight a spec against this runner."""
        ...

    def run(self, spec: OperationSpec) -> Any:
        """Execute one spec and return its typed artifact/result."""
        ...


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Preflight result for one spec on one runner."""
    spec_kind: str
    required_capabilities: frozenset[EngineCapability]
    engine_capabilities: frozenset[EngineCapability]
    artifact_kinds: tuple[str, ...]
    checks: tuple[str, ...] = field(default_factory=tuple)
    errors: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def missing_capabilities(self) -> set[EngineCapability]:
        """Return engine capabilities that the spec requires but the engine lacks."""
        return set(self.required_capabilities) - set(self.engine_capabilities)

    @property
    def valid(self) -> bool:
        """Whether the plan has no capability gaps and no validation errors."""
        return not self.missing_capabilities and not self.errors

    def require_capabilities(self) -> None:
        """Raise if the bound engine cannot satisfy the spec's requirements."""
        missing = sorted(cap.value for cap in self.missing_capabilities)
        if missing:
            raise CapabilityError(f"Engine is missing required capabilities: {missing}")

    def estimated_artifacts(self) -> tuple[str, ...]:
        """Return the artifact kinds this run is expected to produce."""
        return self.artifact_kinds

    def require_checks(self) -> None:
        """Raise if any non-capability validation checks failed."""
        if self.errors:
            raise SpecValidationError("; ".join(self.errors))

    def validate(self) -> "ExecutionPlan":
        """Raise on any plan failure and return the validated plan on success."""
        self.require_capabilities()
        self.require_checks()
        return self
