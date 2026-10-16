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

    def run(
        self,
        spec: OperationSpec,
        *,
        workflow_context: Any | None = None,
        progress_callback: Any | None = None,
    ) -> Any:
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

    admission_checks: tuple[Any, ...] = field(default_factory=tuple)
    effective_config: dict[str, Any] = field(default_factory=dict)
    workload_envelope: Any | None = None

    @property
    def ready(self) -> bool:
        checks = [item for item in self.admission_checks if item.phase in {"static", "readiness"}]
        return self.valid and any(item.phase == "readiness" for item in checks) and all(item.status in {"passed", "not_applicable"} for item in checks)

    @property
    def qualified(self) -> bool:
        checks = [item for item in self.admission_checks if item.phase == "qualification"]
        return self.ready and bool(checks) and all(item.status in {"passed", "not_applicable"} for item in checks)

    @property
    def missing_capabilities(self) -> set[EngineCapability]:
        """Return engine capabilities that the spec requires but the engine lacks."""
        return set(self.required_capabilities) - set(self.engine_capabilities)

    @property
    def valid(self) -> bool:
        """Whether the plan has no capability gaps and no validation errors."""
        return not self.missing_capabilities and not self.errors and not any(item.phase == "static" and item.status == "failed" for item in self.admission_checks)

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
        errors = [*self.errors, *(item.message for item in self.admission_checks if item.phase == "static" and item.status == "failed")]
        if errors:
            raise SpecValidationError("; ".join(errors))

    def validate(self) -> "ExecutionPlan":
        """Raise on any plan failure and return the validated plan on success."""
        self.require_capabilities()
        self.require_checks()
        return self
