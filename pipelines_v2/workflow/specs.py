"""Workflow specs for multi-step orchestration."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.operations import operation_spec_from_dict
from pipelines_v2.runtime import ExecutionPlan


@dataclass(frozen=True, slots=True)
class StepRef:
    """Reference to the artifact produced by a prior workflow step."""
    step: str
    output: str = "artifact"

    kind: ClassVar[str] = "step_ref"

    def feature(self, name: str) -> "StepFeatureRef":
        """Reference one named feature from the referenced step."""
        return StepFeatureRef(step=self.step, feature_name=name)

    def label(self, name: str) -> "StepLabelRef":
        """Reference one named label from the referenced step."""
        return StepLabelRef(step=self.step, label_name=name)

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "output": self.output}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StepRef":
        return cls(
            step=str(payload["step"]),
            output=str(payload.get("output", "artifact")),
        )


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    """One named workflow node executed on one named runner."""
    name: str
    runner: str
    spec: Any
    depends_on: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise SpecValidationError("WorkflowStep name cannot be empty")
        if not self.runner.strip():
            raise SpecValidationError(f"WorkflowStep {self.name!r} requires a runner")

    def resolved_depends_on(self) -> tuple[str, ...]:
        """Return explicit plus inferred dependencies for this step."""
        return tuple(sorted(set(self.depends_on) | _step_dependencies_in_value(self.spec)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "runner": self.runner,
            "spec": self.spec.to_dict(),
            "depends_on": list(self.depends_on),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkflowStep":
        return cls(
            name=str(payload["name"]),
            runner=str(payload["runner"]),
            spec=operation_spec_from_dict(dict(payload["spec"])),
            depends_on=tuple(str(item) for item in payload.get("depends_on", ())),
        )


@dataclass(frozen=True, slots=True)
class WorkflowStepPlan:
    """Preflight result for one workflow step."""
    name: str
    runner: str
    depends_on: tuple[str, ...]
    execution: ExecutionPlan


@dataclass(frozen=True, slots=True)
class StepFeatureRef:
    """Reference to a feature produced by a prior workflow step."""
    step: str
    feature_name: str
    layer_index: int | None = None

    kind: ClassVar[str] = "step_feature_ref"

    def layer(self, layer: int) -> "StepFeatureRef":
        """Narrow this ref to one feature layer."""
        return StepFeatureRef(step=self.step, feature_name=self.feature_name, layer_index=layer)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "step": self.step,
            "feature_name": self.feature_name,
        }
        if self.layer_index is not None:
            payload["layer_index"] = self.layer_index
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StepFeatureRef":
        return cls(
            step=str(payload["step"]),
            feature_name=str(payload["feature_name"]),
            layer_index=int(payload["layer_index"]) if payload.get("layer_index") is not None else None,
        )


@dataclass(frozen=True, slots=True)
class StepLabelRef:
    """Reference to a label payload produced by a prior workflow step."""
    step: str
    label_name: str

    kind: ClassVar[str] = "step_label_ref"

    def equals(self, value: Any) -> Any:
        """Build an equality predicate over this step-produced label."""
        from pipelines_v2.data.datasets import LabelPredicate

        return LabelPredicate(label_set=self, op="equals", value=value)

    def runtime_secrets(self) -> tuple[Any, ...]:
        return ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "label_name": self.label_name,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StepLabelRef":
        return cls(
            step=str(payload["step"]),
            label_name=str(payload["label_name"]),
        )


@dataclass(frozen=True, slots=True)
class WorkflowPlan:
    """Plan output for a whole workflow."""
    name: str | None
    steps: tuple[WorkflowStepPlan, ...]


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    """A DAG of workflow steps executed by a ``WorkflowOrchestrator``."""
    steps: tuple[WorkflowStep, ...]
    name: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        names = [step.name for step in self.steps]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise SpecValidationError(f"Duplicate workflow step names: {sorted(duplicates)}")
        known = set(names)
        for step in self.steps:
            if step.name in step.resolved_depends_on():
                raise SpecValidationError(f"Workflow step {step.name!r} cannot depend on itself")
            missing = sorted(set(step.resolved_depends_on()) - known)
            if missing:
                raise SpecValidationError(
                    f"Workflow step {step.name!r} depends on unknown steps: {missing}"
                )
        _ordered_steps(self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the workflow into a JSON-safe payload."""
        return {
            "kind": "workflow",
            "schema_version": self.schema_version,
            "name": self.name,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "WorkflowSpec":
        kind = payload.get("kind")
        if kind not in (None, "workflow"):
            raise SpecValidationError(f"WorkflowSpec expected kind 'workflow', got {kind!r}")
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            name=payload.get("name"),
            steps=tuple(WorkflowStep.from_dict(step) for step in payload.get("steps", ())),
        )

    def ordered_steps(self) -> tuple[WorkflowStep, ...]:
        """Return steps in dependency order."""
        return _ordered_steps(self.steps)


def _ordered_steps(steps: tuple[WorkflowStep, ...] | list[WorkflowStep]) -> tuple[WorkflowStep, ...]:
    by_name = {step.name: step for step in steps}
    dependencies = {step.name: step.resolved_depends_on() for step in steps}
    indegree = {step.name: len(dependencies[step.name]) for step in steps}
    children: dict[str, list[str]] = {step.name: [] for step in steps}
    for step in steps:
        for dep in dependencies[step.name]:
            children[dep].append(step.name)

    ready = deque(step.name for step in steps if indegree[step.name] == 0)
    ordered: list[WorkflowStep] = []

    while ready:
        name = ready.popleft()
        ordered.append(by_name[name])
        for child in children[name]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)

    if len(ordered) != len(by_name):
        remaining = sorted(name for name, degree in indegree.items() if degree > 0)
        raise SpecValidationError(f"Workflow contains a dependency cycle across steps: {remaining}")

    return tuple(ordered)


def _step_dependencies_in_value(value: Any) -> set[str]:
    if isinstance(value, (StepRef, StepFeatureRef, StepLabelRef)):
        return {value.step}
    if hasattr(value, "__dataclass_fields__"):
        deps: set[str] = set()
        for field_name in value.__dataclass_fields__:
            deps.update(_step_dependencies_in_value(getattr(value, field_name)))
        return deps
    if isinstance(value, tuple | list):
        deps: set[str] = set()
        for item in value:
            deps.update(_step_dependencies_in_value(item))
        return deps
    if isinstance(value, dict):
        deps: set[str] = set()
        for item in value.values():
            deps.update(_step_dependencies_in_value(item))
        return deps
    return set()
