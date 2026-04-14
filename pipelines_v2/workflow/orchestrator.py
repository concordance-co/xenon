"""Workflow orchestration over named runners."""

from __future__ import annotations

import dataclasses
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Mapping

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.operations.specs import CaptureSpec, TokenSelector
from pipelines_v2.runtime import Runner
from pipelines_v2.workflow.specs import StepFeatureRef, StepLabelRef, StepRef, WorkflowPlan, WorkflowSpec, WorkflowStepPlan


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    """Collected outputs from a completed workflow run."""
    step_results: dict[str, Any] = field(default_factory=dict)

    def step(self, name: str) -> Any:
        """Return the result produced by one named workflow step."""
        return self.step_results[name]


@dataclass(slots=True)
class WorkflowOrchestrator:
    """Execute a workflow over named runners with dependency-aware fanout."""
    runners: Mapping[str, Runner]
    max_parallelism: int | None = None

    def plan(self, workflow: WorkflowSpec) -> WorkflowPlan:
        """Preflight each workflow step against its assigned runner."""
        workflow_errors = _workflow_section_metadata_errors(workflow)
        step_plans: list[WorkflowStepPlan] = []
        for step in workflow.ordered_steps():
            try:
                runner = self.runners[step.runner]
            except KeyError as exc:
                raise SpecValidationError(
                    f"Workflow step {step.name!r} references unknown runner {step.runner!r}"
                ) from exc
            execution = runner.plan(step.spec)
            extra_errors = tuple(workflow_errors.get(step.name, ()))
            if extra_errors:
                execution = dataclasses.replace(
                    execution,
                    errors=tuple(execution.errors) + extra_errors,
                )
            step_plans.append(
                WorkflowStepPlan(
                    name=step.name,
                    runner=step.runner,
                    depends_on=step.resolved_depends_on(),
                    execution=execution,
                )
            )
        return WorkflowPlan(name=workflow.name, steps=tuple(step_plans))

    def run(self, workflow: WorkflowSpec) -> WorkflowResult:
        """Execute a workflow, resolving step refs as dependencies complete."""
        plan = self.plan(workflow)
        for step in plan.steps:
            step.execution.validate()
        ordered_steps = workflow.ordered_steps()
        step_by_name = {step.name: step for step in ordered_steps}
        dependencies = {step.name: set(step.resolved_depends_on()) for step in ordered_steps}

        results: dict[str, Any] = {}
        pending = set(step_by_name)
        running: dict[Future[Any], str] = {}
        max_workers = self.max_parallelism or max(1, len(ordered_steps))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while pending or running:
                ready = [
                    step_by_name[name]
                    for name in sorted(pending)
                    if dependencies[name].issubset(results)
                ]
                for step in ready:
                    runner = self.runners[step.runner]
                    resolved_spec = _resolve_step_refs(step.spec, results)
                    future = pool.submit(runner.run, resolved_spec)
                    running[future] = step.name
                    pending.remove(step.name)

                if not running:
                    unresolved = sorted(pending)
                    raise SpecValidationError(
                        f"Workflow could not make progress; unresolved steps remain: {unresolved}"
                    )

                done, _ = wait(set(running), return_when=FIRST_COMPLETED)
                for future in done:
                    step_name = running.pop(future)
                    try:
                        results[step_name] = future.result()
                    except Exception:
                        for outstanding in running:
                            outstanding.cancel()
                        raise
        return WorkflowResult(step_results=results)


def _resolve_step_refs(value: Any, results: Mapping[str, Any]) -> Any:
    if isinstance(value, StepRef):
        try:
            return results[value.step]
        except KeyError as exc:
            raise SpecValidationError(f"Workflow step reference {value.step!r} is not available yet") from exc
    if isinstance(value, StepFeatureRef):
        try:
            artifact = results[value.step]
        except KeyError as exc:
            raise SpecValidationError(f"Workflow step reference {value.step!r} is not available yet") from exc
        feature = artifact.feature(value.feature_name)
        return feature.layer(value.layer_index) if value.layer_index is not None else feature
    if isinstance(value, StepLabelRef):
        try:
            artifact = results[value.step]
        except KeyError as exc:
            raise SpecValidationError(f"Workflow step reference {value.step!r} is not available yet") from exc
        return artifact.label(value.label_name)
    if dataclasses.is_dataclass(value):
        fields = {field.name: _resolve_step_refs(getattr(value, field.name), results) for field in dataclasses.fields(value)}
        return type(value)(**fields)
    if isinstance(value, tuple):
        return tuple(_resolve_step_refs(item, results) for item in value)
    if isinstance(value, list):
        return [_resolve_step_refs(item, results) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_step_refs(item, results) for key, item in value.items()}
    return value


def _workflow_section_metadata_errors(workflow: WorkflowSpec) -> dict[str, tuple[str, ...]]:
    step_by_name = {step.name: step for step in workflow.steps}
    errors: dict[str, tuple[str, ...]] = {}
    for step in workflow.steps:
        token_selector = getattr(step.spec, "tokens", None)
        feature_ref = getattr(step.spec, "feature", None)
        if not isinstance(token_selector, TokenSelector) or token_selector.kind != "section":
            continue
        if not isinstance(feature_ref, StepFeatureRef):
            continue
        source_step = step_by_name.get(feature_ref.step)
        if source_step is None or not isinstance(source_step.spec, CaptureSpec):
            continue
        if source_step.spec.provides_token_sections():
            continue
        errors[step.name] = (
            "Step uses TokenSelector.section(...), but upstream capture step "
            f"{source_step.name!r} does not define prompt_metadata_builder=... "
            "and does not carry explicit token_sections on every materialized example.",
        )
    return errors
