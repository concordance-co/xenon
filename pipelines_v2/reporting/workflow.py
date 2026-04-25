"""Workflow helpers used by report surfaces."""

from __future__ import annotations

from pipelines_v2.operations.reports import ReportSpec
from pipelines_v2.workflow.specs import WorkflowSpec, WorkflowStep


def resolve_report_step(
    workflow: WorkflowSpec,
    *,
    step_name: str | None,
    selector_label: str = "step_name",
) -> WorkflowStep:
    """Resolve the report step from a workflow, with clear ambiguity errors."""

    report_steps = [step for step in workflow.ordered_steps() if isinstance(step.spec, ReportSpec)]
    if not report_steps:
        raise RuntimeError("Workflow run does not contain any report steps")
    if step_name is None:
        if len(report_steps) == 1:
            return report_steps[0]
        names = [step.name for step in report_steps]
        raise RuntimeError(f"Workflow run contains multiple report steps; choose one with {selector_label}: {names}")
    for step in report_steps:
        if step.name == step_name:
            return step
    raise RuntimeError(f"Workflow run does not contain report step {step_name!r}")


__all__ = ["resolve_report_step"]
