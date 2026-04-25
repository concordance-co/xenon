"""Shared runner planning helpers."""

from __future__ import annotations

from pipelines_v2.core.types import OperationSpec
from pipelines_v2.operations.interventions import PatchedGenerationSpec
from pipelines_v2.operations.interventions.runtime import patched_generation_plan_errors


def spec_plan_errors(spec: OperationSpec) -> list[str]:
    """Return operation-level planning errors common to runner backends."""

    errors: list[str] = []
    engine = spec.bound_engine()
    if engine is not None:
        planning_errors = getattr(engine, "planning_errors", None)
        if callable(planning_errors):
            errors.extend(str(error) for error in planning_errors(spec))
    if isinstance(spec, PatchedGenerationSpec):
        errors.extend(patched_generation_plan_errors(spec))
    return errors


__all__ = ["spec_plan_errors"]
