"""Shared runner planning helpers."""

from __future__ import annotations

from pipelines_v2.core.types import OperationSpec
from pipelines_v2.operations.interventions import PatchedGenerationSpec
from pipelines_v2.operations.interventions.runtime import patched_generation_plan_errors


def spec_plan_errors(spec: OperationSpec, *, remote_assets: bool = False) -> list[str]:
    """Return operation-level planning errors common to runner backends."""

    from pipelines_v2.operations.registry import operation_registration

    errors: list[str] = []
    if operation_registration(spec) is None:
        errors.append(f"Unregistered operation kind: {spec.kind!r}")
    engine = spec.bound_engine()
    if engine is not None:
        planning_errors = getattr(engine, "planning_errors", None)
        if callable(planning_errors):
            if remote_assets and engine.identity().get('kind') in {'torch','torch_native'}:
                errors.extend(str(error) for error in planning_errors(spec, check_assets=False))
            else:
                errors.extend(str(error) for error in planning_errors(spec))
    if isinstance(spec, PatchedGenerationSpec):
        supported = getattr(engine, "supported_patch_operators", lambda: frozenset())()
        if spec.patch.operator not in supported:
            errors.append(f"Patch operator {spec.patch.operator!r} is not supported by this engine")
        errors.extend(patched_generation_plan_errors(spec))
    return errors


__all__ = ["spec_plan_errors"]
