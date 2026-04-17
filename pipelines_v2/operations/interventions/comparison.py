"""Artifact-bound comparison specs for intervention workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pipelines_v2.core.types import OperationSpec, RuntimeSecret, SpecValidationError
from pipelines_v2.operations.common._shared import (
    analysis_runtime_spec,
    merge_string_tuples,
    runtime_secrets_from_refs,
    spec_value_from_dict,
)


@dataclass(frozen=True, slots=True)
class PatchComparisonSpec(OperationSpec):
    """Artifact-bound comparison over baseline and named patched generation runs."""

    baseline: Any = None
    variants: Mapping[str, Any] = field(default_factory=dict)
    row_evaluator: Any = None

    kind: ClassVar[str] = "patch_comparison"

    def __post_init__(self) -> None:
        if self.baseline is None:
            raise SpecValidationError("PatchComparisonSpec requires baseline")
        normalized_variants = {
            str(name): value
            for name, value in dict(self.variants).items()
            if str(name).strip()
        }
        if not normalized_variants:
            raise SpecValidationError("PatchComparisonSpec requires at least one named variant")
        if self.row_evaluator is None:
            raise SpecValidationError("PatchComparisonSpec requires row_evaluator")
        object.__setattr__(self, "variants", normalized_variants)

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.baseline, self.variants)

    def runtime_spec(self) -> Any | None:
        runtime_spec = analysis_runtime_spec()
        from pipelines_v2.engine.base import PythonRuntimeSpec

        if not isinstance(runtime_spec, PythonRuntimeSpec):
            return runtime_spec
        local_python_sources = tuple(runtime_spec.local_python_sources)
        row_builder = self.row_evaluator
        if row_builder is not None and hasattr(row_builder, "local_python_sources"):
            local_python_sources = merge_string_tuples(
                local_python_sources,
                tuple(row_builder.local_python_sources),
            )
        return PythonRuntimeSpec(
            python_version=runtime_spec.python_version,
            pip_packages=runtime_spec.pip_packages,
            env=dict(runtime_spec.env),
            secrets=runtime_spec.secrets,
            local_python_sources=local_python_sources,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PatchComparisonSpec":
        from pipelines_v2.operations.common.builders import TransformBuilder

        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            baseline=spec_value_from_dict(payload.get("baseline")),
            variants={
                str(name): spec_value_from_dict(value)
                for name, value in dict(payload.get("variants", {})).items()
            },
            row_evaluator=(
                TransformBuilder.from_dict(dict(payload["row_evaluator"]))
                if payload.get("row_evaluator") is not None
                else None
            ),
        )


__all__ = ["PatchComparisonSpec"]
