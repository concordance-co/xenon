"""General escape-hatch transform specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping

from pipelines_v2.core.types import OperationSpec, SpecValidationError
from pipelines_v2.operations.common._shared import (
    analysis_runtime_spec_for_refs,
    runtime_secrets_from_refs,
    spec_value_from_dict,
)
from pipelines_v2.operations.common.builders import TransformBuilder


@dataclass(frozen=True, slots=True)
class TransformSpec(OperationSpec):
    """Run a user-defined transform function over named runtime inputs."""

    builder: TransformBuilder | None = None
    inputs: Mapping[str, Any] = field(default_factory=dict)
    inline: bool = False

    kind: ClassVar[str] = "transform"

    def __post_init__(self) -> None:
        if self.builder is None:
            raise SpecValidationError("TransformSpec requires a builder")
        if not self.inputs:
            raise SpecValidationError("TransformSpec requires at least one named input")
        empty = sorted(name for name in self.inputs if not str(name).strip())
        if empty:
            raise SpecValidationError(f"TransformSpec input names cannot be empty: {empty}")

    def runtime_secrets(self) -> tuple[Any, ...]:
        return runtime_secrets_from_refs(self.inputs)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec_for_refs(
            self.inputs,
            local_python_sources=self.builder.local_python_sources if self.builder is not None else (),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransformSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            builder=TransformBuilder.from_dict(dict(payload["builder"])),
            inputs={str(key): spec_value_from_dict(value) for key, value in dict(payload.get("inputs", {})).items()},
            inline=bool(payload.get("inline", False)),
        )


__all__ = ["TransformSpec"]
