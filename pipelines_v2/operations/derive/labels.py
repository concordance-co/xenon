"""Derived label operation specs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Mapping

from pipelines_v2.core.types import OperationSpec, RuntimeSecret, SpecValidationError
from pipelines_v2.operations.common._shared import (
    analysis_runtime_spec,
    merge_string_tuples,
    runtime_pip_packages_from_refs,
    runtime_secrets_from_refs,
    spec_value_from_dict,
)


@dataclass(frozen=True, slots=True)
class LabelMapSpec(OperationSpec):
    """Remap one label vocabulary into a new derived label set."""

    source: Any = None
    mapping: Mapping[str, Any] = field(default_factory=dict)
    output_name: str = "mapped_label"
    strict: bool = True
    default_value: Any = None

    kind: ClassVar[str] = "label_map"

    def __post_init__(self) -> None:
        if not str(self.output_name).strip():
            raise SpecValidationError("LabelMapSpec output_name cannot be empty")

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.source)

    def runtime_spec(self) -> Any | None:
        runtime_spec = analysis_runtime_spec()
        packages = runtime_pip_packages_from_refs(self.source)
        if not packages:
            return runtime_spec
        return replace(runtime_spec, pip_packages=merge_string_tuples(runtime_spec.pip_packages, packages))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelMapSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            source=spec_value_from_dict(payload.get("source")),
            mapping={str(key): value for key, value in dict(payload.get("mapping", {})).items()},
            output_name=str(payload.get("output_name", "mapped_label")),
            strict=bool(payload.get("strict", True)),
            default_value=payload.get("default_value"),
        )


@dataclass(frozen=True, slots=True)
class LabelFieldsSpec(OperationSpec):
    """Extract named fields from a structured label payload."""

    source: Any = None
    fields: Mapping[str, str] = field(default_factory=dict)
    strict: bool = True

    kind: ClassVar[str] = "label_fields"

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.source)

    def runtime_spec(self) -> Any | None:
        runtime_spec = analysis_runtime_spec()
        packages = runtime_pip_packages_from_refs(self.source)
        if not packages:
            return runtime_spec
        return replace(runtime_spec, pip_packages=merge_string_tuples(runtime_spec.pip_packages, packages))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelFieldsSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            source=spec_value_from_dict(payload.get("source")),
            fields={str(key): str(value) for key, value in dict(payload.get("fields", {})).items()},
            strict=bool(payload.get("strict", True)),
        )


__all__ = ["LabelFieldsSpec", "LabelMapSpec"]
