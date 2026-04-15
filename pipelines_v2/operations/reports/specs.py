"""Report-packaging specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence

from pipelines_v2.core.types import OperationSpec
from pipelines_v2.operations.common._shared import analysis_runtime_spec, spec_value_from_dict


@dataclass(frozen=True, slots=True)
class ReportSpec(OperationSpec):
    """Package existing step or artifact outputs into a report artifact."""

    inputs: Sequence[Any] = field(default_factory=tuple)
    template: str = "summary"
    output_dir: str | None = None

    kind: ClassVar[str] = "report"

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    def semantic_dict(self) -> dict[str, Any]:
        data = super(ReportSpec, self).semantic_dict()
        data.pop("output_dir", None)
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReportSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            inputs=tuple(spec_value_from_dict(value) for value in payload.get("inputs", ())),
            template=str(payload.get("template", "summary")),
            output_dir=payload.get("output_dir"),
        )


__all__ = ["ReportSpec"]
