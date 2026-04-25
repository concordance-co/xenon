"""Structured projection scoring specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence

from pipelines_v2.core.types import OperationSpec, RuntimeSecret
from .slices import SectionSelector
from pipelines_v2.operations.common._shared import analysis_runtime_spec, runtime_secrets_from_refs, spec_value_from_dict
from pipelines_v2.operations.common.tokens import TokenPooling


@dataclass(frozen=True, slots=True)
class ProjectionSpec(OperationSpec):
    """Score repeated semantic slices against one or more coordinates."""

    feature: Any = None
    coordinates: Sequence[Any] = field(default_factory=tuple)
    slices: SectionSelector = field(default_factory=SectionSelector.all)
    rows: Any = None
    layers: Sequence[int] = field(default_factory=tuple)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)
    metric: str = "signed_dot"
    summaries: Sequence[str] = field(default_factory=tuple)
    emit_labels: bool = False

    kind: ClassVar[str] = "projection"

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.feature, self.coordinates, self.rows)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProjectionSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            coordinates=tuple(spec_value_from_dict(item) for item in payload.get("coordinates", ())),
            slices=SectionSelector.from_dict(payload.get("slices")),
            rows=spec_value_from_dict(payload.get("rows")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
            metric=str(payload.get("metric", "signed_dot")),
            summaries=tuple(str(item) for item in payload.get("summaries", ())),
            emit_labels=bool(payload.get("emit_labels", False)),
        )


__all__ = ["ProjectionSpec"]
