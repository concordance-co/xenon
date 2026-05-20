"""Geometry and projection analysis specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

from pipelines_v2.core.types import OperationSpec, RuntimeSecret
from pipelines_v2.operations.common._shared import (
    analysis_runtime_spec,
    row_selector_from_dict,
    runtime_secrets_from_refs,
    spec_value_from_dict,
)
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector


@dataclass(frozen=True, slots=True)
class GeometrySpec(OperationSpec):
    """Project activations into a low-dimensional geometry view."""

    feature: Any = None
    rows: Any = None
    method: str = "pca"
    layers: Sequence[int] = field(default_factory=tuple)
    label: Any | None = None
    color_by: Mapping[str, Any] = field(default_factory=dict)
    subset: Any | None = None
    normalize: str | None = "rms_per_row"
    components: int = 2
    tokens: TokenSelector = field(default_factory=TokenSelector.full_sequence)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)

    kind: ClassVar[str] = "geometry"

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.feature, self.rows, self.label, self.color_by, self.subset)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GeometrySpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            rows=row_selector_from_dict(payload.get("rows")),
            method=str(payload.get("method", "pca")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            label=spec_value_from_dict(payload.get("label")),
            color_by={str(key): spec_value_from_dict(value) for key, value in dict(payload.get("color_by", {})).items()},
            subset=spec_value_from_dict(payload.get("subset")),
            normalize=payload.get("normalize"),
            components=int(payload.get("components", 2)),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "full_sequence"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
        )


__all__ = ["GeometrySpec"]
