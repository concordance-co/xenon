"""Centroid-learning specs for intervention-friendly representation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence

from pipelines_v2.core.types import OperationSpec, RuntimeSecret, SpecValidationError
from pipelines_v2.operations.common._shared import (
    analysis_runtime_spec,
    runtime_secrets_from_refs,
    spec_value_from_dict,
)
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector


@dataclass(frozen=True, slots=True)
class CentroidSpec(OperationSpec):
    """Compute raw residual centroids keyed by a label/group value."""

    feature: Any = None
    by: Any = None
    layers: Sequence[int] = field(default_factory=tuple)
    rows: Any = None
    tokens: TokenSelector = field(default_factory=TokenSelector.last)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)
    subspace: Any = None

    kind: ClassVar[str] = "centroid"

    def __post_init__(self) -> None:
        if self.feature is None:
            raise SpecValidationError("CentroidSpec requires feature")
        if self.by is None:
            raise SpecValidationError("CentroidSpec requires by")

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.by, self.rows, self.subspace)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CentroidSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            by=spec_value_from_dict(payload.get("by")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            rows=spec_value_from_dict(payload.get("rows")),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "last"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
            subspace=spec_value_from_dict(payload.get("subspace")),
        )


__all__ = ["CentroidSpec"]
