"""Derived feature contrast operation specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

from pipelines_v2.core.types import OperationSpec, RuntimeSecret, SpecValidationError
from pipelines_v2.operations.common._shared import analysis_runtime_spec, runtime_secrets_from_refs, spec_value_from_dict
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector


@dataclass(frozen=True, slots=True)
class PairDeltaSpec(OperationSpec):
    """Compute paired positive-minus-negative deltas and emit a derived feature."""

    feature: Any = None
    case: Any = None
    positive: Any = None
    negative: Any = None
    layers: Sequence[int] = field(default_factory=tuple)
    tokens: TokenSelector = field(default_factory=TokenSelector.last)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)
    output_feature_name: str = "delta"
    labels: Mapping[str, Any] = field(default_factory=dict)
    propagate_from: str = "positive"

    kind: ClassVar[str] = "pair_delta"

    def __post_init__(self) -> None:
        if self.propagate_from not in {"positive", "negative"}:
            raise SpecValidationError("PairDeltaSpec propagate_from must be 'positive' or 'negative'")
        if not str(self.output_feature_name).strip():
            raise SpecValidationError("PairDeltaSpec output_feature_name cannot be empty")

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.feature, self.case, self.positive, self.negative, self.labels)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PairDeltaSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            case=spec_value_from_dict(payload.get("case")),
            positive=spec_value_from_dict(payload.get("positive")),
            negative=spec_value_from_dict(payload.get("negative")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "last"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
            output_feature_name=str(payload.get("output_feature_name", "delta")),
            labels={str(name): spec_value_from_dict(value) for name, value in dict(payload.get("labels", {})).items()},
            propagate_from=str(payload.get("propagate_from", "positive")),
        )


__all__ = ["PairDeltaSpec"]
