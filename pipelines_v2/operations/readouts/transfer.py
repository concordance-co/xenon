"""Transfer-readout operation specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

from pipelines_v2.core.types import OperationSpec, RuntimeSecret
from pipelines_v2.operations.common._shared import (
    analysis_runtime_spec_for_refs,
    row_selector_from_dict,
    runtime_secrets_from_refs,
    spec_value_from_dict,
)
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector


@dataclass(frozen=True, slots=True)
class TransferProbeSpec(OperationSpec):
    """Train within one cohort and evaluate transfer to another cohort."""

    feature: Any = None
    rows: Any = None
    labels: Any = None
    group_by: Any | None = None
    cohort_by: Any | None = None
    cohort_values: Sequence[Any] = field(default_factory=tuple)
    split_by: Mapping[str, Any] = field(default_factory=dict)
    train_values: Sequence[Any] = field(default_factory=tuple)
    test_values: Sequence[Any] = field(default_factory=tuple)
    regularization: Sequence[float] = field(default_factory=tuple)
    metrics: Sequence[str] = field(default_factory=lambda: ("balanced_accuracy", "auroc"))
    baselines: Sequence[str] = field(default_factory=tuple)
    compare_within_baseline: bool = True
    compare_direction_similarity: bool = True
    tokens: TokenSelector = field(default_factory=TokenSelector.full_sequence)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)
    persist_predictions: bool = False

    kind: ClassVar[str] = "transfer_probe"

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.feature, self.rows, self.labels, self.group_by, self.cohort_by, self.split_by)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec_for_refs(self.feature, self.rows, self.labels, self.group_by, self.cohort_by, self.split_by)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransferProbeSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            rows=row_selector_from_dict(payload.get("rows")),
            labels=spec_value_from_dict(payload.get("labels")),
            group_by=spec_value_from_dict(payload.get("group_by")),
            cohort_by=spec_value_from_dict(payload.get("cohort_by")),
            cohort_values=tuple(payload.get("cohort_values", ())),
            split_by={str(key): spec_value_from_dict(value) for key, value in dict(payload.get("split_by", {})).items()},
            train_values=tuple(payload.get("train_values", ())),
            test_values=tuple(payload.get("test_values", ())),
            regularization=tuple(float(value) for value in payload.get("regularization", ())),
            metrics=tuple(str(item) for item in payload.get("metrics", ("balanced_accuracy", "auroc"))),
            baselines=tuple(str(item) for item in payload.get("baselines", ())),
            compare_within_baseline=bool(payload.get("compare_within_baseline", True)),
            compare_direction_similarity=bool(payload.get("compare_direction_similarity", True)),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "full_sequence"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
            persist_predictions=bool(payload.get("persist_predictions", False)),
        )


__all__ = ["TransferProbeSpec"]
