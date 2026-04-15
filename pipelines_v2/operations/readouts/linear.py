"""Linear readout specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence

from pipelines_v2.core.types import OperationSpec, RuntimeSecret
from pipelines_v2.operations.common._shared import (
    analysis_runtime_spec,
    row_selector_from_dict,
    runtime_secrets_from_refs,
    spec_value_from_dict,
)
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector


@dataclass(frozen=True, slots=True)
class ProbeSpec(OperationSpec):
    """Train and evaluate a linear probe over one captured feature family."""

    feature: Any = None
    rows: Any = None
    labels: Any = None
    group_by: Any = None
    split: Any = None
    tokens: TokenSelector = field(default_factory=TokenSelector.full_sequence)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)
    train_values: Sequence[Any] = field(default_factory=lambda: ("train",))
    test_values: Sequence[Any] = field(default_factory=lambda: ("test",))
    folds: int = 5
    baselines: Sequence[str] = field(default_factory=tuple)
    metrics: Sequence[str] = field(default_factory=tuple)
    persist_predictions: bool = False

    kind: ClassVar[str] = "probe"

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.rows, self.labels, self.group_by, self.split)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProbeSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            rows=row_selector_from_dict(payload.get("rows")),
            labels=spec_value_from_dict(payload.get("labels")),
            group_by=spec_value_from_dict(payload.get("group_by")),
            split=spec_value_from_dict(payload.get("split")),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "full_sequence"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
            train_values=tuple(payload.get("train_values", ("train",))),
            test_values=tuple(payload.get("test_values", ("test",))),
            folds=int(payload.get("folds", 5)),
            baselines=tuple(str(item) for item in payload.get("baselines", ())),
            metrics=tuple(str(item) for item in payload.get("metrics", ())),
            persist_predictions=bool(payload.get("persist_predictions", False)),
        )


__all__ = ["ProbeSpec"]
