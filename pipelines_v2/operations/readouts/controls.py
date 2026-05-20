"""Controlled and residualized readout specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence

from pipelines_v2.core.types import OperationSpec, RuntimeSecret
from pipelines_v2.operations.common._shared import (
    analysis_runtime_spec_for_refs,
    row_selector_from_dict,
    runtime_secrets_from_refs,
    spec_value_from_dict,
)
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector


@dataclass(frozen=True, slots=True)
class ResidualizedProbeSpec(OperationSpec):
    """Probe a target after projecting out a nuisance linear subspace."""

    feature: Any = None
    rows: Any = None
    labels: Any = None
    residualize_against: Any = None
    group_by: Any | None = None
    metrics: Sequence[str] = field(default_factory=lambda: ("balanced_accuracy", "auroc"))
    tokens: TokenSelector = field(default_factory=TokenSelector.full_sequence)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)

    kind: ClassVar[str] = "residualized_probe"

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.feature, self.rows, self.labels, self.residualize_against, self.group_by)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec_for_refs(self.feature, self.rows, self.labels, self.residualize_against, self.group_by)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResidualizedProbeSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            rows=row_selector_from_dict(payload.get("rows")),
            labels=spec_value_from_dict(payload.get("labels")),
            residualize_against=spec_value_from_dict(payload.get("residualize_against")),
            group_by=spec_value_from_dict(payload.get("group_by")),
            metrics=tuple(str(item) for item in payload.get("metrics", ("balanced_accuracy", "auroc"))),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "full_sequence"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
        )


__all__ = ["ResidualizedProbeSpec"]
