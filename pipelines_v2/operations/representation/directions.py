"""Direction-finding specs over captured features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence

from pipelines_v2.core.types import OperationSpec, RuntimeSecret
from pipelines_v2.operations.common._shared import analysis_runtime_spec, runtime_secrets_from_refs, spec_value_from_dict
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector


@dataclass(frozen=True, slots=True)
class DirectionSpec(OperationSpec):
    """Compute a direction from positive and negative example groups."""

    feature: Any = None
    positive: Any = None
    negative: Any = None
    group_by: Any = None
    layers: Sequence[int] = field(default_factory=tuple)
    tokens: TokenSelector = field(default_factory=TokenSelector.last)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)

    kind: ClassVar[str] = "direction"

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.positive, self.negative, self.group_by)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DirectionSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            positive=spec_value_from_dict(payload.get("positive")),
            negative=spec_value_from_dict(payload.get("negative")),
            group_by=spec_value_from_dict(payload.get("group_by")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "last"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
        )


__all__ = ["DirectionSpec"]
