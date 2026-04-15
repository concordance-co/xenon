"""Representation decomposition specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence

from pipelines_v2.core.types import OperationSpec, RuntimeSecret
from pipelines_v2.operations.common._shared import analysis_runtime_spec, runtime_secrets_from_refs, spec_value_from_dict
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector


@dataclass(frozen=True, slots=True)
class BasisSpec(OperationSpec):
    """Learn a basis over captured features, currently PCA only."""

    feature: Any = None
    method: str = "pca"
    by: Any = None
    layers: Sequence[int] = field(default_factory=tuple)
    components: int = 8
    tokens: TokenSelector = field(default_factory=TokenSelector.last)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)

    kind: ClassVar[str] = "basis"

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.by)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BasisSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            method=str(payload.get("method", "pca")),
            by=spec_value_from_dict(payload.get("by")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            components=int(payload.get("components", 8)),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "last"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
        )


__all__ = ["BasisSpec"]
