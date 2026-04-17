"""Subspace-learning specs for intervention-friendly representation artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

from pipelines_v2.core.types import OperationSpec, RuntimeSecret
from pipelines_v2.operations.common._shared import analysis_runtime_spec, runtime_secrets_from_refs, spec_value_from_dict
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector


@dataclass(frozen=True, slots=True)
class SubspaceSpec(OperationSpec):
    """Learn a standardized PCA subspace intended for activation patching."""

    feature: Any = None
    layers: Sequence[int] = field(default_factory=tuple)
    components: int = 8
    tokens: TokenSelector = field(default_factory=TokenSelector.last)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)
    named_components_by_layer: Mapping[int, Mapping[str, int]] = field(default_factory=dict)

    kind: ClassVar[str] = "subspace"

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.feature)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SubspaceSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            components=int(payload.get("components", 8)),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "last"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
            named_components_by_layer={
                int(layer): {
                    str(name): int(index)
                    for name, index in dict(named).items()
                }
                for layer, named in dict(payload.get("named_components_by_layer", {})).items()
            },
        )


__all__ = ["SubspaceSpec"]
