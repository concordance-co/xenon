"""Typed source artifacts for intervention workloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence

from pipelines_v2.core.types import OperationSpec, RuntimeSecret, SpecValidationError
from pipelines_v2.operations.common._shared import (
    analysis_runtime_spec,
    runtime_secrets_from_refs,
    spec_value_from_dict,
)


@dataclass(frozen=True, slots=True)
class ActivationBankSpec(OperationSpec):
    """Materialize a residual feature into a compact donor activation bank."""

    feature: Any = None
    layers: Sequence[int] = field(default_factory=tuple)
    rows: Any = None

    kind: ClassVar[str] = "activation_bank"

    def __post_init__(self) -> None:
        if self.feature is None:
            raise SpecValidationError("ActivationBankSpec requires feature")

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.feature, self.rows)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActivationBankSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            rows=spec_value_from_dict(payload.get("rows")),
        )


@dataclass(frozen=True, slots=True)
class ExplicitPathEdge:
    """One weighted residual read->write edge."""

    source_layer: int
    write_layer: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_layer", int(self.source_layer))
        object.__setattr__(self, "write_layer", int(self.write_layer))
        object.__setattr__(self, "weight", float(self.weight))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExplicitPathEdge":
        return cls(
            source_layer=int(payload["source_layer"]),
            write_layer=int(payload["write_layer"]),
            weight=float(payload.get("weight", 1.0)),
        )


@dataclass(frozen=True, slots=True)
class ExplicitPathMaskSpec(OperationSpec):
    """Package a reusable explicit residual path mask artifact."""

    edges: tuple[ExplicitPathEdge, ...] = field(default_factory=tuple)

    kind: ClassVar[str] = "explicit_path_mask"

    def __post_init__(self) -> None:
        normalized = tuple(
            edge if isinstance(edge, ExplicitPathEdge) else ExplicitPathEdge.from_dict(dict(edge))
            for edge in self.edges
        )
        if not normalized:
            raise SpecValidationError("ExplicitPathMaskSpec requires at least one edge")
        object.__setattr__(self, "edges", normalized)

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return ()

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExplicitPathMaskSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            edges=tuple(ExplicitPathEdge.from_dict(dict(edge)) for edge in payload.get("edges", ())),
        )


__all__ = ["ActivationBankSpec", "ExplicitPathEdge", "ExplicitPathMaskSpec"]
