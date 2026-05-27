"""Capture-site specs for model-bound activation collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from pipelines_v2.core.types import EngineCapability
from pipelines_v2.operations.common.schemas import TensorStorage
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector


@dataclass(frozen=True, slots=True)
class ResidualSite:
    """Capture request for residual activations at one named site."""

    name: str
    site: str
    layers: Sequence[int]
    tokens: TokenSelector = field(default_factory=TokenSelector.last)
    pooling: TokenPooling | None = None
    storage: TensorStorage = field(default_factory=TensorStorage)

    def required_capabilities(self) -> set[EngineCapability]:
        return {EngineCapability.RESIDUAL_CAPTURE}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResidualSite":
        return cls(
            name=str(payload["name"]),
            site=str(payload["site"]),
            layers=[int(layer) for layer in payload.get("layers", ())],
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "last"})),
            pooling=(
                TokenPooling.from_dict(dict(payload["pooling"]))
                if payload.get("pooling") is not None
                else None
            ),
            storage=TensorStorage.from_dict(payload.get("storage", {})),
        )


@dataclass(frozen=True, slots=True)
class RoutingRecord:
    """One requested MoE router output family."""

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def gate_logits(cls, *, dtype: str = "float16") -> "RoutingRecord":
        return cls(kind="gate_logits", params={"dtype": dtype})

    @classmethod
    def gate_probs(cls, *, dtype: str = "float16") -> "RoutingRecord":
        return cls(kind="gate_probs", params={"dtype": dtype})

    @classmethod
    def routing_decisions(cls, *, required: bool = True) -> "RoutingRecord":
        return cls(kind="routing_decisions", params={"required": required})

    @classmethod
    def topk_from_gate(cls, *, k: int, include_weights: bool = True) -> "RoutingRecord":
        return cls(kind="topk_from_gate", params={"k": k, "include_weights": include_weights})

    @classmethod
    def expert_load(cls, *, source: str) -> "RoutingRecord":
        return cls(kind="expert_load", params={"source": source})

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoutingRecord":
        return cls(kind=str(payload["kind"]), params=dict(payload.get("params", {})))


@dataclass(frozen=True, slots=True)
class MoERoutingSite:
    """Capture request for MoE router outputs across layers and tokens."""

    name: str
    layers: Sequence[int]
    tokens: TokenSelector = field(default_factory=TokenSelector.last)
    record: Sequence[RoutingRecord] = field(default_factory=lambda: (RoutingRecord.gate_logits(),))

    def required_capabilities(self) -> set[EngineCapability]:
        return {EngineCapability.MOE_ROUTING_CAPTURE}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MoERoutingSite":
        return cls(
            name=str(payload["name"]),
            layers=[int(layer) for layer in payload.get("layers", ())],
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "last"})),
            record=[RoutingRecord.from_dict(record) for record in payload.get("record", ())],
        )


CaptureSite = ResidualSite | MoERoutingSite

__all__ = [
    "CaptureSite",
    "MoERoutingSite",
    "ResidualSite",
    "RoutingRecord",
]
