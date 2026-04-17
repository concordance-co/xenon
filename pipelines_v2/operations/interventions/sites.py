"""Intervention site types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from pipelines_v2.core.types import SpecValidationError


@dataclass(frozen=True, slots=True)
class InterventionSite:
    """Abstract write location for an intervention family."""

    site: str
    layers: tuple[int, ...]

    kind: ClassVar[str] = "intervention_site"

    def __post_init__(self) -> None:
        site = str(self.site or "").strip()
        if not site:
            raise SpecValidationError("InterventionSite requires a non-empty site name")
        normalized_layers = tuple(int(layer) for layer in self.layers)
        if not normalized_layers:
            raise SpecValidationError("InterventionSite requires at least one layer")
        object.__setattr__(self, "site", site)
        object.__setattr__(self, "layers", normalized_layers)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InterventionSite":
        kind = str(payload.get("kind") or ResidualInterventionSite.kind)
        if kind == ResidualInterventionSite.kind:
            return ResidualInterventionSite.from_dict(payload)
        raise SpecValidationError(f"Unsupported intervention site kind: {kind!r}")


@dataclass(frozen=True, slots=True)
class ResidualInterventionSite(InterventionSite):
    """Prompt-side residual write location for activation patching."""

    kind: ClassVar[str] = "residual_intervention_site"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResidualInterventionSite":
        return cls(
            site=str(payload["site"]),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
        )


def site_from_payload(payload: Any) -> InterventionSite:
    if isinstance(payload, InterventionSite):
        return payload
    if isinstance(payload, Mapping):
        return InterventionSite.from_dict(payload)
    raise SpecValidationError(f"Expected intervention site mapping, got {type(payload).__name__}")


__all__ = ["InterventionSite", "ResidualInterventionSite", "site_from_payload"]
