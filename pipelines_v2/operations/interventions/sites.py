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
        loader = _INTERVENTION_SITE_LOADERS.get(kind)
        if loader is None:
            raise SpecValidationError(f"Unsupported intervention site kind: {kind!r}")
        if kind != ResidualInterventionSite.kind and (
            type(payload.get("schema_version", 1)) is not int or payload.get("schema_version", 1) != 1
        ):
            raise SpecValidationError(f"Unsupported intervention site schema_version: {payload.get('schema_version')!r}")
        site = loader(payload)
        if not isinstance(site, InterventionSite) or site.kind != kind:
            raise SpecValidationError("Intervention site loader returned an incompatible kind")
        return site


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


_INTERVENTION_SITE_LOADERS: dict[str, Any] = {
    ResidualInterventionSite.kind: ResidualInterventionSite.from_dict,
}


def register_intervention_site(kind: str, loader: Any) -> None:
    """Register a write-site codec without changing residual legacy behavior."""
    from pipelines_v2.core.registry import register_kind

    if not callable(loader):
        raise TypeError("Intervention site loader must be callable")
    register_kind(_INTERVENTION_SITE_LOADERS, kind, loader, label="Intervention site")


def site_from_payload(payload: Any) -> InterventionSite:
    if isinstance(payload, InterventionSite):
        return payload
    if isinstance(payload, Mapping):
        return InterventionSite.from_dict(payload)
    raise SpecValidationError(f"Expected intervention site mapping, got {type(payload).__name__}")


__all__ = ["InterventionSite", "ResidualInterventionSite", "site_from_payload", "register_intervention_site"]
