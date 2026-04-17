"""Activation patch recipe types."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

from pipelines_v2.core.types import RuntimeSecret, SpecValidationError, to_primitive, to_semantic_primitive
from pipelines_v2.operations.common._shared import runtime_secrets_from_refs, spec_value_from_dict
from pipelines_v2.operations.common.tokens import TokenSelector

from ._shared import (
    normalize_component_map,
    normalize_layer_map,
    token_selector_from_payload,
)
from .sites import InterventionSite, ResidualInterventionSite, site_from_payload


@dataclass(frozen=True, slots=True)
class ActivationPatchSpec:
    """Abstract residual intervention recipe."""

    write_site: InterventionSite = field(
        default_factory=lambda: ResidualInterventionSite(site="resid_post", layers=(0,))
    )
    target_tokens: TokenSelector = field(default_factory=TokenSelector.last)
    source_layer_map: Mapping[int, int] = field(default_factory=dict)
    strength: float = 1.0

    kind: ClassVar[str] = "activation_patch"
    operator_name: ClassVar[str] = "activation_patch"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_layer_map", normalize_layer_map(self.source_layer_map))
        object.__setattr__(self, "strength", float(self.strength))

    @property
    def operator(self) -> str:
        return str(type(self).operator_name)

    def to_dict(self) -> dict[str, Any]:
        payload = to_primitive(self)
        if not isinstance(payload, dict):
            raise TypeError("ActivationPatchSpec serializer returned non-dict data")
        payload["kind"] = self.kind
        return payload

    def semantic_dict(self) -> dict[str, Any]:
        payload = {
            field.name: to_semantic_primitive(getattr(self, field.name))
            for field in self.__dataclass_fields__.values()
        }
        payload["kind"] = self.kind
        return payload

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return ()

    def source_layer_for(self, write_layer: int) -> int:
        return int(self.source_layer_map.get(int(write_layer), int(write_layer)))

    def component_indices_for(self, write_layer: int) -> tuple[int, ...]:
        return ()

    def requires_pairing(self) -> bool:
        return False

    def uses_activation_bank(self) -> bool:
        return False

    def uses_subspace(self) -> bool:
        return False

    def uses_direction(self) -> bool:
        return False

    def uses_centroid(self) -> bool:
        return False

    def uses_path_mask(self) -> bool:
        return False

    def is_interchange(self) -> bool:
        return False

    def is_residual_path(self) -> bool:
        return False

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActivationPatchSpec":
        kind = str(payload.get("kind") or "").strip()
        loader = PATCH_LOADERS.get(kind)
        if loader is None:
            raise SpecValidationError(f"Unsupported activation patch kind: {kind!r}")
        return loader(payload)


@dataclass(frozen=True, slots=True)
class InterchangePatch(ActivationPatchSpec):
    """Residual donor-row interchange from an activation bank."""

    activation_bank: Any = None
    donor_tokens: TokenSelector | None = None

    kind: ClassVar[str] = "interchange_patch"
    operator_name: ClassVar[str] = "interchange"

    def __post_init__(self) -> None:
        ActivationPatchSpec.__post_init__(self)
        if self.activation_bank is None:
            raise SpecValidationError("InterchangePatch requires activation_bank")

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.activation_bank)

    def requires_pairing(self) -> bool:
        return True

    def uses_activation_bank(self) -> bool:
        return True

    def is_interchange(self) -> bool:
        return True

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InterchangePatch":
        return cls(
            write_site=site_from_payload(payload.get("write_site", {})),
            target_tokens=token_selector_from_payload(payload.get("target_tokens"), default=TokenSelector.last()) or TokenSelector.last(),
            source_layer_map=normalize_layer_map(payload.get("source_layer_map")),
            strength=float(payload.get("strength", 1.0)),
            activation_bank=spec_value_from_dict(payload.get("activation_bank")),
            donor_tokens=token_selector_from_payload(payload.get("donor_tokens")),
        )


@dataclass(frozen=True, slots=True)
class ProjectOutPatch(ActivationPatchSpec):
    """Project activations out of a subspace."""

    subspace: Any = None
    component_indices_by_layer: Mapping[int, tuple[int, ...]] = field(default_factory=dict)

    kind: ClassVar[str] = "project_out_patch"
    operator_name: ClassVar[str] = "project_out"

    def __post_init__(self) -> None:
        ActivationPatchSpec.__post_init__(self)
        object.__setattr__(self, "component_indices_by_layer", normalize_component_map(self.component_indices_by_layer))
        if self.subspace is None:
            raise SpecValidationError("ProjectOutPatch requires subspace")

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.subspace)

    def uses_subspace(self) -> bool:
        return True

    def component_indices_for(self, write_layer: int) -> tuple[int, ...]:
        return tuple(self.component_indices_by_layer.get(int(write_layer), ()))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectOutPatch":
        return cls(
            write_site=site_from_payload(payload.get("write_site", {})),
            target_tokens=token_selector_from_payload(payload.get("target_tokens"), default=TokenSelector.last()) or TokenSelector.last(),
            source_layer_map=normalize_layer_map(payload.get("source_layer_map")),
            strength=float(payload.get("strength", 1.0)),
            subspace=spec_value_from_dict(payload.get("subspace")),
            component_indices_by_layer=normalize_component_map(payload.get("component_indices_by_layer")),
        )


@dataclass(frozen=True, slots=True)
class AddDirectionPatch(ActivationPatchSpec):
    """Add a learned direction in raw or standardized coordinates."""

    direction: Any = None
    subspace: Any = None
    component_indices_by_layer: Mapping[int, tuple[int, ...]] = field(default_factory=dict)

    kind: ClassVar[str] = "add_direction_patch"
    operator_name: ClassVar[str] = "add_direction"

    def __post_init__(self) -> None:
        ActivationPatchSpec.__post_init__(self)
        object.__setattr__(self, "component_indices_by_layer", normalize_component_map(self.component_indices_by_layer))
        if self.direction is None:
            raise SpecValidationError("AddDirectionPatch requires direction")

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.direction, self.subspace)

    def uses_direction(self) -> bool:
        return True

    def uses_subspace(self) -> bool:
        return self.subspace is not None

    def component_indices_for(self, write_layer: int) -> tuple[int, ...]:
        return tuple(self.component_indices_by_layer.get(int(write_layer), ()))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AddDirectionPatch":
        return cls(
            write_site=site_from_payload(payload.get("write_site", {})),
            target_tokens=token_selector_from_payload(payload.get("target_tokens"), default=TokenSelector.last()) or TokenSelector.last(),
            source_layer_map=normalize_layer_map(payload.get("source_layer_map")),
            strength=float(payload.get("strength", 1.0)),
            direction=spec_value_from_dict(payload.get("direction")),
            subspace=spec_value_from_dict(payload.get("subspace")),
            component_indices_by_layer=normalize_component_map(payload.get("component_indices_by_layer")),
        )


@dataclass(frozen=True, slots=True)
class SwapMeanPatch(ActivationPatchSpec):
    """Move the section mean toward a named centroid."""

    centroids: Any = None
    centroid_name: str = ""

    kind: ClassVar[str] = "swap_mean_patch"
    operator_name: ClassVar[str] = "swap_mean"

    def __post_init__(self) -> None:
        ActivationPatchSpec.__post_init__(self)
        if self.centroids is None:
            raise SpecValidationError("SwapMeanPatch requires centroids")
        centroid_name = str(self.centroid_name or "").strip()
        if not centroid_name:
            raise SpecValidationError("SwapMeanPatch requires centroid_name")
        object.__setattr__(self, "centroid_name", centroid_name)

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.centroids)

    def uses_centroid(self) -> bool:
        return True

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SwapMeanPatch":
        return cls(
            write_site=site_from_payload(payload.get("write_site", {})),
            target_tokens=token_selector_from_payload(payload.get("target_tokens"), default=TokenSelector.last()) or TokenSelector.last(),
            source_layer_map=normalize_layer_map(payload.get("source_layer_map")),
            strength=float(payload.get("strength", 1.0)),
            centroids=spec_value_from_dict(payload.get("centroids")),
            centroid_name=str(payload.get("centroid_name", "")),
        )


@dataclass(frozen=True, slots=True)
class SwapComponentsPatch(ActivationPatchSpec):
    """Replace selected subspace coordinates with those from a named centroid."""

    subspace: Any = None
    centroids: Any = None
    centroid_name: str = ""
    component_indices_by_layer: Mapping[int, tuple[int, ...]] = field(default_factory=dict)

    kind: ClassVar[str] = "swap_components_patch"
    operator_name: ClassVar[str] = "swap_components"

    def __post_init__(self) -> None:
        ActivationPatchSpec.__post_init__(self)
        object.__setattr__(self, "component_indices_by_layer", normalize_component_map(self.component_indices_by_layer))
        if self.subspace is None:
            raise SpecValidationError("SwapComponentsPatch requires subspace")
        if self.centroids is None:
            raise SpecValidationError("SwapComponentsPatch requires centroids")
        centroid_name = str(self.centroid_name or "").strip()
        if not centroid_name:
            raise SpecValidationError("SwapComponentsPatch requires centroid_name")
        object.__setattr__(self, "centroid_name", centroid_name)

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.subspace, self.centroids)

    def uses_subspace(self) -> bool:
        return True

    def uses_centroid(self) -> bool:
        return True

    def component_indices_for(self, write_layer: int) -> tuple[int, ...]:
        return tuple(self.component_indices_by_layer.get(int(write_layer), ()))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SwapComponentsPatch":
        return cls(
            write_site=site_from_payload(payload.get("write_site", {})),
            target_tokens=token_selector_from_payload(payload.get("target_tokens"), default=TokenSelector.last()) or TokenSelector.last(),
            source_layer_map=normalize_layer_map(payload.get("source_layer_map")),
            strength=float(payload.get("strength", 1.0)),
            subspace=spec_value_from_dict(payload.get("subspace")),
            centroids=spec_value_from_dict(payload.get("centroids")),
            centroid_name=str(payload.get("centroid_name", "")),
            component_indices_by_layer=normalize_component_map(payload.get("component_indices_by_layer")),
        )


@dataclass(frozen=True, slots=True)
class RandomControlPatch(ActivationPatchSpec):
    """Project out a random matched-norm control direction."""

    subspace: Any = None
    component_indices_by_layer: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    random_seed: int = 0
    match_projected_norm: bool = True

    kind: ClassVar[str] = "random_control_patch"
    operator_name: ClassVar[str] = "random_control"

    def __post_init__(self) -> None:
        ActivationPatchSpec.__post_init__(self)
        object.__setattr__(self, "component_indices_by_layer", normalize_component_map(self.component_indices_by_layer))
        object.__setattr__(self, "random_seed", int(self.random_seed))
        object.__setattr__(self, "match_projected_norm", bool(self.match_projected_norm))
        if self.subspace is None:
            raise SpecValidationError("RandomControlPatch requires subspace")

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.subspace)

    def uses_subspace(self) -> bool:
        return True

    def component_indices_for(self, write_layer: int) -> tuple[int, ...]:
        return tuple(self.component_indices_by_layer.get(int(write_layer), ()))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RandomControlPatch":
        return cls(
            write_site=site_from_payload(payload.get("write_site", {})),
            target_tokens=token_selector_from_payload(payload.get("target_tokens"), default=TokenSelector.last()) or TokenSelector.last(),
            source_layer_map=normalize_layer_map(payload.get("source_layer_map")),
            strength=float(payload.get("strength", 1.0)),
            subspace=spec_value_from_dict(payload.get("subspace")),
            component_indices_by_layer=normalize_component_map(payload.get("component_indices_by_layer")),
            random_seed=int(payload.get("random_seed", 0)),
            match_projected_norm=bool(payload.get("match_projected_norm", True)),
        )


@dataclass(frozen=True, slots=True)
class ResidualPathPatch(ActivationPatchSpec):
    """Residual read/write path transport driven by an explicit path mask."""

    activation_bank: Any = None
    path_mask: Any = None
    read_tokens: TokenSelector | None = None
    transport: str = "replace"

    kind: ClassVar[str] = "residual_path_patch"
    operator_name: ClassVar[str] = "residual_path"

    def __post_init__(self) -> None:
        ActivationPatchSpec.__post_init__(self)
        transport = str(self.transport or "").strip().lower()
        if transport not in {"replace", "delta"}:
            raise SpecValidationError("ResidualPathPatch transport must be one of {'replace', 'delta'}")
        object.__setattr__(self, "transport", transport)
        if self.activation_bank is None:
            raise SpecValidationError("ResidualPathPatch requires activation_bank")
        if self.path_mask is None:
            raise SpecValidationError("ResidualPathPatch requires path_mask")

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.activation_bank, self.path_mask)

    def requires_pairing(self) -> bool:
        return True

    def uses_activation_bank(self) -> bool:
        return True

    def uses_path_mask(self) -> bool:
        return True

    def is_residual_path(self) -> bool:
        return True

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResidualPathPatch":
        return cls(
            write_site=site_from_payload(payload.get("write_site", {})),
            target_tokens=token_selector_from_payload(payload.get("target_tokens"), default=TokenSelector.last()) or TokenSelector.last(),
            source_layer_map=normalize_layer_map(payload.get("source_layer_map")),
            strength=float(payload.get("strength", 1.0)),
            activation_bank=spec_value_from_dict(payload.get("activation_bank")),
            path_mask=spec_value_from_dict(payload.get("path_mask")),
            read_tokens=token_selector_from_payload(payload.get("read_tokens")),
            transport=str(payload.get("transport", "replace")),
        )


PATCH_LOADERS: dict[str, Any] = {
    InterchangePatch.kind: InterchangePatch.from_dict,
    ProjectOutPatch.kind: ProjectOutPatch.from_dict,
    AddDirectionPatch.kind: AddDirectionPatch.from_dict,
    SwapMeanPatch.kind: SwapMeanPatch.from_dict,
    SwapComponentsPatch.kind: SwapComponentsPatch.from_dict,
    RandomControlPatch.kind: RandomControlPatch.from_dict,
    ResidualPathPatch.kind: ResidualPathPatch.from_dict,
}


__all__ = [
    "ActivationPatchSpec",
    "AddDirectionPatch",
    "InterchangePatch",
    "ProjectOutPatch",
    "RandomControlPatch",
    "ResidualPathPatch",
    "SwapComponentsPatch",
    "SwapMeanPatch",
]
