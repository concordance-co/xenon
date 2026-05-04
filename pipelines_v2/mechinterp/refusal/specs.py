"""Refusal-direction operation specs.

These specs encode the reusable pieces of "Refusal in Language Models Is
Mediated by a Single Direction" while delegating generic vector math,
projection scoring, and intervention execution to existing pipelines_v2
operation families.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

from pipelines_v2.core.types import OperationSpec, RuntimeSecret
from pipelines_v2.operations.common._shared import analysis_runtime_spec, runtime_secrets_from_refs, spec_value_from_dict
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector
from pipelines_v2.operations.projections import SectionSelector


@dataclass(frozen=True, slots=True)
class RefusalDirectionSpec(OperationSpec):
    """Compute a harmful-minus-harmless refusal direction."""

    feature: Any = None
    harmful_when: Any = None
    harmless_when: Any = None
    group_by: Any = None
    layers: Sequence[int] = field(default_factory=tuple)
    tokens: TokenSelector = field(default_factory=TokenSelector.last)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)
    name: str = "refusal_direction"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "refusal_direction"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(int(layer) for layer in self.layers))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "metadata", {str(key): value for key, value in dict(self.metadata).items()})

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.feature, self.harmful_when, self.harmless_when, self.group_by)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefusalDirectionSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            harmful_when=spec_value_from_dict(payload.get("harmful_when")),
            harmless_when=spec_value_from_dict(payload.get("harmless_when")),
            group_by=spec_value_from_dict(payload.get("group_by")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "last"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
            name=str(payload.get("name", "refusal_direction")),
            metadata={str(key): value for key, value in dict(payload.get("metadata", {})).items()},
        )


@dataclass(frozen=True, slots=True)
class RefusalScoreSpec(OperationSpec):
    """Score captured sections against a refusal direction."""

    feature: Any = None
    direction: Any = None
    layers: Sequence[int] = field(default_factory=tuple)
    slices: SectionSelector = field(default_factory=SectionSelector.all)
    rows: Any = None
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)
    metric: str = "signed_dot"
    summaries: Sequence[str] = field(default_factory=lambda: ("mean",))
    emit_labels: bool = True

    kind: ClassVar[str] = "refusal_score"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(int(layer) for layer in self.layers))
        object.__setattr__(self, "metric", str(self.metric))
        object.__setattr__(self, "summaries", tuple(str(item) for item in self.summaries))

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.feature, self.direction, self.rows)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefusalScoreSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            direction=spec_value_from_dict(payload.get("direction")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            slices=SectionSelector.from_dict(payload.get("slices")),
            rows=spec_value_from_dict(payload.get("rows")),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
            metric=str(payload.get("metric", "signed_dot")),
            summaries=tuple(str(item) for item in payload.get("summaries", ("mean",))),
            emit_labels=bool(payload.get("emit_labels", True)),
        )


@dataclass(frozen=True, slots=True)
class RefusalDirectionSelectionSpec(OperationSpec):
    """Select the refusal layer with largest validation projection separation."""

    direction: Any = None
    scores: Any = None
    harmful_when: Any = None
    harmless_when: Any = None
    layers: Sequence[int] = field(default_factory=tuple)
    summary_metric: str = "mean"
    name: str = "selected_refusal_direction"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "refusal_direction_selection"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(int(layer) for layer in self.layers))
        object.__setattr__(self, "summary_metric", str(self.summary_metric))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "metadata", {str(key): value for key, value in dict(self.metadata).items()})

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.direction, self.scores, self.harmful_when, self.harmless_when)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefusalDirectionSelectionSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            direction=spec_value_from_dict(payload.get("direction")),
            scores=spec_value_from_dict(payload.get("scores")),
            harmful_when=spec_value_from_dict(payload.get("harmful_when")),
            harmless_when=spec_value_from_dict(payload.get("harmless_when")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            summary_metric=str(payload.get("summary_metric", "mean")),
            name=str(payload.get("name", "selected_refusal_direction")),
            metadata={str(key): value for key, value in dict(payload.get("metadata", {})).items()},
        )


@dataclass(frozen=True, slots=True)
class RefusalAblationSubspaceSpec(OperationSpec):
    """Convert a refusal direction into a one-component subspace for ProjectOutPatch."""

    direction: Any = None
    layers: Sequence[int] = field(default_factory=tuple)
    name: str = "refusal_direction_component"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "refusal_ablation_subspace"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(int(layer) for layer in self.layers))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "metadata", {str(key): value for key, value in dict(self.metadata).items()})

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.direction)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RefusalAblationSubspaceSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            direction=spec_value_from_dict(payload.get("direction")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            name=str(payload.get("name", "refusal_direction_component")),
            metadata={str(key): value for key, value in dict(payload.get("metadata", {})).items()},
        )


__all__ = [
    "RefusalAblationSubspaceSpec",
    "RefusalDirectionSelectionSpec",
    "RefusalDirectionSpec",
    "RefusalScoreSpec",
]
