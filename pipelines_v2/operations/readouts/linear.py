"""Linear readout specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

from pipelines_v2.core.types import OperationSpec, RuntimeSecret
from pipelines_v2.operations.common._shared import (
    analysis_runtime_spec_for_refs,
    row_selector_from_dict,
    runtime_secrets_from_refs,
    spec_value_from_dict,
)
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector


@dataclass(frozen=True, slots=True)
class ProbeSpec(OperationSpec):
    """Train and evaluate a linear probe over one captured feature family."""

    feature: Any = None
    rows: Any = None
    labels: Any = None
    group_by: Any = None
    split: Any = None
    tokens: TokenSelector = field(default_factory=TokenSelector.full_sequence)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)
    train_values: Sequence[Any] = field(default_factory=lambda: ("train",))
    train_stages: Sequence[Sequence[Any]] = field(default_factory=tuple)
    stage_epochs: Sequence[int] = field(default_factory=tuple)
    test_values: Sequence[Any] = field(default_factory=lambda: ("test",))
    folds: int = 5
    baselines: Sequence[str] = field(default_factory=tuple)
    metrics: Sequence[str] = field(default_factory=tuple)
    persist_predictions: bool = False
    persist_model: bool = False

    kind: ClassVar[str] = "probe"

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.rows, self.labels, self.group_by, self.split)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec_for_refs(self.rows, self.labels, self.group_by, self.split)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProbeSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            rows=row_selector_from_dict(payload.get("rows")),
            labels=spec_value_from_dict(payload.get("labels")),
            group_by=spec_value_from_dict(payload.get("group_by")),
            split=spec_value_from_dict(payload.get("split")),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "full_sequence"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
            train_values=tuple(payload.get("train_values", ("train",))),
            train_stages=tuple(tuple(stage) for stage in payload.get("train_stages", ())),
            stage_epochs=tuple(int(value) for value in payload.get("stage_epochs", ())),
            test_values=tuple(payload.get("test_values", ("test",))),
            folds=int(payload.get("folds", 5)),
            baselines=tuple(str(item) for item in payload.get("baselines", ())),
            metrics=tuple(str(item) for item in payload.get("metrics", ())),
            persist_predictions=bool(payload.get("persist_predictions", False)),
            persist_model=bool(payload.get("persist_model", False)),
        )


@dataclass(frozen=True, slots=True)
class PersistedProbeImportSpec(OperationSpec):
    """Import a trained linear probe into Xenon's canonical probe artifact shape."""

    path: str = ""
    name: str | None = None
    format: str = "json"
    model: str | None = None
    feature_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "persisted_probe_import"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PersistedProbeImportSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            path=str(payload["path"]),
            name=str(payload["name"]) if payload.get("name") is not None else None,
            format=str(payload.get("format", "json")),
            model=str(payload["model"]) if payload.get("model") is not None else None,
            feature_name=str(payload["feature_name"]) if payload.get("feature_name") is not None else None,
            metadata={str(key): value for key, value in dict(payload.get("metadata", {})).items()},
        )


@dataclass(frozen=True, slots=True)
class PersistedProbeInferenceSpec(OperationSpec):
    """Apply a persisted linear probe artifact to captured features."""

    feature: Any = None
    probe: Any = None
    rows: Any = None
    tokens: TokenSelector = field(default_factory=TokenSelector.full_sequence)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)
    layers: Sequence[int] = field(default_factory=tuple)
    emit_labels: bool = True
    score_name: str | None = None

    kind: ClassVar[str] = "persisted_probe_inference"

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.rows)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec_for_refs(self.rows)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PersistedProbeInferenceSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            probe=spec_value_from_dict(payload.get("probe")),
            rows=row_selector_from_dict(payload.get("rows")),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "full_sequence"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
            layers=tuple(int(value) for value in payload.get("layers", ())),
            emit_labels=bool(payload.get("emit_labels", True)),
            score_name=str(payload["score_name"]) if payload.get("score_name") is not None else None,
        )


__all__ = ["PersistedProbeImportSpec", "PersistedProbeInferenceSpec", "ProbeSpec"]
