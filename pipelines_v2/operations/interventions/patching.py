"""Activation patching specs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

from pipelines_v2.core.types import EngineCapability, OperationSpec, RuntimeSecret, SpecValidationError
from pipelines_v2.data.datasets import Dataset
from pipelines_v2.operations.common.tokens import TokenSelector

if TYPE_CHECKING:
    from pipelines_v2.engine.base import Engine


@dataclass(frozen=True, slots=True)
class ActivationPatchSpec(OperationSpec):
    """Planned model-bound intervention spec for activation patching."""

    engine: "Engine | None" = None
    dataset: Dataset = field(default_factory=lambda: Dataset.from_examples(()))
    basis: Any = None
    site: Any = None
    layers: Sequence[int] = field(default_factory=tuple)
    tokens: TokenSelector = field(default_factory=TokenSelector.last)
    mode: str = "project_out"
    components: Sequence[str] = field(default_factory=tuple)
    strengths: Sequence[float] = field(default_factory=tuple)
    controls: Sequence[str] = field(default_factory=tuple)
    metrics: Sequence[Any] = field(default_factory=tuple)

    kind: ClassVar[str] = "activation_patch"

    def __post_init__(self) -> None:
        if self.engine is None:
            raise SpecValidationError("ActivationPatchSpec requires an engine")

    def to_dict(self) -> dict[str, Any]:
        data = super(ActivationPatchSpec, self).to_dict()
        data["engine"] = self.engine.identity() if self.engine is not None else None
        return data

    def semantic_dict(self) -> dict[str, Any]:
        data = super(ActivationPatchSpec, self).semantic_dict()
        data["engine"] = self.engine.semantic_identity() if self.engine is not None else None
        return data

    def required_capabilities(self) -> set[EngineCapability]:
        return {EngineCapability.ACTIVATION_PATCHING}

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return self.dataset.runtime_secrets()

    def bound_engine(self) -> "Engine | None":
        return self.engine

    def runtime_spec(self) -> Any | None:
        return self.engine.runtime_spec() if self.engine is not None else None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActivationPatchSpec":
        from pipelines_v2.engine import engine_from_dict

        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            engine=engine_from_dict(dict(payload["engine"])),
            dataset=Dataset.from_dict(payload.get("dataset", {})),
            basis=payload.get("basis"),
            site=payload.get("site"),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "last"})),
            mode=str(payload.get("mode", "project_out")),
            components=tuple(str(item) for item in payload.get("components", ())),
            strengths=tuple(float(item) for item in payload.get("strengths", ())),
            controls=tuple(str(item) for item in payload.get("controls", ())),
            metrics=tuple(payload.get("metrics", ())),
        )


__all__ = ["ActivationPatchSpec"]
