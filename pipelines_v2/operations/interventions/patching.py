"""Activation patching specs."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Sequence

from pipelines_v2.core.types import EngineCapability, OperationSpec, RuntimeSecret, SpecValidationError
from pipelines_v2.data.datasets import Dataset
from pipelines_v2.operations.capture import GenerationSpec
from pipelines_v2.operations.common._shared import (
    example_has_explicit_token_sections,
    merge_string_tuples,
    runtime_secrets_from_refs,
    spec_value_from_dict,
)
from pipelines_v2.operations.common.builders import PromptMetadataBuilder, TransformBuilder
from pipelines_v2.operations.common.tokens import TokenSelector

if TYPE_CHECKING:
    from pipelines_v2.engine.base import Engine


def _requires_token_sections(tokens: TokenSelector | None) -> bool:
    return isinstance(tokens, TokenSelector) and tokens.kind == "section"


def _dataset_provides_token_sections(dataset: Dataset) -> bool:
    if dataset.is_deferred:
        return False
    return all(example_has_explicit_token_sections(example) for example in dataset.examples)


@dataclass(frozen=True, slots=True)
class ResidualInterventionSite:
    """Prompt-side residual write location for activation interchange."""

    site: str
    layers: Sequence[int]

    kind: ClassVar[str] = "residual_intervention_site"

    def __post_init__(self) -> None:
        if not str(self.site).strip():
            raise SpecValidationError("ResidualInterventionSite requires a non-empty site name")
        normalized_layers = tuple(int(layer) for layer in self.layers)
        if not normalized_layers:
            raise SpecValidationError("ResidualInterventionSite requires at least one layer")
        object.__setattr__(self, "site", str(self.site))
        object.__setattr__(self, "layers", normalized_layers)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResidualInterventionSite":
        return cls(
            site=str(payload["site"]),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
        )


@dataclass(frozen=True, slots=True)
class ActivationPatchControl:
    """One named donor-control path evaluated alongside the main patch."""

    name: str
    donor_when: Any = None
    donor_tokens: TokenSelector | None = None

    kind: ClassVar[str] = "activation_patch_control"

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise SpecValidationError("ActivationPatchControl requires a non-empty name")
        if self.donor_when is None:
            raise SpecValidationError("ActivationPatchControl requires donor_when")
        object.__setattr__(self, "name", str(self.name))

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.donor_when)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActivationPatchControl":
        donor_tokens_payload = payload.get("donor_tokens")
        return cls(
            name=str(payload["name"]),
            donor_when=spec_value_from_dict(payload.get("donor_when")),
            donor_tokens=(
                TokenSelector.from_dict(donor_tokens_payload)
                if isinstance(donor_tokens_payload, Mapping)
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ActivationPatchSpec(OperationSpec):
    """Prompt-side residual activation interchange over paired donor/target examples."""

    engine: "Engine | None" = None
    dataset: Dataset = field(default_factory=lambda: Dataset.from_examples(()))
    source_feature: Any = None
    pair_by: Any = None
    target_when: Any = None
    donor_when: Any = None
    controls: Sequence[ActivationPatchControl] = field(default_factory=tuple)
    write_site: ResidualInterventionSite = field(
        default_factory=lambda: ResidualInterventionSite(site="resid_post", layers=(0,))
    )
    target_tokens: TokenSelector = field(default_factory=TokenSelector.last)
    donor_tokens: TokenSelector | None = None
    generation: GenerationSpec = field(default_factory=GenerationSpec)
    prompt_metadata_builder: PromptMetadataBuilder | None = None
    row_evaluator: TransformBuilder | None = None

    kind: ClassVar[str] = "activation_patch"

    def __post_init__(self) -> None:
        if self.engine is None:
            raise SpecValidationError("ActivationPatchSpec requires an engine")
        if self.source_feature is None:
            raise SpecValidationError("ActivationPatchSpec requires source_feature")
        if self.pair_by is None:
            raise SpecValidationError("ActivationPatchSpec requires pair_by")
        if self.target_when is None:
            raise SpecValidationError("ActivationPatchSpec requires target_when")
        if self.donor_when is None:
            raise SpecValidationError("ActivationPatchSpec requires donor_when")
        if self.row_evaluator is None:
            raise SpecValidationError("ActivationPatchSpec requires row_evaluator")
        if not self.generation.enabled or int(self.generation.max_tokens or 0) <= 0:
            raise SpecValidationError(
                "ActivationPatchSpec requires generation.enabled=True and generation.max_tokens > 0"
            )
        if _requires_token_sections(self.target_tokens) and not self._provides_target_token_sections():
            raise SpecValidationError(
                "ActivationPatchSpec uses TokenSelector.section(...) for target_tokens, "
                "but no target-side token-section metadata source is defined. "
                "Provide prompt_metadata_builder=... or explicit metadata['token_sections'] on every example."
            )

    def _provides_target_token_sections(self) -> bool:
        if self.prompt_metadata_builder is not None:
            return True
        return _dataset_provides_token_sections(self.dataset)

    def to_dict(self) -> dict[str, Any]:
        data = super(ActivationPatchSpec, self).to_dict()
        data["engine"] = self.engine.identity() if self.engine is not None else None
        return data

    def semantic_dict(self) -> dict[str, Any]:
        data = super(ActivationPatchSpec, self).semantic_dict()
        data["engine"] = self.engine.semantic_identity() if self.engine is not None else None
        return data

    def required_capabilities(self) -> set[EngineCapability]:
        return {
            EngineCapability.GENERATION,
            EngineCapability.ACTIVATION_PATCHING,
            EngineCapability.REQUEST_SCOPED_INTERVENTIONS,
        }

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(
            self.dataset,
            self.source_feature,
            self.pair_by,
            self.target_when,
            self.donor_when,
            tuple(self.controls),
        )

    def bound_engine(self) -> "Engine | None":
        return self.engine

    def runtime_spec(self) -> Any | None:
        if self.engine is None:
            return None
        runtime_spec = self.engine.runtime_spec()
        from pipelines_v2.engine.base import PythonRuntimeSpec

        if not isinstance(runtime_spec, PythonRuntimeSpec):
            return runtime_spec
        local_python_sources = tuple(runtime_spec.local_python_sources)
        if self.prompt_metadata_builder is not None:
            local_python_sources = merge_string_tuples(
                local_python_sources,
                self.prompt_metadata_builder.local_python_sources,
            )
        if self.row_evaluator is not None:
            local_python_sources = merge_string_tuples(
                local_python_sources,
                self.row_evaluator.local_python_sources,
            )
        return PythonRuntimeSpec(
            python_version=runtime_spec.python_version,
            pip_packages=merge_string_tuples(
                runtime_spec.pip_packages,
                self.dataset.runtime_pip_packages(),
            ),
            env=dict(runtime_spec.env),
            secrets=runtime_spec.secrets,
            local_python_sources=local_python_sources,
        )

    def resolve_dataset(self) -> "ActivationPatchSpec":
        if not self.dataset.is_deferred:
            return self
        return replace(self, dataset=self.dataset.resolve())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActivationPatchSpec":
        from pipelines_v2.engine import engine_from_dict

        donor_tokens_payload = payload.get("donor_tokens")
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            engine=engine_from_dict(dict(payload["engine"])),
            dataset=Dataset.from_dict(payload.get("dataset", {})),
            source_feature=spec_value_from_dict(payload.get("source_feature")),
            pair_by=spec_value_from_dict(payload.get("pair_by")),
            target_when=spec_value_from_dict(payload.get("target_when")),
            donor_when=spec_value_from_dict(payload.get("donor_when")),
            controls=tuple(
                ActivationPatchControl.from_dict(dict(item))
                for item in payload.get("controls", ())
            ),
            write_site=ResidualInterventionSite.from_dict(payload.get("write_site", {})),
            target_tokens=TokenSelector.from_dict(payload.get("target_tokens", {"kind": "last"})),
            donor_tokens=(
                TokenSelector.from_dict(donor_tokens_payload)
                if isinstance(donor_tokens_payload, Mapping)
                else None
            ),
            generation=GenerationSpec.from_dict(payload.get("generation", {})),
            prompt_metadata_builder=(
                PromptMetadataBuilder.from_dict(dict(payload["prompt_metadata_builder"]))
                if payload.get("prompt_metadata_builder") is not None
                else None
            ),
            row_evaluator=(
                TransformBuilder.from_dict(dict(payload["row_evaluator"]))
                if payload.get("row_evaluator") is not None
                else None
            ),
        )


__all__ = [
    "ActivationPatchControl",
    "ActivationPatchSpec",
    "ResidualInterventionSite",
]
