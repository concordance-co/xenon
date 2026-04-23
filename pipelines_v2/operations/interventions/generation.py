"""Runnable generation specs for intervention workflows."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar

from pipelines_v2.core.types import EngineCapability, OperationSpec, RuntimeSecret, SpecValidationError
from pipelines_v2.data.datasets import Dataset
from pipelines_v2.operations.capture import GenerationSpec
from pipelines_v2.operations.common._shared import (
    merge_string_tuples,
    runtime_secrets_from_refs,
    spec_value_from_dict,
)
from pipelines_v2.operations.common.builders import PromptMetadataBuilder

from ._shared import dataset_provides_token_sections, requires_token_sections
from .recipes import ActivationPatchSpec

if TYPE_CHECKING:
    from pipelines_v2.engine.base import Engine


@dataclass(frozen=True, slots=True)
class GenerationRunSpec(OperationSpec):
    """Model-bound raw generation over a selected target row set."""

    engine: "Engine | None" = None
    dataset: Dataset = field(default_factory=lambda: Dataset.from_examples(()))
    select_when: Any = None
    generation: GenerationSpec = field(default_factory=GenerationSpec)

    kind: ClassVar[str] = "generation_run"

    def __post_init__(self) -> None:
        if self.engine is None:
            raise SpecValidationError("GenerationRunSpec requires an engine")
        if not self.generation.enabled or (
            self.generation.max_tokens is not None and int(self.generation.max_tokens or 0) <= 0
        ):
            raise SpecValidationError(
                "GenerationRunSpec requires generation.enabled=True and generation.max_tokens > 0 or None"
            )

    def to_dict(self) -> dict[str, Any]:
        data = super(GenerationRunSpec, self).to_dict()
        data["engine"] = self.engine.identity() if self.engine is not None else None
        return data

    def semantic_dict(self) -> dict[str, Any]:
        data = super(GenerationRunSpec, self).semantic_dict()
        data["engine"] = self.engine.semantic_identity() if self.engine is not None else None
        return data

    def required_capabilities(self) -> set[EngineCapability]:
        caps = {EngineCapability.GENERATION}
        if self.generation.structured_output is not None:
            caps.add(EngineCapability.STRUCTURED_OUTPUT)
        return caps

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.dataset, self.select_when)

    def bound_engine(self) -> "Engine | None":
        return self.engine

    def runtime_spec(self) -> Any | None:
        if self.engine is None:
            return None
        runtime_spec = self.engine.runtime_spec()
        from pipelines_v2.engine.base import PythonRuntimeSpec

        if not isinstance(runtime_spec, PythonRuntimeSpec):
            return runtime_spec
        return PythonRuntimeSpec(
            python_version=runtime_spec.python_version,
            pip_packages=merge_string_tuples(
                runtime_spec.pip_packages,
                self.dataset.runtime_pip_packages(),
            ),
            env=dict(runtime_spec.env),
            secrets=runtime_spec.secrets,
            local_python_sources=runtime_spec.local_python_sources,
        )

    def resolve_dataset(self) -> "GenerationRunSpec":
        if not self.dataset.is_deferred:
            return self
        return replace(self, dataset=self.dataset.resolve())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GenerationRunSpec":
        from pipelines_v2.engine import engine_from_dict

        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            engine=engine_from_dict(dict(payload["engine"])),
            dataset=Dataset.from_dict(payload.get("dataset", {})),
            select_when=spec_value_from_dict(payload.get("select_when")),
            generation=GenerationSpec.from_dict(payload.get("generation", {})),
        )


@dataclass(frozen=True, slots=True)
class PatchedGenerationSpec(OperationSpec):
    """Model-bound generation under a typed activation-patch recipe."""

    engine: "Engine | None" = None
    dataset: Dataset = field(default_factory=lambda: Dataset.from_examples(()))
    patch: ActivationPatchSpec | None = None
    select_when: Any = None
    pair_by: Any = None
    target_when: Any = None
    donor_when: Any = None
    generation: GenerationSpec = field(default_factory=GenerationSpec)
    prompt_metadata_builder: PromptMetadataBuilder | None = None

    kind: ClassVar[str] = "patched_generation"

    def __post_init__(self) -> None:
        if self.engine is None:
            raise SpecValidationError("PatchedGenerationSpec requires an engine")
        if self.patch is None:
            raise SpecValidationError("PatchedGenerationSpec requires patch")
        if not self.generation.enabled or (
            self.generation.max_tokens is not None and int(self.generation.max_tokens or 0) <= 0
        ):
            raise SpecValidationError(
                "PatchedGenerationSpec requires generation.enabled=True and generation.max_tokens > 0 or None"
            )
        patch = self.patch
        if patch.requires_pairing():
            if self.pair_by is None:
                raise SpecValidationError(
                    f"PatchedGenerationSpec requires pair_by for {patch.operator} patches"
                )
            if self.target_when is None:
                raise SpecValidationError(
                    f"PatchedGenerationSpec requires target_when for {patch.operator} patches"
                )
            if self.donor_when is None:
                raise SpecValidationError(
                    f"PatchedGenerationSpec requires donor_when for {patch.operator} patches"
                )
        else:
            if self.pair_by is not None or self.target_when is not None or self.donor_when is not None:
                raise SpecValidationError(
                    f"PatchedGenerationSpec pair_by/target_when/donor_when are not used for {patch.operator} patches"
                )
        if requires_token_sections(patch.target_tokens) and not self._provides_target_token_sections():
            raise SpecValidationError(
                "PatchedGenerationSpec uses TokenSelector.section(...) for patch.target_tokens, "
                "but no target-side token-section metadata source is defined. "
                "Provide prompt_metadata_builder=... or explicit metadata['token_sections'] on every example."
            )

    def _provides_target_token_sections(self) -> bool:
        if self.prompt_metadata_builder is not None:
            return True
        return dataset_provides_token_sections(self.dataset)

    def to_dict(self) -> dict[str, Any]:
        data = super(PatchedGenerationSpec, self).to_dict()
        data["engine"] = self.engine.identity() if self.engine is not None else None
        return data

    def semantic_dict(self) -> dict[str, Any]:
        data = super(PatchedGenerationSpec, self).semantic_dict()
        data["engine"] = self.engine.semantic_identity() if self.engine is not None else None
        return data

    def required_capabilities(self) -> set[EngineCapability]:
        caps = {
            EngineCapability.GENERATION,
            EngineCapability.ACTIVATION_PATCHING,
            EngineCapability.REQUEST_SCOPED_INTERVENTIONS,
        }
        if self.generation.structured_output is not None:
            caps.add(EngineCapability.STRUCTURED_OUTPUT)
        return caps

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(
            self.dataset,
            self.patch,
            self.select_when,
            self.pair_by,
            self.target_when,
            self.donor_when,
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
        runtime_env = dict(runtime_spec.env)
        if self.patch.operator in {"project_out", "add_direction", "swap_mean", "swap_components", "random_control"}:
            runtime_env["XENON_ACTIVATION_PATCH_COMPILED_OPERATOR"] = "subspace"
        return PythonRuntimeSpec(
            python_version=runtime_spec.python_version,
            pip_packages=merge_string_tuples(
                runtime_spec.pip_packages,
                self.dataset.runtime_pip_packages(),
            ),
            env=runtime_env,
            secrets=runtime_spec.secrets,
            local_python_sources=local_python_sources,
        )

    def resolve_dataset(self) -> "PatchedGenerationSpec":
        if not self.dataset.is_deferred:
            return self
        return replace(self, dataset=self.dataset.resolve())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PatchedGenerationSpec":
        from pipelines_v2.engine import engine_from_dict

        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            engine=engine_from_dict(dict(payload["engine"])),
            dataset=Dataset.from_dict(payload.get("dataset", {})),
            patch=ActivationPatchSpec.from_dict(payload.get("patch", {})),
            select_when=spec_value_from_dict(payload.get("select_when")),
            pair_by=spec_value_from_dict(payload.get("pair_by")),
            target_when=spec_value_from_dict(payload.get("target_when")),
            donor_when=spec_value_from_dict(payload.get("donor_when")),
            generation=GenerationSpec.from_dict(payload.get("generation", {})),
            prompt_metadata_builder=(
                PromptMetadataBuilder.from_dict(dict(payload["prompt_metadata_builder"]))
                if payload.get("prompt_metadata_builder") is not None
                else None
            ),
        )


__all__ = ["GenerationRunSpec", "PatchedGenerationSpec"]
