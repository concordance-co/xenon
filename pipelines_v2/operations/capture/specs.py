"""Model-bound capture specs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Sequence

from pipelines_v2.core.types import EngineCapability, OperationSpec, RuntimeSecret, SpecValidationError
from pipelines_v2.data.datasets import Dataset
from pipelines_v2.operations.common._shared import (
    contains_section_token_selector,
    example_has_explicit_token_sections,
    merge_string_tuples,
)
from pipelines_v2.operations.common.builders import PromptMetadataBuilder

from .sites import CaptureSite, MoERoutingSite, ResidualSite

if TYPE_CHECKING:
    from pipelines_v2.engine.base import Engine


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    """Generation settings attached to a capture run."""

    enabled: bool = False
    max_tokens: int = 0
    temperature: float = 0.0
    capture_reasoning: bool = False
    structured_output: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GenerationSpec":
        return cls(
            enabled=bool(payload.get("enabled", False)),
            max_tokens=int(payload.get("max_tokens", 0)),
            temperature=float(payload.get("temperature", 0.0)),
            capture_reasoning=bool(payload.get("capture_reasoning", False)),
            structured_output=payload.get("structured_output"),
        )


@dataclass(frozen=True, slots=True)
class CaptureSpec(OperationSpec):
    """Model-bound capture request over a dataset."""

    engine: "Engine | None" = None
    dataset: Dataset = field(default_factory=lambda: Dataset.from_examples(()))
    sites: Sequence[CaptureSite] = field(default_factory=tuple)
    generation: GenerationSpec = field(default_factory=GenerationSpec)
    prompt_metadata_builder: PromptMetadataBuilder | None = None

    kind: ClassVar[str] = "capture"

    def __post_init__(self) -> None:
        if self.engine is None:
            raise SpecValidationError("CaptureSpec requires an engine")
        if not self.sites and not self.generation.enabled:
            raise SpecValidationError("CaptureSpec requires at least one site or enabled generation")
        names = [site.name for site in self.sites]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise SpecValidationError(f"Duplicate capture site names: {sorted(duplicates)}")
        if self.uses_section_token_selector() and not self.provides_token_sections():
            raise SpecValidationError(
                "CaptureSpec uses TokenSelector.section(...), but no token-section metadata source is defined. "
                "Provide prompt_metadata_builder=... or explicit metadata['token_sections'] for every materialized example."
            )

    def to_dict(self) -> dict[str, Any]:
        data = super(CaptureSpec, self).to_dict()
        data["engine"] = self.engine.identity() if self.engine is not None else None
        return data

    def semantic_dict(self) -> dict[str, Any]:
        data = super(CaptureSpec, self).semantic_dict()
        data["engine"] = self.engine.semantic_identity() if self.engine is not None else None
        return data

    def required_capabilities(self) -> set[EngineCapability]:
        caps: set[EngineCapability] = set()
        for site in self.sites:
            caps.update(site.required_capabilities())
        if self.generation.enabled:
            caps.add(EngineCapability.GENERATION)
        if self.generation.structured_output is not None:
            caps.add(EngineCapability.STRUCTURED_OUTPUT)
        return caps

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return self.dataset.runtime_secrets()

    def bound_engine(self) -> "Engine | None":
        return self.engine

    def runtime_spec(self) -> Any | None:
        if self.engine is None:
            return None
        runtime_spec = self.engine.runtime_spec()
        if self.prompt_metadata_builder is None:
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
            local_python_sources=merge_string_tuples(
                runtime_spec.local_python_sources,
                self.prompt_metadata_builder.local_python_sources,
            ),
        )

    def uses_section_token_selector(self) -> bool:
        return any(contains_section_token_selector(site.tokens) for site in self.sites)

    def provides_token_sections(self) -> bool:
        if self.prompt_metadata_builder is not None:
            return True
        if self.dataset.is_deferred:
            return False
        return all(example_has_explicit_token_sections(example) for example in self.dataset.examples)

    def resolve_dataset(self) -> "CaptureSpec":
        if not self.dataset.is_deferred:
            return self
        return replace(self, dataset=self.dataset.resolve())

    @classmethod
    def from_file(cls, path: str | Path) -> "CaptureSpec":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CaptureSpec":
        if payload.get("kind") not in (None, cls.kind):
            raise SpecValidationError(f"CaptureSpec expected kind {cls.kind!r}, got {payload.get('kind')!r}")
        from pipelines_v2.engine import engine_from_dict

        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            engine=engine_from_dict(dict(payload["engine"])),
            dataset=Dataset.from_dict(payload.get("dataset", {})),
            sites=[capture_site_from_dict(site) for site in payload.get("sites", ())],
            generation=GenerationSpec.from_dict(payload.get("generation", {})),
            prompt_metadata_builder=(
                PromptMetadataBuilder.from_dict(dict(payload["prompt_metadata_builder"]))
                if payload.get("prompt_metadata_builder") is not None
                else None
            ),
        )


def capture_site_from_dict(payload: Mapping[str, Any]) -> CaptureSite:
    if "site" in payload:
        return ResidualSite.from_dict(payload)
    if "record" in payload:
        return MoERoutingSite.from_dict(payload)
    raise SpecValidationError(f"Unknown capture site payload: {payload}")


__all__ = [
    "CaptureSpec",
    "GenerationSpec",
    "capture_site_from_dict",
]
