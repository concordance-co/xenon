"""Serializable operation specs and site selectors."""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Sequence

from pipelines_v2.core.paths import find_workspace_root, resolve_workspace_path
from pipelines_v2.core.types import EngineCapability, OperationSpec, RuntimeSecret, SpecValidationError
from pipelines_v2.data.datasets import CaseSet, Dataset, LabelPredicate, LabelSet

if TYPE_CHECKING:
    from pipelines_v2.engine.base import Engine


@dataclass(frozen=True, slots=True)
class TokenSelector:
    """Select token positions from a captured token axis."""
    kind: str
    value: Any = None

    @classmethod
    def last(cls) -> "TokenSelector":
        """Select only the last captured token."""
        return cls(kind="last")

    @classmethod
    def full_sequence(cls) -> "TokenSelector":
        """Select the full captured token sequence."""
        return cls(kind="full_sequence")

    @classmethod
    def slice(cls, start: int, stop: int | None = None) -> "TokenSelector":
        """Select a Python-style token slice over the captured sequence."""
        return cls(kind="slice", value={"start": start, "stop": stop})

    @classmethod
    def section(cls, name: str) -> "TokenSelector":
        """Select a named token section such as ``STRATEGY`` or ``SETTINGS``."""
        return cls(kind="section", value=name)

    def resolve(
        self,
        token_count: int,
        *,
        token_sections: Mapping[str, Sequence[int]] | None = None,
    ) -> list[int]:
        """Resolve the selector into concrete token indices."""
        if token_count <= 0:
            return []
        if self.kind == "last":
            return [token_count - 1]
        if self.kind == "full_sequence":
            return list(range(token_count))
        if self.kind == "slice":
            start = int(self.value["start"])
            stop = self.value.get("stop")
            return list(range(token_count))[start:stop]
        if self.kind == "section":
            section_name = str(self.value)
            if token_sections is None or section_name not in token_sections:
                raise SpecValidationError(f"Token selector section {section_name!r} is not available for this example")
            return [int(position) for position in token_sections[section_name]]
        raise SpecValidationError(f"Unsupported token selector: {self.kind}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TokenSelector":
        return cls(kind=str(payload["kind"]), value=payload.get("value"))


@dataclass(frozen=True, slots=True)
class PromptMetadataBuilder:
    """Serializable ref to a user-defined function that derives prompt metadata.

    The referenced function must be importable in the execution runtime and accept
    one positional argument: the rendered prompt text. It must return a mapping.
    """

    import_path: str
    local_python_sources: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        module_name, _, function_name = str(self.import_path).partition(":")
        if not module_name or not function_name:
            raise SpecValidationError(
                "PromptMetadataBuilder import_path must have the form 'module.path:function_name'"
            )
        object.__setattr__(self, "import_path", f"{module_name}:{function_name}")
        object.__setattr__(
            self,
            "local_python_sources",
            tuple(str(source) for source in self.local_python_sources if str(source).strip()),
        )

    @classmethod
    def from_function(
        cls,
        function: Any,
        *,
        local_python_sources: Sequence[str] | None = None,
    ) -> "PromptMetadataBuilder":
        """Build a serializable ref from a top-level Python function."""
        import_path, sources = _callable_import_ref(
            function,
            local_python_sources=local_python_sources,
            label="Prompt metadata builders",
        )
        return cls(
            import_path=import_path,
            local_python_sources=sources,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "import_path": self.import_path,
            "local_python_sources": list(self.local_python_sources),
        }

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "import_path": self.import_path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromptMetadataBuilder":
        return cls(
            import_path=str(payload["import_path"]),
            local_python_sources=tuple(str(source) for source in payload.get("local_python_sources", ())),
        )

    def build(self, rendered_prompt: str) -> dict[str, Any]:
        """Import and execute the builder against one rendered prompt."""
        function = _load_importable_function(
            self.import_path,
            label="Prompt metadata builder",
            local_python_sources=self.local_python_sources,
        )
        payload = function(rendered_prompt)
        if not isinstance(payload, Mapping):
            raise TypeError(
                "Prompt metadata builder must return a mapping, "
                f"got {type(payload).__name__}"
            )
        return dict(payload)


@dataclass(frozen=True, slots=True)
class TransformResult:
    """Structured return value for ``TransformSpec`` builders."""

    payload: Mapping[str, Any] = field(default_factory=dict)
    labels: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    example_keys: Sequence[str] | None = None


@dataclass(frozen=True, slots=True)
class TransformBuilder:
    """Serializable ref to a user-defined transform function.

    The referenced function must be importable in the execution runtime. It will
    be called with keyword arguments from ``TransformSpec.inputs`` and must return
    either ``TransformResult`` or a mapping with compatible keys.
    """

    import_path: str
    local_python_sources: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        module_name, _, function_name = str(self.import_path).partition(":")
        if not module_name or not function_name:
            raise SpecValidationError(
                "TransformBuilder import_path must have the form 'module.path:function_name'"
            )
        object.__setattr__(self, "import_path", f"{module_name}:{function_name}")
        object.__setattr__(
            self,
            "local_python_sources",
            tuple(str(source) for source in self.local_python_sources if str(source).strip()),
        )

    @classmethod
    def from_function(
        cls,
        function: Any,
        *,
        local_python_sources: Sequence[str] | None = None,
    ) -> "TransformBuilder":
        import_path, sources = _callable_import_ref(
            function,
            local_python_sources=local_python_sources,
            label="Transform builders",
        )
        return cls(import_path=import_path, local_python_sources=sources)

    def to_dict(self) -> dict[str, Any]:
        return {
            "import_path": self.import_path,
            "local_python_sources": list(self.local_python_sources),
        }

    def semantic_dict(self) -> dict[str, Any]:
        return {
            "import_path": self.import_path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransformBuilder":
        return cls(
            import_path=str(payload["import_path"]),
            local_python_sources=tuple(str(source) for source in payload.get("local_python_sources", ())),
        )

    def build(self, inputs: Mapping[str, Any]) -> TransformResult | Mapping[str, Any]:
        function = _load_importable_function(
            self.import_path,
            label="Transform builder",
            local_python_sources=self.local_python_sources,
        )
        return function(**dict(inputs))


@dataclass(frozen=True, slots=True)
class TokenPooling:
    """Reduce selected token vectors to one vector per example."""
    kind: str

    @classmethod
    def mean(cls) -> "TokenPooling":
        """Average across the selected token positions."""
        return cls(kind="mean")

    @classmethod
    def last(cls) -> "TokenPooling":
        """Take the last selected token position."""
        return cls(kind="last")

    @classmethod
    def first(cls) -> "TokenPooling":
        """Take the first selected token position."""
        return cls(kind="first")

    def from_count(self, token_count: int) -> list[int]:
        """Resolve pooling to the token indices it will consume."""
        if token_count <= 0:
            return []
        if self.kind == "mean":
            return list(range(token_count))
        if self.kind == "last":
            return [token_count - 1]
        if self.kind == "first":
            return [0]
        raise SpecValidationError(f"Unsupported token pooling mode: {self.kind}")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TokenPooling":
        return cls(kind=str(payload["kind"]))


@dataclass(frozen=True, slots=True)
class TensorStorage:
    """Requested storage policy for captured tensor payloads."""
    dtype: str = "float16"
    format: str = "safetensors"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TensorStorage":
        return cls(
            dtype=str(payload.get("dtype", "float16")),
            format=str(payload.get("format", "safetensors")),
        )


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    """Generation settings attached to a capture run."""
    enabled: bool = False
    max_tokens: int = 0
    temperature: float = 0.0
    capture_reasoning: bool = False
    structured_output: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerationSpec":
        return cls(
            enabled=bool(payload.get("enabled", False)),
            max_tokens=int(payload.get("max_tokens", 0)),
            temperature=float(payload.get("temperature", 0.0)),
            capture_reasoning=bool(payload.get("capture_reasoning", False)),
            structured_output=payload.get("structured_output"),
        )


@dataclass(frozen=True, slots=True)
class ResidualSite:
    """Capture request for residual activations at one named site."""
    name: str
    site: str
    layers: Sequence[int]
    tokens: TokenSelector = field(default_factory=TokenSelector.last)
    storage: TensorStorage = field(default_factory=TensorStorage)

    def required_capabilities(self) -> set[EngineCapability]:
        return {EngineCapability.RESIDUAL_CAPTURE}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResidualSite":
        return cls(
            name=str(payload["name"]),
            site=str(payload["site"]),
            layers=[int(layer) for layer in payload.get("layers", ())],
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "last"})),
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
    def from_dict(cls, payload: dict[str, Any]) -> "RoutingRecord":
        return cls(kind=str(payload["kind"]), params=dict(payload.get("params", {})))


@dataclass(frozen=True, slots=True)
class MoERoutingSite:
    """Capture request for MoE router outputs across layers/tokens."""
    name: str
    layers: Sequence[int]
    tokens: TokenSelector = field(default_factory=TokenSelector.last)
    record: Sequence[RoutingRecord] = field(default_factory=lambda: (RoutingRecord.gate_logits(),))

    def required_capabilities(self) -> set[EngineCapability]:
        return {EngineCapability.MOE_ROUTING_CAPTURE}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MoERoutingSite":
        return cls(
            name=str(payload["name"]),
            layers=[int(layer) for layer in payload.get("layers", ())],
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "last"})),
            record=[RoutingRecord.from_dict(record) for record in payload.get("record", ())],
        )


CaptureSite = ResidualSite | MoERoutingSite


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
        """Return the engine capabilities required by this capture."""
        caps: set[EngineCapability] = set()
        for site in self.sites:
            caps.update(site.required_capabilities())
        if self.generation.enabled:
            caps.add(EngineCapability.GENERATION)
        if self.generation.structured_output is not None:
            caps.add(EngineCapability.STRUCTURED_OUTPUT)
        return caps

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        """Return runtime secrets required to resolve the capture dataset."""
        return self.dataset.runtime_secrets()

    def bound_engine(self) -> "Engine | None":
        """Return the engine that should execute this capture."""
        return self.engine

    def runtime_spec(self) -> Any | None:
        """Return runtime requirements declared by the bound engine."""
        if self.engine is None:
            return None
        runtime_spec = self.engine.runtime_spec()
        if self.prompt_metadata_builder is None:
            return runtime_spec
        from pipelines_v2.engine.base import PythonRuntimeSpec

        if not isinstance(runtime_spec, PythonRuntimeSpec):
            return runtime_spec
        return PythonRuntimeSpec(
            python_version=runtime_spec.python_version,
            pip_packages=runtime_spec.pip_packages,
            env=dict(runtime_spec.env),
            secrets=runtime_spec.secrets,
            local_python_sources=_merge_string_tuples(
                runtime_spec.local_python_sources,
                self.prompt_metadata_builder.local_python_sources,
            ),
        )

    def uses_section_token_selector(self) -> bool:
        """Whether any capture site requests named prompt sections."""
        return any(_contains_section_token_selector(site.tokens) for site in self.sites)

    def provides_token_sections(self) -> bool:
        """Whether capture has an explicit way to construct section metadata."""
        if self.prompt_metadata_builder is not None:
            return True
        if self.dataset.is_deferred:
            return False
        return all(_example_has_explicit_token_sections(example) for example in self.dataset.examples)

    def resolve_dataset(self) -> "CaptureSpec":
        """Resolve a deferred dataset inside the current runtime."""
        if not self.dataset.is_deferred:
            return self
        return replace(self, dataset=self.dataset.resolve())

    @classmethod
    def from_file(cls, path: str | Path) -> "CaptureSpec":
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
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
            sites=[_capture_site_from_dict(site) for site in payload.get("sites", ())],
            generation=GenerationSpec.from_dict(payload.get("generation", {})),
            prompt_metadata_builder=(
                PromptMetadataBuilder.from_dict(dict(payload["prompt_metadata_builder"]))
                if payload.get("prompt_metadata_builder") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ProbeSpec(OperationSpec):
    """Train and evaluate a linear probe over one captured feature family."""
    feature: Any = None
    labels: Any = None
    group_by: Any = None
    split: Any = None
    tokens: TokenSelector = field(default_factory=TokenSelector.full_sequence)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)
    train_values: Sequence[Any] = field(default_factory=lambda: ("train",))
    test_values: Sequence[Any] = field(default_factory=lambda: ("test",))
    folds: int = 5
    baselines: Sequence[str] = field(default_factory=tuple)
    metrics: Sequence[str] = field(default_factory=tuple)

    kind: ClassVar[str] = "probe"

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return _runtime_secrets_from_refs(self.labels, self.group_by, self.split)

    def runtime_spec(self) -> Any | None:
        return _analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProbeSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=_spec_value_from_dict(payload.get("feature")),
            labels=_spec_value_from_dict(payload.get("labels")),
            group_by=_spec_value_from_dict(payload.get("group_by")),
            split=_spec_value_from_dict(payload.get("split")),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "full_sequence"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
            train_values=tuple(payload.get("train_values", ("train",))),
            test_values=tuple(payload.get("test_values", ("test",))),
            folds=int(payload.get("folds", 5)),
            baselines=tuple(str(item) for item in payload.get("baselines", ())),
            metrics=tuple(str(item) for item in payload.get("metrics", ())),
        )


@dataclass(frozen=True, slots=True)
class DirectionSpec(OperationSpec):
    """Compute a direction from positive and negative example groups."""
    feature: Any = None
    positive: Any = None
    negative: Any = None
    group_by: Any = None
    layers: Sequence[int] = field(default_factory=tuple)
    tokens: TokenSelector = field(default_factory=TokenSelector.last)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)

    kind: ClassVar[str] = "direction"

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return _runtime_secrets_from_refs(self.positive, self.negative, self.group_by)

    def runtime_spec(self) -> Any | None:
        return _analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DirectionSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=_spec_value_from_dict(payload.get("feature")),
            positive=_spec_value_from_dict(payload.get("positive")),
            negative=_spec_value_from_dict(payload.get("negative")),
            group_by=_spec_value_from_dict(payload.get("group_by")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "last"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
        )


@dataclass(frozen=True, slots=True)
class BasisSpec(OperationSpec):
    """Learn a basis over captured features, currently PCA only."""
    feature: Any = None
    method: str = "pca"
    by: Any = None
    layers: Sequence[int] = field(default_factory=tuple)
    components: int = 8
    tokens: TokenSelector = field(default_factory=TokenSelector.last)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)

    kind: ClassVar[str] = "basis"

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return _runtime_secrets_from_refs(self.by)

    def runtime_spec(self) -> Any | None:
        return _analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BasisSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=_spec_value_from_dict(payload.get("feature")),
            method=str(payload.get("method", "pca")),
            by=_spec_value_from_dict(payload.get("by")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            components=int(payload.get("components", 8)),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "last"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
        )


@dataclass(frozen=True, slots=True)
class PairDeltaSpec(OperationSpec):
    """Compute paired positive-minus-negative deltas and emit a derived feature."""
    feature: Any = None
    case: Any = None
    positive: Any = None
    negative: Any = None
    layers: Sequence[int] = field(default_factory=tuple)
    tokens: TokenSelector = field(default_factory=TokenSelector.last)
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)
    output_feature_name: str = "delta"
    labels: Mapping[str, Any] = field(default_factory=dict)
    propagate_from: str = "positive"

    kind: ClassVar[str] = "pair_delta"

    def __post_init__(self) -> None:
        if self.propagate_from not in {"positive", "negative"}:
            raise SpecValidationError("PairDeltaSpec propagate_from must be 'positive' or 'negative'")
        if not str(self.output_feature_name).strip():
            raise SpecValidationError("PairDeltaSpec output_feature_name cannot be empty")

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return _runtime_secrets_from_refs(self.feature, self.case, self.positive, self.negative, self.labels)

    def runtime_spec(self) -> Any | None:
        return _analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PairDeltaSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=_spec_value_from_dict(payload.get("feature")),
            case=_spec_value_from_dict(payload.get("case")),
            positive=_spec_value_from_dict(payload.get("positive")),
            negative=_spec_value_from_dict(payload.get("negative")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "last"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
            output_feature_name=str(payload.get("output_feature_name", "delta")),
            labels={str(name): _spec_value_from_dict(value) for name, value in dict(payload.get("labels", {})).items()},
            propagate_from=str(payload.get("propagate_from", "positive")),
        )


@dataclass(frozen=True, slots=True)
class LabelMapSpec(OperationSpec):
    """Remap one label vocabulary into a new derived label set."""
    source: Any = None
    mapping: Mapping[str, Any] = field(default_factory=dict)
    output_name: str = "mapped_label"
    strict: bool = True
    default_value: Any = None

    kind: ClassVar[str] = "label_map"

    def __post_init__(self) -> None:
        if not str(self.output_name).strip():
            raise SpecValidationError("LabelMapSpec output_name cannot be empty")

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return _runtime_secrets_from_refs(self.source)

    def runtime_spec(self) -> Any | None:
        return _analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelMapSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            source=_spec_value_from_dict(payload.get("source")),
            mapping={str(key): value for key, value in dict(payload.get("mapping", {})).items()},
            output_name=str(payload.get("output_name", "mapped_label")),
            strict=bool(payload.get("strict", True)),
            default_value=payload.get("default_value"),
        )


@dataclass(frozen=True, slots=True)
class LabelFieldsSpec(OperationSpec):
    """Extract named fields from a structured label payload."""
    source: Any = None
    fields: Mapping[str, str] = field(default_factory=dict)
    strict: bool = True

    kind: ClassVar[str] = "label_fields"

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return _runtime_secrets_from_refs(self.source)

    def runtime_spec(self) -> Any | None:
        return _analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LabelFieldsSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            source=_spec_value_from_dict(payload.get("source")),
            fields={str(key): str(value) for key, value in dict(payload.get("fields", {})).items()},
            strict=bool(payload.get("strict", True)),
        )


@dataclass(frozen=True, slots=True)
class ActivationPatchSpec(OperationSpec):
    """Planned model-bound intervention spec for activation patching.

    This spec is serializable today but execution is not implemented yet.
    """
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


@dataclass(frozen=True, slots=True)
class ReportSpec(OperationSpec):
    """Package existing step or artifact outputs into a report artifact."""
    inputs: Sequence[Any] = field(default_factory=tuple)
    template: str = "summary"
    output_dir: str | None = None

    kind: ClassVar[str] = "report"

    def runtime_spec(self) -> Any | None:
        return _analysis_runtime_spec()

    def semantic_dict(self) -> dict[str, Any]:
        data = super(ReportSpec, self).semantic_dict()
        data.pop("output_dir", None)
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReportSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            inputs=tuple(_spec_value_from_dict(value) for value in payload.get("inputs", ())),
            template=str(payload.get("template", "summary")),
            output_dir=payload.get("output_dir"),
        )


@dataclass(frozen=True, slots=True)
class TransformSpec(OperationSpec):
    """Run a user-defined transform function over named runtime inputs."""

    builder: TransformBuilder | None = None
    inputs: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "transform"

    def __post_init__(self) -> None:
        if self.builder is None:
            raise SpecValidationError("TransformSpec requires a builder")
        if not self.inputs:
            raise SpecValidationError("TransformSpec requires at least one named input")
        empty = sorted(name for name in self.inputs if not str(name).strip())
        if empty:
            raise SpecValidationError(f"TransformSpec input names cannot be empty: {empty}")

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return _runtime_secrets_from_refs(self.inputs)

    def runtime_spec(self) -> Any | None:
        runtime_spec = _analysis_runtime_spec()
        from pipelines_v2.engine.base import PythonRuntimeSpec

        if not isinstance(runtime_spec, PythonRuntimeSpec):
            return runtime_spec
        return PythonRuntimeSpec(
            python_version=runtime_spec.python_version,
            pip_packages=runtime_spec.pip_packages,
            env=dict(runtime_spec.env),
            secrets=runtime_spec.secrets,
            local_python_sources=_merge_string_tuples(
                runtime_spec.local_python_sources,
                self.builder.local_python_sources,
            ),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TransformSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            builder=TransformBuilder.from_dict(dict(payload["builder"])),
            inputs={str(key): _spec_value_from_dict(value) for key, value in dict(payload.get("inputs", {})).items()},
        )


def _capture_site_from_dict(payload: dict[str, Any]) -> CaptureSite:
    if "site" in payload:
        return ResidualSite.from_dict(payload)
    if "record" in payload:
        return MoERoutingSite.from_dict(payload)
    raise SpecValidationError(f"Unknown capture site payload: {payload}")


def _analysis_runtime_spec() -> Any:
    from pipelines_v2.engine.base import PythonRuntimeSpec

    return PythonRuntimeSpec(
        pip_packages=(
            "numpy",
            "scikit-learn",
            "safetensors",
            "pyarrow",
            "psycopg[binary]",
        ),
        local_python_sources=("pipelines_v2",),
    )


def _callable_import_ref(
    function: Any,
    *,
    local_python_sources: Sequence[str] | None,
    label: str,
) -> tuple[str, tuple[str, ...]]:
    if not callable(function):
        raise TypeError(f"{label}.from_function(...) expects a callable")
    qualname = str(getattr(function, "__qualname__", getattr(function, "__name__", "")))
    if "<locals>" in qualname or "." in qualname:
        raise SpecValidationError(f"{label} must be top-level named functions")
    source_file = inspect.getsourcefile(function)
    if source_file is None:
        raise SpecValidationError(f"Could not determine source file for {label[:-1].lower()}")
    source_path = Path(source_file).resolve()
    workspace_root = find_workspace_root(source_path)

    if local_python_sources is None:
        try:
            relative_path = source_path.relative_to(workspace_root)
        except ValueError as exc:
            raise SpecValidationError(f"{label} source file must live under the current workspace") from exc
        sources = (".",)
    else:
        sources = tuple(str(source) for source in local_python_sources if str(source).strip())
        if not sources:
            raise SpecValidationError(f"{label} local_python_sources cannot be empty when provided")
        relative_path = None
        for source in sources:
            source_root = resolve_workspace_path(source, workspace_root=workspace_root)
            try:
                relative_path = source_path.relative_to(source_root)
                break
            except ValueError:
                continue
        if relative_path is None:
            raise SpecValidationError(
                f"{label} source file {source_path} is not under any declared local_python_sources: {list(sources)}"
            )

    if relative_path.suffix != ".py":
        raise SpecValidationError(f"{label} source file must be a Python module")
    module_name = ".".join(relative_path.with_suffix("").parts)
    return f"{module_name}:{getattr(function, '__name__', '')}", sources


def _load_importable_function(
    import_path: str,
    *,
    label: str,
    local_python_sources: Sequence[str] = (),
) -> Any:
    module_name, _, function_name = import_path.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_name = str(getattr(exc, "name", "") or "")
        if missing_name and missing_name != module_name and not module_name.startswith(f"{missing_name}."):
            raise
        workspace_root = find_workspace_root()
        source_paths = [
            str(resolve_workspace_path(source, workspace_root=workspace_root))
            for source in local_python_sources
        ]
        added_paths: list[str] = []
        for source_path in source_paths:
            if source_path not in sys.path:
                sys.path.insert(0, source_path)
                added_paths.append(source_path)
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as retry_exc:
            raise SpecValidationError(
                f"{label} module {module_name!r} could not be imported. "
                f"Tried local_python_sources={list(local_python_sources)}"
            ) from retry_exc
        finally:
            for source_path in reversed(added_paths):
                try:
                    sys.path.remove(source_path)
                except ValueError:
                    pass
    try:
        return getattr(module, function_name)
    except AttributeError as exc:
        raise SpecValidationError(
            f"{label} function {function_name!r} is not defined in module {module_name!r}"
        ) from exc


def spec_uses_section_token_selector(value: Any) -> bool:
    """Return whether a spec or nested value uses ``TokenSelector.section(...)``."""
    return _contains_section_token_selector(value)


def _runtime_secrets_from_refs(*values: Any) -> tuple[RuntimeSecret, ...]:
    secrets: dict[str, RuntimeSecret] = {}
    for value in values:
        for secret in _iter_runtime_secrets(value):
            secrets.setdefault(secret.env_var, secret)
    return tuple(secrets[key] for key in sorted(secrets))


def _iter_runtime_secrets(value: Any) -> tuple[RuntimeSecret, ...]:
    if value is None:
        return ()
    if hasattr(value, "runtime_secrets"):
        return tuple(value.runtime_secrets())
    if isinstance(value, tuple | list):
        secrets: list[RuntimeSecret] = []
        for item in value:
            secrets.extend(_iter_runtime_secrets(item))
        return tuple(secrets)
    if isinstance(value, dict):
        secrets = []
        for item in value.values():
            secrets.extend(_iter_runtime_secrets(item))
        return tuple(secrets)
    return ()


def _contains_section_token_selector(value: Any) -> bool:
    if isinstance(value, TokenSelector):
        return value.kind == "section"
    if value is None:
        return False
    if hasattr(value, "__dataclass_fields__"):
        return any(_contains_section_token_selector(getattr(value, field_name)) for field_name in value.__dataclass_fields__)
    if isinstance(value, tuple | list):
        return any(_contains_section_token_selector(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_section_token_selector(item) for item in value.values())
    return False


def _example_has_explicit_token_sections(example: Any) -> bool:
    metadata = getattr(example, "metadata", None)
    if not isinstance(metadata, Mapping):
        return False
    token_sections = metadata.get("token_sections")
    return isinstance(token_sections, Mapping) and bool(token_sections)


def _merge_string_tuples(*values: Sequence[str]) -> tuple[str, ...]:
    merged: list[str] = []
    for value in values:
        for item in value:
            normalized = str(item)
            if normalized not in merged:
                merged.append(normalized)
    return tuple(merged)


def _spec_value_from_dict(value: Any) -> Any:
    if isinstance(value, list):
        return [_spec_value_from_dict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_spec_value_from_dict(item) for item in value)
    if not isinstance(value, dict):
        return value

    kind = value.get("kind")
    if kind == LabelSet.kind:
        return LabelSet.from_dict(value)
    if kind == CaseSet.kind:
        return CaseSet.from_dict(value)
    if kind == LabelPredicate.kind:
        return LabelPredicate.from_dict(value)

    from pipelines_v2.storage.artifacts import (
        ArtifactLabelRef,
        CaptureArtifact,
        FeatureLayerRef,
        FeatureRef,
        OperationArtifact,
    )
    from pipelines_v2.workflow import StepFeatureRef, StepLabelRef, StepRef

    if kind == FeatureRef.kind:
        return FeatureRef.from_dict(value)
    if kind == FeatureLayerRef.kind:
        return FeatureLayerRef.from_dict(value)
    if kind == ArtifactLabelRef.kind:
        return ArtifactLabelRef.from_dict(value)
    if kind == CaptureArtifact.kind:
        return CaptureArtifact.from_dict(value)
    if kind == OperationArtifact.kind:
        return OperationArtifact.from_dict(value)
    if kind == StepRef.kind:
        return StepRef.from_dict(value)
    if kind == StepFeatureRef.kind:
        return StepFeatureRef.from_dict(value)
    if kind == StepLabelRef.kind:
        return StepLabelRef.from_dict(value)
    return {str(key): _spec_value_from_dict(item) for key, item in value.items()}
