"""Emotion-vector operation specs.

These specs encode the reusable pieces from the Transformer Circuits emotions
work as first-class mech-interp helpers while delegating generic scoring and
intervention mechanics to the existing projection and patching stacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

from pipelines_v2.core.types import OperationSpec, RuntimeSecret, SpecValidationError
from pipelines_v2.mechinterp._shared import hf_token_from_env, merge_runtime_secrets
from pipelines_v2.operations.common._shared import (
    analysis_runtime_spec,
    runtime_secrets_from_refs,
    spec_value_from_dict,
)
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector
from pipelines_v2.operations.projections import SectionSelector


EMOTION_VECTOR_SPACE_KIND = "emotion_vector_space_result"


@dataclass(frozen=True, slots=True)
class EmotionPrecomputedVectorSpaceSpec(OperationSpec):
    """Load a released or user-provided emotion vector space.

    The artifact should contain one or more named concepts, each with vectors
    keyed by layer. JSON is the canonical interchange format. NPZ is supported
    for compact local/HF artifacts with arrays named either
    ``layer_{layer}__{concept}`` or ``{concept}`` when ``select_layer`` is set.
    """

    path: str | None = None
    repo_id: str | None = None
    filename: str | None = None
    revision: str | None = None
    format: str = "json"
    select_layer: int | None = None
    normalize: str = "l2"
    vector_space_kind: str = "story"
    token_env_var: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "emotion_precomputed_vector_space"

    def __post_init__(self) -> None:
        if self.path is None and (self.repo_id is None or self.filename is None):
            raise SpecValidationError(
                "EmotionPrecomputedVectorSpaceSpec requires path=... or repo_id=... plus filename=..."
            )
        if self.path is not None:
            object.__setattr__(self, "path", str(self.path))
        if self.repo_id is not None:
            object.__setattr__(self, "repo_id", str(self.repo_id))
        if self.filename is not None:
            object.__setattr__(self, "filename", str(self.filename))
        if self.revision is not None:
            object.__setattr__(self, "revision", str(self.revision))
        if self.select_layer is not None:
            object.__setattr__(self, "select_layer", int(self.select_layer))
        if self.token_env_var is not None:
            object.__setattr__(self, "token_env_var", str(self.token_env_var))
        object.__setattr__(self, "format", str(self.format))
        object.__setattr__(self, "normalize", str(self.normalize))
        object.__setattr__(self, "vector_space_kind", str(self.vector_space_kind))
        object.__setattr__(self, "metadata", {str(key): value for key, value in dict(self.metadata).items()})

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        if self.token_env_var is None:
            return ()
        return (RuntimeSecret(env_var=self.token_env_var),)

    def runtime_spec(self) -> Any | None:
        from pipelines_v2.engine.base import PythonRuntimeSpec

        packages = ["numpy"]
        if self.repo_id is not None:
            packages.append("huggingface_hub")
        if str(self.format).lower() in {"torch", "pt", "torch_tensor"}:
            packages.append("torch")
        base = analysis_runtime_spec(extra_pip_packages=tuple(packages))
        if not isinstance(base, PythonRuntimeSpec):
            return base
        return PythonRuntimeSpec(
            python_version=base.python_version,
            pip_packages=base.pip_packages,
            env=dict(base.env),
            secrets=merge_runtime_secrets(base.secrets, self.runtime_secrets()),
            local_python_sources=base.local_python_sources,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmotionPrecomputedVectorSpaceSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            path=str(payload["path"]) if payload.get("path") is not None else None,
            repo_id=str(payload["repo_id"]) if payload.get("repo_id") is not None else None,
            filename=str(payload["filename"]) if payload.get("filename") is not None else None,
            revision=str(payload["revision"]) if payload.get("revision") is not None else None,
            format=str(payload.get("format", "json")),
            select_layer=int(payload["select_layer"]) if payload.get("select_layer") is not None else None,
            normalize=str(payload.get("normalize", "l2")),
            vector_space_kind=str(payload.get("vector_space_kind", "story")),
            token_env_var=str(payload["token_env_var"]) if payload.get("token_env_var") is not None else None,
            metadata={str(key): value for key, value in dict(payload.get("metadata", {})).items()},
        )


@dataclass(frozen=True, slots=True)
class EmotionVectorSpaceSpec(OperationSpec):
    """Derive story-style emotion concept vectors from captured activations.

    The default token selector follows the paper's story-vector recipe:
    average activations over each story beginning at token 50, average per
    emotion concept, subtract the across-concept mean, and optionally project
    out top neutral-transcript principal components.
    """

    feature: Any = None
    concept_by: Any = None
    layers: Sequence[int] = field(default_factory=tuple)
    tokens: TokenSelector = field(default_factory=lambda: TokenSelector.slice(50, None))
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)
    neutral_feature: Any = None
    neutral_variance_threshold: float | None = 0.5
    normalize: str = "l2"
    min_examples_per_concept: int = 1
    vector_space_kind: str = "story"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "emotion_vector_space"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(int(layer) for layer in self.layers))
        object.__setattr__(self, "min_examples_per_concept", int(self.min_examples_per_concept))
        if self.neutral_variance_threshold is not None:
            threshold = float(self.neutral_variance_threshold)
            if threshold <= 0.0 or threshold > 1.0:
                raise SpecValidationError("neutral_variance_threshold must be in (0, 1]")
            object.__setattr__(self, "neutral_variance_threshold", threshold)
        object.__setattr__(self, "normalize", str(self.normalize))
        object.__setattr__(self, "vector_space_kind", str(self.vector_space_kind))
        object.__setattr__(self, "metadata", {str(key): value for key, value in dict(self.metadata).items()})

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.feature, self.concept_by, self.neutral_feature)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmotionVectorSpaceSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            concept_by=spec_value_from_dict(payload.get("concept_by")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "slice", "value": {"start": 50, "stop": None}})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
            neutral_feature=spec_value_from_dict(payload.get("neutral_feature")),
            neutral_variance_threshold=(
                float(payload["neutral_variance_threshold"])
                if payload.get("neutral_variance_threshold") is not None
                else None
            ),
            normalize=str(payload.get("normalize", "l2")),
            min_examples_per_concept=int(payload.get("min_examples_per_concept", 1)),
            vector_space_kind=str(payload.get("vector_space_kind", "story")),
            metadata={str(key): value for key, value in dict(payload.get("metadata", {})).items()},
        )


@dataclass(frozen=True, slots=True)
class EmotionScoreSpec(OperationSpec):
    """Score captured sections against selected emotions from a vector space."""

    feature: Any = None
    vector_space: Any = None
    concepts: Sequence[str] = field(default_factory=tuple)
    layers: Sequence[int] = field(default_factory=tuple)
    slices: SectionSelector = field(default_factory=SectionSelector.all)
    rows: Any = None
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)
    metric: str = "signed_dot"
    summaries: Sequence[str] = field(default_factory=lambda: ("mean", "min", "max", "std", "trend"))
    emit_labels: bool = True

    kind: ClassVar[str] = "emotion_score"

    def __post_init__(self) -> None:
        object.__setattr__(self, "concepts", tuple(str(item) for item in self.concepts if str(item).strip()))
        object.__setattr__(self, "layers", tuple(int(layer) for layer in self.layers))
        object.__setattr__(self, "metric", str(self.metric))
        object.__setattr__(self, "summaries", tuple(str(item) for item in self.summaries))

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.feature, self.vector_space, self.rows)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmotionScoreSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            vector_space=spec_value_from_dict(payload.get("vector_space")),
            concepts=tuple(str(item) for item in payload.get("concepts", ())),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            slices=SectionSelector.from_dict(payload.get("slices")),
            rows=spec_value_from_dict(payload.get("rows")),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
            metric=str(payload.get("metric", "signed_dot")),
            summaries=tuple(str(item) for item in payload.get("summaries", ("mean", "min", "max", "std", "trend"))),
            emit_labels=bool(payload.get("emit_labels", True)),
        )


@dataclass(frozen=True, slots=True)
class EmotionDirectionSpec(OperationSpec):
    """Export one emotion vector as a direction usable by AddDirectionPatch."""

    vector_space: Any = None
    concept: str = ""
    layers: Sequence[int] = field(default_factory=tuple)
    source: str = "vector"
    scale: float = 1.0
    residual_norm_by_layer: Mapping[int, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "emotion_direction"

    def __post_init__(self) -> None:
        concept = str(self.concept or "").strip()
        if not concept:
            raise SpecValidationError("EmotionDirectionSpec requires concept")
        object.__setattr__(self, "concept", concept)
        object.__setattr__(self, "layers", tuple(int(layer) for layer in self.layers))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(
            self,
            "residual_norm_by_layer",
            {int(layer): float(value) for layer, value in dict(self.residual_norm_by_layer).items()},
        )
        object.__setattr__(self, "metadata", {str(key): value for key, value in dict(self.metadata).items()})

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.vector_space)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmotionDirectionSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            vector_space=spec_value_from_dict(payload.get("vector_space")),
            concept=str(payload.get("concept", "")),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            source=str(payload.get("source", "vector")),
            scale=float(payload.get("scale", 1.0)),
            residual_norm_by_layer={int(layer): float(value) for layer, value in dict(payload.get("residual_norm_by_layer", {})).items()},
            metadata={str(key): value for key, value in dict(payload.get("metadata", {})).items()},
        )


@dataclass(frozen=True, slots=True)
class EmotionGeometrySpec(OperationSpec):
    """Compute geometry diagnostics over an emotion vector space."""

    vector_space: Any = None
    concepts: Sequence[str] = field(default_factory=tuple)
    layers: Sequence[int] = field(default_factory=tuple)
    pca_components: int = 3
    cluster_count: int | None = None

    kind: ClassVar[str] = "emotion_geometry"

    def __post_init__(self) -> None:
        object.__setattr__(self, "concepts", tuple(str(item) for item in self.concepts if str(item).strip()))
        object.__setattr__(self, "layers", tuple(int(layer) for layer in self.layers))
        object.__setattr__(self, "pca_components", int(self.pca_components))
        if self.cluster_count is not None:
            object.__setattr__(self, "cluster_count", int(self.cluster_count))

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.vector_space)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec(extra_pip_packages=("scikit-learn",))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EmotionGeometrySpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            vector_space=spec_value_from_dict(payload.get("vector_space")),
            concepts=tuple(str(item) for item in payload.get("concepts", ())),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            pca_components=int(payload.get("pca_components", 3)),
            cluster_count=int(payload["cluster_count"]) if payload.get("cluster_count") is not None else None,
        )


_hf_token = hf_token_from_env


__all__ = [
    "EMOTION_VECTOR_SPACE_KIND",
    "EmotionDirectionSpec",
    "EmotionGeometrySpec",
    "EmotionPrecomputedVectorSpaceSpec",
    "EmotionScoreSpec",
    "EmotionVectorSpaceSpec",
    "_hf_token",
]
