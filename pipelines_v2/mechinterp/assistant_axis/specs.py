"""Assistant Axis operation specs and known-model metadata.

The assistant-axis APIs are domain-specific mech-interp helpers layered on top
of the generic projection primitives in ``pipelines_v2.operations.projections``.
They encode the Yora/Assistant Axis workflow as first-class operation specs:
load a released assistant-axis direction, derive a direction from captured role
play/default activations, or score captured text spans against that direction.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

from pipelines_v2.core.types import OperationSpec, RuntimeSecret, SpecValidationError
from pipelines_v2.operations.common._shared import (
    analysis_runtime_spec,
    runtime_secrets_from_refs,
    spec_value_from_dict,
)
from pipelines_v2.operations.common.tokens import TokenPooling, TokenSelector
from pipelines_v2.operations.projections import SectionSelector

ASSISTANT_AXIS_VECTOR_REPO = "lu-christina/assistant-axis-vectors"


# These are the model/layer choices released with the Assistant Axis assets.
# Unknown models remain usable, but callers must provide layers explicitly and
# should treat scores as exploratory until layer and capping calibration exist.
KNOWN_ASSISTANT_AXIS_MODELS: dict[str, dict[str, Any]] = {
    "google/gemma-2-27b-it": {
        "short_name": "gemma-2-27b",
        "display_name": "Gemma 2 27B",
        "target_layer": 22,
        "total_layers": 46,
        "hf_axis_filename": "gemma-2-27b/assistant_axis.pt",
        "hf_default_vector_filename": "gemma-2-27b/default_vector.pt",
    },
    "gemma-2-27b": {
        "alias_for": "google/gemma-2-27b-it",
    },
    "qwen/qwen3-32b": {
        "short_name": "qwen-3-32b",
        "display_name": "Qwen 3 32B",
        "target_layer": 32,
        "total_layers": 64,
        "hf_axis_filename": "qwen-3-32b/assistant_axis.pt",
        "hf_default_vector_filename": "qwen-3-32b/default_vector.pt",
        "hf_capping_config_filename": "qwen-3-32b/capping_config.pt",
        "capping_experiment": "layers_46:54-p0.25",
    },
    "qwen/qwen-3-32b": {
        "alias_for": "qwen/qwen3-32b",
    },
    "qwen/qwen3-32b-instruct": {
        "alias_for": "qwen/qwen3-32b",
    },
    "qwen-3-32b": {
        "alias_for": "qwen/qwen3-32b",
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "short_name": "llama-3.3-70b",
        "display_name": "Llama 3.3 70B",
        "target_layer": 40,
        "total_layers": 80,
        "hf_axis_filename": "llama-3.3-70b/assistant_axis.pt",
        "hf_default_vector_filename": "llama-3.3-70b/default_vector.pt",
        "hf_capping_config_filename": "llama-3.3-70b/capping_config.pt",
        "capping_experiment": "layers_56:72-p0.25",
    },
    "meta-llama/llama-3.3-70b": {
        "alias_for": "meta-llama/llama-3.3-70b-instruct",
    },
    "llama-3.3-70b": {
        "alias_for": "meta-llama/llama-3.3-70b-instruct",
    },
    "llama-3.3-70b-instruct": {
        "alias_for": "meta-llama/llama-3.3-70b-instruct",
    },
}


def assistant_axis_model_config(model_id: str | None) -> dict[str, Any] | None:
    """Return the released Assistant Axis config for a model id or alias.

    The lookup is case-insensitive and resolves aliases such as
    ``qwen-3-32b`` to the canonical released model entry. The returned mapping
    is a copy so callers can attach it to payload metadata without mutating the
    registry.
    """

    if model_id is None:
        return None
    normalized = _normalize_model_id(model_id)
    if not normalized:
        return None
    config = KNOWN_ASSISTANT_AXIS_MODELS.get(normalized)
    if config is None:
        return None
    alias_for = config.get("alias_for")
    if alias_for is not None:
        return dict(KNOWN_ASSISTANT_AXIS_MODELS[str(alias_for)])
    return dict(config)


@dataclass(frozen=True, slots=True)
class AssistantAxisPrecomputedCoordinateSpec(OperationSpec):
    """Load a released Assistant Axis coordinate as a canonical coordinate.

    This is the shortest path for supported models: it downloads the vector
    artifact from ``repo_id`` (by default
    ``lu-christina/assistant-axis-vectors``), optionally selects one layer, and
    emits the same ``coordinate_result`` payload consumed by
    ``ProjectionSpec``. Use ``filename`` for custom repositories or
    unreleased layouts; otherwise a known ``model_id`` supplies the filename
    and target layer.
    """

    model_id: str = ""
    repo_id: str = ASSISTANT_AXIS_VECTOR_REPO
    filename: str | None = None
    select_layer: int | None = None
    normalize: str = "l2"
    name: str = "assistant_axis"
    token_env_var: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "assistant_axis_precomputed_coordinate"

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", str(self.model_id))
        object.__setattr__(self, "repo_id", str(self.repo_id))
        if self.filename is not None:
            object.__setattr__(self, "filename", str(self.filename))
        if self.select_layer is not None:
            object.__setattr__(self, "select_layer", int(self.select_layer))
        if self.token_env_var is not None:
            object.__setattr__(self, "token_env_var", str(self.token_env_var))
        object.__setattr__(self, "metadata", {str(key): value for key, value in dict(self.metadata).items()})

        config = assistant_axis_model_config(self.model_id)
        if config is None and self.filename is None:
            raise SpecValidationError(
                "AssistantAxisPrecomputedCoordinateSpec requires either a known model_id "
                "or an explicit Hugging Face filename"
            )

    def resolved_filename(self) -> str:
        """Return the HF artifact filename after model-id alias resolution."""

        if self.filename is not None:
            return str(self.filename)
        config = assistant_axis_model_config(self.model_id)
        if config is None:
            raise SpecValidationError(f"No precomputed Assistant Axis filename known for {self.model_id!r}")
        return str(config["hf_axis_filename"])

    def resolved_layer(self) -> int | None:
        """Return the selected vector layer, if this coordinate is layer-bound."""

        if self.select_layer is not None:
            return int(self.select_layer)
        config = assistant_axis_model_config(self.model_id)
        return int(config["target_layer"]) if config is not None else None

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        """Declare the optional HF token environment variable for remote runs."""

        if self.token_env_var is None:
            return ()
        return (RuntimeSecret(env_var=self.token_env_var),)

    def runtime_spec(self) -> Any | None:
        from pipelines_v2.engine.base import PythonRuntimeSpec

        base = analysis_runtime_spec(extra_pip_packages=("torch", "huggingface_hub"))
        if not isinstance(base, PythonRuntimeSpec):
            return base
        return PythonRuntimeSpec(
            python_version=base.python_version,
            pip_packages=base.pip_packages,
            env=dict(base.env),
            secrets=_merge_runtime_secrets(base.secrets, self.runtime_secrets()),
            local_python_sources=base.local_python_sources,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssistantAxisPrecomputedCoordinateSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            model_id=str(payload.get("model_id", "")),
            repo_id=str(payload.get("repo_id", ASSISTANT_AXIS_VECTOR_REPO)),
            filename=str(payload["filename"]) if payload.get("filename") is not None else None,
            select_layer=int(payload["select_layer"]) if payload.get("select_layer") is not None else None,
            normalize=str(payload.get("normalize", "l2")),
            name=str(payload.get("name", "assistant_axis")),
            token_env_var=str(payload["token_env_var"]) if payload.get("token_env_var") is not None else None,
            metadata={str(key): value for key, value in dict(payload.get("metadata", {})).items()},
        )


@dataclass(frozen=True, slots=True)
class AssistantAxisVectorSpec(OperationSpec):
    """Derive an Assistant Axis coordinate from captured activations.

    The spec expects a feature artifact containing response activations,
    usually residual activations over the generated-token section. Examples are
    split into default-assistant responses via ``default_when`` and role-play
    responses via ``role_when`` or the complement of ``default_when``. Role
    examples are grouped by ``role_by``; each retained role contributes one
    mean vector, and the final direction is:

    ``mean(default responses) - mean(per-role role-play means)``

    ``score_by`` and ``score_values`` optionally restrict role-play examples to
    high-quality or high-intensity rows before the direction is computed.
    """

    feature: Any = None
    role_by: Any = None
    default_when: Any = None
    role_when: Any = None
    score_by: Any = None
    score_values: Sequence[Any] = field(default_factory=lambda: (3,))
    min_role_examples_per_role: int = 50
    min_default_examples: int = 1
    layers: Sequence[int] = field(default_factory=tuple)
    tokens: TokenSelector = field(default_factory=lambda: TokenSelector.section("generated"))
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)
    normalize: str = "l2"
    name: str = "assistant_axis"
    model_id: str | None = None
    warn_unknown_model: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    kind: ClassVar[str] = "assistant_axis_vector"

    def __post_init__(self) -> None:
        object.__setattr__(self, "min_role_examples_per_role", int(self.min_role_examples_per_role))
        object.__setattr__(self, "min_default_examples", int(self.min_default_examples))
        object.__setattr__(self, "layers", tuple(int(layer) for layer in self.layers))
        object.__setattr__(self, "score_values", tuple(self.score_values))
        object.__setattr__(self, "metadata", {str(key): value for key, value in dict(self.metadata).items()})
        if self.model_id is not None:
            object.__setattr__(self, "model_id", str(self.model_id))
        _warn_if_unknown_model(self.model_id, self.warn_unknown_model)

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(
            self.feature,
            self.role_by,
            self.default_when,
            self.role_when,
            self.score_by,
        )

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssistantAxisVectorSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            role_by=spec_value_from_dict(payload.get("role_by")),
            default_when=spec_value_from_dict(payload.get("default_when")),
            role_when=spec_value_from_dict(payload.get("role_when")),
            score_by=spec_value_from_dict(payload.get("score_by")),
            score_values=tuple(payload.get("score_values", (3,))),
            min_role_examples_per_role=int(payload.get("min_role_examples_per_role", 50)),
            min_default_examples=int(payload.get("min_default_examples", 1)),
            layers=tuple(int(layer) for layer in payload.get("layers", ())),
            tokens=TokenSelector.from_dict(payload.get("tokens", {"kind": "section", "value": "generated"})),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
            normalize=str(payload.get("normalize", "l2")),
            name=str(payload.get("name", "assistant_axis")),
            model_id=str(payload["model_id"]) if payload.get("model_id") is not None else None,
            warn_unknown_model=bool(payload.get("warn_unknown_model", True)),
            metadata={str(key): value for key, value in dict(payload.get("metadata", {})).items()},
        )


@dataclass(frozen=True, slots=True)
class AssistantAxisScoreSpec(OperationSpec):
    """Score captured activation slices against an Assistant Axis coordinate.

    This is a thin assistant-axis wrapper around ``ProjectionSpec``. It adds
    known-model layer resolution and assistant-axis metadata, then delegates the
    actual per-section pooling and dot/cosine scoring to the generic projection
    executor.
    """

    feature: Any = None
    axis: Any = None
    model_id: str | None = None
    layer: int | None = None
    slices: SectionSelector = field(default_factory=lambda: SectionSelector.named("generated"))
    rows: Any = None
    pooling: TokenPooling = field(default_factory=TokenPooling.mean)
    metric: str = "signed_dot"
    summaries: Sequence[str] = field(default_factory=lambda: ("mean", "min", "max", "std", "trend"))
    emit_labels: bool = True
    warn_unknown_model: bool = True

    kind: ClassVar[str] = "assistant_axis_score"

    def __post_init__(self) -> None:
        if self.model_id is not None:
            object.__setattr__(self, "model_id", str(self.model_id))
        if self.layer is not None:
            object.__setattr__(self, "layer", int(self.layer))
        object.__setattr__(self, "summaries", tuple(str(item) for item in self.summaries))
        _warn_if_unknown_model(self.model_id, self.warn_unknown_model)

    def resolved_layer(self) -> int:
        """Return the explicit layer or the released target layer for model_id."""

        if self.layer is not None:
            return int(self.layer)
        config = assistant_axis_model_config(self.model_id)
        if config is None:
            raise SpecValidationError(
                "AssistantAxisScoreSpec requires layer=... when model_id is not one of the known Assistant Axis models"
            )
        return int(config["target_layer"])

    def runtime_secrets(self) -> tuple[RuntimeSecret, ...]:
        return runtime_secrets_from_refs(self.feature, self.axis, self.rows)

    def runtime_spec(self) -> Any | None:
        return analysis_runtime_spec()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AssistantAxisScoreSpec":
        return cls(
            schema_version=int(payload.get("schema_version", 1)),
            feature=spec_value_from_dict(payload.get("feature")),
            axis=spec_value_from_dict(payload.get("axis")),
            model_id=str(payload["model_id"]) if payload.get("model_id") is not None else None,
            layer=int(payload["layer"]) if payload.get("layer") is not None else None,
            slices=SectionSelector.from_dict(payload.get("slices")),
            rows=spec_value_from_dict(payload.get("rows")),
            pooling=TokenPooling.from_dict(payload.get("pooling", {"kind": "mean"})),
            metric=str(payload.get("metric", "signed_dot")),
            summaries=tuple(str(item) for item in payload.get("summaries", ("mean", "min", "max", "std", "trend"))),
            emit_labels=bool(payload.get("emit_labels", True)),
            warn_unknown_model=bool(payload.get("warn_unknown_model", True)),
        )


def _normalize_model_id(model_id: str | None) -> str:
    return str(model_id or "").strip().lower()


def _warn_if_unknown_model(model_id: str | None, enabled: bool) -> None:
    if not enabled or model_id is None or assistant_axis_model_config(model_id) is not None:
        return
    warnings.warn(
        "Assistant Axis does not know the best layer or activation-cap configuration "
        f"for model {model_id!r}. Treat the result as exploratory until layer and capping "
        "calibration have been discovered for this model.",
        stacklevel=3,
    )


def _hf_token(token_env_var: str | None) -> str | None:
    if token_env_var is None:
        return None
    token = os.environ.get(token_env_var)
    if not token:
        raise RuntimeError(f"Missing required environment variable: {token_env_var}")
    return token


def _merge_runtime_secrets(*values: Sequence[RuntimeSecret]) -> tuple[RuntimeSecret, ...]:
    merged: dict[str, RuntimeSecret] = {}
    for value in values:
        for secret in value:
            merged.setdefault(secret.env_var, secret)
    return tuple(merged.values())


__all__ = [
    "ASSISTANT_AXIS_VECTOR_REPO",
    "AssistantAxisPrecomputedCoordinateSpec",
    "AssistantAxisScoreSpec",
    "AssistantAxisVectorSpec",
    "KNOWN_ASSISTANT_AXIS_MODELS",
    "assistant_axis_model_config",
    "_hf_token",
]
