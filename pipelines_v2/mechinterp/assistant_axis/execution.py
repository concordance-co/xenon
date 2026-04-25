"""Execution for Assistant Axis specs.

The functions here keep assistant-axis-specific semantics in
``pipelines_v2.mechinterp`` while reusing the generic projection data model.
Vector-producing operations return canonical ``coordinate_result`` payloads;
scoring delegates to ``ProjectionSpec`` and only re-tags the result with
assistant-axis metadata.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.operations.execution.common import (
    OperationExecutionResult,
    feature_matrices,
    resolve_values_map,
)
from pipelines_v2.operations.execution.projections import run_projection
from pipelines_v2.operations.projections import ProjectionSpec
from pipelines_v2.operations.projections._coordinates import load_coordinate_import_payload

from .specs import (
    AssistantAxisPrecomputedCoordinateSpec,
    AssistantAxisScoreSpec,
    AssistantAxisVectorSpec,
    assistant_axis_model_config,
    _hf_token,
)


def run_assistant_axis_precomputed_coordinate(spec: AssistantAxisPrecomputedCoordinateSpec) -> OperationExecutionResult:
    """Download a released Assistant Axis artifact and normalize it for scoring.

    The downloaded tensor or axis dictionary is converted into the canonical
    coordinate payload shape used throughout projection operations:
    ``{"kind": "coordinate_result", "layers": {"<layer>": {"vector": ...}}}``.
    """

    from huggingface_hub import hf_hub_download

    config = assistant_axis_model_config(spec.model_id)
    filename = spec.resolved_filename()
    layer = spec.resolved_layer()
    path = hf_hub_download(
        repo_id=spec.repo_id,
        filename=filename,
        repo_type="dataset",
        token=_hf_token(spec.token_env_var),
    )
    metadata = {
        "source": "precomputed_huggingface",
        "repo_id": spec.repo_id,
        "filename": filename,
        "model_id": spec.model_id,
        **(config or {}),
        **dict(spec.metadata),
    }
    payload = load_coordinate_import_payload(
        path=path,
        format="torch_tensor_or_axis_dict",
        select_layer=layer,
        normalize=spec.normalize,
        name=spec.name,
        metadata=metadata,
    )
    return OperationExecutionResult(
        payload=payload,
        example_coverage={
            "materialized": True,
            "example_count": 0,
            "example_keys": [],
        },
    )


def run_assistant_axis_vector(spec: AssistantAxisVectorSpec) -> OperationExecutionResult:
    """Compute an Assistant Axis direction from captured activation features.

    The feature source must already contain the token span requested by
    ``spec.tokens``. The default is the generated response section, matching the
    usual Assistant Axis setup where response activations are contrasted between
    ordinary assistant answers and role-playing answers.
    """

    if spec.role_by is None:
        raise SpecValidationError("AssistantAxisVectorSpec requires role_by")
    if spec.default_when is None:
        raise SpecValidationError("AssistantAxisVectorSpec requires default_when")

    matrices, example_keys = feature_matrices(
        spec.feature,
        layers=tuple(spec.layers) if spec.layers else None,
        token_selector=spec.tokens,
        token_pooling=spec.pooling,
    )
    role_values = {str(key): value for key, value in resolve_values_map(spec.role_by, label="role_by").items()}
    default_keys = {str(key) for key in spec.default_when.resolve_example_keys() if str(key) in set(example_keys)}
    if len(default_keys) < int(spec.min_default_examples):
        raise SpecValidationError(
            "AssistantAxisVectorSpec default_when matched too few examples: "
            f"{len(default_keys)} < {int(spec.min_default_examples)}"
        )

    role_allowed_keys = _role_allowed_keys(spec=spec, example_keys=example_keys, default_keys=default_keys)
    if not role_allowed_keys:
        raise SpecValidationError("AssistantAxisVectorSpec found no role-playing examples")

    config = assistant_axis_model_config(spec.model_id)
    warnings = []
    if spec.model_id is not None and config is None:
        warnings.append(
            "Unknown Assistant Axis model: best layer and activation-cap configuration should be discovered."
        )

    layers: dict[str, Any] = {}
    for layer, matrix in matrices.items():
        index_by_key = {str(key): index for index, key in enumerate(example_keys)}
        default_indices = [index_by_key[key] for key in sorted(default_keys)]
        role_indices_by_name: dict[str, list[int]] = defaultdict(list)
        for key in sorted(role_allowed_keys):
            role_name = str(role_values.get(key, ""))
            if not role_name or key not in index_by_key:
                continue
            role_indices_by_name[role_name].append(index_by_key[key])

        role_vectors: list[np.ndarray] = []
        dropped_roles: dict[str, int] = {}
        retained_counts: dict[str, int] = {}
        for role_name, indices in sorted(role_indices_by_name.items()):
            count = len(indices)
            if count < int(spec.min_role_examples_per_role):
                dropped_roles[role_name] = count
                continue
            retained_counts[role_name] = count
            role_vectors.append(matrix[np.asarray(indices, dtype=np.int64)].mean(axis=0))

        if not role_vectors:
            raise SpecValidationError(
                "AssistantAxisVectorSpec retained no roles after min_role_examples_per_role filtering"
            )

        default_mean = matrix[np.asarray(default_indices, dtype=np.int64)].mean(axis=0)
        role_mean = np.stack(role_vectors, axis=0).mean(axis=0)
        raw_axis = (default_mean - role_mean).astype(np.float32)
        vector, norm = _normalize_vector(raw_axis, normalize=spec.normalize)
        layers[str(layer)] = {
            "vector": vector.astype(np.float32).tolist(),
            "raw_vector": raw_axis.tolist(),
            "norm": float(norm),
            "default_count": len(default_indices),
            "role_example_count": int(sum(retained_counts.values())),
            "role_vector_count": len(role_vectors),
            "retained_role_counts": retained_counts,
            "dropped_role_counts": dropped_roles,
        }

    payload = {
        "kind": "coordinate_result",
        "coordinate_kind": "direction",
        "name": spec.name,
        "layers": layers,
        "metadata": {
            "source": "assistant_axis_vector_spec",
            "formula": "mean(default_response_activations) - mean(per_role_role_playing_vectors)",
            "model_id": spec.model_id,
            "known_model_config": config,
            "warnings": warnings,
            **dict(spec.metadata),
        },
        "summary": {
            "layer_count": len(layers),
            "default_count": len(default_keys),
            "role_candidate_count": len(role_allowed_keys),
            "score_filtered": spec.score_by is not None,
            "score_values": list(spec.score_values),
            "normalized": spec.normalize,
        },
    }
    return OperationExecutionResult(
        payload=payload,
        example_coverage={
            "materialized": True,
            "example_count": len(example_keys),
            "example_keys": list(example_keys),
        },
    )


def run_assistant_axis_score(spec: AssistantAxisScoreSpec) -> OperationExecutionResult:
    """Score selected sections against an Assistant Axis coordinate.

    This function exists so assistant-axis workflows can use a domain-specific
    spec while preserving the generic projection execution path, payload rows,
    summaries, and optional emitted labels.
    """

    projection = ProjectionSpec(
        feature=spec.feature,
        coordinates=[spec.axis],
        slices=spec.slices,
        rows=spec.rows,
        layers=[spec.resolved_layer()],
        pooling=spec.pooling,
        metric=spec.metric,
        summaries=spec.summaries,
        emit_labels=spec.emit_labels,
    )
    result = run_projection(projection)
    payload = dict(result.payload)
    payload["kind"] = "assistant_axis_score_result"
    payload["assistant_axis"] = {
        "model_id": spec.model_id,
        "layer": spec.resolved_layer(),
        "known_model_config": assistant_axis_model_config(spec.model_id),
    }
    return OperationExecutionResult(
        payload=payload,
        labels=result.labels,
        metadata=result.metadata,
        example_coverage=result.example_coverage,
    )


def _role_allowed_keys(
    *,
    spec: AssistantAxisVectorSpec,
    example_keys: Sequence[str],
    default_keys: set[str],
) -> set[str]:
    """Return role-play example keys after role/default and score filtering."""

    feature_keys = {str(key) for key in example_keys}
    if spec.role_when is None:
        role_keys = feature_keys - default_keys
    else:
        role_keys = {str(key) for key in spec.role_when.resolve_example_keys() if str(key) in feature_keys}
    if spec.score_by is None:
        return role_keys
    score_values = resolve_values_map(spec.score_by, label="score_by")
    allowed_scores = {str(value) for value in spec.score_values}
    return {
        key
        for key in role_keys
        if key in score_values and str(score_values[key]) in allowed_scores
    }


def _normalize_vector(vector: np.ndarray, *, normalize: str) -> tuple[np.ndarray, float]:
    """Normalize a computed assistant-axis vector and return its raw norm."""

    raw = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(raw))
    mode = str(normalize).strip().lower()
    if mode in {"none", ""}:
        return raw, norm
    if mode == "l2":
        if norm <= 0:
            return raw, norm
        return (raw / norm).astype(np.float32), norm
    raise SpecValidationError(f"Unsupported Assistant Axis normalization mode: {normalize!r}")
