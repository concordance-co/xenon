"""Execution for emotion-vector mech-interp specs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.operations.execution.common import (
    OperationExecutionResult,
    feature_matrices,
    resolve_values_map,
)
from pipelines_v2.operations.common.vectors import coordinate_name_key, normalize_vector
from pipelines_v2.operations.execution.projections import run_projection
from pipelines_v2.operations.projections import ProjectionSpec

from .specs import (
    EMOTION_VECTOR_SPACE_KIND,
    EmotionDirectionSpec,
    EmotionGeometrySpec,
    EmotionPrecomputedVectorSpaceSpec,
    EmotionScoreSpec,
    EmotionVectorSpaceSpec,
    _hf_token,
)


def run_emotion_precomputed_vector_space(spec: EmotionPrecomputedVectorSpaceSpec) -> OperationExecutionResult:
    """Load a precomputed emotion vector space into the canonical payload."""

    path = _download_or_resolve_path(spec)
    payload = _load_vector_space_payload(
        path=path,
        format=spec.format,
        select_layer=spec.select_layer,
        normalize=spec.normalize,
        vector_space_kind=spec.vector_space_kind,
        metadata={
            "source": "precomputed",
            "path": str(path),
            "repo_id": spec.repo_id,
            "filename": spec.filename,
            "revision": spec.revision,
            **dict(spec.metadata),
        },
    )
    return OperationExecutionResult(
        payload=payload,
        example_coverage={"materialized": True, "example_count": 0, "example_keys": []},
    )


def run_emotion_vector_space(spec: EmotionVectorSpaceSpec) -> OperationExecutionResult:
    """Derive concept vectors by concept mean minus the across-concept mean."""

    if spec.concept_by is None:
        raise SpecValidationError("EmotionVectorSpaceSpec requires concept_by")
    matrices, example_keys = feature_matrices(
        spec.feature,
        layers=tuple(spec.layers) if spec.layers else None,
        token_selector=spec.tokens,
        token_pooling=spec.pooling,
    )
    concept_values = {str(key): str(value) for key, value in resolve_values_map(spec.concept_by, label="concept_by").items()}
    concepts = sorted({concept_values[key] for key in example_keys if key in concept_values and concept_values[key]})
    if not concepts:
        raise SpecValidationError("EmotionVectorSpaceSpec concept_by resolved no concepts for captured examples")

    neutral_components = _neutral_components_by_layer(spec)
    layers: dict[str, Any] = {}
    dropped: dict[str, int] = {}
    used_keys: set[str] = set()

    for layer, matrix in matrices.items():
        index_by_key = {str(key): index for index, key in enumerate(example_keys)}
        concept_means: dict[str, np.ndarray] = {}
        counts: dict[str, int] = {}
        for concept in concepts:
            keys = [key for key in example_keys if concept_values.get(key) == concept]
            if len(keys) < int(spec.min_examples_per_concept):
                dropped[concept] = len(keys)
                continue
            indices = [index_by_key[key] for key in keys]
            counts[concept] = len(indices)
            used_keys.update(keys)
            concept_means[concept] = matrix[np.asarray(indices, dtype=np.int64)].mean(axis=0).astype(np.float32)
        if not concept_means:
            raise SpecValidationError("EmotionVectorSpaceSpec retained no concepts after min_examples_per_concept")

        global_mean = np.stack(list(concept_means.values()), axis=0).mean(axis=0).astype(np.float32)
        layer_components = neutral_components.get(int(layer), np.zeros((0, matrix.shape[1]), dtype=np.float32))
        layer_payload: dict[str, Any] = {
            "concepts": {},
            "global_mean": global_mean.tolist(),
            "neutral_projector": {
                "component_count": int(layer_components.shape[0]),
                "variance_threshold": spec.neutral_variance_threshold,
            },
        }
        if layer_components.size:
            layer_payload["neutral_projector"]["components"] = layer_components.astype(np.float32).tolist()

        for concept, mean in sorted(concept_means.items()):
            raw = (mean - global_mean).astype(np.float32)
            if layer_components.size:
                raw = _project_out_components(raw, layer_components)
            vector, norm = normalize_vector(raw, normalize=spec.normalize, error_label="emotion vector")
            layer_payload["concepts"][concept] = {
                "vector": vector.astype(np.float32).tolist(),
                "raw_vector": raw.astype(np.float32).tolist(),
                "norm": float(norm),
                "count": int(counts[concept]),
            }
        layers[str(int(layer))] = layer_payload

    payload = {
        "kind": EMOTION_VECTOR_SPACE_KIND,
        "vector_space_kind": spec.vector_space_kind,
        "feature": _feature_name(spec.feature),
        "layers": layers,
        "metadata": {
            "source": "emotion_vector_space_spec",
            "formula": "mean(concept examples) - mean(concept means)",
            "token_selector": {"kind": spec.tokens.kind, "value": spec.tokens.value},
            "pooling": spec.pooling.kind,
            "neutral_project_out": spec.neutral_feature is not None,
            **dict(spec.metadata),
        },
        "summary": {
            "layer_count": len(layers),
            "concept_count": len(_all_concepts(layers)),
            "used_example_count": len(used_keys),
            "dropped_concepts": dropped,
            "normalized": spec.normalize,
        },
    }
    return OperationExecutionResult(
        payload=payload,
        example_coverage={
            "materialized": True,
            "example_count": len(used_keys),
            "example_keys": sorted(used_keys),
        },
    )


def run_emotion_score(spec: EmotionScoreSpec) -> OperationExecutionResult:
    """Score activation slices against selected emotion concepts."""

    vector_space = _resolve_vector_space(spec.vector_space)
    coordinates = _coordinates_from_vector_space(
        vector_space,
        concepts=spec.concepts,
        layers=spec.layers,
    )
    emotion_by_coordinate = _emotion_by_coordinate_name(coordinates)
    projection = ProjectionSpec(
        feature=spec.feature,
        coordinates=coordinates,
        slices=spec.slices,
        rows=spec.rows,
        layers=spec.layers,
        pooling=spec.pooling,
        metric=spec.metric,
        summaries=spec.summaries,
        emit_labels=spec.emit_labels,
    )
    result = run_projection(projection)
    payload = dict(result.payload)
    payload["kind"] = "emotion_score_result"
    payload["vector_space_kind"] = str(vector_space.get("vector_space_kind") or "")
    payload["emotion_vector_space"] = _vector_space_summary(vector_space)
    for row in payload.get("rows", []):
        if isinstance(row, dict):
            row["emotion"] = emotion_by_coordinate.get(str(row.get("coordinate") or ""), str(row.get("coordinate") or ""))
    for row in payload.get("example_summaries", []):
        if isinstance(row, dict):
            row["emotion"] = emotion_by_coordinate.get(str(row.get("coordinate") or ""), str(row.get("coordinate") or ""))
    return OperationExecutionResult(
        payload=payload,
        labels=result.labels,
        metadata=result.metadata,
        example_coverage=result.example_coverage,
    )


def run_emotion_direction(spec: EmotionDirectionSpec) -> OperationExecutionResult:
    """Export one emotion concept as a direction_result for steering."""

    vector_space = _resolve_vector_space(spec.vector_space)
    available_layers = _selected_layers(vector_space, layers=spec.layers)
    if not available_layers:
        raise SpecValidationError("EmotionDirectionSpec did not resolve any layers")

    layers: dict[str, Any] = {}
    for layer in available_layers:
        concept_payload = _concept_payload(vector_space, layer=layer, concept=spec.concept)
        source_name = str(spec.source).strip().lower()
        if source_name == "raw_vector":
            base = np.asarray(concept_payload.get("raw_vector"), dtype=np.float32)
        elif source_name == "vector":
            base = np.asarray(concept_payload.get("vector"), dtype=np.float32)
        else:
            raise SpecValidationError("EmotionDirectionSpec source must be 'vector' or 'raw_vector'")
        residual_norm = float(spec.residual_norm_by_layer.get(int(layer), 1.0))
        raw = (base * float(spec.scale) * residual_norm).astype(np.float32)
        unit, norm = normalize_vector(raw, normalize="l2", error_label="emotion direction")
        layers[str(layer)] = {
            "vector": unit.astype(np.float32).tolist(),
            "raw_vector": raw.tolist(),
            "norm": float(norm),
            "emotion": spec.concept,
            "source": source_name,
            "scale": float(spec.scale),
            "residual_norm": residual_norm,
        }

    payload = {
        "kind": "direction_result",
        "feature": _feature_name(spec.vector_space),
        "name": f"emotion__{coordinate_name_key(spec.concept)}",
        "layers": layers,
        "metadata": {
            "source": "emotion_direction_spec",
            "emotion": spec.concept,
            "vector_space_kind": vector_space.get("vector_space_kind"),
            "steering_units": "AddDirectionPatch strength multiplies raw_vector; use residual_norm_by_layer to encode residual-norm fractions.",
            **dict(spec.metadata),
        },
        "summary": {
            "layer_count": len(layers),
            "emotion": spec.concept,
            "source": spec.source,
            "scale": float(spec.scale),
        },
    }
    return OperationExecutionResult(
        payload=payload,
        example_coverage={"materialized": True, "example_count": 0, "example_keys": []},
    )


def run_emotion_geometry(spec: EmotionGeometrySpec) -> OperationExecutionResult:
    """Compute cosine/PCA/cluster diagnostics for a vector space."""

    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    vector_space = _resolve_vector_space(spec.vector_space)
    layers = _selected_layers(vector_space, layers=spec.layers)
    layer_results: dict[str, Any] = {}
    for layer in layers:
        concepts = _selected_concepts(vector_space, layer=layer, concepts=spec.concepts)
        matrix = np.stack(
            [
                np.asarray(_concept_payload(vector_space, layer=layer, concept=concept)["vector"], dtype=np.float32)
                for concept in concepts
            ],
            axis=0,
        )
        cosine = _cosine_matrix(matrix)
        pca_components = max(1, min(int(spec.pca_components), matrix.shape[0], matrix.shape[1]))
        pca = PCA(n_components=pca_components)
        coords = pca.fit_transform(matrix)
        layer_payload: dict[str, Any] = {
            "concepts": concepts,
            "cosine_similarity": cosine.astype(np.float32).tolist(),
            "pca": {
                "coordinates": coords.astype(np.float32).tolist(),
                "components": pca.components_.astype(np.float32).tolist(),
                "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32).tolist(),
            },
        }
        if spec.cluster_count is not None:
            k = max(1, min(int(spec.cluster_count), len(concepts)))
            labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(matrix)
            layer_payload["clusters"] = {concept: int(label) for concept, label in zip(concepts, labels, strict=True)}
        layer_results[str(layer)] = layer_payload

    payload = {
        "kind": "emotion_geometry_result",
        "vector_space_kind": vector_space.get("vector_space_kind"),
        "layers": layer_results,
        "summary": {
            "layer_count": len(layer_results),
            "concept_count": len(_geometry_concepts(layer_results)) if layer_results else 0,
            "pca_components": int(spec.pca_components),
            "cluster_count": spec.cluster_count,
        },
    }
    return OperationExecutionResult(
        payload=payload,
        example_coverage={"materialized": True, "example_count": 0, "example_keys": []},
    )


def _download_or_resolve_path(spec: EmotionPrecomputedVectorSpaceSpec) -> Path:
    if spec.path is not None:
        return Path(spec.path).expanduser()
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=str(spec.repo_id),
            filename=str(spec.filename),
            repo_type="dataset",
            revision=spec.revision,
            token=_hf_token(spec.token_env_var),
        )
    )


def _load_vector_space_payload(
    *,
    path: Path,
    format: str,
    select_layer: int | None,
    normalize: str,
    vector_space_kind: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Emotion vector-space artifact does not exist: {path}")
    normalized = str(format).strip().lower()
    if normalized == "json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return _coerce_vector_space_payload(
            payload,
            select_layer=select_layer,
            normalize=normalize,
            vector_space_kind=vector_space_kind,
            metadata=metadata,
        )
    if normalized == "npz":
        arrays = np.load(path)
        layers: dict[str, dict[str, Any]] = {}
        for key in arrays.files:
            if "__" in key and key.startswith("layer_"):
                layer_part, concept = key.split("__", 1)
                layer = int(layer_part.removeprefix("layer_"))
            elif select_layer is not None:
                layer = int(select_layer)
                concept = key
            else:
                raise SpecValidationError(
                    "NPZ emotion vector spaces require keys like 'layer_12__happy' or select_layer=..."
                )
            layers.setdefault(str(layer), {"concepts": {}})
            raw = np.asarray(arrays[key], dtype=np.float32)
            vector, norm = normalize_vector(raw, normalize=normalize, error_label="emotion vector")
            layers[str(layer)]["concepts"][str(concept)] = {
                "vector": vector.tolist(),
                "raw_vector": raw.tolist(),
                "norm": float(norm),
            }
        return _vector_space_payload(
            layers=layers,
            vector_space_kind=vector_space_kind,
            metadata=metadata,
            normalized=normalize,
        )
    raise SpecValidationError(f"Unsupported emotion vector-space format: {format!r}")


def _coerce_vector_space_payload(
    payload: Any,
    *,
    select_layer: int | None,
    normalize: str,
    vector_space_kind: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("Emotion vector-space JSON payload must be a mapping")
    if str(payload.get("kind") or "") == EMOTION_VECTOR_SPACE_KIND:
        result = dict(payload)
        result["metadata"] = {**dict(result.get("metadata", {})), **dict(metadata)}
        if select_layer is not None:
            raw_layers = result.get("layers")
            if not isinstance(raw_layers, Mapping) or str(int(select_layer)) not in raw_layers:
                raise SpecValidationError(f"Emotion vector-space payload is missing selected layer {select_layer}")
            result["layers"] = {str(int(select_layer)): raw_layers[str(int(select_layer))]}
        return result

    # Compact JSON shape: {"happy": [...], "sad": [...]} for one selected layer,
    # or {"layers": {"12": {"happy": [...]}}}.
    if "layers" in payload:
        source_layers = payload["layers"]
    else:
        if select_layer is None:
            raise SpecValidationError("Compact emotion vector-space JSON requires select_layer=...")
        source_layers = {str(int(select_layer)): payload}
    if not isinstance(source_layers, Mapping):
        raise TypeError("Emotion vector-space layers must be a mapping")

    layers: dict[str, dict[str, Any]] = {}
    for layer_name, concept_map in source_layers.items():
        if not isinstance(concept_map, Mapping):
            raise TypeError("Emotion vector-space layer entries must be mappings")
        layer_payload = {"concepts": {}}
        for concept, raw_payload in concept_map.items():
            if isinstance(raw_payload, Mapping):
                raw_vector = raw_payload.get("raw_vector", raw_payload.get("vector"))
            else:
                raw_vector = raw_payload
            raw = np.asarray(raw_vector, dtype=np.float32)
            if raw.ndim != 1:
                raise SpecValidationError(f"Emotion vector {concept!r} at layer {layer_name!r} must be rank-1")
            vector, norm = normalize_vector(raw, normalize=normalize, error_label="emotion vector")
            layer_payload["concepts"][str(concept)] = {
                "vector": vector.tolist(),
                "raw_vector": raw.tolist(),
                "norm": float(norm),
            }
        layers[str(int(layer_name))] = layer_payload
    return _vector_space_payload(
        layers=layers,
        vector_space_kind=vector_space_kind,
        metadata=metadata,
        normalized=normalize,
    )


def _vector_space_payload(
    *,
    layers: Mapping[str, Any],
    vector_space_kind: str,
    metadata: Mapping[str, Any],
    normalized: str,
) -> dict[str, Any]:
    return {
        "kind": EMOTION_VECTOR_SPACE_KIND,
        "vector_space_kind": str(vector_space_kind),
        "layers": {str(layer): dict(payload) for layer, payload in layers.items()},
        "metadata": dict(metadata),
        "summary": {
            "layer_count": len(layers),
            "concept_count": len(_all_concepts(layers)),
            "normalized": str(normalized),
        },
    }


def _neutral_components_by_layer(spec: EmotionVectorSpaceSpec) -> dict[int, np.ndarray]:
    if spec.neutral_feature is None or spec.neutral_variance_threshold is None:
        return {}
    from sklearn.decomposition import PCA

    matrices, _ = feature_matrices(
        spec.neutral_feature,
        layers=tuple(spec.layers) if spec.layers else None,
        token_selector=spec.tokens,
        token_pooling=spec.pooling,
    )
    components: dict[int, np.ndarray] = {}
    for layer, matrix in matrices.items():
        if matrix.shape[0] < 2:
            components[int(layer)] = np.zeros((0, matrix.shape[1]), dtype=np.float32)
            continue
        n_components = max(1, min(matrix.shape[0], matrix.shape[1]))
        pca = PCA(n_components=n_components)
        pca.fit(matrix.astype(np.float32))
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        count = int(np.searchsorted(cumulative, float(spec.neutral_variance_threshold), side="left") + 1)
        count = max(1, min(count, n_components))
        components[int(layer)] = pca.components_[:count].astype(np.float32)
    return components


def _resolve_vector_space(source: Any) -> Mapping[str, Any]:
    payload = source.result() if hasattr(source, "result") else source
    if not isinstance(payload, Mapping):
        raise TypeError(f"Emotion vector-space source must resolve to a mapping, got {type(payload).__name__}")
    if str(payload.get("kind") or "") != EMOTION_VECTOR_SPACE_KIND:
        raise SpecValidationError(f"Expected {EMOTION_VECTOR_SPACE_KIND}, got {payload.get('kind')!r}")
    layers = payload.get("layers")
    if not isinstance(layers, Mapping) or not layers:
        raise SpecValidationError("Emotion vector-space payload must contain non-empty layers")
    return payload


def _coordinates_from_vector_space(
    vector_space: Mapping[str, Any],
    *,
    concepts: Sequence[str],
    layers: Sequence[int],
) -> list[dict[str, Any]]:
    selected_layers = _selected_layers(vector_space, layers=layers)
    if not selected_layers:
        raise SpecValidationError("EmotionScoreSpec did not resolve any vector-space layers")
    selected_concepts = sorted(set(concepts)) if concepts else _selected_concepts(vector_space, layer=selected_layers[0], concepts=())
    coordinates: list[dict[str, Any]] = []
    names_by_concept: dict[str, str] = {}
    for concept in selected_concepts:
        coordinate_name = f"emotion__{coordinate_name_key(concept)}"
        if coordinate_name in names_by_concept:
            raise SpecValidationError(
                "Emotion concepts must map to unique projection coordinate names; "
                f"{concept!r} and {names_by_concept[coordinate_name]!r} both map to {coordinate_name!r}"
            )
        names_by_concept[coordinate_name] = concept
        coordinate_layers: dict[str, Any] = {}
        for layer in selected_layers:
            payload = _concept_payload(vector_space, layer=layer, concept=concept)
            coordinate_layers[str(layer)] = {
                "vector": list(payload["vector"]),
                "raw_vector": list(payload.get("raw_vector", payload["vector"])),
                "norm": float(payload.get("norm", 0.0)),
            }
        coordinates.append(
            {
                "kind": "coordinate_result",
                "coordinate_kind": "direction",
                "name": coordinate_name,
                "layers": coordinate_layers,
                "metadata": {
                    "emotion": concept,
                    "vector_space_kind": vector_space.get("vector_space_kind"),
                },
            }
        )
    return coordinates


def _emotion_by_coordinate_name(coordinates: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    values: dict[str, str] = {}
    for coordinate in coordinates:
        name = str(coordinate.get("name") or "")
        metadata = coordinate.get("metadata")
        if name and isinstance(metadata, Mapping):
            values[name] = str(metadata.get("emotion") or name)
    return values


def _selected_layers(vector_space: Mapping[str, Any], *, layers: Sequence[int]) -> list[int]:
    raw_layers = vector_space.get("layers")
    if not isinstance(raw_layers, Mapping):
        return []
    available = sorted(int(layer) for layer in raw_layers)
    if not layers:
        return available
    requested = [int(layer) for layer in layers]
    missing = [layer for layer in requested if layer not in set(available)]
    if missing:
        raise SpecValidationError(f"Emotion vector-space is missing requested layers: {missing}")
    return requested


def _selected_concepts(vector_space: Mapping[str, Any], *, layer: int, concepts: Sequence[str]) -> list[str]:
    raw_layers = vector_space.get("layers")
    layer_payload = raw_layers.get(str(int(layer))) if isinstance(raw_layers, Mapping) else None
    if not isinstance(layer_payload, Mapping) or not isinstance(layer_payload.get("concepts"), Mapping):
        raise SpecValidationError(f"Emotion vector-space layer {layer} is missing concepts")
    available = sorted(str(concept) for concept in layer_payload["concepts"])
    if not concepts:
        return available
    requested = [str(concept) for concept in concepts]
    missing = [concept for concept in requested if concept not in set(available)]
    if missing:
        raise SpecValidationError(f"Emotion vector-space layer {layer} is missing requested concepts: {missing}")
    return requested


def _concept_payload(vector_space: Mapping[str, Any], *, layer: int, concept: str) -> Mapping[str, Any]:
    raw_layers = vector_space.get("layers")
    layer_payload = raw_layers.get(str(int(layer))) if isinstance(raw_layers, Mapping) else None
    if not isinstance(layer_payload, Mapping):
        raise SpecValidationError(f"Emotion vector-space is missing layer {layer}")
    concepts = layer_payload.get("concepts")
    if not isinstance(concepts, Mapping) or concept not in concepts:
        raise SpecValidationError(f"Emotion vector-space layer {layer} is missing concept {concept!r}")
    payload = concepts[concept]
    if not isinstance(payload, Mapping) or payload.get("vector") is None:
        raise SpecValidationError(f"Emotion vector-space concept {concept!r} at layer {layer} is missing a vector")
    return payload


def _all_concepts(layers: Mapping[str, Any]) -> list[str]:
    concepts: set[str] = set()
    for layer_payload in layers.values():
        if isinstance(layer_payload, Mapping) and isinstance(layer_payload.get("concepts"), Mapping):
            concepts.update(str(concept) for concept in layer_payload["concepts"])
    return sorted(concepts)


def _geometry_concepts(layers: Mapping[str, Any]) -> list[str]:
    concepts: set[str] = set()
    for layer_payload in layers.values():
        if isinstance(layer_payload, Mapping) and isinstance(layer_payload.get("concepts"), Sequence):
            concepts.update(str(concept) for concept in layer_payload["concepts"])
    return sorted(concepts)


def _vector_space_summary(vector_space: Mapping[str, Any]) -> dict[str, Any]:
    layers = vector_space.get("layers")
    layer_count = len(layers) if isinstance(layers, Mapping) else 0
    return {
        "kind": vector_space.get("kind"),
        "vector_space_kind": vector_space.get("vector_space_kind"),
        "layer_count": layer_count,
        "concept_count": len(_all_concepts(layers if isinstance(layers, Mapping) else {})),
        "metadata": dict(vector_space.get("metadata", {})) if isinstance(vector_space.get("metadata"), Mapping) else {},
    }


def _project_out_components(vector: np.ndarray, components: np.ndarray) -> np.ndarray:
    if components.size == 0:
        return vector.astype(np.float32)
    rows = np.asarray(components, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows[None, :]
    return (vector - (vector @ rows.T) @ rows).astype(np.float32)


def _cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    unit = values / safe
    return unit @ unit.T


def _feature_name(source: Any) -> str | None:
    name = getattr(source, "name", None)
    if name is not None:
        return str(name)
    artifact_id = getattr(source, "id", None)
    if artifact_id is not None:
        return str(artifact_id)
    return None


__all__ = [
    "run_emotion_direction",
    "run_emotion_geometry",
    "run_emotion_precomputed_vector_space",
    "run_emotion_score",
    "run_emotion_vector_space",
]
