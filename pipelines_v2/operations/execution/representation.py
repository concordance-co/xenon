"""Execution helpers for representation-analysis specs."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.data.datasets import LabelPredicate
from pipelines_v2.operations.execution.common import resolve_values_map
from pipelines_v2.operations.representation import BasisSpec, CentroidSpec, DirectionSpec, GeometrySpec, SubspaceSpec

from .common import (
    OperationExecutionResult,
    align_example_keys_to_rows,
    collapse_matrix_by_group,
    feature_matrices,
    filter_matrix_by_keys,
    feature_name,
    ordered_groups,
    ordered_values,
    subset_example_keys,
)


def run_direction(spec: DirectionSpec) -> OperationExecutionResult:
    if not isinstance(spec.positive, LabelPredicate) or not isinstance(spec.negative, LabelPredicate):
        raise SpecValidationError("DirectionSpec requires positive and negative LabelPredicate refs")

    matrices, example_keys = feature_matrices(
        spec.feature,
        layers=tuple(spec.layers) if spec.layers else None,
        token_selector=spec.tokens,
        token_pooling=spec.pooling,
    )
    positive_keys = {key for key in spec.positive.resolve_example_keys() if key in set(example_keys)}
    negative_keys = {key for key in spec.negative.resolve_example_keys() if key in set(example_keys)}
    if not positive_keys or not negative_keys:
        raise SpecValidationError("DirectionSpec requires at least one positive and one negative example")

    layers: dict[str, Any] = {}
    for layer, X in matrices.items():
        layer_example_keys = list(example_keys)
        if spec.group_by is not None:
            groups = ordered_groups(spec.group_by, example_keys)
            if groups is None:
                raise SpecValidationError("DirectionSpec group_by did not resolve to any groups")
            X, layer_example_keys = collapse_matrix_by_group(
                X,
                example_keys,
                groups,
                positive_keys=positive_keys,
                negative_keys=negative_keys,
            )
        if spec.group_by is not None:
            positive_indices = [index for index, key in enumerate(layer_example_keys) if key.startswith("positive::")]
            negative_indices = [index for index, key in enumerate(layer_example_keys) if key.startswith("negative::")]
        else:
            index_by_key = {key: index for index, key in enumerate(layer_example_keys)}
            positive_indices = [index_by_key[key] for key in sorted(positive_keys)]
            negative_indices = [index_by_key[key] for key in sorted(negative_keys)]
        if not positive_indices or not negative_indices:
            raise SpecValidationError("DirectionSpec produced empty positive or negative selections")
        pos = X[positive_indices]
        neg = X[negative_indices]
        vector = pos.mean(axis=0) - neg.mean(axis=0)
        norm = float(np.linalg.norm(vector))
        unit = vector / norm if norm > 0 else vector
        layer_payload: dict[str, Any] = {
            "vector": unit.tolist(),
            "raw_vector": vector.astype(np.float32).tolist(),
            "norm": norm,
            "positive_count": len(positive_keys),
            "negative_count": len(negative_keys),
        }
        if spec.subspace is not None:
            subspace_layer = _subspace_layer_payload(spec.subspace, int(layer))
            safe_scale = np.asarray(subspace_layer["safe_scale"], dtype=np.float32)
            components = np.asarray(subspace_layer["components"], dtype=np.float32)
            if components.ndim == 1:
                components = components[None, :]
            standardized_vector = np.asarray(vector, dtype=np.float32) / safe_scale
            weights = standardized_vector @ components.T if components.size else np.zeros((0,), dtype=np.float32)
            layer_payload["subspace_weights"] = weights.astype(np.float32).tolist()
            layer_payload["subspace_component_count"] = int(components.shape[0])
        layers[str(layer)] = layer_payload

    payload = {
        "kind": "direction_result",
        "feature": feature_name(spec.feature),
        "layers": layers,
        "summary": {
            "layer_count": len(layers),
            "positive_count": len(positive_keys),
            "negative_count": len(negative_keys),
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


def run_basis(spec: BasisSpec) -> OperationExecutionResult:
    if spec.method != "pca":
        raise NotImplementedError(f"BasisSpec method {spec.method!r} is not implemented yet")

    matrices, example_keys = feature_matrices(
        spec.feature,
        layers=tuple(spec.layers) if spec.layers else None,
        token_selector=spec.tokens,
        token_pooling=spec.pooling,
    )
    label_values = ordered_values(spec.by, example_keys, label="by") if spec.by is not None else None

    layers: dict[str, Any] = {}
    for layer, X in matrices.items():
        n_components = max(1, min(spec.components, X.shape[0], X.shape[1]))
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(X)
        layer_payload: dict[str, Any] = {
            "components": pca.components_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "mean": pca.mean_.tolist(),
            "example_count": X.shape[0],
            "component_count": n_components,
        }
        if label_values is not None:
            grouped: dict[str, Any] = {}
            for label in sorted({str(item) for item in label_values}):
                mask = np.asarray([str(item) == label for item in label_values], dtype=bool)
                grouped[label] = transformed[mask].mean(axis=0).tolist()
            layer_payload["group_centroids"] = grouped
        layers[str(layer)] = layer_payload

    payload = {
        "kind": "basis_result",
        "method": spec.method,
        "feature": feature_name(spec.feature),
        "layers": layers,
        "summary": {
            "layer_count": len(layers),
            "component_count": spec.components,
            "grouped": label_values is not None,
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


def run_subspace(spec: SubspaceSpec) -> OperationExecutionResult:
    matrices, example_keys = feature_matrices(
        spec.feature,
        layers=tuple(spec.layers) if spec.layers else None,
        token_selector=spec.tokens,
        token_pooling=spec.pooling,
    )

    layers: dict[str, Any] = {}
    for layer, X in matrices.items():
        mean = X.mean(axis=0).astype(np.float32)
        scale = X.std(axis=0).astype(np.float32)
        safe_scale = np.where(scale == 0, 1.0, scale).astype(np.float32)
        standardized = ((X - mean) / safe_scale).astype(np.float32)
        n_components = max(1, min(spec.components, standardized.shape[0], standardized.shape[1]))
        pca = PCA(n_components=n_components)
        pca.fit(standardized)
        named_components = {
            str(name): int(index)
            for name, index in dict(spec.named_components_by_layer.get(int(layer), {})).items()
            if 0 <= int(index) < int(n_components)
        }
        layers[str(layer)] = {
            "method": "standardized_pca",
            "mean": mean.tolist(),
            "scale": scale.tolist(),
            "safe_scale": safe_scale.tolist(),
            "components": pca.components_.astype(np.float32).tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32).tolist(),
            "example_count": int(X.shape[0]),
            "component_count": int(n_components),
            "named_components": named_components,
        }

    payload = {
        "kind": "subspace_result",
        "feature": feature_name(spec.feature),
        "layers": layers,
        "summary": {
            "layer_count": len(layers),
            "component_count": int(spec.components),
            "method": "standardized_pca",
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


def run_centroid(spec: CentroidSpec) -> OperationExecutionResult:
    matrices, feature_example_keys = feature_matrices(
        spec.feature,
        layers=tuple(spec.layers) if spec.layers else None,
        token_selector=spec.tokens,
        token_pooling=spec.pooling,
    )
    row_keys = align_example_keys_to_rows(feature_example_keys, spec.rows, label="CentroidSpec")
    value_map = {str(key): value for key, value in resolve_values_map(spec.by, label="CentroidSpec.by").items()}
    labels = sorted({str(value_map[key]) for key in row_keys})
    if not labels:
        raise SpecValidationError("CentroidSpec by did not resolve any centroid labels")

    layers: dict[str, Any] = {}
    for layer, X_all in matrices.items():
        X = filter_matrix_by_keys(X_all, feature_example_keys, row_keys)
        index_by_key = {key: index for index, key in enumerate(row_keys)}
        centroids: dict[str, Any] = {}
        counts: dict[str, int] = {}
        for label in labels:
            label_keys = [key for key in row_keys if str(value_map[key]) == label]
            label_indices = [index_by_key[key] for key in label_keys]
            centroid = X[np.asarray(label_indices, dtype=np.int64)].mean(axis=0).astype(np.float32)
            centroids[label] = centroid.tolist()
            counts[label] = len(label_keys)
        layer_payload: dict[str, Any] = {
            "centroids": centroids,
            "counts": counts,
            "example_count": int(X.shape[0]),
            "centroid_count": len(centroids),
        }
        if spec.subspace is not None:
            subspace_layer = _subspace_layer_payload(spec.subspace, int(layer))
            mean = np.asarray(subspace_layer["mean"], dtype=np.float32)
            safe_scale = np.asarray(subspace_layer["safe_scale"], dtype=np.float32)
            components = np.asarray(subspace_layer["components"], dtype=np.float32)
            if components.ndim == 1:
                components = components[None, :]
            standardized = {
                label: (((np.asarray(vector, dtype=np.float32) - mean) / safe_scale) @ components.T).astype(np.float32).tolist()
                for label, vector in centroids.items()
            }
            layer_payload["standardized_centroids"] = standardized
            layer_payload["subspace_component_count"] = int(components.shape[0])
        layers[str(layer)] = layer_payload

    payload = {
        "kind": "centroid_result",
        "feature": feature_name(spec.feature),
        "layers": layers,
        "summary": {
            "layer_count": len(layers),
            "label_count": len(labels),
            "example_count": len(row_keys),
        },
    }
    return OperationExecutionResult(
        payload=payload,
        example_coverage={
            "materialized": True,
            "example_count": len(row_keys),
            "example_keys": list(row_keys),
        },
    )


def run_geometry(spec: GeometrySpec) -> OperationExecutionResult:
    method = str(spec.method).lower()
    if method not in {"pca", "lda"}:
        raise NotImplementedError(f"GeometrySpec method {spec.method!r} is not implemented yet")

    matrices, feature_example_keys = feature_matrices(
        spec.feature,
        layers=tuple(spec.layers) if spec.layers else None,
        token_selector=spec.tokens,
        token_pooling=spec.pooling,
    )
    row_keys = align_example_keys_to_rows(feature_example_keys, spec.rows, label="GeometrySpec")
    selected_keys = subset_example_keys(row_keys, spec.subset)
    if not selected_keys:
        raise SpecValidationError("GeometrySpec subset did not leave any examples")
    label_values = ordered_values(spec.label, row_keys, label="label") if spec.label is not None else None
    color_values = {
        name: ordered_values(source, row_keys, label=f"color_by[{name}]")
        for name, source in spec.color_by.items()
    }

    layers: list[dict[str, Any]] = []
    for layer, X_all in matrices.items():
        X_row = filter_matrix_by_keys(X_all, feature_example_keys, row_keys)
        X = filter_matrix_by_keys(X_row, row_keys, selected_keys)
        X = _normalize_matrix(X, normalize=spec.normalize)
        layer_payload: dict[str, Any] = {
            "layer": int(layer),
            "example_count": int(X.shape[0]),
            "selected_example_keys": list(selected_keys),
        }
        if method == "pca":
            n_components = max(1, min(int(spec.components), X.shape[0], X.shape[1]))
            pca = PCA(n_components=n_components)
            projected = pca.fit_transform(X)
            layer_payload.update(
                {
                    "components": projected.tolist(),
                    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                    "component_count": int(n_components),
                }
            )
        else:
            if label_values is None:
                raise SpecValidationError("GeometrySpec method='lda' requires label")
            label_index = {key: index for index, key in enumerate(row_keys)}
            subset_labels = [label_values[label_index[key]] for key in selected_keys]
            encoded_labels, class_names = _encode_geometry_labels(subset_labels)
            n_classes = len(class_names)
            max_components = min(int(spec.components), max(1, n_classes - 1), X.shape[1])
            lda = LinearDiscriminantAnalysis(n_components=max_components)
            projected = lda.fit_transform(X, encoded_labels)
            layer_payload.update(
                {
                    "components": projected.tolist(),
                    "label_name": getattr(spec.label, "name", None),
                    "class_names": class_names,
                    "component_count": int(max_components),
                    "explained_variance_ratio": (
                        lda.explained_variance_ratio_.tolist()
                        if hasattr(lda, "explained_variance_ratio_")
                        else None
                    ),
                }
            )
        if label_values is not None:
            label_index = {key: index for index, key in enumerate(row_keys)}
            subset_labels = [label_values[label_index[key]] for key in selected_keys]
            layer_payload["labels"] = [str(value) for value in subset_labels]
        if color_values:
            label_index = {key: index for index, key in enumerate(row_keys)}
            layer_payload["color_by"] = {
                name: [value_list[label_index[key]] for key in selected_keys]
                for name, value_list in color_values.items()
            }
        layers.append(layer_payload)

    payload = {
        "kind": "geometry_result",
        "feature": feature_name(spec.feature),
        "method": method,
        "normalize": spec.normalize,
        "layers": layers,
        "summary": {
            "layer_count": len(layers),
            "example_count": len(selected_keys),
            "method": method,
            "subset_applied": spec.subset is not None,
        },
    }
    return OperationExecutionResult(
        payload=payload,
        example_coverage={
            "materialized": True,
            "example_count": len(selected_keys),
            "example_keys": list(selected_keys),
        },
    )


def _normalize_matrix(X: np.ndarray, *, normalize: str | None) -> np.ndarray:
    if normalize is None:
        return X.astype(np.float32)
    if normalize == "rms_per_row":
        rms = np.sqrt(np.mean(np.square(X), axis=1, keepdims=True))
        rms = np.maximum(rms, 1e-8)
        return (X / rms).astype(np.float32)
    raise NotImplementedError(f"GeometrySpec normalize mode {normalize!r} is not implemented yet")


def _encode_geometry_labels(values: list[Any]) -> tuple[np.ndarray, list[str]]:
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(np.asarray(values, dtype=object))
    return encoded.astype(np.int64), [str(item) for item in encoder.classes_]


def _subspace_layer_payload(value: Any, layer: int) -> dict[str, Any]:
    if value is None or not hasattr(value, "result"):
        raise SpecValidationError("Expected a subspace operation artifact ref")
    payload = value.result()
    if not isinstance(payload, dict):
        raise SpecValidationError("Subspace payload must be a mapping")
    if str(payload.get("kind") or "") != "subspace_result":
        raise SpecValidationError("Expected subspace_result payload")
    layers = payload.get("layers")
    if not isinstance(layers, dict):
        raise SpecValidationError("Subspace payload is missing layers")
    layer_payload = layers.get(str(int(layer)))
    if not isinstance(layer_payload, dict):
        raise SpecValidationError(f"Subspace payload is missing layer {int(layer)}")
    if "safe_scale" not in layer_payload:
        raise SpecValidationError(f"Subspace payload layer {int(layer)} is missing safe_scale")
    return layer_payload
