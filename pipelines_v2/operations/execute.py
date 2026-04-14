"""Execution helpers for artifact-bound operation specs."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from pipelines_v2.core.types import SpecValidationError, stable_hash
from pipelines_v2.data.datasets import CaseSet, LabelPredicate, LabelSet
from pipelines_v2.operations.specs import (
    BasisSpec,
    DirectionSpec,
    LabelFieldsSpec,
    LabelMapSpec,
    PairDeltaSpec,
    ProbeSpec,
    ReportSpec,
    TransformResult,
    TransformSpec,
    TokenSelector,
    TokenPooling,
)
from pipelines_v2.storage.artifacts import ArtifactLabelRef, CaptureArtifact, FeatureLayerRef, FeatureRef, OperationArtifact


@dataclass(frozen=True, slots=True)
class OperationExecutionResult:
    payload: dict[str, Any] = field(default_factory=dict)
    features: dict[str, dict[str, Any]] = field(default_factory=dict)
    labels: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    example_coverage: dict[str, Any] = field(default_factory=dict)


def execute_artifact_operation(spec: Any) -> OperationExecutionResult:
    if isinstance(spec, ProbeSpec):
        return _run_probe(spec)
    if isinstance(spec, DirectionSpec):
        return _run_direction(spec)
    if isinstance(spec, BasisSpec):
        return _run_basis(spec)
    if isinstance(spec, PairDeltaSpec):
        return _run_pair_delta(spec)
    if isinstance(spec, LabelMapSpec):
        return _run_label_map(spec)
    if isinstance(spec, LabelFieldsSpec):
        return _run_label_fields(spec)
    if isinstance(spec, TransformSpec):
        return _run_transform(spec)
    if isinstance(spec, ReportSpec):
        return _run_report(spec)
    raise NotImplementedError(f"Artifact-bound execution is not implemented for {type(spec).__name__}")


def _run_probe(spec: ProbeSpec) -> OperationExecutionResult:
    matrices, example_keys = _feature_matrices(
        spec.feature,
        token_selector=spec.tokens,
        token_pooling=spec.pooling,
    )
    labels = _ordered_values(spec.labels, example_keys, label="labels")
    groups = _ordered_groups(spec.group_by, example_keys)
    split = _ordered_groups(spec.split, example_keys) if spec.split is not None else None

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(np.asarray(labels, dtype=object))
    if len(encoder.classes_) < 2:
        raise SpecValidationError("ProbeSpec requires at least two label classes")

    requested_metrics = tuple(spec.metrics) if spec.metrics else ("accuracy", "balanced_accuracy", "selectivity")
    layer_results: list[dict[str, Any]] = []
    for layer, X in matrices.items():
        layer_results.append(
            _probe_layer(
                layer=layer,
                X=X,
                y=encoded,
                groups=groups,
                split=split,
                train_values=tuple(spec.train_values),
                test_values=tuple(spec.test_values),
                folds=spec.folds,
                baselines=tuple(spec.baselines),
                metrics=requested_metrics,
            )
        )

    best_metric = "balanced_accuracy" if "balanced_accuracy" in requested_metrics else requested_metrics[0]
    best = max(layer_results, key=lambda item: float(item.get(best_metric, 0.0)))
    payload = {
        "kind": "probe_result",
        "feature": _feature_name(spec.feature),
        "label_name": getattr(spec.labels, "name", None),
        "class_names": [str(item) for item in encoder.classes_],
        "layers": layer_results,
        "summary": {
            "best_layer": best["layer"],
            "best_metric": best_metric,
            "best_value": best.get(best_metric),
            "example_count": len(example_keys),
            "group_count": len(set(groups)) if groups is not None else None,
            "split_mode": best.get("split_mode"),
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


def _run_direction(spec: DirectionSpec) -> OperationExecutionResult:
    if not isinstance(spec.positive, LabelPredicate) or not isinstance(spec.negative, LabelPredicate):
        raise SpecValidationError("DirectionSpec requires positive and negative LabelPredicate refs")

    matrices, example_keys = _feature_matrices(
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
            groups = _ordered_groups(spec.group_by, example_keys)
            if groups is None:
                raise SpecValidationError("DirectionSpec group_by did not resolve to any groups")
            X, layer_example_keys = _collapse_matrix_by_group(
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
        layers[str(layer)] = {
            "vector": unit.tolist(),
            "norm": norm,
            "positive_count": len(positive_keys),
            "negative_count": len(negative_keys),
        }

    payload = {
        "kind": "direction_result",
        "feature": _feature_name(spec.feature),
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


def _run_basis(spec: BasisSpec) -> OperationExecutionResult:
    if spec.method != "pca":
        raise NotImplementedError(f"BasisSpec method {spec.method!r} is not implemented yet")

    matrices, example_keys = _feature_matrices(
        spec.feature,
        layers=tuple(spec.layers) if spec.layers else None,
        token_selector=spec.tokens,
        token_pooling=spec.pooling,
    )
    label_values = _ordered_values(spec.by, example_keys, label="by") if spec.by is not None else None

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
        "feature": _feature_name(spec.feature),
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


def _run_pair_delta(spec: PairDeltaSpec) -> OperationExecutionResult:
    if not isinstance(spec.positive, LabelPredicate) or not isinstance(spec.negative, LabelPredicate):
        raise SpecValidationError("PairDeltaSpec requires positive and negative LabelPredicate refs")

    matrices, example_keys = _feature_matrices(
        spec.feature,
        layers=tuple(spec.layers) if spec.layers else None,
        token_selector=spec.tokens,
        token_pooling=spec.pooling,
    )
    case_values = _ordered_values(spec.case, example_keys, label="case")
    positive_keys = {key for key in spec.positive.resolve_example_keys() if key in set(example_keys)}
    negative_keys = {key for key in spec.negative.resolve_example_keys() if key in set(example_keys)}
    if not positive_keys or not negative_keys:
        raise SpecValidationError("PairDeltaSpec requires at least one positive and one negative example")

    groups: dict[str, list[str]] = {}
    for example_key, case_value in zip(example_keys, case_values, strict=False):
        groups.setdefault(str(case_value), []).append(example_key)

    case_keys: list[str] = []
    side_keys: dict[str, list[str]] = {}
    for case_key, members in groups.items():
        pos_members = [key for key in members if key in positive_keys]
        neg_members = [key for key in members if key in negative_keys]
        if not pos_members or not neg_members:
            continue
        case_keys.append(case_key)
        side_keys[f"positive::{case_key}"] = pos_members
        side_keys[f"negative::{case_key}"] = neg_members

    if not case_keys:
        raise SpecValidationError("PairDeltaSpec could not find any case with both positive and negative members")

    feature_layers: dict[str, Any] = {}
    index_by_example = {key: index for index, key in enumerate(example_keys)}
    for layer, X in matrices.items():
        layer_payload: dict[str, Any] = {}
        for case_key in case_keys:
            pos_members = side_keys[f"positive::{case_key}"]
            neg_members = side_keys[f"negative::{case_key}"]
            pos_vec = X[[index_by_example[key] for key in pos_members]].mean(axis=0)
            neg_vec = X[[index_by_example[key] for key in neg_members]].mean(axis=0)
            delta = (pos_vec - neg_vec).astype(np.float32)
            layer_payload[case_key] = {
                "tokens": [0],
                "values": np.expand_dims(delta, axis=0),
                "prompt_hash": stable_hash([spec.output_feature_name, case_key])[:24],
            }
        feature_layers[str(layer)] = layer_payload

    propagated_example_keys = {
        case_key: side_keys[f"{spec.propagate_from}::{case_key}"]
        for case_key in case_keys
    }
    label_payloads = {
        name: _label_payload_from_grouped_source(
            name=name,
            source=source,
            grouped_example_keys=propagated_example_keys,
        )
        for name, source in dict(spec.labels).items()
    }

    feature_payload = {
        "kind": "residual",
        "site": "pair_delta",
        "storage": {"dtype": "float32", "format": "safetensors"},
        "layers": feature_layers,
    }
    payload = {
        "kind": "pair_delta_result",
        "feature": spec.output_feature_name,
        "pair_count": len(case_keys),
        "layer_count": len(feature_layers),
        "propagate_from": spec.propagate_from,
        "labels": sorted(label_payloads),
    }
    return OperationExecutionResult(
        payload=payload,
        features={spec.output_feature_name: feature_payload},
        labels=label_payloads,
        example_coverage={
            "materialized": True,
            "example_count": len(case_keys),
            "example_keys": case_keys,
        },
    )


def _run_transform(spec: TransformSpec) -> OperationExecutionResult:
    raw_result = spec.builder.build(spec.inputs)
    result = _coerce_transform_result(raw_result)
    labels = {
        str(name): _coerce_transform_label_payload(str(name), values)
        for name, values in result.labels.items()
    }
    example_keys = (
        [str(key) for key in result.example_keys]
        if result.example_keys is not None
        else _infer_transform_example_keys(labels)
    )
    payload = dict(result.payload)
    payload.setdefault("kind", "transform_result")
    return OperationExecutionResult(
        payload=payload,
        labels=labels,
        metadata=dict(result.metadata),
        example_coverage={
            "materialized": example_keys is not None,
            "example_count": len(example_keys) if example_keys is not None else None,
            **({"example_keys": list(example_keys)} if example_keys is not None else {}),
        },
    )


def _run_label_map(spec: LabelMapSpec) -> OperationExecutionResult:
    values = _resolve_values_map(spec.source, label="source")
    mapped: dict[str, Any] = {}
    missing: set[str] = set()
    for key, value in values.items():
        mapping_key = str(value)
        if mapping_key in spec.mapping:
            mapped[str(key)] = spec.mapping[mapping_key]
        elif spec.strict:
            missing.add(mapping_key)
        else:
            mapped[str(key)] = spec.default_value
    if missing:
        raise SpecValidationError(
            f"LabelMapSpec output {spec.output_name!r} is missing mappings for source values: {sorted(missing)}"
        )
    payload = {
        "kind": "label_map_result",
        "source_name": getattr(spec.source, "name", None),
        "output_name": spec.output_name,
        "mapped_count": len(mapped),
        "strict": spec.strict,
    }
    labels = {spec.output_name: _make_label_payload(spec.output_name, mapped)}
    return OperationExecutionResult(
        payload=payload,
        labels=labels,
        example_coverage={
            "materialized": True,
            "example_count": len(mapped),
            "example_keys": sorted(mapped),
        },
    )


def _run_label_fields(spec: LabelFieldsSpec) -> OperationExecutionResult:
    values = _resolve_values_map(spec.source, label="source")
    extracted: dict[str, dict[str, Any]] = {output_name: {} for output_name in spec.fields}
    missing_fields: set[str] = set()
    for key, value in values.items():
        raw = _coerce_mapping_value(value, label="LabelFieldsSpec source value")
        for output_name, field_name in spec.fields.items():
            if field_name in raw:
                extracted[output_name][str(key)] = raw[field_name]
            elif spec.strict:
                missing_fields.add(field_name)
    if missing_fields:
        raise SpecValidationError(
            f"LabelFieldsSpec is missing requested fields in source payloads: {sorted(missing_fields)}"
        )
    labels = {
        output_name: _make_label_payload(output_name, output_values)
        for output_name, output_values in extracted.items()
    }
    payload = {
        "kind": "label_fields_result",
        "source_name": getattr(spec.source, "name", None),
        "label_names": sorted(labels),
        "strict": spec.strict,
    }
    example_keys = sorted(next(iter(extracted.values()))) if extracted else []
    return OperationExecutionResult(
        payload=payload,
        labels=labels,
        example_coverage={
            "materialized": True,
            "example_count": len(example_keys),
            "example_keys": example_keys,
        },
    )


def _run_report(spec: ReportSpec) -> OperationExecutionResult:
    inputs = [_summarize_report_input(item) for item in spec.inputs]
    example_keys = _report_example_keys(spec.inputs)
    payload = {
        "kind": "report_result",
        "template": spec.template,
        "output_dir": spec.output_dir,
        "inputs": inputs,
        "summary": {
            "template": spec.template,
            "input_count": len(inputs),
            "example_count": len(example_keys) if example_keys is not None else None,
        },
    }
    example_coverage = {
        "materialized": example_keys is not None,
        "example_count": len(example_keys) if example_keys is not None else None,
    }
    if example_keys is not None:
        example_coverage["example_keys"] = example_keys
    return OperationExecutionResult(payload=payload, example_coverage=example_coverage)


def _probe_layer(
    *,
    layer: int,
    X: NDArray[np.float32],
    y: NDArray[np.int64],
    groups: NDArray[np.object_] | None,
    split: NDArray[np.object_] | None,
    train_values: Sequence[Any],
    test_values: Sequence[Any],
    folds: int,
    baselines: Sequence[str],
    metrics: Sequence[str],
) -> dict[str, Any]:
    splits, split_mode = _classification_splits(
        y=y,
        groups=groups,
        split=split,
        train_values=train_values,
        test_values=test_values,
        folds=folds,
    )
    if not splits:
        raise SpecValidationError("ProbeSpec could not produce any validation splits")

    accuracy_scores: list[float] = []
    balanced_scores: list[float] = []
    auroc_scores: list[float] = []
    baseline_scores: dict[str, list[float]] = {name: [] for name in baselines}
    compute_shuffled_control = "shuffled_label" in baseline_scores or "selectivity" in metrics
    shuffled_control_scores: list[float] = []

    for fold_index, (train_idx, test_idx) in enumerate(splits):
        model = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                class_weight="balanced",
                max_iter=2000,
                tol=1e-3,
                random_state=42 + fold_index,
            ),
        )
        model.fit(X[train_idx], y[train_idx])
        predictions = model.predict(X[test_idx])
        probabilities = model.predict_proba(X[test_idx]) if "auroc" in metrics else None
        accuracy_scores.append(float(accuracy_score(y[test_idx], predictions)))
        balanced_scores.append(float(balanced_accuracy_score(y[test_idx], predictions)))
        if probabilities is not None and probabilities.shape[1] == 2 and len(np.unique(y[test_idx])) > 1:
            from sklearn.metrics import roc_auc_score

            auroc_scores.append(float(roc_auc_score(y[test_idx], probabilities[:, 1])))

        if "majority" in baseline_scores:
            baseline = DummyClassifier(strategy="most_frequent")
            baseline.fit(X[train_idx], y[train_idx])
            baseline_scores["majority"].append(float(accuracy_score(y[test_idx], baseline.predict(X[test_idx]))))
        if compute_shuffled_control:
            rng = np.random.default_rng(seed=fold_index)
            shuffled = np.array(y[train_idx], copy=True)
            rng.shuffle(shuffled)
            control = make_pipeline(
                StandardScaler(),
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-4,
                    class_weight="balanced",
                    max_iter=2000,
                    tol=1e-3,
                    random_state=4042 + fold_index,
                ),
            )
            control.fit(X[train_idx], shuffled)
            shuffled_control_scores.append(float(accuracy_score(y[test_idx], control.predict(X[test_idx]))))

    if "shuffled_label" in baseline_scores:
        baseline_scores["shuffled_label"] = shuffled_control_scores

    baseline_majority = float(np.mean(baseline_scores["majority"])) if "majority" in baseline_scores else None
    baseline_shuffled = (
        float(np.mean(shuffled_control_scores))
        if shuffled_control_scores
        else (baseline_majority if baseline_majority is not None else 0.0)
    )
    accuracy = float(np.mean(accuracy_scores))
    balanced = float(np.mean(balanced_scores))

    result = {
        "layer": int(layer),
        "split_mode": split_mode,
        "example_count": int(X.shape[0]),
        "class_count": int(len(np.unique(y))),
    }
    if "accuracy" in metrics:
        result["accuracy"] = round(accuracy, 4)
    if "balanced_accuracy" in metrics:
        result["balanced_accuracy"] = round(balanced, 4)
    if "auroc" in metrics:
        result["auroc"] = round(float(np.mean(auroc_scores)), 4) if auroc_scores else None
    if "selectivity" in metrics:
        result["selectivity"] = round(accuracy - float(baseline_shuffled or 0.0), 4)
    if "majority" in baseline_scores:
        result["baseline_majority"] = round(float(baseline_majority), 4) if baseline_majority is not None else None
    if "shuffled_label" in baseline_scores:
        result["baseline_shuffled"] = round(float(baseline_shuffled), 4) if baseline_shuffled is not None else None
    return result


def _classification_splits(
    *,
    y: NDArray[np.int64],
    groups: NDArray[np.object_] | None,
    split: NDArray[np.object_] | None,
    train_values: Sequence[Any],
    test_values: Sequence[Any],
    folds: int,
) -> tuple[list[tuple[NDArray[np.int64], NDArray[np.int64]]], str]:
    if split is not None:
        train_allowed = {str(value) for value in train_values}
        test_allowed = {str(value) for value in test_values}
        split_labels = np.asarray([str(value) for value in split], dtype=object)
        train = np.asarray([index for index, value in enumerate(split_labels) if value in train_allowed], dtype=np.int64)
        test = np.asarray([index for index, value in enumerate(split_labels) if value in test_allowed], dtype=np.int64)
        if train.size == 0 or test.size == 0:
            return [], "fixed"
        if groups is not None:
            train_groups = set(str(value) for value in groups[train])
            test_groups = set(str(value) for value in groups[test])
            overlap = sorted(train_groups & test_groups)
            if overlap:
                raise SpecValidationError(
                    f"ProbeSpec fixed split leaks groups across train/test: {overlap[:5]}"
                )
        return [(train, test)], "fixed"

    if groups is not None:
        unique_groups = np.unique(groups)
        if len(unique_groups) < 2:
            return [], "group_kfold"
        splitter = GroupKFold(n_splits=max(2, min(folds, len(unique_groups))))
        return (
            [(train.astype(np.int64), test.astype(np.int64)) for train, test in splitter.split(np.zeros_like(y), y, groups)],
            "group_kfold",
        )

    class_counts = np.bincount(y)
    min_class = int(class_counts.min())
    if min_class < 2:
        return [], "stratified_kfold"
    splitter = StratifiedKFold(n_splits=max(2, min(folds, len(y), min_class)), shuffle=True, random_state=42)
    return (
        [(train.astype(np.int64), test.astype(np.int64)) for train, test in splitter.split(np.zeros_like(y), y)],
        "stratified_kfold",
    )


def _feature_matrices(
    feature: Any,
    *,
    layers: Sequence[int] | None = None,
    token_selector: TokenSelector | None = None,
    token_pooling: TokenPooling | None = None,
) -> tuple[dict[int, NDArray[np.float32]], list[str]]:
    if isinstance(feature, FeatureLayerRef):
        layer_payload = feature.load()
        example_keys = sorted(layer_payload)
        return {
            int(feature.layer): _matrix_from_layer_payload(
                layer_payload,
                example_keys,
                token_selector=token_selector,
                token_pooling=token_pooling,
            )
        }, example_keys
    if isinstance(feature, FeatureRef):
        payload = feature.load()
        if payload.get("kind") != "residual":
            raise NotImplementedError("Only residual feature refs are supported for artifact-bound ops right now")
        available_layers = sorted(int(layer) for layer in payload["layers"])
        selected_layers = [layer for layer in available_layers if layers is None or layer in set(layers)]
        if not selected_layers:
            raise SpecValidationError("No requested layers were present in the feature payload")
        example_keys = sorted(payload["layers"][str(selected_layers[0])])
        return {
            int(layer): _matrix_from_layer_payload(
                payload["layers"][str(layer)],
                example_keys,
                token_selector=token_selector,
                token_pooling=token_pooling,
            )
            for layer in selected_layers
        }, example_keys
    raise TypeError(f"Unsupported feature reference type: {type(feature).__name__}")


def _matrix_from_layer_payload(
    layer_payload: Mapping[str, Any],
    example_keys: Sequence[str],
    *,
    token_selector: TokenSelector | None,
    token_pooling: TokenPooling | None,
) -> NDArray[np.float32]:
    rows: list[NDArray[np.float32]] = []
    for key in example_keys:
        record = dict(layer_payload[key])
        values = np.asarray(record["values"], dtype=np.float32)
        if values.ndim != 2:
            raise TypeError("Residual layer payload values must be rank-2")
        if token_selector is not None:
            selected = token_selector.resolve(
                values.shape[0],
                token_sections=_coerce_token_sections(record.get("token_sections")),
            )
            if not selected:
                raise SpecValidationError("Token selector did not match any captured positions")
            values = values[selected]
        rows.append(_pool_token_values(values, token_pooling=token_pooling))
    return np.stack(rows, axis=0).astype(np.float32)


def _pool_token_values(
    values: NDArray[np.float32],
    *,
    token_pooling: TokenPooling | None,
) -> NDArray[np.float32]:
    pooling = token_pooling or TokenPooling.mean()
    indices = pooling.from_count(int(values.shape[0]))
    if not indices:
        raise SpecValidationError("Token pooling did not match any token positions")
    selected = values[np.asarray(indices, dtype=np.int64)]
    if pooling.kind == "mean":
        return selected.mean(axis=0).astype(np.float32)
    if pooling.kind == "last":
        return selected[-1].astype(np.float32)
    if pooling.kind == "first":
        return selected[0].astype(np.float32)
    raise SpecValidationError(f"Unsupported token pooling mode: {pooling.kind}")


def _ordered_values(source: Any, example_keys: Sequence[str], *, label: str) -> list[Any]:
    values = _resolve_values_map(source, label=label)
    return [values[key] for key in example_keys]


def _ordered_groups(groups: Any, example_keys: Sequence[str]) -> NDArray[np.object_] | None:
    if groups is None:
        return None
    values = _resolve_values_map(groups, label="groups")
    return np.asarray([values[key] for key in example_keys], dtype=object)


def _feature_name(feature: Any) -> str:
    if isinstance(feature, FeatureLayerRef):
        return f"{feature.feature.name}:layer:{feature.layer}"
    if isinstance(feature, FeatureRef):
        return feature.name
    return type(feature).__name__


def _summarize_report_input(value: Any) -> dict[str, Any]:
    if isinstance(value, (CaptureArtifact, OperationArtifact)):
        manifest = value.manifest()
        summary: Any | None = None
        if isinstance(value, OperationArtifact):
            try:
                summary = value.summary()
            except Exception:
                summary = None
        return {
            "artifact_id": manifest.artifact_id,
            "artifact_kind": manifest.artifact_kind,
            "created_at": manifest.created_at,
            "summary": summary,
        }
    if isinstance(value, Mapping):
        return dict(value)
    return {"value": str(value)}


def _resolve_values_map(source: Any, *, label: str) -> Mapping[str, Any]:
    if isinstance(source, (LabelSet, CaseSet, ArtifactLabelRef)):
        return source.resolve_values()
    raise SpecValidationError(f"Expected a label/case ref for {label}, got {type(source).__name__}")


def _resolve_optional_values(source: Any) -> Mapping[str, Any] | None:
    if source is None:
        return None
    return _resolve_values_map(source, label="optional_labels")


def _coerce_token_sections(raw: Any) -> dict[str, list[int]] | None:
    if not isinstance(raw, Mapping):
        return None
    return {
        str(name): [int(position) for position in positions]
        for name, positions in raw.items()
        if isinstance(positions, Sequence)
    }


def _make_label_payload(name: str, values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "label",
        "name": name,
        "values": {str(key): value for key, value in values.items()},
    }


def _label_payload_from_grouped_source(
    *,
    name: str,
    source: Any,
    grouped_example_keys: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    values = _resolve_values_map(source, label=name)
    grouped_values: dict[str, Any] = {}
    for output_key, example_keys in grouped_example_keys.items():
        selected = [values[key] for key in example_keys]
        if not selected:
            raise SpecValidationError(f"PairDeltaSpec label {name!r} had no source values for {output_key!r}")
        if not _all_equal(selected):
            raise SpecValidationError(
                f"PairDeltaSpec label {name!r} is not constant across the selected source examples for {output_key!r}"
            )
        grouped_values[output_key] = selected[0]
    return _make_label_payload(name, grouped_values)


def _all_equal(values: Sequence[Any]) -> bool:
    if not values:
        return True
    baseline = stable_hash(values[0])
    return all(stable_hash(value) == baseline for value in values[1:])


def _coerce_mapping_value(value: Any, *, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            parsed = json.loads(stripped)
            if isinstance(parsed, Mapping):
                return parsed
    raise SpecValidationError(f"{label} must be a mapping or JSON object string")


def _coerce_transform_result(value: Any) -> TransformResult:
    if isinstance(value, TransformResult):
        return value
    if not isinstance(value, Mapping):
        raise SpecValidationError(
            "Transform builder must return TransformResult or a mapping with keys like "
            "'payload', 'labels', 'metadata', and 'example_keys'"
        )
    return TransformResult(
        payload=_coerce_mapping_field(value.get("payload", {}), label="TransformResult payload"),
        labels=_coerce_nested_mapping_field(value.get("labels", {}), label="TransformResult labels"),
        metadata=_coerce_mapping_field(value.get("metadata", {}), label="TransformResult metadata"),
        example_keys=tuple(value.get("example_keys")) if value.get("example_keys") is not None else None,
    )


def _coerce_mapping_field(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecValidationError(f"{label} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _coerce_nested_mapping_field(value: Any, *, label: str) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        raise SpecValidationError(f"{label} must be a mapping")
    normalized: dict[str, Mapping[str, Any]] = {}
    for key, item in value.items():
        if not isinstance(item, Mapping):
            raise SpecValidationError(f"{label}[{key!r}] must be a mapping")
        normalized[str(key)] = {str(inner_key): inner_value for inner_key, inner_value in item.items()}
    return normalized


def _coerce_transform_label_payload(name: str, values: Mapping[str, Any]) -> dict[str, Any]:
    if "kind" in values and values.get("kind") == "label" and "values" in values:
        raw_values = values.get("values")
        if not isinstance(raw_values, Mapping):
            raise SpecValidationError(f"Transform label payload {name!r} must contain mapping 'values'")
        return _make_label_payload(name, raw_values)
    return _make_label_payload(name, values)


def _infer_transform_example_keys(labels: Mapping[str, Mapping[str, Any]]) -> list[str] | None:
    if not labels:
        return None
    key_sets = [
        {str(key) for key in payload.get("values", {})}
        for payload in labels.values()
        if isinstance(payload, Mapping)
    ]
    if not key_sets:
        return None
    intersection = set.intersection(*(set(keys) for keys in key_sets))
    return sorted(intersection)


def _report_example_keys(inputs: Sequence[Any]) -> list[str] | None:
    keys: set[str] = set()
    found_any = False
    for item in inputs:
        if not isinstance(item, (CaptureArtifact, OperationArtifact)):
            continue
        coverage = item.manifest().example_coverage
        raw_keys = coverage.get("example_keys")
        if not isinstance(raw_keys, Sequence) or isinstance(raw_keys, str):
            continue
        keys.update(str(key) for key in raw_keys)
        found_any = True
    if not found_any:
        return None
    return sorted(keys)


def _collapse_matrix_by_group(
    X: NDArray[np.float32],
    example_keys: Sequence[str],
    groups: NDArray[np.object_],
    *,
    positive_keys: set[str],
    negative_keys: set[str],
) -> tuple[NDArray[np.float32], list[str]]:
    grouped: dict[str, list[int]] = {}
    for index, group in enumerate(groups.tolist()):
        grouped.setdefault(str(group), []).append(index)

    rows: list[NDArray[np.float32]] = []
    row_keys: list[str] = []
    for group_key, indices in grouped.items():
        member_keys = [example_keys[index] for index in indices]
        is_positive = any(key in positive_keys for key in member_keys)
        is_negative = any(key in negative_keys for key in member_keys)
        if is_positive and is_negative:
            raise SpecValidationError(f"DirectionSpec group {group_key!r} mixes positive and negative members")
        if not is_positive and not is_negative:
            continue
        rows.append(X[np.asarray(indices, dtype=np.int64)].mean(axis=0))
        row_keys.append(f"{'positive' if is_positive else 'negative'}::{group_key}")

    if not rows:
        return np.empty((0, X.shape[1]), dtype=np.float32), []
    return np.stack(rows, axis=0).astype(np.float32), row_keys
