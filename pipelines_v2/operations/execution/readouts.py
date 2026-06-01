"""Execution helpers for readout and baseline specs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.operations.readouts import (
    PersistedProbeImportSpec,
    PersistedProbeInferenceSpec,
    ProbeSpec,
    ResidualizedProbeSpec,
    TextBaselineSpec,
    TransferProbeSpec,
)

from .common import (
    OperationExecutionResult,
    align_example_keys_to_rows,
    encode_labels,
    feature_matrices,
    feature_name,
    filter_matrix_by_keys,
    make_label_payload,
    ordered_groups,
    ordered_values,
    reference_example_keys,
)


def run_probe(spec: ProbeSpec) -> OperationExecutionResult:
    matrices, feature_example_keys = feature_matrices(
        spec.feature,
        token_selector=spec.tokens,
        token_pooling=spec.pooling,
    )
    example_keys = align_example_keys_to_rows(feature_example_keys, spec.rows, label="ProbeSpec")
    matrices = {
        layer: filter_matrix_by_keys(X, feature_example_keys, example_keys)
        for layer, X in matrices.items()
    }
    labels = ordered_values(spec.labels, example_keys, label="labels")
    groups = ordered_groups(spec.group_by, example_keys)
    split = ordered_groups(spec.split, example_keys) if spec.split is not None else None

    encoded, classes = encode_labels(labels)
    if len(classes) < 2:
        raise SpecValidationError("ProbeSpec requires at least two label classes")

    requested_metrics = tuple(spec.metrics) if spec.metrics else ("accuracy", "balanced_accuracy", "selectivity")
    layer_results: list[dict[str, Any]] = []
    for layer, X in matrices.items():
        layer_results.append(
            probe_layer(
                layer=layer,
                X=X,
                y=encoded,
                groups=groups,
                split=split,
                train_values=tuple(spec.train_values),
                train_stages=tuple(tuple(stage) for stage in spec.train_stages),
                stage_epochs=tuple(spec.stage_epochs),
                test_values=tuple(spec.test_values),
                folds=spec.folds,
                baselines=tuple(spec.baselines),
                metrics=requested_metrics,
                example_keys=example_keys,
                class_names=classes,
                persist_predictions=spec.persist_predictions,
                persist_model=spec.persist_model,
            )
        )

    best_metric = "balanced_accuracy" if "balanced_accuracy" in requested_metrics else requested_metrics[0]
    best = max(layer_results, key=lambda item: float(item.get(best_metric, 0.0)))
    persisted_layers = [
        layer_result["persisted_probe"]
        for layer_result in layer_results
        if isinstance(layer_result.get("persisted_probe"), Mapping)
    ]
    payload = {
        "kind": "probe_result",
        "feature": feature_name(spec.feature),
        "label_name": getattr(spec.labels, "name", None),
        "class_names": classes,
        "layers": layer_results,
        "summary": {
            "best_layer": best["layer"],
            "best_metric": best_metric,
            "best_value": best.get(best_metric),
            "example_count": len(example_keys),
            "group_count": len(set(groups.tolist())) if groups is not None else None,
            "split_mode": best.get("split_mode"),
        },
    }
    if persisted_layers:
        payload["persisted_probe"] = {
            "kind": "persisted_probe",
            "name": getattr(spec.labels, "name", None) or "probe",
            "feature_name": feature_name(spec.feature),
            "class_names": classes,
            "layers": persisted_layers,
            "metadata": {
                "source_kind": "ProbeSpec",
                "label_name": getattr(spec.labels, "name", None),
                "train_values": [str(value) for value in spec.train_values],
                "test_values": [str(value) for value in spec.test_values],
                "train_stages": [[str(value) for value in stage] for stage in spec.train_stages],
                "stage_epochs": [int(value) for value in spec.stage_epochs],
                "tokens": {"kind": spec.tokens.kind, "value": spec.tokens.value},
                "pooling": {"kind": spec.pooling.kind},
            },
            "summary": {
                "layer_count": len(persisted_layers),
                "layers": [int(layer["layer"]) for layer in persisted_layers],
                "class_count": len(classes),
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


def run_persisted_probe_import(spec: PersistedProbeImportSpec) -> OperationExecutionResult:
    payload = _load_persisted_probe_payload(spec)
    layers = [_normalize_probe_layer(layer, source_path=spec.path) for layer in payload.get("layers", ())]
    if not layers:
        raise SpecValidationError("PersistedProbeImportSpec requires at least one layer with coefficients")

    class_names = [str(value) for value in payload.get("class_names", ())]
    if len(class_names) < 2:
        raise SpecValidationError("PersistedProbeImportSpec requires at least two class_names")

    imported = {
        "kind": "persisted_probe",
        "name": spec.name or str(payload.get("name") or Path(spec.path).stem),
        "model": spec.model or payload.get("model"),
        "feature_name": spec.feature_name or payload.get("feature_name"),
        "class_names": class_names,
        "layers": layers,
        "metadata": {
            **{str(key): value for key, value in dict(payload.get("metadata", {})).items()},
            **{str(key): value for key, value in dict(spec.metadata).items()},
            "source_path": spec.path,
            "source_format": spec.format,
        },
        "summary": {
            "layer_count": len(layers),
            "layers": [int(layer["layer"]) for layer in layers],
            "class_count": len(class_names),
        },
    }
    return OperationExecutionResult(payload=imported)


def run_persisted_probe_inference(spec: PersistedProbeInferenceSpec) -> OperationExecutionResult:
    probe = _resolve_persisted_probe(spec.probe)
    requested_layers = tuple(int(layer) for layer in spec.layers)
    probe_layers = {
        int(layer["layer"]): layer
        for layer in probe.get("layers", ())
        if isinstance(layer, Mapping) and layer.get("layer") is not None
    }
    selected_layers = requested_layers or tuple(sorted(probe_layers))
    missing_layers = [layer for layer in selected_layers if layer not in probe_layers]
    if missing_layers:
        raise SpecValidationError(f"PersistedProbeInferenceSpec probe does not contain requested layers: {missing_layers}")

    matrices, feature_example_keys = feature_matrices(
        spec.feature,
        layers=selected_layers,
        token_selector=spec.tokens,
        token_pooling=spec.pooling,
    )
    example_keys = align_example_keys_to_rows(feature_example_keys, spec.rows, label="PersistedProbeInferenceSpec")
    matrices = {
        layer: filter_matrix_by_keys(X, feature_example_keys, example_keys)
        for layer, X in matrices.items()
        if layer in selected_layers
    }

    class_names = [str(value) for value in probe.get("class_names", ())]
    if len(class_names) < 2:
        raise SpecValidationError("Persisted probe payload requires at least two class_names")
    positive_class = class_names[-1]
    score_base = _safe_label_component(spec.score_name or str(probe.get("name") or "persisted_probe"))

    rows: list[dict[str, Any]] = []
    labels: dict[str, dict[str, Any]] = {}
    for layer in selected_layers:
        X = matrices.get(layer)
        if X is None:
            continue
        layer_payload = probe_layers[layer]
        scores, probabilities, predictions = _apply_persisted_probe_layer(X, layer_payload, class_names=class_names)
        positive_index = len(class_names) - 1
        positive_probabilities = probabilities[:, positive_index]
        score_values: dict[str, Any] = {}
        prediction_values: dict[str, Any] = {}
        for row_index, example_key in enumerate(example_keys):
            probability_by_class = {
                class_name: round(float(probabilities[row_index, class_index]), 6)
                for class_index, class_name in enumerate(class_names)
            }
            row = {
                "example_key": str(example_key),
                "layer": int(layer),
                "probe": probe.get("name"),
                "score": round(float(scores[row_index]), 6),
                "probability": round(float(positive_probabilities[row_index]), 6),
                "positive_class": positive_class,
                "prediction": str(predictions[row_index]),
                "probability_by_class": probability_by_class,
            }
            rows.append(row)
            score_values[str(example_key)] = row["probability"]
            prediction_values[str(example_key)] = row["prediction"]
        if spec.emit_labels:
            labels[f"{score_base}__layer_{layer}__probability_{_safe_label_component(positive_class)}"] = make_label_payload(
                f"{score_base}__layer_{layer}__probability_{_safe_label_component(positive_class)}",
                score_values,
            )
            labels[f"{score_base}__layer_{layer}__prediction"] = make_label_payload(
                f"{score_base}__layer_{layer}__prediction",
                prediction_values,
            )

    payload = {
        "kind": "persisted_probe_inference_result",
        "feature": feature_name(spec.feature),
        "probe": {
            "name": probe.get("name"),
            "model": probe.get("model"),
            "feature_name": probe.get("feature_name"),
            "class_names": class_names,
        },
        "layers": list(selected_layers),
        "rows": rows,
        "summary": {
            "row_count": len(rows),
            "example_count": len(example_keys),
            "layer_count": len(selected_layers),
            "positive_class": positive_class,
        },
    }
    return OperationExecutionResult(
        payload=payload,
        labels=labels,
        example_coverage={
            "materialized": True,
            "example_count": len(example_keys),
            "example_keys": list(example_keys),
        },
    )


def run_transfer_probe(spec: TransferProbeSpec) -> OperationExecutionResult:
    matrices, feature_example_keys = feature_matrices(
        spec.feature,
        token_selector=spec.tokens,
        token_pooling=spec.pooling,
    )
    example_keys = align_example_keys_to_rows(feature_example_keys, spec.rows, label="TransferProbeSpec")
    matrices = {
        layer: filter_matrix_by_keys(X, feature_example_keys, example_keys)
        for layer, X in matrices.items()
    }
    label_values = ordered_values(spec.labels, example_keys, label="labels")
    y, class_names = encode_labels(label_values)
    if len(class_names) < 2:
        raise SpecValidationError("TransferProbeSpec requires at least two label classes")
    groups = ordered_groups(spec.group_by, example_keys)
    cohort_values = ordered_values(spec.cohort_by, example_keys, label="cohort_by") if spec.cohort_by is not None else []
    split_values = {
        name: ordered_values(source, example_keys, label=f"split_by[{name}]")
        for name, source in spec.split_by.items()
    }

    selected_cohorts = [str(value) for value in spec.cohort_values] if spec.cohort_values else sorted({str(value) for value in cohort_values})
    regularization = tuple(spec.regularization) or (1.0,)

    layers: list[dict[str, Any]] = []
    if split_values:
        if not selected_cohorts and cohort_values:
            selected_cohorts = sorted({str(value) for value in cohort_values})
        for layer, X in matrices.items():
            layers.append(
                _transfer_split_layer(
                    layer=layer,
                    X=X,
                    y=y,
                    class_names=class_names,
                    example_keys=example_keys,
                    groups=groups,
                    cohort_values=cohort_values,
                    selected_cohorts=selected_cohorts,
                    split_values=split_values,
                    train_values=tuple(spec.train_values) or ("train",),
                    test_values=tuple(spec.test_values) or ("test",),
                    regularization=regularization,
                    metrics=tuple(spec.metrics),
                    persist_predictions=spec.persist_predictions,
                )
            )
    else:
        if spec.cohort_by is None:
            raise SpecValidationError("TransferProbeSpec requires cohort_by unless split_by is provided")
        if not selected_cohorts:
            raise SpecValidationError("TransferProbeSpec could not infer any cohort values")
        for layer, X in matrices.items():
            layers.append(
                _transfer_cross_layer(
                    layer=layer,
                    X=X,
                    y=y,
                    class_names=class_names,
                    example_keys=example_keys,
                    groups=groups,
                    cohort_values=cohort_values,
                    selected_cohorts=selected_cohorts,
                    regularization=regularization,
                    metrics=tuple(spec.metrics),
                    compare_within_baseline=spec.compare_within_baseline,
                    compare_direction_similarity=spec.compare_direction_similarity,
                    persist_predictions=spec.persist_predictions,
                )
            )

    payload = {
        "kind": "transfer_probe_result",
        "feature": feature_name(spec.feature),
        "label_name": getattr(spec.labels, "name", None),
        "class_names": class_names,
        "layers": layers,
        "summary": {
            "layer_count": len(layers),
            "cohort_count": len(selected_cohorts),
            "split_names": sorted(split_values),
            "regularization": list(regularization),
            "mode": "split_holdout" if split_values else "cross_cohort_transfer",
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


def run_text_baseline(spec: TextBaselineSpec) -> OperationExecutionResult:
    text_example_keys = reference_example_keys(spec.text, label="text")
    example_keys = align_example_keys_to_rows(text_example_keys, spec.rows, label="TextBaselineSpec")
    if not example_keys:
        raise SpecValidationError("TextBaselineSpec requires a text ref with materializable example keys")
    text_values = [str(value) for value in ordered_values(spec.text, example_keys, label="text")]
    label_values = ordered_values(spec.labels, example_keys, label="labels")
    y, class_names = encode_labels(label_values)
    if len(class_names) < 2:
        raise SpecValidationError("TextBaselineSpec requires at least two label classes")
    groups = ordered_groups(spec.group_by, example_keys)
    cohort_values = ordered_values(spec.cohort_by, example_keys, label="cohort_by") if spec.cohort_by is not None else []
    split_values = {
        name: ordered_values(source, example_keys, label=f"split_by[{name}]")
        for name, source in spec.split_by.items()
    }
    selected_cohorts = [str(value) for value in spec.cohort_values] if spec.cohort_values else sorted({str(value) for value in cohort_values})
    regularization = tuple(spec.regularization) or (1.0,)

    if split_values:
        results = _text_split_results(
            texts=text_values,
            y=y,
            class_names=class_names,
            example_keys=example_keys,
            groups=groups,
            cohort_values=cohort_values,
            selected_cohorts=selected_cohorts,
            split_values=split_values,
            train_values=tuple(spec.train_values) or ("train",),
            test_values=tuple(spec.test_values) or ("test",),
            model=spec.model,
            regularization=regularization,
            metrics=tuple(spec.metrics),
            persist_predictions=spec.persist_predictions,
        )
        mode = "split_holdout"
    elif spec.cohort_by is not None:
        results = _text_cross_results(
            texts=text_values,
            y=y,
            class_names=class_names,
            example_keys=example_keys,
            groups=groups,
            cohort_values=cohort_values,
            selected_cohorts=selected_cohorts,
            model=spec.model,
            regularization=regularization,
            metrics=tuple(spec.metrics),
            persist_predictions=spec.persist_predictions,
        )
        mode = "cross_cohort_transfer"
    else:
        results = _text_grouped_cv_results(
            texts=text_values,
            y=y,
            class_names=class_names,
            groups=groups,
            example_keys=example_keys,
            model=spec.model,
            regularization=regularization,
            metrics=tuple(spec.metrics),
            persist_predictions=spec.persist_predictions,
        )
        mode = "grouped_cv"

    payload = {
        "kind": "text_baseline_result",
        "text_name": getattr(spec.text, "name", None),
        "label_name": getattr(spec.labels, "name", None),
        "class_names": class_names,
        "model": spec.model,
        "mode": mode,
        "results": results,
        "summary": {
            "mode": mode,
            "example_count": len(example_keys),
            "class_count": len(class_names),
            "cohort_count": len(selected_cohorts),
            "split_names": sorted(split_values),
            "regularization": list(regularization),
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


def run_residualized_probe(spec: ResidualizedProbeSpec) -> OperationExecutionResult:
    matrices, feature_example_keys = feature_matrices(
        spec.feature,
        token_selector=spec.tokens,
        token_pooling=spec.pooling,
    )
    example_keys = align_example_keys_to_rows(feature_example_keys, spec.rows, label="ResidualizedProbeSpec")
    matrices = {
        layer: filter_matrix_by_keys(X, feature_example_keys, example_keys)
        for layer, X in matrices.items()
    }
    label_values = ordered_values(spec.labels, example_keys, label="labels")
    y_target, target_classes = encode_labels(label_values)
    nuisance_values = ordered_values(spec.residualize_against, example_keys, label="residualize_against")
    y_nuisance, nuisance_classes = encode_labels(nuisance_values)
    groups = ordered_groups(spec.group_by, example_keys)

    layers: list[dict[str, Any]] = []
    for layer, X in matrices.items():
        family_model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=4000,
            random_state=42,
        )
        family_model.fit(X, y_nuisance)
        W = np.asarray(family_model.coef_, dtype=np.float64)
        family_acc_raw = float(balanced_accuracy_score(y_nuisance, family_model.predict(X)))
        projector = W.T @ np.linalg.pinv(W @ W.T) @ W
        X_null = (X.astype(np.float64) - X.astype(np.float64) @ projector).astype(np.float32)

        family_model_null = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=4000,
            random_state=42,
        )
        family_model_null.fit(X_null, y_nuisance)
        family_acc_null = float(balanced_accuracy_score(y_nuisance, family_model_null.predict(X_null)))

        raw_metrics = grouped_cv_activation(
            X=X,
            y=y_target,
            groups=groups,
            metrics=tuple(spec.metrics),
            C=1.0,
        )
        null_metrics = grouped_cv_activation(
            X=X_null,
            y=y_target,
            groups=groups,
            metrics=tuple(spec.metrics),
            C=1.0,
        )

        layers.append(
            {
                "layer": int(layer),
                "nuisance_class_names": nuisance_classes,
                "target_class_names": target_classes,
                "family_subspace_rank": int(np.linalg.matrix_rank(W)),
                "nuisance_accuracy_raw_training_fit": round(family_acc_raw, 4),
                "nuisance_accuracy_on_null_training_fit": round(family_acc_null, 4),
                "raw_probe": raw_metrics,
                "residualized_probe": null_metrics,
                "delta_raw_minus_null": {
                    metric: (
                        round(float(raw_metrics[metric]) - float(null_metrics[metric]), 4)
                        if raw_metrics.get(metric) is not None and null_metrics.get(metric) is not None
                        else None
                    )
                    for metric in spec.metrics
                },
            }
        )

    def _best_metric(section: str, metric: str) -> float | None:
        values = [
            float(layer[section][metric])
            for layer in layers
            if isinstance(layer.get(section), dict) and layer[section].get(metric) is not None
        ]
        return round(max(values), 4) if values else None

    nuisance_null_values = [
        float(layer["nuisance_accuracy_on_null_training_fit"])
        for layer in layers
        if layer.get("nuisance_accuracy_on_null_training_fit") is not None
    ]
    payload = {
        "kind": "residualized_probe_result",
        "feature": feature_name(spec.feature),
        "label_name": getattr(spec.labels, "name", None),
        "residualize_against_name": getattr(spec.residualize_against, "name", None),
        "layers": layers,
        "summary": {
            "layer_count": len(layers),
            "example_count": len(example_keys),
            "best_raw_balanced_accuracy": _best_metric("raw_probe", "balanced_accuracy"),
            "best_residualized_balanced_accuracy": _best_metric("residualized_probe", "balanced_accuracy"),
            "best_raw_auroc": _best_metric("raw_probe", "auroc"),
            "best_residualized_auroc": _best_metric("residualized_probe", "auroc"),
            "min_nuisance_accuracy_on_null_training_fit": (
                round(min(nuisance_null_values), 4) if nuisance_null_values else None
            ),
            "max_nuisance_accuracy_on_null_training_fit": (
                round(max(nuisance_null_values), 4) if nuisance_null_values else None
            ),
            "residualization_diagnostic": (
                "nuisance_still_decodable"
                if nuisance_null_values and max(nuisance_null_values) >= 0.75
                else "nuisance_reduced"
                if nuisance_null_values
                else "nuisance_not_measured"
            ),
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


def _load_persisted_probe_payload(spec: PersistedProbeImportSpec) -> Mapping[str, Any]:
    if str(spec.format).strip().lower() != "json":
        raise SpecValidationError(f"Unsupported persisted probe import format: {spec.format!r}")
    path = Path(spec.path)
    if not path.exists():
        raise SpecValidationError(f"Persisted probe import path does not exist: {spec.path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise SpecValidationError("Persisted probe import payload must be a JSON object")
    if payload.get("kind") == "persisted_probe":
        return payload
    if payload.get("kind") == "probe_result" and isinstance(payload.get("persisted_probe"), Mapping):
        return payload["persisted_probe"]
    return _normalize_external_probe_payload(payload)


def _normalize_external_probe_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    layers = payload.get("layers")
    if not isinstance(layers, Sequence) or isinstance(layers, str):
        if "coef" in payload or "coefficients" in payload:
            layers = [payload]
        else:
            raise SpecValidationError("Persisted probe import payload must contain layers or coefficients")
    return {
        "kind": "persisted_probe",
        "name": payload.get("name"),
        "model": payload.get("model"),
        "feature_name": payload.get("feature_name") or payload.get("feature"),
        "class_names": payload.get("class_names") or payload.get("classes") or ("negative", "positive"),
        "layers": list(layers),
        "metadata": payload.get("metadata", {}),
    }


def _normalize_probe_layer(layer: Any, *, source_path: str) -> dict[str, Any]:
    if not isinstance(layer, Mapping):
        raise SpecValidationError("Persisted probe layer must be a JSON object")
    coef = layer.get("coef", layer.get("coefficients", layer.get("weights")))
    if coef is None:
        raise SpecValidationError(f"Persisted probe layer in {source_path} is missing coefficients")
    coef_array = np.asarray(coef, dtype=np.float64)
    if coef_array.ndim == 1:
        coef_array = coef_array.reshape(1, -1)
    if coef_array.ndim != 2:
        raise SpecValidationError("Persisted probe coefficients must be rank-1 or rank-2")
    intercept = np.asarray(layer.get("intercept", layer.get("bias", 0.0)), dtype=np.float64)
    if intercept.ndim == 0:
        intercept = intercept.reshape(1)
    if intercept.ndim != 1:
        raise SpecValidationError("Persisted probe intercept must be scalar or rank-1")
    if intercept.size not in {1, coef_array.shape[0]}:
        raise SpecValidationError("Persisted probe intercept width does not match coefficient rows")
    scaler_mean = layer.get("scaler_mean", layer.get("mean"))
    scaler_scale = layer.get("scaler_scale", layer.get("scale"))
    normalized = {
        "layer": int(layer.get("layer", 0)),
        "coef": coef_array.astype(float).tolist(),
        "intercept": intercept.astype(float).tolist(),
        "scaler_mean": _optional_float_vector(scaler_mean, label="scaler_mean"),
        "scaler_scale": _optional_float_vector(scaler_scale, label="scaler_scale"),
        "threshold": float(layer.get("threshold", 0.5)),
        "metadata": dict(layer.get("metadata", {})) if isinstance(layer.get("metadata"), Mapping) else {},
    }
    for key in ("training_mode", "train_stages", "stage_epochs"):
        if key in layer:
            normalized[key] = layer[key]
    return normalized


def _optional_float_vector(value: Any, *, label: str) -> list[float] | None:
    if value is None:
        return None
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1:
        raise SpecValidationError(f"Persisted probe {label} must be rank-1")
    return vector.astype(float).tolist()


def _resolve_persisted_probe(probe: Any) -> Mapping[str, Any]:
    if isinstance(probe, Mapping):
        payload = probe
    elif hasattr(probe, "result"):
        payload = probe.result()
    else:
        raise SpecValidationError(f"PersistedProbeInferenceSpec probe must be a mapping or operation artifact, got {type(probe).__name__}")
    if isinstance(payload, Mapping) and payload.get("kind") == "probe_result" and isinstance(payload.get("persisted_probe"), Mapping):
        payload = payload["persisted_probe"]
    if not isinstance(payload, Mapping) or payload.get("kind") != "persisted_probe":
        raise SpecValidationError("PersistedProbeInferenceSpec probe must resolve to a persisted_probe payload")
    return payload


def _apply_persisted_probe_layer(
    X: NDArray[np.float32],
    layer: Mapping[str, Any],
    *,
    class_names: Sequence[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    coef = np.asarray(layer.get("coef"), dtype=np.float64)
    intercept = np.asarray(layer.get("intercept", [0.0]), dtype=np.float64)
    if coef.ndim != 2:
        raise SpecValidationError("Persisted probe layer coefficients must be rank-2")
    X64 = X.astype(np.float64)
    mean = layer.get("scaler_mean")
    scale = layer.get("scaler_scale")
    if mean is not None:
        mean_array = np.asarray(mean, dtype=np.float64)
        if mean_array.shape[0] != X64.shape[1]:
            raise SpecValidationError("Persisted probe scaler_mean width does not match feature width")
        X64 = X64 - mean_array
    if scale is not None:
        scale_array = np.asarray(scale, dtype=np.float64)
        if scale_array.shape[0] != X64.shape[1]:
            raise SpecValidationError("Persisted probe scaler_scale width does not match feature width")
        X64 = X64 / np.where(scale_array == 0.0, 1.0, scale_array)
    if coef.shape[1] != X64.shape[1]:
        raise SpecValidationError("Persisted probe coefficient width does not match feature width")

    logits = X64 @ coef.T + intercept.reshape(1, -1)
    if len(class_names) == 2 and logits.shape[1] == 1:
        positive = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        probabilities = np.stack([1.0 - positive, positive], axis=1)
        predictions = [str(class_names[int(value >= float(layer.get("threshold", 0.5)))]) for value in positive]
        return logits[:, 0], probabilities, predictions
    if logits.shape[1] != len(class_names):
        raise SpecValidationError("Persisted probe class count does not match logits width")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=1, keepdims=True)
    predictions = [str(class_names[int(index)]) for index in probabilities.argmax(axis=1)]
    scores = probabilities[:, -1]
    return scores, probabilities, predictions


def _safe_label_component(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(value).strip().lower()).strip("_") or "value"


def _serialize_probe_model(
    *,
    scaler: StandardScaler,
    model: SGDClassifier,
    layer: int,
    class_names: Sequence[str] | None,
    training_example_count: int,
    training_mode: str,
    train_stages: Sequence[Sequence[Any]] = (),
    stage_epochs: Sequence[int] = (),
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "layer": int(layer),
        "coef": np.asarray(model.coef_, dtype=np.float64).astype(float).tolist(),
        "intercept": np.asarray(model.intercept_, dtype=np.float64).astype(float).tolist(),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float64).astype(float).tolist(),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float64).astype(float).tolist(),
        "threshold": 0.5,
        "training_mode": training_mode,
        "metadata": {
            "estimator": "sklearn.linear_model.SGDClassifier",
            "loss": "log_loss",
            "alpha": 1e-4,
            "class_names": [str(value) for value in class_names] if class_names is not None else [],
            "training_example_count": int(training_example_count),
            "training_mode": training_mode,
        },
    }
    if train_stages:
        resolved_epochs = _resolve_stage_epochs(train_stages, stage_epochs)
        payload["train_stages"] = [[str(value) for value in stage] for stage in train_stages]
        payload["stage_epochs"] = [int(value) for value in resolved_epochs]
        payload["metadata"]["train_stages"] = payload["train_stages"]
        payload["metadata"]["stage_epochs"] = payload["stage_epochs"]
    return payload


def probe_layer(
    *,
    layer: int,
    X: NDArray[np.float32],
    y: NDArray[np.int64],
    groups: NDArray[np.object_] | None,
    split: NDArray[np.object_] | None,
    train_values: Sequence[Any],
    train_stages: Sequence[Sequence[Any]],
    stage_epochs: Sequence[int],
    test_values: Sequence[Any],
    folds: int,
    baselines: Sequence[str],
    metrics: Sequence[str],
    example_keys: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    persist_predictions: bool = False,
    persist_model: bool = False,
) -> dict[str, Any]:
    effective_train_values = tuple(train_values)
    normalized_train_stages = tuple(tuple(stage) for stage in train_stages if stage)
    if normalized_train_stages and not effective_train_values:
        effective_train_values = tuple(
            dict.fromkeys(str(value) for stage in normalized_train_stages for value in stage)
        )
    splits, split_mode = classification_splits(
        y=y,
        groups=groups,
        split=split,
        train_values=effective_train_values,
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
    prediction_rows: list[dict[str, Any]] = []

    for fold_index, (train_idx, test_idx) in enumerate(splits):
        scaler, model = _fit_probe_model(
            X=X,
            y=y,
            train_idx=train_idx,
            split=split,
            train_stages=normalized_train_stages,
            stage_epochs=stage_epochs,
            seed=42 + fold_index,
        )
        X_test = scaler.transform(X[test_idx])
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test) if "auroc" in metrics else None
        metric_payload = compute_metric_payload(y[test_idx], predictions, probabilities, metrics=metrics)
        if "accuracy" in metric_payload:
            accuracy_scores.append(float(metric_payload["accuracy"]))
        if "balanced_accuracy" in metric_payload:
            balanced_scores.append(float(metric_payload["balanced_accuracy"]))
        auroc_value = metric_payload.get("auroc")
        if auroc_value is not None:
            auroc_scores.append(float(auroc_value))
        if persist_predictions and example_keys is not None:
            prediction_rows.extend(
                _serialize_prediction_rows(
                    example_keys=[example_keys[int(index)] for index in test_idx.tolist()],
                    y_true=y[test_idx],
                    predictions=predictions,
                    probabilities=probabilities,
                    class_names=class_names,
                    context={
                        "layer": int(layer),
                        "evaluation_kind": "probe",
                        "split_mode": split_mode,
                        "fold_index": int(fold_index),
                    },
                )
            )

        if "majority" in baseline_scores:
            baseline = DummyClassifier(strategy="most_frequent")
            baseline.fit(X[train_idx], y[train_idx])
            baseline_scores["majority"].append(float(accuracy_score(y[test_idx], baseline.predict(X[test_idx]))))
        if compute_shuffled_control:
            rng = np.random.default_rng(seed=fold_index)
            shuffled = np.array(y[train_idx], copy=True)
            rng.shuffle(shuffled)
            control_scaler = StandardScaler()
            X_train_control = control_scaler.fit_transform(X[train_idx])
            X_test_control = control_scaler.transform(X[test_idx])
            control = _make_probe_estimator(seed=4042 + fold_index)
            control.fit(X_train_control, shuffled)
            shuffled_control_scores.append(float(accuracy_score(y[test_idx], control.predict(X_test_control))))

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
    if persist_predictions:
        result["test_predictions"] = prediction_rows
        result["test_prediction_count"] = len(prediction_rows)
    if normalized_train_stages:
        result["training_mode"] = "staged_finetune"
        result["train_stages"] = [list(stage) for stage in normalized_train_stages]
        result["stage_epochs"] = list(_resolve_stage_epochs(normalized_train_stages, stage_epochs))
    if persist_model:
        if split_mode == "fixed":
            train_indices = splits[0][0]
            training_mode = "staged_finetune" if normalized_train_stages else "fixed_train_values"
        else:
            train_indices = np.arange(X.shape[0], dtype=np.int64)
            training_mode = "all_rows"
        final_scaler, final_model = _fit_probe_model(
            X=X,
            y=y,
            train_idx=train_indices,
            split=split,
            train_stages=normalized_train_stages,
            stage_epochs=stage_epochs,
            seed=9000 + int(layer),
        )
        result["persisted_probe"] = _serialize_probe_model(
            scaler=final_scaler,
            model=final_model,
            layer=layer,
            class_names=class_names,
            training_example_count=int(train_indices.size),
            training_mode=training_mode,
            train_stages=normalized_train_stages,
            stage_epochs=stage_epochs,
        )
    return result


def _make_probe_estimator(
    *,
    seed: int,
    class_weight: str | dict[int, float] | None = "balanced",
) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        class_weight=class_weight,
        max_iter=2000,
        tol=1e-3,
        random_state=seed,
    )


def _resolve_stage_epochs(
    train_stages: Sequence[Sequence[Any]],
    stage_epochs: Sequence[int],
) -> tuple[int, ...]:
    if not train_stages:
        return ()
    if not stage_epochs:
        return tuple(1 for _ in train_stages)
    if len(stage_epochs) != len(train_stages):
        raise SpecValidationError("ProbeSpec stage_epochs must match train_stages length")
    resolved = tuple(int(value) for value in stage_epochs)
    if any(value <= 0 for value in resolved):
        raise SpecValidationError("ProbeSpec stage_epochs values must be positive")
    return resolved


def _stage_indices_for_split(
    split: NDArray[np.object_],
    train_stages: Sequence[Sequence[Any]],
) -> list[NDArray[np.int64]]:
    split_labels = np.asarray([str(value) for value in split], dtype=object)
    stage_indices: list[NDArray[np.int64]] = []
    for stage in train_stages:
        allowed = {str(value) for value in stage}
        indices = np.asarray([index for index, value in enumerate(split_labels) if value in allowed], dtype=np.int64)
        if indices.size == 0:
            raise SpecValidationError(f"ProbeSpec train stage {tuple(stage)!r} selected zero rows")
        stage_indices.append(indices)
    return stage_indices


def _fit_probe_model(
    *,
    X: NDArray[np.float32],
    y: NDArray[np.int64],
    train_idx: NDArray[np.int64],
    split: NDArray[np.object_] | None,
    train_stages: Sequence[Sequence[Any]],
    stage_epochs: Sequence[int],
    seed: int,
) -> tuple[StandardScaler, SGDClassifier]:
    scaler = StandardScaler()
    if not train_stages:
        X_train = scaler.fit_transform(X[train_idx])
        model = _make_probe_estimator(seed=seed)
        model.fit(X_train, y[train_idx])
        return scaler, model

    if split is None:
        raise SpecValidationError("ProbeSpec train_stages requires a fixed split label source")

    stage_indices = _stage_indices_for_split(split, train_stages)
    allowed_train = set(train_idx.tolist())
    for stage in stage_indices:
        if not set(stage.tolist()).issubset(allowed_train):
            raise SpecValidationError("ProbeSpec train_stages selected rows outside the fixed training split")

    union_train_idx = np.unique(np.concatenate(stage_indices))
    scaler.fit(X[union_train_idx])
    epochs = _resolve_stage_epochs(train_stages, stage_epochs)
    classes = np.unique(y[union_train_idx]).astype(np.int64)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y[union_train_idx],
    )
    class_weight = {int(label): float(weight) for label, weight in zip(classes.tolist(), weights.tolist(), strict=True)}
    model = _make_probe_estimator(seed=seed, class_weight=class_weight)
    rng = np.random.default_rng(seed=seed)
    first_call = True
    for stage_idx, stage_epoch_count in zip(stage_indices, epochs):
        X_stage = scaler.transform(X[stage_idx])
        y_stage = y[stage_idx]
        for _ in range(stage_epoch_count):
            order = rng.permutation(X_stage.shape[0])
            if first_call:
                model.partial_fit(X_stage[order], y_stage[order], classes=classes)
                first_call = False
            else:
                model.partial_fit(X_stage[order], y_stage[order])
    if first_call:
        raise SpecValidationError("ProbeSpec train_stages did not produce any training updates")
    return scaler, model


def classification_splits(
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
            return [], "stratified_group_kfold"
        class_counts = np.bincount(y)
        min_class = int(class_counts.min()) if class_counts.size else 0
        n_splits = max(2, min(folds, len(unique_groups), min_class if min_class > 0 else len(unique_groups)))
        if min_class >= 2:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            return (
                [(train.astype(np.int64), test.astype(np.int64)) for train, test in splitter.split(np.zeros_like(y), y, groups)],
                "stratified_group_kfold",
            )
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


def maybe_compute_auroc(
    y_true: NDArray[np.int64],
    probabilities: NDArray[np.float64] | NDArray[np.float32] | None,
) -> float | None:
    if probabilities is None or len(np.unique(y_true)) < 2:
        return None
    if probabilities.ndim != 2 or probabilities.shape[0] != y_true.shape[0]:
        return None
    if probabilities.shape[1] == 2:
        return float(roc_auc_score(y_true, probabilities[:, 1]))
    labels = list(range(probabilities.shape[1]))
    return float(roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro", labels=labels))


def grouped_cv_activation(
    *,
    X: NDArray[np.float32],
    y: NDArray[np.int64],
    groups: NDArray[np.object_] | None,
    metrics: Sequence[str],
    C: float,
    seed: int = 42,
    example_keys: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    prediction_context: dict[str, Any] | None = None,
    persist_predictions: bool = False,
) -> dict[str, Any]:
    splits, split_mode = classification_splits(
        y=y,
        groups=groups,
        split=None,
        train_values=(),
        test_values=(),
        folds=5,
    )
    if not splits:
        return _empty_metrics(metrics, split_mode=split_mode, example_count=int(X.shape[0]))
    scores = [
        _evaluate_activation_split(
            X,
            y,
            train_idx,
            test_idx,
            metrics=metrics,
            C=C,
            seed=seed + fold,
            example_keys=example_keys if persist_predictions else None,
            class_names=class_names if persist_predictions else None,
            prediction_context={
                **(prediction_context or {}),
                "split_mode": split_mode,
                "fold_index": int(fold),
            } if persist_predictions else None,
        )
        for fold, (train_idx, test_idx) in enumerate(splits)
    ]
    return _aggregate_metric_runs(scores, split_mode=split_mode, example_count=int(X.shape[0]))


def fixed_split_activation(
    *,
    X: NDArray[np.float32],
    y: NDArray[np.int64],
    groups: NDArray[np.object_] | None,
    split_labels: Sequence[Any],
    train_values: Sequence[Any],
    test_values: Sequence[Any],
    metrics: Sequence[str],
    C: float,
    seed: int = 42,
    example_keys: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    prediction_context: dict[str, Any] | None = None,
    persist_predictions: bool = False,
) -> dict[str, Any]:
    splits, split_mode = classification_splits(
        y=y,
        groups=groups,
        split=np.asarray([str(value) for value in split_labels], dtype=object),
        train_values=train_values,
        test_values=test_values,
        folds=2,
    )
    if not splits:
        return _empty_metrics(metrics, split_mode=split_mode, example_count=int(X.shape[0]))
    train_idx, test_idx = splits[0]
    result = _evaluate_activation_split(
        X,
        y,
        train_idx,
        test_idx,
        metrics=metrics,
        C=C,
        seed=seed,
        example_keys=example_keys if persist_predictions else None,
        class_names=class_names if persist_predictions else None,
        prediction_context={
            **(prediction_context or {}),
            "split_mode": split_mode,
            "fold_index": 0,
        } if persist_predictions else None,
    )
    result["split_mode"] = split_mode
    result["example_count"] = int(X.shape[0])
    return result


def grouped_cv_text(
    *,
    texts: Sequence[str],
    y: NDArray[np.int64],
    groups: NDArray[np.object_] | None,
    model: str,
    metrics: Sequence[str],
    C: float,
    seed: int = 42,
    example_keys: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    prediction_context: dict[str, Any] | None = None,
    persist_predictions: bool = False,
) -> dict[str, Any]:
    splits, split_mode = classification_splits(
        y=y,
        groups=groups,
        split=None,
        train_values=(),
        test_values=(),
        folds=5,
    )
    if not splits:
        return _empty_metrics(metrics, split_mode=split_mode, example_count=len(texts))
    scores = [
        _evaluate_text_split(
            texts,
            y,
            train_idx,
            test_idx,
            model=model,
            metrics=metrics,
            C=C,
            seed=seed + fold,
            example_keys=example_keys if persist_predictions else None,
            class_names=class_names if persist_predictions else None,
            prediction_context={
                **(prediction_context or {}),
                "split_mode": split_mode,
                "fold_index": int(fold),
            } if persist_predictions else None,
        )
        for fold, (train_idx, test_idx) in enumerate(splits)
    ]
    return _aggregate_metric_runs(scores, split_mode=split_mode, example_count=len(texts))


def fixed_split_text(
    *,
    texts: Sequence[str],
    y: NDArray[np.int64],
    groups: NDArray[np.object_] | None,
    split_labels: Sequence[Any],
    train_values: Sequence[Any],
    test_values: Sequence[Any],
    model: str,
    metrics: Sequence[str],
    C: float,
    seed: int = 42,
    example_keys: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    prediction_context: dict[str, Any] | None = None,
    persist_predictions: bool = False,
) -> dict[str, Any]:
    splits, split_mode = classification_splits(
        y=y,
        groups=groups,
        split=np.asarray([str(value) for value in split_labels], dtype=object),
        train_values=train_values,
        test_values=test_values,
        folds=2,
    )
    if not splits:
        return _empty_metrics(metrics, split_mode=split_mode, example_count=len(texts))
    train_idx, test_idx = splits[0]
    result = _evaluate_text_split(
        texts,
        y,
        train_idx,
        test_idx,
        model=model,
        metrics=metrics,
        C=C,
        seed=seed,
        example_keys=example_keys if persist_predictions else None,
        class_names=class_names if persist_predictions else None,
        prediction_context={
            **(prediction_context or {}),
            "split_mode": split_mode,
            "fold_index": 0,
        } if persist_predictions else None,
    )
    result["split_mode"] = split_mode
    result["example_count"] = len(texts)
    return result


def _transfer_cross_layer(
    *,
    layer: int,
    X: NDArray[np.float32],
    y: NDArray[np.int64],
    class_names: Sequence[str],
    example_keys: Sequence[str],
    groups: NDArray[np.object_] | None,
    cohort_values: Sequence[Any],
    selected_cohorts: Sequence[str],
    regularization: Sequence[float],
    metrics: Sequence[str],
    compare_within_baseline: bool,
    compare_direction_similarity: bool,
    persist_predictions: bool,
) -> dict[str, Any]:
    cohort_labels = np.asarray([str(value) for value in cohort_values], dtype=object)
    bundles = {}
    for cohort in selected_cohorts:
        mask = cohort_labels == cohort
        if not mask.any():
            continue
        bundles[cohort] = {
            "X": X[mask],
            "y": y[mask],
            "groups": groups[mask] if groups is not None else None,
            "example_keys": [example_keys[index] for index in np.nonzero(mask)[0].tolist()],
        }

    within: dict[str, Any] = {}
    directions: dict[str, Any] = {}
    if compare_within_baseline:
        for cohort, bundle in bundles.items():
            within[cohort] = grouped_cv_activation(
                X=bundle["X"],
                y=bundle["y"],
                groups=bundle["groups"],
                metrics=metrics,
                C=regularization[0],
                example_keys=bundle["example_keys"],
                class_names=class_names,
                prediction_context={
                    "layer": int(layer),
                    "evaluation_kind": "within_cohort_baseline",
                    "cohort": cohort,
                    "C": float(regularization[0]),
                },
                persist_predictions=persist_predictions,
            )
    if compare_direction_similarity:
        for cohort, bundle in bundles.items():
            if len(np.unique(bundle["y"])) < 2:
                directions[cohort] = None
                continue
            pipe = _make_activation_logreg(C=regularization[0], seed=42)
            pipe.fit(bundle["X"], bundle["y"])
            directions[cohort] = extract_direction(pipe)

    transfers: dict[str, Any] = {}
    for train_cohort in selected_cohorts:
        for test_cohort in selected_cohorts:
            if train_cohort == test_cohort:
                continue
            train_bundle = bundles.get(train_cohort)
            test_bundle = bundles.get(test_cohort)
            key = f"{train_cohort}_to_{test_cohort}"
            if train_bundle is None or test_bundle is None:
                transfers[key] = {"balanced_accuracy": None}
                continue
            if len(regularization) == 1:
                result = _evaluate_activation_transfer(
                    train_bundle["X"],
                    train_bundle["y"],
                    test_bundle["X"],
                    test_bundle["y"],
                    metrics=metrics,
                    C=regularization[0],
                    example_keys_test=test_bundle["example_keys"] if persist_predictions else None,
                    class_names=class_names if persist_predictions else None,
                    prediction_context={
                        "layer": int(layer),
                        "evaluation_kind": "cross_cohort_transfer",
                        "split_mode": "cross_transfer",
                        "train_cohort": train_cohort,
                        "test_cohort": test_cohort,
                        "C": float(regularization[0]),
                    } if persist_predictions else None,
                )
                if compare_within_baseline:
                    baseline = within.get(test_cohort, {})
                    result["transfer_delta_vs_test_within"] = _metric_delta(
                        result.get("balanced_accuracy"),
                        baseline.get("balanced_accuracy"),
                    )
                transfers[key] = result
            else:
                sweep = []
                for C in regularization:
                    result = _evaluate_activation_transfer(
                        train_bundle["X"],
                        train_bundle["y"],
                        test_bundle["X"],
                        test_bundle["y"],
                        metrics=metrics,
                        C=C,
                        example_keys_test=test_bundle["example_keys"] if persist_predictions else None,
                        class_names=class_names if persist_predictions else None,
                        prediction_context={
                            "layer": int(layer),
                            "evaluation_kind": "cross_cohort_transfer",
                            "split_mode": "cross_transfer",
                            "train_cohort": train_cohort,
                            "test_cohort": test_cohort,
                            "C": float(C),
                        } if persist_predictions else None,
                    )
                    result["C"] = float(C)
                    sweep.append(result)
                transfers[key] = {"regularization_sweep": sweep}

    cosine = {}
    if compare_direction_similarity:
        for left in selected_cohorts:
            for right in selected_cohorts:
                if left >= right:
                    continue
                cosine[f"{left}_vs_{right}"] = cosine_similarity(directions.get(left), directions.get(right))

    return {
        "layer": int(layer),
        "class_names": list(class_names),
        "within_cohort_baseline": within,
        "cross_cohort_transfer": transfers,
        "direction_similarity": cosine,
    }


def _transfer_split_layer(
    *,
    layer: int,
    X: NDArray[np.float32],
    y: NDArray[np.int64],
    class_names: Sequence[str],
    example_keys: Sequence[str],
    groups: NDArray[np.object_] | None,
    cohort_values: Sequence[Any],
    selected_cohorts: Sequence[str],
    split_values: dict[str, list[Any]],
    train_values: Sequence[Any],
    test_values: Sequence[Any],
    regularization: Sequence[float],
    metrics: Sequence[str],
    persist_predictions: bool,
) -> dict[str, Any]:
    cohort_labels = np.asarray([str(value) for value in cohort_values], dtype=object) if cohort_values else None
    split_results: dict[str, Any] = {}
    for split_name, split_labels in split_values.items():
        split_labels_arr = np.asarray([str(value) for value in split_labels], dtype=object)
        if cohort_labels is None or not selected_cohorts:
            split_results[split_name] = _fixed_split_payload(
                X=X,
                y=y,
                example_keys=example_keys,
                groups=groups,
                split_labels=split_labels_arr,
                train_values=train_values,
                test_values=test_values,
                metrics=metrics,
                regularization=regularization,
                class_names=class_names,
                prediction_context={
                    "layer": int(layer),
                    "evaluation_kind": "split_holdout",
                    "split_name": split_name,
                } if persist_predictions else None,
                persist_predictions=persist_predictions,
            )
            continue
        per_cohort: dict[str, Any] = {}
        for cohort in selected_cohorts:
            mask = cohort_labels == cohort
            if not mask.any():
                continue
            per_cohort[cohort] = _fixed_split_payload(
                X=X[mask],
                y=y[mask],
                example_keys=[example_keys[index] for index in np.nonzero(mask)[0].tolist()],
                groups=groups[mask] if groups is not None else None,
                split_labels=split_labels_arr[mask],
                train_values=train_values,
                test_values=test_values,
                metrics=metrics,
                regularization=regularization,
                class_names=class_names,
                prediction_context={
                    "layer": int(layer),
                    "evaluation_kind": "split_holdout",
                    "split_name": split_name,
                    "cohort": cohort,
                } if persist_predictions else None,
                persist_predictions=persist_predictions,
            )
        split_results[split_name] = per_cohort
    return {
        "layer": int(layer),
        "class_names": list(class_names),
        "split_results": split_results,
    }


def _fixed_split_payload(
    *,
    X: NDArray[np.float32],
    y: NDArray[np.int64],
    example_keys: Sequence[str],
    groups: NDArray[np.object_] | None,
    split_labels: NDArray[np.object_],
    train_values: Sequence[Any],
    test_values: Sequence[Any],
    metrics: Sequence[str],
    regularization: Sequence[float],
    class_names: Sequence[str],
    prediction_context: dict[str, Any] | None = None,
    persist_predictions: bool = False,
) -> Any:
    if len(regularization) == 1:
        return fixed_split_activation(
            X=X,
            y=y,
            example_keys=example_keys,
            groups=groups,
            split_labels=split_labels,
            train_values=train_values,
            test_values=test_values,
            metrics=metrics,
            C=regularization[0],
            class_names=class_names,
            prediction_context={
                **(prediction_context or {}),
                "C": float(regularization[0]),
            } if persist_predictions else None,
            persist_predictions=persist_predictions,
        )
    return {
        "regularization_sweep": [
            {
                "C": float(C),
                **fixed_split_activation(
                    X=X,
                    y=y,
                    example_keys=example_keys,
                    groups=groups,
                    split_labels=split_labels,
                    train_values=train_values,
                    test_values=test_values,
                    metrics=metrics,
                    C=C,
                    class_names=class_names,
                    prediction_context={
                        **(prediction_context or {}),
                        "C": float(C),
                    } if persist_predictions else None,
                    persist_predictions=persist_predictions,
                ),
            }
            for C in regularization
        ]
    }


def _text_cross_results(
    *,
    texts: Sequence[str],
    y: NDArray[np.int64],
    class_names: Sequence[str],
    example_keys: Sequence[str],
    groups: NDArray[np.object_] | None,
    cohort_values: Sequence[Any],
    selected_cohorts: Sequence[str],
    model: str,
    regularization: Sequence[float],
    metrics: Sequence[str],
    persist_predictions: bool,
) -> dict[str, Any]:
    cohort_labels = np.asarray([str(value) for value in cohort_values], dtype=object)
    bundles = {}
    for cohort in selected_cohorts:
        mask = cohort_labels == cohort
        if not mask.any():
            continue
        bundles[cohort] = {
            "texts": [texts[index] for index in np.nonzero(mask)[0].tolist()],
            "y": y[mask],
            "groups": groups[mask] if groups is not None else None,
            "example_keys": [example_keys[index] for index in np.nonzero(mask)[0].tolist()],
        }

    within = {
        cohort: grouped_cv_text(
            texts=bundle["texts"],
            y=bundle["y"],
            groups=bundle["groups"],
            model=model,
            metrics=metrics,
            C=regularization[0],
            example_keys=bundle["example_keys"],
            class_names=class_names,
            prediction_context={
                "evaluation_kind": "within_cohort_baseline",
                "cohort": cohort,
                "model": model,
                "C": float(regularization[0]),
            },
            persist_predictions=persist_predictions,
        )
        for cohort, bundle in bundles.items()
    }

    transfers: dict[str, Any] = {}
    for train_cohort in selected_cohorts:
        for test_cohort in selected_cohorts:
            if train_cohort == test_cohort:
                continue
            train_bundle = bundles.get(train_cohort)
            test_bundle = bundles.get(test_cohort)
            key = f"{train_cohort}_to_{test_cohort}"
            if train_bundle is None or test_bundle is None:
                transfers[key] = {"balanced_accuracy": None}
                continue
            if len(regularization) == 1:
                result = _evaluate_text_transfer(
                    train_bundle["texts"],
                    train_bundle["y"],
                    test_bundle["texts"],
                    test_bundle["y"],
                    model=model,
                    metrics=metrics,
                    C=regularization[0],
                    example_keys_test=test_bundle["example_keys"] if persist_predictions else None,
                    class_names=class_names if persist_predictions else None,
                    prediction_context={
                        "evaluation_kind": "cross_cohort_transfer",
                        "split_mode": "cross_transfer",
                        "train_cohort": train_cohort,
                        "test_cohort": test_cohort,
                        "model": model,
                        "C": float(regularization[0]),
                    } if persist_predictions else None,
                )
                result["transfer_delta_vs_test_within"] = _metric_delta(
                    result.get("balanced_accuracy"),
                    within.get(test_cohort, {}).get("balanced_accuracy"),
                )
                transfers[key] = result
            else:
                transfers[key] = {
                    "regularization_sweep": [
                        {
                            "C": float(C),
                            **_evaluate_text_transfer(
                                train_bundle["texts"],
                                train_bundle["y"],
                                test_bundle["texts"],
                                test_bundle["y"],
                                model=model,
                                metrics=metrics,
                                C=C,
                                example_keys_test=test_bundle["example_keys"] if persist_predictions else None,
                                class_names=class_names if persist_predictions else None,
                                prediction_context={
                                    "evaluation_kind": "cross_cohort_transfer",
                                    "split_mode": "cross_transfer",
                                    "train_cohort": train_cohort,
                                    "test_cohort": test_cohort,
                                    "model": model,
                                    "C": float(C),
                                } if persist_predictions else None,
                            ),
                        }
                        for C in regularization
                    ]
                }
    return {
        "class_names": list(class_names),
        "within_cohort_baseline": within,
        "cross_cohort_transfer": transfers,
    }


def _text_split_results(
    *,
    texts: Sequence[str],
    y: NDArray[np.int64],
    class_names: Sequence[str],
    example_keys: Sequence[str],
    groups: NDArray[np.object_] | None,
    cohort_values: Sequence[Any],
    selected_cohorts: Sequence[str],
    split_values: dict[str, list[Any]],
    train_values: Sequence[Any],
    test_values: Sequence[Any],
    model: str,
    regularization: Sequence[float],
    metrics: Sequence[str],
    persist_predictions: bool,
) -> dict[str, Any]:
    cohort_labels = np.asarray([str(value) for value in cohort_values], dtype=object) if cohort_values else None
    results: dict[str, Any] = {"class_names": list(class_names), "split_results": {}}
    for split_name, split_labels in split_values.items():
        split_arr = np.asarray([str(value) for value in split_labels], dtype=object)
        if cohort_labels is None or not selected_cohorts:
            results["split_results"][split_name] = _text_fixed_split_payload(
                texts=texts,
                y=y,
                example_keys=example_keys,
                groups=groups,
                split_labels=split_arr,
                train_values=train_values,
                test_values=test_values,
                model=model,
                regularization=regularization,
                metrics=metrics,
                class_names=class_names,
                prediction_context={
                    "evaluation_kind": "split_holdout",
                    "split_name": split_name,
                    "model": model,
                } if persist_predictions else None,
                persist_predictions=persist_predictions,
            )
            continue
        per_cohort = {}
        for cohort in selected_cohorts:
            mask = cohort_labels == cohort
            if not mask.any():
                continue
            indices = np.nonzero(mask)[0].tolist()
            per_cohort[cohort] = _text_fixed_split_payload(
                texts=[texts[index] for index in indices],
                y=y[mask],
                example_keys=[example_keys[index] for index in indices],
                groups=groups[mask] if groups is not None else None,
                split_labels=split_arr[mask],
                train_values=train_values,
                test_values=test_values,
                model=model,
                regularization=regularization,
                metrics=metrics,
                class_names=class_names,
                prediction_context={
                    "evaluation_kind": "split_holdout",
                    "split_name": split_name,
                    "cohort": cohort,
                    "model": model,
                } if persist_predictions else None,
                persist_predictions=persist_predictions,
            )
        results["split_results"][split_name] = per_cohort
    return results


def _text_fixed_split_payload(
    *,
    texts: Sequence[str],
    y: NDArray[np.int64],
    example_keys: Sequence[str],
    groups: NDArray[np.object_] | None,
    split_labels: NDArray[np.object_],
    train_values: Sequence[Any],
    test_values: Sequence[Any],
    model: str,
    regularization: Sequence[float],
    metrics: Sequence[str],
    class_names: Sequence[str],
    prediction_context: dict[str, Any] | None = None,
    persist_predictions: bool = False,
) -> Any:
    if len(regularization) == 1:
        return fixed_split_text(
            texts=texts,
            y=y,
            example_keys=example_keys,
            groups=groups,
            split_labels=split_labels,
            train_values=train_values,
            test_values=test_values,
            model=model,
            metrics=metrics,
            C=regularization[0],
            class_names=class_names,
            prediction_context={
                **(prediction_context or {}),
                "C": float(regularization[0]),
            } if persist_predictions else None,
            persist_predictions=persist_predictions,
        )
    return {
        "regularization_sweep": [
            {
                "C": float(C),
                **fixed_split_text(
                    texts=texts,
                    y=y,
                    example_keys=example_keys,
                    groups=groups,
                    split_labels=split_labels,
                    train_values=train_values,
                    test_values=test_values,
                    model=model,
                    metrics=metrics,
                    C=C,
                    class_names=class_names,
                    prediction_context={
                        **(prediction_context or {}),
                        "C": float(C),
                    } if persist_predictions else None,
                    persist_predictions=persist_predictions,
                ),
            }
            for C in regularization
        ]
    }


def _text_grouped_cv_results(
    *,
    texts: Sequence[str],
    y: NDArray[np.int64],
    class_names: Sequence[str],
    groups: NDArray[np.object_] | None,
    example_keys: Sequence[str],
    model: str,
    regularization: Sequence[float],
    metrics: Sequence[str],
    persist_predictions: bool,
) -> dict[str, Any]:
    if len(regularization) == 1:
        return {
            "class_names": list(class_names),
            "grouped_cv": grouped_cv_text(
                texts=texts,
                y=y,
                groups=groups,
                example_keys=example_keys,
                class_names=class_names,
                model=model,
                metrics=metrics,
                C=regularization[0],
                prediction_context={
                    "evaluation_kind": "grouped_cv",
                    "model": model,
                    "C": float(regularization[0]),
                },
                persist_predictions=persist_predictions,
            ),
        }
    return {
        "class_names": list(class_names),
        "regularization_sweep": [
            {
                "C": float(C),
                **grouped_cv_text(
                    texts=texts,
                    y=y,
                    groups=groups,
                    example_keys=example_keys,
                    class_names=class_names,
                    model=model,
                    metrics=metrics,
                    C=C,
                    prediction_context={
                        "evaluation_kind": "grouped_cv",
                        "model": model,
                        "C": float(C),
                    },
                    persist_predictions=persist_predictions,
                ),
            }
            for C in regularization
        ],
    }


def _evaluate_activation_split(
    X: NDArray[np.float32],
    y: NDArray[np.int64],
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    *,
    metrics: Sequence[str],
    C: float,
    seed: int,
    example_keys: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    prediction_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model = _make_activation_logreg(C=C, seed=seed)
    model.fit(X[train_idx], y[train_idx])
    predictions = model.predict(X[test_idx])
    probabilities = model.predict_proba(X[test_idx]) if "auroc" in metrics else None
    payload = compute_metric_payload(y[test_idx], predictions, probabilities, metrics=metrics)
    if example_keys is not None:
        payload["test_predictions"] = _serialize_prediction_rows(
            example_keys=[example_keys[int(index)] for index in test_idx.tolist()],
            y_true=y[test_idx],
            predictions=predictions,
            probabilities=probabilities,
            class_names=class_names,
            context=prediction_context,
        )
    return payload


def _evaluate_text_split(
    texts: Sequence[str],
    y: NDArray[np.int64],
    train_idx: NDArray[np.int64],
    test_idx: NDArray[np.int64],
    *,
    model: str,
    metrics: Sequence[str],
    C: float,
    seed: int,
    example_keys: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    prediction_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pipeline = _make_text_model(model=model, C=C, seed=seed)
    train_texts = [texts[int(index)] for index in train_idx.tolist()]
    test_texts = [texts[int(index)] for index in test_idx.tolist()]
    pipeline.fit(train_texts, y[train_idx])
    predictions = pipeline.predict(test_texts)
    probabilities = pipeline.predict_proba(test_texts) if "auroc" in metrics else None
    payload = compute_metric_payload(y[test_idx], predictions, probabilities, metrics=metrics)
    if example_keys is not None:
        payload["test_predictions"] = _serialize_prediction_rows(
            example_keys=[example_keys[int(index)] for index in test_idx.tolist()],
            y_true=y[test_idx],
            predictions=predictions,
            probabilities=probabilities,
            class_names=class_names,
            context=prediction_context,
        )
    return payload


def _evaluate_activation_transfer(
    X_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    X_test: NDArray[np.float32],
    y_test: NDArray[np.int64],
    *,
    metrics: Sequence[str],
    C: float,
    seed: int = 42,
    example_keys_test: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    prediction_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return _empty_metrics(metrics, split_mode="cross_transfer", example_count=int(len(y_train) + len(y_test)))
    model = _make_activation_logreg(C=C, seed=seed)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test) if "auroc" in metrics else None
    payload = compute_metric_payload(y_test, predictions, probabilities, metrics=metrics)
    payload["split_mode"] = "cross_transfer"
    payload["n_train"] = int(len(y_train))
    payload["n_test"] = int(len(y_test))
    if example_keys_test is not None:
        payload["test_predictions"] = _serialize_prediction_rows(
            example_keys=example_keys_test,
            y_true=y_test,
            predictions=predictions,
            probabilities=probabilities,
            class_names=class_names,
            context=prediction_context,
        )
        payload["test_prediction_count"] = len(example_keys_test)
    return payload


def _evaluate_text_transfer(
    texts_train: Sequence[str],
    y_train: NDArray[np.int64],
    texts_test: Sequence[str],
    y_test: NDArray[np.int64],
    *,
    model: str,
    metrics: Sequence[str],
    C: float,
    seed: int = 42,
    example_keys_test: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    prediction_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return _empty_metrics(metrics, split_mode="cross_transfer", example_count=int(len(y_train) + len(y_test)))
    pipeline = _make_text_model(model=model, C=C, seed=seed)
    pipeline.fit(list(texts_train), y_train)
    predictions = pipeline.predict(list(texts_test))
    probabilities = pipeline.predict_proba(list(texts_test)) if "auroc" in metrics else None
    payload = compute_metric_payload(y_test, predictions, probabilities, metrics=metrics)
    payload["split_mode"] = "cross_transfer"
    payload["n_train"] = int(len(y_train))
    payload["n_test"] = int(len(y_test))
    if example_keys_test is not None:
        payload["test_predictions"] = _serialize_prediction_rows(
            example_keys=example_keys_test,
            y_true=y_test,
            predictions=predictions,
            probabilities=probabilities,
            class_names=class_names,
            context=prediction_context,
        )
        payload["test_prediction_count"] = len(example_keys_test)
    return payload


def compute_metric_payload(
    y_true: NDArray[np.int64],
    predictions: NDArray[np.int64],
    probabilities: NDArray[np.float64] | NDArray[np.float32] | None,
    *,
    metrics: Sequence[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if "accuracy" in metrics:
        payload["accuracy"] = round(float(accuracy_score(y_true, predictions)), 4)
    if "balanced_accuracy" in metrics:
        payload["balanced_accuracy"] = round(float(balanced_accuracy_score(y_true, predictions)), 4)
    if "auroc" in metrics:
        auroc = maybe_compute_auroc(y_true, probabilities)
        payload["auroc"] = round(auroc, 4) if auroc is not None else None
    if len(np.unique(y_true)) == 2:
        negative_label = int(np.min(y_true))
        positive_label = int(np.max(y_true))
        negative_mask = y_true == negative_label
        positive_mask = y_true == positive_label
        fp = int(np.sum((predictions == positive_label) & negative_mask))
        fn = int(np.sum((predictions == negative_label) & positive_mask))
        tp = int(np.sum((predictions == positive_label) & positive_mask))
        tn = int(np.sum((predictions == negative_label) & negative_mask))
        neg_count = int(np.sum(negative_mask))
        pos_count = int(np.sum(positive_mask))
        payload["true_negative_count"] = tn
        payload["false_positive_count"] = fp
        payload["false_negative_count"] = fn
        payload["true_positive_count"] = tp
        payload["false_positive_rate"] = round(fp / neg_count, 4) if neg_count else None
        payload["false_negative_rate"] = round(fn / pos_count, 4) if pos_count else None
        payload["true_positive_rate"] = round(tp / pos_count, 4) if pos_count else None
        payload["true_negative_rate"] = round(tn / neg_count, 4) if neg_count else None
    return payload


def _serialize_prediction_rows(
    *,
    example_keys: Sequence[str],
    y_true: NDArray[np.int64],
    predictions: NDArray[np.int64],
    probabilities: NDArray[np.float64] | NDArray[np.float32] | None,
    class_names: Sequence[str] | None,
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    class_labels = list(class_names) if class_names is not None else None
    binary_labels = sorted(int(value) for value in np.unique(y_true).tolist()) if len(np.unique(y_true)) == 2 else None
    for idx, example_key in enumerate(example_keys):
        row: dict[str, Any] = {
            "example_key": str(example_key),
            "true_label_index": int(y_true[idx]),
            "predicted_label_index": int(predictions[idx]),
            "correct": bool(int(y_true[idx]) == int(predictions[idx])),
        }
        if context:
            for key, value in context.items():
                if isinstance(value, np.integer):
                    row[str(key)] = int(value)
                elif isinstance(value, np.floating):
                    row[str(key)] = float(value)
                elif isinstance(value, np.bool_):
                    row[str(key)] = bool(value)
                else:
                    row[str(key)] = value
        if class_labels is not None:
            row["true_label"] = str(class_labels[int(y_true[idx])])
            row["predicted_label"] = str(class_labels[int(predictions[idx])])
        if probabilities is not None and probabilities.ndim == 2 and idx < probabilities.shape[0]:
            probs = probabilities[idx]
            row["class_probabilities"] = [round(float(value), 6) for value in probs.tolist()]
            if probs.shape[0] == 2:
                row["positive_class_probability"] = round(float(probs[1]), 6)
        if binary_labels is not None:
            negative_label, positive_label = binary_labels
            truth = int(y_true[idx])
            pred = int(predictions[idx])
            if truth == positive_label and pred == positive_label:
                row["binary_outcome"] = "true_positive"
            elif truth == negative_label and pred == positive_label:
                row["binary_outcome"] = "false_positive"
            elif truth == positive_label and pred == negative_label:
                row["binary_outcome"] = "false_negative"
            else:
                row["binary_outcome"] = "true_negative"
        rows.append(row)
    return rows


def _aggregate_metric_runs(
    runs: Sequence[dict[str, Any]],
    *,
    split_mode: str,
    example_count: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "split_mode": split_mode,
        "example_count": int(example_count),
    }
    numeric_keys = sorted({key for run in runs for key, value in run.items() if isinstance(value, (int, float))})
    for key in numeric_keys:
        values = [float(run[key]) for run in runs if run.get(key) is not None]
        payload[key] = round(float(np.mean(values)), 4) if values else None
        payload[f"{key}_std"] = round(float(np.std(values)), 4) if values else None
    prediction_rows = [
        row
        for run in runs
        for row in (run.get("test_predictions") or [])
        if isinstance(row, dict)
    ]
    if prediction_rows:
        payload["test_predictions"] = prediction_rows
        payload["test_prediction_count"] = len(prediction_rows)
    payload["n_folds"] = int(len(runs))
    return payload


def _empty_metrics(metrics: Sequence[str], *, split_mode: str, example_count: int) -> dict[str, Any]:
    payload = {
        "split_mode": split_mode,
        "example_count": int(example_count),
    }
    for metric in metrics:
        payload[str(metric)] = None
    return payload


def _metric_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return round(float(left) - float(right), 4)


def _make_activation_logreg(*, C: float, seed: int) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=4000,
            random_state=seed,
        ),
    )


def _make_text_model(*, model: str, C: float, seed: int) -> Any:
    if model != "countvectorizer_logreg":
        raise NotImplementedError(f"Unsupported TextBaselineSpec model family: {model!r}")
    return make_pipeline(
        CountVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=4000,
            random_state=seed,
        ),
    )


def extract_direction(model: Any) -> NDArray[np.float32]:
    classifier = model.named_steps["logisticregression"]
    coef = np.asarray(classifier.coef_, dtype=np.float32)
    return coef[0] if coef.shape[0] == 1 else coef.mean(axis=0)


def cosine_similarity(left: NDArray[np.float32] | None, right: NDArray[np.float32] | None) -> float | None:
    if left is None or right is None:
        return None
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    left_norm = float(np.linalg.norm(left64))
    right_norm = float(np.linalg.norm(right64))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return round(float(np.dot(left64, right64) / (left_norm * right_norm)), 4)
