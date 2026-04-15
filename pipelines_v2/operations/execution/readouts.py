"""Execution helpers for readout and baseline specs."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipelines_v2.core.types import SpecValidationError
from pipelines_v2.operations.readouts import ProbeSpec, ResidualizedProbeSpec, TextBaselineSpec, TransferProbeSpec

from .common import (
    OperationExecutionResult,
    align_example_keys_to_rows,
    encode_labels,
    feature_matrices,
    feature_name,
    filter_matrix_by_keys,
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
                test_values=tuple(spec.test_values),
                folds=spec.folds,
                baselines=tuple(spec.baselines),
                metrics=requested_metrics,
                example_keys=example_keys,
                class_names=classes,
            )
        )

    best_metric = "balanced_accuracy" if "balanced_accuracy" in requested_metrics else requested_metrics[0]
    best = max(layer_results, key=lambda item: float(item.get(best_metric, 0.0)))
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
    return OperationExecutionResult(
        payload=payload,
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
                    groups=groups,
                    cohort_values=cohort_values,
                    selected_cohorts=selected_cohorts,
                    regularization=regularization,
                    metrics=tuple(spec.metrics),
                    compare_within_baseline=spec.compare_within_baseline,
                    compare_direction_similarity=spec.compare_direction_similarity,
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
        )
        mode = "split_holdout"
    elif spec.cohort_by is not None:
        results = _text_cross_results(
            texts=text_values,
            y=y,
            class_names=class_names,
            groups=groups,
            cohort_values=cohort_values,
            selected_cohorts=selected_cohorts,
            model=spec.model,
            regularization=regularization,
            metrics=tuple(spec.metrics),
        )
        mode = "cross_cohort_transfer"
    else:
        results = _text_grouped_cv_results(
            texts=text_values,
            y=y,
            class_names=class_names,
            groups=groups,
            model=spec.model,
            regularization=regularization,
            metrics=tuple(spec.metrics),
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

    payload = {
        "kind": "residualized_probe_result",
        "feature": feature_name(spec.feature),
        "label_name": getattr(spec.labels, "name", None),
        "residualize_against_name": getattr(spec.residualize_against, "name", None),
        "layers": layers,
        "summary": {
            "layer_count": len(layers),
            "example_count": len(example_keys),
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


def probe_layer(
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
    example_keys: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    splits, split_mode = classification_splits(
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
    fixed_split_predictions: list[dict[str, Any]] | None = None

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
        metric_payload = compute_metric_payload(y[test_idx], predictions, probabilities, metrics=metrics)
        if "accuracy" in metric_payload:
            accuracy_scores.append(float(metric_payload["accuracy"]))
        if "balanced_accuracy" in metric_payload:
            balanced_scores.append(float(metric_payload["balanced_accuracy"]))
        auroc_value = metric_payload.get("auroc")
        if auroc_value is not None:
            auroc_scores.append(float(auroc_value))
        if split is not None and len(splits) == 1 and example_keys is not None:
            fixed_split_predictions = _serialize_prediction_rows(
                example_keys=[example_keys[int(index)] for index in test_idx.tolist()],
                y_true=y[test_idx],
                predictions=predictions,
                probabilities=probabilities,
                class_names=class_names,
            )

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
    if fixed_split_predictions is not None:
        result["test_predictions"] = fixed_split_predictions
    return result


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
    scores = [_evaluate_activation_split(X, y, train_idx, test_idx, metrics=metrics, C=C, seed=seed + fold) for fold, (train_idx, test_idx) in enumerate(splits)]
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
        example_keys=example_keys,
        class_names=class_names,
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
    scores = [_evaluate_text_split(texts, y, train_idx, test_idx, model=model, metrics=metrics, C=C, seed=seed + fold) for fold, (train_idx, test_idx) in enumerate(splits)]
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
        example_keys=example_keys,
        class_names=class_names,
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
    groups: NDArray[np.object_] | None,
    cohort_values: Sequence[Any],
    selected_cohorts: Sequence[str],
    regularization: Sequence[float],
    metrics: Sequence[str],
    compare_within_baseline: bool,
    compare_direction_similarity: bool,
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
    groups: NDArray[np.object_] | None,
    cohort_values: Sequence[Any],
    selected_cohorts: Sequence[str],
    model: str,
    regularization: Sequence[float],
    metrics: Sequence[str],
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
        }

    within = {
        cohort: grouped_cv_text(
            texts=bundle["texts"],
            y=bundle["y"],
            groups=bundle["groups"],
            model=model,
            metrics=metrics,
            C=regularization[0],
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
    model: str,
    regularization: Sequence[float],
    metrics: Sequence[str],
) -> dict[str, Any]:
    if len(regularization) == 1:
        return {
            "class_names": list(class_names),
            "grouped_cv": grouped_cv_text(
                texts=texts,
                y=y,
                groups=groups,
                model=model,
                metrics=metrics,
                C=regularization[0],
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
                    model=model,
                    metrics=metrics,
                    C=C,
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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    class_labels = list(class_names) if class_names is not None else None
    for idx, example_key in enumerate(example_keys):
        row: dict[str, Any] = {
            "example_key": str(example_key),
            "true_label_index": int(y_true[idx]),
            "predicted_label_index": int(predictions[idx]),
        }
        if class_labels is not None:
            row["true_label"] = str(class_labels[int(y_true[idx])])
            row["predicted_label"] = str(class_labels[int(predictions[idx])])
        if probabilities is not None and probabilities.ndim == 2 and idx < probabilities.shape[0]:
            probs = probabilities[idx]
            row["class_probabilities"] = [round(float(value), 6) for value in probs.tolist()]
            if probs.shape[0] == 2:
                row["positive_class_probability"] = round(float(probs[1]), 6)
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
