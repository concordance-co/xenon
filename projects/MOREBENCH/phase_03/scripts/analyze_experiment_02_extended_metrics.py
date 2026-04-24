from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipelines_v2.api import ModalVolumeStore, PostgresCatalog, PostgresSource, TokenPooling, TokenSelector, TransferPolicy
from pipelines_v2.operations.execution.common import feature_matrices
from pipelines_v2.storage.artifacts import CaptureArtifact, OperationArtifact, artifact_from_manifest


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
ARTIFACT_ROOT = "/data/artifacts/morebench_phase_03_experiment02"
ARTIFACT_STORE_NAME = "xenon-data"
FEATURE_NAME = "generated_sequence_residual"
REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_extended_analysis")


def _catalog() -> PostgresCatalog:
    return PostgresCatalog(source=PostgresSource.from_env(DB_ENV_VAR))


def _store() -> ModalVolumeStore:
    return ModalVolumeStore(
        name=ARTIFACT_STORE_NAME,
        root=ARTIFACT_ROOT,
        transfer_policy=TransferPolicy(allow_large_transfer=True),
    )


def _load_operation_artifact(artifact_id: str) -> OperationArtifact:
    manifest = _catalog().load_artifact(artifact_id)
    if manifest is None:
        raise RuntimeError(f"Could not load artifact manifest {artifact_id!r}")
    artifact = artifact_from_manifest(manifest, store=_store())
    if not isinstance(artifact, OperationArtifact):
        raise TypeError(f"Artifact {artifact_id!r} is not an operation artifact")
    return artifact


def _load_capture_artifact(artifact_id: str) -> CaptureArtifact:
    manifest = _catalog().load_artifact(artifact_id)
    if manifest is None:
        raise RuntimeError(f"Could not load artifact manifest {artifact_id!r}")
    artifact = artifact_from_manifest(manifest, store=_store())
    if not isinstance(artifact, CaptureArtifact):
        raise TypeError(f"Artifact {artifact_id!r} is not a capture artifact")
    return artifact


def _row_from_example(example: dict[str, Any]) -> dict[str, Any]:
    labels = dict(example.get("labels", {}))
    return {
        "example_key": str(example["key"]),
        "group_id": str(labels.get("group_id") or ""),
        "split": str(labels.get("split") or ""),
        "prime_condition": str(labels.get("prime_condition") or ""),
        "prime_family": str(labels.get("prime_family") or ""),
        "generated_text": str(labels.get("generated_text") or ""),
    }


def _load_rows_and_matrices(
    *,
    capture_dataset_id: str,
    capture_id: str,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    dataset_artifact = _load_operation_artifact(capture_dataset_id)
    capture_artifact = _load_capture_artifact(capture_id)
    payload = dataset_artifact.result()
    dataset_payload = payload.get("dataset") if isinstance(payload, dict) else None
    if not isinstance(dataset_payload, dict):
        raise TypeError(f"Capture dataset artifact {capture_dataset_id!r} missing serialized dataset")
    rows = [_row_from_example(example) for example in list(dataset_payload.get("examples", []))]
    row_map = {row["example_key"]: row for row in rows}

    matrices, example_keys = feature_matrices(
        capture_artifact.feature(FEATURE_NAME),
        token_selector=TokenSelector.full_sequence(),
        token_pooling=TokenPooling.mean(),
    )
    missing = sorted(set(row_map) - set(example_keys))
    if missing:
        raise RuntimeError(f"Feature rows missing {len(missing)} dataset example keys; sample={missing[:5]}")
    ordered_rows = [row_map[key] for key in example_keys]
    ordered_matrices = {layer: matrix.astype(np.float32) for layer, matrix in matrices.items()}
    return ordered_rows, ordered_matrices


def _class_support(labels: list[str]) -> dict[str, int]:
    support: dict[str, int] = {}
    for label in labels:
        support[label] = support.get(label, 0) + 1
    return dict(sorted(support.items()))


def _multiclass_probe_metrics(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    )
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    probs = model.predict_proba(test_x)
    result: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(test_y, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(test_y, preds)), 4),
        "test_example_count": int(test_y.shape[0]),
        "class_support_test": _class_support([class_names[int(idx)] for idx in test_y.tolist()]),
    }

    valid_aurocs: list[float] = []
    per_class: dict[str, Any] = {}
    for class_index, class_name in enumerate(class_names):
        y_true = (test_y == class_index).astype(int)
        y_pred = (preds == class_index).astype(int)
        support = int(y_true.sum())
        class_result = {
            "support": support,
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "auroc": None,
        }
        if 0 < support < len(y_true):
            auroc = float(roc_auc_score(y_true, probs[:, class_index]))
            class_result["auroc"] = round(auroc, 4)
            valid_aurocs.append(auroc)
        per_class[class_name] = class_result
    result["macro_ovr_auroc"] = round(float(np.mean(valid_aurocs)), 4) if valid_aurocs else None
    result["per_class_ovr"] = per_class
    return result


def _multiclass_text_metrics(train_texts: list[str], train_y: np.ndarray, test_texts: list[str], test_y: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    model = make_pipeline(
        CountVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=4000,
            random_state=42,
        ),
    )
    model.fit(train_texts, train_y)
    preds = model.predict(test_texts)
    probs = model.predict_proba(test_texts)
    result: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(test_y, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(test_y, preds)), 4),
        "test_example_count": int(test_y.shape[0]),
        "class_support_test": _class_support([class_names[int(idx)] for idx in test_y.tolist()]),
    }
    valid_aurocs: list[float] = []
    per_class: dict[str, Any] = {}
    for class_index, class_name in enumerate(class_names):
        y_true = (test_y == class_index).astype(int)
        y_pred = (preds == class_index).astype(int)
        support = int(y_true.sum())
        class_result = {
            "support": support,
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "auroc": None,
        }
        if 0 < support < len(y_true):
            auroc = float(roc_auc_score(y_true, probs[:, class_index]))
            class_result["auroc"] = round(auroc, 4)
            valid_aurocs.append(auroc)
        per_class[class_name] = class_result
    result["macro_ovr_auroc"] = round(float(np.mean(valid_aurocs)), 4) if valid_aurocs else None
    result["per_class_ovr"] = per_class
    return result


def _binary_metrics_from_scores(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    threshold = float(np.median(scores))
    preds = (scores >= threshold).astype(int)
    result: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, preds)), 4),
        "auroc": None,
        "support_positive": int(labels.sum()),
        "support_negative": int((1 - labels).sum()),
    }
    if 0 < int(labels.sum()) < len(labels):
        result["auroc"] = round(float(roc_auc_score(labels, scores)), 4)
    return result


def _binary_probe_metrics(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict[str, Any]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=4000,
            random_state=42,
        ),
    )
    model.fit(train_x, train_y)
    probs = model.predict_proba(test_x)[:, 1]
    return _binary_metrics_from_scores(probs, test_y)


def _binary_text_metrics(train_texts: list[str], train_y: np.ndarray, test_texts: list[str], test_y: np.ndarray) -> dict[str, Any]:
    model = make_pipeline(
        CountVectorizer(ngram_range=(1, 2), min_df=1),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=4000,
            random_state=42,
        ),
    )
    model.fit(train_texts, train_y)
    probs = model.predict_proba(test_texts)[:, 1]
    return _binary_metrics_from_scores(probs, test_y)


def _direction_ranking(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict[str, Any]:
    pos = train_x[train_y == 1]
    neg = train_x[train_y == 0]
    if pos.shape[0] == 0 or neg.shape[0] == 0:
        raise RuntimeError("Direction ranking requires both classes in train")
    direction = pos.mean(axis=0) - neg.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm > 0:
        direction = direction / norm
    scores = test_x @ direction
    return {
        "auroc": round(float(roc_auc_score(test_y, scores)), 4) if 0 < int(test_y.sum()) < len(test_y) else None,
        "score_mean_positive": round(float(scores[test_y == 1].mean()), 4) if int(test_y.sum()) > 0 else None,
        "score_mean_negative": round(float(scores[test_y == 0].mean()), 4) if int((1 - test_y).sum()) > 0 else None,
        "test_example_count": int(test_y.shape[0]),
        "support_positive": int(test_y.sum()),
        "support_negative": int((1 - test_y).sum()),
    }


def run_analysis(*, capture_dataset_id: str, capture_id: str) -> dict[str, Any]:
    rows, matrices = _load_rows_and_matrices(capture_dataset_id=capture_dataset_id, capture_id=capture_id)
    splits = np.asarray([row["split"] for row in rows], dtype=object)
    train_mask = splits == "train"
    test_mask = splits == "test"
    train_rows = [row for row, keep in zip(rows, train_mask.tolist(), strict=True) if keep]
    test_rows = [row for row, keep in zip(rows, test_mask.tolist(), strict=True) if keep]

    class_names = sorted({row["prime_condition"] for row in rows})
    class_index = {name: idx for idx, name in enumerate(class_names)}
    y = np.asarray([class_index[row["prime_condition"]] for row in rows], dtype=np.int64)
    train_y = y[train_mask]
    test_y = y[test_mask]
    train_texts = [row["generated_text"] for row in train_rows]
    test_texts = [row["generated_text"] for row in test_rows]

    multiclass_probe: list[dict[str, Any]] = []
    for layer, matrix in sorted(matrices.items()):
        metrics = _multiclass_probe_metrics(matrix[train_mask], train_y, matrix[test_mask], test_y, class_names)
        metrics["layer"] = layer
        multiclass_probe.append(metrics)
    multiclass_text = _multiclass_text_metrics(train_texts, train_y, test_texts, test_y, class_names)

    binary_labels = np.asarray([0 if row["prime_condition"] == "generic_ethics_control" else 1 for row in rows], dtype=np.int64)
    binary_probe: list[dict[str, Any]] = []
    for layer, matrix in sorted(matrices.items()):
        metrics = _binary_probe_metrics(
            matrix[train_mask],
            binary_labels[train_mask],
            matrix[test_mask],
            binary_labels[test_mask],
        )
        metrics["layer"] = layer
        binary_probe.append(metrics)
    binary_text = _binary_text_metrics(
        train_texts,
        binary_labels[train_mask],
        test_texts,
        binary_labels[test_mask],
    )

    ud_mask = np.asarray([row["prime_condition"] in {"utilitarian", "deontology"} for row in rows], dtype=bool)
    ud_rows = [row for row, keep in zip(rows, ud_mask.tolist(), strict=True) if keep]
    ud_splits = np.asarray([row["split"] for row in ud_rows], dtype=object)
    ud_train_mask = ud_splits == "train"
    ud_test_mask = ud_splits == "test"
    ud_labels = np.asarray([1 if row["prime_condition"] == "utilitarian" else 0 for row in ud_rows], dtype=np.int64)
    ranking_probe: list[dict[str, Any]] = []
    ud_matrices = {layer: matrix[ud_mask] for layer, matrix in matrices.items()}
    ud_train_texts = [row["generated_text"] for row in ud_rows if row["split"] == "train"]
    ud_test_texts = [row["generated_text"] for row in ud_rows if row["split"] == "test"]
    for layer, matrix in sorted(ud_matrices.items()):
        metrics = _direction_ranking(
            matrix[ud_train_mask],
            ud_labels[ud_train_mask],
            matrix[ud_test_mask],
            ud_labels[ud_test_mask],
        )
        metrics["layer"] = layer
        ranking_probe.append(metrics)
    ranking_text = _binary_text_metrics(
        ud_train_texts,
        ud_labels[ud_train_mask],
        ud_test_texts,
        ud_labels[ud_test_mask],
    )

    return {
        "capture_dataset_id": capture_dataset_id,
        "capture_id": capture_id,
        "train_example_count": int(train_mask.sum()),
        "test_example_count": int(test_mask.sum()),
        "test_prime_support": _class_support([row["prime_condition"] for row in test_rows]),
        "multiclass_text_baseline": multiclass_text,
        "multiclass_probe": multiclass_probe,
        "binary_framework_vs_generic_text_baseline": binary_text,
        "binary_framework_vs_generic_probe": binary_probe,
        "ranking_utilitarian_vs_deontology_text_baseline": ranking_text,
        "ranking_utilitarian_vs_deontology_probe": ranking_probe,
        "ranking_utilitarian_vs_deontology_support": {
            "train": _class_support([row["prime_condition"] for row in ud_rows if row["split"] == "train"]),
            "test": _class_support([row["prime_condition"] for row in ud_rows if row["split"] == "test"]),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dataset-id", required=True)
    parser.add_argument("--capture-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_analysis(capture_dataset_id=args.capture_dataset_id, capture_id=args.capture_id)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "description_only_extended_metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
