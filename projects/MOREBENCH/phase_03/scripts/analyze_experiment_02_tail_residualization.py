from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from projects.MOREBENCH.phase_03.scripts.analyze_experiment_02_family_transfer_tail import (
    FamilySpec,
    _encode_labels,
    _merge_family_tail_data,
)


REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_tail_residualization")


def _build_text_model(kind: str):
    if kind == "count_logreg":
        return make_pipeline(
            CountVectorizer(ngram_range=(1, 2), min_df=1),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=42),
        )
    if kind == "tfidf_logreg":
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=42),
        )
    if kind == "tfidf_knn":
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1),
            KNeighborsClassifier(n_neighbors=1, metric="cosine"),
        )
    raise ValueError(f"Unknown text model kind: {kind}")


def _macro_ovr_auroc(
    *,
    probs: np.ndarray,
    test_y: np.ndarray,
    class_names: list[str],
) -> tuple[float | None, dict[str, Any]]:
    valid: list[float] = []
    per_class: dict[str, Any] = {}
    for class_index, class_name in enumerate(class_names):
        y_true = (test_y == class_index).astype(int)
        support = int(y_true.sum())
        recall = None
        auroc = None
        if support > 0:
            preds = (np.argmax(probs, axis=1) == class_index).astype(int)
            recall = round(float(recall_score(y_true, preds, zero_division=0)), 4)
        if 0 < support < len(y_true):
            value = float(roc_auc_score(y_true, probs[:, class_index]))
            auroc = round(value, 4)
            valid.append(value)
        per_class[class_name] = {"support": support, "recall": recall, "auroc": auroc}
    macro = round(float(np.mean(valid)), 4) if valid else None
    return macro, per_class


def _metrics(preds: np.ndarray, probs: np.ndarray, test_y: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    macro_auroc, per_class = _macro_ovr_auroc(probs=probs, test_y=test_y, class_names=class_names)
    return {
        "accuracy": round(float(accuracy_score(test_y, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(test_y, preds)), 4),
        "macro_ovr_auroc": macro_auroc,
        "per_class": per_class,
    }


def _fit_text_model(kind: str, train_texts: list[str], train_y: np.ndarray, test_texts: list[str], test_y: np.ndarray, class_names: list[str]) -> tuple[Any, np.ndarray, np.ndarray, dict[str, Any]]:
    model = _build_text_model(kind)
    model.fit(train_texts, train_y)
    train_probs = model.predict_proba(train_texts)
    test_probs = model.predict_proba(test_texts)
    test_preds = model.predict(test_texts)
    return model, train_probs, test_probs, _metrics(test_preds, test_probs, test_y, class_names)


def _probe_metrics(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=42),
    )
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    probs = model.predict_proba(test_x)
    return _metrics(preds, probs, test_y, class_names)


def _residualize_from_text_probs(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_text_probs: np.ndarray,
    test_text_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_design = np.concatenate(
        [np.ones((train_text_probs.shape[0], 1), dtype=np.float32), train_text_probs.astype(np.float32)],
        axis=1,
    )
    test_design = np.concatenate(
        [np.ones((test_text_probs.shape[0], 1), dtype=np.float32), test_text_probs.astype(np.float32)],
        axis=1,
    )
    beta, *_ = np.linalg.lstsq(train_design, train_x.astype(np.float32), rcond=None)
    residual_train = train_x.astype(np.float32) - train_design @ beta
    residual_test = test_x.astype(np.float32) - test_design @ beta
    return residual_train, residual_test


def _theory_only_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    per_class = dict(metrics.get("per_class", {}))
    filtered = {
        name: payload
        for name, payload in per_class.items()
        if name != "generic_ethics_control" and int(payload.get("support") or 0) > 0
    }
    recalls = [float(payload["recall"]) for payload in filtered.values() if payload.get("recall") is not None]
    aucs = [float(payload["auroc"]) for payload in filtered.values() if payload.get("auroc") is not None]
    return {
        "classes": sorted(filtered),
        "macro_theory_recall": round(float(np.mean(recalls)), 4) if recalls else None,
        "macro_theory_auroc": round(float(np.mean(aucs)), 4) if aucs else None,
    }


def _evaluate_fold(
    *,
    rows: list[dict[str, Any]],
    matrices: dict[int, np.ndarray],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    fold_name: str,
) -> dict[str, Any]:
    y, class_names = _encode_labels(rows)
    train_rows = [row for row, keep in zip(rows, train_mask.tolist(), strict=True) if keep]
    test_rows = [row for row, keep in zip(rows, test_mask.tolist(), strict=True) if keep]
    train_texts = [row["tail_text"] for row in train_rows]
    test_texts = [row["tail_text"] for row in test_rows]
    train_y = y[train_mask]
    test_y = y[test_mask]

    text_candidates: list[dict[str, Any]] = []
    fitted_by_name: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for kind in ("count_logreg", "tfidf_logreg", "tfidf_knn"):
        _, train_probs, test_probs, metrics = _fit_text_model(kind, train_texts, train_y, test_texts, test_y, class_names)
        entry = {"name": kind, **metrics, "theory_only": _theory_only_summary(metrics)}
        text_candidates.append(entry)
        fitted_by_name[kind] = (train_probs, test_probs, entry)
    best_text = max(
        text_candidates,
        key=lambda item: float("-inf") if item["macro_ovr_auroc"] is None else item["macro_ovr_auroc"],
    )
    best_kind = str(best_text["name"])
    train_probs, test_probs, _ = fitted_by_name[best_kind]

    residualized_by_layer: list[dict[str, Any]] = []
    for layer, matrix in sorted(matrices.items()):
        residual_train, residual_test = _residualize_from_text_probs(
            matrix[train_mask],
            matrix[test_mask],
            train_probs,
            test_probs,
        )
        metrics = _probe_metrics(residual_train, train_y, residual_test, test_y, class_names)
        residualized_by_layer.append(
            {
                "layer": layer,
                **metrics,
                "theory_only": _theory_only_summary(metrics),
                "residualized_probe_minus_best_text_balanced_accuracy": round(
                    metrics["balanced_accuracy"] - float(best_text["balanced_accuracy"]),
                    4,
                ),
                "residualized_probe_minus_best_text_macro_ovr_auroc": (
                    None
                    if metrics["macro_ovr_auroc"] is None or best_text["macro_ovr_auroc"] is None
                    else round(metrics["macro_ovr_auroc"] - float(best_text["macro_ovr_auroc"]), 4)
                ),
            }
        )
    best_residualized = max(
        residualized_by_layer,
        key=lambda item: float("-inf") if item["macro_ovr_auroc"] is None else item["macro_ovr_auroc"],
    )
    return {
        "fold_name": fold_name,
        "train_example_count": int(train_mask.sum()),
        "test_example_count": int(test_mask.sum()),
        "test_prime_support": {
            label: sum(1 for row in test_rows if row["prime_condition"] == label)
            for label in sorted({row["prime_condition"] for row in test_rows})
        },
        "text_baselines": text_candidates,
        "best_text_baseline_by_macro_ovr_auroc": best_text,
        "residualized_probe_by_layer": residualized_by_layer,
        "best_residualized_probe_layer": best_residualized,
    }


def run_analysis(
    *,
    description_capture_dataset_id: str,
    description_capture_id: str,
    name_capture_dataset_id: str | None,
    name_capture_id: str,
    alias_capture_dataset_id: str,
    alias_capture_id: str,
    tail_fraction: float,
    name_capture_dataset_json: str | None = None,
) -> dict[str, Any]:
    family_specs = [
        FamilySpec("description_only", description_capture_dataset_id, description_capture_id),
        FamilySpec("name_only", name_capture_dataset_id, name_capture_id, capture_dataset_json=name_capture_dataset_json),
        FamilySpec("alias_only", alias_capture_dataset_id, alias_capture_id),
    ]
    rows, matrices, _ = _merge_family_tail_data(family_specs, tail_fraction=tail_fraction)
    prime_families = np.asarray([row["prime_family"] for row in rows], dtype=object)
    folds = {
        "holdout_alias_only": {
            "train_mask": prime_families != "alias_only",
            "test_mask": prime_families == "alias_only",
        },
        "holdout_description_only": {
            "train_mask": prime_families != "description_only",
            "test_mask": prime_families == "description_only",
        },
    }
    payload: dict[str, Any] = {"tail_fraction": tail_fraction, "folds": {}}
    for name, split in folds.items():
        payload["folds"][name] = _evaluate_fold(
            rows=rows,
            matrices=matrices,
            train_mask=np.asarray(split["train_mask"], dtype=bool),
            test_mask=np.asarray(split["test_mask"], dtype=bool),
            fold_name=name,
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--description-capture-dataset-id", required=True)
    parser.add_argument("--description-capture-id", required=True)
    parser.add_argument("--name-capture-dataset-id", default=None)
    parser.add_argument("--name-capture-id", required=True)
    parser.add_argument("--name-capture-dataset-json", default=None)
    parser.add_argument("--alias-capture-dataset-id", required=True)
    parser.add_argument("--alias-capture-id", required=True)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "tail_residualization.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_analysis(
        description_capture_dataset_id=args.description_capture_dataset_id,
        description_capture_id=args.description_capture_id,
        name_capture_dataset_id=args.name_capture_dataset_id,
        name_capture_id=args.name_capture_id,
        alias_capture_dataset_id=args.alias_capture_dataset_id,
        alias_capture_id=args.alias_capture_id,
        tail_fraction=float(args.tail_fraction),
        name_capture_dataset_json=args.name_capture_dataset_json,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
