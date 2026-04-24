from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from projects.MOREBENCH.phase_03.scripts.analyze_experiment_02_extended_metrics import _load_rows_and_matrices


CAPTURE_DATASET_ID = "transform_1_4a60e2ca"
CAPTURE_ID = "capture_1_34cdfd7923d9"
TARGET_PRIMES = ("deontology", "virtue_ethics")
TARGET_FAMILY = "description_only"
REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_deont_vs_virtue_group_holdout")
REPORT_PATH = REPORT_DIR / "report.md"
SUMMARY_PATH = REPORT_DIR / "summary.json"


def _binary_metrics(labels: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
    return {
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, preds)), 4),
        "auroc": round(float(roc_auc_score(labels, probs)), 4),
        "positive_count": int(labels.sum()),
        "negative_count": int((1 - labels).sum()),
    }


def _fit_activation_logo(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    logo = LeaveOneGroupOut()
    probs = np.zeros(labels.shape[0], dtype=np.float64)
    preds = np.zeros(labels.shape[0], dtype=np.int32)
    fold_rows: list[dict[str, Any]] = []

    for train_idx, test_idx in logo.split(matrix, labels, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=4000,
                random_state=42,
            ),
        )
        model.fit(matrix[train_idx], labels[train_idx])
        fold_probs = model.predict_proba(matrix[test_idx])[:, 1]
        fold_preds = model.predict(matrix[test_idx])
        probs[test_idx] = fold_probs
        preds[test_idx] = fold_preds
        fold_rows.append(
            {
                "held_out_group": str(groups[test_idx][0]),
                "test_size": int(test_idx.shape[0]),
                "positive_count": int(labels[test_idx].sum()),
                "negative_count": int(test_idx.shape[0] - labels[test_idx].sum()),
            }
        )

    metrics = _binary_metrics(labels, probs, preds)
    metrics["fold_count"] = len(fold_rows)
    metrics["folds"] = fold_rows
    return metrics


def _fit_text_logo(
    texts: list[str],
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    logo = LeaveOneGroupOut()
    probs = np.zeros(labels.shape[0], dtype=np.float64)
    preds = np.zeros(labels.shape[0], dtype=np.int32)

    for train_idx, test_idx in logo.split(texts, labels, groups):
        model = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(3, 5)),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=4000,
                random_state=42,
            ),
        )
        train_texts = [texts[idx] for idx in train_idx.tolist()]
        test_texts = [texts[idx] for idx in test_idx.tolist()]
        model.fit(train_texts, labels[train_idx])
        probs[test_idx] = model.predict_proba(test_texts)[:, 1]
        preds[test_idx] = model.predict(test_texts)

    return _binary_metrics(labels, probs, preds)


def _fit_length_logo(
    texts: list[str],
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict[str, Any]:
    logo = LeaveOneGroupOut()
    lengths = np.asarray([[len(text)] for text in texts], dtype=np.float32)
    probs = np.zeros(labels.shape[0], dtype=np.float64)
    preds = np.zeros(labels.shape[0], dtype=np.int32)

    for train_idx, test_idx in logo.split(lengths, labels, groups):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=4000,
                random_state=42,
            ),
        )
        model.fit(lengths[train_idx], labels[train_idx])
        probs[test_idx] = model.predict_proba(lengths[test_idx])[:, 1]
        preds[test_idx] = model.predict(lengths[test_idx])

    return _binary_metrics(labels, probs, preds)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows, matrices = _load_rows_and_matrices(
        capture_dataset_id=CAPTURE_DATASET_ID,
        capture_id=CAPTURE_ID,
    )
    selected = [
        row
        for row in rows
        if row["prime_family"] == TARGET_FAMILY and row["prime_condition"] in TARGET_PRIMES
    ]
    groups_to_primes: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        groups_to_primes[row["group_id"]].add(row["prime_condition"])
    complete_groups = sorted(
        group_id for group_id, primes in groups_to_primes.items() if set(primes) == set(TARGET_PRIMES)
    )
    filtered_indices = [
        idx for idx, row in enumerate(rows)
        if row["prime_family"] == TARGET_FAMILY
        and row["prime_condition"] in TARGET_PRIMES
        and row["group_id"] in complete_groups
    ]

    filtered_rows = [rows[idx] for idx in filtered_indices]
    texts = [row["generated_text"] for row in filtered_rows]
    labels = np.asarray([1 if row["prime_condition"] == "deontology" else 0 for row in filtered_rows], dtype=np.int32)
    groups = np.asarray([row["group_id"] for row in filtered_rows], dtype=object)
    lengths = [len(text) for text in texts]

    class_counts = Counter(row["prime_condition"] for row in filtered_rows)
    group_example_counts = Counter(row["group_id"] for row in filtered_rows)

    text_metrics = _fit_text_logo(texts, labels, groups)
    length_metrics = _fit_length_logo(texts, labels, groups)

    layer_results: list[dict[str, Any]] = []
    best_layer: dict[str, Any] | None = None
    for layer, full_matrix in sorted(matrices.items()):
        matrix = full_matrix[np.asarray(filtered_indices, dtype=np.int32)]
        metrics = _fit_activation_logo(matrix, labels, groups)
        metrics["layer"] = int(layer)
        metrics["probe_minus_text_auroc"] = round(float(metrics["auroc"] - text_metrics["auroc"]), 4)
        metrics["probe_minus_length_auroc"] = round(float(metrics["auroc"] - length_metrics["auroc"]), 4)
        layer_results.append(metrics)
        if best_layer is None or metrics["auroc"] > best_layer["auroc"]:
            best_layer = metrics

    summary = {
        "analysis": "experiment_02_deont_vs_virtue_group_holdout",
        "capture_dataset_id": CAPTURE_DATASET_ID,
        "capture_id": CAPTURE_ID,
        "prime_family": TARGET_FAMILY,
        "target_primes": list(TARGET_PRIMES),
        "complete_group_count": len(complete_groups),
        "complete_groups": complete_groups,
        "row_count": len(filtered_rows),
        "group_example_counts": dict(sorted(group_example_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "text_baseline_char_tfidf_logo": text_metrics,
        "length_baseline_logo": length_metrics,
        "layer_results": layer_results,
        "best_layer": best_layer,
        "notes": {
            "split": "leave-one-group-out over dilemma groups that retained both deontology and virtue rows in the old strict description_only capture",
            "caveat": "old strict copy filtering reduced the usable matrix from 30 groups to 21 complete groups",
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Experiment 02 Deontology vs Virtue Group Holdout",
        "",
        f"- capture dataset artifact: `{CAPTURE_DATASET_ID}`",
        f"- capture artifact: `{CAPTURE_ID}`",
        f"- prime family: `{TARGET_FAMILY}`",
        f"- target: `deontology` vs `virtue_ethics`",
        f"- usable complete groups after old strict filtering: `{len(complete_groups)}`",
        f"- usable rows: `{len(filtered_rows)}`",
        "",
        "## Group Coverage",
        f"- complete groups: `{', '.join(complete_groups)}`",
        f"- class counts: `{dict(sorted(class_counts.items()))}`",
        "",
        "## Group-Holdout Baselines",
        f"- char-TF-IDF text AUROC: `{text_metrics['auroc']}`",
        f"- char-TF-IDF text balanced accuracy: `{text_metrics['balanced_accuracy']}`",
        f"- length-only AUROC: `{length_metrics['auroc']}`",
        f"- length-only balanced accuracy: `{length_metrics['balanced_accuracy']}`",
        "",
        "## Layer Results",
    ]
    for item in layer_results:
        report_lines.append(
            f"- layer `{item['layer']}`: probe AUROC `{item['auroc']}`, BA `{item['balanced_accuracy']}`, "
            f"delta vs text `{item['probe_minus_text_auroc']}`, delta vs length `{item['probe_minus_length_auroc']}`"
        )
    if best_layer is not None:
        report_lines.extend(
            [
                "",
                "## Best Layer",
                f"- best layer: `{best_layer['layer']}`",
                f"- probe AUROC: `{best_layer['auroc']}`",
                f"- probe BA: `{best_layer['balanced_accuracy']}`",
                f"- delta vs text AUROC: `{best_layer['probe_minus_text_auroc']}`",
                f"- delta vs length AUROC: `{best_layer['probe_minus_length_auroc']}`",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Read",
            "- This is a true dilemma-group holdout over the surviving complete groups, not the earlier bank-only split.",
            "- It only answers the question on the old strict description-only capture substrate.",
            "- It does not test name_only or full 30-group coverage because those activations are not available in usable form.",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "summary_path": str(SUMMARY_PATH),
        "report_path": str(REPORT_PATH),
        "best_layer": best_layer,
        "text_baseline": text_metrics,
        "length_baseline": length_metrics,
        "complete_group_count": len(complete_groups),
        "row_count": len(filtered_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
