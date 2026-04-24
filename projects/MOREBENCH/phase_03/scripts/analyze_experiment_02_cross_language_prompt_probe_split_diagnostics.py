from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import modal
import numpy as np
from safetensors.numpy import load as load_safetensors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipelines_v2.storage.features import decode_feature_payload
from projects.MOREBENCH.phase_03.specs import experiment_02_cross_language_prompt_probe_allpairs_capture_workflow as allpairs


CATALOG_ROOT = Path("artifacts") / "morebench_phase03_experiment02_cross_language_prompt_probe_allpairs_catalog"
ALLPAIRS_SUMMARY_PATH = Path(
    "projects/MOREBENCH/phase_03/reports/experiment_02_cross_language_prompt_probe_allpairs/summary.json"
)
REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_cross_language_prompt_probe_split_diagnostics")
REPORT_PATH = REPORT_DIR / "report.md"
SUMMARY_PATH = REPORT_DIR / "summary.json"

LANGUAGE_ORDER = ("en", "es", "zh")
SOURCE_FAMILY_ORDER = ("daily_dilemmas", "ai_risk_dilemmas", "expert_written_collab")
PAIR_ORDER = list(combinations(allpairs.TARGET_PRIMES, 2))


def _artifact_manifest(capture_artifact_id: str) -> dict[str, Any]:
    path = CATALOG_ROOT / f"{capture_artifact_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _modal_relative_path(path: str) -> str:
    return path.removeprefix("/data/")


def _load_capture_feature(ref: dict[str, Any]) -> dict[str, Any]:
    volume = modal.Volume.from_name(str(ref["name"]))
    metadata = json.loads(b"".join(volume.read_file(_modal_relative_path(str(ref["metadata_path"])))))
    tensors = load_safetensors(b"".join(volume.read_file(_modal_relative_path(str(ref["tensor_path"])))))
    return decode_feature_payload(metadata, tensors)


def _load_records() -> list[dict[str, Any]]:
    dataset = allpairs.build_dataset()
    records: list[dict[str, Any]] = []
    for example in dataset.examples:
        labels = dict(example.labels)
        prompt_messages = list(example.prompt)
        prompt_text = "\n\n".join(str(message.get("content") or "") for message in prompt_messages)
        records.append(
            {
                "key": str(example.key),
                "group_id": str(labels["group_id"]),
                "prime_condition": str(labels["prime_condition"]),
                "language_code": str(labels["language_code"]),
                "source_family": str(labels["source_family"]),
                "prompt_text": prompt_text,
            }
        )
    return records


def _label_for(record: dict[str, Any], positive_label: str) -> int:
    return 1 if record["prime_condition"] == positive_label else 0


def _vector_for(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2:
        return np.asarray(array[-1], dtype=np.float32)
    return np.asarray(array, dtype=np.float32)


def _candidate_thresholds(probs: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(probs, dtype=np.float64))
    if values.size == 0:
        return np.asarray([0.5], dtype=np.float64)
    mids = (values[:-1] + values[1:]) / 2.0 if values.size > 1 else np.asarray([], dtype=np.float64)
    thresholds = np.concatenate(
        [
            np.asarray([0.0], dtype=np.float64),
            mids,
            np.asarray([1.0], dtype=np.float64),
        ]
    )
    return np.unique(np.clip(thresholds, 0.0, 1.0))


def _best_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    thresholds = _candidate_thresholds(probs)
    best_threshold = 0.5
    best_bal_acc = -1.0
    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        bal_acc = float(balanced_accuracy_score(y_true, preds))
        if bal_acc > best_bal_acc or (bal_acc == best_bal_acc and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_bal_acc = bal_acc
            best_threshold = float(threshold)
    return best_threshold, best_bal_acc


def _fit_text_probs(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    model = LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear")
    model.fit(x_train, train_labels)
    train_probs = model.predict_proba(x_train)[:, 1]
    test_probs = model.predict_proba(x_test)[:, 1]
    return train_probs, test_probs


def _fit_probe_probs(
    train_vectors: np.ndarray,
    train_labels: list[int],
    test_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear"),
    )
    model.fit(train_vectors, train_labels)
    train_probs = model.predict_proba(train_vectors)[:, 1]
    test_probs = model.predict_proba(test_vectors)[:, 1]
    return train_probs, test_probs


def _score_metrics(
    train_labels: list[int],
    train_probs: np.ndarray,
    test_labels: list[int],
    test_probs: np.ndarray,
) -> dict[str, float]:
    y_train = np.asarray(train_labels, dtype=np.int32)
    y_test = np.asarray(test_labels, dtype=np.int32)
    train_opt_threshold, train_opt_bal_acc = _best_threshold(y_train, train_probs)
    test_opt_threshold, test_opt_bal_acc = _best_threshold(y_test, test_probs)
    preds_050 = (test_probs >= 0.5).astype(int)
    bal_acc_050 = float(balanced_accuracy_score(y_test, preds_050))
    preds_train_opt = (test_probs >= train_opt_threshold).astype(int)
    bal_acc_train_opt = float(balanced_accuracy_score(y_test, preds_train_opt))
    auc = float(roc_auc_score(y_test, test_probs))
    pos_mask = y_test == 1
    neg_mask = y_test == 0
    return {
        "auroc": round(auc, 4),
        "balanced_accuracy_050": round(bal_acc_050, 4),
        "balanced_accuracy_train_opt": round(bal_acc_train_opt, 4),
        "balanced_accuracy_test_opt": round(test_opt_bal_acc, 4),
        "train_opt_threshold": round(float(train_opt_threshold), 4),
        "test_opt_threshold": round(float(test_opt_threshold), 4),
        "train_opt_bal_acc": round(float(train_opt_bal_acc), 4),
        "test_positive_mean_score": round(float(np.mean(test_probs[pos_mask])), 4),
        "test_negative_mean_score": round(float(np.mean(test_probs[neg_mask])), 4),
    }


def _aggregate_metric(rows: list[dict[str, float]], key: str) -> float:
    return round(float(np.mean([row[key] for row in rows])), 4)


def _pair_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]}__vs__{pair[1]}"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    allpairs_summary = json.loads(ALLPAIRS_SUMMARY_PATH.read_text(encoding="utf-8"))
    capture_artifact_id = str(allpairs_summary["capture_artifact_id"])
    manifest = _artifact_manifest(capture_artifact_id)
    feature_ref = manifest["storage_refs"]["features"]["prompt_eos_residual"]
    feature_payload = _load_capture_feature(feature_ref)
    records = _load_records()

    calibration_results: dict[str, Any] = {}
    source_holdout_results: dict[str, Any] = {}

    for pair in PAIR_ORDER:
        negative_label, positive_label = pair
        pair_key = _pair_label(pair)
        best_layer = str(allpairs_summary["pair_results"][pair_key]["best_layer"])
        layer_map = feature_payload["layers"][best_layer]
        pair_records = [record for record in records if record["prime_condition"] in pair]

        calibration_cells: list[dict[str, Any]] = []
        for train_lang in LANGUAGE_ORDER:
            train_rows = [record for record in pair_records if record["language_code"] == train_lang]
            train_texts = [record["prompt_text"] for record in train_rows]
            train_labels = [_label_for(record, positive_label) for record in train_rows]
            x_train = np.stack([_vector_for(layer_map[record["key"]]["values"]) for record in train_rows], axis=0)
            for test_lang in LANGUAGE_ORDER:
                if train_lang == test_lang:
                    continue
                test_rows = [record for record in pair_records if record["language_code"] == test_lang]
                test_texts = [record["prompt_text"] for record in test_rows]
                test_labels = [_label_for(record, positive_label) for record in test_rows]
                x_test = np.stack([_vector_for(layer_map[record["key"]]["values"]) for record in test_rows], axis=0)

                probe_train_probs, probe_test_probs = _fit_probe_probs(x_train, train_labels, x_test)
                text_train_probs, text_test_probs = _fit_text_probs(train_texts, train_labels, test_texts)

                calibration_cells.append(
                    {
                        "train_language": train_lang,
                        "test_language": test_lang,
                        "probe": _score_metrics(train_labels, probe_train_probs, test_labels, probe_test_probs),
                        "text": _score_metrics(train_labels, text_train_probs, test_labels, text_test_probs),
                    }
                )

        calibration_results[pair_key] = {
            "best_layer": best_layer,
            "cells": calibration_cells,
            "mean_probe_auroc": _aggregate_metric([cell["probe"] for cell in calibration_cells], "auroc"),
            "mean_probe_balanced_accuracy_050": _aggregate_metric(
                [cell["probe"] for cell in calibration_cells], "balanced_accuracy_050"
            ),
            "mean_probe_balanced_accuracy_train_opt": _aggregate_metric(
                [cell["probe"] for cell in calibration_cells], "balanced_accuracy_train_opt"
            ),
            "mean_probe_balanced_accuracy_test_opt": _aggregate_metric(
                [cell["probe"] for cell in calibration_cells], "balanced_accuracy_test_opt"
            ),
            "mean_text_balanced_accuracy_050": _aggregate_metric(
                [cell["text"] for cell in calibration_cells], "balanced_accuracy_050"
            ),
        }

        source_cells: list[dict[str, Any]] = []
        for heldout_family in SOURCE_FAMILY_ORDER:
            for train_lang in LANGUAGE_ORDER:
                train_rows = [
                    record
                    for record in pair_records
                    if record["source_family"] != heldout_family and record["language_code"] == train_lang
                ]
                train_texts = [record["prompt_text"] for record in train_rows]
                train_labels = [_label_for(record, positive_label) for record in train_rows]
                x_train = np.stack([_vector_for(layer_map[record["key"]]["values"]) for record in train_rows], axis=0)
                for test_lang in LANGUAGE_ORDER:
                    if train_lang == test_lang:
                        continue
                    test_rows = [
                        record
                        for record in pair_records
                        if record["source_family"] == heldout_family and record["language_code"] == test_lang
                    ]
                    test_texts = [record["prompt_text"] for record in test_rows]
                    test_labels = [_label_for(record, positive_label) for record in test_rows]
                    x_test = np.stack([_vector_for(layer_map[record["key"]]["values"]) for record in test_rows], axis=0)

                    probe_train_probs, probe_test_probs = _fit_probe_probs(x_train, train_labels, x_test)
                    text_train_probs, text_test_probs = _fit_text_probs(train_texts, train_labels, test_texts)

                    source_cells.append(
                        {
                            "heldout_source_family": heldout_family,
                            "train_language": train_lang,
                            "test_language": test_lang,
                            "probe": _score_metrics(train_labels, probe_train_probs, test_labels, probe_test_probs),
                            "text": _score_metrics(train_labels, text_train_probs, test_labels, text_test_probs),
                        }
                    )

        source_holdout_results[pair_key] = {
            "best_layer": best_layer,
            "cells": source_cells,
            "mean_probe_auroc": _aggregate_metric([cell["probe"] for cell in source_cells], "auroc"),
            "mean_probe_balanced_accuracy_050": _aggregate_metric(
                [cell["probe"] for cell in source_cells], "balanced_accuracy_050"
            ),
            "mean_text_auroc": _aggregate_metric([cell["text"] for cell in source_cells], "auroc"),
            "mean_text_balanced_accuracy_050": _aggregate_metric(
                [cell["text"] for cell in source_cells], "balanced_accuracy_050"
            ),
            "mean_probe_minus_text_auroc": round(
                _aggregate_metric([cell["probe"] for cell in source_cells], "auroc")
                - _aggregate_metric([cell["text"] for cell in source_cells], "auroc"),
                4,
            ),
            "mean_probe_minus_text_balanced_accuracy_050": round(
                _aggregate_metric([cell["probe"] for cell in source_cells], "balanced_accuracy_050")
                - _aggregate_metric([cell["text"] for cell in source_cells], "balanced_accuracy_050"),
                4,
            ),
        }

    calibration_ranked = sorted(
        calibration_results.items(),
        key=lambda item: (
            item[1]["mean_probe_balanced_accuracy_test_opt"] - item[1]["mean_probe_balanced_accuracy_050"],
            item[1]["mean_probe_auroc"],
        ),
        reverse=True,
    )
    source_ranked = sorted(
        source_holdout_results.items(),
        key=lambda item: (
            item[1]["mean_probe_minus_text_balanced_accuracy_050"],
            item[1]["mean_probe_minus_text_auroc"],
        ),
        reverse=True,
    )

    summary = {
        "capture_artifact_id": capture_artifact_id,
        "calibration": {
            "pair_results": calibration_results,
            "ranked_pairs_by_threshold_rescue": [pair_key for pair_key, _ in calibration_ranked],
        },
        "source_family_cross_language_holdout": {
            "pair_results": source_holdout_results,
            "ranked_pairs_by_probe_minus_text_balanced_accuracy": [pair_key for pair_key, _ in source_ranked],
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        "# Experiment 02 Cross-Language Prompt Probe Split Diagnostics",
        "",
        f"- capture artifact: `{capture_artifact_id}`",
        "",
        "## Calibration Summary",
    ]
    for pair_key, payload in calibration_ranked:
        rescue = round(payload["mean_probe_balanced_accuracy_test_opt"] - payload["mean_probe_balanced_accuracy_050"], 4)
        report_lines.append(
            f"- `{pair_key}`: probe AUROC `{payload['mean_probe_auroc']}`, "
            f"probe bal acc @0.5 `{payload['mean_probe_balanced_accuracy_050']}`, "
            f"probe bal acc @train-opt `{payload['mean_probe_balanced_accuracy_train_opt']}`, "
            f"probe bal acc @test-opt `{payload['mean_probe_balanced_accuracy_test_opt']}`, "
            f"text bal acc @0.5 `{payload['mean_text_balanced_accuracy_050']}`, "
            f"oracle rescue `{rescue}`"
        )

    report_lines.extend(["", "## Source-Family + Cross-Language Holdout Summary"])
    for pair_key, payload in source_ranked:
        report_lines.append(
            f"- `{pair_key}`: text AUROC `{payload['mean_text_auroc']}`, "
            f"text bal acc `{payload['mean_text_balanced_accuracy_050']}`, "
            f"probe AUROC `{payload['mean_probe_auroc']}`, "
            f"probe bal acc `{payload['mean_probe_balanced_accuracy_050']}`, "
            f"delta AUROC `{payload['mean_probe_minus_text_auroc']}`, "
            f"delta bal acc `{payload['mean_probe_minus_text_balanced_accuracy_050']}`"
        )

    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
