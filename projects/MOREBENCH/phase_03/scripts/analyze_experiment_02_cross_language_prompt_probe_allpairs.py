from __future__ import annotations

import argparse
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
REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_cross_language_prompt_probe_allpairs")
REPORT_PATH = REPORT_DIR / "report.md"
SUMMARY_PATH = REPORT_DIR / "summary.json"

LANGUAGE_ORDER = ("en", "es", "zh")
PRIME_ORDER = allpairs.TARGET_PRIMES
PAIR_ORDER = list(combinations(PRIME_ORDER, 2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-artifact-id", required=True)
    parser.add_argument("--random-control-permutations", type=int, default=64)
    return parser.parse_args()


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
                "prompt_text": prompt_text,
                "is_generic_control": bool(labels.get("is_generic_control", False)),
            }
        )
    return records


def _attach_prompt_token_counts(records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    records_by_key = {record["key"]: record for record in records}
    for row in manifest.get("metadata", {}).get("example_metadata", []):
        if not isinstance(row, dict):
            continue
        key = str(row.get("example_key") or "")
        if key in records_by_key:
            records_by_key[key]["prompt_token_count"] = int(row.get("prompt_token_count") or 0)


def _vector_for(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2:
        return np.asarray(array[-1], dtype=np.float32)
    return np.asarray(array, dtype=np.float32)


def _rows_by_lang(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        lang: [record for record in records if record["language_code"] == lang]
        for lang in LANGUAGE_ORDER
    }


def _label_for(record: dict[str, Any], positive_label: str) -> int:
    return 1 if record["prime_condition"] == positive_label else 0


def _fit_text_metrics(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
) -> tuple[float, float]:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    model = LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear")
    model.fit(x_train, train_labels)
    probs = model.predict_proba(x_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return float(roc_auc_score(test_labels, probs)), float(balanced_accuracy_score(test_labels, preds))


def _fit_probe_metrics(
    train_vectors: np.ndarray,
    train_labels: list[int],
    test_vectors: np.ndarray,
    test_labels: list[int],
) -> tuple[float, float]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear"),
    )
    model.fit(train_vectors, train_labels)
    probs = model.predict_proba(test_vectors)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return float(roc_auc_score(test_labels, probs)), float(balanced_accuracy_score(test_labels, preds))


def _fit_direction(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=4000,
            random_state=42,
            solver="liblinear",
        ),
    )
    model.fit(matrix, labels)
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    coef = np.asarray(classifier.coef_[0], dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(scale == 0.0, 1.0, scale)
    direction = coef / scale
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        return direction.astype(np.float32)
    return (direction / norm).astype(np.float32)


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    left_norm = float(np.linalg.norm(left64))
    right_norm = float(np.linalg.norm(right64))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return round(float(np.dot(left64, right64) / (left_norm * right_norm)), 4)


def _text_matrices(
    pair_records: list[dict[str, Any]],
    positive_label: str,
) -> tuple[dict[str, dict[str, float]], float, dict[str, dict[str, float]], float]:
    rows_by_lang = _rows_by_lang(pair_records)
    auc_matrix: dict[str, dict[str, float]] = {}
    balanced_accuracy_matrix: dict[str, dict[str, float]] = {}
    cross_aucs: list[float] = []
    cross_balanced_accuracies: list[float] = []
    for train_lang in LANGUAGE_ORDER:
        train_rows = rows_by_lang[train_lang]
        train_texts = [record["prompt_text"] for record in train_rows]
        train_labels = [_label_for(record, positive_label) for record in train_rows]
        auc_matrix[train_lang] = {}
        balanced_accuracy_matrix[train_lang] = {}
        for test_lang in LANGUAGE_ORDER:
            test_rows = rows_by_lang[test_lang]
            test_texts = [record["prompt_text"] for record in test_rows]
            test_labels = [_label_for(record, positive_label) for record in test_rows]
            auc, balanced_accuracy = _fit_text_metrics(train_texts, train_labels, test_texts, test_labels)
            auc_matrix[train_lang][test_lang] = round(auc, 4)
            balanced_accuracy_matrix[train_lang][test_lang] = round(balanced_accuracy, 4)
            if train_lang != test_lang:
                cross_aucs.append(auc)
                cross_balanced_accuracies.append(balanced_accuracy)
    return (
        auc_matrix,
        round(float(np.mean(cross_aucs)), 4),
        balanced_accuracy_matrix,
        round(float(np.mean(cross_balanced_accuracies)), 4),
    )


def _probe_matrices(
    feature_payload: dict[str, Any],
    pair_records: list[dict[str, Any]],
    positive_label: str,
) -> tuple[
    dict[str, dict[str, dict[str, float]]],
    dict[str, float],
    dict[str, dict[str, dict[str, float]]],
    dict[str, float],
    dict[str, np.ndarray],
]:
    rows_by_lang = _rows_by_lang(pair_records)
    auc_matrices_by_layer: dict[str, dict[str, dict[str, float]]] = {}
    mean_cross_auc_by_layer: dict[str, float] = {}
    balanced_accuracy_matrices_by_layer: dict[str, dict[str, dict[str, float]]] = {}
    mean_cross_balanced_accuracy_by_layer: dict[str, float] = {}
    pair_directions: dict[str, np.ndarray] = {}
    for layer in sorted(int(layer_str) for layer_str in feature_payload["layers"]):
        layer_map = feature_payload["layers"][str(layer)]
        auc_matrix: dict[str, dict[str, float]] = {}
        balanced_accuracy_matrix: dict[str, dict[str, float]] = {}
        cross_aucs: list[float] = []
        cross_balanced_accuracies: list[float] = []
        full_matrix = np.stack([_vector_for(layer_map[record["key"]]["values"]) for record in pair_records], axis=0)
        full_labels = np.asarray([_label_for(record, positive_label) for record in pair_records], dtype=np.int32)
        pair_directions[str(layer)] = _fit_direction(full_matrix, full_labels)
        for train_lang in LANGUAGE_ORDER:
            train_rows = rows_by_lang[train_lang]
            x_train = np.stack([_vector_for(layer_map[record["key"]]["values"]) for record in train_rows], axis=0)
            y_train = [_label_for(record, positive_label) for record in train_rows]
            auc_matrix[train_lang] = {}
            balanced_accuracy_matrix[train_lang] = {}
            for test_lang in LANGUAGE_ORDER:
                test_rows = rows_by_lang[test_lang]
                x_test = np.stack([_vector_for(layer_map[record["key"]]["values"]) for record in test_rows], axis=0)
                y_test = [_label_for(record, positive_label) for record in test_rows]
                auc, balanced_accuracy = _fit_probe_metrics(x_train, y_train, x_test, y_test)
                auc_matrix[train_lang][test_lang] = round(auc, 4)
                balanced_accuracy_matrix[train_lang][test_lang] = round(balanced_accuracy, 4)
                if train_lang != test_lang:
                    cross_aucs.append(auc)
                    cross_balanced_accuracies.append(balanced_accuracy)
        auc_matrices_by_layer[str(layer)] = auc_matrix
        mean_cross_auc_by_layer[str(layer)] = round(float(np.mean(cross_aucs)), 4)
        balanced_accuracy_matrices_by_layer[str(layer)] = balanced_accuracy_matrix
        mean_cross_balanced_accuracy_by_layer[str(layer)] = round(float(np.mean(cross_balanced_accuracies)), 4)
    return (
        auc_matrices_by_layer,
        mean_cross_auc_by_layer,
        balanced_accuracy_matrices_by_layer,
        mean_cross_balanced_accuracy_by_layer,
        pair_directions,
    )


def _random_label_control(
    feature_payload: dict[str, Any],
    pair_records: list[dict[str, Any]],
    positive_label: str,
    *,
    layer: str,
    permutations: int,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows_by_lang = _rows_by_lang(pair_records)
    layer_map = feature_payload["layers"][layer]
    mean_values: list[float] = []
    for _ in range(permutations):
        pair_scores: list[float] = []
        for train_lang in LANGUAGE_ORDER:
            train_rows = rows_by_lang[train_lang]
            x_train = np.stack([_vector_for(layer_map[record["key"]]["values"]) for record in train_rows], axis=0)
            y_train_true = np.asarray([_label_for(record, positive_label) for record in train_rows], dtype=np.int64)
            y_train = rng.permutation(y_train_true).tolist()
            for test_lang in LANGUAGE_ORDER:
                if train_lang == test_lang:
                    continue
                test_rows = rows_by_lang[test_lang]
                x_test = np.stack([_vector_for(layer_map[record["key"]]["values"]) for record in test_rows], axis=0)
                y_test = [_label_for(record, positive_label) for record in test_rows]
                auc, _ = _fit_probe_metrics(x_train, y_train, x_test, y_test)
                pair_scores.append(float(auc))
        mean_values.append(float(np.mean(pair_scores)))
    values = np.asarray(mean_values, dtype=np.float32)
    return {
        "layer": layer,
        "permutations": permutations,
        "mean_cross_language_auroc_mean": round(float(np.mean(values)), 4),
        "mean_cross_language_auroc_p95": round(float(np.quantile(values, 0.95)), 4),
        "share_mean_cross_language_auroc_ge_080": round(float(np.mean(values >= 0.80)), 4),
    }


def _best_layer(mean_cross_by_layer: dict[str, float], prompt_text_mean: float) -> tuple[str, float]:
    best_layer = max(
        mean_cross_by_layer,
        key=lambda layer: (mean_cross_by_layer[layer] - prompt_text_mean, mean_cross_by_layer[layer], -int(layer)),
    )
    return best_layer, round(mean_cross_by_layer[best_layer] - prompt_text_mean, 4)


def _pair_label(pair: tuple[str, str]) -> str:
    return f"{pair[0]}__vs__{pair[1]}"


def main() -> None:
    args = _parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = _artifact_manifest(args.capture_artifact_id)
    feature_ref = manifest["storage_refs"]["features"]["prompt_eos_residual"]
    feature_payload = _load_capture_feature(feature_ref)
    records = _load_records()
    _attach_prompt_token_counts(records, manifest)

    pair_results: dict[str, Any] = {}
    l32_directions: dict[str, np.ndarray] = {}
    for pair in PAIR_ORDER:
        negative_label, positive_label = pair
        pair_records = [record for record in records if record["prime_condition"] in pair]
        (
            prompt_text_matrix,
            mean_prompt_text,
            prompt_text_balanced_accuracy_matrix,
            mean_prompt_text_balanced_accuracy,
        ) = _text_matrices(pair_records, positive_label)
        (
            probe_matrices_by_layer,
            mean_probe_by_layer,
            probe_balanced_accuracy_matrices_by_layer,
            mean_probe_balanced_accuracy_by_layer,
            directions,
        ) = _probe_matrices(feature_payload, pair_records, positive_label)
        best_layer, best_delta = _best_layer(mean_probe_by_layer, mean_prompt_text)
        random_control = _random_label_control(
            feature_payload,
            pair_records,
            positive_label,
            layer=best_layer,
            permutations=args.random_control_permutations,
        )
        cross_script = {
            "en->zh": probe_matrices_by_layer[best_layer]["en"]["zh"],
            "zh->en": probe_matrices_by_layer[best_layer]["zh"]["en"],
            "es->zh": probe_matrices_by_layer[best_layer]["es"]["zh"],
            "zh->es": probe_matrices_by_layer[best_layer]["zh"]["es"],
        }
        emergence = {
            "layer0_le_060": bool(mean_probe_by_layer["0"] <= 0.60),
            "layer4_ge_layer0": bool(mean_probe_by_layer["4"] >= mean_probe_by_layer["0"]),
            "layer8_ge_layer4": bool(mean_probe_by_layer["8"] >= mean_probe_by_layer["4"]),
            "layer16_ge_layer8": bool(mean_probe_by_layer["16"] >= mean_probe_by_layer["8"]),
        }
        pair_key = _pair_label(pair)
        pair_results[pair_key] = {
            "negative_label": negative_label,
            "positive_label": positive_label,
            "row_count": len(pair_records),
            "prompt_text_matrix": prompt_text_matrix,
            "mean_cross_language_prompt_text_auroc": mean_prompt_text,
            "prompt_text_balanced_accuracy_matrix": prompt_text_balanced_accuracy_matrix,
            "mean_cross_language_prompt_text_balanced_accuracy": mean_prompt_text_balanced_accuracy,
            "mean_cross_language_probe_auroc_by_layer": mean_probe_by_layer,
            "mean_cross_language_probe_balanced_accuracy_by_layer": mean_probe_balanced_accuracy_by_layer,
            "best_layer": best_layer,
            "best_layer_mean_cross_language_probe_auroc": mean_probe_by_layer[best_layer],
            "best_layer_mean_cross_language_probe_balanced_accuracy": mean_probe_balanced_accuracy_by_layer[best_layer],
            "best_layer_delta_vs_prompt_text": best_delta,
            "best_layer_delta_vs_prompt_text_balanced_accuracy": round(
                mean_probe_balanced_accuracy_by_layer[best_layer] - mean_prompt_text_balanced_accuracy,
                4,
            ),
            "best_layer_cross_script_pairs": cross_script,
            "best_layer_probe_balanced_accuracy_matrix": probe_balanced_accuracy_matrices_by_layer[best_layer],
            "random_label_control": random_control,
            "emergence_shape": emergence,
        }
        l32_directions[pair_key] = directions["32"]

    cosine_matrix: dict[str, dict[str, float | None]] = {}
    for left_key, left_direction in l32_directions.items():
        cosine_matrix[left_key] = {}
        for right_key, right_direction in l32_directions.items():
            cosine_matrix[left_key][right_key] = _cosine(left_direction, right_direction)

    strongest_pairs = sorted(
        pair_results.items(),
        key=lambda item: (
            item[1]["best_layer_delta_vs_prompt_text"],
            item[1]["best_layer_mean_cross_language_probe_auroc"],
        ),
        reverse=True,
    )

    summary = {
        "capture_artifact_id": args.capture_artifact_id,
        "pair_count": len(pair_results),
        "pair_order": [_pair_label(pair) for pair in PAIR_ORDER],
        "pair_results": pair_results,
        "l32_pair_direction_cosines": cosine_matrix,
        "strongest_pairs": [pair_key for pair_key, _ in strongest_pairs],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        "# Experiment 02 Cross-Language Prompt Probe All-Pairs",
        "",
        f"- capture artifact: `{args.capture_artifact_id}`",
        f"- pair count: `{len(pair_results)}`",
        "",
        "## Pair Summary",
    ]
    for pair_key, payload in strongest_pairs:
        report_lines.append(
            f"- `{pair_key}`: text `{payload['mean_cross_language_prompt_text_auroc']}`, "
            f"text bal acc `{payload['mean_cross_language_prompt_text_balanced_accuracy']}`, "
            f"best layer `{payload['best_layer']}`, probe `{payload['best_layer_mean_cross_language_probe_auroc']}`, "
            f"probe bal acc `{payload['best_layer_mean_cross_language_probe_balanced_accuracy']}`, "
            f"delta `{payload['best_layer_delta_vs_prompt_text']}`, "
            f"bal acc delta `{payload['best_layer_delta_vs_prompt_text_balanced_accuracy']}`, "
            f"random p95 `{payload['random_label_control']['mean_cross_language_auroc_p95']}`"
        )
    report_lines.extend(["", "## L32 Pair-Direction Cosines"])
    for left_key in summary["pair_order"]:
        row = cosine_matrix[left_key]
        compact = ", ".join(f"{right_key}={row[right_key]}" for right_key in summary["pair_order"])
        report_lines.append(f"- `{left_key}`: {compact}")
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
