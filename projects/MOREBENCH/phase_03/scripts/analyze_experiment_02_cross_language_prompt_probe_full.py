from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import modal
import numpy as np
from safetensors.numpy import load as load_safetensors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipelines_v2.storage.features import decode_feature_payload


CATALOG_ROOT = Path("artifacts") / "morebench_phase03_experiment02_cross_language_prompt_probe_full_catalog"
TRANSFORM_RESULT_PATH = (
    Path("artifacts")
    / "morebench_phase03_experiment02_cross_language_prompt_probe_full"
    / "transform_33d92c1d07d0_339727c1"
    / "result.json"
)
REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_cross_language_prompt_probe_full")
REPORT_PATH = REPORT_DIR / "report.md"
SUMMARY_PATH = REPORT_DIR / "summary.json"

LANGUAGE_ORDER = ("en", "es", "zh")
TRACKED_PAIRS = (("en", "zh"), ("zh", "en"), ("es", "zh"), ("zh", "es"))
THEORY_TERM_AUDIT = {"phronesis", "temperance", "virtue", "categorical", "imperative", "kantian"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-artifact-id", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=256)
    parser.add_argument("--random-control-permutations", type=int, default=256)
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
    payload = json.loads(TRANSFORM_RESULT_PATH.read_text(encoding="utf-8"))
    dataset = payload["dataset"]
    records: list[dict[str, Any]] = []
    for example in dataset["examples"]:
        labels = dict(example.get("labels", {}))
        prompt_messages = list(example.get("prompt", []))
        prompt_text = "\n\n".join(str(message.get("content") or "") for message in prompt_messages)
        records.append(
            {
                "key": str(example["key"]),
                "group_id": str(labels["group_id"]),
                "prime_condition": str(labels["prime_condition"]),
                "language_code": str(labels["language_code"]),
                "language_name": str(labels["language_name"]),
                "source_family": str(labels.get("source_family") or ""),
                "context": str(labels.get("context") or ""),
                "prompt_text": prompt_text,
            }
        )
    return records


def _attach_prompt_token_counts(records: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    records_by_key = {record["key"]: record for record in records}
    metadata_rows = manifest.get("metadata", {}).get("example_metadata", [])
    for row in metadata_rows:
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


def _label_for(record: dict[str, Any]) -> int:
    return 1 if record["prime_condition"] == "deontology" else 0


def _fit_text_auc(train_texts: list[str], train_labels: list[int], test_texts: list[str], test_labels: list[int]) -> float:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    model = LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear")
    model.fit(x_train, train_labels)
    probs = model.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(test_labels, probs))


def _fit_probe_auc(train_vectors: np.ndarray, train_labels: list[int], test_vectors: np.ndarray, test_labels: list[int]) -> float:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear"),
    )
    model.fit(train_vectors, train_labels)
    probs = model.predict_proba(test_vectors)[:, 1]
    return float(roc_auc_score(test_labels, probs))


def _fit_scalar_auc(train_values: list[float], train_labels: list[int], test_values: list[float], test_labels: list[int]) -> float:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear"),
    )
    x_train = np.asarray(train_values, dtype=np.float32).reshape(-1, 1)
    x_test = np.asarray(test_values, dtype=np.float32).reshape(-1, 1)
    model.fit(x_train, train_labels)
    probs = model.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(test_labels, probs))


def _rows_by_lang(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        lang: [record for record in records if record["language_code"] == lang]
        for lang in LANGUAGE_ORDER
    }


def _grouped_rows_by_lang(records: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {lang: defaultdict(list) for lang in LANGUAGE_ORDER}
    for record in records:
        grouped[record["language_code"]][record["group_id"]].append(record)
    return grouped


def _text_matrix_from_rows(rows_by_lang: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, dict[str, float]], float]:
    matrix: dict[str, dict[str, float]] = {}
    cross_values: list[float] = []
    for train_lang in LANGUAGE_ORDER:
        train_rows = rows_by_lang[train_lang]
        train_texts = [record["prompt_text"] for record in train_rows]
        train_labels = [_label_for(record) for record in train_rows]
        matrix[train_lang] = {}
        for test_lang in LANGUAGE_ORDER:
            test_rows = rows_by_lang[test_lang]
            test_texts = [record["prompt_text"] for record in test_rows]
            test_labels = [_label_for(record) for record in test_rows]
            auc = float(_fit_text_auc(train_texts, train_labels, test_texts, test_labels))
            matrix[train_lang][test_lang] = round(auc, 4)
            if train_lang != test_lang:
                cross_values.append(auc)
    return matrix, round(float(np.mean(cross_values)), 4)


def _length_matrix_from_rows(rows_by_lang: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, dict[str, float]], float]:
    matrix: dict[str, dict[str, float]] = {}
    cross_values: list[float] = []
    for train_lang in LANGUAGE_ORDER:
        train_rows = rows_by_lang[train_lang]
        train_values = [float(record["prompt_token_count"]) for record in train_rows]
        train_labels = [_label_for(record) for record in train_rows]
        matrix[train_lang] = {}
        for test_lang in LANGUAGE_ORDER:
            test_rows = rows_by_lang[test_lang]
            test_values = [float(record["prompt_token_count"]) for record in test_rows]
            test_labels = [_label_for(record) for record in test_rows]
            auc = float(_fit_scalar_auc(train_values, train_labels, test_values, test_labels))
            matrix[train_lang][test_lang] = round(auc, 4)
            if train_lang != test_lang:
                cross_values.append(auc)
    return matrix, round(float(np.mean(cross_values)), 4)


def _probe_matrix_for_layer(
    rows_by_lang: dict[str, list[dict[str, Any]]],
    feature_payload: dict[str, Any],
    *,
    layer: str,
) -> tuple[dict[str, dict[str, float]], float]:
    layer_map = feature_payload["layers"][layer]
    matrix: dict[str, dict[str, float]] = {}
    cross_values: list[float] = []
    for train_lang in LANGUAGE_ORDER:
        train_rows = rows_by_lang[train_lang]
        x_train = np.stack([_vector_for(layer_map[record["key"]]["values"]) for record in train_rows], axis=0)
        y_train = [_label_for(record) for record in train_rows]
        matrix[train_lang] = {}
        for test_lang in LANGUAGE_ORDER:
            test_rows = rows_by_lang[test_lang]
            x_test = np.stack([_vector_for(layer_map[record["key"]]["values"]) for record in test_rows], axis=0)
            y_test = [_label_for(record) for record in test_rows]
            auc = float(_fit_probe_auc(x_train, y_train, x_test, y_test))
            matrix[train_lang][test_lang] = round(auc, 4)
            if train_lang != test_lang:
                cross_values.append(auc)
    return matrix, round(float(np.mean(cross_values)), 4)


def _probe_matrices(
    feature_payload: dict[str, Any],
    records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, float]]:
    rows_by_lang = _rows_by_lang(records)
    matrices_by_layer: dict[str, dict[str, dict[str, float]]] = {}
    mean_cross_by_layer: dict[str, float] = {}
    for layer in sorted(int(layer_str) for layer_str in feature_payload["layers"]):
        matrix, mean_cross = _probe_matrix_for_layer(rows_by_lang, feature_payload, layer=str(layer))
        matrices_by_layer[str(layer)] = matrix
        mean_cross_by_layer[str(layer)] = mean_cross
    return matrices_by_layer, mean_cross_by_layer


def _cross_script_pairs_by_layer(probe_matrices_by_layer: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    return {
        layer: {
            f"{src}->{dst}": probe_matrices_by_layer[layer][src][dst]
            for src, dst in TRACKED_PAIRS
        }
        for layer in sorted(probe_matrices_by_layer, key=lambda value: int(value))
    }


def _non_english_prompt_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    language_details: dict[str, list[dict[str, Any]]] = {"es": [], "zh": []}
    token_counter_zh: dict[str, int] = {}
    for record in records:
        lang = record["language_code"]
        if lang not in {"es", "zh"}:
            continue
        word_tokens = re.findall(r"[^\W_]+", record["prompt_text"], flags=re.UNICODE)
        lowered_ascii = [
            token.lower()
            for token in word_tokens
            if token.isascii() and any(char.isalpha() for char in token)
        ]
        english_theory_terms = [token for token in lowered_ascii if token in THEORY_TERM_AUDIT]
        detail = {
            "example_key": record["key"],
            "group_id": record["group_id"],
            "prime_condition": record["prime_condition"],
            "ascii_tokens": lowered_ascii,
            "english_theory_terms": english_theory_terms,
        }
        language_details[lang].append(detail)
        if lang == "zh":
            for token in lowered_ascii:
                token_counter_zh[token] = token_counter_zh.get(token, 0) + 1
    return {
        "es_row_count": len(language_details["es"]),
        "zh_row_count": len(language_details["zh"]),
        "es_rows_with_english_theory_terms": sum(1 for item in language_details["es"] if item["english_theory_terms"]),
        "zh_rows_with_english_theory_terms": sum(1 for item in language_details["zh"] if item["english_theory_terms"]),
        "zh_rows_with_ascii_tokens": sum(1 for item in language_details["zh"] if item["ascii_tokens"]),
        "zh_top_ascii_tokens": sorted(token_counter_zh.items(), key=lambda item: (-item[1], item[0]))[:20],
        "details": language_details,
    }


def _random_label_control(
    feature_payload: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    layer: str,
    seed: int = 0,
    permutations: int = 256,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    rows_by_lang = _rows_by_lang(records)
    layer_map = feature_payload["layers"][layer]
    ordered_pair_values: dict[str, list[float]] = {
        f"{train_lang}->{test_lang}": []
        for train_lang in LANGUAGE_ORDER
        for test_lang in LANGUAGE_ORDER
        if train_lang != test_lang
    }
    mean_values: list[float] = []
    for _ in range(permutations):
        pair_scores: list[float] = []
        for train_lang in LANGUAGE_ORDER:
            train_rows = rows_by_lang[train_lang]
            x_train = np.stack([_vector_for(layer_map[record["key"]]["values"]) for record in train_rows], axis=0)
            y_train_true = np.asarray([_label_for(record) for record in train_rows], dtype=np.int64)
            y_train = rng.permutation(y_train_true).tolist()
            for test_lang in LANGUAGE_ORDER:
                if train_lang == test_lang:
                    continue
                test_rows = rows_by_lang[test_lang]
                x_test = np.stack([_vector_for(layer_map[record["key"]]["values"]) for record in test_rows], axis=0)
                y_test = [_label_for(record) for record in test_rows]
                auc = float(_fit_probe_auc(x_train, y_train, x_test, y_test))
                ordered_pair_values[f"{train_lang}->{test_lang}"].append(auc)
                pair_scores.append(auc)
        mean_values.append(float(np.mean(pair_scores)))
    summary_pairs = {
        pair: {
            "mean": round(float(np.mean(values)), 4),
            "p95": round(float(np.quantile(values, 0.95)), 4),
            "max": round(float(np.max(values)), 4),
        }
        for pair, values in ordered_pair_values.items()
    }
    mean_values_array = np.asarray(mean_values, dtype=np.float32)
    return {
        "layer": layer,
        "permutations": permutations,
        "mean_cross_language_auroc_mean": round(float(np.mean(mean_values_array)), 4),
        "mean_cross_language_auroc_p95": round(float(np.quantile(mean_values_array, 0.95)), 4),
        "mean_cross_language_auroc_max": round(float(np.max(mean_values_array)), 4),
        "share_mean_cross_language_auroc_ge_060": round(float(np.mean(mean_values_array >= 0.60)), 4),
        "share_mean_cross_language_auroc_ge_080": round(float(np.mean(mean_values_array >= 0.80)), 4),
        "ordered_pair_summary": summary_pairs,
    }


def _best_layer(mean_cross_by_layer: dict[str, float], prompt_text_mean: float) -> tuple[str, float]:
    best_layer = max(
        mean_cross_by_layer,
        key=lambda layer: (mean_cross_by_layer[layer] - prompt_text_mean, mean_cross_by_layer[layer], -int(layer)),
    )
    return best_layer, round(mean_cross_by_layer[best_layer] - prompt_text_mean, 4)


def _ci(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    return {
        "mean": round(float(np.mean(array)), 4),
        "lo": round(float(np.quantile(array, 0.025)), 4),
        "hi": round(float(np.quantile(array, 0.975)), 4),
    }


def _bootstrap_best_layer(
    records: list[dict[str, Any]],
    feature_payload: dict[str, Any],
    *,
    best_layer: str,
    resamples: int,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    group_ids = sorted({record["group_id"] for record in records})
    grouped = _grouped_rows_by_lang(records)
    prompt_text_means: list[float] = []
    probe_means: list[float] = []
    deltas: list[float] = []
    pair_values: dict[str, list[float]] = {f"{src}->{dst}": [] for src, dst in TRACKED_PAIRS}

    for _ in range(resamples):
        sampled_groups = rng.choice(group_ids, size=len(group_ids), replace=True).tolist()
        rows_by_lang = {
            lang: [
                row
                for group_id in sampled_groups
                for row in grouped[lang][group_id]
            ]
            for lang in LANGUAGE_ORDER
        }
        _, prompt_text_mean = _text_matrix_from_rows(rows_by_lang)
        probe_matrix, probe_mean = _probe_matrix_for_layer(rows_by_lang, feature_payload, layer=best_layer)
        prompt_text_means.append(prompt_text_mean)
        probe_means.append(probe_mean)
        deltas.append(probe_mean - prompt_text_mean)
        for src, dst in TRACKED_PAIRS:
            pair_values[f"{src}->{dst}"].append(float(probe_matrix[src][dst]))

    return {
        "resamples": resamples,
        "prompt_text_mean_cross_language_auroc": _ci(prompt_text_means),
        "best_layer_mean_cross_language_auroc": _ci(probe_means),
        "best_layer_delta_vs_prompt_text": _ci(deltas),
        "best_layer_cross_script_pairs": {
            pair: _ci(values)
            for pair, values in pair_values.items()
        },
    }


def _write_report(summary: dict[str, Any]) -> None:
    text_matrix = summary["prompt_text_matrix"]
    length_matrix = summary["prompt_length_matrix"]
    best_layer = summary["best_layer"]
    best_matrix = summary["probe_matrices_by_layer"][best_layer]
    cross_script_pairs = summary["cross_script_pairs_by_layer"]
    residue = summary["non_english_prompt_audit"]
    random_control = summary["random_label_control"]
    bootstrap = summary["group_bootstrap"]
    report = f"""# Experiment 02 Cross-Language Prompt Probe Full 30

Prompt-final residual probe on the translated `English / Spanish / Simplified Chinese` full-30 prompt set for `deontology` vs `virtue_ethics`.

## Run
- capture artifact: `{summary['capture_artifact_id']}`
- example count: `{summary['row_count']}`
- group count: `{summary['group_count']}`
- layers: `{', '.join(str(layer) for layer in summary['captured_layers'])}`

## Prompt Text Baseline
Raw prompt-text char-TF-IDF AUROC matrix:

| train -> test | `en` | `es` | `zh` |
| --- | ---: | ---: | ---: |
| `en` | `{text_matrix['en']['en']:.2f}` | `{text_matrix['en']['es']:.2f}` | `{text_matrix['en']['zh']:.2f}` |
| `es` | `{text_matrix['es']['en']:.2f}` | `{text_matrix['es']['es']:.2f}` | `{text_matrix['es']['zh']:.2f}` |
| `zh` | `{text_matrix['zh']['en']:.2f}` | `{text_matrix['zh']['es']:.2f}` | `{text_matrix['zh']['zh']:.2f}` |

- mean cross-language prompt-text AUROC: `{summary['mean_cross_language_prompt_text_auroc']:.4f}`
- grouped bootstrap 95% CI: `[{bootstrap['prompt_text_mean_cross_language_auroc']['lo']:.4f}, {bootstrap['prompt_text_mean_cross_language_auroc']['hi']:.4f}]`

Prompt-length-only baseline (`prompt_token_count`) cross-language AUROC matrix:

| train -> test | `en` | `es` | `zh` |
| --- | ---: | ---: | ---: |
| `en` | `{length_matrix['en']['en']:.2f}` | `{length_matrix['en']['es']:.2f}` | `{length_matrix['en']['zh']:.2f}` |
| `es` | `{length_matrix['es']['en']:.2f}` | `{length_matrix['es']['es']:.2f}` | `{length_matrix['es']['zh']:.2f}` |
| `zh` | `{length_matrix['zh']['en']:.2f}` | `{length_matrix['zh']['es']:.2f}` | `{length_matrix['zh']['zh']:.2f}` |

- mean cross-language prompt-length AUROC: `{summary['mean_cross_language_prompt_length_auroc']:.4f}`

## Prompt-Final Probe
Mean cross-language prompt-final probe AUROC by layer:

| layer | mean cross-language AUROC |
| --- | ---: |
""" + "\n".join(
        f"| `{layer}` | `{summary['mean_cross_language_probe_auroc_by_layer'][str(layer)]:.4f}` |"
        for layer in summary["captured_layers"]
    ) + f"""

Best layer by cross-language delta over prompt text:
- best layer: `{best_layer}`
- mean cross-language AUROC: `{summary['mean_cross_language_probe_auroc_by_layer'][best_layer]:.4f}`
- grouped bootstrap 95% CI: `[{bootstrap['best_layer_mean_cross_language_auroc']['lo']:.4f}, {bootstrap['best_layer_mean_cross_language_auroc']['hi']:.4f}]`
- delta vs prompt-text baseline: `{summary['best_layer_delta_vs_prompt_text']:.4f}`
- grouped bootstrap delta 95% CI: `[{bootstrap['best_layer_delta_vs_prompt_text']['lo']:.4f}, {bootstrap['best_layer_delta_vs_prompt_text']['hi']:.4f}]`

Best-layer cross-language probe matrix:

| train -> test | `en` | `es` | `zh` |
| --- | ---: | ---: | ---: |
| `en` | `{best_matrix['en']['en']:.2f}` | `{best_matrix['en']['es']:.2f}` | `{best_matrix['en']['zh']:.2f}` |
| `es` | `{best_matrix['es']['en']:.2f}` | `{best_matrix['es']['es']:.2f}` | `{best_matrix['es']['zh']:.2f}` |
| `zh` | `{best_matrix['zh']['en']:.2f}` | `{best_matrix['zh']['es']:.2f}` | `{best_matrix['zh']['zh']:.2f}` |

Cross-script ordered pairs by layer:

| layer | `en->zh` | `zh->en` | `es->zh` | `zh->es` |
| --- | ---: | ---: | ---: | ---: |
""" + "\n".join(
        f"| `{layer}` | `{cross_script_pairs[str(layer)]['en->zh']:.2f}` | `{cross_script_pairs[str(layer)]['zh->en']:.2f}` | `{cross_script_pairs[str(layer)]['es->zh']:.2f}` | `{cross_script_pairs[str(layer)]['zh->es']:.2f}` |"
        for layer in summary["captured_layers"]
    ) + f"""

Best-layer grouped bootstrap 95% CIs for cross-script ordered pairs:
- `en->zh`: `[{bootstrap['best_layer_cross_script_pairs']['en->zh']['lo']:.4f}, {bootstrap['best_layer_cross_script_pairs']['en->zh']['hi']:.4f}]`
- `zh->en`: `[{bootstrap['best_layer_cross_script_pairs']['zh->en']['lo']:.4f}, {bootstrap['best_layer_cross_script_pairs']['zh->en']['hi']:.4f}]`
- `es->zh`: `[{bootstrap['best_layer_cross_script_pairs']['es->zh']['lo']:.4f}, {bootstrap['best_layer_cross_script_pairs']['es->zh']['hi']:.4f}]`
- `zh->es`: `[{bootstrap['best_layer_cross_script_pairs']['zh->es']['lo']:.4f}, {bootstrap['best_layer_cross_script_pairs']['zh->es']['hi']:.4f}]`

## Red-Team Checks
- Spanish prompt rows with English theory-term residue: `{residue['es_rows_with_english_theory_terms']}/{residue['es_row_count']}`
- Chinese prompt rows with English theory-term residue: `{residue['zh_rows_with_english_theory_terms']}/{residue['zh_row_count']}`
- Chinese prompt rows with ASCII residue: `{residue['zh_rows_with_ascii_tokens']}/{residue['zh_row_count']}`
- Top ASCII residues in Chinese prompts: `{', '.join(f'{token} ({count})' for token, count in residue['zh_top_ascii_tokens'][:10])}`
- Random-label control at best layer:
  - mean cross-language AUROC under permutation: `{random_control['mean_cross_language_auroc_mean']:.4f}`
  - 95th percentile: `{random_control['mean_cross_language_auroc_p95']:.4f}`
  - max over `{random_control['permutations']}` permutations: `{random_control['mean_cross_language_auroc_max']:.4f}`
  - share of permutations with mean cross-language AUROC `>= 0.60`: `{random_control['share_mean_cross_language_auroc_ge_060']:.4f}`
  - share with `>= 0.80`: `{random_control['share_mean_cross_language_auroc_ge_080']:.4f}`

## Interpretation
- The scale-up target is not whether any single pair is high, but whether the prompt-final probe still opens clear room over the prompt-text baseline once we use all `30` dilemmas.
- The most important structural check is the emergence curve. A signal that starts near chance at early layers and rises through the stack is much harder to explain as a surface-text shortcut than a flat ceiling from `L0`.
- The cross-script ordered pairs are the strongest subtest because they break the easy English-character path.
- The grouped bootstrap is the main guardrail here. We should trust the full run only if the best-layer delta stays comfortably above zero and the cross-script pairs remain high.

## Recommendation
- {summary['recommendation']}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = _artifact_manifest(args.capture_artifact_id)
    feature_ref = manifest["storage_refs"]["features"]["prompt_eos_residual"]
    feature_payload = _load_capture_feature(feature_ref)
    records = _load_records()
    _attach_prompt_token_counts(records, manifest)

    rows_by_lang = _rows_by_lang(records)
    prompt_text_matrix, mean_prompt_text = _text_matrix_from_rows(rows_by_lang)
    prompt_length_matrix, mean_prompt_length = _length_matrix_from_rows(rows_by_lang)
    probe_matrices_by_layer, mean_probe_by_layer = _probe_matrices(feature_payload, records)
    best_layer, best_delta = _best_layer(mean_probe_by_layer, mean_prompt_text)
    cross_script_pairs = _cross_script_pairs_by_layer(probe_matrices_by_layer)
    prompt_audit = _non_english_prompt_audit(records)
    random_label_control = _random_label_control(
        feature_payload,
        records,
        layer=best_layer,
        permutations=args.random_control_permutations,
    )
    group_bootstrap = _bootstrap_best_layer(
        records,
        feature_payload,
        best_layer=best_layer,
        resamples=args.bootstrap_resamples,
    )

    best_mean = mean_probe_by_layer[best_layer]
    delta_ci = group_bootstrap["best_layer_delta_vs_prompt_text"]
    en_zh = probe_matrices_by_layer[best_layer]["en"]["zh"]
    zh_en = probe_matrices_by_layer[best_layer]["zh"]["en"]
    emergence_shape = (
        mean_probe_by_layer["0"] <= 0.60
        and mean_probe_by_layer["4"] >= mean_probe_by_layer["0"]
        and mean_probe_by_layer["8"] >= mean_probe_by_layer["4"]
        and mean_probe_by_layer["16"] >= mean_probe_by_layer["8"]
    )
    strong_reopening = (
        best_delta >= 0.25
        and delta_ci["lo"] > 0.0
        and emergence_shape
        and min(en_zh, zh_en) >= 0.80
    )
    moderate_reopening = (
        best_delta >= 0.15
        and delta_ci["lo"] > 0.0
        and best_mean >= 0.75
    )

    if strong_reopening:
        recommendation = (
            "The full 30-dilemma prompt-side run supports a strong representational reopening on deontology vs virtue_ethics. "
            "The next step is a targeted follow-up control set, not another broad search."
        )
    elif moderate_reopening:
        recommendation = (
            "The full 30-dilemma prompt-side run is suggestive enough to keep the line alive, but it still needs one more control pass "
            "before we should treat it as a defended Level 2 claim."
        )
    else:
        recommendation = (
            "The full 30-dilemma prompt-side run does not clear the reopening threshold cleanly enough. Treat this as useful evidence, "
            "not as a defended representational claim."
        )

    summary = {
        "capture_artifact_id": args.capture_artifact_id,
        "row_count": len(records),
        "group_count": len({record["group_id"] for record in records}),
        "captured_layers": [int(layer) for layer in sorted(int(layer) for layer in feature_payload["layers"])],
        "prompt_text_matrix": prompt_text_matrix,
        "mean_cross_language_prompt_text_auroc": mean_prompt_text,
        "prompt_length_matrix": prompt_length_matrix,
        "mean_cross_language_prompt_length_auroc": mean_prompt_length,
        "probe_matrices_by_layer": probe_matrices_by_layer,
        "mean_cross_language_probe_auroc_by_layer": mean_probe_by_layer,
        "cross_script_pairs_by_layer": cross_script_pairs,
        "best_layer": best_layer,
        "best_layer_delta_vs_prompt_text": best_delta,
        "non_english_prompt_audit": prompt_audit,
        "random_label_control": random_label_control,
        "group_bootstrap": group_bootstrap,
        "emergence_shape": {
            "layer0_le_060": bool(mean_probe_by_layer["0"] <= 0.60),
            "layer4_ge_layer0": bool(mean_probe_by_layer["4"] >= mean_probe_by_layer["0"]),
            "layer8_ge_layer4": bool(mean_probe_by_layer["8"] >= mean_probe_by_layer["4"]),
            "layer16_ge_layer8": bool(mean_probe_by_layer["16"] >= mean_probe_by_layer["8"]),
            "monotonic_rise_through_16": bool(emergence_shape),
        },
        "decision": {
            "strong_reopening": bool(strong_reopening),
            "moderate_reopening": bool(moderate_reopening),
            "best_layer_delta_ci_excludes_zero": bool(delta_ci["lo"] > 0.0),
            "best_layer_cross_script_en_zh_ge_080": bool(en_zh >= 0.80),
            "best_layer_cross_script_zh_en_ge_080": bool(zh_en >= 0.80),
            "random_label_share_ge_080_le_010": bool(random_label_control["share_mean_cross_language_auroc_ge_080"] <= 0.10),
        },
        "recommendation": recommendation,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(summary)


if __name__ == "__main__":
    main()
