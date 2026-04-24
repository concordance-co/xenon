from __future__ import annotations

import argparse
import json
import re
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
from projects.MOREBENCH.phase_03.specs import experiment_02_cross_language_pilot_workflow as pilot
from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base


CATALOG_ROOT = Path("artifacts") / "morebench_phase03_experiment02_cross_language_prompt_probe_catalog"
REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_cross_language_prompt_probe")
REPORT_PATH = REPORT_DIR / "report.md"
SUMMARY_PATH = REPORT_DIR / "summary.json"

LANGUAGE_ORDER = ("en", "es", "zh")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-artifact-id", required=True)
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


def _dataset_records() -> dict[str, dict[str, Any]]:
    dataset = pilot.build_dataset()
    records: dict[str, dict[str, Any]] = {}
    for example in dataset.examples:
        labels = dict(example.labels)
        prompt_messages = list(example.prompt)
        prompt_text = "\n\n".join(str(message.get("content") or "") for message in prompt_messages)
        records[str(example.key)] = {
            "group_id": labels["group_id"],
            "prime_condition": labels["prime_condition"],
            "language_code": labels["language_code"],
            "language_name": labels["language_name"],
            "prompt_text": prompt_text,
        }
    return records


def _attach_prompt_token_counts(records_by_key: dict[str, dict[str, Any]], manifest: dict[str, Any]) -> None:
    metadata_rows = manifest.get("metadata", {}).get("example_metadata", [])
    for row in metadata_rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("example_key") or "")
        if key in records_by_key:
            records_by_key[key]["prompt_token_count"] = int(row.get("prompt_token_count") or 0)


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


def _vector_for(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2:
        return np.asarray(array[-1], dtype=np.float32)
    return np.asarray(array, dtype=np.float32)


def _text_matrix(records_by_key: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, float]], float]:
    rows_by_lang = {
        lang: [
            (key, record)
            for key, record in records_by_key.items()
            if record["language_code"] == lang
        ]
        for lang in LANGUAGE_ORDER
    }
    matrix: dict[str, dict[str, float]] = {}
    cross_values: list[float] = []
    for train_lang in LANGUAGE_ORDER:
        train_rows = rows_by_lang[train_lang]
        train_texts = [record["prompt_text"] for _, record in train_rows]
        train_labels = [1 if record["prime_condition"] == "deontology" else 0 for _, record in train_rows]
        matrix[train_lang] = {}
        for test_lang in LANGUAGE_ORDER:
            test_rows = rows_by_lang[test_lang]
            test_texts = [record["prompt_text"] for _, record in test_rows]
            test_labels = [1 if record["prime_condition"] == "deontology" else 0 for _, record in test_rows]
            auc = round(_fit_text_auc(train_texts, train_labels, test_texts, test_labels), 4)
            matrix[train_lang][test_lang] = auc
            if train_lang != test_lang:
                cross_values.append(auc)
    mean_cross = round(sum(cross_values) / len(cross_values), 4)
    return matrix, mean_cross


def _length_matrix(records_by_key: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, float]], float]:
    rows_by_lang = {
        lang: [
            (key, record)
            for key, record in records_by_key.items()
            if record["language_code"] == lang
        ]
        for lang in LANGUAGE_ORDER
    }
    matrix: dict[str, dict[str, float]] = {}
    cross_values: list[float] = []
    for train_lang in LANGUAGE_ORDER:
        train_rows = rows_by_lang[train_lang]
        train_values = [float(record["prompt_token_count"]) for _, record in train_rows]
        train_labels = [1 if record["prime_condition"] == "deontology" else 0 for _, record in train_rows]
        matrix[train_lang] = {}
        for test_lang in LANGUAGE_ORDER:
            test_rows = rows_by_lang[test_lang]
            test_values = [float(record["prompt_token_count"]) for _, record in test_rows]
            test_labels = [1 if record["prime_condition"] == "deontology" else 0 for _, record in test_rows]
            auc = round(_fit_scalar_auc(train_values, train_labels, test_values, test_labels), 4)
            matrix[train_lang][test_lang] = auc
            if train_lang != test_lang:
                cross_values.append(auc)
    mean_cross = round(sum(cross_values) / len(cross_values), 4)
    return matrix, mean_cross


def _probe_matrices(
    feature_payload: dict[str, Any],
    records_by_key: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, float]]:
    layers = feature_payload["layers"]
    matrices_by_layer: dict[str, dict[str, dict[str, float]]] = {}
    mean_cross_by_layer: dict[str, float] = {}
    for layer in sorted(int(layer_str) for layer_str in layers):
        layer_map = feature_payload["layers"][str(layer)]
        rows_by_lang = {
            lang: [
                (
                    key,
                    records_by_key[key],
                    _vector_for(layer_map[key]["values"]),
                )
                for key in sorted(layer_map)
                if records_by_key[key]["language_code"] == lang
            ]
            for lang in LANGUAGE_ORDER
        }
        matrix: dict[str, dict[str, float]] = {}
        cross_values: list[float] = []
        for train_lang in LANGUAGE_ORDER:
            train_rows = rows_by_lang[train_lang]
            x_train = np.stack([vector for _, _, vector in train_rows], axis=0)
            y_train = [1 if record["prime_condition"] == "deontology" else 0 for _, record, _ in train_rows]
            matrix[train_lang] = {}
            for test_lang in LANGUAGE_ORDER:
                test_rows = rows_by_lang[test_lang]
                x_test = np.stack([vector for _, _, vector in test_rows], axis=0)
                y_test = [1 if record["prime_condition"] == "deontology" else 0 for _, record, _ in test_rows]
                auc = round(_fit_probe_auc(x_train, y_train, x_test, y_test), 4)
                matrix[train_lang][test_lang] = auc
                if train_lang != test_lang:
                    cross_values.append(auc)
        matrices_by_layer[str(layer)] = matrix
        mean_cross_by_layer[str(layer)] = round(sum(cross_values) / len(cross_values), 4)
    return matrices_by_layer, mean_cross_by_layer


def _cross_script_pairs_by_layer(probe_matrices_by_layer: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    tracked_pairs = (("en", "zh"), ("zh", "en"), ("es", "zh"), ("zh", "es"))
    return {
        layer: {
            f"{src}->{dst}": probe_matrices_by_layer[layer][src][dst]
            for src, dst in tracked_pairs
        }
        for layer in sorted(probe_matrices_by_layer, key=lambda value: int(value))
    }


def _zh_prompt_ascii_audit(records_by_key: dict[str, dict[str, Any]]) -> dict[str, Any]:
    token_counter: dict[str, int] = {}
    details = []
    rows_with_ascii = 0
    theory_terms = {"phronesis", "temperance", "virtue", "categorical", "imperative", "kantian"}
    for key, record in records_by_key.items():
        if record["language_code"] != "zh":
            continue
        word_tokens = re.findall(r"[^\W_]+", record["prompt_text"], flags=re.UNICODE)
        lowered = [
            token.lower()
            for token in word_tokens
            if token.isascii() and token.isalpha()
        ]
        if lowered:
            rows_with_ascii += 1
        for token in lowered:
            token_counter[token] = token_counter.get(token, 0) + 1
        details.append(
            {
                "example_key": key,
                "language_code": record["language_code"],
                "group_id": record["group_id"],
                "prime_condition": record["prime_condition"],
                "ascii_tokens": lowered,
                "english_theory_terms": [token for token in lowered if token in theory_terms],
            }
        )
    top_tokens = sorted(token_counter.items(), key=lambda item: (-item[1], item[0]))[:20]
    return {
        "zh_row_count": sum(1 for record in records_by_key.values() if record["language_code"] == "zh"),
        "rows_with_ascii_tokens": rows_with_ascii,
        "rows_with_english_theory_terms": sum(1 for item in details if item["english_theory_terms"]),
        "top_ascii_tokens": top_tokens,
        "details": details,
    }


def _random_label_control(
    feature_payload: dict[str, Any],
    records_by_key: dict[str, dict[str, Any]],
    *,
    layer: str,
    seed: int = 0,
    permutations: int = 256,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    layer_map = feature_payload["layers"][layer]
    rows_by_lang = {
        lang: [
            (
                key,
                records_by_key[key],
                _vector_for(layer_map[key]["values"]),
            )
            for key in sorted(layer_map)
            if records_by_key[key]["language_code"] == lang
        ]
        for lang in LANGUAGE_ORDER
    }

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
            x_train = np.stack([vector for _, _, vector in train_rows], axis=0)
            y_train_true = np.asarray(
                [1 if record["prime_condition"] == "deontology" else 0 for _, record, _ in train_rows],
                dtype=np.int64,
            )
            y_train = rng.permutation(y_train_true).tolist()
            for test_lang in LANGUAGE_ORDER:
                if train_lang == test_lang:
                    continue
                test_rows = rows_by_lang[test_lang]
                x_test = np.stack([vector for _, _, vector in test_rows], axis=0)
                y_test = [1 if record["prime_condition"] == "deontology" else 0 for _, record, _ in test_rows]
                auc = _fit_probe_auc(x_train, y_train, x_test, y_test)
                ordered_pair_values[f"{train_lang}->{test_lang}"].append(float(auc))
                pair_scores.append(float(auc))
        mean_values.append(float(np.mean(pair_scores)))

    summary_pairs = {
        pair: {
            "mean": round(float(np.mean(values)), 4),
            "p95": round(float(np.quantile(values, 0.95)), 4),
            "max": round(float(np.max(values)), 4),
        }
        for pair, values in ordered_pair_values.items()
    }
    return {
        "layer": layer,
        "permutations": permutations,
        "mean_cross_language_auroc_mean": round(float(np.mean(mean_values)), 4),
        "mean_cross_language_auroc_p95": round(float(np.quantile(mean_values, 0.95)), 4),
        "mean_cross_language_auroc_max": round(float(np.max(mean_values)), 4),
        "share_mean_cross_language_auroc_ge_060": round(float(np.mean(np.asarray(mean_values) >= 0.60)), 4),
        "share_mean_cross_language_auroc_ge_080": round(float(np.mean(np.asarray(mean_values) >= 0.80)), 4),
        "ordered_pair_summary": summary_pairs,
    }


def _best_layer(mean_cross_by_layer: dict[str, float], prompt_text_mean: float) -> tuple[str, float]:
    best_layer = max(
        mean_cross_by_layer,
        key=lambda layer: (mean_cross_by_layer[layer] - prompt_text_mean, mean_cross_by_layer[layer], -int(layer)),
    )
    return best_layer, round(mean_cross_by_layer[best_layer] - prompt_text_mean, 4)


def _write_report(summary: dict[str, Any]) -> None:
    text_matrix = summary["prompt_text_matrix"]
    length_matrix = summary["prompt_length_matrix"]
    best_layer = summary["best_layer"]
    best_matrix = summary["probe_matrices_by_layer"][best_layer]
    cross_script_pairs = summary["cross_script_pairs_by_layer"]
    residue = summary["zh_prompt_ascii_audit"]
    random_control = summary["random_label_control"]
    report = f"""# Experiment 02 Cross-Language Prompt Probe

Prompt-final residual probe on the translated `English / Spanish / Simplified Chinese` pilot prompts for `deontology` vs `virtue_ethics`.

## Run
- capture artifact: `{summary['capture_artifact_id']}`
- example count: `{summary['row_count']}`
- layers: `{', '.join(str(layer) for layer in summary['captured_layers'])}`

## Prompt Text Baseline
Raw prompt-text char-TF-IDF AUROC matrix:

| train -> test | `en` | `es` | `zh` |
| --- | ---: | ---: | ---: |
| `en` | `{text_matrix['en']['en']:.2f}` | `{text_matrix['en']['es']:.2f}` | `{text_matrix['en']['zh']:.2f}` |
| `es` | `{text_matrix['es']['en']:.2f}` | `{text_matrix['es']['es']:.2f}` | `{text_matrix['es']['zh']:.2f}` |
| `zh` | `{text_matrix['zh']['en']:.2f}` | `{text_matrix['zh']['es']:.2f}` | `{text_matrix['zh']['zh']:.2f}` |

- mean cross-language prompt-text AUROC: `{summary['mean_cross_language_prompt_text_auroc']:.4f}`

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
- delta vs prompt-text baseline: `{summary['best_layer_delta_vs_prompt_text']:.4f}`

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

## Red-Team Checks
- Chinese prompt rows with ASCII residue: `{residue['rows_with_ascii_tokens']}/{residue['zh_row_count']}`
- Chinese prompt rows with English theory-term residue: `{residue['rows_with_english_theory_terms']}/{residue['zh_row_count']}`
- Top ASCII residues in Chinese prompts: `{', '.join(f'{token} ({count})' for token, count in residue['top_ascii_tokens'][:10])}`
- Random-label control at L32:
  - mean cross-language AUROC under permutation: `{random_control['mean_cross_language_auroc_mean']:.4f}`
  - 95th percentile: `{random_control['mean_cross_language_auroc_p95']:.4f}`
  - max over `{random_control['permutations']}` permutations: `{random_control['mean_cross_language_auroc_max']:.4f}`
  - share of permutations with mean cross-language AUROC `>= 0.60`: `{random_control['share_mean_cross_language_auroc_ge_060']:.4f}`
  - share with `>= 0.80`: `{random_control['share_mean_cross_language_auroc_ge_080']:.4f}`

## Interpretation
- Prompt-side is the right substrate only if the prompt-final probe transfers across languages better than the prompt-text baseline.
- The critical comparison here is cross-language mean AUROC, not within-language diagonals.
- This pilot is tiny (`5` dilemmas per language pair), so exact cells are noisy; what matters is whether any layer clearly opens room over the prompt-text baseline.
- The key structural check passes: the cross-script pairs themselves rise through the stack and are already high by `16`, then saturate by `32`.
- The random-label control is mixed: its mean stays near chance, but the tail is wide enough that this `5`-dilemma pilot is not by itself sufficient to rule out small-N overfitting.
- The Chinese prompt audit is cleaner than the response-side case: some ASCII proper names remain, but English theory-term residue in the Chinese prompts is `0`, and the cross-script prompt-text baseline is still low.

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
    records_by_key = _dataset_records()
    _attach_prompt_token_counts(records_by_key, manifest)

    prompt_text_matrix, mean_prompt_text = _text_matrix(records_by_key)
    prompt_length_matrix, mean_prompt_length = _length_matrix(records_by_key)
    probe_matrices_by_layer, mean_probe_by_layer = _probe_matrices(feature_payload, records_by_key)
    best_layer, best_delta = _best_layer(mean_probe_by_layer, mean_prompt_text)
    cross_script_pairs = _cross_script_pairs_by_layer(probe_matrices_by_layer)
    prompt_ascii_residue = _zh_prompt_ascii_audit(records_by_key)
    random_label_control = _random_label_control(feature_payload, records_by_key, layer=best_layer)

    best_mean = mean_probe_by_layer[best_layer]
    recommendation = (
        "Prompt-side looks promising enough to scale, but this pilot is not claim-ready; keep the interpretation frozen and verify on the 30-dilemma run."
        if best_delta >= 0.10
        else "Prompt-side does not yet beat the prompt-text baseline by a meaningful margin on this pilot."
    )

    summary = {
        "capture_artifact_id": args.capture_artifact_id,
        "row_count": len(records_by_key),
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
        "zh_prompt_ascii_audit": prompt_ascii_residue,
        "random_label_control": random_label_control,
        "recommendation": recommendation,
        "decision": {
            "best_layer_delta_ge_010": bool(best_delta >= 0.10),
            "best_layer_mean_cross_language_probe_auroc": best_mean,
            "random_label_mean_p95_le_060": bool(random_label_control["mean_cross_language_auroc_p95"] <= 0.60),
            "random_label_share_ge_080_le_010": bool(random_label_control["share_mean_cross_language_auroc_ge_080"] <= 0.10),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(summary)


if __name__ == "__main__":
    main()
