from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import modal
import numpy as np
from safetensors.numpy import load as load_safetensors
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipelines_v2.storage.features import decode_feature_payload
from projects.MOREBENCH.phase_03.scripts.analyze_experiment_02_extended_metrics import _load_rows_and_matrices


PROMPT_CATALOG_ROOT = Path("artifacts") / "morebench_phase03_experiment02_cross_language_prompt_probe_full_catalog"
PROMPT_TRANSFORM_RESULT = (
    Path("artifacts")
    / "morebench_phase03_experiment02_cross_language_prompt_probe_full"
    / "transform_33d92c1d07d0_339727c1"
    / "result.json"
)
REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_cross_language_prompt_probe_followups")
REPORT_PATH = REPORT_DIR / "report.md"
SUMMARY_PATH = REPORT_DIR / "summary.json"

PROMPT_CAPTURE_ID = "capture_1_2c011b403d39"
RESPONSE_CAPTURE_DATASET_ID = "transform_1_4a60e2ca"
RESPONSE_CAPTURE_ID = "capture_1_34cdfd7923d9"
RESPONSE_PRIME_FAMILY = "description_only"
TARGET_PRIMES = ("deontology", "virtue_ethics")
THEORY_NAME_PATTERNS = (
    "deontology",
    "kantian",
    "virtue ethics",
    "aristotelian",
    "康德",
    "亚里士多德",
    "义务论",
    "德性伦理",
    "deontología",
    "ética de la virtud",
)


def _artifact_manifest(capture_artifact_id: str) -> dict[str, Any]:
    path = PROMPT_CATALOG_ROOT / f"{capture_artifact_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _modal_relative_path(path: str) -> str:
    return path.removeprefix("/data/")


def _load_capture_feature(ref: dict[str, Any]) -> dict[str, Any]:
    volume = modal.Volume.from_name(str(ref["name"]))
    metadata = json.loads(b"".join(volume.read_file(_modal_relative_path(str(ref["metadata_path"])))))
    tensors = load_safetensors(b"".join(volume.read_file(_modal_relative_path(str(ref["tensor_path"])))))
    return decode_feature_payload(metadata, tensors)


def _load_prompt_records() -> list[dict[str, Any]]:
    payload = json.loads(PROMPT_TRANSFORM_RESULT.read_text(encoding="utf-8"))
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
                "prompt_text": prompt_text,
            }
        )
    return records


def _response_subset() -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    rows, matrices = _load_rows_and_matrices(
        capture_dataset_id=RESPONSE_CAPTURE_DATASET_ID,
        capture_id=RESPONSE_CAPTURE_ID,
    )
    selected = [
        row
        for row in rows
        if row["prime_family"] == RESPONSE_PRIME_FAMILY and row["prime_condition"] in TARGET_PRIMES
    ]
    groups_to_primes: dict[str, set[str]] = {}
    for row in selected:
        groups_to_primes.setdefault(row["group_id"], set()).add(row["prime_condition"])
    complete_groups = sorted(group for group, primes in groups_to_primes.items() if set(primes) == set(TARGET_PRIMES))
    keep_indices = [
        idx
        for idx, row in enumerate(rows)
        if row["prime_family"] == RESPONSE_PRIME_FAMILY
        and row["prime_condition"] in TARGET_PRIMES
        and row["group_id"] in complete_groups
    ]
    filtered_rows = [rows[idx] for idx in keep_indices]
    filtered_matrices = {layer: matrix[np.asarray(keep_indices, dtype=np.int32)] for layer, matrix in matrices.items()}
    return filtered_rows, filtered_matrices


def _vector_for(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2:
        return np.asarray(array[-1], dtype=np.float32)
    return np.asarray(array, dtype=np.float32)


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


def _fit_probe_auc(train_vectors: np.ndarray, train_labels: list[int], test_vectors: np.ndarray, test_labels: list[int]) -> float:
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
    model.fit(train_vectors, train_labels)
    probs = model.predict_proba(test_vectors)[:, 1]
    return float(roc_auc_score(test_labels, probs))


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    left_norm = float(np.linalg.norm(left64))
    right_norm = float(np.linalg.norm(right64))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return round(float(np.dot(left64, right64) / (left_norm * right_norm)), 4)


def _rows_by_lang(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        lang: [record for record in records if record["language_code"] == lang]
        for lang in ("en", "es", "zh")
    }


def _label_for(record: dict[str, Any]) -> int:
    return 1 if record["prime_condition"] == "deontology" else 0


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
    mean_values: list[float] = []
    for _ in range(permutations):
        pair_scores: list[float] = []
        for train_lang in ("en", "es", "zh"):
            train_rows = rows_by_lang[train_lang]
            x_train = np.stack([_vector_for(layer_map[record["key"]]["values"]) for record in train_rows], axis=0)
            y_train_true = np.asarray([_label_for(record) for record in train_rows], dtype=np.int64)
            y_train = rng.permutation(y_train_true).tolist()
            for test_lang in ("en", "es", "zh"):
                if train_lang == test_lang:
                    continue
                test_rows = rows_by_lang[test_lang]
                x_test = np.stack([_vector_for(layer_map[record["key"]]["values"]) for record in test_rows], axis=0)
                y_test = [_label_for(record) for record in test_rows]
                pair_scores.append(float(_fit_probe_auc(x_train, y_train, x_test, y_test)))
        mean_values.append(float(np.mean(pair_scores)))
    values = np.asarray(mean_values, dtype=np.float32)
    return {
        "layer": layer,
        "permutations": permutations,
        "mean_cross_language_auroc_mean": round(float(np.mean(values)), 4),
        "mean_cross_language_auroc_p95": round(float(np.quantile(values, 0.95)), 4),
        "mean_cross_language_auroc_max": round(float(np.max(values)), 4),
        "share_mean_cross_language_auroc_ge_060": round(float(np.mean(values >= 0.60)), 4),
        "share_mean_cross_language_auroc_ge_080": round(float(np.mean(values >= 0.80)), 4),
    }


def _cross_layer_projection(
    feature_payload: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    source_layer: str,
    target_layers: tuple[str, ...],
) -> dict[str, Any]:
    rows_by_lang = _rows_by_lang(records)
    source_map = feature_payload["layers"][source_layer]
    result: dict[str, Any] = {}
    for target_layer in target_layers:
        target_map = feature_payload["layers"][target_layer]
        matrix: dict[str, dict[str, float]] = {}
        cross_values: list[float] = []
        for train_lang in ("en", "es", "zh"):
            train_rows = rows_by_lang[train_lang]
            train_matrix = np.stack([_vector_for(source_map[record["key"]]["values"]) for record in train_rows], axis=0)
            train_labels = np.asarray([_label_for(record) for record in train_rows], dtype=np.int32)
            direction = _fit_direction(train_matrix, train_labels)
            matrix[train_lang] = {}
            for test_lang in ("en", "es", "zh"):
                test_rows = rows_by_lang[test_lang]
                test_matrix = np.stack([_vector_for(target_map[record["key"]]["values"]) for record in test_rows], axis=0)
                test_labels = np.asarray([_label_for(record) for record in test_rows], dtype=np.int32)
                scores = test_matrix @ direction
                auc = float(roc_auc_score(test_labels, scores))
                matrix[train_lang][test_lang] = round(auc, 4)
                if train_lang != test_lang:
                    cross_values.append(auc)
        result[target_layer] = {
            "matrix": matrix,
            "mean_cross_language_auroc": round(float(np.mean(cross_values)), 4),
        }
    return result


def _prompt_name_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows_with_names: list[dict[str, Any]] = []
    pattern_counts = {pattern: 0 for pattern in THEORY_NAME_PATTERNS}
    for record in records:
        prompt_text = record["prompt_text"]
        low = prompt_text.lower()
        hits = [pattern for pattern in THEORY_NAME_PATTERNS if pattern.lower() in low or pattern in prompt_text]
        if hits:
            rows_with_names.append(
                {
                    "example_key": record["key"],
                    "group_id": record["group_id"],
                    "language_code": record["language_code"],
                    "prime_condition": record["prime_condition"],
                    "hits": hits,
                }
            )
            for hit in hits:
                pattern_counts[hit] += 1
    return {
        "row_count": len(records),
        "rows_with_theory_name_hits": len(rows_with_names),
        "pattern_counts": pattern_counts,
        "rows": rows_with_names[:20],
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    prompt_manifest = _artifact_manifest(PROMPT_CAPTURE_ID)
    prompt_feature_ref = prompt_manifest["storage_refs"]["features"]["prompt_eos_residual"]
    prompt_feature_payload = _load_capture_feature(prompt_feature_ref)
    prompt_records = _load_prompt_records()
    prompt_name_audit = _prompt_name_audit(prompt_records)

    overlapping_groups = sorted({row["group_id"] for row in _response_subset()[0]})
    prompt_records_overlap = [
        record
        for record in prompt_records
        if record["group_id"] in overlapping_groups
    ]
    prompt_records_by_key = {record["key"]: record for record in prompt_records_overlap}
    prompt_labels = np.asarray(
        [1 if record["prime_condition"] == "deontology" else 0 for record in prompt_records_overlap],
        dtype=np.int32,
    )

    prompt_directions: dict[int, np.ndarray] = {}
    for layer_str, layer_map in prompt_feature_payload["layers"].items():
        layer = int(layer_str)
        matrix = np.stack(
            [_vector_for(layer_map[record["key"]]["values"]) for record in prompt_records_overlap],
            axis=0,
        )
        prompt_directions[layer] = _fit_direction(matrix, prompt_labels)

    response_rows, response_matrices = _response_subset()
    response_labels = np.asarray(
        [1 if row["prime_condition"] == "deontology" else 0 for row in response_rows],
        dtype=np.int32,
    )
    response_directions = {
        int(layer): _fit_direction(matrix.astype(np.float32), response_labels)
        for layer, matrix in response_matrices.items()
    }

    prompt_l32 = prompt_directions[32]
    cosine_vs_response = {
        str(layer): _cosine(prompt_l32, direction)
        for layer, direction in sorted(response_directions.items())
    }
    prompt_internal = {
        "16_vs_32": _cosine(prompt_directions[16], prompt_directions[32]),
        "24_vs_32": _cosine(prompt_directions[24], prompt_directions[32]),
        "32_vs_40": _cosine(prompt_directions[32], prompt_directions[40]),
    }
    random_label_controls = {
        layer: _random_label_control(prompt_feature_payload, prompt_records, layer=layer)
        for layer in ("16", "24", "32")
    }
    cross_layer_projection = _cross_layer_projection(
        prompt_feature_payload,
        prompt_records,
        source_layer="32",
        target_layers=("16", "24", "40"),
    )

    summary = {
        "prompt_capture_id": PROMPT_CAPTURE_ID,
        "response_capture_dataset_id": RESPONSE_CAPTURE_DATASET_ID,
        "response_capture_id": RESPONSE_CAPTURE_ID,
        "overlap_group_count": len(overlapping_groups),
        "prompt_name_audit": prompt_name_audit,
        "prompt_internal_direction_cosines": prompt_internal,
        "prompt_l32_vs_response_direction_cosines": cosine_vs_response,
        "random_label_controls": random_label_controls,
        "cross_layer_projection_from_l32": cross_layer_projection,
        "read": {
            "name_ablation_by_construction": bool(prompt_name_audit["rows_with_theory_name_hits"] == 0),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    report_lines = [
        "# Experiment 02 Cross-Language Prompt Probe Follow-Ups",
        "",
        "## Prompt Name Audit",
        f"- prompt rows audited: `{prompt_name_audit['row_count']}`",
        f"- rows with theory-name hits: `{prompt_name_audit['rows_with_theory_name_hits']}`",
        f"- pattern counts: `{prompt_name_audit['pattern_counts']}`",
        "",
        "## Direction Cosines",
        f"- overlapping groups used for prompt-vs-response comparison: `{len(overlapping_groups)}`",
        f"- prompt internal cosine `L16 vs L32`: `{prompt_internal['16_vs_32']}`",
        f"- prompt internal cosine `L24 vs L32`: `{prompt_internal['24_vs_32']}`",
        f"- prompt internal cosine `L32 vs L40`: `{prompt_internal['32_vs_40']}`",
        "",
        "### Prompt L32 vs Old Response-Side Directions",
    ]
    for layer, value in cosine_vs_response.items():
        report_lines.append(f"- response layer `{layer}` cosine with prompt `L32`: `{value}`")
    report_lines.extend(["", "## Random-Label Controls"])
    for layer in ("16", "24", "32"):
        item = random_label_controls[layer]
        report_lines.append(
            f"- layer `{layer}`: mean `{item['mean_cross_language_auroc_mean']}`, p95 `{item['mean_cross_language_auroc_p95']}`, "
            f"max `{item['mean_cross_language_auroc_max']}`, share `>= 0.80` `{item['share_mean_cross_language_auroc_ge_080']}`"
        )
    report_lines.extend(["", "## Cross-Layer Projection From L32"])
    for target_layer in ("16", "24", "40"):
        item = cross_layer_projection[target_layer]
        report_lines.append(f"- target layer `{target_layer}` mean cross-language AUROC using `L32` directions: `{item['mean_cross_language_auroc']}`")
        for train_lang in ("en", "es", "zh"):
            row = item["matrix"][train_lang]
            report_lines.append(
                f"  - `{train_lang}` direction -> `en {row['en']}`, `es {row['es']}`, `zh {row['zh']}`"
            )
    report_lines.extend(
        [
            "",
            "## Read",
            "- The full-30 prompt-side asset is already description-only rather than name-only.",
            "- If the prompt name audit remains at zero hits, the original neutral-tag control is effectively satisfied by construction for framework-name tokens.",
            "- The cosine table should be read as a relationship check, not as a success metric. High cosine means the prompt-side direction and old response-side direction point similarly in residual space; low cosine means the prompt-side reopening is geometrically distinct from the old response-side readout.",
            "- The random-label controls at `L16`, `L24`, and `L32` tell us whether the low inter-layer cosines are compatible with real structure rather than small-N memorization.",
            "- The cross-layer projection test asks a different question: whether the `L32` separator itself is already present at earlier layers without retraining.",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
