from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import modal
import numpy as np
from safetensors.numpy import load as load_safetensors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipelines_v2.storage.features import decode_feature_payload


CATALOG_ROOT = Path("artifacts") / "morebench_phase03_experiment02_cross_language_prompt_probe_dense_catalog"
TRANSFORM_RESULT_PATH = (
    Path("artifacts")
    / "morebench_phase03_experiment02_cross_language_prompt_probe_full"
    / "transform_33d92c1d07d0_339727c1"
    / "result.json"
)
REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_cross_language_prompt_probe_dense")
REPORT_PATH = REPORT_DIR / "report.md"
SUMMARY_PATH = REPORT_DIR / "summary.json"

LANGUAGE_ORDER = ("en", "es", "zh")
TRACKED_PAIRS = (("en", "zh"), ("zh", "en"), ("es", "zh"), ("zh", "es"))


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


def _load_records() -> list[dict[str, Any]]:
    payload = json.loads(TRANSFORM_RESULT_PATH.read_text(encoding="utf-8"))
    dataset = payload["dataset"]
    records: list[dict[str, Any]] = []
    for example in dataset["examples"]:
        labels = dict(example.get("labels", {}))
        records.append(
            {
                "key": str(example["key"]),
                "group_id": str(labels["group_id"]),
                "prime_condition": str(labels["prime_condition"]),
                "language_code": str(labels["language_code"]),
            }
        )
    return records


def _vector_for(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2:
        return np.asarray(array[-1], dtype=np.float32)
    return np.asarray(array, dtype=np.float32)


def _label_for(record: dict[str, Any]) -> int:
    return 1 if record["prime_condition"] == "deontology" else 0


def _fit_probe_auc(train_vectors: np.ndarray, train_labels: list[int], test_vectors: np.ndarray, test_labels: list[int]) -> float:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced", solver="liblinear"),
    )
    model.fit(train_vectors, train_labels)
    probs = model.predict_proba(test_vectors)[:, 1]
    return float(roc_auc_score(test_labels, probs))


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


def main() -> None:
    args = _parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = _artifact_manifest(args.capture_artifact_id)
    feature_ref = manifest["storage_refs"]["features"]["prompt_eos_residual"]
    feature_payload = _load_capture_feature(feature_ref)
    records = _load_records()

    rows_by_lang = {
        lang: [record for record in records if record["language_code"] == lang]
        for lang in LANGUAGE_ORDER
    }

    probe_matrices_by_layer: dict[str, dict[str, dict[str, float]]] = {}
    mean_cross_by_layer: dict[str, float] = {}
    directions: dict[str, np.ndarray] = {}
    for layer in sorted(int(layer_str) for layer_str in feature_payload["layers"]):
        layer_map = feature_payload["layers"][str(layer)]
        matrix: dict[str, dict[str, float]] = {}
        cross_values: list[float] = []
        full_matrix = np.stack([_vector_for(layer_map[record["key"]]["values"]) for record in records], axis=0)
        full_labels = np.asarray([_label_for(record) for record in records], dtype=np.int32)
        directions[str(layer)] = _fit_direction(full_matrix, full_labels)
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
        probe_matrices_by_layer[str(layer)] = matrix
        mean_cross_by_layer[str(layer)] = round(float(np.mean(cross_values)), 4)

    cross_script_pairs = {
        layer: {f"{src}->{dst}": probe_matrices_by_layer[layer][src][dst] for src, dst in TRACKED_PAIRS}
        for layer in sorted(probe_matrices_by_layer, key=lambda value: int(value))
    }
    direction_cosines = {}
    ordered_layers = [str(layer) for layer in sorted(int(layer_str) for layer_str in feature_payload["layers"])]
    for left, right in zip(ordered_layers[:-1], ordered_layers[1:], strict=True):
        direction_cosines[f"{left}_vs_{right}"] = _cosine(directions[left], directions[right])
    direction_cosines["16_vs_32"] = _cosine(directions["16"], directions["32"])
    direction_cosines["24_vs_32"] = _cosine(directions["24"], directions["32"])
    direction_cosines["28_vs_32"] = _cosine(directions["28"], directions["32"])

    summary = {
        "capture_artifact_id": args.capture_artifact_id,
        "row_count": len(records),
        "layers": [int(layer) for layer in ordered_layers],
        "probe_matrices_by_layer": probe_matrices_by_layer,
        "mean_cross_language_probe_auroc_by_layer": mean_cross_by_layer,
        "cross_script_pairs_by_layer": cross_script_pairs,
        "direction_cosines": direction_cosines,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Experiment 02 Cross-Language Prompt Probe Dense Layer Sweep",
        "",
        f"- capture artifact: `{args.capture_artifact_id}`",
        f"- example count: `{len(records)}`",
        f"- layers: `{', '.join(ordered_layers)}`",
        "",
        "## Mean Cross-Language AUROC By Layer",
    ]
    for layer in ordered_layers:
        report_lines.append(f"- layer `{layer}`: `{mean_cross_by_layer[layer]}`")
    report_lines.extend(["", "## Cross-Script Ordered Pairs"])
    for layer in ordered_layers:
        pairs = cross_script_pairs[layer]
        report_lines.append(
            f"- layer `{layer}`: `en->zh={pairs['en->zh']}`, `zh->en={pairs['zh->en']}`, `es->zh={pairs['es->zh']}`, `zh->es={pairs['zh->es']}`"
        )
    report_lines.extend(["", "## Direction Cosines"])
    for key, value in direction_cosines.items():
        report_lines.append(f"- `{key}`: `{value}`")
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
