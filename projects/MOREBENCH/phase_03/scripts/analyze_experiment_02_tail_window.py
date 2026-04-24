from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipelines_v2.api import ModalVolumeStore, PostgresCatalog, PostgresSource, TransferPolicy
from pipelines_v2.storage.artifacts import CaptureArtifact, OperationArtifact, artifact_from_manifest


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
ARTIFACT_ROOT = "/data/artifacts/morebench_phase_03_experiment02"
ARTIFACT_STORE_NAME = "xenon-data"
FEATURE_NAME = "generated_sequence_residual"
REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_tail_window_analysis")


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
        "generated_text": str(labels.get("generated_text") or ""),
    }


def _suffix_text(text: str, fraction: float) -> str:
    pieces = text.split()
    if not pieces:
        return text
    start = min(len(pieces) - 1, max(0, int(math.floor(len(pieces) * (1.0 - fraction)))))
    return " ".join(pieces[start:])


def _class_support(labels: list[str]) -> dict[str, int]:
    support: dict[str, int] = {}
    for label in labels:
        support[label] = support.get(label, 0) + 1
    return dict(sorted(support.items()))


def _encode_labels(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    class_names = sorted({row["prime_condition"] for row in rows})
    index = {name: idx for idx, name in enumerate(class_names)}
    return np.asarray([index[row["prime_condition"]] for row in rows], dtype=np.int64), class_names


def _load_rows(capture_dataset_id: str) -> list[dict[str, Any]]:
    dataset_artifact = _load_operation_artifact(capture_dataset_id)
    payload = dataset_artifact.result()
    dataset_payload = payload.get("dataset") if isinstance(payload, dict) else None
    if not isinstance(dataset_payload, dict):
        raise TypeError(f"Capture dataset artifact {capture_dataset_id!r} missing serialized dataset")
    return [_row_from_example(example) for example in list(dataset_payload.get("examples", []))]


def _load_tail_matrices(
    capture_id: str,
    *,
    fraction: float,
) -> tuple[dict[int, np.ndarray], list[str], dict[str, Any]]:
    capture_artifact = _load_capture_artifact(capture_id)
    feature = capture_artifact.feature(FEATURE_NAME)
    payload = feature.load()
    if payload.get("kind") != "residual":
        raise TypeError(f"Expected residual feature payload, got {payload.get('kind')!r}")

    layers = sorted(int(layer) for layer in payload["layers"])
    example_keys = sorted(payload["layers"][str(layers[0])])
    matrices: dict[int, np.ndarray] = {}
    token_counts: list[int] = []
    kept_counts: list[int] = []

    for layer in layers:
        layer_payload = payload["layers"][str(layer)]
        rows: list[np.ndarray] = []
        for key in example_keys:
            record = dict(layer_payload[key])
            values = np.asarray(record["values"], dtype=np.float32)
            if values.ndim != 2:
                raise TypeError("Residual capture values must be rank-2")
            token_count = int(values.shape[0])
            start = max(0, min(token_count - 1, int(math.floor(token_count * (1.0 - fraction)))))
            selected = values[start:]
            if selected.shape[0] <= 0:
                raise RuntimeError(f"Tail selection was empty for example {key!r} at layer {layer}")
            if layer == layers[0]:
                token_counts.append(token_count)
                kept_counts.append(int(selected.shape[0]))
            rows.append(selected.mean(axis=0).astype(np.float32))
        matrices[layer] = np.stack(rows, axis=0).astype(np.float32)

    stats = {
        "tail_fraction": fraction,
        "token_count_summary": {
            "min": int(min(token_counts)),
            "median": float(np.median(token_counts)),
            "max": int(max(token_counts)),
        },
        "tail_token_count_summary": {
            "min": int(min(kept_counts)),
            "median": float(np.median(kept_counts)),
            "max": int(max(kept_counts)),
        },
    }
    return matrices, example_keys, stats


def _text_metrics(train_texts: list[str], train_y: np.ndarray, test_texts: list[str], test_y: np.ndarray) -> dict[str, Any]:
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
    return {
        "accuracy": round(float(accuracy_score(test_y, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(test_y, preds)), 4),
    }


def _probe_metrics(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict[str, Any]:
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
    preds = model.predict(test_x)
    return {
        "accuracy": round(float(accuracy_score(test_y, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(test_y, preds)), 4),
    }


def analyze(
    *,
    capture_dataset_id: str,
    capture_id: str,
    tail_fraction: float,
) -> dict[str, Any]:
    rows = _load_rows(capture_dataset_id)
    matrices, example_keys, tail_stats = _load_tail_matrices(capture_id, fraction=tail_fraction)
    row_map = {row["example_key"]: row for row in rows}
    ordered_rows = [row_map[key] for key in example_keys]
    y, class_names = _encode_labels(ordered_rows)

    train_mask = np.asarray([row["split"] == "train" for row in ordered_rows], dtype=bool)
    test_mask = np.asarray([row["split"] == "test" for row in ordered_rows], dtype=bool)

    train_rows = [row for row, keep in zip(ordered_rows, train_mask.tolist(), strict=True) if keep]
    test_rows = [row for row, keep in zip(ordered_rows, test_mask.tolist(), strict=True) if keep]
    train_texts = [_suffix_text(row["generated_text"], tail_fraction) for row in train_rows]
    test_texts = [_suffix_text(row["generated_text"], tail_fraction) for row in test_rows]
    train_y = y[train_mask]
    test_y = y[test_mask]

    text_result = _text_metrics(train_texts, train_y, test_texts, test_y)
    probe_by_layer: list[dict[str, Any]] = []
    for layer, matrix in sorted(matrices.items()):
        probe_result = _probe_metrics(matrix[train_mask], train_y, matrix[test_mask], test_y)
        probe_by_layer.append(
            {
                "layer": layer,
                **probe_result,
                "probe_minus_text_baseline_balanced_accuracy": round(
                    probe_result["balanced_accuracy"] - text_result["balanced_accuracy"],
                    4,
                ),
            }
        )
    best_probe = max(probe_by_layer, key=lambda item: item["balanced_accuracy"])
    return {
        "capture_dataset_id": capture_dataset_id,
        "capture_id": capture_id,
        "tail_fraction": tail_fraction,
        "text_tail_method": "last_25_percent_of_whitespace_tokens",
        "class_names": class_names,
        "train_example_count": int(train_mask.sum()),
        "test_example_count": int(test_mask.sum()),
        "train_class_support": _class_support([row["prime_condition"] for row in train_rows]),
        "test_class_support": _class_support([row["prime_condition"] for row in test_rows]),
        "tail_stats": tail_stats,
        "text_baseline": text_result,
        "best_probe_layer": best_probe,
        "probe_by_layer": probe_by_layer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze tail-window probe/text metrics on existing Experiment 2 capture")
    parser.add_argument("--capture-dataset-id", required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "tail_window_metrics.json")
    args = parser.parse_args()

    result = analyze(
        capture_dataset_id=args.capture_dataset_id,
        capture_id=args.capture_id,
        tail_fraction=float(args.tail_fraction),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
