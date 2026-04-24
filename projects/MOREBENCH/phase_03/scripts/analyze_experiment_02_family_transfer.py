from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipelines_v2.api import (
    ModalVolumeStore,
    PostgresCatalog,
    PostgresSource,
    TransferPolicy,
    TokenPooling,
    TokenSelector,
)
from pipelines_v2.operations.execution.common import feature_matrices
from pipelines_v2.storage.artifacts import CaptureArtifact, OperationArtifact, artifact_from_manifest


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
ARTIFACT_ROOT = "/data/artifacts/morebench_phase_03_experiment02"
ARTIFACT_STORE_NAME = "xenon-data"
FEATURE_NAME = "generated_sequence_residual"
REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_family_transfer_analysis")


@dataclass(frozen=True, slots=True)
class FamilySpec:
    family: str
    capture_dataset_id: str
    capture_id: str


def _catalog() -> PostgresCatalog:
    return PostgresCatalog(source=PostgresSource.from_env(DB_ENV_VAR))


def _store() -> ModalVolumeStore:
    return ModalVolumeStore(
        name=ARTIFACT_STORE_NAME,
        root=ARTIFACT_ROOT,
        transfer_policy=TransferPolicy(allow_large_transfer=True),
    )


def _load_operation_artifact(artifact_id: str) -> OperationArtifact:
    catalog = _catalog()
    store = _store()
    manifest = catalog.load_artifact(artifact_id)
    if manifest is None:
        raise RuntimeError(f"Could not load artifact manifest {artifact_id!r}")
    artifact = artifact_from_manifest(manifest, store=store)
    if not isinstance(artifact, OperationArtifact):
        raise TypeError(f"Artifact {artifact_id!r} is not an operation artifact")
    return artifact


def _load_capture_artifact(artifact_id: str) -> CaptureArtifact:
    catalog = _catalog()
    store = _store()
    manifest = catalog.load_artifact(artifact_id)
    if manifest is None:
        raise RuntimeError(f"Could not load artifact manifest {artifact_id!r}")
    artifact = artifact_from_manifest(manifest, store=store)
    if not isinstance(artifact, CaptureArtifact):
        raise TypeError(f"Artifact {artifact_id!r} is not a capture artifact")
    return artifact


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z'-]+", text.lower())
        if len(token) >= 5
    }


def _row_from_example(example: dict[str, Any], *, family_fallback: str) -> dict[str, Any]:
    labels = dict(example.get("labels", {}))
    return {
        "example_key": str(example["key"]),
        "group_id": str(labels.get("group_id") or ""),
        "split": str(labels.get("split") or ""),
        "prime_condition": str(labels.get("prime_condition") or ""),
        "prime_family": str(labels.get("prime_family") or family_fallback),
        "alias_bank": str(labels.get("alias_bank") or ""),
        "generated_text": str(labels.get("generated_text") or ""),
        "source_family": str(labels.get("source_family") or ""),
    }


def _load_family_rows(spec: FamilySpec) -> tuple[list[dict[str, Any]], dict[int, tuple[np.ndarray, list[str]]]]:
    dataset_artifact = _load_operation_artifact(spec.capture_dataset_id)
    capture_artifact = _load_capture_artifact(spec.capture_id)

    payload = dataset_artifact.result()
    dataset_payload = payload.get("dataset") if isinstance(payload, dict) else None
    if not isinstance(dataset_payload, dict):
        raise TypeError(f"Capture dataset artifact {spec.capture_dataset_id!r} missing serialized dataset")
    examples = list(dataset_payload.get("examples", []))
    rows = [_row_from_example(example, family_fallback=spec.family) for example in examples]
    row_keys = {row["example_key"] for row in rows}

    matrices, example_keys = feature_matrices(
        capture_artifact.feature(FEATURE_NAME),
        token_selector=TokenSelector.full_sequence(),
        token_pooling=TokenPooling.mean(),
    )
    if set(example_keys) != row_keys:
        missing = sorted(row_keys - set(example_keys))
        extra = sorted(set(example_keys) - row_keys)
        raise RuntimeError(
            f"Feature/row mismatch for {spec.family}: missing_feature={missing[:5]} extra_feature={extra[:5]}"
        )
    return rows, {layer: (matrix, example_keys) for layer, matrix in matrices.items()}


def _merge_family_data(
    family_specs: list[FamilySpec],
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray]]:
    all_rows: list[dict[str, Any]] = []
    per_layer_vectors: dict[int, list[np.ndarray]] = {}

    for spec in family_specs:
        rows, matrices = _load_family_rows(spec)
        row_map = {row["example_key"]: row for row in rows}
        ordered_keys = None
        for layer, (matrix, example_keys) in sorted(matrices.items()):
            ordered_keys = example_keys
            per_layer_vectors.setdefault(layer, [])
            per_layer_vectors[layer].append(matrix)
        if ordered_keys is None:
            continue
        all_rows.extend(row_map[key] for key in ordered_keys)

    merged_matrices = {
        layer: np.concatenate(chunks, axis=0).astype(np.float32)
        for layer, chunks in sorted(per_layer_vectors.items())
    }
    return all_rows, merged_matrices


def _encode_labels(rows: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    class_names = sorted({row["prime_condition"] for row in rows})
    index = {name: idx for idx, name in enumerate(class_names)}
    return np.asarray([index[row["prime_condition"]] for row in rows], dtype=np.int64), class_names


def _class_support(labels: list[str]) -> dict[str, int]:
    support: dict[str, int] = {}
    for label in labels:
        support[label] = support.get(label, 0) + 1
    return dict(sorted(support.items()))


def _text_baseline(train_texts: list[str], train_y: np.ndarray, test_texts: list[str], test_y: np.ndarray) -> dict[str, float]:
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


def _probe_baseline(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray) -> dict[str, float]:
    model = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=2000,
            tol=1e-3,
            random_state=42,
        ),
    )
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    return {
        "accuracy": round(float(accuracy_score(test_y, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(test_y, preds)), 4),
    }


def _evaluate_split(
    rows: list[dict[str, Any]],
    matrices: dict[int, np.ndarray],
    *,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> dict[str, Any]:
    y, class_names = _encode_labels(rows)
    train_rows = [row for row, keep in zip(rows, train_mask.tolist(), strict=True) if keep]
    test_rows = [row for row, keep in zip(rows, test_mask.tolist(), strict=True) if keep]
    train_texts = [row["generated_text"] for row in train_rows]
    test_texts = [row["generated_text"] for row in test_rows]
    train_y = y[train_mask]
    test_y = y[test_mask]
    train_classes = sorted({class_names[int(idx)] for idx in train_y.tolist()})
    test_classes = sorted({class_names[int(idx)] for idx in test_y.tolist()})
    payload: dict[str, Any] = {
        "class_names": class_names,
        "train_example_count": int(train_mask.sum()),
        "test_example_count": int(test_mask.sum()),
        "train_class_names": train_classes,
        "test_class_names": test_classes,
        "train_prime_support": _class_support([row["prime_condition"] for row in train_rows]),
        "test_prime_support": _class_support([row["prime_condition"] for row in test_rows]),
    }
    if len(set(train_y.tolist())) < 2 or len(set(test_y.tolist())) < 2:
        payload["error"] = "Split does not contain at least two classes in both train and test"
        return payload

    text_result = _text_baseline(train_texts, train_y, test_texts, test_y)
    probe_by_layer: list[dict[str, Any]] = []
    for layer, X in sorted(matrices.items()):
        probe_result = _probe_baseline(X[train_mask], train_y, X[test_mask], test_y)
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
    best_layer = max(probe_by_layer, key=lambda item: item["balanced_accuracy"])
    payload.update({
        "text_baseline": text_result,
        "best_probe_layer": best_layer,
        "probe_by_layer": probe_by_layer,
    })
    return payload


def _family_pair_vocab_overlap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        by_key[(row["group_id"], row["prime_condition"], row["prime_family"])] = _content_tokens(row["generated_text"])
    families = sorted({row["prime_family"] for row in rows})
    overlaps: dict[str, list[float]] = {}
    for left in families:
        for right in families:
            if left >= right:
                continue
            pair_key = f"{left}__vs__{right}"
            values: list[float] = []
            for group_id in sorted({row["group_id"] for row in rows}):
                for prime_condition in sorted({row["prime_condition"] for row in rows}):
                    left_tokens = by_key.get((group_id, prime_condition, left))
                    right_tokens = by_key.get((group_id, prime_condition, right))
                    if not left_tokens or not right_tokens:
                        continue
                    union = left_tokens | right_tokens
                    if union:
                        values.append(len(left_tokens & right_tokens) / len(union))
            overlaps[pair_key] = values
    return {
        pair: {
            "pair_count": len(values),
            "mean_jaccard": round(float(np.mean(values)), 4) if values else None,
            "median_jaccard": round(float(np.median(values)), 4) if values else None,
        }
        for pair, values in sorted(overlaps.items())
    }


def run_analysis(
    *,
    description_capture_dataset_id: str,
    description_capture_id: str,
    name_capture_dataset_id: str,
    name_capture_id: str,
    alias_capture_dataset_id: str,
    alias_capture_id: str,
) -> dict[str, Any]:
    family_specs = [
        FamilySpec("description_only", description_capture_dataset_id, description_capture_id),
        FamilySpec("name_only", name_capture_dataset_id, name_capture_id),
        FamilySpec("alias_only", alias_capture_dataset_id, alias_capture_id),
    ]
    rows, matrices = _merge_family_data(family_specs)
    families = sorted({row["prime_family"] for row in rows})
    splits = np.asarray([row["split"] for row in rows], dtype=object)
    prime_families = np.asarray([row["prime_family"] for row in rows], dtype=object)

    within_family: dict[str, Any] = {}
    for family in families:
        mask = prime_families == family
        family_rows = [row for row, keep in zip(rows, mask.tolist(), strict=True) if keep]
        family_matrices = {layer: X[mask] for layer, X in matrices.items()}
        family_splits = splits[mask]
        within_family[family] = _evaluate_split(
            family_rows,
            family_matrices,
            train_mask=family_splits == "train",
            test_mask=family_splits == "test",
        )

    mixed_group_holdout = _evaluate_split(
        rows,
        matrices,
        train_mask=splits == "train",
        test_mask=splits == "test",
    )

    leave_one_family_out: dict[str, Any] = {}
    for heldout in families:
        leave_one_family_out[heldout] = _evaluate_split(
            rows,
            matrices,
            train_mask=prime_families != heldout,
            test_mask=prime_families == heldout,
        )

    return {
        "families": families,
        "row_count": len(rows),
        "vocabulary_overlap": _family_pair_vocab_overlap(rows),
        "within_family_group_holdout": within_family,
        "mixed_family_group_holdout": mixed_group_holdout,
        "leave_one_family_out": leave_one_family_out,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--description-capture-dataset-id", required=True)
    parser.add_argument("--description-capture-id", required=True)
    parser.add_argument("--name-capture-dataset-id", required=True)
    parser.add_argument("--name-capture-id", required=True)
    parser.add_argument("--alias-capture-dataset-id", required=True)
    parser.add_argument("--alias-capture-id", required=True)
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
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_DIR / "multifamily_analysis.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
