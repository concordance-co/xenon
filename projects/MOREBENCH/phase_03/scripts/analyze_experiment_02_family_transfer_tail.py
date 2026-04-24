from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipelines_v2.api import ModalVolumeStore, PostgresCatalog, PostgresSource, TransferPolicy
from pipelines_v2.storage.artifacts import CaptureArtifact, OperationArtifact, artifact_from_manifest


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
ARTIFACT_ROOT = "/data/artifacts/morebench_phase_03_experiment02"
ARTIFACT_STORE_NAME = "xenon-data"
FEATURE_NAME = "generated_sequence_residual"
REPORT_DIR = Path("projects/MOREBENCH/phase_03/reports/experiment_02_family_transfer_tail_analysis")


@dataclass(frozen=True, slots=True)
class FamilySpec:
    family: str
    capture_dataset_id: str | None
    capture_id: str
    capture_dataset_json: str | None = None


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
        "generated_text": str(labels.get("generated_text") or ""),
    }


def _tail_text(text: str, fraction: float) -> str:
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
        class_result = {
            "support": support,
            "recall": None,
            "auroc": None,
        }
        if support > 0:
            preds = (np.argmax(probs, axis=1) == class_index).astype(int)
            class_result["recall"] = round(float(recall_score(y_true, preds, zero_division=0)), 4)
        if 0 < support < len(y_true):
            auroc = float(roc_auc_score(y_true, probs[:, class_index]))
            class_result["auroc"] = round(auroc, 4)
            valid.append(auroc)
        per_class[class_name] = class_result
    macro = round(float(np.mean(valid)), 4) if valid else None
    return macro, per_class


def _metrics_from_predictions(
    *,
    preds: np.ndarray,
    probs: np.ndarray,
    test_y: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    macro_auroc, per_class = _macro_ovr_auroc(probs=probs, test_y=test_y, class_names=class_names)
    return {
        "accuracy": round(float(accuracy_score(test_y, preds)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(test_y, preds)), 4),
        "macro_ovr_auroc": macro_auroc,
        "per_class": per_class,
    }


def _fit_text_baseline(
    kind: str,
    train_texts: list[str],
    train_y: np.ndarray,
    test_texts: list[str],
    test_y: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    if kind == "count_logreg":
        model = make_pipeline(
            CountVectorizer(ngram_range=(1, 2), min_df=1),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=42),
        )
    elif kind == "tfidf_logreg":
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=42),
        )
    elif kind == "tfidf_knn":
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1),
            KNeighborsClassifier(n_neighbors=1, metric="cosine"),
        )
    else:
        raise ValueError(f"Unknown text baseline kind: {kind}")
    model.fit(train_texts, train_y)
    preds = model.predict(test_texts)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(test_texts)
    else:
        raise RuntimeError(f"Model {kind} does not expose predict_proba")
    return {
        "name": kind,
        **_metrics_from_predictions(preds=preds, probs=probs, test_y=test_y, class_names=class_names),
    }


def _probe_metrics(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=4000, random_state=42),
    )
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    probs = model.predict_proba(test_x)
    return _metrics_from_predictions(preds=preds, probs=probs, test_y=test_y, class_names=class_names)


def _load_family_tail_rows(
    spec: FamilySpec,
    *,
    tail_fraction: float,
) -> tuple[list[dict[str, Any]], dict[int, tuple[np.ndarray, list[str]]], dict[str, Any]]:
    capture_artifact = _load_capture_artifact(spec.capture_id)
    if spec.capture_dataset_json:
        dataset_payload = json.loads(Path(spec.capture_dataset_json).read_text(encoding="utf-8"))
    else:
        if not spec.capture_dataset_id:
            raise TypeError(f"Family {spec.family!r} requires either capture_dataset_id or capture_dataset_json")
        dataset_artifact = _load_operation_artifact(spec.capture_dataset_id)
        payload = dataset_artifact.result()
        dataset_payload = payload.get("dataset") if isinstance(payload, dict) else None
        if not isinstance(dataset_payload, dict):
            raise TypeError(f"Capture dataset artifact {spec.capture_dataset_id!r} missing serialized dataset")
    examples = list(dataset_payload.get("examples", []))
    rows = [_row_from_example(example, family_fallback=spec.family) for example in examples]
    row_keys = {row["example_key"] for row in rows}

    feature = capture_artifact.feature(FEATURE_NAME)
    feature_payload = feature.load()
    if feature_payload.get("kind") != "residual":
        raise TypeError(f"Expected residual feature payload for {spec.family}, got {feature_payload.get('kind')!r}")

    layers = sorted(int(layer) for layer in feature_payload["layers"])
    example_keys = sorted(feature_payload["layers"][str(layers[0])])
    if set(example_keys) != row_keys:
        missing = sorted(row_keys - set(example_keys))
        extra = sorted(set(example_keys) - row_keys)
        raise RuntimeError(
            f"Feature/row mismatch for {spec.family}: missing_feature={missing[:5]} extra_feature={extra[:5]}"
        )

    token_counts: list[int] = []
    kept_counts: list[int] = []
    matrices: dict[int, tuple[np.ndarray, list[str]]] = {}
    for layer in layers:
        layer_payload = feature_payload["layers"][str(layer)]
        pooled_rows: list[np.ndarray] = []
        for key in example_keys:
            record = dict(layer_payload[key])
            values = np.asarray(record["values"], dtype=np.float32)
            if values.ndim != 2:
                raise TypeError("Residual layer payload values must be rank-2")
            token_count = int(values.shape[0])
            start = max(0, min(token_count - 1, int(math.floor(token_count * (1.0 - tail_fraction)))))
            selected = values[start:]
            if selected.shape[0] <= 0:
                raise RuntimeError(f"Tail selection was empty for example {key!r} at layer {layer}")
            if layer == layers[0]:
                token_counts.append(token_count)
                kept_counts.append(int(selected.shape[0]))
            pooled_rows.append(selected.mean(axis=0).astype(np.float32))
        matrices[layer] = (np.stack(pooled_rows, axis=0).astype(np.float32), example_keys)

    tail_stats = {
        "token_count_summary": {
            "min": int(min(token_counts)) if token_counts else 0,
            "median": float(np.median(token_counts)) if token_counts else 0.0,
            "max": int(max(token_counts)) if token_counts else 0,
        },
        "tail_token_count_summary": {
            "min": int(min(kept_counts)) if kept_counts else 0,
            "median": float(np.median(kept_counts)) if kept_counts else 0.0,
            "max": int(max(kept_counts)) if kept_counts else 0,
        },
    }
    return rows, matrices, tail_stats


def _merge_family_tail_data(
    family_specs: list[FamilySpec],
    *,
    tail_fraction: float,
) -> tuple[list[dict[str, Any]], dict[int, np.ndarray], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    per_layer_vectors: dict[int, list[np.ndarray]] = {}
    tail_stats: dict[str, Any] = {}

    for spec in family_specs:
        rows, matrices, family_stats = _load_family_tail_rows(spec, tail_fraction=tail_fraction)
        row_map = {row["example_key"]: row for row in rows}
        ordered_keys = None
        for layer, (matrix, example_keys) in sorted(matrices.items()):
            ordered_keys = example_keys
            per_layer_vectors.setdefault(layer, [])
            per_layer_vectors[layer].append(matrix)
        if ordered_keys is None:
            continue
        ordered_rows = [row_map[key] for key in ordered_keys]
        for row in ordered_rows:
            row["tail_text"] = _tail_text(row["generated_text"], tail_fraction)
        all_rows.extend(ordered_rows)
        tail_stats[spec.family] = family_stats

    merged_matrices = {
        layer: np.concatenate(chunks, axis=0).astype(np.float32)
        for layer, chunks in sorted(per_layer_vectors.items())
    }
    return all_rows, merged_matrices, tail_stats


def _family_pair_vocab_overlap(rows: list[dict[str, Any]], *, text_key: str) -> dict[str, Any]:
    by_key: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        by_key[(row["group_id"], row["prime_condition"], row["prime_family"])] = _content_tokens(str(row[text_key]))
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
    train_texts = [row["tail_text"] for row in train_rows]
    test_texts = [row["tail_text"] for row in test_rows]
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

    text_results = [
        _fit_text_baseline(kind, train_texts, train_y, test_texts, test_y, class_names)
        for kind in ("count_logreg", "tfidf_logreg", "tfidf_knn")
    ]
    best_text_by_ba = max(text_results, key=lambda item: item["balanced_accuracy"])
    best_text_by_auroc = max(
        text_results,
        key=lambda item: float("-inf") if item["macro_ovr_auroc"] is None else item["macro_ovr_auroc"],
    )

    probe_by_layer: list[dict[str, Any]] = []
    for layer, X in sorted(matrices.items()):
        probe_result = _probe_metrics(X[train_mask], train_y, X[test_mask], test_y, class_names)
        probe_by_layer.append(
            {
                "layer": layer,
                **probe_result,
                "probe_minus_best_text_balanced_accuracy": round(
                    probe_result["balanced_accuracy"] - best_text_by_ba["balanced_accuracy"],
                    4,
                ),
                "probe_minus_best_text_macro_ovr_auroc": (
                    None
                    if probe_result["macro_ovr_auroc"] is None or best_text_by_auroc["macro_ovr_auroc"] is None
                    else round(probe_result["macro_ovr_auroc"] - best_text_by_auroc["macro_ovr_auroc"], 4)
                ),
            }
        )
    best_probe_layer = max(probe_by_layer, key=lambda item: item["balanced_accuracy"])
    payload.update(
        {
            "text_baselines": text_results,
            "best_text_baseline_by_balanced_accuracy": best_text_by_ba,
            "best_text_baseline_by_macro_ovr_auroc": best_text_by_auroc,
            "best_probe_layer": best_probe_layer,
            "probe_by_layer": probe_by_layer,
        }
    )
    return payload


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
    rows, matrices, tail_stats = _merge_family_tail_data(family_specs, tail_fraction=tail_fraction)
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
        "tail_fraction": tail_fraction,
        "families": families,
        "row_count": len(rows),
        "tail_stats": tail_stats,
        "full_text_vocabulary_overlap": _family_pair_vocab_overlap(rows, text_key="generated_text"),
        "tail_text_vocabulary_overlap": _family_pair_vocab_overlap(rows, text_key="tail_text"),
        "within_family_group_holdout": within_family,
        "mixed_family_group_holdout": mixed_group_holdout,
        "leave_one_family_out": leave_one_family_out,
    }


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
    parser.add_argument("--output", type=Path, default=REPORT_DIR / "multifamily_tail_analysis.json")
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
