from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from pipelines.interp.local_capture import _artifact_basename_for_row
from projects.DX_TERMINAL.counterfactual.analysis import train_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run first-pass prompt-confusion mechanistic analysis on a filtered Phase 03 slice."
    )
    parser.add_argument(
        "--slice-json",
        type=Path,
        required=True,
        help="Path to a mechanistic slice JSON produced by build_mechanistic_slice.py.",
    )
    parser.add_argument(
        "--activations-dir",
        type=Path,
        required=True,
        help="Capture directory containing metadata.parquet and residual_stream/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("projects/DX_TERMINAL/prompt_confusion/phase_03/reports"),
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated captured layer numbers, or 'all'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def _parse_layers(raw: str, captured_layers: list[int]) -> list[int]:
    if raw.strip().lower() == "all":
        return list(captured_layers)
    requested = [int(part.strip()) for part in raw.split(",") if part.strip()]
    missing = sorted(set(requested) - set(captured_layers))
    if missing:
        raise ValueError(f"Requested layers not captured: {missing}")
    return requested


def _load_slice(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _flatten_slice_rows(slice_payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for pair in slice_payload.get("pairs", []):
        aligned = dict(pair["aligned"])
        strong = dict(pair["strong_conflict"])
        matched_pair_id = str(pair["matched_pair_id"])
        family = str(pair["strategy_family"])
        changed_output = bool(pair.get("changed_output"))

        aligned["matched_pair_id"] = matched_pair_id
        aligned["strategy_family"] = family
        aligned["member"] = "aligned"
        aligned["conflict_label"] = 0
        aligned["source_label"] = None
        aligned["changed_output"] = changed_output

        strong["matched_pair_id"] = matched_pair_id
        strong["strategy_family"] = family
        strong["member"] = "strong_conflict"
        strong["conflict_label"] = 1
        strong["source_label"] = (
            1 if strong.get("readout_side") == "setting" else 0 if strong.get("readout_side") == "strategy" else None
        )
        strong["changed_output"] = changed_output

        rows.extend([aligned, strong])
        pair_rows.append(
            {
                "matched_pair_id": matched_pair_id,
                "strategy_family": family,
                "changed_output": changed_output,
                "aligned": aligned,
                "strong_conflict": strong,
            }
        )
    return rows, pair_rows


def _load_metadata(activations_dir: Path) -> tuple[list[dict[str, Any]], list[int]]:
    meta_path = activations_dir / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    rows = pq.read_table(meta_path).to_pylist()
    captured_layers: list[int] | None = None
    for row in rows:
        raw = row.get("captured_layers")
        if raw:
            if isinstance(raw, str):
                captured_layers = [int(v) for v in json.loads(raw)]
            else:
                captured_layers = [int(v) for v in raw]
            break
    if captured_layers is None:
        num_layers = int(rows[0].get("num_layers_captured") or 0)
        captured_layers = list(range(num_layers))
    return [dict(row) for row in rows], captured_layers


def _join_rows_to_metadata(rows: list[dict[str, Any]], meta_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    meta_by_row_key: dict[str, dict[str, Any]] = {}
    for meta in meta_rows:
        row_key = meta.get("row_key")
        if row_key is not None and str(row_key).strip():
            meta_by_row_key[str(row_key)] = dict(meta)

    joined: list[dict[str, Any]] = []
    missing = 0
    for row in rows:
        row_key = row.get("row_key")
        if row_key is None:
            missing += 1
            continue
        meta = meta_by_row_key.get(str(row_key))
        if meta is None:
            missing += 1
            continue
        merged = {**row, **meta}
        joined.append(merged)
    return joined, missing


def _load_last_token_vectors(
    rows: list[dict[str, Any]],
    *,
    activations_dir: Path,
    layers: list[int],
    num_workers: int,
) -> dict[str, dict[int, np.ndarray]]:
    from safetensors.numpy import load_file

    residual_dir = activations_dir / "residual_stream"
    if not residual_dir.exists():
        raise FileNotFoundError(residual_dir)

    def _load_one(row: dict[str, Any]) -> tuple[str, dict[int, np.ndarray] | None]:
        artifact_id = row.get("artifact_id") or _artifact_basename_for_row(row)
        path = residual_dir / f"{artifact_id}.safetensors"
        if not path.exists():
            return str(row["example_id"]), None
        data = load_file(str(path))
        residual = data.get("residual_stream")
        if residual is None or residual.ndim != 3:
            return str(row["example_id"]), None
        captured_layers_raw = row.get("captured_layers")
        if isinstance(captured_layers_raw, str):
            captured_layers = [int(v) for v in json.loads(captured_layers_raw)]
        elif captured_layers_raw:
            captured_layers = [int(v) for v in captured_layers_raw]
        else:
            captured_layers = list(range(int(residual.shape[0])))
        layer_to_idx = {layer: idx for idx, layer in enumerate(captured_layers)}
        vectors: dict[int, np.ndarray] = {}
        for layer in layers:
            idx = layer_to_idx.get(layer)
            if idx is None:
                continue
            vectors[layer] = residual[idx, -1].astype(np.float32)
        return str(row["example_id"]), vectors

    results: dict[str, dict[int, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        futures = {pool.submit(_load_one, row): str(row["example_id"]) for row in rows}
        for future in as_completed(futures):
            example_id, vectors = future.result()
            if vectors is not None:
                results[example_id] = vectors
    return results


def _split_group_ids(group_ids: list[str], *, seed: int, test_fraction: float) -> tuple[set[str], set[str]]:
    ordered = sorted(set(group_ids))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n_test = max(1, int(len(ordered) * test_fraction))
    return set(ordered[n_test:]), set(ordered[:n_test])


def _binary_probe_metrics(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    probe = train_probe(X_train, y_train, seed=seed)
    pred = probe.predict(X_test)
    prob = probe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_positive_rate": float(np.mean(y_train)),
        "test_positive_rate": float(np.mean(y_test)),
    }
    if len(np.unique(y_test)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_test, prob))
    else:
        metrics["auroc"] = None
    return metrics


def _build_probe_dataset(
    rows: list[dict[str, Any]],
    activations: dict[str, dict[int, np.ndarray]],
    *,
    label_key: str,
) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    for row in rows:
        label = row.get(label_key)
        if label is None:
            continue
        vectors = activations.get(str(row["example_id"]))
        if not vectors:
            continue
        dataset.append(
            {
                "example_id": str(row["example_id"]),
                "matched_pair_id": str(row["matched_pair_id"]),
                "strategy_family": str(row["strategy_family"]),
                "environment_pressure_bucket": str(row["environment_pressure_bucket"]),
                "label": int(label),
                "vectors": vectors,
            }
        )
    return dataset


def _run_row_probe(
    dataset: list[dict[str, Any]],
    *,
    layers: list[int],
    seed: int,
    test_fraction: float,
) -> dict[str, Any]:
    group_ids = [str(row["matched_pair_id"]) for row in dataset]
    train_groups, test_groups = _split_group_ids(group_ids, seed=seed, test_fraction=test_fraction)
    results: list[dict[str, Any]] = []
    for layer in layers:
        train_rows = [row for row in dataset if row["matched_pair_id"] in train_groups and layer in row["vectors"]]
        test_rows = [row for row in dataset if row["matched_pair_id"] in test_groups and layer in row["vectors"]]
        if not train_rows or not test_rows:
            continue
        y_train = np.asarray([row["label"] for row in train_rows], dtype=np.int64)
        y_test = np.asarray([row["label"] for row in test_rows], dtype=np.int64)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        X_train = np.stack([row["vectors"][layer] for row in train_rows])
        X_test = np.stack([row["vectors"][layer] for row in test_rows])
        metrics = _binary_probe_metrics(X_train, y_train, X_test, y_test, seed=seed)
        metrics["layer"] = int(layer)
        results.append(metrics)
    best = max(results, key=lambda row: (row.get("balanced_accuracy") or -1.0, row.get("auroc") or -1.0), default=None)
    return {"per_layer": results, "best_layer": best}


def _build_delta_dataset(
    pair_rows: list[dict[str, Any]],
    activations: dict[str, dict[int, np.ndarray]],
) -> list[dict[str, Any]]:
    dataset: list[dict[str, Any]] = []
    for pair in pair_rows:
        strong = pair["strong_conflict"]
        label = strong.get("source_label")
        if label is None:
            continue
        aligned_vectors = activations.get(str(pair["aligned"]["example_id"]))
        strong_vectors = activations.get(str(strong["example_id"]))
        if not aligned_vectors or not strong_vectors:
            continue
        delta_vectors: dict[int, np.ndarray] = {}
        for layer, strong_vec in strong_vectors.items():
            aligned_vec = aligned_vectors.get(layer)
            if aligned_vec is None:
                continue
            delta_vectors[layer] = strong_vec - aligned_vec
        if not delta_vectors:
            continue
        dataset.append(
            {
                "matched_pair_id": str(pair["matched_pair_id"]),
                "strategy_family": str(pair["strategy_family"]),
                "environment_pressure_bucket": str(strong["environment_pressure_bucket"]),
                "label": int(label),
                "vectors": delta_vectors,
            }
        )
    return dataset


def _run_pca_summary(
    dataset: list[dict[str, Any]],
    *,
    layers: list[int],
) -> dict[str, Any]:
    from sklearn.decomposition import PCA

    per_layer: list[dict[str, Any]] = []
    for layer in layers:
        layer_rows = [row for row in dataset if layer in row["vectors"]]
        if len(layer_rows) < 4:
            continue
        X = np.stack([row["vectors"][layer] for row in layer_rows])
        y = np.asarray([row["label"] for row in layer_rows], dtype=np.int64)
        pca = PCA(n_components=min(5, X.shape[0], X.shape[1]))
        coords = pca.fit_transform(X)
        layer_result: dict[str, Any] = {
            "layer": int(layer),
            "n_rows": int(len(layer_rows)),
            "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_[:3]],
            "pc1_mean_by_label": {},
        }
        for label in sorted(set(int(v) for v in y.tolist())):
            mask = y == label
            layer_result["pc1_mean_by_label"][str(label)] = float(np.mean(coords[mask, 0]))
        if len(np.unique(y)) == 2:
            layer_result["pc1_gap"] = abs(
                layer_result["pc1_mean_by_label"]["1"] - layer_result["pc1_mean_by_label"]["0"]
            )
            centroid0 = np.mean(coords[y == 0, :2], axis=0)
            centroid1 = np.mean(coords[y == 1, :2], axis=0)
            layer_result["pc12_centroid_distance"] = float(np.linalg.norm(centroid1 - centroid0))
        per_layer.append(layer_result)
    best = max(
        per_layer,
        key=lambda row: (row.get("pc12_centroid_distance") or 0.0, row.get("pc1_gap") or 0.0),
        default=None,
    )
    return {"per_layer": per_layer, "best_layer": best}


def _family_breakdown(rows: list[dict[str, Any]], label_key: str) -> dict[str, Any]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        family = str(row["strategy_family"])
        label = row.get(label_key)
        grouped[family][str(label)] += 1
    return {family: dict(counter) for family, counter in grouped.items()}


def main() -> None:
    args = parse_args()
    slice_payload = _load_slice(args.slice_json)
    slice_rows, pair_rows = _flatten_slice_rows(slice_payload)

    meta_rows, captured_layers = _load_metadata(args.activations_dir)
    layers = _parse_layers(args.layers, captured_layers)
    joined_rows, missing_meta = _join_rows_to_metadata(slice_rows, meta_rows)
    activations = _load_last_token_vectors(
        joined_rows,
        activations_dir=args.activations_dir,
        layers=layers,
        num_workers=args.num_workers,
    )

    joined_rows = [row for row in joined_rows if str(row["example_id"]) in activations]
    pair_rows = [
        pair
        for pair in pair_rows
        if str(pair["aligned"]["example_id"]) in activations and str(pair["strong_conflict"]["example_id"]) in activations
    ]

    conflict_dataset = _build_probe_dataset(joined_rows, activations, label_key="conflict_label")
    source_dataset = _build_probe_dataset(
        [row for row in joined_rows if row["member"] == "strong_conflict"],
        activations,
        label_key="source_label",
    )
    delta_dataset = _build_delta_dataset(pair_rows, activations)

    conflict_probe = _run_row_probe(
        conflict_dataset,
        layers=layers,
        seed=args.seed,
        test_fraction=args.test_fraction,
    )
    source_probe = _run_row_probe(
        source_dataset,
        layers=layers,
        seed=args.seed,
        test_fraction=args.test_fraction,
    )
    delta_probe = _run_row_probe(
        delta_dataset,
        layers=layers,
        seed=args.seed,
        test_fraction=args.test_fraction,
    )

    conflict_pca = _run_pca_summary(conflict_dataset, layers=layers)
    source_pca = _run_pca_summary(source_dataset, layers=layers)
    delta_pca = _run_pca_summary(delta_dataset, layers=layers)

    result = {
        "slice_json": str(args.slice_json),
        "activations_dir": str(args.activations_dir),
        "layers": layers,
        "summary": {
            "slice_rows": len(slice_rows),
            "joined_rows": len(joined_rows),
            "matched_pairs": len(pair_rows),
            "missing_metadata_rows": int(missing_meta),
            "activation_rows": len(activations),
            "conflict_dataset_rows": len(conflict_dataset),
            "source_dataset_rows": len(source_dataset),
            "delta_dataset_rows": len(delta_dataset),
            "conflict_family_breakdown": _family_breakdown(conflict_dataset, "label"),
            "source_family_breakdown": _family_breakdown(source_dataset, "label"),
            "delta_family_breakdown": _family_breakdown(delta_dataset, "label"),
        },
        "probes": {
            "conflict_vs_non_conflict": conflict_probe,
            "source_strategy_vs_setting": source_probe,
            "pair_delta_source_strategy_vs_setting": delta_probe,
        },
        "pca": {
            "conflict_vs_non_conflict": conflict_pca,
            "source_strategy_vs_setting": source_pca,
            "pair_delta_source_strategy_vs_setting": delta_pca,
        },
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.slice_json.stem.replace("mechanistic_slice_", "mechanistic_analysis_")
    output_path = output_dir / f"{stem}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_path": str(output_path), "summary": result["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
