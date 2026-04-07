"""Counterfactual structure analysis for pre/post market representations.

Focuses on Dataset B captures, where ACTIVE SETTINGS edits occur after the
market section. The goal is to measure:

1. Position-wise decodability of market labels before vs after settings.
2. How much downstream section states stay inside the pre-market subspace.

Outputs JSON summaries that are directly about representation timing:
whether a label is present in row-level market states, and whether settings
apply an in-subspace reweighting or an orthogonal policy shift downstream.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from projects.DX_TERMINAL.counterfactual.analysis import (
    evaluate_probe_per_snapshot,
    linear_cka,
    load_dataset_b_spec,
    preload_all_activations,
    run_probe_transfer,
)


@dataclass
class CounterfactualStructureConfig:
    activations_dir: Path = Path("data/activations/counterfactual")
    experiment_id: str = "default"
    output_dir: Path = Path("data/analysis_results/counterfactual_structure")
    train_variant: str = "settings_all1"
    compare_variant: str = "settings_all5"
    row_key: str = "row_mean"
    variance_threshold: float = 0.9
    layers: list[int] | None = None
    seed: int = 42
    downstream_positions: tuple[str, ...] = field(default_factory=lambda: (
        "settings_eos",
        "portfolio_eos",
        "constraints_eos",
        "prev_decisions_eos",
        "last_token",
    ))
    pre_positions: tuple[str, ...] = field(default_factory=lambda: (
        "row_mean",
        "row_eos",
    ))

    @property
    def run_dir(self) -> Path:
        return self.activations_dir / self.experiment_id

    @property
    def results_dir(self) -> Path:
        return self.output_dir / self.experiment_id


def _row_matrix(
    acts: dict[str, np.ndarray],
    *,
    n_rows: int,
    layer: int,
    row_key: str,
) -> np.ndarray | None:
    rows: list[np.ndarray] = []
    for i in range(n_rows):
        key = f"{row_key}_{i}"
        if key not in acts:
            return None
        rows.append(acts[key][layer].astype(np.float32))
    if not rows:
        return None
    return np.stack(rows, axis=0)


def collect_row_groups(
    snapshots: list[dict[str, Any]],
    split_ids: set[str],
    variant: str,
    label_name: str,
    layer: int,
    row_key: str,
    cache: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    """Collect grouped row-level examples for a row position."""
    groups: list[dict[str, Any]] = []
    for snap in snapshots:
        sid = snap["snapshot_id"]
        if sid not in split_ids:
            continue
        acts = cache.get(f"{sid}_{variant}", {})
        if not acts:
            continue

        n_rows = int(snap["n_rows"])
        X = _row_matrix(acts, n_rows=n_rows, layer=layer, row_key=row_key)
        if X is None:
            continue

        y = np.array(snap["labels"].get(label_name, [0] * n_rows), dtype=np.int64)
        groups.append({
            "X": X,
            "y": y,
            "snapshot_id": sid,
            "vault_day": f"{snap.get('vault_address', sid)[:10]}_{snap.get('snap_date', '')}",
        })
    return groups


def collect_concat_groups(
    snapshots: list[dict[str, Any]],
    split_ids: set[str],
    variant: str,
    label_name: str,
    layer: int,
    row_key: str,
    position_key: str,
    cache: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    """Collect row groups with a downstream position concatenated to each row."""
    groups: list[dict[str, Any]] = []
    for snap in snapshots:
        sid = snap["snapshot_id"]
        if sid not in split_ids:
            continue
        acts = cache.get(f"{sid}_{variant}", {})
        if not acts or position_key not in acts:
            continue

        n_rows = int(snap["n_rows"])
        row_X = _row_matrix(acts, n_rows=n_rows, layer=layer, row_key=row_key)
        if row_X is None:
            continue

        pos = acts[position_key][layer].astype(np.float32)
        pos_tiled = np.repeat(pos[None, :], n_rows, axis=0)
        X = np.concatenate([row_X, pos_tiled], axis=1)
        y = np.array(snap["labels"].get(label_name, [0] * n_rows), dtype=np.int64)
        groups.append({
            "X": X,
            "y": y,
            "snapshot_id": sid,
            "vault_day": f"{snap.get('vault_address', sid)[:10]}_{snap.get('snap_date', '')}",
        })
    return groups


def collect_market_state_matrix(
    snapshots: list[dict[str, Any]],
    variant: str,
    layer: int,
    row_key: str,
    cache: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, list[str]]:
    """Collect one market-state vector per snapshot by averaging row states."""
    mat: list[np.ndarray] = []
    ids: list[str] = []
    for snap in snapshots:
        sid = snap["snapshot_id"]
        acts = cache.get(f"{sid}_{variant}", {})
        if not acts:
            continue
        X = _row_matrix(acts, n_rows=int(snap["n_rows"]), layer=layer, row_key=row_key)
        if X is None:
            continue
        mat.append(X.mean(axis=0))
        ids.append(sid)
    if not mat:
        return np.empty((0, 0), dtype=np.float32), []
    return np.stack(mat, axis=0), ids


def collect_position_matrix(
    snapshots: list[dict[str, Any]],
    variant: str,
    layer: int,
    position_key: str,
    cache: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, list[str]]:
    """Collect one downstream position vector per snapshot."""
    mat: list[np.ndarray] = []
    ids: list[str] = []
    for snap in snapshots:
        sid = snap["snapshot_id"]
        acts = cache.get(f"{sid}_{variant}", {})
        if not acts or position_key not in acts:
            continue
        mat.append(acts[position_key][layer].astype(np.float32))
        ids.append(sid)
    if not mat:
        return np.empty((0, 0), dtype=np.float32), []
    return np.stack(mat, axis=0), ids


def _align_matrices_by_id(
    ids_a: list[str],
    X_a: np.ndarray,
    ids_b: list[str],
    X_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align two matrices by snapshot id order."""
    index_b = {sid: i for i, sid in enumerate(ids_b)}
    aligned_a: list[np.ndarray] = []
    aligned_b: list[np.ndarray] = []
    common_ids: list[str] = []
    for i, sid in enumerate(ids_a):
        j = index_b.get(sid)
        if j is None:
            continue
        aligned_a.append(X_a[i])
        aligned_b.append(X_b[j])
        common_ids.append(sid)
    if not aligned_a:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32), []
    return np.stack(aligned_a), np.stack(aligned_b), common_ids


def _align_three_matrices_by_id(
    ids_a: list[str],
    X_a: np.ndarray,
    ids_b: list[str],
    X_b: np.ndarray,
    ids_c: list[str],
    X_c: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Align three matrices onto a common snapshot-id order."""
    index_b = {sid: i for i, sid in enumerate(ids_b)}
    index_c = {sid: i for i, sid in enumerate(ids_c)}
    aligned_a: list[np.ndarray] = []
    aligned_b: list[np.ndarray] = []
    aligned_c: list[np.ndarray] = []
    common_ids: list[str] = []

    for i, sid in enumerate(ids_a):
        j = index_b.get(sid)
        k = index_c.get(sid)
        if j is None or k is None:
            continue
        aligned_a.append(X_a[i])
        aligned_b.append(X_b[j])
        aligned_c.append(X_c[k])
        common_ids.append(sid)

    if not aligned_a:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            [],
        )
    return (
        np.stack(aligned_a),
        np.stack(aligned_b),
        np.stack(aligned_c),
        common_ids,
    )


def fit_market_subspace(
    X: np.ndarray,
    variance_threshold: float = 0.9,
) -> dict[str, Any]:
    """Fit a PCA-style orthonormal basis capturing the market subspace."""
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("fit_market_subspace requires a non-empty 2D matrix")

    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    var = s ** 2
    total = float(var.sum())
    if total <= 1e-12:
        basis = np.eye(X.shape[1], 1, dtype=np.float32)
        return {
            "mean": mean.astype(np.float32),
            "basis": basis,
            "effective_dim": 1,
            "explained_variance": 0.0,
        }

    explained = np.cumsum(var) / total
    k = int(np.searchsorted(explained, variance_threshold, side="left")) + 1
    basis = vt[:k].T.astype(np.float32)
    return {
        "mean": mean.astype(np.float32),
        "basis": basis,
        "effective_dim": k,
        "explained_variance": float(explained[k - 1]),
    }


def subspace_energy_ratio(
    X: np.ndarray,
    basis: np.ndarray,
    mean: np.ndarray | None = None,
) -> float:
    """Fraction of energy in X captured by the given orthonormal basis."""
    if X.ndim != 2 or X.shape[0] == 0:
        return 0.0
    Xc = X - mean if mean is not None else X
    total = float(np.sum(Xc ** 2))
    if total <= 1e-12:
        return 0.0
    coeffs = Xc @ basis
    parallel = coeffs @ basis.T
    parallel_energy = float(np.sum(parallel ** 2))
    return parallel_energy / total


def analyze_position_subspace(
    market_a: np.ndarray,
    position_a: np.ndarray,
    position_b: np.ndarray,
    *,
    variance_threshold: float,
) -> dict[str, Any]:
    """Measure how downstream positions relate to the market subspace."""
    fitted = fit_market_subspace(market_a, variance_threshold=variance_threshold)
    mean = fitted["mean"]
    basis = fitted["basis"]

    result = {
        "effective_dim": fitted["effective_dim"],
        "explained_variance": fitted["explained_variance"],
        "position_a_subspace_ratio": subspace_energy_ratio(position_a, basis, mean=mean),
        "position_b_subspace_ratio": subspace_energy_ratio(position_b, basis, mean=mean),
        "delta_parallel_ratio": subspace_energy_ratio(position_b - position_a, basis, mean=None),
        "delta_orthogonal_ratio": 0.0,
        "mean_delta_norm": float(np.mean(np.linalg.norm(position_b - position_a, axis=1))),
        "n_snapshots": int(position_a.shape[0]),
    }
    result["delta_orthogonal_ratio"] = max(0.0, 1.0 - result["delta_parallel_ratio"])
    if position_a.shape[0] >= 2:
        result["position_cka_a"] = linear_cka(market_a, position_a)
        result["position_cka_b"] = linear_cka(market_a, position_b)
    else:
        result["position_cka_a"] = None
        result["position_cka_b"] = None
    return result


def _mean_metric(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _run_transfer_for_groups(
    train_groups: list[dict[str, Any]],
    within_groups: list[dict[str, Any]],
    transfer_groups: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    if not train_groups or not within_groups or not transfer_groups:
        return {"error": "insufficient_data"}
    y_train = np.concatenate([g["y"] for g in train_groups])
    if len(np.unique(y_train)) < 2:
        return {"error": "single_class_train"}

    result = run_probe_transfer(
        train_groups,
        within_groups,
        transfer_groups,
        seed=seed,
    )
    return {
        "within_auroc": _mean_metric(result["within_metrics"]["auroc"]),
        "transfer_auroc": _mean_metric(result["transfer_metrics"]["auroc"]),
        "within_hit1": _mean_metric(result["within_metrics"]["hit_at_1"]),
        "transfer_hit1": _mean_metric(result["transfer_metrics"]["hit_at_1"]),
        "within_mrr": _mean_metric(result["within_metrics"]["mrr"]),
        "transfer_mrr": _mean_metric(result["transfer_metrics"]["mrr"]),
        "within_balanced_accuracy": _mean_metric(result["within_metrics"]["balanced_accuracy"]),
        "transfer_balanced_accuracy": _mean_metric(result["transfer_metrics"]["balanced_accuracy"]),
        "transfer_gap": result["transfer_gap"],
    }


def run_position_probe_sweep(
    *,
    snapshots: list[dict[str, Any]],
    train_ids: set[str],
    test_ids: set[str],
    cache: dict[str, dict[str, np.ndarray]],
    layers: list[int],
    label_names: list[str],
    config: CounterfactualStructureConfig,
) -> dict[str, Any]:
    """Run within/transfer probes across pre and post representations."""
    results: dict[str, Any] = {"layers": layers, "labels": {}}

    for label_name in label_names:
        label_results: dict[str, list[dict[str, Any]]] = {}

        for row_key in config.pre_positions:
            per_layer: list[dict[str, Any]] = []
            for layer in layers:
                train_groups = collect_row_groups(
                    snapshots, train_ids, config.train_variant,
                    label_name, layer, row_key, cache,
                )
                within_groups = collect_row_groups(
                    snapshots, test_ids, config.train_variant,
                    label_name, layer, row_key, cache,
                )
                transfer_groups = collect_row_groups(
                    snapshots, test_ids, config.compare_variant,
                    label_name, layer, row_key, cache,
                )
                layer_result = _run_transfer_for_groups(
                    train_groups, within_groups, transfer_groups, seed=config.seed,
                )
                layer_result["layer"] = layer
                per_layer.append(layer_result)
            label_results[row_key] = per_layer

        for position_key in config.downstream_positions:
            rep_key = f"{config.row_key}+{position_key}"
            per_layer = []
            for layer in layers:
                train_groups = collect_concat_groups(
                    snapshots, train_ids, config.train_variant,
                    label_name, layer, config.row_key, position_key, cache,
                )
                within_groups = collect_concat_groups(
                    snapshots, test_ids, config.train_variant,
                    label_name, layer, config.row_key, position_key, cache,
                )
                transfer_groups = collect_concat_groups(
                    snapshots, test_ids, config.compare_variant,
                    label_name, layer, config.row_key, position_key, cache,
                )
                layer_result = _run_transfer_for_groups(
                    train_groups, within_groups, transfer_groups, seed=config.seed,
                )
                layer_result["layer"] = layer
                per_layer.append(layer_result)
            label_results[rep_key] = per_layer

        results["labels"][label_name] = label_results

    return results


def run_subspace_retention(
    *,
    snapshots: list[dict[str, Any]],
    cache: dict[str, dict[str, np.ndarray]],
    layers: list[int],
    config: CounterfactualStructureConfig,
) -> dict[str, Any]:
    """Measure market-subspace preservation across downstream positions."""
    results: dict[str, Any] = {
        "layers": layers,
        "row_key": config.row_key,
        "variance_threshold": config.variance_threshold,
        "positions": {},
    }

    for position_key in config.downstream_positions:
        per_layer: list[dict[str, Any]] = []
        for layer in layers:
            market_a, market_ids = collect_market_state_matrix(
                snapshots, config.train_variant, layer, config.row_key, cache,
            )
            pos_a, pos_a_ids = collect_position_matrix(
                snapshots, config.train_variant, layer, position_key, cache,
            )
            pos_b, pos_b_ids = collect_position_matrix(
                snapshots, config.compare_variant, layer, position_key, cache,
            )
            market_aligned, pos_a_aligned, pos_b_aligned, ids_final = _align_three_matrices_by_id(
                market_ids, market_a, pos_a_ids, pos_a, pos_b_ids, pos_b,
            )
            if not ids_final:
                per_layer.append({"layer": layer, "error": "insufficient_data"})
                continue

            stats = analyze_position_subspace(
                market_aligned,
                pos_a_aligned,
                pos_b_aligned,
                variance_threshold=config.variance_threshold,
            )
            stats["layer"] = layer
            per_layer.append(stats)
        results["positions"][position_key] = per_layer

    return results


def run_counterfactual_structure(
    config: CounterfactualStructureConfig,
) -> dict[str, Any]:
    """Run Dataset B pre/post structure analysis and write result JSON."""
    spec = load_dataset_b_spec()
    snapshots = spec["snapshots"]
    train_ids = spec["train_ids"]
    test_ids = spec["test_ids"]

    if not snapshots:
        return {"error": "dataset_b_not_found"}

    sample_id = f"{snapshots[0]['snapshot_id']}_{config.train_variant}"
    from projects.DX_TERMINAL.counterfactual.analysis import load_pooled_activations
    sample_acts = load_pooled_activations(config.run_dir, sample_id)
    if not sample_acts:
        return {"error": "no_activations_found"}

    for key in ("last_token", f"{config.row_key}_0"):
        if key in sample_acts:
            num_layers = sample_acts[key].shape[0]
            break
    else:
        return {"error": "no_position_keys_found"}

    layers = config.layers or list(range(num_layers))
    label_names = list(snapshots[0].get("labels", {}).keys())

    capture_ids: list[str] = []
    for snap in snapshots:
        sid = snap["snapshot_id"]
        capture_ids.append(f"{sid}_{config.train_variant}")
        capture_ids.append(f"{sid}_{config.compare_variant}")
    cache = preload_all_activations(config.run_dir, capture_ids)

    results = {
        "probe_sweep": run_position_probe_sweep(
            snapshots=snapshots,
            train_ids=train_ids,
            test_ids=test_ids,
            cache=cache,
            layers=layers,
            label_names=label_names,
            config=config,
        ),
        "subspace_retention": run_subspace_retention(
            snapshots=snapshots,
            cache=cache,
            layers=layers,
            config=config,
        ),
    }

    config.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = config.results_dir / "structure_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    return results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Counterfactual structure analysis")
    p.add_argument("--activations-dir", type=Path, default=Path("data/activations/counterfactual"))
    p.add_argument("--experiment-id", default="default")
    p.add_argument("--output-dir", type=Path, default=Path("data/analysis_results/counterfactual_structure"))
    p.add_argument("--train-variant", default="settings_all1")
    p.add_argument("--compare-variant", default="settings_all5")
    p.add_argument("--row-key", choices=["row_mean", "row_eos"], default="row_mean")
    p.add_argument("--variance-threshold", type=float, default=0.9)
    p.add_argument("--layers", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    parsed_layers: list[int] | None = None
    if args.layers:
        parsed_layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    config = CounterfactualStructureConfig(
        activations_dir=args.activations_dir,
        experiment_id=args.experiment_id,
        output_dir=args.output_dir,
        train_variant=args.train_variant,
        compare_variant=args.compare_variant,
        row_key=args.row_key,
        variance_threshold=args.variance_threshold,
        layers=parsed_layers,
        seed=args.seed,
    )
    results = run_counterfactual_structure(config)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
