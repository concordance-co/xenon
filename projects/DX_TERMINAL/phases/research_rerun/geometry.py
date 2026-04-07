from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from pipelines.db import connect_neon
from projects.DX_TERMINAL.phases.counterfactual.analysis import orthogonal_procrustes, preload_all_activations
from projects.DX_TERMINAL.synthetic_market.shared.synthetic_manifold_analysis import _evaluate_regression_probe, _train_regression_probe


RISK_CONTEXTS = tuple(f"risk_{level}" for level in range(1, 6))
BASE_CONTEXT = "risk_3"
ROW_KEYS = ("row_mean", "row_eos")


@dataclass
class ResearchRiskGeometryConfig:
    research_activations_dir: Path = Path("data/activations/research_rerun")
    output_dir: Path = Path("data/analysis_results/research_risk_geometry")
    experiment_id: str = "real_risk_geometry_bridge_v1"
    seed: int = 42
    test_fraction: float = 0.2
    num_workers: int = 8

    @property
    def run_dir(self) -> Path:
        return self.research_activations_dir / self.experiment_id

    @property
    def results_dir(self) -> Path:
        return self.output_dir / self.experiment_id


def _matrix_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0.0:
        return None
    return float(np.dot(av, bv) / denom)


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    if av.size == 0 or bv.size == 0 or av.size != bv.size:
        return None
    if np.allclose(av, av[0]) or np.allclose(bv, bv[0]):
        return None
    corr = spearmanr(av, bv).correlation
    if corr is None or np.isnan(corr):
        return None
    return float(corr)


def _pairwise_distance_vector(coords: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for left_idx in range(coords.shape[0]):
        for right_idx in range(left_idx + 1, coords.shape[0]):
            values.append(float(np.linalg.norm(coords[left_idx] - coords[right_idx])))
    return np.asarray(values, dtype=np.float32)


def _center_rows(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


def _fit_identity(_: np.ndarray, __: np.ndarray) -> np.ndarray:
    return np.eye(2, dtype=np.float32)


def _fit_orthogonal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return orthogonal_procrustes(x, y).astype(np.float32)


def _fit_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    r = orthogonal_procrustes(x, y)
    xr = x @ r
    denom = float(np.sum(xr * xr))
    scale = 1.0 if denom <= 1e-12 else float(np.sum(xr * y) / denom)
    return (scale * r).astype(np.float32)


def _fit_diagonal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    scales = []
    for axis in range(x.shape[1]):
        denom = float(np.dot(x[:, axis], x[:, axis]))
        scales.append(1.0 if denom <= 1e-12 else float(np.dot(x[:, axis], y[:, axis]) / denom))
    return np.diag(np.asarray(scales, dtype=np.float32))


def _fit_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    matrix, *_ = np.linalg.lstsq(x, y, rcond=None)
    return np.asarray(matrix, dtype=np.float32)


TRANSFORM_FAMILIES: dict[str, Any] = {
    "identity": _fit_identity,
    "orthogonal": _fit_orthogonal,
    "similarity": _fit_similarity,
    "diagonal": _fit_diagonal,
    "linear": _fit_linear,
}


def _polar_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    return u @ vt


def _matrix_summary(matrix: np.ndarray) -> dict[str, Any]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rotation = _polar_rotation(matrix)
    angle = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    svals = np.sort(np.asarray(singular_values, dtype=np.float64))[::-1]
    anisotropy = None
    if len(svals) >= 2 and abs(svals[1]) > 1e-12:
        anisotropy = float(svals[0] / svals[1])
    return {
        "matrix": matrix.tolist(),
        "determinant": float(np.linalg.det(matrix)),
        "rotation_angle_deg": angle,
        "singular_values": [float(v) for v in svals.tolist()],
        "anisotropy_ratio": anisotropy,
    }


def _load_risk_prompt_rows(experiment_id: str) -> list[dict[str, Any]]:
    conn = connect_neon()
    try:
        rows = conn.execute(
            """
            SELECT
                p.prompt_id AS capture_id,
                p.base_example_id,
                p.variant,
                p.n_rows,
                p.row_order,
                p.metadata
            FROM research_rerun_prompts p
            WHERE p.experiment_id = %s
              AND p.experiment_group = 'risk_geometry'
            ORDER BY p.base_example_id, p.variant
            """,
            [experiment_id],
        ).fetchall()
    finally:
        conn.close()
    parsed: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        metadata = record.get("metadata")
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        record["metadata"] = metadata or {}
        parsed.append(record)
    return parsed


def _risk_level(context_variant: str) -> int | None:
    if not str(context_variant).startswith("risk_"):
        return None
    try:
        value = int(str(context_variant).split("_", 1)[1])
    except ValueError:
        return None
    if 1 <= value <= 5:
        return value
    return None


def _ordered_contexts(prompt_rows: list[dict[str, Any]]) -> list[str]:
    available = sorted({str(row["variant"]) for row in prompt_rows if _risk_level(str(row["variant"])) is not None})
    return sorted(available, key=lambda name: int(_risk_level(name) or 0))


def _transfer_pairs(contexts: list[str]) -> list[tuple[str, str]]:
    pairs = [(BASE_CONTEXT, BASE_CONTEXT)]
    for context in contexts:
        if context == BASE_CONTEXT:
            continue
        pairs.append((BASE_CONTEXT, context))
    return pairs


def _deformation_pairs(contexts: list[str]) -> list[tuple[str, str]]:
    ordered = list(contexts)
    if len(ordered) < 2:
        return []
    pairs = [(ordered[idx], ordered[idx + 1]) for idx in range(len(ordered) - 1)]
    if ordered[0] != ordered[-1]:
        pairs.append((ordered[0], ordered[-1]))
    return pairs


def _split_example_ids(example_ids: list[str], *, seed: int, test_fraction: float) -> tuple[set[str], set[str]]:
    unique_ids = sorted({str(example_id) for example_id in example_ids})
    if not unique_ids:
        return set(), set()
    if len(unique_ids) == 1:
        return {unique_ids[0]}, {unique_ids[0]}
    rng = np.random.default_rng(seed)
    shuffled = list(unique_ids)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_fraction)))
    n_test = min(len(shuffled) - 1, n_test)
    test_ids = set(shuffled[:n_test])
    train_ids = set(shuffled[n_test:])
    return train_ids, test_ids


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _selected_prompt_rows(prompt_row: dict[str, Any]) -> list[tuple[str, int]]:
    metadata = prompt_row.get("metadata") or {}
    symbols = list(metadata.get("selected_symbols") or [])
    indices = list(metadata.get("selected_row_indices") or [])
    if len(symbols) != len(indices):
        return []
    return [(str(symbol), int(idx)) for symbol, idx in zip(symbols, indices, strict=False)]


def _collect_coordinate_rows(
    *,
    prompt_rows: list[dict[str, Any]],
    activation_cache: dict[str, dict[str, np.ndarray]],
    example_ids: set[str],
    row_key: str,
    layer: int,
    axis_index: int,
    context_variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    for row in prompt_rows:
        if str(row["variant"]) != context_variant:
            continue
        if str(row["base_example_id"]) not in example_ids:
            continue
        acts = activation_cache.get(str(row["capture_id"]))
        if not acts:
            continue
        metadata = row.get("metadata") or {}
        base_coords = metadata.get("base_coords") or {}
        for symbol, row_index in _selected_prompt_rows(row):
            key = f"{row_key}_{row_index}"
            coords = base_coords.get(symbol)
            if key not in acts or coords is None or len(coords) <= axis_index:
                continue
            X_rows.append(acts[key][layer].astype(np.float32))
            y_rows.append(float(coords[axis_index]))
    if not X_rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.stack(X_rows), np.asarray(y_rows, dtype=np.float32)


def _decode_examples(
    *,
    prompt_rows: list[dict[str, Any]],
    activation_cache: dict[str, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
    x_probe: Any,
    y_probe: Any,
) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    for row in prompt_rows:
        acts = activation_cache.get(str(row["capture_id"]))
        if not acts:
            continue
        metadata = row.get("metadata") or {}
        selected = _selected_prompt_rows(row)
        if len(selected) < 4:
            continue
        base_coords = metadata.get("base_coords") or {}
        score_coords = metadata.get("score_coords") or {}
        decoded_coords: list[list[float]] = []
        base_points: list[list[float]] = []
        score_points: list[list[float]] = []
        symbols: list[str] = []
        for symbol, row_index in selected:
            key = f"{row_key}_{row_index}"
            if key not in acts or symbol not in base_coords or symbol not in score_coords:
                decoded_coords = []
                break
            vec = acts[key][layer].astype(np.float32).reshape(1, -1)
            x_hat = float(x_probe.predict(vec)[0])
            y_hat = float(y_probe.predict(vec)[0])
            decoded_coords.append([x_hat, y_hat])
            base_points.append([float(base_coords[symbol][0]), float(base_coords[symbol][1])])
            score_points.append([float(score_coords[symbol][0]), float(score_coords[symbol][1])])
            symbols.append(symbol)
        if len(decoded_coords) != len(selected):
            continue
        decoded_arr = _center_rows(np.asarray(decoded_coords, dtype=np.float32))
        base_arr = _center_rows(np.asarray(base_points, dtype=np.float32))
        score_arr = _center_rows(np.asarray(score_points, dtype=np.float32))
        decoded.append(
            {
                "base_example_id": str(row["base_example_id"]),
                "context_variant": str(row["variant"]),
                "symbols": symbols,
                "decoded_centered": decoded_arr,
                "base_centered": base_arr,
                "score_centered": score_arr,
                "geometry_vec": _pairwise_distance_vector(decoded_arr),
                "base_geometry_vec": _pairwise_distance_vector(base_arr),
                "score_geometry_vec": _pairwise_distance_vector(score_arr),
            }
        )
    return decoded


def _context_realignment_metrics(examples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(examples) < 2:
        return {"error": "insufficient_examples"}
    base_spearmans: list[float] = []
    score_spearmans: list[float] = []
    for example in examples:
        act_vec = np.asarray(example["geometry_vec"], dtype=np.float32)
        base_vec = np.asarray(example["base_geometry_vec"], dtype=np.float32)
        score_vec = np.asarray(example["score_geometry_vec"], dtype=np.float32)
        base_corr = _safe_spearman(base_vec, act_vec)
        if base_corr is not None:
            base_spearmans.append(base_corr)
        score_corr = _safe_spearman(score_vec, act_vec)
        if score_corr is not None:
            score_spearmans.append(score_corr)
    return {
        "n_examples": len(examples),
        "base_distance_spearman_mean": _mean(base_spearmans),
        "score_distance_spearman_mean": _mean(score_spearmans),
        "score_over_base_margin": None
        if not base_spearmans or not score_spearmans
        else float(np.mean(score_spearmans) - np.mean(base_spearmans)),
    }


def _context_deformation_metrics(
    examples: list[dict[str, Any]],
    *,
    source_context: str,
    target_context: str,
) -> dict[str, Any]:
    by_key = {
        (str(example["base_example_id"]), str(example["context_variant"])): example
        for example in examples
    }
    spearmans: list[float] = []
    cosines: list[float] = []
    activation_norms: list[float] = []
    score_norms: list[float] = []
    paired = 0
    for example_id in sorted({str(example["base_example_id"]) for example in examples}):
        source = by_key.get((example_id, source_context))
        target = by_key.get((example_id, target_context))
        if source is None or target is None:
            continue
        act_delta = np.asarray(target["geometry_vec"], dtype=np.float32) - np.asarray(source["geometry_vec"], dtype=np.float32)
        score_delta = np.asarray(target["score_geometry_vec"], dtype=np.float32) - np.asarray(source["score_geometry_vec"], dtype=np.float32)
        paired += 1
        activation_norms.append(float(np.linalg.norm(act_delta)))
        score_norms.append(float(np.linalg.norm(score_delta)))
        corr = _safe_spearman(score_delta, act_delta)
        if corr is not None:
            spearmans.append(corr)
        denom = float(np.linalg.norm(score_delta) * np.linalg.norm(act_delta))
        if denom > 1e-12:
            cosines.append(float(np.dot(score_delta, act_delta) / denom))
    return {
        "n_examples": paired,
        "deformation_spearman_mean": _mean(spearmans),
        "deformation_cosine_mean": _mean(cosines),
        "activation_delta_norm_mean": _mean(activation_norms),
        "score_delta_norm_mean": _mean(score_norms),
    }


def _summarize_context_transfer(results: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for target_name, by_transfer in results.items():
        summary[target_name] = {}
        for transfer_key, per_row_key in by_transfer.items():
            best: tuple[str, float, int] | None = None
            for row_key, per_layer in per_row_key.items():
                for metrics in per_layer:
                    score = metrics.get("r2")
                    if score is None:
                        continue
                    if best is None or float(score) > best[1]:
                        best = (row_key, float(score), int(metrics["layer"]))
            summary[target_name][transfer_key] = None if best is None else {
                "representation": best[0],
                "r2": best[1],
                "layer": best[2],
            }
    return summary


def _summarize_best_metric(results: dict[str, Any], *, margin_key: str, acc_key: str) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for context_name, per_row_key in results.items():
        best: tuple[str, float, int, float | None] | None = None
        for row_key, per_layer in per_row_key.items():
            for metrics in per_layer:
                score = metrics.get(margin_key)
                if score is None:
                    continue
                acc = metrics.get(acc_key)
                if best is None or float(score) > best[1]:
                    best = (
                        row_key,
                        float(score),
                        int(metrics["layer"]),
                        None if acc is None else float(acc),
                    )
        summary[context_name] = None if best is None else {
            "representation": best[0],
            "margin": best[1],
            "layer": best[2],
            "nn_accuracy": best[3],
        }
    return summary


def _select_states(results: dict[str, Any], contexts: list[str]) -> dict[str, dict[str, Any]]:
    transfer = results["context_transfer"]
    transfer_keys = [f"{left}_to_{right}" for left, right in _transfer_pairs(contexts)]
    best_early: tuple[str, int, float] | None = None
    for row_key in ROW_KEYS:
        sample_key = transfer_keys[0]
        sample = transfer["base_x"].get(sample_key, {}).get(row_key, [])
        for layer_idx in range(len(sample)):
            values: list[float] = []
            for transfer_key in transfer_keys:
                for target_name in ("base_x", "base_y"):
                    metric = transfer[target_name][transfer_key][row_key][layer_idx].get("r2")
                    if metric is not None:
                        values.append(float(metric))
            if values:
                score = float(np.mean(values))
                if best_early is None or score > best_early[2]:
                    best_early = (row_key, layer_idx, score)

    best_late: tuple[str, int, float] | None = None
    realignment = results["context_realignment"]
    for row_key in ROW_KEYS:
        sample = realignment[contexts[0]].get(row_key, [])
        for layer_idx in range(len(sample)):
            values = []
            for context in contexts:
                margin = realignment[context][row_key][layer_idx].get("score_over_base_margin")
                if margin is not None:
                    values.append(float(margin))
            if values:
                score = float(np.mean(values))
                if best_late is None or score > best_late[2]:
                    best_late = (row_key, layer_idx, score)

    return {
        "early": {
            "row_key": best_early[0],
            "layer": best_early[1],
            "selection_score": best_early[2],
        } if best_early else {"error": "insufficient_transfer"},
        "late": {
            "row_key": best_late[0],
            "layer": best_late[1],
            "selection_score": best_late[2],
        } if best_late else {"error": "insufficient_realignment"},
    }


def _stack_points(pairs: list[tuple[dict[str, Any], dict[str, Any]]], *, key: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for source, target in pairs:
        xs.append(np.asarray(source[key], dtype=np.float32))
        ys.append(np.asarray(target[key], dtype=np.float32))
    if not xs:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _collect_pair_entries(
    examples: list[dict[str, Any]],
    *,
    source_context: str,
    target_context: str,
    example_ids: set[str] | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_key = {
        (str(example["base_example_id"]), str(example["context_variant"])): example
        for example in examples
    }
    ids = sorted({str(example["base_example_id"]) for example in examples})
    if example_ids is not None:
        ids = [example_id for example_id in ids if example_id in example_ids]
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for example_id in ids:
        source = by_key.get((example_id, source_context))
        target = by_key.get((example_id, target_context))
        if source is None or target is None:
            continue
        pairs.append((source, target))
    return pairs


def _r2_score_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return None
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - ss_res / ss_tot)


def _evaluate_transform(pairs: list[tuple[dict[str, Any], dict[str, Any]]], matrix: np.ndarray) -> dict[str, Any]:
    if not pairs:
        return {"error": "insufficient_pairs"}
    x_points, y_points = _stack_points(pairs, key="decoded_centered")
    y_hat = x_points @ matrix
    r2_x = _r2_score_1d(y_points[:, 0], y_hat[:, 0])
    r2_y = _r2_score_1d(y_points[:, 1], y_hat[:, 1])
    spearmans: list[float] = []
    cosines: list[float] = []
    target_score_spearmans: list[float] = []
    base_latent_spearmans: list[float] = []
    for source, target in pairs:
        pred = np.asarray(source["decoded_centered"], dtype=np.float32) @ matrix
        pred_vec = _pairwise_distance_vector(pred)
        target_vec = np.asarray(target["geometry_vec"], dtype=np.float32)
        score_vec = np.asarray(target["score_geometry_vec"], dtype=np.float32)
        base_vec = np.asarray(target["base_geometry_vec"], dtype=np.float32)
        corr = spearmanr(pred_vec, target_vec).correlation
        if corr is not None and not np.isnan(corr):
            spearmans.append(float(corr))
        denom = float(np.linalg.norm(pred_vec) * np.linalg.norm(target_vec))
        if denom > 1e-12:
            cosines.append(float(np.dot(pred_vec, target_vec) / denom))
        score_corr = spearmanr(pred_vec, score_vec).correlation
        if score_corr is not None and not np.isnan(score_corr):
            target_score_spearmans.append(float(score_corr))
        base_corr = spearmanr(pred_vec, base_vec).correlation
        if base_corr is not None and not np.isnan(base_corr):
            base_latent_spearmans.append(float(base_corr))
    return {
        "n_examples": len(pairs),
        "coord_r2_x": r2_x,
        "coord_r2_y": r2_y,
        "coord_r2_mean": _mean([value for value in (r2_x, r2_y) if value is not None]),
        "distance_spearman_mean": _mean(spearmans),
        "distance_cosine_mean": _mean(cosines),
        "score_distance_spearman_mean": _mean(target_score_spearmans),
        "latent_distance_spearman_mean": _mean(base_latent_spearmans),
    }


def _fit_and_evaluate_family(
    *,
    train_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    test_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    family: str,
) -> dict[str, Any]:
    x_train, y_train = _stack_points(train_pairs, key="decoded_centered")
    if x_train.size == 0 or y_train.size == 0:
        return {"error": "insufficient_train_pairs"}
    matrix = TRANSFORM_FAMILIES[family](x_train, y_train)
    metrics = _evaluate_transform(test_pairs, matrix)
    metrics.update(_matrix_summary(matrix))
    metrics["family"] = family
    return metrics


def _compose_matrices(matrices: list[np.ndarray]) -> np.ndarray:
    result = np.eye(2, dtype=np.float32)
    for matrix in matrices:
        result = result @ matrix
    return result


def run_research_risk_geometry_analysis(config: ResearchRiskGeometryConfig) -> dict[str, Any]:
    prompt_rows = _load_risk_prompt_rows(config.experiment_id)
    if not prompt_rows:
        return {"error": "no_risk_geometry_prompts"}
    contexts = _ordered_contexts(prompt_rows)
    if len(contexts) < 5:
        return {"error": "incomplete_risk_ladder", "contexts": contexts}

    capture_ids = [str(row["capture_id"]) for row in prompt_rows]
    activation_cache = preload_all_activations(config.run_dir, capture_ids, max_workers=config.num_workers)
    if not activation_cache:
        return {"error": "no_research_rerun_activations"}

    sample_acts = next(iter(activation_cache.values()))
    layers = list(range(int(sample_acts["last_token"].shape[0])))
    example_ids = sorted({str(row["base_example_id"]) for row in prompt_rows})
    train_example_ids, test_example_ids = _split_example_ids(example_ids, seed=config.seed, test_fraction=config.test_fraction)

    results: dict[str, Any] = {
        "experiment_id": config.experiment_id,
        "contexts": contexts,
        "base_context": BASE_CONTEXT,
        "n_examples": len(example_ids),
        "layers": layers,
        "row_keys": list(ROW_KEYS),
        "context_transfer": {"base_x": {}, "base_y": {}},
        "context_realignment": {},
        "context_deformation": {},
    }

    for target_name in ("base_x", "base_y"):
        axis_index = 0 if target_name == "base_x" else 1
        for source_context, target_context in _transfer_pairs(contexts):
            transfer_key = f"{source_context}_to_{target_context}"
            results["context_transfer"][target_name][transfer_key] = {}
            for row_key in ROW_KEYS:
                per_layer: list[dict[str, Any]] = []
                for layer in layers:
                    X_train, y_train = _collect_coordinate_rows(
                        prompt_rows=prompt_rows,
                        activation_cache=activation_cache,
                        example_ids=train_example_ids,
                        row_key=row_key,
                        layer=layer,
                        axis_index=axis_index,
                        context_variant=source_context,
                    )
                    X_test, y_test = _collect_coordinate_rows(
                        prompt_rows=prompt_rows,
                        activation_cache=activation_cache,
                        example_ids=test_example_ids,
                        row_key=row_key,
                        layer=layer,
                        axis_index=axis_index,
                        context_variant=target_context,
                    )
                    if X_train.size == 0 or X_test.size == 0:
                        per_layer.append({"layer": layer, "error": "insufficient_rows"})
                        continue
                    probe = _train_regression_probe(X_train, y_train)
                    metrics = _evaluate_regression_probe(probe, X_test, y_test)
                    per_layer.append({"layer": layer, **metrics})
                results["context_transfer"][target_name][transfer_key][row_key] = per_layer

    for context in contexts:
        results["context_realignment"][context] = {}
        for row_key in ROW_KEYS:
            per_layer: list[dict[str, Any]] = []
            for layer in layers:
                X_train, y_x_train = _collect_coordinate_rows(
                    prompt_rows=prompt_rows,
                    activation_cache=activation_cache,
                    example_ids=train_example_ids,
                    row_key=row_key,
                    layer=layer,
                    axis_index=0,
                    context_variant=BASE_CONTEXT,
                )
                _, y_y_train = _collect_coordinate_rows(
                    prompt_rows=prompt_rows,
                    activation_cache=activation_cache,
                    example_ids=train_example_ids,
                    row_key=row_key,
                    layer=layer,
                    axis_index=1,
                    context_variant=BASE_CONTEXT,
                )
                if X_train.size == 0:
                    per_layer.append({"layer": layer, "error": "insufficient_coordinate_rows"})
                    continue
                x_probe = _train_regression_probe(X_train, y_x_train)
                y_probe = _train_regression_probe(X_train, y_y_train)
                decoded = _decode_examples(
                    prompt_rows=[row for row in prompt_rows if str(row["variant"]) == context],
                    activation_cache=activation_cache,
                    row_key=row_key,
                    layer=layer,
                    x_probe=x_probe,
                    y_probe=y_probe,
                )
                metrics = _context_realignment_metrics(decoded)
                per_layer.append({"layer": layer, **metrics})
            results["context_realignment"][context][row_key] = per_layer

    for source_context, target_context in _deformation_pairs(contexts):
        pair_key = f"{source_context}_to_{target_context}"
        results["context_deformation"][pair_key] = {}
        for row_key in ROW_KEYS:
            per_layer: list[dict[str, Any]] = []
            for layer in layers:
                X_train, y_x_train = _collect_coordinate_rows(
                    prompt_rows=prompt_rows,
                    activation_cache=activation_cache,
                    example_ids=train_example_ids,
                    row_key=row_key,
                    layer=layer,
                    axis_index=0,
                    context_variant=BASE_CONTEXT,
                )
                _, y_y_train = _collect_coordinate_rows(
                    prompt_rows=prompt_rows,
                    activation_cache=activation_cache,
                    example_ids=train_example_ids,
                    row_key=row_key,
                    layer=layer,
                    axis_index=1,
                    context_variant=BASE_CONTEXT,
                )
                if X_train.size == 0:
                    per_layer.append({"layer": layer, "error": "insufficient_coordinate_rows"})
                    continue
                x_probe = _train_regression_probe(X_train, y_x_train)
                y_probe = _train_regression_probe(X_train, y_y_train)
                decoded = _decode_examples(
                    prompt_rows=prompt_rows,
                    activation_cache=activation_cache,
                    row_key=row_key,
                    layer=layer,
                    x_probe=x_probe,
                    y_probe=y_probe,
                )
                metrics = _context_deformation_metrics(
                    decoded,
                    source_context=source_context,
                    target_context=target_context,
                )
                per_layer.append({"layer": layer, **metrics})
            results["context_deformation"][pair_key][row_key] = per_layer

    results["summary"] = {
        "context_transfer": _summarize_context_transfer(results["context_transfer"]),
        "context_realignment": _summarize_best_metric(
            results["context_realignment"],
            margin_key="score_over_base_margin",
            acc_key="score_distance_spearman_mean",
        ),
        "context_deformation": _summarize_best_metric(
            results["context_deformation"],
            margin_key="deformation_spearman_mean",
            acc_key="deformation_cosine_mean",
        ),
    }

    selected_states = _select_states(results, contexts)
    results["selected_states"] = selected_states
    results["states"] = {}

    if all("error" not in state for state in selected_states.values()):
        for state_name, state in selected_states.items():
            row_key = str(state["row_key"])
            layer = int(state["layer"])
            X_train, y_x_train = _collect_coordinate_rows(
                prompt_rows=prompt_rows,
                activation_cache=activation_cache,
                example_ids=train_example_ids,
                row_key=row_key,
                layer=layer,
                axis_index=0,
                context_variant=BASE_CONTEXT,
            )
            _, y_y_train = _collect_coordinate_rows(
                prompt_rows=prompt_rows,
                activation_cache=activation_cache,
                example_ids=train_example_ids,
                row_key=row_key,
                layer=layer,
                axis_index=1,
                context_variant=BASE_CONTEXT,
            )
            X_test, y_x_test = _collect_coordinate_rows(
                prompt_rows=prompt_rows,
                activation_cache=activation_cache,
                example_ids=test_example_ids,
                row_key=row_key,
                layer=layer,
                axis_index=0,
                context_variant=BASE_CONTEXT,
            )
            _, y_y_test = _collect_coordinate_rows(
                prompt_rows=prompt_rows,
                activation_cache=activation_cache,
                example_ids=test_example_ids,
                row_key=row_key,
                layer=layer,
                axis_index=1,
                context_variant=BASE_CONTEXT,
            )
            x_probe = _train_regression_probe(X_train, y_x_train)
            y_probe = _train_regression_probe(X_train, y_y_train)
            decoded = _decode_examples(
                prompt_rows=prompt_rows,
                activation_cache=activation_cache,
                row_key=row_key,
                layer=layer,
                x_probe=x_probe,
                y_probe=y_probe,
            )
            pair_results: dict[str, Any] = {}
            pair_matrices: dict[str, dict[str, np.ndarray]] = {}
            for source_context, target_context in _deformation_pairs(contexts):
                pair_key = f"{source_context}_to_{target_context}"
                train_pairs = _collect_pair_entries(
                    decoded,
                    source_context=source_context,
                    target_context=target_context,
                    example_ids=train_example_ids,
                )
                test_pairs = _collect_pair_entries(
                    decoded,
                    source_context=source_context,
                    target_context=target_context,
                    example_ids=test_example_ids,
                )
                family_results: dict[str, Any] = {}
                family_matrices: dict[str, np.ndarray] = {}
                for family in TRANSFORM_FAMILIES:
                    result = _fit_and_evaluate_family(
                        train_pairs=train_pairs,
                        test_pairs=test_pairs,
                        family=family,
                    )
                    family_results[family] = result
                    if "matrix" in result:
                        family_matrices[family] = np.asarray(result["matrix"], dtype=np.float32)
                pair_results[pair_key] = family_results
                pair_matrices[pair_key] = family_matrices

            composition_results: dict[str, Any] = {}
            adjacent_keys = [f"{left}_to_{right}" for left, right in zip(contexts[:-1], contexts[1:])]
            direct_key = f"{contexts[0]}_to_{contexts[-1]}"
            direct_test_pairs = _collect_pair_entries(
                decoded,
                source_context=contexts[0],
                target_context=contexts[-1],
                example_ids=test_example_ids,
            )
            for family in TRANSFORM_FAMILIES:
                if any(family not in pair_matrices.get(pair_key, {}) for pair_key in adjacent_keys + [direct_key]):
                    continue
                composed = _compose_matrices([pair_matrices[pair_key][family] for pair_key in adjacent_keys])
                direct = pair_matrices[direct_key][family]
                composition_results[family] = {
                    "composed": {**_evaluate_transform(direct_test_pairs, composed), **_matrix_summary(composed)},
                    "direct": {**_evaluate_transform(direct_test_pairs, direct), **_matrix_summary(direct)},
                    "matrix_cosine": _matrix_cosine_similarity(composed, direct),
                    "frobenius_gap": float(np.linalg.norm(composed - direct)),
                }

            summary = {
                "best_family_by_pair": {},
                "best_composition_family": None,
            }
            best_composition: tuple[str, float] | None = None
            for pair_key, family_results in pair_results.items():
                best_family: tuple[str, float] | None = None
                for family, metrics in family_results.items():
                    score = metrics.get("coord_r2_mean")
                    if score is None:
                        continue
                    if best_family is None or float(score) > best_family[1]:
                        best_family = (family, float(score))
                summary["best_family_by_pair"][pair_key] = None if best_family is None else {
                    "family": best_family[0],
                    "coord_r2_mean": best_family[1],
                }
            for family, metrics in composition_results.items():
                matrix_cos = metrics.get("matrix_cosine")
                if matrix_cos is not None and (best_composition is None or float(matrix_cos) > best_composition[1]):
                    best_composition = (family, float(matrix_cos))
            if best_composition is not None:
                summary["best_composition_family"] = {
                    "family": best_composition[0],
                    "matrix_cosine": best_composition[1],
                }

            results["states"][state_name] = {
                "row_key": row_key,
                "layer": layer,
                "coordinate_probe_metrics": {
                    "base_x_market_r2": _evaluate_regression_probe(x_probe, X_test, y_x_test),
                    "base_y_market_r2": _evaluate_regression_probe(y_probe, X_test, y_y_test),
                },
                "pair_transforms": pair_results,
                "composition": composition_results,
                "summary": summary,
            }

    config.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.results_dir / "results.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote research risk-geometry analysis to {output_path}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze real DX risk-geometry rerun captures")
    parser.add_argument("--experiment-id", default="real_risk_geometry_bridge_v1")
    parser.add_argument("--research-activations-dir", type=Path, default=Path("data/activations/research_rerun"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis_results/research_risk_geometry"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    config = ResearchRiskGeometryConfig(
        research_activations_dir=args.research_activations_dir,
        output_dir=args.output_dir,
        experiment_id=args.experiment_id,
        seed=args.seed,
        test_fraction=args.test_fraction,
        num_workers=args.num_workers,
    )
    results = run_research_risk_geometry_analysis(config)
    print(json.dumps(results.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
