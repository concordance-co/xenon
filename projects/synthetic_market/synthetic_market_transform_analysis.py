from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from projects.counterfactual.analysis import orthogonal_procrustes
from projects.synthetic_market.synthetic_manifold_analysis import (
    _evaluate_regression_probe,
    _load_structure_tables,
    _mean,
    _preload_pooled_residuals,
    _train_regression_probe,
)
from projects.synthetic_market.synthetic_market_representation_analysis import (
    SET_GEOMETRY_CONTROL_FAMILY,
    SET_GEOMETRY_COORDS_BY_SCENARIO,
    _collect_set_geometry_coordinate_rows_for_context,
    _ordered_set_geometry_context_variants,
    _pairwise_distance_vector_for_profiles,
    _parse_set_geometry_example_id,
    _score_distance_vector_for_profiles,
    _set_geometry_context_deformation_pairs,
    _set_geometry_context_transfer_pairs,
    _split_example_ids,
)


@dataclass
class SyntheticMarketTransformConfig:
    structure_dir: Path = Path("data/activations/synthetic_structure/phase11_set_geometry_risk_ladder_v1")
    phase11_results_path: Path = Path(
        "data/analysis_results/synthetic_market_representation/phase11_set_geometry_risk_ladder_v1/results.json"
    )
    output_dir: Path = Path("data/analysis_results/synthetic_market_transform/phase12_risk_ladder_transforms_v1")
    phase_name: str = "phase11_set_geometry_risk_ladder_v1"
    seed: int = 42
    test_fraction: float = 0.2
    num_workers: int = 8


def _matrix_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float | None:
    av = np.asarray(a, dtype=np.float64).reshape(-1)
    bv = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom == 0.0:
        return None
    return float(np.dot(av, bv) / denom)


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
    if denom <= 1e-12:
        scale = 1.0
    else:
        scale = float(np.sum(xr * y) / denom)
    return (scale * r).astype(np.float32)


def _fit_diagonal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    scales = []
    for axis in range(x.shape[1]):
        denom = float(np.dot(x[:, axis], x[:, axis]))
        if denom <= 1e-12:
            scales.append(1.0)
        else:
            scales.append(float(np.dot(x[:, axis], y[:, axis]) / denom))
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


def _decode_set_geometry_examples(
    *,
    asset_rows: list[dict[str, Any]],
    activation_cache: dict[int, dict[str, np.ndarray]],
    row_key: str,
    layer: int,
    x_probe: Any,
    y_probe: Any,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in asset_rows:
        if str(row.get("family")) != SET_GEOMETRY_CONTROL_FAMILY:
            continue
        parsed = _parse_set_geometry_example_id(row.get("example_id"))
        if parsed is None:
            continue
        _, style_idx, perm_idx, scale_idx = parsed
        scenario = str(row.get("family_variant"))
        context_variant = str(row.get("context_variant"))
        profile_id = str(row.get("profile_id") or "")
        if profile_id not in SET_GEOMETRY_COORDS_BY_SCENARIO.get(scenario, {}):
            continue
        log_id = int(row["log_id"])
        acts = activation_cache.get(log_id)
        if not acts:
            continue
        key = f"{row_key}_{int(row['row_index'])}"
        if key not in acts:
            continue
        act = acts[key][layer].astype(np.float32).reshape(1, -1)
        x_hat = float(x_probe.predict(act)[0])
        y_hat = float(y_probe.predict(act)[0])
        example_key = (str(row["example_id"]), context_variant)
        entry = grouped.setdefault(
            example_key,
            {
                "example_id": str(row["example_id"]),
                "context_variant": context_variant,
                "scenario": scenario,
                "style_idx": style_idx,
                "perm_idx": perm_idx,
                "scale_idx": scale_idx,
                "decoded_coords": {},
                "score_coords": {},
                "latent_coords": {},
            },
        )
        entry["decoded_coords"][profile_id] = (x_hat, y_hat)
        entry["score_coords"][profile_id] = (
            float(row.get("attractiveness_score", 0.0)),
            float(row.get("risk_adjusted_score", 0.0)),
        )
        entry["latent_coords"][profile_id] = tuple(
            float(v) for v in SET_GEOMETRY_COORDS_BY_SCENARIO[scenario][profile_id]
        )

    finalized: list[dict[str, Any]] = []
    for (_, _), entry in sorted(grouped.items()):
        ordered_profiles = tuple(SET_GEOMETRY_COORDS_BY_SCENARIO[entry["scenario"]])
        if any(profile_id not in entry["decoded_coords"] for profile_id in ordered_profiles):
            continue
        decoded = np.asarray([entry["decoded_coords"][profile_id] for profile_id in ordered_profiles], dtype=np.float32)
        score = np.asarray([entry["score_coords"][profile_id] for profile_id in ordered_profiles], dtype=np.float32)
        latent = np.asarray([entry["latent_coords"][profile_id] for profile_id in ordered_profiles], dtype=np.float32)
        decoded_centered = _center_rows(decoded)
        score_centered = _center_rows(score)
        latent_centered = _center_rows(latent)
        geom_vec = _pairwise_distance_vector(decoded_centered)
        score_vec = _pairwise_distance_vector(score_centered)
        latent_vec = _pairwise_distance_vector(latent_centered)
        finalized.append({
            **entry,
            "ordered_profiles": ordered_profiles,
            "decoded_centered": decoded_centered,
            "score_centered": score_centered,
            "latent_centered": latent_centered,
            "geometry_vec": geom_vec,
            "score_geometry_vec": score_vec,
            "latent_geometry_vec": latent_vec,
        })
    return finalized


def _collect_pair_entries(
    examples: list[dict[str, Any]],
    *,
    source_context: str,
    target_context: str,
    example_ids: set[str] | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_key = {
        (str(example["example_id"]), str(example["context_variant"])): example
        for example in examples
    }
    candidate_ids = sorted({str(example["example_id"]) for example in examples})
    if example_ids is not None:
        candidate_ids = [example_id for example_id in candidate_ids if example_id in example_ids]
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for example_id in candidate_ids:
        source = by_key.get((example_id, source_context))
        target = by_key.get((example_id, target_context))
        if source is None or target is None:
            continue
        pairs.append((source, target))
    return pairs


def _stack_points(pairs: list[tuple[dict[str, Any], dict[str, Any]]], *, key: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for source, target in pairs:
        xs.append(np.asarray(source[key], dtype=np.float32))
        ys.append(np.asarray(target[key], dtype=np.float32))
    if not xs:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _r2_score_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return None
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return float(1.0 - ss_res / ss_tot)


def _evaluate_transform(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    matrix: np.ndarray,
) -> dict[str, Any]:
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
        latent_vec = np.asarray(target["latent_geometry_vec"], dtype=np.float32)

        corr = spearmanr(pred_vec, target_vec).correlation
        if corr is not None and not np.isnan(corr):
            spearmans.append(float(corr))
        denom = float(np.linalg.norm(pred_vec) * np.linalg.norm(target_vec))
        if denom > 1e-12:
            cosines.append(float(np.dot(pred_vec, target_vec) / denom))
        score_corr = spearmanr(pred_vec, score_vec).correlation
        if score_corr is not None and not np.isnan(score_corr):
            target_score_spearmans.append(float(score_corr))
        latent_corr = spearmanr(pred_vec, latent_vec).correlation
        if latent_corr is not None and not np.isnan(latent_corr):
            base_latent_spearmans.append(float(latent_corr))

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


def _select_states(phase11_results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    transfer = phase11_results.get("set_geometry_context_transfer", {})
    transfer_contexts = _ordered_set_geometry_context_variants(
        list(phase11_results.get("set_geometry_context_realignment", {}).keys())
    )
    transfer_pairs = _set_geometry_context_transfer_pairs(transfer_contexts)
    transfer_keys = [f"{left}_to_{right}" for left, right in transfer_pairs]
    available_targets = [target_name for target_name in ("latent_x", "latent_y") if target_name in transfer]
    if not transfer_keys or not available_targets:
        return {
            "early": {"error": "insufficient_context_transfer"},
            "late": {"error": "insufficient_context_realignment"},
        }

    best_early: tuple[str, int, float] | None = None
    for row_key in ("row_mean", "row_eos"):
        sample_target = available_targets[0]
        sample_key = next(
            (
                transfer_key
                for transfer_key in transfer_keys
                if transfer_key in transfer.get(sample_target, {})
                and row_key in transfer[sample_target][transfer_key]
            ),
            None,
        )
        if sample_key is None:
            continue
        n_layers = len(transfer[sample_target][sample_key][row_key])
        for layer_idx in range(n_layers):
            values: list[float] = []
            for transfer_key in transfer_keys:
                for target_name in available_targets:
                    target_transfers = transfer.get(target_name, {})
                    if transfer_key not in target_transfers or row_key not in target_transfers[transfer_key]:
                        continue
                    metric = target_transfers[transfer_key][row_key][layer_idx].get("r2")
                    if metric is not None:
                        values.append(float(metric))
            if not values:
                continue
            score = float(np.mean(values))
            if best_early is None or score > best_early[2]:
                best_early = (row_key, layer_idx, score)

    realignment = phase11_results.get("set_geometry_context_realignment", {})
    contexts = _ordered_set_geometry_context_variants(list(realignment.keys()))
    if not contexts:
        return {
            "early": {"error": "insufficient_context_transfer"},
            "late": {"error": "insufficient_context_realignment"},
        }
    best_late: tuple[str, int, float] | None = None
    for row_key in ("row_mean", "row_eos"):
        if row_key not in realignment.get(contexts[0], {}):
            continue
        n_layers = len(realignment[contexts[0]][row_key])
        for layer_idx in range(n_layers):
            values = []
            for context in contexts:
                if row_key not in realignment.get(context, {}):
                    continue
                margin = realignment[context][row_key][layer_idx].get("score_over_base_margin")
                if margin is not None:
                    values.append(float(margin))
            if not values:
                continue
            score = float(np.mean(values))
            if best_late is None or score > best_late[2]:
                best_late = (row_key, layer_idx, score)

    if best_early is None:
        return {
            "early": {"error": "insufficient_context_transfer"},
            "late": {"error": "insufficient_context_realignment"},
        }
    if best_late is None:
        return {
            "early": {"row_key": best_early[0], "layer": best_early[1], "selection_score": best_early[2]},
            "late": {"error": "insufficient_context_realignment"},
        }
    return {
        "early": {"row_key": best_early[0], "layer": best_early[1], "selection_score": best_early[2]},
        "late": {"row_key": best_late[0], "layer": best_late[1], "selection_score": best_late[2]},
    }


def run_synthetic_market_transform_analysis(config: SyntheticMarketTransformConfig) -> dict[str, Any]:
    phase11_results = json.loads(config.phase11_results_path.read_text())
    selected_states = _select_states(phase11_results)
    if any("error" in state for state in selected_states.values()):
        return {
            "phase_name": config.phase_name,
            "selected_states": selected_states,
            "error": "insufficient_context_ladder_for_transform_analysis",
        }

    meta_rows, tick_rows, asset_rows = _load_structure_tables(config.structure_dir)
    if not meta_rows:
        return {"error": "no_synthetic_structure_metadata"}

    set_rows = [row for row in asset_rows if str(row.get("family")) == SET_GEOMETRY_CONTROL_FAMILY]
    if not set_rows:
        return {"error": "no_set_geometry_rows"}
    log_ids = sorted({int(row["log_id"]) for row in set_rows})
    activation_cache = _preload_pooled_residuals(
        config.structure_dir,
        log_ids,
        max_workers=config.num_workers,
    )
    if not activation_cache:
        return {"error": "no_pooled_residuals"}

    context_variants = _ordered_set_geometry_context_variants([str(row.get("context_variant")) for row in set_rows])
    example_ids = sorted({str(row.get("example_id")) for row in set_rows})
    train_example_ids, test_example_ids = _split_example_ids(
        example_ids,
        seed=config.seed,
        test_fraction=config.test_fraction,
    )

    analysis: dict[str, Any] = {
        "phase_name": config.phase_name,
        "selected_states": selected_states,
        "contexts": context_variants,
        "n_examples": len(example_ids),
        "transform_pairs": [f"{left}_to_{right}" for left, right in _set_geometry_context_deformation_pairs(context_variants)],
        "states": {},
    }

    for state_name, state in selected_states.items():
        row_key = str(state["row_key"])
        layer = int(state["layer"])

        x_train, y_x_train = _collect_set_geometry_coordinate_rows_for_context(
            example_ids=train_example_ids,
            asset_rows=set_rows,
            activation_cache=activation_cache,
            row_key=row_key,
            layer=layer,
            axis_index=0,
            context_variant="market_only",
        )
        _, y_y_train = _collect_set_geometry_coordinate_rows_for_context(
            example_ids=train_example_ids,
            asset_rows=set_rows,
            activation_cache=activation_cache,
            row_key=row_key,
            layer=layer,
            axis_index=1,
            context_variant="market_only",
        )
        x_test, y_x_test = _collect_set_geometry_coordinate_rows_for_context(
            example_ids=test_example_ids,
            asset_rows=set_rows,
            activation_cache=activation_cache,
            row_key=row_key,
            layer=layer,
            axis_index=0,
            context_variant="market_only",
        )
        _, y_y_test = _collect_set_geometry_coordinate_rows_for_context(
            example_ids=test_example_ids,
            asset_rows=set_rows,
            activation_cache=activation_cache,
            row_key=row_key,
            layer=layer,
            axis_index=1,
            context_variant="market_only",
        )
        if x_train.size == 0 or x_test.size == 0:
            analysis["states"][state_name] = {"error": "insufficient_coordinate_probe_data"}
            continue

        x_probe = _train_regression_probe(x_train, y_x_train)
        y_probe = _train_regression_probe(x_train, y_y_train)
        coord_probe_metrics = {
            "latent_x_market_r2": _evaluate_regression_probe(x_probe, x_test, y_x_test),
            "latent_y_market_r2": _evaluate_regression_probe(y_probe, x_test, y_y_test),
        }

        decoded_examples = _decode_set_geometry_examples(
            asset_rows=set_rows,
            activation_cache=activation_cache,
            row_key=row_key,
            layer=layer,
            x_probe=x_probe,
            y_probe=y_probe,
        )

        pair_results: dict[str, Any] = {}
        pair_matrices: dict[str, dict[str, np.ndarray]] = {}
        for source_context, target_context in _set_geometry_context_deformation_pairs(context_variants):
            pair_key = f"{source_context}_to_{target_context}"
            train_pairs = _collect_pair_entries(
                decoded_examples,
                source_context=source_context,
                target_context=target_context,
                example_ids=train_example_ids,
            )
            test_pairs = _collect_pair_entries(
                decoded_examples,
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
        adjacent_pairs = _set_geometry_context_deformation_pairs(context_variants)
        adjacent_keys = [f"{left}_to_{right}" for left, right in adjacent_pairs]
        direct_key = f"{context_variants[0]}_to_{context_variants[-1]}"
        direct_test_pairs = _collect_pair_entries(
            decoded_examples,
            source_context=context_variants[0],
            target_context=context_variants[-1],
            example_ids=test_example_ids,
        )
        if len(context_variants) >= 2:
            for family in TRANSFORM_FAMILIES:
                if any(family not in pair_matrices[pair_key] for pair_key in adjacent_keys + [direct_key]):
                    continue
                composed = _compose_matrices([pair_matrices[pair_key][family] for pair_key in adjacent_keys])
                direct = pair_matrices[direct_key][family]
                composed_eval = _evaluate_transform(direct_test_pairs, composed)
                direct_eval = _evaluate_transform(direct_test_pairs, direct)
                composition_results[family] = {
                    "composed": {
                        **composed_eval,
                        **_matrix_summary(composed),
                    },
                    "direct": {
                        **direct_eval,
                        **_matrix_summary(direct),
                    },
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
            if matrix_cos is None:
                continue
            if best_composition is None or float(matrix_cos) > best_composition[1]:
                best_composition = (family, float(matrix_cos))
        if best_composition is not None:
            summary["best_composition_family"] = {
                "family": best_composition[0],
                "matrix_cosine": best_composition[1],
            }

        analysis["states"][state_name] = {
            "row_key": row_key,
            "layer": layer,
            "coordinate_probe_metrics": coord_probe_metrics,
            "pair_transforms": pair_results,
            "composition": composition_results,
            "summary": summary,
        }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.output_dir / "results.json"
    output_path.write_text(json.dumps(analysis, indent=2))
    print(f"Wrote synthetic market transform analysis to {output_path}", flush=True)
    return analysis
