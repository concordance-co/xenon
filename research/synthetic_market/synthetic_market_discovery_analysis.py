"""Discovery-oriented PCA analysis for DX-like synthetic market prompts.

This module is the first step of the revised program:

1. Capture market-only prompts in a realistic DX-like surface form.
2. Pool section states such as `market_mean` and `market_eos`.
3. Run PCA across prompts and ask what the dominant directions correlate with.

The goal is not to impose a latent basis up front. The goal is to learn which
prompt-level directions the model appears to use before any context ladder is
applied.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression

from research.synthetic_market.synthetic_manifold_analysis import (
    _load_structure_tables,
    _preload_pooled_residuals,
)


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(np.std(values)) if values else 0.0


def _leader_gap(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values, reverse=True)
    return float(ordered[0] - ordered[1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    corr = spearmanr(x, y).correlation
    if corr is None or np.isnan(corr):
        return None
    return float(corr)


def _group_asset_rows(asset_rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in asset_rows:
        grouped.setdefault(int(row["log_id"]), []).append(dict(row))
    for log_id in grouped:
        grouped[log_id].sort(key=lambda row: int(row["row_index"]))
    return grouped


def _build_prompt_features(
    tick_rows: list[dict[str, Any]],
    asset_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    *,
    context_variant: str,
    family_allowlist: tuple[str, ...],
) -> list[dict[str, Any]]:
    asset_by_log = _group_asset_rows(asset_rows)
    tick_by_log = {
        int(row["log_id"]): dict(row)
        for row in tick_rows
        if str(row.get("context_variant")) == context_variant
        and (not family_allowlist or str(row.get("family")) in family_allowlist)
    }
    meta_by_log = {int(row["log_id"]): dict(row) for row in metadata_rows}

    rows: list[dict[str, Any]] = []
    for log_id, tick in sorted(tick_by_log.items()):
        asset_list = asset_by_log.get(log_id)
        if not asset_list:
            continue
        meta = meta_by_log.get(log_id, {})

        pct_5m = [float(row["pct_5m"]) for row in asset_list]
        pct_1h = [float(row["pct_1h"]) for row in asset_list]
        flow_5m = [float(row["net_flow_5m"]) for row in asset_list]
        vol_5m = [float(row["vol_5m"]) for row in asset_list]
        vol_1h = [float(row["vol_1h"]) for row in asset_list]
        traders_5m = [float(row["unique_traders_5m"]) for row in asset_list]
        top20 = [float(row["top20_holder_pct"]) for row in asset_list]
        edge = [float(row["edge_after_fee_score"]) for row in asset_list]
        risk_adj = [float(row["risk_adjusted_score"]) for row in asset_list]
        attractiveness = [float(row["attractiveness_score"]) for row in asset_list]

        rows.append({
            "log_id": log_id,
            "family": str(tick["family"]),
            "family_variant": str(tick["family_variant"]),
            "user_chars": float(tick.get("user_chars", 0)),
            "n_rows": float(tick.get("n_rows", len(asset_list))),
            "seq_len": float(meta.get("seq_len", 0)),
            "pct_5m_mean": _safe_mean(pct_5m),
            "pct_5m_max": max(pct_5m),
            "pct_5m_std": _safe_std(pct_5m),
            "pct_5m_gap": _leader_gap(pct_5m),
            "pct_1h_mean": _safe_mean(pct_1h),
            "pct_1h_max": max(pct_1h),
            "pct_1h_std": _safe_std(pct_1h),
            "pct_1h_gap": _leader_gap(pct_1h),
            "net_flow_5m_mean": _safe_mean(flow_5m),
            "net_flow_5m_max": max(flow_5m),
            "net_flow_5m_std": _safe_std(flow_5m),
            "net_flow_5m_gap": _leader_gap(flow_5m),
            "vol_5m_mean": _safe_mean(vol_5m),
            "vol_1h_mean": _safe_mean(vol_1h),
            "unique_traders_5m_mean": _safe_mean(traders_5m),
            "unique_traders_5m_max": max(traders_5m),
            "unique_traders_5m_sum": float(sum(traders_5m)),
            "top20_holder_pct_mean": _safe_mean(top20),
            "top20_holder_pct_min": min(top20),
            "top20_holder_pct_max": max(top20),
            "top20_holder_pct_std": _safe_std(top20),
            "edge_after_fee_mean": _safe_mean(edge),
            "edge_after_fee_max": max(edge),
            "edge_after_fee_gap": _leader_gap(edge),
            "risk_adjusted_mean": _safe_mean(risk_adj),
            "risk_adjusted_max": max(risk_adj),
            "risk_adjusted_gap": _leader_gap(risk_adj),
            "attractiveness_mean": _safe_mean(attractiveness),
            "attractiveness_max": max(attractiveness),
            "attractiveness_gap": _leader_gap(attractiveness),
        })
    return rows


def _numeric_feature_table(rows: list[dict[str, Any]]) -> tuple[list[int], list[str], np.ndarray]:
    if not rows:
        return [], [], np.zeros((0, 0), dtype=np.float32)
    feature_names = [
        "user_chars",
        "n_rows",
        "seq_len",
        "pct_5m_mean",
        "pct_5m_max",
        "pct_5m_std",
        "pct_5m_gap",
        "pct_1h_mean",
        "pct_1h_max",
        "pct_1h_std",
        "pct_1h_gap",
        "net_flow_5m_mean",
        "net_flow_5m_max",
        "net_flow_5m_std",
        "net_flow_5m_gap",
        "vol_5m_mean",
        "vol_1h_mean",
        "unique_traders_5m_mean",
        "unique_traders_5m_max",
        "unique_traders_5m_sum",
        "top20_holder_pct_mean",
        "top20_holder_pct_min",
        "top20_holder_pct_max",
        "top20_holder_pct_std",
        "edge_after_fee_mean",
        "edge_after_fee_max",
        "edge_after_fee_gap",
        "risk_adjusted_mean",
        "risk_adjusted_max",
        "risk_adjusted_gap",
        "attractiveness_mean",
        "attractiveness_max",
        "attractiveness_gap",
    ]
    log_ids = [int(row["log_id"]) for row in rows]
    X = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32)
    return log_ids, feature_names, X


def _top_correlations(
    scores: np.ndarray,
    feature_matrix: np.ndarray,
    feature_names: list[str],
    *,
    top_k: int,
    allowed_features: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_idx, feature_name in enumerate(feature_names):
        if feature_name not in allowed_features:
            continue
        corr = _spearman(scores, feature_matrix[:, feature_idx])
        if corr is None:
            continue
        rows.append({
            "feature": feature_name,
            "spearman": corr,
            "abs_spearman": abs(corr),
        })
    rows.sort(key=lambda row: (-float(row["abs_spearman"]), str(row["feature"])))
    return rows[:top_k]


def _nuisance_matrix(
    feature_matrix: np.ndarray,
    feature_names: list[str],
    *,
    nuisance_features: set[str],
) -> np.ndarray:
    nuisance_names = [name for name in feature_names if name in nuisance_features]
    if not nuisance_names:
        return np.zeros((feature_matrix.shape[0], 0), dtype=np.float32)
    indices = [feature_names.index(name) for name in nuisance_names]
    return feature_matrix[:, indices].astype(np.float32)


def _residualize_activations(
    activations: np.ndarray,
    nuisance: np.ndarray,
) -> np.ndarray:
    activations = np.asarray(activations, dtype=np.float32)
    nuisance = np.asarray(nuisance, dtype=np.float32)
    if activations.ndim != 2:
        raise ValueError("activations must be 2D")
    if nuisance.ndim != 2:
        raise ValueError("nuisance must be 2D")
    if activations.shape[0] != nuisance.shape[0]:
        raise ValueError("activations and nuisance must have the same number of rows")
    if nuisance.shape[1] == 0:
        return activations.copy()
    reg = LinearRegression()
    reg.fit(nuisance, activations)
    predicted = reg.predict(nuisance).astype(np.float32)
    return activations - predicted


def _compute_pca(
    X: np.ndarray,
    *,
    max_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    centered = X - mean
    scale = centered.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale == 0.0] = 1.0
    standardized = centered / scale
    u, s, vt = np.linalg.svd(standardized, full_matrices=False)
    n_components = min(max_components, vt.shape[0])
    components = vt[:n_components].astype(np.float32)
    scores = (u[:, :n_components] * s[:n_components]).astype(np.float32)
    explained_variance = (s[:n_components] ** 2) / max(1, standardized.shape[0] - 1)
    total_variance = float(np.sum((s ** 2) / max(1, standardized.shape[0] - 1)))
    explained_ratio = (
        explained_variance / total_variance if total_variance > 0.0 else np.zeros_like(explained_variance)
    ).astype(np.float32)
    return mean, scale, components, scores, explained_ratio


@dataclass
class SyntheticMarketDiscoveryConfig:
    structure_dir: Path = Path("data/activations/synthetic_structure/phase15_market_basis_discovery_v1")
    output_dir: Path = Path("data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1")
    phase_name: str = "phase15_market_basis_discovery_v1"
    context_variant: str = "market_only"
    state_keys: tuple[str, ...] = ("market_mean", "market_eos")
    family_allowlist: tuple[str, ...] = ("market_basis_scalar", "market_basis_coupled")
    layers: list[int] | None = None
    max_components: int = 5
    num_workers: int = 8
    residualize_nuisance: bool = False


def run_synthetic_market_discovery_analysis(config: SyntheticMarketDiscoveryConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows, tick_rows, asset_rows = _load_structure_tables(config.structure_dir)
    prompt_rows = _build_prompt_features(
        tick_rows,
        asset_rows,
        metadata_rows,
        context_variant=config.context_variant,
        family_allowlist=config.family_allowlist,
    )
    if not prompt_rows:
        result = {"error": "no_prompt_rows"}
        (config.output_dir / "results.json").write_text(json.dumps(result, indent=2))
        return result

    log_ids, feature_names, feature_matrix = _numeric_feature_table(prompt_rows)
    nuisance_features = {"user_chars", "seq_len", "n_rows"}
    market_features = set(feature_names) - nuisance_features
    activation_cache = _preload_pooled_residuals(config.structure_dir, log_ids, max_workers=config.num_workers)
    available_layers = sorted({
        tensor.shape[0] - 1
        for log_id in log_ids
        for tensor in activation_cache.get(log_id, {}).values()
        if tensor.ndim == 2
    })
    if not available_layers:
        result = {"error": "no_pooled_residuals"}
        (config.output_dir / "results.json").write_text(json.dumps(result, indent=2))
        return result

    layers = config.layers or list(range(max(available_layers) + 1))
    results: dict[str, Any] = {
        "phase_name": config.phase_name,
        "context_variant": config.context_variant,
        "n_prompts": len(log_ids),
        "feature_names": feature_names,
        "nuisance_features": sorted(nuisance_features),
        "market_features": sorted(market_features),
        "residualize_nuisance": bool(config.residualize_nuisance),
        "states": {},
    }
    basis_payload: dict[str, np.ndarray] = {}

    for state_key in config.state_keys:
        state_results: dict[str, Any] = {}
        for layer in layers:
            X_rows: list[np.ndarray] = []
            kept_indices: list[int] = []
            for idx, log_id in enumerate(log_ids):
                acts = activation_cache.get(log_id)
                if not acts or state_key not in acts:
                    continue
                tensor = acts[state_key]
                if tensor.ndim != 2 or layer >= tensor.shape[0]:
                    continue
                X_rows.append(tensor[layer].astype(np.float32))
                kept_indices.append(idx)
            if len(X_rows) < 3:
                continue

            X = np.stack(X_rows)
            feature_subset = feature_matrix[kept_indices]
            nuisance_subset = _nuisance_matrix(
                feature_subset,
                feature_names,
                nuisance_features=nuisance_features,
            )
            X_for_pca = (
                _residualize_activations(X, nuisance_subset)
                if config.residualize_nuisance
                else X
            )
            mean, scale, components, scores, explained_ratio = _compute_pca(
                X_for_pca,
                max_components=config.max_components,
            )
            variances = explained_ratio.astype(np.float64)
            participation_ratio = float((variances.sum() ** 2) / np.square(variances).sum()) if np.square(variances).sum() > 0 else 0.0

            pc_rows: list[dict[str, Any]] = []
            for pc_idx in range(scores.shape[1]):
                pc_scores = scores[:, pc_idx]
                pc_rows.append({
                    "pc_index": pc_idx + 1,
                    "explained_variance_ratio": float(explained_ratio[pc_idx]),
                    "top_market_correlations": _top_correlations(
                        pc_scores,
                        feature_subset,
                        feature_names,
                        top_k=6,
                        allowed_features=market_features,
                    ),
                    "top_nuisance_correlations": _top_correlations(
                        pc_scores,
                        feature_subset,
                        feature_names,
                        top_k=3,
                        allowed_features=nuisance_features,
                    ),
                })

            state_results[str(layer)] = {
                "n_prompts": len(X_rows),
                "explained_variance_ratio": [float(value) for value in explained_ratio],
                "participation_ratio_top_components": participation_ratio,
                "residualize_nuisance": bool(config.residualize_nuisance),
                "pcs": pc_rows,
            }
            basis_prefix = f"{state_key}_layer_{layer}"
            basis_payload[f"{basis_prefix}__mean"] = mean
            basis_payload[f"{basis_prefix}__scale"] = scale
            basis_payload[f"{basis_prefix}__components"] = components
        results["states"][state_key] = state_results

    (config.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    if basis_payload:
        np.savez_compressed(config.output_dir / "pca_basis.npz", **basis_payload)
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run discovery PCA on DX-like synthetic market prompts.")
    parser.add_argument("--structure-dir", type=Path, default=Path("data/activations/synthetic_structure/phase15_market_basis_discovery_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1"))
    parser.add_argument("--phase-name", default="phase15_market_basis_discovery_v1")
    parser.add_argument("--context-variant", default="market_only")
    parser.add_argument("--layers", default="")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--residualize-nuisance", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    layers = [int(token) for token in args.layers.split(",") if token.strip()] or None
    result = run_synthetic_market_discovery_analysis(
        SyntheticMarketDiscoveryConfig(
            structure_dir=args.structure_dir,
            output_dir=args.output_dir,
            phase_name=args.phase_name,
            context_variant=args.context_variant,
            layers=layers,
            num_workers=args.num_workers,
            residualize_nuisance=args.residualize_nuisance,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
