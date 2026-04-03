"""Phase 16 synthetic context-order analysis.

Compares matched A/B/C prompt variants built from the same synthetic market:

- A: market only
- B: market then context after
- C: context before then market

The focus is whether context-before-market changes the market representation at
market_mean / market_eos, and whether B and C converge again by last_token.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from projects.synthetic_market.synthetic_manifold_analysis import (
    _load_structure_tables,
    _preload_pooled_residuals,
)
from projects.synthetic_market.synthetic_market_discovery_analysis import (
    _nuisance_matrix,
    _residualize_activations,
)


CONTEXT_GROUPS = {
    "risk": {
        "A": "market_only",
        "B": "risk_5_after_market",
        "C": "risk_5_before_market",
    },
    "affordance": {
        "A": "market_only",
        "B": "affordance_5_after_market",
        "C": "affordance_5_before_market",
    },
}


def _cosine_similarity_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.where(denom == 0.0, 1.0, denom)
    return np.sum(a * b, axis=1) / denom


def _mean_l2_rows(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1).mean())


def _projection_scores(X: np.ndarray, *, mean: np.ndarray, scale: np.ndarray, components: np.ndarray) -> np.ndarray:
    standardized = (X - mean) / np.where(scale == 0.0, 1.0, scale)
    return standardized @ components.T


def _build_prompt_nuisance_rows(
    tick_rows: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, float]], list[str]]:
    meta_by_log = {int(row["log_id"]): dict(row) for row in metadata_rows}
    nuisance_rows: dict[int, dict[str, float]] = {}
    feature_names = ["user_chars", "n_rows", "seq_len"]
    for row in tick_rows:
        log_id = int(row["log_id"])
        meta = meta_by_log.get(log_id, {})
        nuisance_rows[log_id] = {
            "user_chars": float(row.get("user_chars", len(str(row.get("user_prompt", ""))))),
            "n_rows": float(row.get("n_rows", row.get("num_assets", 0))),
            "seq_len": float(meta.get("seq_len", 0)),
        }
    return nuisance_rows, feature_names


def _matched_examples(
    tick_rows: list[dict[str, Any]],
    group_name: str,
) -> list[dict[str, Any]]:
    required = CONTEXT_GROUPS[group_name]
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in tick_rows:
        key = (str(row["example_id"]), str(row["family"]), str(row["family_variant"]))
        grouped.setdefault(key, {})[str(row["context_variant"])] = dict(row)

    matched: list[dict[str, Any]] = []
    for (example_id, family, family_variant), variants in grouped.items():
        if not all(ctx in variants for ctx in required.values()):
            continue
        matched.append({
            "example_id": example_id,
            "family": family,
            "family_variant": family_variant,
            "A_log_id": int(variants[required["A"]]["log_id"]),
            "B_log_id": int(variants[required["B"]]["log_id"]),
            "C_log_id": int(variants[required["C"]]["log_id"]),
        })
    matched.sort(key=lambda row: (row["family"], row["family_variant"], row["example_id"]))
    return matched


@dataclass
class SyntheticMarketContextOrderConfig:
    structure_dir: Path = Path("data/activations/synthetic_structure/phase16_context_order_v1")
    output_dir: Path = Path("data/analysis_results/synthetic_market_context_order/phase16_context_order_v1")
    phase_name: str = "phase16_context_order_v1"
    basis_npz_path: Path = Path(
        "data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/pca_basis.npz"
    )
    basis_results_path: Path = Path(
        "data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/results.json"
    )
    state_keys: tuple[str, ...] = ("market_mean", "market_eos")
    integration_state_keys: tuple[str, ...] = ("last_token", "active_settings_eos", "portfolio_eos", "constraints_eos")
    layers: list[int] | None = None
    num_workers: int = 8
    cross_basis_overrides: dict[str, str] | None = None
    cross_basis_layers: tuple[int, ...] = (40, 42)


def run_synthetic_market_context_order_analysis(config: SyntheticMarketContextOrderConfig) -> dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows, tick_rows, _asset_rows = _load_structure_tables(config.structure_dir)
    nuisance_rows, nuisance_feature_names = _build_prompt_nuisance_rows(tick_rows, metadata_rows)
    tick_by_log = {int(row["log_id"]): dict(row) for row in tick_rows}
    all_log_ids = sorted(tick_by_log)
    activation_cache = _preload_pooled_residuals(config.structure_dir, all_log_ids, max_workers=config.num_workers)

    available_layers = sorted({
        tensor.shape[0] - 1
        for acts in activation_cache.values()
        for tensor in acts.values()
        if getattr(tensor, "ndim", 0) == 2
    })
    if not available_layers:
        result = {"error": "no_pooled_residuals"}
        (config.output_dir / "results.json").write_text(json.dumps(result, indent=2))
        return result
    layers = config.layers or list(range(max(available_layers) + 1))

    basis = np.load(config.basis_npz_path)
    basis_results = json.loads(config.basis_results_path.read_text())

    results: dict[str, Any] = {
        "phase_name": config.phase_name,
        "basis_phase_name": basis_results.get("phase_name"),
        "basis_residualize_nuisance": basis_results.get("residualize_nuisance", False),
        "cross_basis_overrides": config.cross_basis_overrides or {},
        "cross_basis_layers": list(config.cross_basis_layers),
        "groups": {},
    }
    cross_basis_overrides = config.cross_basis_overrides or {}
    cross_basis_layers = set(int(layer) for layer in config.cross_basis_layers)

    for group_name in CONTEXT_GROUPS:
        matched = _matched_examples(tick_rows, group_name)
        group_result: dict[str, Any] = {
            "n_matched_examples": len(matched),
            "state_results": {},
            "integration_results": {},
        }

        for state_key in config.state_keys:
            state_payload: dict[str, Any] = {}
            for layer in layers:
                a_rows: list[np.ndarray] = []
                b_rows: list[np.ndarray] = []
                c_rows: list[np.ndarray] = []
                nuisance_stack: list[list[float]] = []
                for row in matched:
                    a = activation_cache.get(row["A_log_id"], {})
                    b = activation_cache.get(row["B_log_id"], {})
                    c = activation_cache.get(row["C_log_id"], {})
                    if state_key not in a or state_key not in b or state_key not in c:
                        continue
                    a_tensor = a[state_key]
                    b_tensor = b[state_key]
                    c_tensor = c[state_key]
                    if layer >= a_tensor.shape[0] or layer >= b_tensor.shape[0] or layer >= c_tensor.shape[0]:
                        continue
                    a_rows.append(a_tensor[layer].astype(np.float32))
                    b_rows.append(b_tensor[layer].astype(np.float32))
                    c_rows.append(c_tensor[layer].astype(np.float32))
                    nuisance_stack.extend([
                        [nuisance_rows[row["A_log_id"]][name] for name in nuisance_feature_names],
                        [nuisance_rows[row["B_log_id"]][name] for name in nuisance_feature_names],
                        [nuisance_rows[row["C_log_id"]][name] for name in nuisance_feature_names],
                    ])

                if not a_rows:
                    continue
                A = np.stack(a_rows)
                B = np.stack(b_rows)
                C = np.stack(c_rows)
                raw_stack = np.concatenate([A, B, C], axis=0)
                nuisance = np.asarray(nuisance_stack, dtype=np.float32)
                resid_stack = _residualize_activations(raw_stack, nuisance)
                n = A.shape[0]
                A_resid, B_resid, C_resid = resid_stack[:n], resid_stack[n : 2 * n], resid_stack[2 * n :]

                row = {
                    "n_examples": int(n),
                    "ab_cosine_mean": float(_cosine_similarity_rows(A, B).mean()),
                    "ac_cosine_mean": float(_cosine_similarity_rows(A, C).mean()),
                    "perception_gap": float(_cosine_similarity_rows(A, B).mean() - _cosine_similarity_rows(A, C).mean()),
                    "ab_l2_mean": _mean_l2_rows(A, B),
                    "ac_l2_mean": _mean_l2_rows(A, C),
                }

                basis_prefix = f"{state_key}_layer_{layer}"
                comp_key = f"{basis_prefix}__components"
                if comp_key in basis:
                    mean = basis[f"{basis_prefix}__mean"]
                    scale = basis[f"{basis_prefix}__scale"]
                    components = basis[comp_key]
                    A_scores = _projection_scores(A_resid, mean=mean, scale=scale, components=components)
                    B_scores = _projection_scores(B_resid, mean=mean, scale=scale, components=components)
                    C_scores = _projection_scores(C_resid, mean=mean, scale=scale, components=components)
                    row["ab_pc_l2_mean"] = _mean_l2_rows(A_scores[:, :3], B_scores[:, :3])
                    row["ac_pc_l2_mean"] = _mean_l2_rows(A_scores[:, :3], C_scores[:, :3])
                    row["pc_shift"] = []
                    basis_layer = basis_results.get("states", {}).get(state_key, {}).get(str(layer), {})
                    pcs = basis_layer.get("pcs", [])
                    for pc_idx in range(min(3, A_scores.shape[1])):
                        feature = ""
                        if pc_idx < len(pcs) and pcs[pc_idx].get("top_market_correlations"):
                            feature = pcs[pc_idx]["top_market_correlations"][0]["feature"]
                        row["pc_shift"].append({
                            "pc_index": pc_idx + 1,
                            "feature": feature,
                            "ab_abs_mean": float(np.mean(np.abs(A_scores[:, pc_idx] - B_scores[:, pc_idx]))),
                            "ac_abs_mean": float(np.mean(np.abs(A_scores[:, pc_idx] - C_scores[:, pc_idx]))),
                        })
                override_state = cross_basis_overrides.get(state_key)
                if override_state and layer in cross_basis_layers:
                    override_prefix = f"{override_state}_layer_{layer}"
                    override_comp_key = f"{override_prefix}__components"
                    if override_comp_key in basis:
                        override_mean = basis[f"{override_prefix}__mean"]
                        override_scale = basis[f"{override_prefix}__scale"]
                        override_components = basis[override_comp_key]
                        A_override = _projection_scores(
                            A_resid, mean=override_mean, scale=override_scale, components=override_components
                        )
                        B_override = _projection_scores(
                            B_resid, mean=override_mean, scale=override_scale, components=override_components
                        )
                        C_override = _projection_scores(
                            C_resid, mean=override_mean, scale=override_scale, components=override_components
                        )
                        override_basis_layer = basis_results.get("states", {}).get(override_state, {}).get(str(layer), {})
                        override_pcs = override_basis_layer.get("pcs", [])
                        row["cross_basis_projection"] = {
                            "basis_state": override_state,
                            "ab_pc_l2_mean": _mean_l2_rows(A_override[:, :3], B_override[:, :3]),
                            "ac_pc_l2_mean": _mean_l2_rows(A_override[:, :3], C_override[:, :3]),
                            "pc_shift": [],
                        }
                        for pc_idx in range(min(3, A_override.shape[1])):
                            feature = ""
                            if pc_idx < len(override_pcs) and override_pcs[pc_idx].get("top_market_correlations"):
                                feature = override_pcs[pc_idx]["top_market_correlations"][0]["feature"]
                            row["cross_basis_projection"]["pc_shift"].append({
                                "pc_index": pc_idx + 1,
                                "feature": feature,
                                "ab_abs_mean": float(np.mean(np.abs(A_override[:, pc_idx] - B_override[:, pc_idx]))),
                                "ac_abs_mean": float(np.mean(np.abs(A_override[:, pc_idx] - C_override[:, pc_idx]))),
                            })
                state_payload[str(layer)] = row
            group_result["state_results"][state_key] = state_payload

        for state_key in config.integration_state_keys:
            state_payload: dict[str, Any] = {}
            for layer in layers:
                b_rows: list[np.ndarray] = []
                c_rows: list[np.ndarray] = []
                for row in matched:
                    b = activation_cache.get(row["B_log_id"], {})
                    c = activation_cache.get(row["C_log_id"], {})
                    if state_key not in b or state_key not in c:
                        continue
                    b_tensor = b[state_key]
                    c_tensor = c[state_key]
                    if layer >= b_tensor.shape[0] or layer >= c_tensor.shape[0]:
                        continue
                    b_rows.append(b_tensor[layer].astype(np.float32))
                    c_rows.append(c_tensor[layer].astype(np.float32))
                if not b_rows:
                    continue
                B = np.stack(b_rows)
                C = np.stack(c_rows)
                state_payload[str(layer)] = {
                    "n_examples": int(B.shape[0]),
                    "bc_cosine_mean": float(_cosine_similarity_rows(B, C).mean()),
                    "bc_l2_mean": _mean_l2_rows(B, C),
                }
            group_result["integration_results"][state_key] = state_payload

        results["groups"][group_name] = group_result

    (config.output_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 16 synthetic context-order analysis.")
    parser.add_argument("--structure-dir", type=Path, default=Path("data/activations/synthetic_structure/phase16_context_order_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/analysis_results/synthetic_market_context_order/phase16_context_order_v1"))
    parser.add_argument("--phase-name", default="phase16_context_order_v1")
    parser.add_argument(
        "--basis-npz-path",
        type=Path,
        default=Path("data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/pca_basis.npz"),
    )
    parser.add_argument(
        "--basis-results-path",
        type=Path,
        default=Path("data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/results.json"),
    )
    parser.add_argument("--layers", default="")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--cross-basis-overrides",
        default="",
        help="Comma-separated state overrides like market_eos:market_mean",
    )
    parser.add_argument(
        "--cross-basis-layers",
        default="40,42",
        help="Comma-separated layers for cross-basis projections",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    layers = [int(token) for token in args.layers.split(",") if token.strip()] or None
    cross_basis_overrides = {}
    for token in args.cross_basis_overrides.split(","):
        token = token.strip()
        if not token:
            continue
        src, dst = token.split(":", 1)
        cross_basis_overrides[src.strip()] = dst.strip()
    cross_basis_layers = tuple(int(token) for token in args.cross_basis_layers.split(",") if token.strip())
    result = run_synthetic_market_context_order_analysis(
        SyntheticMarketContextOrderConfig(
            structure_dir=args.structure_dir,
            output_dir=args.output_dir,
            phase_name=args.phase_name,
            basis_npz_path=args.basis_npz_path,
            basis_results_path=args.basis_results_path,
            layers=layers,
            num_workers=args.num_workers,
            cross_basis_overrides=cross_basis_overrides or None,
            cross_basis_layers=cross_basis_layers,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
