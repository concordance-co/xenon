from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SECTION_ORDER = (
    "preamble",
    "market",
    "active_strategies",
    "active_settings",
    "portfolio",
    "constraints",
    "price_impact_limits",
    "prev_decisions",
    "last_token",
)


@dataclass(slots=True)
class SyntheticMarketPatchingAnalysisConfig:
    baseline_dir: Path
    intervention_dir: Path
    output_dir: Path
    control_dir: Path | None = None
    min_layer: int = 0
    top_k: int = 20
    basis_npz_path: Path | None = None
    basis_state_key: str = "market_mean"
    basis_components: int = 4


def classify_state_key(key: str) -> dict[str, Any]:
    if key == "last_token":
        return {
            "state_key": key,
            "state_kind": "terminal",
            "section_name": "last_token",
            "pooling": "last_token",
            "row_index": None,
            "sort_group": SECTION_ORDER.index("last_token"),
        }

    row_match = re.match(r"^(row_(mean|eos))_(\d+)$", key)
    if row_match:
        pooling = row_match.group(2)
        row_index = int(row_match.group(3))
        return {
            "state_key": key,
            "state_kind": "row",
            "section_name": "market",
            "pooling": pooling,
            "row_index": row_index,
            "sort_group": SECTION_ORDER.index("market"),
        }

    section_match = re.match(r"^([a-z_]+)_(mean|eos)$", key)
    if section_match:
        section_name = section_match.group(1)
        pooling = section_match.group(2)
        return {
            "state_key": key,
            "state_kind": "section",
            "section_name": section_name,
            "pooling": pooling,
            "row_index": None,
            "sort_group": SECTION_ORDER.index(section_name) if section_name in SECTION_ORDER else len(SECTION_ORDER),
        }

    return {
        "state_key": key,
        "state_kind": "other",
        "section_name": "other",
        "pooling": "other",
        "row_index": None,
        "sort_group": len(SECTION_ORDER) + 1,
    }


def _load_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return pq.read_table(path).to_pylist()


def _load_pooled(path: Path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file

    if not path.exists():
        raise FileNotFoundError(path)
    return load_file(str(path))


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    out = np.zeros((a.shape[0],), dtype=np.float64)
    valid = denom > 1e-12
    if np.any(valid):
        out[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return out


def _safe_ratio(num: float, denom: float) -> float | None:
    if abs(denom) < 1e-12:
        return None
    return float(num / denom)


def _projection_scores(vec: np.ndarray, *, mean: np.ndarray, scale: np.ndarray, components: np.ndarray) -> np.ndarray:
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    standardized = (vec - mean) / safe_scale
    return standardized @ components.T


def _parse_patch_stats(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _mean_abs(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(abs(v) for v in values) / len(values))


def run_synthetic_market_patching_analysis(config: SyntheticMarketPatchingAnalysisConfig) -> dict[str, Any]:
    baseline_meta = _load_metadata(config.baseline_dir / "metadata.parquet")
    intervention_meta = _load_metadata(config.intervention_dir / "metadata.parquet")
    control_meta = _load_metadata(config.control_dir / "metadata.parquet") if config.control_dir is not None else []

    baseline_ids = {int(row["log_id"]) for row in baseline_meta}
    intervention_by_id = {int(row["log_id"]): row for row in intervention_meta}
    control_by_id = {int(row["log_id"]): row for row in control_meta}

    common_ids = sorted(baseline_ids & intervention_by_id.keys())
    if config.control_dir is not None:
        common_ids = sorted(set(common_ids) & control_by_id.keys())

    if not common_ids:
        raise ValueError("No shared log_ids between baseline/intervention/control runs")

    basis = None
    if config.basis_npz_path is not None:
        basis = np.load(config.basis_npz_path)

    state_acc: dict[tuple[str, int], dict[str, Any]] = {}
    intervention_patch_acc: dict[int, dict[str, Any]] = {}
    control_patch_acc: dict[int, dict[str, Any]] = {}

    for log_id in common_ids:
        baseline = _load_pooled(config.baseline_dir / "residual" / f"{log_id}.safetensors")
        intervention = _load_pooled(config.intervention_dir / "residual" / f"{log_id}.safetensors")
        control = (
            _load_pooled(config.control_dir / "residual" / f"{log_id}.safetensors")
            if config.control_dir is not None
            else None
        )

        shared_keys = set(baseline) & set(intervention)
        if control is not None:
            shared_keys &= set(control)

        for key in sorted(shared_keys):
            base = np.asarray(baseline[key], dtype=np.float32)
            intv = np.asarray(intervention[key], dtype=np.float32)
            ctrl = np.asarray(control[key], dtype=np.float32) if control is not None else None
            if base.shape != intv.shape or base.ndim != 2:
                continue
            if ctrl is not None and ctrl.shape != base.shape:
                continue

            delta_intv = np.linalg.norm(intv - base, axis=1)
            cosine_intv = _safe_cosine(base, intv)
            delta_ctrl = np.linalg.norm(ctrl - base, axis=1) if ctrl is not None else None
            cosine_ctrl = _safe_cosine(base, ctrl) if ctrl is not None else None
            base_norm = np.linalg.norm(base, axis=1)

            state_info = classify_state_key(key)
            for layer_idx in range(base.shape[0]):
                if layer_idx < int(config.min_layer):
                    continue
                acc = state_acc.setdefault(
                    (key, int(layer_idx)),
                    {
                        **state_info,
                        "layer": int(layer_idx),
                        "count": 0,
                        "sum_delta_intervention": 0.0,
                        "sum_delta_control": 0.0,
                        "sum_cosine_intervention": 0.0,
                        "sum_cosine_control": 0.0,
                        "sum_base_norm": 0.0,
                        "sum_basis_shift_intervention": 0.0,
                        "sum_basis_shift_control": 0.0,
                        "basis_count": 0,
                    },
                )
                acc["count"] += 1
                acc["sum_delta_intervention"] += float(delta_intv[layer_idx])
                acc["sum_cosine_intervention"] += float(cosine_intv[layer_idx])
                acc["sum_base_norm"] += float(base_norm[layer_idx])
                if delta_ctrl is not None and cosine_ctrl is not None:
                    acc["sum_delta_control"] += float(delta_ctrl[layer_idx])
                    acc["sum_cosine_control"] += float(cosine_ctrl[layer_idx])
                if basis is not None:
                    basis_prefix = f"{config.basis_state_key}_layer_{layer_idx}"
                    comp_key = f"{basis_prefix}__components"
                    mean_key = f"{basis_prefix}__mean"
                    scale_key = f"{basis_prefix}__scale"
                    if comp_key in basis and mean_key in basis and scale_key in basis:
                        components = basis[comp_key][: int(config.basis_components)]
                        mean = basis[mean_key]
                        scale = basis[scale_key]
                        base_scores = _projection_scores(base[layer_idx], mean=mean, scale=scale, components=components)
                        intv_scores = _projection_scores(intv[layer_idx], mean=mean, scale=scale, components=components)
                        acc["sum_basis_shift_intervention"] += float(np.mean(np.abs(intv_scores - base_scores)))
                        if ctrl is not None:
                            ctrl_scores = _projection_scores(ctrl[layer_idx], mean=mean, scale=scale, components=components)
                            acc["sum_basis_shift_control"] += float(np.mean(np.abs(ctrl_scores - base_scores)))
                        acc["basis_count"] += 1

        for raw_layer, stats in _parse_patch_stats(intervention_by_id[log_id].get("patch_stats_json")).items():
            layer = int(raw_layer)
            acc = intervention_patch_acc.setdefault(layer, {"count": 0, "delta_norm_raw": 0.0, "selected_proj_norm_before": 0.0, "selected_coeff_after_abs_max": []})
            acc["count"] += 1
            acc["delta_norm_raw"] += float(stats.get("delta_norm_raw", 0.0))
            acc["selected_proj_norm_before"] += float(stats.get("selected_proj_norm_before", 0.0))
            coeff_after = stats.get("selected_coeff_after")
            if isinstance(coeff_after, list):
                acc["selected_coeff_after_abs_max"].append(max(abs(float(v)) for v in coeff_after) if coeff_after else 0.0)

        if config.control_dir is not None:
            for raw_layer, stats in _parse_patch_stats(control_by_id[log_id].get("patch_stats_json")).items():
                layer = int(raw_layer)
                acc = control_patch_acc.setdefault(layer, {"count": 0, "delta_norm_raw": 0.0, "selected_proj_norm_before": 0.0, "selected_coeff_after_abs_max": []})
                acc["count"] += 1
                acc["delta_norm_raw"] += float(stats.get("delta_norm_raw", 0.0))
                acc["selected_proj_norm_before"] += float(stats.get("selected_proj_norm_before", 0.0))
                coeff_after = stats.get("selected_coeff_after")
                if isinstance(coeff_after, list):
                    acc["selected_coeff_after_abs_max"].append(max(abs(float(v)) for v in coeff_after) if coeff_after else 0.0)

    summary_rows: list[dict[str, Any]] = []
    for (_, _), acc in sorted(state_acc.items(), key=lambda item: (item[1]["sort_group"], item[1]["state_kind"], item[1]["state_key"], item[1]["layer"])):
        count = max(1, int(acc["count"]))
        mean_delta_intervention = float(acc["sum_delta_intervention"] / count)
        mean_cosine_intervention = float(acc["sum_cosine_intervention"] / count)
        mean_delta_control = float(acc["sum_delta_control"] / count) if config.control_dir is not None else None
        mean_cosine_control = float(acc["sum_cosine_control"] / count) if config.control_dir is not None else None
        mean_base_norm = float(acc["sum_base_norm"] / count)
        mean_basis_shift_intervention = (
            float(acc["sum_basis_shift_intervention"] / max(1, int(acc["basis_count"])))
            if acc["basis_count"] > 0
            else None
        )
        mean_basis_shift_control = (
            float(acc["sum_basis_shift_control"] / max(1, int(acc["basis_count"])))
            if config.control_dir is not None and acc["basis_count"] > 0
            else None
        )
        delta_margin = (
            float(mean_delta_intervention - mean_delta_control)
            if mean_delta_control is not None
            else None
        )
        row = {
            "state_key": acc["state_key"],
            "state_kind": acc["state_kind"],
            "section_name": acc["section_name"],
            "pooling": acc["pooling"],
            "row_index": acc["row_index"],
            "layer": int(acc["layer"]),
            "count": count,
            "mean_base_norm": mean_base_norm,
            "mean_delta_intervention": mean_delta_intervention,
            "mean_delta_control": mean_delta_control,
            "delta_margin": delta_margin,
            "delta_ratio": _safe_ratio(mean_delta_intervention, mean_delta_control) if mean_delta_control is not None else None,
            "mean_cosine_intervention": mean_cosine_intervention,
            "mean_cosine_control": mean_cosine_control,
            "mean_basis_shift_intervention": mean_basis_shift_intervention,
            "mean_basis_shift_control": mean_basis_shift_control,
            "basis_shift_margin": (
                float(mean_basis_shift_intervention - mean_basis_shift_control)
                if mean_basis_shift_intervention is not None and mean_basis_shift_control is not None
                else None
            ),
        }
        summary_rows.append(row)

    patch_rows: list[dict[str, Any]] = []
    for layer in sorted(intervention_patch_acc):
        intv = intervention_patch_acc[layer]
        ctrl = control_patch_acc.get(layer)
        intv_count = max(1, int(intv["count"]))
        row = {
            "layer": int(layer),
            "intervention_count": intv_count,
            "intervention_delta_norm_raw": float(intv["delta_norm_raw"] / intv_count),
            "intervention_selected_proj_norm_before": float(intv["selected_proj_norm_before"] / intv_count),
            "intervention_selected_coeff_after_abs_max": _mean_abs(intv["selected_coeff_after_abs_max"]),
            "control_count": int(ctrl["count"]) if ctrl is not None else None,
            "control_delta_norm_raw": float(ctrl["delta_norm_raw"] / max(1, int(ctrl["count"]))) if ctrl is not None else None,
            "control_selected_proj_norm_before": float(ctrl["selected_proj_norm_before"] / max(1, int(ctrl["count"]))) if ctrl is not None else None,
            "control_selected_coeff_after_abs_max": _mean_abs(ctrl["selected_coeff_after_abs_max"]) if ctrl is not None else None,
        }
        patch_rows.append(row)

    section_rows = [row for row in summary_rows if row["state_kind"] != "row"]
    row_rows = [row for row in summary_rows if row["state_kind"] == "row"]
    top_section_rows = sorted(
        [row for row in section_rows if row["delta_margin"] is not None],
        key=lambda row: (row["delta_margin"], row["mean_delta_intervention"]),
        reverse=True,
    )[: config.top_k]
    top_section_basis_rows = sorted(
        [row for row in section_rows if row["basis_shift_margin"] is not None],
        key=lambda row: (row["basis_shift_margin"], row["mean_basis_shift_intervention"]),
        reverse=True,
    )[: config.top_k]
    top_row_rows = sorted(
        [row for row in row_rows if row["delta_margin"] is not None],
        key=lambda row: (row["delta_margin"], row["mean_delta_intervention"]),
        reverse=True,
    )[: config.top_k]
    top_row_basis_rows = sorted(
        [row for row in row_rows if row["basis_shift_margin"] is not None],
        key=lambda row: (row["basis_shift_margin"], row["mean_basis_shift_intervention"]),
        reverse=True,
    )[: config.top_k]

    direct_target = next(
        (
            row
            for row in summary_rows
            if row["state_key"] == "market_mean" and row["layer"] == 4
        ),
        None,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(summary_rows), config.output_dir / "state_layer_summary.parquet", compression="snappy")
    if patch_rows:
        pq.write_table(pa.Table.from_pylist(patch_rows), config.output_dir / "patch_stats_summary.parquet", compression="snappy")

    result = {
        "baseline_dir": str(config.baseline_dir),
        "intervention_dir": str(config.intervention_dir),
        "control_dir": str(config.control_dir) if config.control_dir is not None else None,
        "num_shared_examples": len(common_ids),
        "shared_log_ids": common_ids,
        "direct_target_market_mean_l4": direct_target,
        "patch_stats_summary": patch_rows,
        "top_section_state_layers": top_section_rows,
        "top_section_basis_state_layers": top_section_basis_rows,
        "top_row_state_layers": top_row_rows,
        "top_row_basis_state_layers": top_row_basis_rows,
        "num_state_layer_rows": len(summary_rows),
    }
    (config.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze synthetic market patching runs against a baseline structure directory.")
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--intervention-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-layer", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--basis-npz-path", type=Path, default=None)
    parser.add_argument("--basis-state-key", default="market_mean")
    parser.add_argument("--basis-components", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    result = run_synthetic_market_patching_analysis(
        SyntheticMarketPatchingAnalysisConfig(
            baseline_dir=args.baseline_dir,
            intervention_dir=args.intervention_dir,
            control_dir=args.control_dir,
            output_dir=args.output_dir,
            min_layer=int(args.min_layer),
            top_k=int(args.top_k),
            basis_npz_path=args.basis_npz_path,
            basis_state_key=args.basis_state_key,
            basis_components=int(args.basis_components),
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
