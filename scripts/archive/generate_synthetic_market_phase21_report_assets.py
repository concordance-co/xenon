from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ASSET_DIR = ROOT / "data" / "report_assets" / "synthetic_market_phase21_restoration"

AXIS_CONFIGS: dict[str, dict[str, Any]] = {
    "leader": {
        "label": "Leader",
        "short_label": "Leader (L4)",
        "pair_metric": "vol_1h_max",
        "pair_mode": "denoise",
        "target_layer": 4,
        "components_per_layer": 4,
        "batch_size": 8,
        "baseline_run_name": "phase21_restoration_baseline_leader_denoise_v3",
        "intervention_run_name": "phase21_restoration_swapcomponents_leader_denoise_v3",
        "baseline_app_id": "ap-oTncJihjTy7LtjY41itDlr",
        "intervention_app_id": "ap-hlOLdjWejZArX0SDs12IH0",
        "analysis_results": Path("/tmp/phase21_restoration_analysis_v3/output/results.json"),
        "analysis_metadata": Path("/tmp/phase21_restoration_analysis_v3/output/metadata.parquet"),
        "baseline_results": Path("/tmp/phase21_restoration_baseline_leader_denoise_v3_results.json"),
        "baseline_metadata": Path("/tmp/phase21_restoration_baseline_leader_denoise_v3_metadata.parquet"),
        "intervention_results": Path("/tmp/phase21_restoration_swapcomponents_leader_denoise_v3_results.json"),
        "intervention_metadata": Path("/tmp/phase21_restoration_swapcomponents_leader_denoise_v3_metadata.parquet"),
    },
    "dispersion": {
        "label": "Dispersion",
        "short_label": "Dispersion (L35)",
        "pair_metric": "pct_1h_mad",
        "pair_mode": "denoise",
        "target_layer": 35,
        "components_per_layer": 4,
        "batch_size": 32,
        "baseline_run_name": "phase21_restoration_baseline_dispersion_denoise_v1",
        "intervention_run_name": "phase21_restoration_swapcomponents_dispersion_denoise_v1",
        "baseline_app_id": "ap-DQIfivft1xogoXUVXhqDal",
        "intervention_app_id": "ap-wtpY11KtnoHWB4vnxNG1II",
        "analysis_results": Path("/tmp/phase21_restoration_dispersion_analysis_v1/output/results.json"),
        "analysis_metadata": Path("/tmp/phase21_restoration_dispersion_analysis_v1/output/metadata.parquet"),
        "baseline_results": Path("/tmp/phase21_restoration_baseline_dispersion_denoise_v1_results.json"),
        "baseline_metadata": Path("/tmp/phase21_restoration_baseline_dispersion_denoise_v1_metadata.parquet"),
        "intervention_results": Path("/tmp/phase21_restoration_swapcomponents_dispersion_denoise_v1_results.json"),
        "intervention_metadata": Path("/tmp/phase21_restoration_swapcomponents_dispersion_denoise_v1_metadata.parquet"),
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def _count_not_none(rows: list[dict[str, Any]], key: str) -> int:
    return sum(row.get(key) is not None for row in rows)


def _count_true(rows: list[dict[str, Any]], key: str) -> int:
    return sum(bool(row.get(key)) for row in rows if row.get(key) is not None)


def _build_count_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "tool_name_restorable_count": _count_not_none(rows, "source_tool_name_restored"),
        "tool_name_restored_count": _count_true(rows, "source_tool_name_restored"),
        "tool_name_backfire_pool": _count_not_none(rows, "source_tool_name_backfire"),
        "tool_name_backfire_count": _count_true(rows, "source_tool_name_backfire"),
        "tool_token_restorable_count": _count_not_none(rows, "source_tool_token_restored"),
        "tool_token_restored_count": _count_true(rows, "source_tool_token_restored"),
        "tool_token_backfire_pool": _count_not_none(rows, "source_tool_token_backfire"),
        "tool_token_backfire_count": _count_true(rows, "source_tool_token_backfire"),
        "spend_restorable_count": _count_not_none(rows, "source_tool_spend_pct_improved"),
        "spend_improved_count": _count_true(rows, "source_tool_spend_pct_improved"),
        "spend_full_restoration_count": _count_true(rows, "source_tool_spend_pct_restored"),
        "spend_backfire_pool": _count_not_none(rows, "source_tool_spend_pct_backfired"),
        "spend_backfire_count": _count_true(rows, "source_tool_spend_pct_backfired"),
        "generated_restorable_count": _count_not_none(rows, "source_generated_token_count_improved"),
        "generated_improved_count": _count_true(rows, "source_generated_token_count_improved"),
        "generated_full_restoration_count": _count_true(rows, "source_generated_token_count_restored"),
        "generated_backfire_pool": _count_not_none(rows, "source_generated_token_count_backfired"),
        "generated_backfire_count": _count_true(rows, "source_generated_token_count_backfired"),
    }


def _axis_main_read(axis_id: str, metrics: dict[str, Any], counts: dict[str, int]) -> str:
    if axis_id == "leader":
        return (
            "Leader is the clearer restoration result. Tool-token match improves, tool-token "
            "restoration beats backfire, and spend moves toward the source more often than not. "
            "Response length is still unstable."
        )
    if axis_id == "dispersion":
        return (
            "Dispersion does not restore the action surface cleanly. Tool-name and tool-token source "
            "match both get worse overall. Spend looks perfect on its restorable subset, but that subset "
            f"is only {counts['spend_restorable_count']} row, so it is not strong evidence by itself. "
            "Response length is much worse than Leader."
        )
    raise ValueError(f"Unknown axis: {axis_id}")


def _build_axis_summary(axis_id: str, config: dict[str, Any]) -> dict[str, Any]:
    analysis = _load_json(config["analysis_results"])
    analysis_rows = _load_rows(config["analysis_metadata"])
    baseline_results = _load_json(config["baseline_results"])
    intervention_results = _load_json(config["intervention_results"])
    counts = _build_count_summary(analysis_rows)

    metrics = dict(analysis)
    metrics["source_tool_token_match_rate_delta"] = (
        metrics.get("source_tool_token_match_rate_intervention", 0.0)
        - metrics.get("source_tool_token_match_rate_baseline", 0.0)
    )
    metrics["source_tool_name_match_rate_delta"] = (
        metrics.get("source_tool_name_match_rate_intervention", 0.0)
        - metrics.get("source_tool_name_match_rate_baseline", 0.0)
    )
    metrics["source_tool_spend_pct_gap_delta"] = (
        metrics.get("mean_source_tool_spend_pct_delta_intervention", 0.0)
        - metrics.get("mean_source_tool_spend_pct_delta_baseline", 0.0)
    )
    metrics["source_generated_token_count_gap_delta"] = (
        metrics.get("mean_source_generated_token_count_delta_intervention", 0.0)
        - metrics.get("mean_source_generated_token_count_delta_baseline", 0.0)
    )

    return {
        "axis_id": axis_id,
        "label": config["label"],
        "short_label": config["short_label"],
        "pair_metric": config["pair_metric"],
        "pair_mode": config["pair_mode"],
        "target_layer": config["target_layer"],
        "components_per_layer": config["components_per_layer"],
        "batch_size": config["batch_size"],
        "baseline_run_name": config["baseline_run_name"],
        "intervention_run_name": config["intervention_run_name"],
        "baseline_app_id": config["baseline_app_id"],
        "intervention_app_id": config["intervention_app_id"],
        "metrics": metrics,
        "counts": counts,
        "main_read": _axis_main_read(axis_id, metrics, counts),
        "baseline_results": baseline_results,
        "intervention_results": intervention_results,
    }


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    axes = {
        axis_id: _build_axis_summary(axis_id, config)
        for axis_id, config in AXIS_CONFIGS.items()
    }

    methodology_scorecard = [
        {"item": "Matched prompt pairs and separately generated source behaviors", "status": "meets"},
        {"item": "Source-driven coefficient swap rather than generic ablation", "status": "meets"},
        {"item": "Compiled non-eager custom-op execution path for the patch itself", "status": "meets"},
        {"item": "Per-row patch diagnostics with zero skips on both axes", "status": "meets"},
        {"item": "Exact token-by-token source activation transplant", "status": "missing"},
        {"item": "Clean restoration of action choice on every tested axis", "status": "partial"},
        {"item": "Clean restoration of response length / verbosity", "status": "missing"},
        {"item": "Path-level validation", "status": "missing"},
    ]

    summary = {
        "date": "28 March 2026",
        "phase_name": "phase21_restoration",
        "model": "Qwen/Qwen3-30B-A3B",
        "sample": {
            "count_per_axis": 48,
            "pair_mode": "denoise",
            "patch_mode": "swap_components",
            "engine_mode": "compiled non-eager custom-op",
            "context_variant": "market_only",
            "max_tokens": 15000,
            "selection_strategy": "ordered",
            "leader_batch_size": axes["leader"]["batch_size"],
            "dispersion_batch_size": axes["dispersion"]["batch_size"],
        },
        "methodology": {
            "pair_mode_explainer": (
                "Each row belongs to a matched pair. In denoise mode, the base row is the lower-valued "
                "member of the pair on the chosen metric, and the source row is the higher-valued member."
            ),
            "swap_components_explainer": (
                "swap_components does not copy every activation token by token. It captures the source row, "
                "averages over the market span, and swaps the selected source coefficients into the base row "
                "over the base market span."
            ),
            "batch_note": (
                "Leader was run earlier at batch size 8. Dispersion was run later at batch size 32 after "
                "the new default was adopted. Both use the same compiled non-eager patch path."
            ),
            "scorecard": methodology_scorecard,
        },
        "overall": {
            "main_read": (
                "Phase 21 is no longer a single-axis story. The Leader axis shows real partial restoration "
                "on tool-token choice and some positive movement on spend, while the Dispersion axis is weak "
                "or negative on tool selection and only clearly positive on spend for a one-row restorable subset. "
                "Both axes remain poor on response length."
            ),
            "best_axis_for_action_choice": "leader",
            "worst_axis_for_response_length": "dispersion",
        },
        "axes": axes,
    }

    (ASSET_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    for axis_id, config in AXIS_CONFIGS.items():
        _copy_if_exists(config["analysis_results"], ASSET_DIR / f"{axis_id}_analysis_results.json")
        _copy_if_exists(config["analysis_metadata"], ASSET_DIR / f"{axis_id}_analysis_metadata.parquet")
        _copy_if_exists(config["baseline_results"], ASSET_DIR / f"{axis_id}_baseline_results.json")
        _copy_if_exists(config["baseline_metadata"], ASSET_DIR / f"{axis_id}_baseline_metadata.parquet")
        _copy_if_exists(
            config["intervention_results"],
            ASSET_DIR / f"{axis_id}_intervention_results.json",
        )
        _copy_if_exists(
            config["intervention_metadata"],
            ASSET_DIR / f"{axis_id}_intervention_metadata.parquet",
        )


if __name__ == "__main__":
    main()
