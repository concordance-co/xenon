from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[6]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ASSET_DIR = ROOT / "data" / "report_assets" / "synthetic_market_phase22_path_validation"

CONFIG: dict[str, Any] = {
    "axis_label": "Leader",
    "pair_metric": "vol_1h_max",
    "pair_mode": "denoise",
    "basis_state_key": "market_mean",
    "lesion_layer": 4,
    "rescue_layer": 40,
    "components_per_layer": 4,
    "batch_size": 32,
    "model": "Qwen/Qwen3-30B-A3B",
    "baseline_run_name": "phase22_leader_path_l40_lesion_v1",
    "intervention_run_name": "phase22_leader_path_l40_rescue_v1",
    "donor_run_name": "phase22_leader_path_l40_donors_v1",
    "baseline_app_id": "ap-M19n6Iw6nERYt4xiNjZh5y",
    "intervention_app_id": "ap-ynwBZaL3LfbofIl5xsO8x3",
    "donor_app_id": "ap-xcOmM10jIAJuIvrY1SDCUW",
    "analysis_results": Path("/tmp/phase22_leader_path_l40_compare_v1_results.json"),
    "analysis_metadata": Path("/tmp/phase22_leader_path_l40_compare_v1_metadata.parquet"),
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


def _build_tool_surface_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    baseline_tool_rows = sum(row.get("baseline_first_tool_name") is not None for row in rows)
    intervention_tool_rows = sum(row.get("intervention_first_tool_name") is not None for row in rows)
    source_tool_rows = sum(row.get("source_first_tool_name") is not None for row in rows)
    all_three_tool_rows = sum(
        row.get("baseline_first_tool_name") is not None
        and row.get("intervention_first_tool_name") is not None
        and row.get("source_first_tool_name") is not None
        for row in rows
    )
    source_length_cap_rows = sum(row.get("source_generated_token_count") == 15000 for row in rows)
    return {
        "baseline_tool_rows": baseline_tool_rows,
        "intervention_tool_rows": intervention_tool_rows,
        "source_tool_rows": source_tool_rows,
        "all_three_tool_rows": all_three_tool_rows,
        "source_length_cap_rows": source_length_cap_rows,
    }


def _build_main_read(metrics: dict[str, Any], counts: dict[str, int], tool_surface: dict[str, int], total_rows: int) -> str:
    if tool_surface["all_three_tool_rows"] <= max(5, int(0.2 * total_rows)):
        return (
            "Phase 22 does not give a clean action-choice path-validation result. The patching mechanics are "
            "clean, but the tool surface is too unstable at full scale: only "
            f"{tool_surface['all_three_tool_rows']} of {total_rows} rows reached parsed tool calls in lesion, "
            "rescue, and source generation. On the full metric table rescue looks worse than lesion, but that "
            "headline is dominated by rows that never reached a comparable tool call."
        )

    tool_restore = metrics.get("source_tool_token_restoration_rate")
    tool_backfire = metrics.get("source_tool_token_backfire_rate")
    spend_improve = metrics.get("source_tool_spend_pct_improvement_rate")
    length_restore = metrics.get("mean_source_generated_token_count_normalized_restoration")

    if (
        tool_restore is not None
        and tool_backfire is not None
        and tool_restore > tool_backfire
        and spend_improve is not None
        and spend_improve >= 0.5
    ):
        verdict = (
            "The rescue intervention restores source-like action behavior better than the lesion alone. "
            "That is positive path-validation evidence for a downstream carrier of the leader signal."
        )
    elif tool_restore is not None and tool_backfire is not None and tool_restore > 0:
        verdict = (
            "The rescue intervention produces partial restoration, but the path-validation result is mixed "
            "rather than clean."
        )
    else:
        verdict = (
            "The rescue intervention does not produce convincing restoration over the lesion baseline, so the "
            "current downstream path hypothesis is weak."
        )

    length_note = ""
    if length_restore is not None and length_restore < 0:
        length_note = (
            f" Response length remains unstable, with negative normalized restoration ({length_restore:.3f})."
        )
    count_note = (
        " Restorable tool-token rows: "
        f"{counts['tool_token_restored_count']} of {counts['tool_token_restorable_count']}."
    )
    return verdict + count_note + length_note


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    analysis = _load_json(CONFIG["analysis_results"])
    analysis_rows = _load_rows(CONFIG["analysis_metadata"])
    counts = _build_count_summary(analysis_rows)
    tool_surface = _build_tool_surface_summary(analysis_rows)

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
        metrics.get("mean_source_tool_spend_pct_gap_intervention", 0.0)
        - metrics.get("mean_source_tool_spend_pct_gap_baseline", 0.0)
    )
    metrics["source_generated_token_count_gap_delta"] = (
        metrics.get("mean_source_generated_token_count_gap_intervention", 0.0)
        - metrics.get("mean_source_generated_token_count_gap_baseline", 0.0)
    )

    methodology_scorecard = [
        {"item": "Matched prompt pairs and separately generated source behaviors", "status": "meets"},
        {"item": "Early lesion plus downstream rescue in the same request-scoped engine path", "status": "meets"},
        {"item": "Compiled non-eager custom-op execution with batch size 32", "status": "meets"},
        {"item": "Batched donor capture on Modal rather than one-by-one residual capture", "status": "meets"},
        {"item": "Per-row patch diagnostics with skip accounting", "status": "meets"},
        {"item": "Stable parsed tool-call surface across lesion, rescue, and source runs", "status": "missing"},
        {"item": "Downstream rescue gives cleaner action restoration than lesion alone", "status": "missing"},
        {"item": "Response length / verbosity also restores cleanly", "status": "missing"},
        {"item": "Further downstream path tracing beyond one rescue site", "status": "missing"},
    ]

    summary = {
        "date": "3 April 2026",
        "phase_name": "phase22_path_validation",
        "model": CONFIG["model"],
        "sample": {
            "count": len(analysis_rows),
            "pair_mode": CONFIG["pair_mode"],
            "pair_metric": CONFIG["pair_metric"],
            "basis_state_key": CONFIG["basis_state_key"],
            "engine_mode": "compiled non-eager custom-op",
            "context_variant": "market_only",
            "max_tokens": 15000,
            "batch_size": CONFIG["batch_size"],
            "selection_strategy": "ordered",
        },
        "experiment": {
            "axis_label": CONFIG["axis_label"],
            "lesion_layer": CONFIG["lesion_layer"],
            "rescue_layer": CONFIG["rescue_layer"],
            "components_per_layer": CONFIG["components_per_layer"],
            "baseline_run_name": CONFIG["baseline_run_name"],
            "intervention_run_name": CONFIG["intervention_run_name"],
            "donor_run_name": CONFIG["donor_run_name"],
            "baseline_app_id": CONFIG["baseline_app_id"],
            "intervention_app_id": CONFIG["intervention_app_id"],
            "donor_app_id": CONFIG["donor_app_id"],
        },
        "methodology": {
            "path_validation_explainer": (
                "Path validation asks whether a downstream rescue can undo the behavioral damage caused by an "
                "earlier lesion. Here the lesion removes the Leader subspace at L4, and the rescue reinserts "
                "source-side coefficients at L40."
            ),
            "lesion_explainer": (
                "The baseline condition is not an untouched prompt. It is the lesion-only run: project_out at "
                "L4 over the matched market span."
            ),
            "rescue_explainer": (
                "The intervention condition keeps the same L4 lesion but adds swap_components at L40 using "
                "donor means from the matched source rows."
            ),
            "scorecard": methodology_scorecard,
        },
        "overall": {
            "main_read": _build_main_read(metrics, counts, tool_surface, len(analysis_rows)),
        },
        "metrics": metrics,
        "counts": counts,
        "tool_surface": tool_surface,
    }

    (ASSET_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    _copy_if_exists(CONFIG["analysis_results"], ASSET_DIR / "comparison_results.json")
    _copy_if_exists(CONFIG["analysis_metadata"], ASSET_DIR / "comparison_metadata.parquet")


if __name__ == "__main__":
    main()
