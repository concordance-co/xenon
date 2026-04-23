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

ASSET_DIR = ROOT / "data" / "report_assets" / "synthetic_market_phase21_restoration"

AXIS_CONFIGS: dict[str, dict[str, Any]] = {
    "leader": {
        "label": "Leader",
        "short_label": "Leader (L4)",
        "pair_metric": "vol_1h_max",
        "pair_mode": "denoise",
        "target_layer": 4,
        "components_per_layer": 4,
        "basis_state_key": "market_mean",
        "batch_size": 16,
        "baseline_run_name": "phase21_restoration_leader_denoise_bs16_v1_baseline",
        "intervention_run_name": "phase21_restoration_leader_denoise_bs16_v1_swap_components",
        "baseline_app_id": "ap-lkJNRuw7IjrYQyLz7qOI41",
        "intervention_app_id": "ap-wg6R0jW9UUyuxOKDXmzhQe",
        "analysis_results": Path("/tmp/phase21_restoration_leader_denoise_bs16_v1_analysis_output/results.json"),
        "analysis_metadata": Path("/tmp/phase21_restoration_leader_denoise_bs16_v1_analysis_output/metadata.parquet"),
        "baseline_results": Path("/tmp/phase21_restoration_leader_denoise_bs16_v1_baseline_results.json"),
        "baseline_metadata": Path("/tmp/phase21_restoration_leader_denoise_bs16_v1_baseline_metadata.parquet"),
        "intervention_results": Path("/tmp/phase21_restoration_leader_denoise_bs16_v1_swap_components_results.json"),
        "intervention_metadata": Path("/tmp/phase21_restoration_leader_denoise_bs16_v1_swap_components_metadata.parquet"),
    },
    "dispersion": {
        "label": "Dispersion",
        "short_label": "Dispersion (L35)",
        "pair_metric": "pct_1h_mad",
        "pair_mode": "denoise",
        "target_layer": 35,
        "components_per_layer": 4,
        "basis_state_key": "market_mean",
        "batch_size": 16,
        "baseline_run_name": "phase21_restoration_dispersion_denoise_bs16_v1_baseline",
        "intervention_run_name": "phase21_restoration_dispersion_denoise_bs16_v1_swap_components",
        "baseline_app_id": "ap-0i5ROVAsweeynqaZvydtgE",
        "intervention_app_id": "ap-jgH7Zj6t8lnRe2fq018SME",
        "analysis_results": Path("/tmp/phase21_restoration_dispersion_denoise_bs16_v1_analysis_output/results.json"),
        "analysis_metadata": Path("/tmp/phase21_restoration_dispersion_denoise_bs16_v1_analysis_output/metadata.parquet"),
        "baseline_results": Path("/tmp/phase21_restoration_dispersion_denoise_bs16_v1_baseline_results.json"),
        "baseline_metadata": Path("/tmp/phase21_restoration_dispersion_denoise_bs16_v1_baseline_metadata.parquet"),
        "intervention_results": Path("/tmp/phase21_restoration_dispersion_denoise_bs16_v1_swap_components_results.json"),
        "intervention_metadata": Path("/tmp/phase21_restoration_dispersion_denoise_bs16_v1_swap_components_metadata.parquet"),
    },
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


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


def _build_decode_review(axis_id: str, config: dict[str, Any]) -> dict[str, Any]:
    baseline_rows = _load_rows(config["baseline_metadata"])
    intervention_rows = _load_rows(config["intervention_metadata"])

    def _scan(rows: list[dict[str, Any]], *, source: bool) -> dict[str, int]:
        if source:
            text_key = "source_generated_text"
            finish_key = "source_finish_reason"
            tool_key = "source_has_tool_call"
            parse_key = "source_tool_call_parse_ok"
            token_key = "source_generated_token_count"
        else:
            text_key = "generated_text"
            finish_key = "finish_reason"
            tool_key = "has_tool_call"
            parse_key = "tool_call_parse_ok"
            token_key = "generated_token_count"
        texts = [str(row.get(text_key) or "") for row in rows]
        return {
            "rows": len(rows),
            "tool_calls": sum(bool(row.get(tool_key)) for row in rows),
            "parse_ok": sum(bool(row.get(parse_key)) for row in rows),
            "finish_stop": sum(str(row.get(finish_key) or "") == "stop" for row in rows),
            "max_token_cap_hits": sum(int(row.get(token_key) or 0) >= 15000 for row in rows),
            "corruption_bangs": sum("!!!!" in text for text in texts),
            "missing_think_close": sum("<think>" in text and "</think>" not in text for text in texts),
        }

    return {
        "axis_id": axis_id,
        "baseline": {
            "base": _scan(baseline_rows, source=False),
            "source": _scan(baseline_rows, source=True),
        },
        "intervention": {
            "base": _scan(intervention_rows, source=False),
            "source": _scan(intervention_rows, source=True),
        },
    }


def _axis_main_read(axis_id: str, metrics: dict[str, Any], counts: dict[str, int]) -> str:
    if axis_id == "leader":
        return (
            "Leader remains the clearer restoration result after the clean rerun. Tool-token "
            "match improves, tool-token restoration beats backfire by a useful margin, and spend "
            "moves toward the source more often than not. Response length is still unstable."
        )
    if axis_id == "dispersion":
        return (
            "Dispersion looks cleaner than the old draft once the degenerate batch-32 run is removed, "
            "but it still does not restore the action surface cleanly. Tool-name stays saturated, "
            "tool-token match gets slightly worse overall, and restoration is about the same size as "
            "backfire. Spend does move toward the source on its small restorable subset, but response "
            "length remains materially worse than Leader."
        )
    raise ValueError(f"Unknown axis: {axis_id}")


def _build_axis_summary(axis_id: str, config: dict[str, Any]) -> dict[str, Any]:
    analysis = _load_json(config["analysis_results"])
    analysis_rows = _load_rows(config["analysis_metadata"])
    baseline_results = _load_json_if_exists(config["baseline_results"])
    intervention_results = _load_json_if_exists(config["intervention_results"])
    counts = _build_count_summary(analysis_rows)
    decode_review = _build_decode_review(axis_id, config)

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
        "basis_state_key": config["basis_state_key"],
        "metrics": metrics,
        "counts": counts,
        "decode_review": decode_review,
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
        "date": "3 April 2026",
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
            "basis_state_key": "market_mean",
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
                "This report replaces the earlier mixed-batch Phase 21 draft. Both axes were rerun on the "
                "same validated `batch_size = 16` path after the batch-32 tool-surface instability was isolated."
            ),
            "decode_review_note": (
                "All four rerun generations are clean: every base row and every source row reached a valid "
                "parsed tool call, all finish reasons are `stop`, and there are zero `!!!!` corruptions or "
                "15000-token cap hits."
            ),
            "scorecard": methodology_scorecard,
        },
        "overall": {
            "main_read": (
                "The validated bs16 rerun preserves the main scientific split from the old draft while fixing "
                "the decoding pathology. Leader still shows the stronger partial restoration signal on action "
                "choice and spend. Dispersion is no longer obviously broken at the decode level, but it remains "
                "weak on action restoration and poor on response length. Phase 22 should therefore stay "
                "leader-first rather than promoting dispersion into the main path-validation target."
            ),
            "best_axis_for_action_choice": "leader",
            "worst_axis_for_response_length": "dispersion",
            "phase22_recommendation": "leader_only",
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
