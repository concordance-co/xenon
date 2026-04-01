from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from research.synthetic_market.synthetic_market_behavior_runner import _extract_first_tool_call_fields
from research.synthetic_market.synthetic_market_behavior_analysis import (
    SyntheticMarketBehaviorAnalysisConfig,
    run_synthetic_market_behavior_analysis,
)


def test_extract_first_tool_call_fields_parses_buy_call():
    fields = _extract_first_tool_call_fields(
        '<think>alpha</think>\n\n<tool_call>\n{"name":"buy_token","arguments":{"token":"MORI","spend_pct":50.0,"strategy":"none","content":"x"}}\n</tool_call>'
    )

    assert fields["has_tool_call"] is True
    assert fields["tool_call_parse_ok"] is True
    assert fields["first_tool_name"] == "buy_token"
    assert fields["first_tool_token"] == "MORI"
    assert fields["first_tool_spend_pct"] == 50.0


def test_extract_first_tool_call_fields_parses_plain_json_tool_call_list():
    fields = _extract_first_tool_call_fields(
        '[{"name":"buy_token","arguments":{"token":"VEXA","spend_pct":25.0,"strategy":"none","content":"y"}}]'
    )

    assert fields["has_tool_call"] is True
    assert fields["tool_call_parse_ok"] is True
    assert fields["first_tool_name"] == "buy_token"
    assert fields["first_tool_token"] == "VEXA"
    assert fields["first_tool_spend_pct"] == 25.0


def test_behavior_analysis_computes_change_rates(tmp_path):
    baseline_dir = tmp_path / "baseline"
    intervention_dir = tmp_path / "intervention"
    output_dir = tmp_path / "output"
    baseline_dir.mkdir()
    intervention_dir.mkdir()

    baseline_rows = [
        {
            "log_id": 1,
            "example_id": "ex1",
            "family": "fam",
            "family_variant": "v1",
            "roster_key": "r00",
            "source_first_tool_name": None,
            "source_first_tool_token": None,
            "source_first_tool_spend_pct": None,
            "first_generated_token_id": 10,
            "first_generated_token_text": "A",
            "generated_text": "alpha",
            "generated_token_count": 2,
            "has_tool_call": True,
            "first_tool_name": "buy_token",
            "first_tool_token": "MORI",
            "first_tool_spend_pct": 50.0,
            "patch_stats_json": "{}",
        },
        {
            "log_id": 2,
            "example_id": "ex2",
            "family": "fam",
            "family_variant": "v2",
            "roster_key": "r01",
            "source_first_tool_name": None,
            "source_first_tool_token": None,
            "source_first_tool_spend_pct": None,
            "first_generated_token_id": 20,
            "first_generated_token_text": "B",
            "generated_text": "beta",
            "generated_token_count": 3,
            "has_tool_call": False,
            "first_tool_name": None,
            "first_tool_token": None,
            "first_tool_spend_pct": None,
            "patch_stats_json": "{}",
        },
    ]
    intervention_rows = [
        {
            "log_id": 1,
            "example_id": "ex1",
            "family": "fam",
            "family_variant": "v1",
            "roster_key": "r00",
            "pair_mode": "denoise",
            "source_first_tool_name": "buy_token",
            "source_first_tool_token": "VEXA",
            "source_first_tool_spend_pct": 25.0,
            "source_generated_token_count": 4,
            "first_generated_token_id": 10,
            "first_generated_token_text": "A",
            "generated_text": "alpha changed",
            "generated_token_count": 4,
            "has_tool_call": True,
            "first_tool_name": "buy_token",
            "first_tool_token": "VEXA",
            "first_tool_spend_pct": 25.0,
            "patch_stats_json": json.dumps(
                {
                    "4": {
                        "layer": 4,
                        "delta_norm_std": 1.5,
                        "mean_norm_before": 10.0,
                        "mean_norm_after": 12.0,
                        "mean_std_norm_before": 2.0,
                        "mean_std_norm_after": 2.5,
                        "selected_proj_norm_before": 3.0,
                    }
                }
            ),
        },
        {
            "log_id": 2,
            "example_id": "ex2",
            "family": "fam",
            "family_variant": "v2",
            "roster_key": "r01",
            "pair_mode": "denoise",
            "source_first_tool_name": "record_observation",
            "source_first_tool_token": None,
            "source_first_tool_spend_pct": None,
            "source_generated_token_count": 1,
            "first_generated_token_id": 21,
            "first_generated_token_text": "C",
            "generated_text": "beta",
            "generated_token_count": 1,
            "has_tool_call": True,
            "first_tool_name": "record_observation",
            "first_tool_token": None,
            "first_tool_spend_pct": None,
            "patch_stats_json": json.dumps(
                {
                    "4": {
                        "layer": 4,
                        "status": "skipped",
                        "reason": "token_span_out_of_bounds:1",
                    }
                }
            ),
        },
    ]

    pq.write_table(pa.Table.from_pylist(baseline_rows), baseline_dir / "metadata.parquet")
    pq.write_table(pa.Table.from_pylist(intervention_rows), intervention_dir / "metadata.parquet")

    result = run_synthetic_market_behavior_analysis(
        SyntheticMarketBehaviorAnalysisConfig(
            baseline_dir=baseline_dir,
            intervention_dir=intervention_dir,
            output_dir=output_dir,
        )
    )

    assert result["count"] == 2
    assert result["first_token_change_rate"] == 0.5
    assert result["text_change_rate"] == 0.5
    assert result["mean_generated_token_count_delta"] == 2.0
    assert result["tool_presence_change_rate"] == 0.5
    assert result["tool_name_change_rate"] == 0.5
    assert result["tool_token_change_rate"] == 0.5
    assert result["mean_tool_spend_pct_delta"] == 25.0
    assert result["patch_applied_rate"] == 0.5
    assert result["patch_skipped_rate"] == 0.5
    assert result["rows_with_patch_stats"] == 2
    assert result["mean_patch_delta_norm_std"] == 1.5
    assert result["family_variant_summary"]["v1"]["tool_token_change_rate"] == 1.0
    assert result["tool_token_change_rate_ci95"] is not None
    assert result["source_tool_name_match_rate_baseline"] == 0.5
    assert result["source_tool_name_match_rate_intervention"] == 1.0
    assert result["source_tool_name_match_rate_delta"] == 0.5
    assert result["source_tool_name_restorable_count"] == 1
    assert result["source_tool_name_restoration_rate"] == 1.0
    assert result["source_tool_name_backfire_rate"] == 0.0
    assert result["source_tool_token_match_rate_baseline"] == 0.0
    assert result["source_tool_token_match_rate_intervention"] == 1.0
    assert result["source_tool_token_match_rate_delta"] == 1.0
    assert result["source_tool_token_restorable_count"] == 1
    assert result["source_tool_token_restoration_rate"] == 1.0
    assert result["mean_source_tool_spend_pct_delta_baseline"] == 25.0
    assert result["mean_source_tool_spend_pct_delta_intervention"] == 0.0
    assert result["source_tool_spend_pct_improvement_rate"] == 1.0
    assert result["source_tool_spend_pct_full_restoration_rate"] == 1.0
    assert result["source_tool_spend_pct_backfire_rate"] == 0.0
    assert result["mean_source_tool_spend_pct_normalized_restoration"] == 1.0
    assert result["mean_source_generated_token_count_delta_baseline"] == 2.0
    assert result["mean_source_generated_token_count_delta_intervention"] == 0.0
    assert result["source_generated_token_count_improvement_rate"] == 1.0
    assert result["source_generated_token_count_full_restoration_rate"] == 1.0
    assert result["source_generated_token_count_backfire_rate"] == 0.0
    assert result["mean_source_generated_token_count_normalized_restoration"] == 1.0
    assert result["paired_row_count"] == 2
    assert result["pair_modes_present"] == ["denoise"]
    assert result["pair_mode_summary"]["denoise"]["source_tool_token_restoration_rate"] == 1.0

    saved = json.loads((output_dir / "results.json").read_text())
    assert saved["count"] == 2
