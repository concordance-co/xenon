from __future__ import annotations

import json

import pyarrow.parquet as pq

from pipelines.interp.synthetic_policy import SyntheticPolicyConfig, build_dataset, generate_dataset


def test_generate_policy_v4_dataset_expected_counts() -> None:
    config = SyntheticPolicyConfig(
        phase_name="actionability_algebra_v4_test",
        scenario_seeds=3,
        variant="v4",
    )
    examples = generate_dataset(config)
    assert len(examples) == 12
    assert {example.family for example in examples} == {"permission_grid"}
    assert {example.context_variant for example in examples} == {"market_only"}


def test_policy_v4_prompts_include_distractors_and_thresholds(tmp_path) -> None:
    result = build_dataset(
        SyntheticPolicyConfig(
            phase_name="actionability_algebra_v4_test",
            output_dir=tmp_path,
            scenario_seeds=1,
            variant="v4",
        )
    )

    assert result["summary"]["n_examples"] == 4

    prompt_lines = [
        json.loads(line)["user_prompt"]
        for line in (tmp_path / "synthetic_market_prompts.jsonl").read_text().splitlines()
    ]

    first_prompt = prompt_lines[0]
    assert "## PORTFOLIO CONTEXT" in first_prompt
    assert "## ACTIVE STRATEGIES" in first_prompt
    assert "## EXECUTION CONSTRAINTS" in first_prompt
    assert "Asset A" in first_prompt

    assert (
        "Reference reserve target from the prior rebalance plan"
        in first_prompt
        or "Realized fee burn over the previous maintenance window"
        in first_prompt
        or "Desk monitoring width for this slate"
        in first_prompt
    )
    assert (
        "Reference only: the previous slippage guard fired"
        in first_prompt
        or "Administrative review cadence remains"
        in first_prompt
        or "Archive note: the last manual remark window was"
        in first_prompt
    )

    tick_rows = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet").to_pylist()
    assert min(row["log_id"] for row in tick_rows) >= 2_145_000_000
