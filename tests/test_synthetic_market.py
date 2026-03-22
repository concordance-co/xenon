from __future__ import annotations

import json

import pyarrow.parquet as pq

from pipelines.interp.synthetic_market import (
    ARCHETYPES,
    SyntheticMarketConfig,
    build_synthetic_market_dataset,
    generate_dataset,
)


def test_generate_dataset_expected_family_counts() -> None:
    config = SyntheticMarketConfig(
        scalar_steps=5,
        pairwise_variants=4,
        archetype_variants=2,
        include_settings_variants=True,
    )
    examples = generate_dataset(config)

    scalar_expected = 3 * 5 * 3
    pairwise_expected = 3 * 4 * 3
    archetype_expected = len(ARCHETYPES) * 2 * 3
    assert len(examples) == scalar_expected + pairwise_expected + archetype_expected

    families = {example.family for example in examples}
    assert families == {"scalar_sweep", "pairwise_tradeoff", "archetype_family"}

    contexts = {example.context_variant for example in examples}
    assert contexts == {"market_only", "low_risk", "high_risk"}


def test_generate_phase2_geometry_dataset_expected_counts(tmp_path) -> None:
    config = SyntheticMarketConfig(
        dataset_preset="phase2_geometry",
        scalar_steps=5,
        scalar_background_variants=2,
        minimal_scalar_templates=1,
        include_settings_variants=False,
    )
    examples = generate_dataset(config)
    assert len(examples) == 45
    families = {example.family for example in examples}
    assert families == {"scalar_sweep_dense", "scalar_sweep_minimal"}
    contexts = {example.context_variant for example in examples}
    assert contexts == {"market_only"}

    build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=tmp_path,
            dataset_preset="phase2_geometry",
            scalar_steps=3,
            scalar_background_variants=1,
            minimal_scalar_templates=1,
            include_settings_variants=False,
        )
    )
    tick_rows = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet").to_pylist()
    assert min(row["log_id"] for row in tick_rows) >= 2_100_000_000


def test_generate_phase3_coupled_geometry_dataset_expected_counts(tmp_path) -> None:
    config = SyntheticMarketConfig(
        dataset_preset="phase3_coupled_geometry",
        coupled_grid_steps=5,
        coupled_background_variants=2,
        coupled_minimal_templates=1,
        include_settings_variants=False,
    )
    examples = generate_dataset(config)
    assert len(examples) == (3 * 2 * 25) + (3 * 1 * 25)
    families = {example.family for example in examples}
    assert families == {"coupled_factor_dense", "coupled_factor_minimal"}
    contexts = {example.context_variant for example in examples}
    assert contexts == {"market_only"}

    build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=tmp_path,
            dataset_preset="phase3_coupled_geometry",
            coupled_grid_steps=5,
            coupled_background_variants=1,
            coupled_minimal_templates=1,
            include_settings_variants=False,
        )
    )
    tick_rows = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet").to_pylist()
    assert min(row["log_id"] for row in tick_rows) >= 2_130_000_000


def test_generate_phase4_market_representation_dataset_expected_counts(tmp_path) -> None:
    config = SyntheticMarketConfig(
        dataset_preset="phase4_market_representation",
        representation_steps=5,
        representation_background_variants=2,
        include_settings_variants=False,
    )
    examples = generate_dataset(config)
    assert len(examples) == (3 * 2 * 5) + (2 * 2)
    families = {example.family for example in examples}
    assert families == {"pairwise_tradeoff_hard", "rank_context_tradeoff"}
    contexts = {example.context_variant for example in examples}
    assert contexts == {"market_only"}

    build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=tmp_path,
            dataset_preset="phase4_market_representation",
            representation_steps=5,
            representation_background_variants=2,
            include_settings_variants=False,
        )
    )
    tick_rows = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet").to_pylist()
    assert min(row["log_id"] for row in tick_rows) >= 2_147_300_000


def test_generate_phase5_symbol_permutation_dataset_expected_counts(tmp_path) -> None:
    config = SyntheticMarketConfig(
        dataset_preset="phase5_symbol_permutation",
        permutation_variants=4,
        include_settings_variants=False,
    )
    examples = generate_dataset(config)
    assert len(examples) == 2 * 4
    families = {example.family for example in examples}
    assert families == {"symbol_permutation_control"}
    contexts = {example.context_variant for example in examples}
    assert contexts == {"market_only"}

    build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=tmp_path,
            dataset_preset="phase5_symbol_permutation",
            permutation_variants=4,
            include_settings_variants=False,
        )
    )
    tick_rows = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet").to_pylist()
    asset_rows = pq.read_table(tmp_path / "synthetic_market_asset_records.parquet").to_pylist()
    assert min(row["log_id"] for row in tick_rows) >= 2_146_900_000
    assert all(row["profile_id"] is not None for row in asset_rows)


def test_generate_phase6_profile_invariance_dataset_expected_counts(tmp_path) -> None:
    config = SyntheticMarketConfig(
        dataset_preset="phase6_profile_invariance",
        permutation_variants=3,
        profile_surface_variants=2,
        include_settings_variants=False,
    )
    examples = generate_dataset(config)
    assert len(examples) == 2 * 3 * 2
    families = {example.family for example in examples}
    assert families == {"profile_invariance_control"}
    contexts = {example.context_variant for example in examples}
    assert contexts == {"market_only"}

    build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=tmp_path,
            dataset_preset="phase6_profile_invariance",
            permutation_variants=3,
            profile_surface_variants=2,
            include_settings_variants=False,
        )
    )
    tick_rows = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet").to_pylist()
    assert min(row["log_id"] for row in tick_rows) >= 2_146_950_000


def test_phase6_profile_invariance_emits_distinct_surface_styles() -> None:
    examples = generate_dataset(
        SyntheticMarketConfig(
            dataset_preset="phase6_profile_invariance",
            permutation_variants=1,
            profile_surface_variants=4,
            include_settings_variants=False,
        )
    )
    prompts = [example.user_prompt for example in examples if example.family_variant == "participation_concentration_tiebreak"]
    assert any("Snapshot:" in prompt for prompt in prompts)
    assert any("Holder concentration (top 20)" in prompt for prompt in prompts)
    assert any("Short-horizon price move" in prompt for prompt in prompts)


def test_rank_context_tradeoff_preserves_focal_pair_across_backgrounds() -> None:
    examples = generate_dataset(
        SyntheticMarketConfig(
            dataset_preset="phase4_market_representation",
            representation_steps=5,
            representation_background_variants=3,
            include_settings_variants=False,
        )
    )
    fixed_pair_rows = [
        example for example in examples
        if example.family == "rank_context_tradeoff"
        and example.family_variant.startswith("fixed_momentum_flow_pair")
    ]
    assert len(fixed_pair_rows) == 3

    focal_metrics = [
        (
            example.assets[0].pct_5m,
            example.assets[0].net_flow_5m,
            example.assets[0].top20_holder_pct,
            example.assets[1].pct_5m,
            example.assets[1].net_flow_5m,
            example.assets[1].top20_holder_pct,
        )
        for example in fixed_pair_rows
    ]
    assert len(set(focal_metrics)) == 1


def test_symbol_permutation_control_relabels_profiles_across_variants() -> None:
    examples = generate_dataset(
        SyntheticMarketConfig(
            dataset_preset="phase5_symbol_permutation",
            permutation_variants=4,
            include_settings_variants=False,
        )
    )
    first_scenario = [
        example for example in examples
        if example.family_variant == "momentum_flow_permuted_market"
    ]
    assert len(first_scenario) == 4

    profile_symbol_pairs = [
        tuple((asset.profile_id, asset.symbol) for asset in example.assets)
        for example in first_scenario
    ]
    assert len(set(profile_symbol_pairs)) == len(first_scenario)


def test_prompts_use_neutral_asset_symbols() -> None:
    config = SyntheticMarketConfig(
        scalar_steps=3,
        pairwise_variants=3,
        archetype_variants=1,
        include_settings_variants=False,
    )
    examples = generate_dataset(config)
    prompt = examples[0].user_prompt
    assert "Asset A" in prompt
    assert "Asset B" in prompt
    assert "POOPCOIN" not in prompt
    assert "HOTDOGZ" not in prompt


def test_settings_variants_change_some_labels() -> None:
    config = SyntheticMarketConfig(
        scalar_steps=3,
        pairwise_variants=3,
        archetype_variants=1,
        include_settings_variants=True,
    )
    examples = generate_dataset(config)

    by_example: dict[str, dict[str, str | None]] = {}
    for example in examples:
        by_example.setdefault(example.example_id, {})[example.context_variant] = example.labels["best_asset"]

    changed = 0
    for variants in by_example.values():
        if {"market_only", "low_risk", "high_risk"} <= set(variants):
            if variants["low_risk"] != variants["high_risk"]:
                changed += 1
    assert changed >= 1


def test_synthetic_outputs_include_stable_log_ids_and_messages(tmp_path) -> None:
    build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=tmp_path,
            scalar_steps=3,
            pairwise_variants=3,
            archetype_variants=1,
            include_settings_variants=False,
        )
    )
    tick_rows = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet").to_pylist()
    log_ids = [row["log_id"] for row in tick_rows]
    assert len(log_ids) == len(set(log_ids))
    assert min(log_ids) >= 2_000_000_000
    first_messages = json.loads(tick_rows[0]["prompt_messages_json"])
    assert [message["role"] for message in first_messages] == ["system", "user"]


def test_build_synthetic_market_dataset_writes_expected_outputs(tmp_path) -> None:
    result = build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=tmp_path,
            scalar_steps=3,
            pairwise_variants=3,
            archetype_variants=1,
            include_settings_variants=False,
        )
    )

    summary = result["summary"]
    assert summary["n_examples"] > 0
    assert summary["context_variants"] == ["market_only"]

    tick_table = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet")
    asset_table = pq.read_table(tmp_path / "synthetic_market_asset_records.parquet")
    pairwise_table = pq.read_table(tmp_path / "synthetic_market_pairwise_records.parquet")

    assert tick_table.num_rows == summary["n_tick_rows"]
    assert asset_table.num_rows == summary["n_asset_rows"]
    assert pairwise_table.num_rows == summary["n_pairwise_rows"]
    assert "log_id" in tick_table.column_names
    assert "prompt_messages_json" in tick_table.column_names
    assert "log_id" in asset_table.column_names
    assert "log_id" in pairwise_table.column_names

    summary_json = json.loads((tmp_path / "synthetic_market_summary.json").read_text())
    assert summary_json["n_examples"] == summary["n_examples"]
