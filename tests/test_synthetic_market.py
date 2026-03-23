from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq

from pipelines.interp.synthetic_market import (
    ARCHETYPES,
    SET_GEOMETRY_SCENARIOS,
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


def test_generate_phase10_set_geometry_context_dataset_expected_counts(tmp_path) -> None:
    config = SyntheticMarketConfig(
        dataset_preset="phase10_set_geometry_context",
        permutation_variants=3,
        profile_surface_variants=2,
        relation_scale_variants=2,
        include_settings_variants=True,
    )
    examples = generate_dataset(config)
    assert len(examples) == 4 * 3 * 2 * 2 * 3
    families = {example.family for example in examples}
    assert families == {"set_geometry_control"}
    contexts = {example.context_variant for example in examples}
    assert contexts == {"market_only", "low_risk", "high_risk"}

    build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=tmp_path,
            dataset_preset="phase10_set_geometry_context",
            permutation_variants=3,
            profile_surface_variants=2,
            relation_scale_variants=2,
            include_settings_variants=True,
        )
    )
    tick_rows = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet").to_pylist()
    assert min(row["log_id"] for row in tick_rows) >= 2_147_150_000


def test_generate_phase11_set_geometry_risk_ladder_dataset_expected_counts(tmp_path) -> None:
    config = SyntheticMarketConfig(
        dataset_preset="phase11_set_geometry_risk_ladder",
        permutation_variants=3,
        profile_surface_variants=2,
        relation_scale_variants=2,
        include_settings_variants=True,
    )
    examples = generate_dataset(config)
    assert len(examples) == 4 * 3 * 2 * 2 * 6
    families = {example.family for example in examples}
    assert families == {"set_geometry_control"}
    contexts = {example.context_variant for example in examples}
    assert contexts == {"market_only", "risk_1", "risk_2", "risk_3", "risk_4", "risk_5"}

    build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=tmp_path,
            dataset_preset="phase11_set_geometry_risk_ladder",
            permutation_variants=3,
            profile_surface_variants=2,
            relation_scale_variants=2,
            include_settings_variants=True,
        )
    )
    tick_rows = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet").to_pylist()
    assert min(row["log_id"] for row in tick_rows) >= 2_147_200_000


def test_generate_phase13_set_geometry_portfolio_ladder_dataset_expected_counts(tmp_path) -> None:
    config = SyntheticMarketConfig(
        dataset_preset="phase13_set_geometry_portfolio_ladder",
        permutation_variants=3,
        profile_surface_variants=2,
        relation_scale_variants=2,
        include_settings_variants=True,
    )
    examples = generate_dataset(config)
    assert len(examples) == 4 * 3 * 2 * 2 * 6
    families = {example.family for example in examples}
    assert families == {"set_geometry_control"}
    contexts = {example.context_variant for example in examples}
    assert contexts == {"market_only", "portfolio_1", "portfolio_2", "portfolio_3", "portfolio_4", "portfolio_5"}

    build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=tmp_path,
            dataset_preset="phase13_set_geometry_portfolio_ladder",
            permutation_variants=3,
            profile_surface_variants=2,
            relation_scale_variants=2,
            include_settings_variants=True,
        )
    )
    tick_rows = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet").to_pylist()
    assert min(row["log_id"] for row in tick_rows) >= 2_147_240_000


def test_generate_phase14_set_geometry_affordance_ladder_dataset_expected_counts(tmp_path) -> None:
    config = SyntheticMarketConfig(
        dataset_preset="phase14_set_geometry_affordance_ladder",
        permutation_variants=3,
        profile_surface_variants=2,
        relation_scale_variants=2,
        include_settings_variants=True,
    )
    examples = generate_dataset(config)
    assert len(examples) == 4 * 3 * 2 * 2 * 6
    families = {example.family for example in examples}
    assert families == {"set_geometry_control"}
    contexts = {example.context_variant for example in examples}
    assert contexts == {"market_only", "affordance_1", "affordance_2", "affordance_3", "affordance_4", "affordance_5"}

    build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=tmp_path,
            dataset_preset="phase14_set_geometry_affordance_ladder",
            permutation_variants=3,
            profile_surface_variants=2,
            relation_scale_variants=2,
            include_settings_variants=True,
        )
    )
    tick_rows = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet").to_pylist()
    assert min(row["log_id"] for row in tick_rows) >= 2_147_250_000


def test_generate_phase15_market_basis_discovery_dataset_expected_counts(tmp_path) -> None:
    config = SyntheticMarketConfig(
        dataset_preset="phase15_market_basis_discovery",
        discovery_scalar_steps=5,
        discovery_background_variants=2,
        discovery_grid_steps=3,
        include_settings_variants=False,
    )
    examples = generate_dataset(config)
    assert len(examples) == (4 * 2 * 5) + (2 * 2 * 9)
    families = {example.family for example in examples}
    assert families == {"market_basis_scalar", "market_basis_coupled"}
    contexts = {example.context_variant for example in examples}
    assert contexts == {"market_only"}

    build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=tmp_path,
            dataset_preset="phase15_market_basis_discovery",
            discovery_scalar_steps=5,
            discovery_background_variants=2,
            discovery_grid_steps=3,
            include_settings_variants=False,
        )
    )
    tick_rows = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet").to_pylist()
    assert min(row["log_id"] for row in tick_rows) >= 2_147_260_000


def test_generate_phase16_context_order_dataset_expected_counts(tmp_path) -> None:
    config = SyntheticMarketConfig(
        dataset_preset="phase16_context_order",
        discovery_scalar_steps=5,
        discovery_background_variants=2,
        discovery_grid_steps=3,
        include_settings_variants=False,
    )
    examples = generate_dataset(config)
    assert len(examples) == ((4 * 2 * 5) + (2 * 2 * 9)) * 5
    families = {example.family for example in examples}
    assert families == {"market_context_order_scalar", "market_context_order_coupled"}
    contexts = {example.context_variant for example in examples}
    assert contexts == {
        "market_only",
        "risk_5_after_market",
        "risk_5_before_market",
        "affordance_5_after_market",
        "affordance_5_before_market",
    }

    build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=tmp_path,
            dataset_preset="phase16_context_order",
            discovery_scalar_steps=5,
            discovery_background_variants=2,
            discovery_grid_steps=3,
            include_settings_variants=False,
        )
    )
    tick_rows = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet").to_pylist()
    assert min(row["log_id"] for row in tick_rows) >= 2_147_270_000


def test_phase15_market_basis_discovery_uses_dx_like_prompt_surface() -> None:
    examples = generate_dataset(
        SyntheticMarketConfig(
            dataset_preset="phase15_market_basis_discovery",
            discovery_scalar_steps=3,
            discovery_background_variants=1,
            discovery_grid_steps=3,
            include_settings_variants=False,
        )
    )
    prompt = examples[0].user_prompt
    assert "synthetic market scenario" not in prompt.lower()
    assert "archetype:" not in prompt.lower()
    assert "## market snapshot" in prompt.lower()
    assert "## active strategies (current only)" in prompt.lower()
    assert "## active settings" in prompt.lower()
    assert "## portfolio context" in prompt.lower()
    assert "## constraints" in prompt.lower()
    assert "## price impact limits" in prompt.lower()
    assert "- nera" in prompt.lower()


def test_phase16_context_order_variants_reorder_prompt_sections() -> None:
    examples = generate_dataset(
        SyntheticMarketConfig(
            dataset_preset="phase16_context_order",
            discovery_scalar_steps=3,
            discovery_background_variants=1,
            discovery_grid_steps=3,
            include_settings_variants=False,
        )
    )
    by_context = {example.context_variant: example.user_prompt for example in examples[:5]}

    after_prompt = by_context["risk_5_after_market"]
    before_prompt = by_context["risk_5_before_market"]

    assert after_prompt.index("## MARKET SNAPSHOT") < after_prompt.index("## ACTIVE SETTINGS")
    assert before_prompt.index("## ACTIVE SETTINGS") < before_prompt.index("## MARKET SNAPSHOT")
    assert "Asset Risk Preference (Risk): 5 / 5" in after_prompt
    assert "Asset Risk Preference (Risk): 5 / 5" in before_prompt


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


def test_generate_phase7_relation_invariance_dataset_expected_counts(tmp_path) -> None:
    config = SyntheticMarketConfig(
        dataset_preset="phase7_relation_invariance",
        permutation_variants=3,
        profile_surface_variants=2,
        relation_roster_variants=4,
        relation_scale_variants=3,
        include_settings_variants=False,
    )
    examples = generate_dataset(config)
    assert len(examples) == 4 * 3 * 2 * 4 * 3
    families = {example.family for example in examples}
    assert families == {"relation_invariance_control"}
    contexts = {example.context_variant for example in examples}
    assert contexts == {"market_only"}

    build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=tmp_path,
            dataset_preset="phase7_relation_invariance",
            permutation_variants=3,
            profile_surface_variants=2,
            relation_roster_variants=4,
            relation_scale_variants=3,
            include_settings_variants=False,
        )
    )
    tick_rows = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet").to_pylist()
    assert min(row["log_id"] for row in tick_rows) >= 2_147_000_000


def test_phase10_set_geometry_context_uses_stronger_settings_language() -> None:
    examples = generate_dataset(
        SyntheticMarketConfig(
            dataset_preset="phase10_set_geometry_context",
            permutation_variants=1,
            profile_surface_variants=1,
            relation_scale_variants=1,
            include_settings_variants=True,
        )
    )
    prompts = {}
    for example in examples:
        prompts.setdefault(example.context_variant, example.user_prompt)
    assert "edge clearly exceeds fees" in prompts["market_only"].lower()
    assert "asset risk preference (risk): 1 / 5" in prompts["low_risk"].lower()
    assert "lower holder concentration" in prompts["low_risk"].lower()
    assert "asset risk preference (risk): 5 / 5" in prompts["high_risk"].lower()
    assert "fresh momentum and thinner participation are acceptable" in prompts["high_risk"].lower()


def test_phase11_set_geometry_risk_ladder_uses_dx_native_settings_language() -> None:
    examples = generate_dataset(
        SyntheticMarketConfig(
            dataset_preset="phase11_set_geometry_risk_ladder",
            permutation_variants=1,
            profile_surface_variants=1,
            relation_scale_variants=1,
            include_settings_variants=True,
        )
    )
    prompts = {}
    for example in examples:
        prompts.setdefault(example.context_variant, example.user_prompt)
    assert "asset risk preference (risk): 1 / 5" in prompts["risk_1"].lower()
    assert "asset risk preference (risk): 3 / 5" in prompts["risk_3"].lower()
    assert "asset risk preference (risk): 5 / 5" in prompts["risk_5"].lower()


def test_phase13_set_geometry_portfolio_ladder_uses_portfolio_context_language() -> None:
    examples = generate_dataset(
        SyntheticMarketConfig(
            dataset_preset="phase13_set_geometry_portfolio_ladder",
            permutation_variants=1,
            profile_surface_variants=1,
            relation_scale_variants=1,
            include_settings_variants=True,
        )
    )
    prompts = {}
    for example in examples:
        prompts.setdefault(example.context_variant, example.user_prompt)
    assert "## portfolio context" in prompts["market_only"].lower()
    assert "no current token holdings" in prompts["market_only"].lower()
    assert "already represents about 24% of deployed capital" in prompts["portfolio_3"].lower()
    assert "very large position" in prompts["portfolio_5"].lower()


def test_phase14_set_geometry_affordance_ladder_uses_constraint_language() -> None:
    examples = generate_dataset(
        SyntheticMarketConfig(
            dataset_preset="phase14_set_geometry_affordance_ladder",
            permutation_variants=1,
            profile_surface_variants=1,
            relation_scale_variants=1,
            include_settings_variants=True,
        )
    )
    prompts = {}
    for example in examples:
        prompts.setdefault(example.context_variant, example.user_prompt)
    assert "## constraints" in prompts["market_only"].lower()
    assert "no hard execution constraints are supplied" in prompts["market_only"].lower()
    assert "## price impact limits (max 900 bps)" in prompts["affordance_1"].lower()
    assert "buy max 12.00% of eth".lower() in prompts["affordance_1"].lower()
    assert "buy max 0.00% of eth".lower() in prompts["affordance_2"].lower()
    assert "buy max 30.00% of eth".lower() in prompts["affordance_5"].lower()


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
    assert any("Top 20 holder pct" in prompt for prompt in prompts)
    assert any("Short-horizon move" in prompt for prompt in prompts)


def test_phase7_relation_invariance_preserves_anchor_order_across_controls() -> None:
    examples = generate_dataset(
        SyntheticMarketConfig(
            dataset_preset="phase7_relation_invariance",
            permutation_variants=2,
            profile_surface_variants=2,
            relation_roster_variants=4,
            relation_scale_variants=3,
            include_settings_variants=False,
        )
    )
    for example in examples:
        asset_rows = {row["profile_id"]: row for row in example.labels["asset_rows"]}
        assert asset_rows["anchor_left"]["attractiveness_score"] > asset_rows["anchor_right"]["attractiveness_score"]


def test_phase7_relation_invariance_changes_anchor_rank_context_across_rosters() -> None:
    examples = generate_dataset(
        SyntheticMarketConfig(
            dataset_preset="phase7_relation_invariance",
            permutation_variants=1,
            profile_surface_variants=1,
            relation_roster_variants=4,
            relation_scale_variants=1,
            include_settings_variants=False,
        )
    )
    by_scenario: dict[str, set[tuple[int, int]]] = {}
    for example in examples:
        asset_rows = {row["profile_id"]: row for row in example.labels["asset_rows"]}
        ranks = (
            int(asset_rows["anchor_left"]["attractiveness_rank"]),
            int(asset_rows["anchor_right"]["attractiveness_rank"]),
        )
        by_scenario.setdefault(example.family_variant, set()).add(ranks)

    assert all(len(rank_pairs) >= 3 for rank_pairs in by_scenario.values())


def test_generate_phase8_contextual_relation_dataset_expected_counts(tmp_path) -> None:
    config = SyntheticMarketConfig(
        dataset_preset="phase8_contextual_relation",
        permutation_variants=3,
        profile_surface_variants=2,
        relation_roster_variants=4,
        relation_scale_variants=3,
        include_settings_variants=False,
    )
    examples = generate_dataset(config)
    assert len(examples) == 4 * 3 * 2 * 4 * 3
    families = {example.family for example in examples}
    assert families == {"relation_invariance_control"}
    contexts = {example.context_variant for example in examples}
    assert contexts == {"market_only"}

    build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=tmp_path,
            dataset_preset="phase8_contextual_relation",
            permutation_variants=3,
            profile_surface_variants=2,
            relation_roster_variants=4,
            relation_scale_variants=3,
            include_settings_variants=False,
        )
    )
    tick_rows = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet").to_pylist()
    assert min(row["log_id"] for row in tick_rows) >= 2_147_050_000


def test_generate_phase9_set_geometry_dataset_expected_counts(tmp_path) -> None:
    config = SyntheticMarketConfig(
        dataset_preset="phase9_set_geometry",
        permutation_variants=4,
        profile_surface_variants=2,
        relation_scale_variants=3,
        include_settings_variants=False,
    )
    examples = generate_dataset(config)
    assert len(examples) == len(SET_GEOMETRY_SCENARIOS) * 4 * 2 * 3
    families = {example.family for example in examples}
    assert families == {"set_geometry_control"}
    contexts = {example.context_variant for example in examples}
    assert contexts == {"market_only"}

    build_synthetic_market_dataset(
        SyntheticMarketConfig(
            output_dir=tmp_path,
            dataset_preset="phase9_set_geometry",
            permutation_variants=3,
            profile_surface_variants=2,
            relation_scale_variants=2,
            include_settings_variants=False,
        )
    )
    tick_rows = pq.read_table(tmp_path / "synthetic_market_tick_records.parquet").to_pylist()
    assert min(row["log_id"] for row in tick_rows) >= 2_147_100_000


def test_phase8_contextual_relation_keeps_anchor_raw_scores_constant_across_scenarios() -> None:
    examples = generate_dataset(
        SyntheticMarketConfig(
            dataset_preset="phase8_contextual_relation",
            permutation_variants=1,
            profile_surface_variants=1,
            relation_roster_variants=1,
            relation_scale_variants=1,
            include_settings_variants=False,
        )
    )
    anchor_signatures_by_scale: dict[float, set[tuple[float | int, ...]]] = {}
    for example in examples:
        asset_rows = {row["profile_id"]: row for row in example.labels["asset_rows"]}
        scale = round(asset_rows["anchor_left"]["pct_5m"] / 5.08, 3)
        anchor_signatures_by_scale.setdefault(scale, set()).add((
            asset_rows["anchor_left"]["pct_5m"],
            asset_rows["anchor_left"]["pct_1h"],
            asset_rows["anchor_left"]["net_flow_5m"],
            asset_rows["anchor_left"]["unique_traders_5m"],
            asset_rows["anchor_left"]["top20_holder_pct"],
            asset_rows["anchor_right"]["pct_5m"],
            asset_rows["anchor_right"]["pct_1h"],
            asset_rows["anchor_right"]["net_flow_5m"],
            asset_rows["anchor_right"]["unique_traders_5m"],
            asset_rows["anchor_right"]["top20_holder_pct"],
            asset_rows["anchor_left"]["attractiveness_score"],
            asset_rows["anchor_right"]["attractiveness_score"],
        ))
    assert len(anchor_signatures_by_scale) >= 2
    assert all(len(signatures) == 1 for signatures in anchor_signatures_by_scale.values())


def test_phase8_contextual_relation_changes_anchor_rank_contexts() -> None:
    examples = generate_dataset(
        SyntheticMarketConfig(
            dataset_preset="phase8_contextual_relation",
            permutation_variants=1,
            profile_surface_variants=1,
            relation_roster_variants=4,
            relation_scale_variants=1,
            include_settings_variants=False,
        )
    )
    rank_pairs = set()
    for example in examples:
        asset_rows = {row["profile_id"]: row for row in example.labels["asset_rows"]}
        rank_pairs.add((
            int(asset_rows["anchor_left"]["attractiveness_rank"]),
            int(asset_rows["anchor_right"]["attractiveness_rank"]),
        ))
    assert len(rank_pairs) >= 3


def test_phase9_set_geometry_preserves_rank_order_across_scenarios() -> None:
    examples = generate_dataset(
        SyntheticMarketConfig(
            dataset_preset="phase9_set_geometry",
            permutation_variants=1,
            profile_surface_variants=1,
            relation_scale_variants=1,
            include_settings_variants=False,
        )
    )
    rank_orders = set()
    for example in examples:
        ordered = tuple(
            row["profile_id"]
            for row in sorted(example.labels["asset_rows"], key=lambda row: row["attractiveness_rank"])
        )
        rank_orders.add(ordered)
    assert rank_orders == {("geo_alpha", "geo_beta", "geo_gamma", "geo_delta")}


def test_phase9_set_geometry_emits_distinct_distance_signatures_under_same_rank_order() -> None:
    examples = generate_dataset(
        SyntheticMarketConfig(
            dataset_preset="phase9_set_geometry",
            permutation_variants=1,
            profile_surface_variants=1,
            relation_scale_variants=1,
            include_settings_variants=False,
        )
    )
    signatures = {}
    for example in examples:
        rows = {
            row["profile_id"]: row
            for row in example.labels["asset_rows"]
        }
        ordered = ("geo_alpha", "geo_beta", "geo_gamma", "geo_delta")
        coords = np.asarray(
            [
                [
                    rows[profile_id]["pct_5m"],
                    rows[profile_id]["net_flow_5m"],
                    rows[profile_id]["unique_traders_5m"],
                    rows[profile_id]["top20_holder_pct"],
                ]
                for profile_id in ordered
            ],
            dtype=np.float32,
        )
        dists = []
        for left in range(len(ordered)):
            for right in range(left + 1, len(ordered)):
                dists.append(float(np.linalg.norm(coords[left] - coords[right])))
        signatures[example.family_variant] = tuple(round(value, 3) for value in dists)
    assert len(set(signatures.values())) == len(SET_GEOMETRY_SCENARIOS)


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
    assert "- A" in prompt
    assert "- B" in prompt
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
