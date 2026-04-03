from __future__ import annotations

from pathlib import Path

from projects.synthetic_market.synthetic_market_behavior_battery import (
    build_behavior_baseline_plan,
    build_behavior_robustness_battery,
    build_behavior_robustness_payload,
)
from projects.synthetic_market.synthetic_market_behavior_runner import SyntheticMarketBehaviorConfig


def test_build_behavior_robustness_battery_expands_targeted_and_random_control_runs(tmp_path: Path):
    base_config = SyntheticMarketBehaviorConfig(
        output_dir=tmp_path / "behavior",
        patch_mode="project_out",
        pair_mode="denoise",
        target_layers=(4, 35),
        component_indices_by_layer={4: (0, 1, 2, 3), 35: (0, 1, 2, 3)},
        components_per_layer=4,
    )

    plan = build_behavior_robustness_battery(
        base_config,
        run_name_prefix="phase20_market_robustness",
        lambda_sweep=(0.25, 1.0),
        neighboring_component_offsets=(0, 1),
        subspace_sizes=(2,),
        random_control_seeds=(13,),
        pair_modes=("denoise", "noise"),
    )

    assert len(plan) == 24

    targeted = [item for item in plan if item.sweep_kind == "targeted"]
    random_controls = [item for item in plan if item.sweep_kind == "random_control"]
    assert len(targeted) == 12
    assert len(random_controls) == 12

    noise_runs = [item for item in plan if item.config.pair_mode == "noise"]
    assert noise_runs

    shifted = next(item for item in plan if "shift_p1" in item.run_name and item.sweep_kind == "targeted")
    assert shifted.config.component_indices_by_layer[4] == (1, 2, 3, 4)
    assert shifted.config.component_indices_by_layer[35] == (1, 2, 3, 4)

    resized = next(item for item in plan if "_k2_" in item.run_name and item.sweep_kind == "targeted")
    assert resized.config.component_indices_by_layer[4] == (0, 1)
    assert resized.config.component_indices_by_layer[35] == (0, 1)
    assert resized.config.components_per_layer == 2

    control = next(item for item in plan if item.sweep_kind == "random_control")
    assert control.config.patch_mode == "random_control"
    assert control.config.random_seed == 13


def test_build_behavior_robustness_battery_skips_invalid_negative_neighbor_offsets(tmp_path: Path):
    base_config = SyntheticMarketBehaviorConfig(
        output_dir=tmp_path / "behavior",
        patch_mode="project_out",
        target_layers=(4,),
        component_indices_by_layer={4: (0, 1)},
        components_per_layer=2,
    )

    plan = build_behavior_robustness_battery(
        base_config,
        run_name_prefix="phase20_market_robustness",
        lambda_sweep=(1.0,),
        neighboring_component_offsets=(-1, 0, 1),
        random_control_seeds=(),
    )

    run_names = {item.run_name for item in plan}
    assert all("shift_m1" not in run_name for run_name in run_names)
    assert any("shift_p1" in run_name for run_name in run_names)


def test_build_behavior_baseline_plan_adds_one_run_per_pair_mode(tmp_path: Path):
    base_config = SyntheticMarketBehaviorConfig(
        output_dir=tmp_path / "behavior",
        patch_mode="project_out",
        pair_mode="denoise",
        target_layers=(4,),
    )

    plan = build_behavior_baseline_plan(
        base_config,
        run_name_prefix="phase20_market_robustness",
        pair_modes=("denoise", "noise"),
    )

    assert [item.run_name for item in plan] == [
        "phase20_market_robustness_baseline_denoise",
        "phase20_market_robustness_baseline_noise",
    ]
    assert all(item.config.patch_mode == "none" for item in plan)
    assert all(item.sweep_kind == "baseline" for item in plan)


def test_build_behavior_robustness_payload_counts_runs_by_kind(tmp_path: Path):
    base_config = SyntheticMarketBehaviorConfig(
        output_dir=tmp_path / "behavior",
        patch_mode="project_out",
        pair_mode="denoise",
        target_layers=(4,),
        component_indices_by_layer={4: (0, 1)},
        components_per_layer=2,
    )

    payload = build_behavior_robustness_payload(
        base_config,
        run_name_prefix="phase20_market_robustness",
        lambda_sweep=(1.0,),
        neighboring_component_offsets=(0,),
        random_control_seeds=(13,),
        include_baselines=True,
    )

    assert payload["count"] == 3
    assert payload["counts_by_sweep_kind"] == {
        "baseline": 1,
        "random_control": 1,
        "targeted": 1,
    }
