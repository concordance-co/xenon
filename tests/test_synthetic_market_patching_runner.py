from __future__ import annotations

import numpy as np

from research.synthetic_market.synthetic_market_patching_runner import (
    SyntheticMarketPatchingConfig,
    _apply_selection_strategy,
    _build_patch_spec,
    _extract_roster_key,
    _parse_component_indices_spec,
)
from pipelines.interp.patching.market_patch import (
    PATCH_MODE_PROJECT_OUT,
    PATCH_MODE_RANDOM_CONTROL,
    PATCH_MODE_SWAP_MEAN,
)


def test_extract_roster_key_from_example_id() -> None:
    assert _extract_roster_key("basis_coupled_pct_5m__net_flow_5m_r03_x01_y02") == "r03"
    assert _extract_roster_key("basis_scalar_pct_1h_r11_s04") == "r11"
    assert _extract_roster_key("no_roster_here") == "unknown"


def test_apply_selection_strategy_round_robins_buckets() -> None:
    rows = [
        {"example_id": "a_r00_x00", "family": "scalar", "family_variant": "pct_5m"},
        {"example_id": "b_r00_x01", "family": "scalar", "family_variant": "pct_5m"},
        {"example_id": "c_r01_x00", "family": "scalar", "family_variant": "pct_5m"},
        {"example_id": "d_r01_x01", "family": "scalar", "family_variant": "pct_5m"},
        {"example_id": "e_r00_x00", "family": "coupled", "family_variant": "pct_5m__net_flow_5m"},
        {"example_id": "f_r00_x01", "family": "coupled", "family_variant": "pct_5m__net_flow_5m"},
    ]

    selected = _apply_selection_strategy(
        rows,
        strategy="stratified_family_variant_roster",
        limit=5,
    )

    assert [row["example_id"] for row in selected] == [
        "a_r00_x00",
        "c_r01_x00",
        "e_r00_x00",
        "b_r00_x01",
        "d_r01_x01",
    ]


def test_build_patch_spec_project_out_uses_named_component_when_requested() -> None:
    config = SyntheticMarketPatchingConfig(
        patch_mode=PATCH_MODE_PROJECT_OUT,
        target_layers=(4,),
        components_per_layer=4,
        direction_name="leader_axis",
    )
    basis_payload = {
        4: {
            "components": np.zeros((4, 8), dtype=np.float32),
            "named_components": {"leader_axis": 0, "other_axis": 2},
        }
    }

    spec = _build_patch_spec(
        config=config,
        market_span=(10, 20),
        basis_payload=basis_payload,
    )

    assert spec.component_indices_by_layer == {4: (0,)}


def test_build_patch_spec_random_control_uses_named_component_when_requested() -> None:
    config = SyntheticMarketPatchingConfig(
        patch_mode=PATCH_MODE_RANDOM_CONTROL,
        target_layers=(35,),
        components_per_layer=4,
        direction_name="dispersion_axis",
    )
    basis_payload = {
        35: {
            "components": np.zeros((4, 8), dtype=np.float32),
            "named_components": {"dispersion_axis": 0, "other_axis": 1},
        }
    }

    spec = _build_patch_spec(
        config=config,
        market_span=(12, 24),
        basis_payload=basis_payload,
    )

    assert spec.component_indices_by_layer == {35: (0,)}


def test_parse_component_indices_spec_supports_global_and_per_layer_forms() -> None:
    assert _parse_component_indices_spec("0,1,3", target_layers=(4, 35)) == {
        4: (0, 1, 3),
        35: (0, 1, 3),
    }
    assert _parse_component_indices_spec("4=0,2;35=1,3", target_layers=(4, 35)) == {
        4: (0, 2),
        35: (1, 3),
    }


def test_build_patch_spec_prefers_explicit_component_indices() -> None:
    config = SyntheticMarketPatchingConfig(
        patch_mode=PATCH_MODE_PROJECT_OUT,
        target_layers=(4,),
        components_per_layer=4,
        component_indices_by_layer={4: (1, 3)},
        direction_name="leader_axis",
    )
    basis_payload = {
        4: {
            "components": np.zeros((4, 8), dtype=np.float32),
            "named_components": {"leader_axis": 0, "other_axis": 2},
        }
    }

    spec = _build_patch_spec(
        config=config,
        market_span=(10, 20),
        basis_payload=basis_payload,
    )

    assert spec.component_indices_by_layer == {4: (1, 3)}


def test_build_patch_spec_swap_mean_uses_donor_means() -> None:
    config = SyntheticMarketPatchingConfig(
        patch_mode=PATCH_MODE_SWAP_MEAN,
        target_layers=(4,),
    )
    basis_payload = {
        4: {
            "components": np.zeros((4, 8), dtype=np.float32),
            "named_components": {},
        }
    }

    spec = _build_patch_spec(
        config=config,
        market_span=(10, 20),
        basis_payload=basis_payload,
        donor_mean_by_layer={4: np.ones((8,), dtype=np.float32)},
    )

    assert 4 in spec.donor_mean_by_layer
