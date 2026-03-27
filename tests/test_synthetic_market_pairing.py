from __future__ import annotations

from pipelines.interp import synthetic_market_pairing as pairing


def test_build_matched_metric_examples_denoise_pairs_low_base_to_high_source(monkeypatch) -> None:
    rows = [
        {
            "log_id": 1,
            "example_id": "ex_r00_a",
            "family": "scalar",
            "family_variant": "pct_5m",
            "context_variant": "market_only",
            "roster_key": "r00",
            "prompt_messages_json": "m1",
        },
        {
            "log_id": 2,
            "example_id": "ex_r00_b",
            "family": "scalar",
            "family_variant": "pct_5m",
            "context_variant": "market_only",
            "roster_key": "r00",
            "prompt_messages_json": "m2",
        },
        {
            "log_id": 3,
            "example_id": "ex_r01_a",
            "family": "scalar",
            "family_variant": "pct_5m",
            "context_variant": "market_only",
            "roster_key": "r01",
            "prompt_messages_json": "m3",
        },
        {
            "log_id": 4,
            "example_id": "ex_r01_b",
            "family": "scalar",
            "family_variant": "pct_5m",
            "context_variant": "market_only",
            "roster_key": "r01",
            "prompt_messages_json": "m4",
        },
    ]

    monkeypatch.setattr(
        pairing,
        "load_prompt_visible_metric_map",
        lambda **_: {
            1: {"pct_1h_mad": 1.0},
            2: {"pct_1h_mad": 4.0},
            3: {"pct_1h_mad": 2.0},
            4: {"pct_1h_mad": 5.0},
        },
    )

    paired = pairing.build_matched_metric_examples(
        rows,
        phase_name="phase15_market_basis_discovery_v1",
        pair_metric="pct_1h_mad",
        pair_mode="denoise",
        min_metric_gap=1.0,
        limit=None,
    )

    assert [row["log_id"] for row in paired] == [1, 3]
    assert [row["source_log_id"] for row in paired] == [2, 4]
    assert paired[0]["pair_metric_gap"] == 3.0
    assert paired[1]["pair_metric_gap"] == 3.0


def test_build_matched_metric_examples_noise_reverses_base_and_source(monkeypatch) -> None:
    rows = [
        {
            "log_id": 10,
            "example_id": "ex_r00_a",
            "family": "scalar",
            "family_variant": "pct_5m",
            "context_variant": "market_only",
            "roster_key": "r00",
            "prompt_messages_json": "m10",
        },
        {
            "log_id": 11,
            "example_id": "ex_r00_b",
            "family": "scalar",
            "family_variant": "pct_5m",
            "context_variant": "market_only",
            "roster_key": "r00",
            "prompt_messages_json": "m11",
        },
    ]

    monkeypatch.setattr(
        pairing,
        "load_prompt_visible_metric_map",
        lambda **_: {
            10: {"vol_1h_max": 10.0},
            11: {"vol_1h_max": 20.0},
        },
    )

    paired = pairing.build_matched_metric_examples(
        rows,
        phase_name="phase15_market_basis_discovery_v1",
        pair_metric="vol_1h_max",
        pair_mode="noise",
        min_metric_gap=0.0,
        limit=None,
    )

    assert len(paired) == 1
    assert paired[0]["log_id"] == 11
    assert paired[0]["source_log_id"] == 10
    assert paired[0]["pair_mode"] == "noise"
