from __future__ import annotations

import numpy as np

from projects.DX_TERMINAL.phases.research_rerun.analysis import (
    _blocked_pair_rows,
    _score_prompt_rows,
    _settings_layer_metrics,
    _settings_triplet_rows,
)


class _FirstDimProbe:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = np.clip(np.asarray(X)[:, 0], 0.0, 1.0)
        return np.stack([1.0 - probs, probs], axis=1)


def _acts(
    *,
    buy0: float,
    buy1: float,
    sell0: float,
    sell1: float,
    trade: float,
    side: float,
    settings: float = 0.0,
) -> dict[str, np.ndarray]:
    return {
        "row_mean_0": np.array([[buy0]], dtype=np.float32),
        "row_mean_1": np.array([[buy1]], dtype=np.float32),
        "row_eos_0": np.array([[sell0]], dtype=np.float32),
        "row_eos_1": np.array([[sell1]], dtype=np.float32),
        "last_token": np.array([[trade]], dtype=np.float32),
        "active_settings_eos": np.array([[settings]], dtype=np.float32),
    }


def test_score_prompt_rows_uses_real_row_order_symbols() -> None:
    prompt_rows = [
        {
            "capture_id": "exp:1:blocked_valence:original",
            "base_example_id": "base-1",
            "experiment_group": "blocked_valence",
            "variant": "original",
            "n_rows": 2,
            "row_order": ["ALPHA", "BETA"],
        }
    ]
    activation_cache = {
        "exp:1:blocked_valence:original": _acts(
            buy0=0.2,
            buy1=0.8,
            sell0=0.9,
            sell1=0.1,
            trade=0.7,
            side=0.8,
        ),
    }

    scored = _score_prompt_rows(
        prompt_rows,
        activation_cache=activation_cache,
        buy_probe=_FirstDimProbe(),
        buy_layer=0,
        buy_row_key="row_mean",
        sell_probe=_FirstDimProbe(),
        sell_layer=0,
        sell_row_key="row_eos",
        trade_probe=_FirstDimProbe(),
        trade_layer=0,
        side_probe=_FirstDimProbe(),
        side_layer=0,
    )

    assert len(scored) == 1
    assert scored[0]["top_buy_symbol"] == "BETA"
    assert scored[0]["top_sell_symbol"] == "ALPHA"
    assert scored[0]["predicted_valence"] == "bullish"


def test_blocked_pair_rows_preserve_capture_ids_and_revealed_asset() -> None:
    blocked = _blocked_pair_rows(
        [
            {
                "capture_id": "exp:1:blocked_valence:original",
                "base_example_id": "base-1",
                "experiment_group": "blocked_valence",
                "variant": "original",
                "block_reason": "high_strategy_present",
                "settings_signature": "1/1/1/1/1",
                "actionability_cell": "observe_only",
                "predicted_valence": "neutral",
                "top_buy_symbol": "ALPHA",
                "top_sell_symbol": "BETA",
                "trade_probability": 0.1,
                "bullish": 0.05,
                "bearish": 0.05,
                "neutral": 0.9,
            },
            {
                "capture_id": "exp:1:blocked_valence:clear_strategies",
                "base_example_id": "base-1",
                "experiment_group": "blocked_valence",
                "variant": "clear_strategies",
                "predicted_valence": "bullish",
                "top_buy_symbol": "ALPHA",
                "top_sell_symbol": "BETA",
                "trade_probability": 0.8,
                "bullish": 0.7,
                "bearish": 0.1,
                "neutral": 0.2,
                "top_buy_score": 0.9,
                "top_sell_score": 0.2,
            },
        ]
    )

    assert len(blocked) == 1
    assert blocked[0]["original_capture_id"] == "exp:1:blocked_valence:original"
    assert blocked[0]["clear_capture_id"] == "exp:1:blocked_valence:clear_strategies"
    assert blocked[0]["revealed_asset"] == "ALPHA"


def test_settings_layer_metrics_use_explicit_capture_ids() -> None:
    triplets = _settings_triplet_rows(
        [
            {
                "capture_id": "custom-exp:101:settings_twist:original",
                "base_example_id": "base-101",
                "experiment_group": "settings_twist",
                "variant": "original",
                "cohort_label": "buy",
                "predicted_valence": "bullish",
                "trade_probability": 0.8,
                "bullish": 0.7,
                "bearish": 0.1,
                "neutral": 0.2,
            },
            {
                "capture_id": "custom-exp:101:settings_twist:settings_all1",
                "base_example_id": "base-101",
                "experiment_group": "settings_twist",
                "variant": "settings_all1",
                "cohort_label": "buy",
                "predicted_valence": "neutral",
                "trade_probability": 0.2,
                "bullish": 0.1,
                "bearish": 0.1,
                "neutral": 0.8,
            },
            {
                "capture_id": "custom-exp:101:settings_twist:settings_all5",
                "base_example_id": "base-101",
                "experiment_group": "settings_twist",
                "variant": "settings_all5",
                "cohort_label": "buy",
                "predicted_valence": "bullish",
                "trade_probability": 0.9,
                "bullish": 0.8,
                "bearish": 0.05,
                "neutral": 0.15,
            },
            {
                "capture_id": "custom-exp:102:settings_twist:original",
                "base_example_id": "base-102",
                "experiment_group": "settings_twist",
                "variant": "original",
                "cohort_label": "sell",
                "predicted_valence": "bearish",
                "trade_probability": 0.7,
                "bullish": 0.1,
                "bearish": 0.6,
                "neutral": 0.3,
            },
            {
                "capture_id": "custom-exp:102:settings_twist:settings_all1",
                "base_example_id": "base-102",
                "experiment_group": "settings_twist",
                "variant": "settings_all1",
                "cohort_label": "sell",
                "predicted_valence": "neutral",
                "trade_probability": 0.3,
                "bullish": 0.1,
                "bearish": 0.2,
                "neutral": 0.7,
            },
            {
                "capture_id": "custom-exp:102:settings_twist:settings_all5",
                "base_example_id": "base-102",
                "experiment_group": "settings_twist",
                "variant": "settings_all5",
                "cohort_label": "sell",
                "predicted_valence": "bearish",
                "trade_probability": 0.85,
                "bullish": 0.05,
                "bearish": 0.8,
                "neutral": 0.15,
            },
        ]
    )

    activation_cache = {
        "custom-exp:101:settings_twist:original": _acts(
            buy0=0.2, buy1=0.8, sell0=0.4, sell1=0.6, trade=0.8, side=0.8, settings=0.4
        ),
        "custom-exp:101:settings_twist:settings_all1": _acts(
            buy0=0.2, buy1=0.8, sell0=0.4, sell1=0.6, trade=0.2, side=0.5, settings=0.1
        ),
        "custom-exp:101:settings_twist:settings_all5": _acts(
            buy0=0.2, buy1=0.8, sell0=0.4, sell1=0.6, trade=0.9, side=0.8, settings=0.9
        ),
        "custom-exp:102:settings_twist:original": _acts(
            buy0=0.9, buy1=0.5, sell0=0.9, sell1=0.1, trade=0.7, side=0.2, settings=0.5
        ),
        "custom-exp:102:settings_twist:settings_all1": _acts(
            buy0=0.9, buy1=0.5, sell0=0.9, sell1=0.1, trade=0.3, side=0.2, settings=0.2
        ),
        "custom-exp:102:settings_twist:settings_all5": _acts(
            buy0=0.9, buy1=0.5, sell0=0.9, sell1=0.1, trade=0.85, side=0.2, settings=0.95
        ),
    }

    metrics = _settings_layer_metrics(triplets, activation_cache, layer=0)

    assert metrics["n_triplets"] == 2
    assert "error" not in metrics
    assert metrics["row_mean_cka_original_all1"] == 1.0
    assert metrics["last_token_cka_all1_all5"] >= 0.0
