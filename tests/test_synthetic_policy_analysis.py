from __future__ import annotations

from research.synthetic_market.synthetic_policy_analysis import _resolve_tick_label


def test_resolve_tick_label_decomposes_permission_bits() -> None:
    base = {"labels": {"permission_mode": "buy_and_sell", "observe_vs_act": "act"}}
    assert _resolve_tick_label(base, "can_buy") == "yes"
    assert _resolve_tick_label(base, "can_sell") == "yes"
    assert _resolve_tick_label(base, "observe_vs_act") == "act"

    buy_only = {"labels": {"permission_mode": "buy_only"}}
    assert _resolve_tick_label(buy_only, "can_buy") == "yes"
    assert _resolve_tick_label(buy_only, "can_sell") == "no"

    sell_only = {"labels": {"permission_mode": "sell_only"}}
    assert _resolve_tick_label(sell_only, "can_buy") == "no"
    assert _resolve_tick_label(sell_only, "can_sell") == "yes"

    observe_only = {"labels": {"permission_mode": "observe_only"}}
    assert _resolve_tick_label(observe_only, "can_buy") == "no"
    assert _resolve_tick_label(observe_only, "can_sell") == "no"
