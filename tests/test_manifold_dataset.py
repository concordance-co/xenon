from __future__ import annotations

from pipelines.datasets.manifold_export import (
    build_asset_records,
    build_manifold_tables,
    build_pairwise_records,
    build_tick_record,
    classify_strategy_clause,
    summarize_strategies,
)


def _sample_payload(*, tool: str = "record_observation", token: str | None = None, spend_pct: float | None = None):
    tool_args = {
        "content": "HIGH status: Restrictions active_compliant -> observing.",
        "strategy": "strategy141",
    }
    if token is not None:
        tool_args["token"] = token
    if spend_pct is not None:
        tool_args["spend_pct"] = spend_pct

    return {
        "id": "vault:req:0",
        "tool": tool,
        "tool_args": tool_args,
        "request_id": "req:0",
        "created_at": "2026-03-18T15:12:13.806476283Z",
        "vault_address": "0xvault",
        "snapshot": {
            "Agent": {
                "CurrentNftId": "7340",
                "VaultAddress": "0xvault",
                "Options": {
                    "trade_size": 4,
                    "trading_activity": 4,
                    "holding_style": 2,
                    "diversification": 1,
                    "asset_risk_preference": 2,
                    "max_trade_amount": 10000,
                    "slippage_bps": 3000,
                    "max_price_impact_bps": 1200,
                },
                "Strategies": [
                    {
                        "content": "Do not sell any tokens.",
                        "strategyPriority": "high",
                    },
                    {
                        "content": "do not buy any token",
                        "strategyPriority": "high",
                    },
                ],
            },
            "Market": {
                "GeneratedAt": "2026-03-18T15:11:51Z",
                "Reaps": {
                    "SourceCandidates": [
                        {"Symbol": "HOTDOGZ"},
                    ],
                    "TargetCandidates": [
                        {"Symbol": "POOPCOIN"},
                    ],
                },
                "Tokens": [
                    {
                        "Name": "Hotdogz",
                        "Symbol": "HOTDOGZ",
                        "CreatedTimestamp": 1772129289,
                        "PriceInEth": 1.2348e-7,
                        "Metrics": {
                            "PctChange1m": 0.0,
                            "PctChange5m": -0.47,
                            "PctChange1h": -8.62,
                            "PctChange6h": -31.81,
                            "PctChange24h": -5.07,
                            "PctChange7d": 59.75,
                            "PctChangeAll": -62.95,
                            "VolumeInEth5m": 0.0748,
                            "VolumeInEth1h": 2.2767,
                            "VolumeInEth6h": 8.43,
                            "VolumeInEth24h": 18.86,
                            "VolumeInEth7d": 42.23,
                            "VolumeInEthAll": 1758.69,
                            "NetFlowInEth5m": -0.0748,
                            "NetFlowInEth1h": -2.13,
                            "HolderCount": 749,
                            "HoldersChange1h": -5,
                            "UniqueTraders5m": 4,
                            "Top20HolderPct": 41.01,
                        },
                    },
                    {
                        "Name": "Poop Coin",
                        "Symbol": "POOPCOIN",
                        "CreatedTimestamp": 1772129297,
                        "PriceInEth": 1.4407e-6,
                        "Metrics": {
                            "PctChange1m": -0.0005,
                            "PctChange5m": -0.019,
                            "PctChange1h": 15.26,
                            "PctChange6h": 40.10,
                            "PctChange24h": 37.98,
                            "PctChange7d": 4.87,
                            "PctChangeAll": 332.22,
                            "VolumeInEth5m": 0.0436,
                            "VolumeInEth1h": 38.69,
                            "VolumeInEth6h": 53.49,
                            "VolumeInEth24h": 58.34,
                            "VolumeInEth7d": 237.42,
                            "VolumeInEthAll": 4299.32,
                            "NetFlowInEth5m": -0.0096,
                            "NetFlowInEth1h": 6.48,
                            "HolderCount": 1334,
                            "HoldersChange1h": -10,
                            "UniqueTraders5m": 10,
                            "Top20HolderPct": 44.02,
                        },
                    },
                ],
            },
            "Memories": [
                {"tool": "record_observation"},
                {"tool": "record_observation"},
                {"tool": "buy_token"},
            ],
            "Portfolio": {
                "EthBalance": 0,
                "Tokens": [
                    {
                        "Name": "Poop Coin",
                        "Symbol": "POOPCOIN",
                        "Balance": 1000,
                        "TimeHeld": 2000,
                        "UnrealizedPnlPercent": 10.0,
                    },
                    {
                        "Name": "Hotdogz",
                        "Symbol": "HOTDOGZ",
                        "Balance": 500,
                        "TimeHeld": 1000,
                        "UnrealizedPnlPercent": -20.0,
                    },
                ],
            },
            "AllowedTools": [
                "buy_token",
                "sell_token",
                "record_observation",
            ],
        },
    }


def test_classify_strategy_clause():
    assert classify_strategy_clause("Do not sell any tokens.") == "hold_rule"
    assert classify_strategy_clause("do not buy any token") == "restriction"
    assert classify_strategy_clause("Buy POOPCOIN now.") == "immediate_action"
    assert classify_strategy_clause("If POOPCOIN drops 20%, buy more.") == "triggered_action"


def test_summarize_strategies():
    payload = _sample_payload()
    strategies = payload["snapshot"]["Agent"]["Strategies"]
    summary = summarize_strategies(
        strategies,
        aliases={"poop coin": "POOPCOIN", "poopcoin": "POOPCOIN", "hotdogz": "HOTDOGZ"},
    )
    assert summary["n_strategies"] == 2
    assert summary["n_high_strategies"] == 2
    assert summary["n_hold_rules"] == 1
    assert summary["n_restrictions"] == 1
    assert summary["blocks_all_buys"] is True
    assert summary["blocks_all_sells"] is True


def test_build_tick_record_forced_observe():
    tick = build_tick_record(_sample_payload(), log_id=123)
    assert tick["log_id"] == 123
    assert tick["decision_type"] == "record_observation"
    assert tick["forced_observe"] is True
    assert tick["can_buy_any"] is False
    assert tick["can_sell_any"] is False
    assert tick["blocks_all_buys"] is True
    assert tick["blocks_all_sells"] is True
    assert tick["n_market_tokens"] == 2
    assert tick["n_held_tokens"] == 2


def test_build_asset_records_assigns_roles_and_feasibility():
    rows = build_asset_records(_sample_payload(), log_id=123)
    assert [r["symbol"] for r in rows] == ["HOTDOGZ", "POOPCOIN"]
    by_symbol = {row["symbol"]: row for row in rows}

    assert by_symbol["HOTDOGZ"]["reap_role"] == "source"
    assert by_symbol["POOPCOIN"]["reap_role"] == "target"
    assert by_symbol["HOTDOGZ"]["buy_allowed"] is False
    assert by_symbol["POOPCOIN"]["sell_allowed"] is False
    assert by_symbol["POOPCOIN"]["is_held"] is True
    assert by_symbol["POOPCOIN"]["forced_observe"] is True
    assert by_symbol["POOPCOIN"]["asset_executed_valence"] == "neutral"
    assert isinstance(by_symbol["HOTDOGZ"]["pct_change_5m_rank_desc"], int)


def test_buy_target_sets_asset_valence():
    payload = _sample_payload(tool="buy_token", token="POOPCOIN", spend_pct=25.0)
    rows = build_asset_records(payload, log_id=77)
    by_symbol = {row["symbol"]: row for row in rows}
    assert by_symbol["POOPCOIN"]["is_buy_target"] is True
    assert by_symbol["POOPCOIN"]["asset_executed_valence"] == "bullish"
    assert by_symbol["HOTDOGZ"]["asset_executed_valence"] == "neutral"


def test_build_pairwise_records():
    rows = build_asset_records(_sample_payload(tool="buy_token", token="POOPCOIN", spend_pct=25.0), log_id=77)
    pairwise = build_pairwise_records(rows)
    assert len(pairwise) == 2
    pair = next(p for p in pairwise if p["symbol_a"] == "POOPCOIN" and p["symbol_b"] == "HOTDOGZ")
    assert pair["a_beats_b_pct_5m"] is True
    assert pair["a_is_buy_target"] is True


def test_build_manifold_tables():
    tick, assets, pairs = build_manifold_tables(_sample_payload(), log_id=9)
    assert tick["log_id"] == 9
    assert len(assets) == 2
    assert len(pairs) == 2
