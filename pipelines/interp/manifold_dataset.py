from __future__ import annotations

import argparse
import json
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from psycopg.rows import dict_row

from pipelines.db import connect_neon

EPS = 1e-9


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _normalize_symbol(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def classify_strategy_clause(text: str) -> str:
    norm = _lower(text)
    if not norm:
        return "unknown"
    if re.search(r"\b(if|when|once)\b.*\b(buy|sell|liquidate|exit|enter)\b", norm):
        return "triggered_action"
    if re.search(r"\b(buy|sell|liquidate|exit|enter|take profit)\b.*\b(now|immediately)\b", norm):
        return "immediate_action"
    if re.search(r"\b(do not sell|don't sell|never sell|hold)\b", norm):
        return "hold_rule"
    if re.search(r"\b(do not buy|don't buy|avoid|only buy|only trade|stay flat|observe only)\b", norm):
        return "restriction"
    if re.search(r"\b(buy|sell|liquidate|exit|enter)\b", norm):
        return "immediate_action"
    return "unknown"


def _build_alias_map(snapshot: dict[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}

    market = snapshot.get("Market") if isinstance(snapshot, dict) else None
    tokens = market.get("Tokens") if isinstance(market, dict) else None
    if isinstance(tokens, list):
        for token in tokens:
            if not isinstance(token, dict):
                continue
            symbol = _normalize_symbol(token.get("Symbol"))
            name = _lower(token.get("Name"))
            if symbol:
                aliases[_lower(symbol)] = symbol
            if name:
                aliases[name] = symbol or name.upper()

    portfolio = snapshot.get("Portfolio") if isinstance(snapshot, dict) else None
    holdings = portfolio.get("Tokens") if isinstance(portfolio, dict) else None
    if isinstance(holdings, list):
        for token in holdings:
            if not isinstance(token, dict):
                continue
            symbol = _normalize_symbol(token.get("Symbol"))
            name = _lower(token.get("Name"))
            if symbol:
                aliases[_lower(symbol)] = symbol
            if name:
                aliases[name] = symbol or name.upper()

    return aliases


def _extract_referenced_symbols(text: str, aliases: dict[str, str]) -> set[str]:
    norm = _lower(text)
    found: set[str] = set()
    for alias, symbol in aliases.items():
        if not alias:
            continue
        if alias in norm:
            found.add(symbol)
    return found


def summarize_strategies(
    strategies: Any,
    *,
    aliases: dict[str, str],
) -> dict[str, Any]:
    summary = {
        "n_strategies": 0,
        "n_high_strategies": 0,
        "n_restrictions": 0,
        "n_hold_rules": 0,
        "n_immediate_actions": 0,
        "n_triggered_actions": 0,
        "blocks_all_buys": False,
        "blocks_all_sells": False,
        "buy_only_symbols": set(),
        "trade_only_symbols": set(),
        "avoid_buy_symbols": set(),
        "hold_symbols": set(),
    }

    if not isinstance(strategies, list):
        return summary

    for item in strategies:
        if not isinstance(item, dict):
            continue
        text = str(item.get("content") or "")
        priority = _lower(item.get("strategyPriority"))
        clause_type = classify_strategy_clause(text)
        symbols = _extract_referenced_symbols(text, aliases)
        norm = _lower(text)

        summary["n_strategies"] += 1
        if priority == "high":
            summary["n_high_strategies"] += 1

        if clause_type == "restriction":
            summary["n_restrictions"] += 1
        elif clause_type == "hold_rule":
            summary["n_hold_rules"] += 1
        elif clause_type == "immediate_action":
            summary["n_immediate_actions"] += 1
        elif clause_type == "triggered_action":
            summary["n_triggered_actions"] += 1

        if re.search(r"\b(do not buy any token|do not buy any tokens|don't buy any token|don't buy any tokens|observe only|stay flat)\b", norm):
            summary["blocks_all_buys"] = True
        if re.search(r"\b(do not sell any token|do not sell any tokens|don't sell any token|don't sell any tokens|never sell any token|never sell any tokens)\b", norm):
            summary["blocks_all_sells"] = True

        if "only buy" in norm:
            summary["buy_only_symbols"].update(symbols)
        if "only trade" in norm:
            summary["trade_only_symbols"].update(symbols)
        if "avoid" in norm:
            summary["avoid_buy_symbols"].update(symbols)
        if re.search(r"\b(do not sell|don't sell|never sell|hold)\b", norm):
            summary["hold_symbols"].update(symbols)

    summary["buy_only_symbols"] = sorted(summary["buy_only_symbols"])
    summary["trade_only_symbols"] = sorted(summary["trade_only_symbols"])
    summary["avoid_buy_symbols"] = sorted(summary["avoid_buy_symbols"])
    summary["hold_symbols"] = sorted(summary["hold_symbols"])
    return summary


def _extract_decision(payload: dict[str, Any]) -> dict[str, Any]:
    action_name = payload.get("tool")
    action_args = payload.get("tool_args")
    if not isinstance(action_args, dict):
        action_args = {}

    completion = payload.get("llm_completion_payload")
    if isinstance(completion, dict):
        choices = completion.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    tool_calls = message.get("tool_calls")
                    if isinstance(tool_calls, list) and tool_calls:
                        first_call = tool_calls[0]
                        if isinstance(first_call, dict):
                            fn = first_call.get("function")
                            if isinstance(fn, dict):
                                if isinstance(fn.get("name"), str):
                                    action_name = fn["name"]
                                args_raw = fn.get("arguments")
                                if isinstance(args_raw, str):
                                    try:
                                        maybe_args = json.loads(args_raw)
                                    except json.JSONDecodeError:
                                        maybe_args = None
                                    if isinstance(maybe_args, dict):
                                        action_args = maybe_args
                    reasoning_content = message.get("reasoning_content")
                else:
                    reasoning_content = None
            else:
                reasoning_content = None
        else:
            reasoning_content = None
    else:
        reasoning_content = None

    decision_type = "other"
    trade_side = None
    executed_valence = "neutral"
    if action_name in {"buy_token", "sell_token"}:
        decision_type = "trade"
        trade_side = "buy" if action_name == "buy_token" else "sell"
        executed_valence = "bullish" if trade_side == "buy" else "bearish"
    elif action_name == "record_observation":
        decision_type = "record_observation"

    target_asset = _normalize_symbol(action_args.get("token"))
    return {
        "action_name": action_name,
        "decision_type": decision_type,
        "trade_side": trade_side,
        "target_asset": target_asset,
        "size": _safe_float(action_args.get("spend_pct")) if action_args.get("spend_pct") is not None else None,
        "observation_text": action_args.get("content"),
        "strategy_ref": action_args.get("strategy"),
        "executed_valence": executed_valence,
        "reasoning_content": reasoning_content,
    }


def _derive_reap_roles(snapshot: dict[str, Any]) -> dict[str, str]:
    roles: dict[str, str] = {}
    market = snapshot.get("Market") if isinstance(snapshot, dict) else None
    reaps = market.get("Reaps") if isinstance(market, dict) else None
    if not isinstance(reaps, dict):
        return roles

    for item in reaps.get("SourceCandidates") or []:
        if isinstance(item, dict):
            symbol = _normalize_symbol(item.get("Symbol"))
            if symbol:
                roles[symbol] = "source"
    for item in reaps.get("TargetCandidates") or []:
        if isinstance(item, dict):
            symbol = _normalize_symbol(item.get("Symbol"))
            if symbol:
                roles[symbol] = "target"
    return roles


def _derive_market_leaders(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    if not rows:
        return {
            "leader_pct_5m": None,
            "leader_net_flow_5m": None,
            "leader_flow_surprise": None,
            "leader_participation_momentum": None,
        }

    def _max_symbol(key: str) -> str | None:
        if not rows:
            return None
        best = max(rows, key=lambda r: _safe_float(r.get(key), float("-inf")))
        return best.get("symbol")

    return {
        "leader_pct_5m": _max_symbol("pct_change_5m"),
        "leader_net_flow_5m": _max_symbol("net_flow_eth_5m"),
        "leader_flow_surprise": _max_symbol("flow_surprise"),
        "leader_participation_momentum": _max_symbol("participation_momentum"),
    }


def build_tick_record(payload: dict[str, Any], *, log_id: int | None = None) -> dict[str, Any]:
    snapshot = payload.get("snapshot") if isinstance(payload, dict) else None
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    agent = snapshot.get("Agent") if isinstance(snapshot.get("Agent"), dict) else {}
    market = snapshot.get("Market") if isinstance(snapshot.get("Market"), dict) else {}
    portfolio = snapshot.get("Portfolio") if isinstance(snapshot.get("Portfolio"), dict) else {}

    aliases = _build_alias_map(snapshot)
    strategies = agent.get("Strategies")
    strategy_summary = summarize_strategies(strategies, aliases=aliases)
    decision = _extract_decision(payload)

    eth_balance = _safe_float(portfolio.get("EthBalance"))
    allowed_tools = snapshot.get("AllowedTools") if isinstance(snapshot.get("AllowedTools"), list) else []
    holdings = portfolio.get("Tokens") if isinstance(portfolio.get("Tokens"), list) else []

    n_held_tokens = 0
    can_sell_any = False
    held_symbols: list[str] = []
    for token in holdings:
        if not isinstance(token, dict):
            continue
        symbol = _normalize_symbol(token.get("Symbol"))
        balance = _safe_float(token.get("Balance"))
        if balance <= 0:
            continue
        n_held_tokens += 1
        if symbol:
            held_symbols.append(symbol)
            if not strategy_summary["blocks_all_sells"]:
                can_sell_any = True

    can_buy_any = (
        "buy_token" in allowed_tools
        and eth_balance > 0
        and not strategy_summary["blocks_all_buys"]
    )

    forced_observe = (
        decision["decision_type"] == "record_observation"
        and not can_buy_any
        and not can_sell_any
    )

    memories = snapshot.get("Memories") if isinstance(snapshot.get("Memories"), list) else []
    recent_actions = [m.get("tool") for m in memories if isinstance(m, dict) and isinstance(m.get("tool"), str)]
    market_tokens = market.get("Tokens") if isinstance(market.get("Tokens"), list) else []
    generated_at = _parse_iso(market.get("GeneratedAt"))
    tick_record = {
        "log_id": log_id,
        "request_id": payload.get("request_id"),
        "execution_id": payload.get("id"),
        "vault_address": payload.get("vault_address") or agent.get("VaultAddress"),
        "nft_id": payload.get("nft_id") or agent.get("CurrentNftId"),
        "created_at": payload.get("created_at"),
        "market_generated_at": market.get("GeneratedAt"),
        "decision_type": decision["decision_type"],
        "action_name": decision["action_name"],
        "trade_side": decision["trade_side"],
        "target_asset": decision["target_asset"],
        "size": decision["size"],
        "executed_valence": decision["executed_valence"],
        "forced_observe": forced_observe,
        "strategy_ref": decision["strategy_ref"],
        "observation_text": decision["observation_text"],
        "reasoning_content": decision["reasoning_content"],
        "trade_size": _safe_int(agent.get("Options", {}).get("trade_size")),
        "trading_activity": _safe_int(agent.get("Options", {}).get("trading_activity")),
        "holding_style": _safe_int(agent.get("Options", {}).get("holding_style")),
        "diversification": _safe_int(agent.get("Options", {}).get("diversification")),
        "asset_risk_preference": _safe_int(agent.get("Options", {}).get("asset_risk_preference")),
        "max_trade_amount": _safe_float(agent.get("Options", {}).get("max_trade_amount")),
        "slippage_bps": _safe_int(agent.get("Options", {}).get("slippage_bps")),
        "max_price_impact_bps": _safe_int(agent.get("Options", {}).get("max_price_impact_bps")),
        "allowed_tools_json": _json_text(allowed_tools),
        "n_allowed_tools": len(allowed_tools),
        "eth_balance": eth_balance,
        "n_market_tokens": len(market_tokens),
        "n_held_tokens": n_held_tokens,
        "held_symbols_json": _json_text(sorted(held_symbols)),
        "has_reap": isinstance(market.get("Reaps"), dict),
        "can_buy_any": can_buy_any,
        "can_sell_any": can_sell_any,
        "blocks_all_buys": strategy_summary["blocks_all_buys"],
        "blocks_all_sells": strategy_summary["blocks_all_sells"],
        "n_strategies": strategy_summary["n_strategies"],
        "n_high_strategies": strategy_summary["n_high_strategies"],
        "n_restrictions": strategy_summary["n_restrictions"],
        "n_hold_rules": strategy_summary["n_hold_rules"],
        "n_immediate_actions": strategy_summary["n_immediate_actions"],
        "n_triggered_actions": strategy_summary["n_triggered_actions"],
        "buy_only_symbols_json": _json_text(strategy_summary["buy_only_symbols"]),
        "trade_only_symbols_json": _json_text(strategy_summary["trade_only_symbols"]),
        "avoid_buy_symbols_json": _json_text(strategy_summary["avoid_buy_symbols"]),
        "hold_symbols_json": _json_text(strategy_summary["hold_symbols"]),
        "recent_memory_count": len(memories),
        "recent_buy_count": sum(1 for tool in recent_actions if tool == "buy_token"),
        "recent_sell_count": sum(1 for tool in recent_actions if tool == "sell_token"),
        "recent_observe_count": sum(1 for tool in recent_actions if tool == "record_observation"),
        "tool_name": payload.get("tool"),
        "status": payload.get("status"),
        "error": payload.get("error"),
    }

    if generated_at is not None:
        tick_record["market_generated_unix"] = generated_at.timestamp()
    return tick_record


def build_asset_records(payload: dict[str, Any], *, log_id: int | None = None) -> list[dict[str, Any]]:
    snapshot = payload.get("snapshot") if isinstance(payload, dict) else None
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    agent = snapshot.get("Agent") if isinstance(snapshot.get("Agent"), dict) else {}
    market = snapshot.get("Market") if isinstance(snapshot.get("Market"), dict) else {}
    portfolio = snapshot.get("Portfolio") if isinstance(snapshot.get("Portfolio"), dict) else {}

    decision = _extract_decision(payload)
    aliases = _build_alias_map(snapshot)
    strategy_summary = summarize_strategies(agent.get("Strategies"), aliases=aliases)
    reaps = _derive_reap_roles(snapshot)
    generated_at = _parse_iso(market.get("GeneratedAt"))

    held_balances: dict[str, float] = {}
    held_times: dict[str, float] = {}
    held_pnls: dict[str, float] = {}
    for token in portfolio.get("Tokens") or []:
        if not isinstance(token, dict):
            continue
        symbol = _normalize_symbol(token.get("Symbol"))
        if not symbol:
            continue
        held_balances[symbol] = _safe_float(token.get("Balance"))
        held_times[symbol] = _safe_float(token.get("TimeHeld"))
        held_pnls[symbol] = _safe_float(token.get("UnrealizedPnlPercent"))

    eth_balance = _safe_float(portfolio.get("EthBalance"))
    allowed_tools = snapshot.get("AllowedTools") if isinstance(snapshot.get("AllowedTools"), list) else []
    can_buy_any = (
        "buy_token" in allowed_tools
        and eth_balance > 0
        and not strategy_summary["blocks_all_buys"]
    )

    rows: list[dict[str, Any]] = []
    for idx, token in enumerate(market.get("Tokens") or []):
        if not isinstance(token, dict):
            continue
        symbol = _normalize_symbol(token.get("Symbol"))
        if not symbol:
            continue

        metrics = token.get("Metrics") if isinstance(token.get("Metrics"), dict) else {}
        created_ts = _safe_int(token.get("CreatedTimestamp"))
        age_hours = None
        if created_ts > 0 and generated_at is not None:
            age_hours = max(0.0, (generated_at.timestamp() - created_ts) / 3600.0)

        vol_5m = _safe_float(metrics.get("VolumeInEth5m"))
        vol_1h = _safe_float(metrics.get("VolumeInEth1h"))
        net_flow_5m = _safe_float(metrics.get("NetFlowInEth5m"))
        unique_traders_5m = _safe_float(metrics.get("UniqueTraders5m"))
        pct_change_5m = _safe_float(metrics.get("PctChange5m"))
        pct_change_1h = _safe_float(metrics.get("PctChange1h"))
        flow_surprise = net_flow_5m / ((vol_1h / 12.0) + EPS)
        participation_momentum = unique_traders_5m * pct_change_5m

        is_held = held_balances.get(symbol, 0.0) > 0
        buy_allowed = can_buy_any
        sell_allowed = (
            "sell_token" in allowed_tools
            and is_held
            and not strategy_summary["blocks_all_sells"]
        )

        if strategy_summary["buy_only_symbols"]:
            buy_allowed = buy_allowed and symbol in strategy_summary["buy_only_symbols"]
            sell_allowed = False
        if strategy_summary["trade_only_symbols"]:
            buy_allowed = buy_allowed and symbol in strategy_summary["trade_only_symbols"]
            sell_allowed = sell_allowed and symbol in strategy_summary["trade_only_symbols"]
        if symbol in strategy_summary["avoid_buy_symbols"]:
            buy_allowed = False
        if symbol in strategy_summary["hold_symbols"]:
            sell_allowed = False

        executed_target = decision["target_asset"] == symbol
        if decision["trade_side"] == "buy" and executed_target:
            executed_valence = "bullish"
        elif decision["trade_side"] == "sell" and executed_target:
            executed_valence = "bearish"
        else:
            executed_valence = "neutral"

        row = {
            "log_id": log_id,
            "request_id": payload.get("request_id"),
            "vault_address": payload.get("vault_address") or agent.get("VaultAddress"),
            "symbol": symbol,
            "name": token.get("Name"),
            "row_index": idx,
            "price_eth": _safe_float(token.get("PriceInEth")),
            "pct_change_1m": _safe_float(metrics.get("PctChange1m")),
            "pct_change_5m": pct_change_5m,
            "pct_change_1h": pct_change_1h,
            "pct_change_6h": _safe_float(metrics.get("PctChange6h")),
            "pct_change_24h": _safe_float(metrics.get("PctChange24h")),
            "pct_change_7d": _safe_float(metrics.get("PctChange7d")),
            "pct_change_all": _safe_float(metrics.get("PctChangeAll")),
            "volume_eth_5m": vol_5m,
            "volume_eth_1h": vol_1h,
            "volume_eth_6h": _safe_float(metrics.get("VolumeInEth6h")),
            "volume_eth_24h": _safe_float(metrics.get("VolumeInEth24h")),
            "volume_eth_7d": _safe_float(metrics.get("VolumeInEth7d")),
            "volume_eth_all": _safe_float(metrics.get("VolumeInEthAll")),
            "net_flow_eth_5m": net_flow_5m,
            "net_flow_eth_1h": _safe_float(metrics.get("NetFlowInEth1h")),
            "holder_count": _safe_int(metrics.get("HolderCount")),
            "holders_change_1h": _safe_int(metrics.get("HoldersChange1h")),
            "unique_traders_5m": _safe_int(metrics.get("UniqueTraders5m")),
            "top20_holder_pct": _safe_float(metrics.get("Top20HolderPct")),
            "age_hours": age_hours,
            "is_new_launch": bool(age_hours is not None and age_hours < 6.0),
            "momentum_divergence_5m_1h": pct_change_5m - pct_change_1h,
            "flow_surprise": flow_surprise,
            "participation_momentum": participation_momentum,
            "participation_efficiency": unique_traders_5m / (vol_5m + EPS),
            "concentration_risk": _safe_float(metrics.get("Top20HolderPct")) / 100.0,
            "reap_role": reaps.get(symbol, "none"),
            "is_held": is_held,
            "held_balance": held_balances.get(symbol, 0.0),
            "held_time_s": held_times.get(symbol, 0.0),
            "held_unrealized_pnl_pct": held_pnls.get(symbol),
            "buy_allowed": buy_allowed,
            "sell_allowed": sell_allowed,
            "is_target_asset": executed_target,
            "is_buy_target": decision["trade_side"] == "buy" and executed_target,
            "is_sell_target": decision["trade_side"] == "sell" and executed_target,
            "asset_executed_valence": executed_valence,
            "decision_type": decision["decision_type"],
            "trade_side": decision["trade_side"],
            "forced_observe": decision["decision_type"] == "record_observation" and not buy_allowed and not sell_allowed,
        }
        rows.append(row)

    leaders = _derive_market_leaders(rows)
    for row in rows:
        row["is_top_pct_5m"] = row["symbol"] == leaders["leader_pct_5m"]
        row["is_top_net_flow_5m"] = row["symbol"] == leaders["leader_net_flow_5m"]
        row["is_top_flow_surprise"] = row["symbol"] == leaders["leader_flow_surprise"]
        row["is_top_participation_momentum"] = row["symbol"] == leaders["leader_participation_momentum"]

    for key in [
        "pct_change_5m",
        "pct_change_1h",
        "net_flow_eth_5m",
        "flow_surprise",
        "participation_momentum",
        "top20_holder_pct",
    ]:
        values = [float(row[key]) for row in rows]
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / max(1, len(values))
        std = math.sqrt(var)
        rank_key = f"{key}_rank_desc"
        z_key = f"{key}_zscore"
        sorted_symbols = [row["symbol"] for row in sorted(rows, key=lambda r: r[key], reverse=True)]
        for row in rows:
            row[z_key] = (row[key] - mean) / std if std > 0 else 0.0
            row[rank_key] = sorted_symbols.index(row["symbol"]) + 1

    return rows


def build_pairwise_records(asset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left in asset_rows:
        for right in asset_rows:
            if left["symbol"] == right["symbol"]:
                continue
            rows.append({
                "log_id": left.get("log_id"),
                "request_id": left.get("request_id"),
                "vault_address": left.get("vault_address"),
                "symbol_a": left["symbol"],
                "symbol_b": right["symbol"],
                "pct_change_5m_delta": left["pct_change_5m"] - right["pct_change_5m"],
                "pct_change_1h_delta": left["pct_change_1h"] - right["pct_change_1h"],
                "net_flow_eth_5m_delta": left["net_flow_eth_5m"] - right["net_flow_eth_5m"],
                "flow_surprise_delta": left["flow_surprise"] - right["flow_surprise"],
                "participation_momentum_delta": left["participation_momentum"] - right["participation_momentum"],
                "concentration_risk_delta": left["concentration_risk"] - right["concentration_risk"],
                "a_beats_b_pct_5m": left["pct_change_5m"] > right["pct_change_5m"],
                "a_beats_b_net_flow_5m": left["net_flow_eth_5m"] > right["net_flow_eth_5m"],
                "a_beats_b_flow_surprise": left["flow_surprise"] > right["flow_surprise"],
                "a_beats_b_participation_momentum": left["participation_momentum"] > right["participation_momentum"],
                "a_is_buy_target": left["is_buy_target"],
                "a_is_sell_target": left["is_sell_target"],
            })
    return rows


def build_manifold_tables(payload: dict[str, Any], *, log_id: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    tick = build_tick_record(payload, log_id=log_id)
    assets = build_asset_records(payload, log_id=log_id)
    pairs = build_pairwise_records(assets)
    return tick, assets, pairs


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def export_from_neon(*, output_dir: Path, limit: int | None = None, where_sql: str | None = None) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    query = """
        SELECT fl.log_id, fl.raw_payload
        FROM full_logs fl
        WHERE fl.raw_payload IS NOT NULL
    """
    params: list[Any] = []
    if where_sql:
        query += f" AND ({where_sql})"
    query += " ORDER BY fl.log_id"
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    tick_rows: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    conn = connect_neon()
    try:
        conn.row_factory = dict_row
        for row in conn.execute(query, params):
            payload = row["raw_payload"]
            if not isinstance(payload, dict):
                continue
            tick, assets, pairs = build_manifold_tables(payload, log_id=row["log_id"])
            tick_rows.append(tick)
            asset_rows.extend(assets)
            pair_rows.extend(pairs)
    finally:
        conn.close()

    _write_parquet(output_dir / "tick_records.parquet", tick_rows)
    _write_parquet(output_dir / "asset_records.parquet", asset_rows)
    _write_parquet(output_dir / "pairwise_records.parquet", pair_rows)
    return {
        "tick_rows": len(tick_rows),
        "asset_rows": len(asset_rows),
        "pair_rows": len(pair_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build market manifold research tables from full_logs.raw_payload")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--where-sql", default=None, help="Optional SQL predicate appended to the full_logs query")
    args = parser.parse_args()

    stats = export_from_neon(output_dir=args.output_dir, limit=args.limit, where_sql=args.where_sql)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
