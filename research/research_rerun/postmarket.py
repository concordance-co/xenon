from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from pipelines.db import connect_neon
from pipelines.interp.counterfactual import build_market_rows, build_settings_edited_variant, parse_market_section

from .core import _build_example_record, _load_source_examples, replace_section_body


RISK_LEVELS = (1, 2, 3, 4, 5)
RISK_BASE_CONTEXT = "risk_3"
AFFORDANCE_CONTEXTS = ("market_only", "affordance_1", "affordance_2", "affordance_3", "affordance_4", "affordance_5")
AFFORDANCE_BASE_CONTEXT = "market_only"

RISK_GROUP = "risk_postmarket_geometry"
AFFORDANCE_GROUP = "affordance_postmarket_geometry"
REAL_ROSTER_WIDTH = 6

PORTFOLIO_HEADER = "## PORTFOLIO CONTEXT"
CONSTRAINTS_HEADER = "## CONSTRAINTS"

PORTFOLIO_INTRO = (
    "ETH sitting idle earns nothing — your job is to find opportunities and deploy into tokens. "
    "Once deployed, focus on HOLDING and monitoring for genuine exit signals — not continuously trading in and out. "
    "Positions need time to develop; the best returns come from conviction holds, not from constant rotation. "
    "If ETH is near zero and all positions have a valid thesis, that is a healthy state — do NOT sell just to rebuild an ETH buffer. "
    "If mostly ETH, look for quality entries.\n\n"
    "**Note**: Unrealized PnL shown below is per-token only. You do NOT have vault-level total PnL. "
    "Do not treat one token's unrealized gain as your overall performance."
)

AFFORDANCE_ETH_BY_CONTEXT = {
    "market_only": 0.250000,
    "affordance_1": 0.120000,
    "affordance_2": 0.055000,
    "affordance_3": 0.020000,
    "affordance_4": 0.006000,
    "affordance_5": 0.001500,
}

AFFORDANCE_MAX_TRADE_BY_CONTEXT = {
    "market_only": 100.0,
    "affordance_1": 80.0,
    "affordance_2": 55.0,
    "affordance_3": 30.0,
    "affordance_4": 12.0,
    "affordance_5": 4.0,
}

AFFORDANCE_PRICE_IMPACT_BPS = {
    "market_only": 1500,
    "affordance_1": 1200,
    "affordance_2": 900,
    "affordance_3": 650,
    "affordance_4": 400,
    "affordance_5": 250,
}

AFFORDANCE_BUY_CAPS = {
    "market_only": {"held": 100.0, "unheld": 100.0},
    "affordance_1": {"held": 80.0, "unheld": 70.0},
    "affordance_2": {"held": 55.0, "unheld": 35.0},
    "affordance_3": {"held": 30.0, "unheld": 12.0},
    "affordance_4": {"held": 12.0, "unheld": 0.0},
    "affordance_5": {"held": 4.0, "unheld": 0.0},
}

AFFORDANCE_HELD_BONUS = {
    "market_only": 0.0,
    "affordance_1": 0.15,
    "affordance_2": 0.30,
    "affordance_3": 0.50,
    "affordance_4": 0.75,
    "affordance_5": 1.00,
}

AFFORDANCE_UNHELD_PENALTY = {
    "market_only": 0.0,
    "affordance_1": 0.40,
    "affordance_2": 0.80,
    "affordance_3": 1.30,
    "affordance_4": 1.90,
    "affordance_5": 2.60,
}


@dataclass(slots=True)
class TokenMetrics:
    symbol: str
    name: str
    pct_5m: float
    pct_1h: float
    net_flow_5m: float
    vol_5m: float
    vol_1h: float
    unique_traders_5m: float


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _human_token_balance(balance: Any) -> str:
    numeric = _safe_float(balance)
    if numeric == 0.0:
        return "0"
    magnitude = abs(numeric)
    decimals = 6 if magnitude < 1_000_000 else 3
    return f"{numeric:,.{decimals}f}".replace(",", "")


def _duration_text(seconds: Any) -> str:
    value = max(int(round(_safe_float(seconds))), 0)
    days, rem = divmod(value, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes and len(parts) < 2:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append("0m")
    return " ".join(parts[:2])


def _metrics_from_market_json(market_json: dict[str, Any]) -> list[TokenMetrics]:
    rows: list[TokenMetrics] = []
    for token in list((market_json or {}).get("Tokens") or []):
        metrics = token.get("Metrics") or {}
        rows.append(
            TokenMetrics(
                symbol=str(token.get("Symbol") or ""),
                name=str(token.get("Name") or token.get("Symbol") or ""),
                pct_5m=_safe_float(metrics.get("PctChange5m")),
                pct_1h=_safe_float(metrics.get("PctChange1h")),
                net_flow_5m=_safe_float(metrics.get("NetFlowInEth5m")),
                vol_5m=_safe_float(metrics.get("VolumeInEth5m")),
                vol_1h=_safe_float(metrics.get("VolumeInEth1h")),
                unique_traders_5m=_safe_float(metrics.get("UniqueTraders5m")),
            )
        )
    return rows


def _zscore(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    std = float(arr.std())
    if std <= 1e-6:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / std


def _risk_alpha(level: int) -> float:
    return {
        1: 0.85,
        2: 0.45,
        3: 0.0,
        4: -0.35,
        5: -0.75,
    }[int(level)]


def _compute_symbol_geometry(rows: list[TokenMetrics]) -> dict[str, dict[str, float]]:
    momentum = [0.55 * row.pct_5m + 0.45 * row.pct_1h for row in rows]
    participation = [0.18 * row.vol_5m + 0.06 * row.vol_1h + 0.22 * row.unique_traders_5m for row in rows]
    flow = [row.net_flow_5m for row in rows]
    traders = [row.unique_traders_5m for row in rows]
    vol_5m = [row.vol_5m for row in rows]
    vol_1h = [row.vol_1h for row in rows]
    abs_pct_5m = [abs(row.pct_5m) for row in rows]
    abs_pct_1h = [abs(row.pct_1h) for row in rows]
    abs_flow = [abs(row.net_flow_5m) for row in rows]

    z_momentum = _zscore(momentum)
    z_participation = _zscore(participation)
    z_flow = _zscore(flow)
    z_traders = _zscore(traders)
    z_vol_5m = _zscore(vol_5m)
    z_vol_1h = _zscore(vol_1h)
    z_abs_pct_5m = _zscore(abs_pct_5m)
    z_abs_pct_1h = _zscore(abs_pct_1h)
    z_abs_flow = _zscore(abs_flow)

    result: dict[str, dict[str, float]] = {}
    for idx, row in enumerate(rows):
        strength = (
            1.00 * float(z_momentum[idx])
            + 0.85 * float(z_flow[idx])
            + 0.35 * float(z_participation[idx])
        )
        stability = (
            0.75 * float(z_traders[idx])
            + 0.25 * float(z_vol_1h[idx])
            + 0.20 * float(z_vol_5m[idx])
            - 0.45 * float(z_abs_pct_5m[idx])
            - 0.25 * float(z_abs_pct_1h[idx])
            - 0.10 * float(z_abs_flow[idx])
        )
        result[row.symbol] = {
            "strength": strength,
            "stability": stability,
        }
    return result


def _select_four_asset_slice(rows: list[TokenMetrics], symbol_geometry: dict[str, dict[str, float]]) -> list[str]:
    if len(rows) <= 4:
        return [row.symbol for row in rows]
    points = np.asarray(
        [[symbol_geometry[row.symbol]["strength"], symbol_geometry[row.symbol]["stability"]] for row in rows],
        dtype=np.float32,
    )
    centroid = points.mean(axis=0)
    selected = [int(np.argmax(np.linalg.norm(points - centroid, axis=1)))]
    while len(selected) < 4:
        best_idx = None
        best_score = None
        for idx in range(len(rows)):
            if idx in selected:
                continue
            dists = [float(np.linalg.norm(points[idx] - points[j])) for j in selected]
            score = min(dists)
            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            break
        selected.append(int(best_idx))
    return [rows[idx].symbol for idx in sorted(selected)]


def _pair_by_spread(symbols: list[str], symbol_geometry: dict[str, dict[str, float]]) -> list[str]:
    if len(symbols) <= 2:
        return list(symbols)
    best_pair: tuple[str, str] | None = None
    best_dist = None
    for left_idx in range(len(symbols)):
        for right_idx in range(left_idx + 1, len(symbols)):
            left = symbols[left_idx]
            right = symbols[right_idx]
            dist = math.dist(
                (
                    symbol_geometry[left]["strength"],
                    symbol_geometry[left]["stability"],
                ),
                (
                    symbol_geometry[right]["strength"],
                    symbol_geometry[right]["stability"],
                ),
            )
            if best_dist is None or dist > best_dist:
                best_dist = dist
                best_pair = (left, right)
    if best_pair is None:
        return symbols[:2]
    return [best_pair[0], best_pair[1]]


def _select_affordance_slice(
    rows: list[TokenMetrics],
    symbol_geometry: dict[str, dict[str, float]],
    held_symbols: set[str],
) -> list[str] | None:
    roster_symbols = [row.symbol for row in rows]
    held = [symbol for symbol in roster_symbols if symbol in held_symbols]
    unheld = [symbol for symbol in roster_symbols if symbol not in held_symbols]
    if len(held) < 2 or len(unheld) < 2:
        return None
    selected = _pair_by_spread(held, symbol_geometry) + _pair_by_spread(unheld, symbol_geometry)
    seen: set[str] = set()
    deduped = []
    for symbol in selected:
        if symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    if len(deduped) != 4:
        return None
    return deduped


def _score_coords_for_risk_level(
    *,
    selected_symbols: list[str],
    symbol_geometry: dict[str, dict[str, float]],
    risk_level: int,
) -> dict[str, list[float]]:
    alpha = _risk_alpha(risk_level)
    coords: dict[str, list[float]] = {}
    for symbol in selected_symbols:
        strength = float(symbol_geometry[symbol]["strength"])
        stability = float(symbol_geometry[symbol]["stability"])
        risk_adjusted = strength + alpha * stability
        coords[symbol] = [strength, risk_adjusted]
    return coords


def _score_coords_for_affordance_level(
    *,
    selected_symbols: list[str],
    symbol_geometry: dict[str, dict[str, float]],
    held_symbols: set[str],
    context_variant: str,
) -> dict[str, list[float]]:
    coords: dict[str, list[float]] = {}
    held_bonus = AFFORDANCE_HELD_BONUS[context_variant]
    unheld_penalty = AFFORDANCE_UNHELD_PENALTY[context_variant]
    for symbol in selected_symbols:
        strength = float(symbol_geometry[symbol]["strength"])
        stability = float(symbol_geometry[symbol]["stability"])
        if symbol in held_symbols:
            adjusted = stability + held_bonus
        else:
            adjusted = stability - unheld_penalty
        coords[symbol] = [strength, adjusted]
    return coords


def _portfolio_holdings(portfolio_json: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], float]:
    tokens = list((portfolio_json or {}).get("Tokens") or [])
    holdings = {str(token.get("Symbol") or ""): token for token in tokens if token.get("Symbol")}
    eth_balance = _safe_float((portfolio_json or {}).get("EthBalance"))
    return holdings, eth_balance


def _format_portfolio_line(token: dict[str, Any]) -> str:
    pnl = _safe_float(token.get("UnrealizedPnlPercent"))
    pnl_text = f"{pnl:+.2f}%"
    return (
        f"- {token.get('Symbol')}: Balance: {_human_token_balance(token.get('Balance'))} | "
        f"Avg Entry: {_safe_float(token.get('AvgEntryPriceInEth')):.18f} ETH | "
        f"Unrealized PnL: {pnl_text} | "
        f"Time Since Last Interaction: {_duration_text(token.get('TimeSinceLastSwapOrGenesisOrReap'))} | "
        f"Time Held: {_duration_text(token.get('TimeHeld'))}"
    )


def _build_portfolio_body(
    *,
    context_variant: str,
    selected_symbols: list[str],
    holdings_by_symbol: dict[str, dict[str, Any]],
) -> str:
    lines = [PORTFOLIO_INTRO, "", f"- ETH: Balance: {AFFORDANCE_ETH_BY_CONTEXT[context_variant]:.6f}"]
    selected_holdings = [holdings_by_symbol[symbol] for symbol in selected_symbols if symbol in holdings_by_symbol]
    for token in selected_holdings:
        lines.append(_format_portfolio_line(token))
    if len(selected_holdings) == 0:
        lines.append(f"- No held positions overlap the selected {len(selected_symbols)}-asset roster.")
    return "\n".join(lines)


def _constraint_line(symbol: str, *, held_symbols: set[str], buy_cap: float) -> str:
    if symbol in held_symbols:
        return f"- {symbol}: BUY max {buy_cap:.2f}% of ETH, SELL max 100.00% of {symbol}"
    return f"- {symbol}: BUY max {buy_cap:.2f}% of ETH"


def _build_constraints_body(
    *,
    context_variant: str,
    roster_symbols: list[str],
    held_symbols: set[str],
) -> str:
    max_trade = AFFORDANCE_MAX_TRADE_BY_CONTEXT[context_variant]
    price_impact = AFFORDANCE_PRICE_IMPACT_BPS[context_variant]
    caps = AFFORDANCE_BUY_CAPS[context_variant]
    lines = [
        f"- Max Trade Amount (Percent): {max_trade:.2f}% of available ETH - **[HIGH] strategies may exceed this limit.**",
        "",
        f"## PRICE IMPACT LIMITS (max {price_impact} bps)",
        "",
        "Max sizes that stay within your price impact tolerance. **[HIGH] strategies may exceed these limits** if needed to fulfill explicit directives.",
        "",
    ]
    for symbol in roster_symbols:
        buy_cap = caps["held"] if symbol in held_symbols else caps["unheld"]
        lines.append(_constraint_line(symbol, held_symbols=held_symbols, buy_cap=buy_cap))
        lines.append("")
    lines.append("You can split trades across multiple decisions if needed.")
    return "\n".join(lines).strip()


def build_affordance_edited_variant(
    user_text: str,
    *,
    context_variant: str,
    roster_symbols: list[str],
    selected_symbols: list[str],
    holdings_by_symbol: dict[str, dict[str, Any]],
) -> str:
    held_symbols = set(holdings_by_symbol)
    result = replace_section_body(
        user_text,
        PORTFOLIO_HEADER,
        _build_portfolio_body(
            context_variant=context_variant,
            selected_symbols=selected_symbols,
            holdings_by_symbol=holdings_by_symbol,
        ),
    )
    result = replace_section_body(
        result,
        CONSTRAINTS_HEADER,
        _build_constraints_body(
            context_variant=context_variant,
            roster_symbols=roster_symbols,
            held_symbols=held_symbols,
        ),
    )
    return result


def _load_candidate_metadata() -> list[dict[str, Any]]:
    with connect_neon() as conn:
        rows = conn.execute(
            """
            SELECT
                example_id,
                log_id,
                created_at,
                vault_address,
                vault_risk_preference,
                market_snapshot_json,
                portfolio_snapshot_json,
                config_snapshot_json
            FROM interp_examples_v0
            WHERE decision_type = 'record_observation'
              AND label_quality = 'high'
              AND context_complete
              AND parse_ok
              AND jsonb_array_length(
                    COALESCE(
                      market_snapshot_json::jsonb->'Tokens',
                      market_snapshot_json::jsonb->'tickers',
                      market_snapshot_json::jsonb->'assets',
                      '[]'::jsonb
                    )
                  ) = %s
              AND strpos(user_text, '## MARKET SNAPSHOT') > 0
              AND strpos(user_text, '## ACTIVE SETTINGS') > 0
              AND strpos(user_text, '## PORTFOLIO CONTEXT') > 0
              AND strpos(user_text, '## CONSTRAINTS') > 0
              AND strpos(user_text, 'No active strategies.') > 0
            ORDER BY created_at
            """,
            [REAL_ROSTER_WIDTH],
        ).fetchall()
    return [dict(row) for row in rows]


def _pick_diverse_rows(
    candidates: list[dict[str, Any]],
    *,
    top_rosters: int,
    per_roster: int,
) -> list[dict[str, Any]]:
    by_roster: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_roster[tuple(row["roster_key"])].append(row)

    chosen: list[dict[str, Any]] = []
    for _, roster_rows in sorted(by_roster.items(), key=lambda kv: len(kv[1]), reverse=True)[:top_rosters]:
        roster_rows = sorted(roster_rows, key=lambda row: str(row["created_at"]))
        if len(roster_rows) <= per_roster:
            chosen.extend(roster_rows)
            continue
        idxs = np.linspace(0, len(roster_rows) - 1, per_roster, dtype=int)
        seen: set[int] = set()
        for idx in idxs.tolist():
            if idx in seen:
                continue
            seen.add(idx)
            chosen.append(roster_rows[idx])
    return chosen


def _prepare_candidates() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _load_candidate_metadata()
    risk_candidates: list[dict[str, Any]] = []
    affordance_candidates: list[dict[str, Any]] = []
    for row in rows:
        market_json = row.get("market_snapshot_json")
        portfolio_json = row.get("portfolio_snapshot_json")
        if isinstance(market_json, str):
            market_json = json.loads(market_json)
        if isinstance(portfolio_json, str):
            portfolio_json = json.loads(portfolio_json)
        metrics_rows = _metrics_from_market_json(market_json or {})
        if len(metrics_rows) != REAL_ROSTER_WIDTH:
            continue
        roster_key = tuple(sorted(token.symbol for token in metrics_rows))
        symbol_geometry = _compute_symbol_geometry(metrics_rows)
        risk_selected_symbols = [row.symbol for row in metrics_rows]

        holdings_by_symbol, eth_balance = _portfolio_holdings(portfolio_json or {})
        held_symbols = set(holdings_by_symbol)
        held_in_roster = [symbol for symbol in risk_selected_symbols if symbol in held_symbols]
        unheld_in_roster = [symbol for symbol in risk_selected_symbols if symbol not in held_symbols]
        affordance_selected_symbols = risk_selected_symbols if held_in_roster and unheld_in_roster else None

        shared = {
            "example_id": row["example_id"],
            "log_id": int(row["log_id"]),
            "created_at": row.get("created_at"),
            "vault_address": row.get("vault_address"),
            "vault_risk_preference": row.get("vault_risk_preference"),
            "market_json": market_json,
            "portfolio_json": portfolio_json,
            "roster_key": roster_key,
            "symbol_geometry": symbol_geometry,
            "metrics_rows": metrics_rows,
            "eth_balance": eth_balance,
            "holdings_by_symbol": holdings_by_symbol,
        }
        risk_candidates.append(
            {
                **shared,
                "selected_symbols": risk_selected_symbols,
            }
        )
        if affordance_selected_symbols is not None and eth_balance > 0.0:
            affordance_candidates.append(
                {
                    **shared,
                    "selected_symbols": affordance_selected_symbols,
                }
            )
    return risk_candidates, affordance_candidates


def _selected_row_indices(row_order: list[str], selected_symbols: list[str]) -> list[int]:
    index_by_symbol = {symbol: idx for idx, symbol in enumerate(row_order)}
    return [int(index_by_symbol[symbol]) for symbol in selected_symbols if symbol in index_by_symbol]


def build_postmarket_geometry_payload(
    *,
    experiment_id: str,
    risk_top_rosters: int = 6,
    risk_per_roster: int = 5,
    affordance_top_rosters: int = 6,
    affordance_per_roster: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    risk_candidates, affordance_candidates = _prepare_candidates()
    selected_risk_rows = _pick_diverse_rows(
        risk_candidates,
        top_rosters=risk_top_rosters,
        per_roster=risk_per_roster,
    )
    selected_affordance_rows = _pick_diverse_rows(
        affordance_candidates,
        top_rosters=affordance_top_rosters,
        per_roster=affordance_per_roster,
    )
    selected_rows = selected_risk_rows + selected_affordance_rows
    log_ids = sorted({int(row["log_id"]) for row in selected_rows})
    with connect_neon() as conn:
        source_examples = _load_source_examples(conn, log_ids)
    missing = [log_id for log_id in log_ids if log_id not in source_examples]
    if missing:
        raise RuntimeError(f"Missing source rows in interp_examples_v0 for log_ids={missing[:10]}")

    examples: list[dict[str, Any]] = []
    prompts: list[dict[str, Any]] = []
    example_cache: dict[int, dict[str, Any]] = {}
    manifest_rows: list[dict[str, Any]] = []

    for selected in selected_rows:
        source_row = source_examples[int(selected["log_id"])]
        example_record = _build_example_record(source_row)
        if int(selected["log_id"]) not in example_cache:
            example_cache[int(selected["log_id"])] = example_record
            examples.append(example_record)

    for group_name, selected_group_rows in (
        (RISK_GROUP, selected_risk_rows),
        (AFFORDANCE_GROUP, selected_affordance_rows),
    ):
        for selected in selected_group_rows:
            source_row = source_examples[int(selected["log_id"])]
            example_record = example_cache[int(selected["log_id"])]
            market_json = example_record["market_json"]
            _, row_texts = parse_market_section(source_row["user_text"])
            market_rows = build_market_rows(market_json, row_texts)
            row_order = [market_row.symbol for market_row in market_rows]
            selected_symbols = list(selected["selected_symbols"])
            selected_indices = _selected_row_indices(row_order, selected_symbols)
            if len(selected_indices) != len(selected_symbols):
                continue

            base_coords = {
                symbol: [
                    float(selected["symbol_geometry"][symbol]["strength"]),
                    float(selected["symbol_geometry"][symbol]["stability"]),
                ]
                for symbol in selected_symbols
            }
            holdings_by_symbol = dict(selected["holdings_by_symbol"])
            held_selected = [symbol for symbol in selected_symbols if symbol in holdings_by_symbol]

            if group_name == RISK_GROUP:
                contexts = [f"risk_{level}" for level in RISK_LEVELS]
                for risk_level in RISK_LEVELS:
                    variant = f"risk_{risk_level}"
                    edited_user = build_settings_edited_variant(
                        source_row["user_text"],
                        {"Asset Risk Preference": risk_level},
                    )
                    score_coords = _score_coords_for_risk_level(
                        selected_symbols=selected_symbols,
                        symbol_geometry=selected["symbol_geometry"],
                        risk_level=risk_level,
                    )
                    metadata = {
                        "source_log_id": int(source_row["log_id"]),
                        "ladder_kind": "risk",
                        "base_context": RISK_BASE_CONTEXT,
                        "risk_level": risk_level,
                        "selected_symbols": selected_symbols,
                        "selected_row_indices": selected_indices,
                        "base_coords": base_coords,
                        "score_coords": score_coords,
                        "held_selected_symbols": held_selected,
                        "roster_key": list(selected["roster_key"]),
                        "roster_width": len(selected_symbols),
                        "original_vault_risk_preference": selected.get("vault_risk_preference"),
                    }
                    prompts.append(
                        {
                            "prompt_id": f"{experiment_id}:{source_row['log_id']}:{group_name}:{variant}",
                            "base_example_id": source_row["example_id"],
                            "experiment_id": experiment_id,
                            "experiment_group": group_name,
                            "cohort_label": "real_postmarket_risk",
                            "variant": variant,
                            "system_text": source_row["system_text"],
                            "user_text": edited_user,
                            "row_order": row_order,
                            "n_rows": len(row_order),
                            "target_asset": None,
                            "block_reason": None,
                            "settings_signature": variant,
                            "actionability_cell": None,
                            "metadata": metadata,
                        }
                    )
                manifest_rows.append(
                    {
                        "experiment_group": group_name,
                        "base_example_id": source_row["example_id"],
                        "log_id": int(source_row["log_id"]),
                        "created_at": source_row.get("created_at"),
                        "vault_address": source_row.get("vault_address"),
                        "roster_key": list(selected["roster_key"]),
                        "selected_symbols": selected_symbols,
                        "selected_row_indices": selected_indices,
                    }
                )
            else:
                roster_symbols = row_order
                for context_variant in AFFORDANCE_CONTEXTS:
                    edited_user = build_affordance_edited_variant(
                        source_row["user_text"],
                        context_variant=context_variant,
                        roster_symbols=roster_symbols,
                        selected_symbols=selected_symbols,
                        holdings_by_symbol=holdings_by_symbol,
                    )
                    score_coords = _score_coords_for_affordance_level(
                        selected_symbols=selected_symbols,
                        symbol_geometry=selected["symbol_geometry"],
                        held_symbols=set(holdings_by_symbol),
                        context_variant=context_variant,
                    )
                    metadata = {
                        "source_log_id": int(source_row["log_id"]),
                        "ladder_kind": "affordance",
                        "base_context": AFFORDANCE_BASE_CONTEXT,
                        "affordance_level": 0 if context_variant == "market_only" else int(context_variant.split("_")[1]),
                        "selected_symbols": selected_symbols,
                        "selected_row_indices": selected_indices,
                        "base_coords": base_coords,
                        "score_coords": score_coords,
                        "held_selected_symbols": held_selected,
                        "roster_key": list(selected["roster_key"]),
                        "roster_width": len(selected_symbols),
                        "eth_balance_display": AFFORDANCE_ETH_BY_CONTEXT[context_variant],
                    }
                    prompts.append(
                        {
                            "prompt_id": f"{experiment_id}:{source_row['log_id']}:{group_name}:{context_variant}",
                            "base_example_id": source_row["example_id"],
                            "experiment_id": experiment_id,
                            "experiment_group": group_name,
                            "cohort_label": "real_postmarket_affordance",
                            "variant": context_variant,
                            "system_text": source_row["system_text"],
                            "user_text": edited_user,
                            "row_order": row_order,
                            "n_rows": len(row_order),
                            "target_asset": None,
                            "block_reason": None,
                            "settings_signature": context_variant,
                            "actionability_cell": None,
                            "metadata": metadata,
                        }
                    )
                manifest_rows.append(
                    {
                        "experiment_group": group_name,
                        "base_example_id": source_row["example_id"],
                        "log_id": int(source_row["log_id"]),
                        "created_at": source_row.get("created_at"),
                        "vault_address": source_row.get("vault_address"),
                        "roster_key": list(selected["roster_key"]),
                        "selected_symbols": selected_symbols,
                        "selected_row_indices": selected_indices,
                        "held_selected_symbols": held_selected,
                    }
                )

    payload = {
        "experiment_id": experiment_id,
        "examples": examples,
        "prompts": prompts,
    }
    summary = {
        "experiment_id": experiment_id,
        "base_examples": len(examples),
        "prompts": len(prompts),
        "risk_base_examples": len(selected_risk_rows),
        "affordance_base_examples": len(selected_affordance_rows),
        "risk_contexts": [f"risk_{level}" for level in RISK_LEVELS],
        "affordance_contexts": list(AFFORDANCE_CONTEXTS),
        "prompt_counts_by_group": {
            group: sum(1 for prompt in prompts if prompt["experiment_group"] == group)
            for group in sorted({prompt["experiment_group"] for prompt in prompts})
        },
        "prompt_counts_by_variant": {
            variant: sum(1 for prompt in prompts if prompt["variant"] == variant)
            for variant in sorted({prompt["variant"] for prompt in prompts})
        },
        "manifest_rows": manifest_rows,
    }
    return payload, summary
