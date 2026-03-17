from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _json_loads(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def _dig(root: Any, *path: str) -> Any:
    cur = root
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
        if cur is None:
            return None
    return cur


def _normalize_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    out: list[dict[str, Any]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str):
            continue
        if isinstance(content, list):
            content = _json_dumps(content)
        if not isinstance(content, str):
            content = "" if content is None else str(content)
        out.append({"role": role, "content": content})
    return out


def _extract_first_text(messages: list[dict[str, Any]], role: str) -> str | None:
    for message in messages:
        if message.get("role") != role:
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


def _extract_context_blocks(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {
            "messages": [],
            "tools": None,
            "market": None,
            "portfolio": None,
            "strategy": None,
            "config": None,
            "memory": None,
            "model_source": None,
            "assistant_content": None,
            "reasoning_content": None,
            "tool_calls": None,
        }

    request_payload = _dig(payload, "llm_request_payload")
    completion_payload = _dig(payload, "llm_completion_payload")
    snapshot = _dig(payload, "snapshot")

    llm_input = _dig(request_payload, "llm_input")
    messages = _normalize_messages(
        _dig(llm_input, "messages")
        or _dig(request_payload, "messages")
    )

    tools = _dig(llm_input, "tools") or _dig(request_payload, "tools")
    market = (
        _dig(snapshot, "Market")
        or _dig(llm_input, "snapshot", "Market")
    )
    portfolio = (
        _dig(snapshot, "Portfolio")
        or _dig(snapshot, "Vault")
        or _dig(llm_input, "snapshot", "Portfolio")
    )
    strategy = (
        _dig(snapshot, "Strategies")
        or _dig(llm_input, "strategies")
    )
    config = (
        _dig(snapshot, "Config")
        or _dig(snapshot, "VaultConfig")
        or _dig(snapshot, "Agent", "Options")
        or _dig(payload, "options")
    )
    memory = (
        _dig(snapshot, "Memories")
        or _dig(llm_input, "memories")
    )

    first_choice_message = _dig(completion_payload, "choices")
    first_message: dict[str, Any] | None = None
    if isinstance(first_choice_message, list) and first_choice_message:
        first = first_choice_message[0]
        if isinstance(first, dict):
            maybe_message = first.get("message")
            if isinstance(maybe_message, dict):
                first_message = maybe_message

    assistant_content = None if first_message is None else first_message.get("content")
    reasoning_content = None if first_message is None else first_message.get("reasoning_content")
    tool_calls = None if first_message is None else first_message.get("tool_calls")
    model_source = (
        _dig(completion_payload, "model")
        or _dig(request_payload, "model")
        or _dig(payload, "model")
    )

    return {
        "messages": messages,
        "tools": tools,
        "market": market,
        "portfolio": portfolio,
        "strategy": strategy,
        "config": config,
        "memory": memory,
        "model_source": model_source,
        "assistant_content": assistant_content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
    }


def _extract_decision(
    *,
    tool_calls: Any,
    tool_fallback: Any,
    tool_args_json: Any,
) -> dict[str, Any]:
    action_name: str | None = None
    action_args: dict[str, Any] = {}

    if isinstance(tool_calls, list) and tool_calls:
        first_call = tool_calls[0]
        if isinstance(first_call, dict):
            function = first_call.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str):
                    action_name = name
                raw_args = function.get("arguments")
                if isinstance(raw_args, str):
                    parsed_args = _json_loads(raw_args)
                    if isinstance(parsed_args, dict):
                        action_args = parsed_args
                elif isinstance(raw_args, dict):
                    action_args = raw_args

    if action_name is None and isinstance(tool_fallback, str):
        action_name = tool_fallback
        parsed = _json_loads(tool_args_json)
        if isinstance(parsed, dict):
            action_args = parsed

    decision_type = "other"
    trade_side = None
    if action_name in {"buy_token", "sell_token"}:
        decision_type = "trade"
        trade_side = "buy" if action_name == "buy_token" else "sell"
    elif action_name == "record_observation":
        decision_type = "record_observation"

    return {
        "action_name": action_name,
        "action_args": action_args,
        "decision_type": decision_type,
        "trade_side": trade_side,
        "asset": action_args.get("token") if isinstance(action_args, dict) else None,
        "size": action_args.get("spend_pct") if isinstance(action_args, dict) else None,
        "observation_text": action_args.get("content") if isinstance(action_args, dict) else None,
    }


def _missing_blocks(
    *,
    has_messages: bool,
    has_market: bool,
    has_portfolio: bool,
    has_strategy: bool,
    has_config: bool,
    has_memory: bool,
    has_tools: bool,
) -> list[str]:
    missing: list[str] = []
    if not has_messages:
        missing.append("messages")
    if not has_market:
        missing.append("market")
    if not has_portfolio:
        missing.append("portfolio")
    if not has_strategy:
        missing.append("strategy")
    if not has_config:
        missing.append("config")
    if not has_memory:
        missing.append("memory")
    if not has_tools:
        missing.append("tools")
    return missing


# ---------------------------------------------------------------------------
# Bulk prefetch helpers
# ---------------------------------------------------------------------------

def _prefetch_swaps(conn, log_ids: list[int], tx_hashes: list[str]):
    by_log: dict[int, dict[str, Any]] = {}
    if log_ids:
        rows = conn.execute(
            """
            SELECT DISTINCT ON (log_id)
                log_id, transaction_hash, side, token_address, token_symbol, effective_price_usd
            FROM swaps
            WHERE log_id = ANY(%s)
            ORDER BY log_id, timestamp DESC NULLS LAST, log_index DESC
            """,
            [log_ids],
        ).fetchall()
        by_log = {int(r["log_id"]): dict(r) for r in rows}

    by_tx: dict[str, dict[str, Any]] = {}
    if tx_hashes:
        rows = conn.execute(
            """
            SELECT DISTINCT ON (transaction_hash)
                transaction_hash, side, token_address, token_symbol, effective_price_usd
            FROM swaps
            WHERE transaction_hash = ANY(%s)
            ORDER BY transaction_hash, timestamp DESC NULLS LAST, log_index DESC
            """,
            [tx_hashes],
        ).fetchall()
        by_tx = {str(r["transaction_hash"]): dict(r) for r in rows}

    return by_log, by_tx


def _prefetch_outcomes(conn, log_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not log_ids:
        return {}
    rows = conn.execute(
        """
        SELECT DISTINCT ON (log_id)
            log_id, pnl_1h_pct, pnl_4h_pct, pnl_1d_pct,
            was_profitable_1h, entry_price_eth, entry_price_usd
        FROM trade_outcomes
        WHERE log_id = ANY(%s)
        ORDER BY log_id
        """,
        [log_ids],
    ).fetchall()
    return {int(r["log_id"]): dict(r) for r in rows}


def _prefetch_vault_configs(conn, vault_addresses: list[str]) -> dict[str, dict[str, Any]]:
    if not vault_addresses:
        return {}
    rows = conn.execute(
        """
        SELECT
            vault_address, trade_size, trading_activity, holding_style,
            diversification, asset_risk_preference
        FROM vaults
        WHERE vault_address = ANY(%s)
        """,
        [vault_addresses],
    ).fetchall()
    return {str(r["vault_address"]): dict(r) for r in rows}


def _prefetch_strategies(conn, vault_addresses: list[str]) -> dict[str, dict[str, Any]]:
    """Return {vault_address: {strategy_id_from_log: None, matched_strategy: ..., vault_strategies: [...]}}."""
    if not vault_addresses:
        return {}
    rows = conn.execute(
        """
        SELECT
            vault_address, strategy_id, content, enabled,
            strategy_priority, expiry, created_block, updated_block
        FROM strategies
        WHERE vault_address = ANY(%s)
        ORDER BY vault_address, enabled DESC NULLS LAST,
            CASE WHEN strategy_id ~ '^[0-9]+$' THEN strategy_id::INTEGER ELSE NULL END DESC NULLS LAST,
            strategy_id DESC
        """,
        [vault_addresses],
    ).fetchall()

    by_vault: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        va = str(r["vault_address"])
        by_vault.setdefault(va, []).append(dict(r))

    return by_vault


def _build_strategy_fallback(
    strategy_map: dict[str, list[dict[str, Any]]],
    vault_address: str,
    strategy_id: str | None,
) -> Any:
    items = strategy_map.get(vault_address)
    if not items:
        return None
    matched = None
    if strategy_id is not None:
        for item in items:
            if str(item.get("strategy_id")) == str(strategy_id):
                matched = item
                break
    return {
        "strategy_id_from_log": strategy_id,
        "matched_strategy": matched,
        "vault_strategies": items,
    }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PrepareConfig:
    limit: int = 50_000
    only_focus_decisions: bool = True
    transform_version: str = "interp_examples_v0.2"
    full_rebuild: bool = False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_prepare(config: PrepareConfig) -> dict[str, int]:
    from pipelines.db import connect_neon, ensure_schema

    conn = connect_neon(autocommit=False)
    conn.row_factory = dict_row
    ensure_schema(conn)

    # ------------------------------------------------------------------
    # Incremental mode: find high-water mark unless full rebuild
    # ------------------------------------------------------------------
    high_water_mark = 0
    if config.full_rebuild:
        conn.execute("TRUNCATE interp_examples_v0")
        logger.info("Full rebuild: truncated interp_examples_v0")
    else:
        hwm_row = conn.execute(
            "SELECT COALESCE(MAX(log_id), 0) AS hwm FROM interp_examples_v0"
        ).fetchone()
        high_water_mark = int(hwm_row["hwm"])
        logger.info("Incremental mode: high_water_mark=%d", high_water_mark)

    # ------------------------------------------------------------------
    # Source query: read raw_payload directly from JSONB column
    # ------------------------------------------------------------------
    focus_filter = ""
    if config.only_focus_decisions:
        focus_filter = "AND l.tool IN ('buy_token', 'sell_token', 'record_observation')"

    query = f"""
        SELECT
            l.id AS log_id,
            l.vault_address,
            l.created_at,
            l.strategy_id,
            l.transaction_hash,
            l.tool,
            l.tool_args_json,
            f.raw_payload,
            f.parse_error,
            f.completion_text,
            f.reasoning_content,
            f.tool_calls_json,
            f.llm_model,
            f.fetched_at
        FROM inference_logs l
        INNER JOIN full_logs f ON f.log_id = l.id
        WHERE f.raw_payload IS NOT NULL
          AND l.id > %s
          {focus_filter}
        ORDER BY l.id ASC
        LIMIT %s
    """
    rows = conn.execute(query, [high_water_mark, config.limit]).fetchall()

    # ------------------------------------------------------------------
    # Bulk prefetch enrichment data
    # ------------------------------------------------------------------
    log_ids = [int(row["log_id"]) for row in rows]
    vault_addresses = sorted(
        {str(row["vault_address"]) for row in rows if row.get("vault_address")}
    )
    tx_hashes = sorted(
        {str(row["transaction_hash"]) for row in rows if row.get("transaction_hash")}
    )

    swap_by_log, swap_by_tx = _prefetch_swaps(conn, log_ids, tx_hashes)
    outcomes_by_log = _prefetch_outcomes(conn, log_ids)
    vault_configs = _prefetch_vault_configs(conn, vault_addresses)
    strategy_map = _prefetch_strategies(conn, vault_addresses)

    # ------------------------------------------------------------------
    # Build rows for bulk upsert via COPY + temp table
    # ------------------------------------------------------------------
    _COLUMNS = (
        "example_id", "log_id", "vault_address", "created_at", "strategy_id", "transaction_hash", "is_trade",
        "prompt_messages_json", "system_text", "user_text", "tools_available_json",
        "market_snapshot_json", "portfolio_snapshot_json", "strategy_snapshot_json", "config_snapshot_json",
        "memory_snapshot_json", "model_source", "assistant_content", "reasoning_content", "tool_calls_json",
        "action_name", "decision_type", "trade_side", "asset", "size", "observation_text",
        "joined_swap", "swap_side", "swap_token_address", "swap_token_symbol", "swap_price_usd",
        "pnl_1h_pct", "pnl_4h_pct", "pnl_1d_pct", "was_profitable_1h",
        "entry_price_usd", "entry_price_eth",
        "vault_trade_size", "vault_trading_activity", "vault_holding_style", "vault_diversification", "vault_risk_preference",
        "parse_ok", "parse_error", "has_messages", "has_tools", "has_market", "has_portfolio", "has_strategy", "has_config", "has_memory",
        "context_complete", "missing_blocks_json", "label_quality", "label_confidence",
        "ingest_version", "transform_version", "built_at",
    )
    _SET_CLAUSE = ", ".join(f"{col}=EXCLUDED.{col}" for col in _COLUMNS if col not in ("example_id", "log_id", "vault_address"))

    inserted = 0
    focused = 0
    error_count = 0
    insert_rows: list[tuple] = []

    try:
        for row in rows:
            log_id = row["log_id"]
            try:
                # raw_payload is already a dict (psycopg auto-deserializes JSONB)
                payload = row["raw_payload"]
                context = _extract_context_blocks(payload)

                if context["strategy"] is None:
                    context["strategy"] = _build_strategy_fallback(
                        strategy_map,
                        vault_address=str(row["vault_address"]),
                        strategy_id=row["strategy_id"],
                    )

                tool_calls = (
                    context["tool_calls"]
                    if context["tool_calls"] is not None
                    else _json_loads(row["tool_calls_json"])
                )
                decision = _extract_decision(
                    tool_calls=tool_calls,
                    tool_fallback=row["tool"],
                    tool_args_json=row["tool_args_json"],
                )

                if config.only_focus_decisions and decision["decision_type"] not in {
                    "trade",
                    "record_observation",
                }:
                    continue
                focused += 1

                messages = context["messages"]
                has_messages = bool(messages)
                has_tools = context["tools"] is not None
                has_market = context["market"] is not None
                has_portfolio = context["portfolio"] is not None
                has_strategy = context["strategy"] is not None
                has_config = context["config"] is not None
                has_memory = context["memory"] is not None
                parse_ok = not bool(row["parse_error"])
                context_complete = (
                    has_messages and has_market and has_portfolio and has_strategy and has_config
                )
                missing = _missing_blocks(
                    has_messages=has_messages,
                    has_market=has_market,
                    has_portfolio=has_portfolio,
                    has_strategy=has_strategy,
                    has_config=has_config,
                    has_memory=has_memory,
                    has_tools=has_tools,
                )
                if parse_ok and context_complete:
                    quality = "high"
                elif parse_ok and len(missing) <= 2 and all(
                    x in {"memory", "tools"} for x in missing
                ):
                    quality = "medium"
                else:
                    quality = "low"

                # Look up enrichment from prefetched maps
                lid = int(log_id)
                swap = swap_by_log.get(lid) or {}
                if not swap and row.get("transaction_hash"):
                    swap = swap_by_tx.get(str(row["transaction_hash"])) or {}
                joined_swap = bool(swap)
                outcome = outcomes_by_log.get(lid) or {}
                vault_cfg = vault_configs.get(str(row["vault_address"])) or {}

                example_id = f"{row['vault_address']}:{log_id}"
                ingest_version = row["fetched_at"]

                insert_rows.append((
                    example_id,
                    int(log_id),
                    row["vault_address"],
                    row["created_at"],
                    row["strategy_id"],
                    row["transaction_hash"],
                    True if decision["decision_type"] == "trade" else False,
                    _json_dumps(messages),
                    _extract_first_text(messages, "system"),
                    _extract_first_text(messages, "user"),
                    _json_dumps(context["tools"]),
                    _json_dumps(context["market"]),
                    _json_dumps(context["portfolio"]),
                    _json_dumps(context["strategy"]),
                    _json_dumps(context["config"]),
                    _json_dumps(context["memory"]),
                    context["model_source"] or row["llm_model"],
                    context["assistant_content"]
                    if context["assistant_content"] is not None
                    else row["completion_text"],
                    context["reasoning_content"]
                    if context["reasoning_content"] is not None
                    else row["reasoning_content"],
                    _json_dumps(tool_calls),
                    decision["action_name"],
                    decision["decision_type"],
                    decision["trade_side"],
                    decision["asset"],
                    None if decision["size"] is None else str(decision["size"]),
                    decision["observation_text"],
                    joined_swap,
                    swap.get("side"),
                    swap.get("token_address"),
                    swap.get("token_symbol"),
                    swap.get("effective_price_usd"),
                    outcome.get("pnl_1h_pct"),
                    outcome.get("pnl_4h_pct"),
                    outcome.get("pnl_1d_pct"),
                    bool(outcome.get("was_profitable_1h")) if outcome.get("was_profitable_1h") is not None else None,
                    outcome.get("entry_price_usd"),
                    outcome.get("entry_price_eth"),
                    vault_cfg.get("trade_size"),
                    vault_cfg.get("trading_activity"),
                    vault_cfg.get("holding_style"),
                    vault_cfg.get("diversification"),
                    vault_cfg.get("asset_risk_preference"),
                    parse_ok,
                    row["parse_error"],
                    has_messages,
                    has_tools,
                    has_market,
                    has_portfolio,
                    has_strategy,
                    has_config,
                    has_memory,
                    context_complete,
                    _json_dumps(missing),
                    quality,
                    quality,
                    ingest_version,
                    config.transform_version,
                    _now_iso(),
                ))

            except Exception as exc:
                error_count += 1
                logger.error(
                    "Failed to process log_id=%s, skipping row", log_id, exc_info=True
                )
                continue

        # ---------------------------------------------------------------
        # Bulk upsert via COPY into temp table, then INSERT ... ON CONFLICT
        # ---------------------------------------------------------------
        if insert_rows:
            col_list = ", ".join(_COLUMNS)
            conn.execute(
                "CREATE TEMP TABLE _prep_staging (LIKE interp_examples_v0 INCLUDING DEFAULTS) ON COMMIT DROP"
            )
            with conn.cursor().copy(
                f"COPY _prep_staging ({col_list}) FROM STDIN"
            ) as copy:
                for tup in insert_rows:
                    copy.write_row(tup)
            result = conn.execute(f"""
                INSERT INTO interp_examples_v0 ({col_list})
                SELECT {col_list} FROM _prep_staging
                ON CONFLICT (example_id) DO UPDATE SET {_SET_CLAUSE}
            """)
            inserted = result.rowcount if result.rowcount and result.rowcount >= 0 else len(insert_rows)

            # Row count validation
            expected = len(insert_rows)
            if abs(inserted - expected) > 0:
                logger.warning(
                    "Row count mismatch: expected=%d, actual upserted=%d (delta=%d)",
                    expected, inserted, inserted - expected,
                )
            else:
                logger.info("Bulk upsert: %d rows written as expected", inserted)

        conn.commit()
    except Exception:
        conn.rollback()
        logger.error(
            "Fatal error during prepare insert loop, rolling back", exc_info=True
        )
        raise

    # ------------------------------------------------------------------
    # Summary stats (all via filtered queries on interp_examples_v0)
    # ------------------------------------------------------------------
    totals = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN decision_type='trade' THEN 1 ELSE 0 END) AS trade_count,
            SUM(CASE WHEN decision_type='record_observation' THEN 1 ELSE 0 END) AS observation_count,
            SUM(CASE WHEN label_quality='high' THEN 1 ELSE 0 END) AS high_quality_count,
            SUM(CASE WHEN context_complete THEN 1 ELSE 0 END) AS context_complete_count,
            SUM(CASE WHEN parse_ok THEN 1 ELSE 0 END) AS parse_ok_count
        FROM interp_examples_v0
        """
    ).fetchone()

    gap_count = conn.execute(
        "SELECT COUNT(*) AS c FROM interp_examples_v0 WHERE NOT parse_ok OR NOT context_complete"
    ).fetchone()["c"]

    conn.close()

    return {
        "rows_scanned": len(rows),
        "rows_focus_kept": focused,
        "rows_written": inserted,
        "row_errors": error_count,
        "total_examples": int(totals["total"] or 0),
        "trade_count": int(totals["trade_count"] or 0),
        "observation_count": int(totals["observation_count"] or 0),
        "high_quality_count": int(totals["high_quality_count"] or 0),
        "context_complete_count": int(totals["context_complete_count"] or 0),
        "parse_ok_count": int(totals["parse_ok_count"] or 0),
        "gap_count": int(gap_count or 0),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build context-complete interp dataset tables from Xenon full logs (Neon Postgres)"
    )
    parser.add_argument("--limit", type=int, default=50_000)
    parser.add_argument(
        "--include-all-decisions",
        action="store_true",
        help="Keep all decisions, not just trade + record_observation",
    )
    parser.add_argument("--transform-version", default="interp_examples_v0.2")
    parser.add_argument(
        "--full-rebuild",
        action="store_true",
        help="Truncate interp_examples_v0 and rebuild from scratch instead of incremental",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = PrepareConfig(
        limit=max(1, int(args.limit)),
        only_focus_decisions=not bool(args.include_all_decisions),
        transform_version=str(args.transform_version),
        full_rebuild=bool(args.full_rebuild),
    )
    stats = run_prepare(cfg)
    print("Interp dataset prep complete")
    for key, value in stats.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
