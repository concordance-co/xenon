from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        [table_name],
    ).fetchone()
    return row is not None


def _json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _json_loads(value: Any) -> Any:
    if not isinstance(value, str) or not value:
        return None
    try:
        return json.loads(value)
    except Exception:
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


_payload_path_prefix_remap: str | None = None


def _init_payload_remap(sample_path: str | None) -> None:
    """Detect if payload paths need remapping (local → Modal) and cache the prefix."""
    global _payload_path_prefix_remap
    if not sample_path:
        return
    path = Path(sample_path)
    if path.exists():
        _payload_path_prefix_remap = ""  # no remap needed
        return
    # Try Modal path
    parts = sample_path.split("full_logs/")
    if len(parts) == 2:
        modal_base = "/data/ingest/full_logs/"
        modal_path = Path(modal_base + parts[1])
        if modal_path.exists():
            # Store the prefix to replace: everything before "full_logs/"
            _payload_path_prefix_remap = parts[0] + "full_logs/"
            print(f"  Payload path remap: '{_payload_path_prefix_remap}' → '/data/ingest/full_logs/'")
            return
    _payload_path_prefix_remap = None  # can't find files


def _load_payload(payload_path: str | None) -> dict[str, Any] | None:
    if not payload_path:
        return None

    if _payload_path_prefix_remap is None:
        return None  # remap detection failed — files aren't available
    elif _payload_path_prefix_remap:
        payload_path = payload_path.replace(
            _payload_path_prefix_remap, "/data/ingest/full_logs/", 1
        )

    path = Path(payload_path)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except (FileNotFoundError, OSError):
        return None
    except Exception:
        return None
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


def _extract_swap_row(conn: sqlite3.Connection, log_id: int, transaction_hash: str | None) -> dict[str, Any]:
    if not _table_exists(conn, "swaps"):
        return {}

    row = conn.execute(
        """
        SELECT
            transaction_hash,
            side,
            token_address,
            token_symbol,
            effective_price_usd
        FROM swaps
        WHERE log_id = ?
        LIMIT 1
        """,
        [log_id],
    ).fetchone()
    if row is None and transaction_hash:
        row = conn.execute(
            """
            SELECT
                transaction_hash,
                side,
                token_address,
                token_symbol,
                effective_price_usd
            FROM swaps
            WHERE transaction_hash = ?
            LIMIT 1
            """,
            [transaction_hash],
        ).fetchone()
    if row is None:
        return {}
    return dict(row)


def _extract_trade_outcome(conn: sqlite3.Connection, log_id: int) -> dict[str, Any]:
    if not _table_exists(conn, "trade_outcomes"):
        return {}
    row = conn.execute(
        """
        SELECT
            pnl_1h_pct,
            pnl_4h_pct,
            pnl_1d_pct,
            was_profitable_1h,
            entry_price_eth,
            entry_price_usd
        FROM trade_outcomes
        WHERE log_id = ?
        LIMIT 1
        """,
        [log_id],
    ).fetchone()
    if row is None:
        return {}
    return dict(row)


def _extract_vault_config(conn: sqlite3.Connection, vault_address: str) -> dict[str, Any]:
    if not _table_exists(conn, "vaults"):
        return {}
    row = conn.execute(
        """
        SELECT
            trade_size,
            trading_activity,
            holding_style,
            diversification,
            asset_risk_preference
        FROM vaults
        WHERE vault_address = ?
        LIMIT 1
        """,
        [vault_address],
    ).fetchone()
    if row is None:
        return {}
    return dict(row)


def _extract_strategy_fallback(
    conn: sqlite3.Connection,
    *,
    vault_address: str,
    strategy_id: str | None,
) -> Any:
    if not _table_exists(conn, "strategies"):
        return None

    rows = conn.execute(
        """
        SELECT
            strategy_id,
            content,
            enabled,
            strategy_priority,
            expiry,
            created_block,
            updated_block
        FROM strategies
        WHERE vault_address = ?
        ORDER BY enabled DESC, CAST(strategy_id AS INTEGER) DESC
        """,
        [vault_address],
    ).fetchall()
    if not rows:
        return None
    items = [dict(r) for r in rows]
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


@dataclass(slots=True)
class PrepareConfig:
    db_path: Path = Path("data/terminal_ingest.db")
    limit: int = 50_000
    only_focus_decisions: bool = True
    trade_sample_size: int = 150
    observation_sample_size: int = 150
    paired_sample_size: int = 100
    transform_version: str = "interp_examples_v0.2"
    export_jsonl: bool = False
    export_parquet: bool = False
    export_dir: Path = Path("data/interp_exports")


def _prefetch_swaps(conn: sqlite3.Connection) -> dict[int, dict]:
    """Load all swaps into a dict keyed by log_id."""
    if not _table_exists(conn, "swaps"):
        return {}
    rows = conn.execute(
        """SELECT log_id, transaction_hash, side, token_address,
                  token_symbol, effective_price_usd
           FROM swaps"""
    ).fetchall()
    by_log: dict[int, dict] = {}
    for r in rows:
        lid = int(r["log_id"]) if r["log_id"] is not None else None
        if lid is not None and lid not in by_log:
            by_log[lid] = dict(r)
    return by_log


def _prefetch_outcomes(conn: sqlite3.Connection) -> dict[int, dict]:
    """Load all trade outcomes into a dict keyed by log_id."""
    if not _table_exists(conn, "trade_outcomes"):
        return {}
    rows = conn.execute(
        """SELECT log_id, pnl_1h_pct, pnl_4h_pct, pnl_1d_pct,
                  was_profitable_1h, entry_price_eth, entry_price_usd
           FROM trade_outcomes"""
    ).fetchall()
    return {int(r["log_id"]): dict(r) for r in rows}


def _prefetch_vault_configs(conn: sqlite3.Connection) -> dict[str, dict]:
    """Load all vault configs into a dict keyed by vault_address."""
    if not _table_exists(conn, "vaults"):
        return {}
    rows = conn.execute(
        """SELECT vault_address, trade_size, trading_activity,
                  holding_style, diversification, asset_risk_preference
           FROM vaults"""
    ).fetchall()
    return {r["vault_address"]: dict(r) for r in rows}


def _prefetch_strategies(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    """Load all strategies grouped by vault_address."""
    if not _table_exists(conn, "strategies"):
        return {}
    rows = conn.execute(
        """SELECT vault_address, strategy_id, content, enabled,
                  strategy_priority, expiry, created_block, updated_block
           FROM strategies
           ORDER BY enabled DESC, CAST(strategy_id AS INTEGER) DESC"""
    ).fetchall()
    by_vault: dict[str, list[dict]] = {}
    for r in rows:
        by_vault.setdefault(r["vault_address"], []).append(dict(r))
    return by_vault


def run_prepare(config: PrepareConfig) -> dict[str, int]:
    if not config.db_path.exists():
        raise FileNotFoundError(f"Database not found: {config.db_path}")

    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA cache_size=-64000;")  # 64MB cache
    conn.execute("PRAGMA mmap_size=268435456;")  # 256MB mmap

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS interp_examples_v0 (
            example_id TEXT PRIMARY KEY,
            log_id INTEGER NOT NULL,
            vault_address TEXT NOT NULL,
            created_at TEXT,
            strategy_id TEXT,
            transaction_hash TEXT,
            is_trade INTEGER NOT NULL,
            prompt_messages_json TEXT,
            system_text TEXT,
            user_text TEXT,
            tools_available_json TEXT,
            market_snapshot_json TEXT,
            portfolio_snapshot_json TEXT,
            strategy_snapshot_json TEXT,
            config_snapshot_json TEXT,
            memory_snapshot_json TEXT,
            model_source TEXT,
            assistant_content TEXT,
            reasoning_content TEXT,
            tool_calls_json TEXT,
            action_name TEXT,
            decision_type TEXT NOT NULL,
            trade_side TEXT,
            asset TEXT,
            size TEXT,
            observation_text TEXT,
            joined_swap INTEGER NOT NULL,
            swap_side TEXT,
            swap_token_address TEXT,
            swap_token_symbol TEXT,
            swap_price_usd REAL,
            pnl_1h_pct REAL,
            pnl_4h_pct REAL,
            pnl_1d_pct REAL,
            was_profitable_1h INTEGER,
            entry_price_usd REAL,
            entry_price_eth REAL,
            vault_trade_size INTEGER,
            vault_trading_activity INTEGER,
            vault_holding_style INTEGER,
            vault_diversification INTEGER,
            vault_risk_preference INTEGER,
            parse_ok INTEGER NOT NULL,
            parse_error TEXT,
            has_messages INTEGER NOT NULL,
            has_tools INTEGER NOT NULL,
            has_market INTEGER NOT NULL,
            has_portfolio INTEGER NOT NULL,
            has_strategy INTEGER NOT NULL,
            has_config INTEGER NOT NULL,
            has_memory INTEGER NOT NULL,
            context_complete INTEGER NOT NULL,
            missing_blocks_json TEXT,
            label_quality TEXT NOT NULL,
            label_confidence TEXT NOT NULL,
            ingest_version TEXT,
            transform_version TEXT NOT NULL,
            built_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_interp_examples_v0_decision
        ON interp_examples_v0(decision_type, context_complete, parse_ok, created_at);
        CREATE INDEX IF NOT EXISTS idx_interp_examples_v0_vault_log
        ON interp_examples_v0(vault_address, log_id);
        """
    )

    query = """
        SELECT
            l.id AS log_id,
            l.vault_address,
            l.created_at,
            l.strategy_id,
            l.transaction_hash,
            l.tool,
            l.tool_args_json,
            f.payload_gz,
            f.payload_path,
            f.parse_error,
            f.completion_text,
            f.reasoning_content,
            f.tool_calls_json,
            f.llm_model,
            f.fetched_at
        FROM inference_logs l
        INNER JOIN full_logs f ON f.log_id = l.id
        ORDER BY l.id DESC
        LIMIT ?
    """
    rows = conn.execute(query, [config.limit]).fetchall()
    print(f"  Queried {len(rows)} rows from inference_logs + full_logs")

    # Check if we have inline payloads or need file-based fallback
    has_inline = rows and rows[0]["payload_gz"] is not None
    if not has_inline and rows:
        _init_payload_remap(rows[0]["payload_path"])
        print("  Using file-based payload loading (legacy DB)")
    else:
        print("  Using inline payload_gz (fast path)")

    # Pre-fetch lookup tables into memory (eliminates per-row queries)
    import time as _time
    t0 = _time.monotonic()
    swap_lookup = _prefetch_swaps(conn)
    outcome_lookup = _prefetch_outcomes(conn)
    vault_cfg_lookup = _prefetch_vault_configs(conn)
    strategy_lookup = _prefetch_strategies(conn)
    print(f"  Prefetched lookups in {_time.monotonic() - t0:.1f}s "
          f"(swaps={len(swap_lookup)}, outcomes={len(outcome_lookup)}, "
          f"vaults={len(vault_cfg_lookup)}, strategies={len(strategy_lookup)})")

    # Get existing example_ids to skip re-processing
    existing_ids: set[str] = set()
    if _table_exists(conn, "interp_examples_v0"):
        existing_ids = {
            r[0] for r in conn.execute("SELECT example_id FROM interp_examples_v0").fetchall()
        }
        print(f"  {len(existing_ids)} existing examples (will skip)")

    rows_to_process = []
    skipped = 0
    for row in rows:
        example_id = f"{row['vault_address']}:{row['log_id']}"
        if example_id in existing_ids:
            skipped += 1
        else:
            rows_to_process.append(row)
    print(f"  {skipped} already exist, {len(rows_to_process)} to process")

    # Pre-load payloads: inline from DB (fast) or from files in parallel (legacy)
    if rows_to_process and has_inline:
        t1 = _time.monotonic()
        payloads = []
        for r in rows_to_process:
            gz = r["payload_gz"]
            if gz:
                try:
                    payloads.append(json.loads(gzip.decompress(gz)))
                except Exception:
                    payloads.append(None)
            else:
                payloads.append(_load_payload(r["payload_path"]))
        loaded = sum(1 for p in payloads if p is not None)
        print(f"  Decompressed {loaded}/{len(payloads)} inline payloads in {_time.monotonic() - t1:.1f}s")
    elif rows_to_process:
        from concurrent.futures import ThreadPoolExecutor
        t1 = _time.monotonic()
        payload_paths = [r["payload_path"] for r in rows_to_process]
        with ThreadPoolExecutor(max_workers=32) as pool:
            payloads = list(pool.map(_load_payload, payload_paths))
        loaded = sum(1 for p in payloads if p is not None)
        print(f"  Loaded {loaded}/{len(payloads)} payloads from files in {_time.monotonic() - t1:.1f}s")
    else:
        payloads = []

    inserted = 0
    focused = 0
    t0 = _time.monotonic()
    for i, row in enumerate(rows_to_process):
        payload = payloads[i]
        context = _extract_context_blocks(payload)
        if context["strategy"] is None:
            vault_addr = str(row["vault_address"])
            strats = strategy_lookup.get(vault_addr, [])
            if strats:
                strategy_id = row["strategy_id"]
                matched = None
                if strategy_id is not None:
                    for s in strats:
                        if str(s.get("strategy_id")) == str(strategy_id):
                            matched = s
                            break
                context["strategy"] = {
                    "strategy_id_from_log": strategy_id,
                    "matched_strategy": matched,
                    "vault_strategies": strats,
                }
        tool_calls = context["tool_calls"] if context["tool_calls"] is not None else _json_loads(row["tool_calls_json"])
        decision = _extract_decision(
            tool_calls=tool_calls,
            tool_fallback=row["tool"],
            tool_args_json=row["tool_args_json"],
        )
        if config.only_focus_decisions and decision["decision_type"] not in {"trade", "record_observation"}:
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
        context_complete = has_messages and has_market and has_portfolio and has_strategy and has_config
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
        elif parse_ok and len(missing) <= 2 and all(x in {"memory", "tools"} for x in missing):
            quality = "medium"
        else:
            quality = "low"

        log_id = int(row["log_id"])
        swap = swap_lookup.get(log_id, {})
        joined_swap = bool(swap)
        outcome = outcome_lookup.get(log_id, {})
        vault_cfg = vault_cfg_lookup.get(str(row["vault_address"]), {})

        example_id = f"{row['vault_address']}:{row['log_id']}"
        ingest_version = row["fetched_at"]
        conn.execute(
            """
            INSERT INTO interp_examples_v0 (
                example_id, log_id, vault_address, created_at, strategy_id, transaction_hash, is_trade,
                prompt_messages_json, system_text, user_text, tools_available_json,
                market_snapshot_json, portfolio_snapshot_json, strategy_snapshot_json, config_snapshot_json,
                memory_snapshot_json, model_source, assistant_content, reasoning_content, tool_calls_json,
                action_name, decision_type, trade_side, asset, size, observation_text,
                joined_swap, swap_side, swap_token_address, swap_token_symbol, swap_price_usd,
                pnl_1h_pct, pnl_4h_pct, pnl_1d_pct, was_profitable_1h,
                entry_price_usd, entry_price_eth,
                vault_trade_size, vault_trading_activity, vault_holding_style, vault_diversification, vault_risk_preference,
                parse_ok, parse_error, has_messages, has_tools, has_market, has_portfolio, has_strategy, has_config, has_memory,
                context_complete, missing_blocks_json, label_quality, label_confidence,
                ingest_version, transform_version, built_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?
            )
            ON CONFLICT(example_id) DO UPDATE SET
                created_at=excluded.created_at,
                strategy_id=excluded.strategy_id,
                transaction_hash=excluded.transaction_hash,
                is_trade=excluded.is_trade,
                prompt_messages_json=excluded.prompt_messages_json,
                system_text=excluded.system_text,
                user_text=excluded.user_text,
                tools_available_json=excluded.tools_available_json,
                market_snapshot_json=excluded.market_snapshot_json,
                portfolio_snapshot_json=excluded.portfolio_snapshot_json,
                strategy_snapshot_json=excluded.strategy_snapshot_json,
                config_snapshot_json=excluded.config_snapshot_json,
                memory_snapshot_json=excluded.memory_snapshot_json,
                model_source=excluded.model_source,
                assistant_content=excluded.assistant_content,
                reasoning_content=excluded.reasoning_content,
                tool_calls_json=excluded.tool_calls_json,
                action_name=excluded.action_name,
                decision_type=excluded.decision_type,
                trade_side=excluded.trade_side,
                asset=excluded.asset,
                size=excluded.size,
                observation_text=excluded.observation_text,
                joined_swap=excluded.joined_swap,
                swap_side=excluded.swap_side,
                swap_token_address=excluded.swap_token_address,
                swap_token_symbol=excluded.swap_token_symbol,
                swap_price_usd=excluded.swap_price_usd,
                pnl_1h_pct=excluded.pnl_1h_pct,
                pnl_4h_pct=excluded.pnl_4h_pct,
                pnl_1d_pct=excluded.pnl_1d_pct,
                was_profitable_1h=excluded.was_profitable_1h,
                entry_price_usd=excluded.entry_price_usd,
                entry_price_eth=excluded.entry_price_eth,
                vault_trade_size=excluded.vault_trade_size,
                vault_trading_activity=excluded.vault_trading_activity,
                vault_holding_style=excluded.vault_holding_style,
                vault_diversification=excluded.vault_diversification,
                vault_risk_preference=excluded.vault_risk_preference,
                parse_ok=excluded.parse_ok,
                parse_error=excluded.parse_error,
                has_messages=excluded.has_messages,
                has_tools=excluded.has_tools,
                has_market=excluded.has_market,
                has_portfolio=excluded.has_portfolio,
                has_strategy=excluded.has_strategy,
                has_config=excluded.has_config,
                has_memory=excluded.has_memory,
                context_complete=excluded.context_complete,
                missing_blocks_json=excluded.missing_blocks_json,
                label_quality=excluded.label_quality,
                label_confidence=excluded.label_confidence,
                ingest_version=excluded.ingest_version,
                transform_version=excluded.transform_version,
                built_at=excluded.built_at
            """,
            (
                example_id,
                int(row["log_id"]),
                row["vault_address"],
                row["created_at"],
                row["strategy_id"],
                row["transaction_hash"],
                1 if decision["decision_type"] == "trade" else 0,
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
                context["assistant_content"] if context["assistant_content"] is not None else row["completion_text"],
                context["reasoning_content"] if context["reasoning_content"] is not None else row["reasoning_content"],
                _json_dumps(tool_calls),
                decision["action_name"],
                decision["decision_type"],
                decision["trade_side"],
                decision["asset"],
                None if decision["size"] is None else str(decision["size"]),
                decision["observation_text"],
                1 if joined_swap else 0,
                swap.get("side"),
                swap.get("token_address"),
                swap.get("token_symbol"),
                swap.get("effective_price_usd"),
                outcome.get("pnl_1h_pct"),
                outcome.get("pnl_4h_pct"),
                outcome.get("pnl_1d_pct"),
                outcome.get("was_profitable_1h"),
                outcome.get("entry_price_usd"),
                outcome.get("entry_price_eth"),
                vault_cfg.get("trade_size"),
                vault_cfg.get("trading_activity"),
                vault_cfg.get("holding_style"),
                vault_cfg.get("diversification"),
                vault_cfg.get("asset_risk_preference"),
                1 if parse_ok else 0,
                row["parse_error"],
                1 if has_messages else 0,
                1 if has_tools else 0,
                1 if has_market else 0,
                1 if has_portfolio else 0,
                1 if has_strategy else 0,
                1 if has_config else 0,
                1 if has_memory else 0,
                1 if context_complete else 0,
                _json_dumps(missing),
                quality,
                quality,
                ingest_version,
                config.transform_version,
                _now_iso(),
            ),
        )
        inserted += 1
        if inserted % 500 == 0:
            conn.commit()
            elapsed = _time.monotonic() - t0
            rate = inserted / elapsed if elapsed > 0 else 0
            print(f"  {inserted} inserted, {i+1}/{len(rows)} scanned ({rate:.0f} rows/s)")

    conn.commit()
    elapsed = _time.monotonic() - t0
    print(f"  Done: {inserted} inserted, {skipped} skipped, {focused} focused in {elapsed:.1f}s")

    conn.executescript(
        """
        DROP TABLE IF EXISTS interp_context_gaps_v0;
        CREATE TABLE interp_context_gaps_v0 AS
        SELECT
            example_id,
            log_id,
            vault_address,
            created_at,
            decision_type,
            parse_ok,
            context_complete,
            missing_blocks_json,
            parse_error,
            label_quality
        FROM interp_examples_v0
        WHERE parse_ok = 0 OR context_complete = 0;

        DROP TABLE IF EXISTS interp_sample_trade_v0;
        CREATE TABLE interp_sample_trade_v0 AS
        WITH ranked AS (
            SELECT *
            FROM interp_examples_v0
            WHERE decision_type = 'trade'
              AND label_quality = 'high'
            ORDER BY created_at DESC, log_id DESC
        )
        SELECT * FROM ranked LIMIT 1;

        DROP TABLE IF EXISTS interp_sample_observation_v0;
        CREATE TABLE interp_sample_observation_v0 AS
        WITH ranked AS (
            SELECT *
            FROM interp_examples_v0
            WHERE decision_type = 'record_observation'
              AND label_quality = 'high'
            ORDER BY created_at DESC, log_id DESC
        )
        SELECT * FROM ranked LIMIT 1;

        DROP TABLE IF EXISTS interp_sample_paired_v0;
        CREATE TABLE interp_sample_paired_v0 AS
        SELECT a.*
        FROM interp_examples_v0 a
        WHERE a.label_quality = 'high'
          AND a.decision_type IN ('trade', 'record_observation')
          AND EXISTS (
              SELECT 1
              FROM interp_examples_v0 b
              WHERE b.vault_address = a.vault_address
                AND b.decision_type != a.decision_type
                AND abs(b.log_id - a.log_id) <= 10000
          )
        ORDER BY a.created_at DESC, a.log_id DESC
        LIMIT 1;
        """
    )

    # Resize sample tables to configured sizes.
    conn.execute("DELETE FROM interp_sample_trade_v0")
    conn.execute(
        """
        INSERT INTO interp_sample_trade_v0
        SELECT * FROM (
            SELECT *
            FROM interp_examples_v0
            WHERE decision_type = 'trade'
              AND label_quality = 'high'
              AND trade_side = 'buy'
            ORDER BY created_at DESC, log_id DESC
            LIMIT ?
        )
        UNION ALL
        SELECT * FROM (
            SELECT *
            FROM interp_examples_v0
            WHERE decision_type = 'trade'
              AND label_quality = 'high'
              AND trade_side = 'sell'
            ORDER BY created_at DESC, log_id DESC
            LIMIT ?
        )
        LIMIT ?
        """,
        [
            max(1, config.trade_sample_size // 2),
            max(1, config.trade_sample_size // 2),
            config.trade_sample_size,
        ],
    )

    conn.execute("DELETE FROM interp_sample_observation_v0")
    conn.execute(
        """
        INSERT INTO interp_sample_observation_v0
        SELECT *
        FROM interp_examples_v0
        WHERE decision_type = 'record_observation'
          AND label_quality = 'high'
        ORDER BY created_at DESC, log_id DESC
        LIMIT ?
        """,
        [config.observation_sample_size],
    )

    conn.execute("DELETE FROM interp_sample_paired_v0")
    conn.execute(
        """
        INSERT INTO interp_sample_paired_v0
        SELECT a.*
        FROM interp_examples_v0 a
        WHERE a.label_quality = 'high'
          AND a.decision_type IN ('trade', 'record_observation')
          AND EXISTS (
              SELECT 1
              FROM interp_examples_v0 b
              WHERE b.vault_address = a.vault_address
                AND b.decision_type != a.decision_type
                AND abs(b.log_id - a.log_id) <= 10000
          )
        ORDER BY a.created_at DESC, a.log_id DESC
        LIMIT ?
        """,
        [config.paired_sample_size],
    )
    conn.commit()

    totals = conn.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN decision_type='trade' THEN 1 ELSE 0 END) AS trade_count,
            SUM(CASE WHEN decision_type='record_observation' THEN 1 ELSE 0 END) AS observation_count,
            SUM(CASE WHEN label_quality='high' THEN 1 ELSE 0 END) AS high_quality_count,
            SUM(CASE WHEN context_complete=1 THEN 1 ELSE 0 END) AS context_complete_count,
            SUM(CASE WHEN parse_ok=1 THEN 1 ELSE 0 END) AS parse_ok_count
        FROM interp_examples_v0
        """
    ).fetchone()
    gap_count = conn.execute("SELECT COUNT(*) AS c FROM interp_context_gaps_v0").fetchone()["c"]
    trade_sample = conn.execute("SELECT COUNT(*) AS c FROM interp_sample_trade_v0").fetchone()["c"]
    obs_sample = conn.execute("SELECT COUNT(*) AS c FROM interp_sample_observation_v0").fetchone()["c"]
    pair_sample = conn.execute("SELECT COUNT(*) AS c FROM interp_sample_paired_v0").fetchone()["c"]
    export_count = 0

    if config.export_jsonl:
        export_count = _export_jsonl_tables(conn, config.export_dir)

    parquet_count = 0
    if config.export_parquet:
        parquet_count = _export_parquet_tables(conn, config.export_dir)

    conn.close()
    return {
        "rows_scanned": len(rows),
        "rows_focus_kept": focused,
        "rows_written": inserted,
        "total_examples": int(totals["total"] or 0),
        "trade_count": int(totals["trade_count"] or 0),
        "observation_count": int(totals["observation_count"] or 0),
        "high_quality_count": int(totals["high_quality_count"] or 0),
        "context_complete_count": int(totals["context_complete_count"] or 0),
        "parse_ok_count": int(totals["parse_ok_count"] or 0),
        "gap_count": int(gap_count or 0),
        "trade_sample_count": int(trade_sample or 0),
        "observation_sample_count": int(obs_sample or 0),
        "paired_sample_count": int(pair_sample or 0),
        "export_files_written": int(export_count),
        "parquet_files_written": int(parquet_count),
    }


def _write_query_to_jsonl(conn: sqlite3.Connection, sql: str, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = conn.execute(sql).fetchall()
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, separators=(",", ":")))
            handle.write("\n")
    return len(rows)


def _export_jsonl_tables(conn: sqlite3.Connection, export_dir: Path) -> int:
    queries = {
        "interp_examples_v0_high_quality.jsonl": """
            SELECT *
            FROM interp_examples_v0
            WHERE label_quality = 'high'
            ORDER BY created_at DESC, log_id DESC
        """,
        "interp_sample_trade_v0.jsonl": """
            SELECT *
            FROM interp_sample_trade_v0
            ORDER BY created_at DESC, log_id DESC
        """,
        "interp_sample_observation_v0.jsonl": """
            SELECT *
            FROM interp_sample_observation_v0
            ORDER BY created_at DESC, log_id DESC
        """,
        "interp_sample_paired_v0.jsonl": """
            SELECT *
            FROM interp_sample_paired_v0
            ORDER BY created_at DESC, log_id DESC
        """,
        "interp_context_gaps_v0.jsonl": """
            SELECT *
            FROM interp_context_gaps_v0
            ORDER BY created_at DESC, log_id DESC
        """,
    }
    files_written = 0
    for name, sql in queries.items():
        out_path = export_dir / name
        _ = _write_query_to_jsonl(conn, sql, out_path)
        files_written += 1
    return files_written


def _export_parquet_tables(conn: sqlite3.Connection, export_dir: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    queries = {
        "interp_examples_v0_high_quality.parquet": """
            SELECT *
            FROM interp_examples_v0
            WHERE label_quality = 'high'
            ORDER BY created_at DESC, log_id DESC
        """,
        "interp_sample_trade_v0.parquet": """
            SELECT *
            FROM interp_sample_trade_v0
            ORDER BY created_at DESC, log_id DESC
        """,
        "interp_sample_observation_v0.parquet": """
            SELECT *
            FROM interp_sample_observation_v0
            ORDER BY created_at DESC, log_id DESC
        """,
        "interp_sample_paired_v0.parquet": """
            SELECT *
            FROM interp_sample_paired_v0
            ORDER BY created_at DESC, log_id DESC
        """,
        "interp_context_gaps_v0.parquet": """
            SELECT *
            FROM interp_context_gaps_v0
            ORDER BY created_at DESC, log_id DESC
        """,
    }

    export_dir.mkdir(parents=True, exist_ok=True)
    files_written = 0

    for name, sql in queries.items():
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        if not rows:
            continue

        data = {col: [row[idx] for row in rows] for idx, col in enumerate(columns)}
        table = pa.Table.from_pydict(data)
        out_path = export_dir / name
        pq.write_table(table, out_path, compression="snappy")
        files_written += 1
        print(f"  Wrote {len(rows)} rows to {out_path}")

    return files_written


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build context-complete interp dataset tables from Xenon full logs"
    )
    parser.add_argument("--db-path", type=Path, default=Path("data/terminal_ingest.db"))
    parser.add_argument("--limit", type=int, default=50_000)
    parser.add_argument(
        "--include-all-decisions",
        action="store_true",
        help="Keep all decisions, not just trade + record_observation",
    )
    parser.add_argument("--trade-sample-size", type=int, default=150)
    parser.add_argument("--observation-sample-size", type=int, default=150)
    parser.add_argument("--paired-sample-size", type=int, default=100)
    parser.add_argument("--transform-version", default="interp_examples_v0.2")
    parser.add_argument(
        "--export-jsonl",
        action="store_true",
        help="Export high-quality and sample tables to JSONL files",
    )
    parser.add_argument(
        "--export-parquet",
        action="store_true",
        help="Export high-quality and sample tables to Parquet files",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("data/interp_exports"),
        help="Directory for exports when --export-jsonl or --export-parquet is enabled",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = PrepareConfig(
        db_path=args.db_path,
        limit=max(1, int(args.limit)),
        only_focus_decisions=not bool(args.include_all_decisions),
        trade_sample_size=max(1, int(args.trade_sample_size)),
        observation_sample_size=max(1, int(args.observation_sample_size)),
        paired_sample_size=max(1, int(args.paired_sample_size)),
        transform_version=str(args.transform_version),
        export_jsonl=bool(args.export_jsonl),
        export_parquet=bool(args.export_parquet),
        export_dir=args.export_dir,
    )
    stats = run_prepare(cfg)
    print("Interp dataset prep complete")
    for key, value in stats.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
