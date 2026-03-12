#!/usr/bin/env python3
"""Rebuild ingest DB from local full_logs JSON.gz files.

This mirrors the Modal rebuild-from-files logic, but runs locally.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import sys
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

# Ensure repo root is on sys.path when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.ingest.db import IngestDatabase
from pipelines.ingest.full_log_parser import parse_full_log


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, default=Path("data/full_logs"))
    p.add_argument("--db-path", type=Path, default=Path("data/terminal_ingest.db"))
    p.add_argument("--backup-dir", type=Path, default=Path("data/rebuild_backups"))
    p.add_argument("--limit", type=int, default=-1, help="-1 means all files")
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--no-reset", action="store_true", help="Do not wipe existing DB first")
    return p.parse_args()


async def _init_schema(db_path: Path) -> None:
    db = IngestDatabase(db_path)
    await db.connect()
    await db.init_schema()
    await db.close()


def main() -> int:
    args = _parse_args()
    raw_dir = args.input_dir
    db_path = args.db_path
    backup_dir = args.backup_dir
    batch_size = max(100, args.batch_size)
    reset_db = not args.no_reset

    if not raw_dir.exists():
        raise SystemExit(f"Missing input dir: {raw_dir}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)

    wal_path = db_path.with_name(f"{db_path.name}-wal")
    shm_path = db_path.with_name(f"{db_path.name}-shm")

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if reset_db and db_path.exists():
        shutil.copy2(db_path, backup_dir / f"{db_path.stem}.pre_rebuild.{ts}.db")
        if wal_path.exists():
            shutil.copy2(wal_path, backup_dir / f"{db_path.stem}.pre_rebuild.{ts}.db-wal")
        if shm_path.exists():
            shutil.copy2(shm_path, backup_dir / f"{db_path.stem}.pre_rebuild.{ts}.db-shm")
        db_path.unlink()
        if wal_path.exists():
            wal_path.unlink()
        if shm_path.exists():
            shm_path.unlink()

    asyncio.run(_init_schema(db_path))

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=OFF;")

    now_iso = datetime.now(UTC).isoformat()

    vault_sql = """
        INSERT INTO vaults (
            vault_address, owner_address, nft_id, nft_name, persona_json,
            trade_size, trading_activity, holding_style, diversification,
            asset_risk_preference, max_trade_amount, slippage_bps, paused, state,
            leaderboard_rank, total_pnl_usd, realized_pnl_usd, unrealized_pnl_usd,
            created_block, updated_block, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(vault_address) DO UPDATE SET
            owner_address=excluded.owner_address,
            nft_id=excluded.nft_id,
            persona_json=excluded.persona_json,
            trade_size=excluded.trade_size,
            trading_activity=excluded.trading_activity,
            holding_style=excluded.holding_style,
            diversification=excluded.diversification,
            asset_risk_preference=excluded.asset_risk_preference,
            max_trade_amount=excluded.max_trade_amount,
            slippage_bps=excluded.slippage_bps,
            paused=excluded.paused,
            state=excluded.state,
            fetched_at=excluded.fetched_at
    """

    strategy_sql = """
        INSERT INTO strategies (
            vault_address, strategy_id, vault_owner_address, content, expiry,
            enabled, strategy_priority, created_block, updated_block, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(vault_address, strategy_id) DO UPDATE SET
            vault_owner_address=excluded.vault_owner_address,
            content=excluded.content,
            expiry=excluded.expiry,
            enabled=excluded.enabled,
            strategy_priority=excluded.strategy_priority,
            created_block=excluded.created_block,
            updated_block=excluded.updated_block,
            fetched_at=excluded.fetched_at
    """

    inference_sql = """
        INSERT INTO inference_logs (
            id, cursor, vault_address, request_id, execution_key, tool, tool_args_json,
            strategy_id, status, inference_duration_ms, error, transaction_hash,
            created_at, completed_at, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            cursor=excluded.cursor,
            vault_address=excluded.vault_address,
            request_id=excluded.request_id,
            execution_key=excluded.execution_key,
            tool=excluded.tool,
            tool_args_json=excluded.tool_args_json,
            strategy_id=excluded.strategy_id,
            status=excluded.status,
            inference_duration_ms=excluded.inference_duration_ms,
            error=excluded.error,
            transaction_hash=excluded.transaction_hash,
            created_at=excluded.created_at,
            completed_at=excluded.completed_at,
            fetched_at=excluded.fetched_at
    """

    full_log_sql = """
        INSERT INTO full_logs (
            log_id, vault_address, payload_path, payload_sha256, payload_size_bytes,
            prompt_text, completion_text, reasoning_content, tool_calls_json,
            llm_model, prompt_tokens, completion_tokens, reasoning_tokens, total_tokens,
            parse_error, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(log_id) DO UPDATE SET
            vault_address=excluded.vault_address,
            payload_path=excluded.payload_path,
            payload_sha256=excluded.payload_sha256,
            payload_size_bytes=excluded.payload_size_bytes,
            prompt_text=excluded.prompt_text,
            completion_text=excluded.completion_text,
            reasoning_content=excluded.reasoning_content,
            tool_calls_json=excluded.tool_calls_json,
            llm_model=excluded.llm_model,
            prompt_tokens=excluded.prompt_tokens,
            completion_tokens=excluded.completion_tokens,
            reasoning_tokens=excluded.reasoning_tokens,
            total_tokens=excluded.total_tokens,
            parse_error=excluded.parse_error,
            fetched_at=excluded.fetched_at
    """

    files = sorted(raw_dir.rglob("*.json.gz"))
    if args.limit >= 0:
        files = files[: args.limit]
    total_files = len(files)
    print(f"Found {total_files} files in {raw_dir}")

    vault_rows: dict[str, tuple] = {}
    strategy_rows: dict[tuple[str, str], tuple] = {}
    inference_rows: list[tuple] = []
    full_rows: list[tuple] = []

    processed = 0
    parse_failures = 0

    def _flush() -> None:
        if vault_rows:
            conn.executemany(vault_sql, list(vault_rows.values()))
            vault_rows.clear()
        if strategy_rows:
            conn.executemany(strategy_sql, list(strategy_rows.values()))
            strategy_rows.clear()
        if inference_rows:
            conn.executemany(inference_sql, inference_rows)
            inference_rows.clear()
        if full_rows:
            conn.executemany(full_log_sql, full_rows)
            full_rows.clear()
        conn.commit()

    for path in files:
        processed += 1
        try:
            raw_bytes = path.read_bytes()
            payload = json.loads(gzip.decompress(raw_bytes).decode("utf-8"))
            if not isinstance(payload, dict):
                parse_failures += 1
                continue

            log_id_raw = payload.get("id")
            if isinstance(log_id_raw, int):
                log_id = log_id_raw
            else:
                stem = path.name[: -len(".json.gz")] if path.name.endswith(".json.gz") else path.stem
                if not stem.isdigit():
                    parse_failures += 1
                    continue
                log_id = int(stem)

            vault_address = payload.get("vault_address")
            if not isinstance(vault_address, str) or not vault_address:
                snapshot = payload.get("snapshot")
                agent = snapshot.get("Agent", {}) if isinstance(snapshot, dict) else {}
                vault_address = agent.get("VaultAddress") if isinstance(agent, dict) else None
            if not isinstance(vault_address, str) or not vault_address:
                parse_failures += 1
                continue

            snapshot = payload.get("snapshot", {})
            agent = snapshot.get("Agent", {}) if isinstance(snapshot, dict) else {}
            options = agent.get("Options", {}) if isinstance(agent, dict) else {}
            persona = agent.get("Persona")
            owner = agent.get("OwnerAddress")
            nft_id = agent.get("CurrentNftId")
            state = agent.get("State")
            paused = agent.get("Paused")

            def _opt(*names: str):
                for n in names:
                    if isinstance(options, dict) and n in options:
                        return options.get(n)
                return None

            vault_rows[vault_address] = (
                vault_address,
                owner,
                str(nft_id) if nft_id is not None else None,
                None,
                json.dumps(persona, ensure_ascii=True) if persona is not None else None,
                _opt("tradeSize", "TradeSize"),
                _opt("tradingActivity", "TradingActivity"),
                _opt("holdingStyle", "HoldingStyle"),
                _opt("diversification", "Diversification"),
                _opt("assetRiskPreference", "AssetRiskPreference"),
                _opt("maxTradeAmount", "MaxTradeAmount"),
                _opt("slippageBps", "SlippageBps"),
                1 if paused else 0 if paused is not None else None,
                state,
                None,
                None,
                None,
                None,
                None,
                None,
                now_iso,
            )

            strategies = agent.get("Strategies") if isinstance(agent, dict) else None
            if isinstance(strategies, list):
                for idx, strategy in enumerate(strategies):
                    if not isinstance(strategy, dict):
                        continue
                    strategy_id = strategy.get("strategyId") or strategy.get("id") or f"recovered_{idx}"
                    sid = str(strategy_id)
                    strategy_rows[(vault_address, sid)] = (
                        vault_address,
                        sid,
                        owner,
                        strategy.get("content"),
                        strategy.get("expiry"),
                        1 if strategy.get("enabled") else 0 if strategy.get("enabled") is not None else None,
                        strategy.get("strategyPriority"),
                        strategy.get("createdBlock"),
                        strategy.get("updatedBlock"),
                        now_iso,
                    )

            tool_args = payload.get("tool_args")
            inference_rows.append(
                (
                    log_id,
                    payload.get("cursor"),
                    vault_address,
                    payload.get("request_id"),
                    payload.get("execution_key"),
                    payload.get("tool"),
                    json.dumps(tool_args, ensure_ascii=True, separators=(",", ":"))
                    if tool_args is not None
                    else None,
                    payload.get("strategy_id") or payload.get("strategyId"),
                    payload.get("status"),
                    payload.get("inference_duration_ms"),
                    payload.get("error"),
                    payload.get("transaction_hash") or payload.get("transactionHash"),
                    payload.get("created_at"),
                    payload.get("completed_at"),
                    now_iso,
                )
            )

            parsed = parse_full_log(payload, include_reasoning=True)
            full_rows.append(
                (
                    log_id,
                    vault_address,
                    str(path),
                    hashlib.sha256(raw_bytes).hexdigest(),
                    len(raw_bytes),
                    parsed.prompt_text,
                    parsed.completion_text,
                    parsed.reasoning_content,
                    parsed.tool_calls_json,
                    parsed.llm_model,
                    parsed.prompt_tokens,
                    parsed.completion_tokens,
                    parsed.reasoning_tokens,
                    parsed.total_tokens,
                    parsed.parse_error,
                    now_iso,
                )
            )
        except Exception:
            parse_failures += 1

        if processed % batch_size == 0:
            _flush()
            print(f"processed={processed}/{total_files} parse_failures={parse_failures}")

    _flush()
    conn.execute("PRAGMA foreign_keys=ON;")
    integrity = conn.execute("PRAGMA integrity_check;").fetchone()
    integrity_out = integrity[0] if integrity else "unknown"

    counts = {}
    for table in ("vaults", "strategies", "inference_logs", "full_logs"):
        row = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()
        counts[table] = int(row[0]) if row else 0
    conn.close()

    print(
        json.dumps(
            {
                "status": "ok" if integrity_out == "ok" else "warning",
                "integrity_check": integrity_out,
                "files_seen": total_files,
                "processed": processed,
                "parse_failures": parse_failures,
                "counts": counts,
                "db_path": str(db_path),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
