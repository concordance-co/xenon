from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import aiosqlite

from pipelines.ingest.full_log_parser import ParsedFullLog
from pipelines.ingest.payload_store import RawPayloadMetadata


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _as_bool_int(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0


@dataclass(slots=True)
class FullLogRecord:
    log_id: int
    vault_address: str | None
    payload_meta: RawPayloadMetadata
    parsed: ParsedFullLog


class IngestDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = await aiosqlite.connect(self.db_path)
        await self.conn.execute("PRAGMA journal_mode=WAL;")
        await self.conn.execute("PRAGMA synchronous=NORMAL;")
        await self.conn.execute("PRAGMA foreign_keys=ON;")

    async def close(self) -> None:
        if self.conn is not None:
            await self.conn.close()
            self.conn = None

    async def init_schema(self) -> None:
        assert self.conn is not None
        await self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS vaults (
                vault_address TEXT PRIMARY KEY,
                owner_address TEXT,
                nft_id TEXT,
                nft_name TEXT,
                persona_json TEXT,
                trade_size INTEGER,
                trading_activity INTEGER,
                holding_style INTEGER,
                diversification INTEGER,
                asset_risk_preference INTEGER,
                max_trade_amount TEXT,
                slippage_bps TEXT,
                paused INTEGER,
                state TEXT,
                leaderboard_rank INTEGER,
                total_pnl_usd REAL,
                realized_pnl_usd REAL,
                unrealized_pnl_usd REAL,
                created_block INTEGER,
                updated_block INTEGER,
                fetched_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS strategies (
                vault_address TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                vault_owner_address TEXT,
                content TEXT,
                expiry INTEGER,
                enabled INTEGER,
                strategy_priority TEXT,
                created_block INTEGER,
                updated_block INTEGER,
                fetched_at TEXT NOT NULL,
                PRIMARY KEY (vault_address, strategy_id),
                FOREIGN KEY (vault_address) REFERENCES vaults(vault_address)
            );

            CREATE TABLE IF NOT EXISTS inference_logs (
                id INTEGER PRIMARY KEY,
                cursor TEXT,
                vault_address TEXT NOT NULL,
                request_id TEXT,
                execution_key TEXT,
                tool TEXT,
                tool_args_json TEXT,
                strategy_id TEXT,
                status TEXT,
                inference_duration_ms INTEGER,
                error TEXT,
                transaction_hash TEXT,
                created_at TEXT,
                completed_at TEXT,
                fetched_at TEXT NOT NULL,
                FOREIGN KEY (vault_address) REFERENCES vaults(vault_address)
            );

            CREATE INDEX IF NOT EXISTS idx_inference_logs_vault
            ON inference_logs(vault_address, id);

            CREATE TABLE IF NOT EXISTS full_logs (
                log_id INTEGER PRIMARY KEY,
                vault_address TEXT,
                payload_path TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                payload_size_bytes INTEGER NOT NULL,
                prompt_text TEXT,
                completion_text TEXT,
                reasoning_content TEXT,
                tool_calls_json TEXT,
                llm_model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                reasoning_tokens INTEGER,
                total_tokens INTEGER,
                parse_error TEXT,
                fetched_at TEXT NOT NULL,
                FOREIGN KEY (log_id) REFERENCES inference_logs(id)
            );
            """
        )
        await self.conn.commit()

    async def upsert_vault(self, leaderboard_item: dict[str, Any], vault_config: dict[str, Any]) -> None:
        assert self.conn is not None
        now = _now_iso()
        await self.conn.execute(
            """
            INSERT INTO vaults (
                vault_address, owner_address, nft_id, nft_name, persona_json,
                trade_size, trading_activity, holding_style, diversification,
                asset_risk_preference, max_trade_amount, slippage_bps, paused, state,
                leaderboard_rank, total_pnl_usd, realized_pnl_usd, unrealized_pnl_usd,
                created_block, updated_block, fetched_at
            ) VALUES (
                :vault_address, :owner_address, :nft_id, :nft_name, :persona_json,
                :trade_size, :trading_activity, :holding_style, :diversification,
                :asset_risk_preference, :max_trade_amount, :slippage_bps, :paused, :state,
                :leaderboard_rank, :total_pnl_usd, :realized_pnl_usd, :unrealized_pnl_usd,
                :created_block, :updated_block, :fetched_at
            )
            ON CONFLICT(vault_address) DO UPDATE SET
                owner_address=excluded.owner_address,
                nft_id=excluded.nft_id,
                nft_name=excluded.nft_name,
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
                leaderboard_rank=excluded.leaderboard_rank,
                total_pnl_usd=excluded.total_pnl_usd,
                realized_pnl_usd=excluded.realized_pnl_usd,
                unrealized_pnl_usd=excluded.unrealized_pnl_usd,
                created_block=excluded.created_block,
                updated_block=excluded.updated_block,
                fetched_at=excluded.fetched_at
            """,
            {
                "vault_address": leaderboard_item.get("vaultAddress") or vault_config.get("vaultAddress"),
                "owner_address": leaderboard_item.get("ownerAddress") or vault_config.get("ownerAddress"),
                "nft_id": leaderboard_item.get("nftId") or vault_config.get("nftId"),
                "nft_name": leaderboard_item.get("nftName") or vault_config.get("nftName"),
                "persona_json": json.dumps(vault_config.get("persona"), ensure_ascii=True),
                "trade_size": vault_config.get("tradeSize"),
                "trading_activity": vault_config.get("tradingActivity"),
                "holding_style": vault_config.get("holdingStyle"),
                "diversification": vault_config.get("diversification"),
                "asset_risk_preference": vault_config.get("assetRiskPreference"),
                "max_trade_amount": vault_config.get("maxTradeAmount"),
                "slippage_bps": vault_config.get("slippageBps"),
                "paused": _as_bool_int(vault_config.get("paused")),
                "state": vault_config.get("state"),
                "leaderboard_rank": leaderboard_item.get("rank"),
                "total_pnl_usd": leaderboard_item.get("totalPnlUsd"),
                "realized_pnl_usd": leaderboard_item.get("realizedPnlUsd"),
                "unrealized_pnl_usd": leaderboard_item.get("unrealizedPnlUsd"),
                "created_block": vault_config.get("createdBlock"),
                "updated_block": vault_config.get("updatedBlock"),
                "fetched_at": now,
            },
        )
        await self.conn.commit()

    async def upsert_strategies(self, vault_address: str, strategies: Iterable[dict[str, Any]]) -> None:
        assert self.conn is not None
        now = _now_iso()
        rows = []
        for strategy in strategies:
            rows.append(
                (
                    vault_address,
                    strategy.get("strategyId"),
                    strategy.get("vaultOwnerAddress"),
                    strategy.get("content"),
                    strategy.get("expiry"),
                    _as_bool_int(strategy.get("enabled")),
                    strategy.get("strategyPriority"),
                    strategy.get("createdBlock"),
                    strategy.get("updatedBlock"),
                    now,
                )
            )
        if rows:
            await self.conn.executemany(
                """
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
                """,
                rows,
            )
            await self.conn.commit()

    async def upsert_inference_logs(self, logs: Iterable[dict[str, Any]]) -> None:
        assert self.conn is not None
        now = _now_iso()
        rows = []
        for log_item in logs:
            rows.append(
                (
                    log_item.get("id"),
                    log_item.get("cursor"),
                    log_item.get("vault_address"),
                    log_item.get("request_id"),
                    log_item.get("execution_key"),
                    log_item.get("tool"),
                    json.dumps(log_item.get("tool_args"), ensure_ascii=True, separators=(",", ":")),
                    log_item.get("strategyId"),
                    log_item.get("status"),
                    log_item.get("inference_duration_ms"),
                    log_item.get("error"),
                    log_item.get("transactionHash"),
                    log_item.get("created_at"),
                    log_item.get("completed_at"),
                    now,
                )
            )
        if rows:
            await self.conn.executemany(
                """
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
                """,
                rows,
            )
            await self.conn.commit()

    async def fetch_existing_full_log_ids(self, log_ids: list[int]) -> set[int]:
        assert self.conn is not None
        if not log_ids:
            return set()
        placeholders = ",".join("?" for _ in log_ids)
        cursor = await self.conn.execute(
            f"SELECT log_id FROM full_logs WHERE log_id IN ({placeholders})",
            log_ids,
        )
        rows = await cursor.fetchall()
        return {int(row[0]) for row in rows}

    async def upsert_full_log(self, record: FullLogRecord) -> None:
        assert self.conn is not None
        await self.conn.execute(
            """
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
            """,
            (
                record.log_id,
                record.vault_address,
                record.payload_meta.payload_path,
                record.payload_meta.payload_sha256,
                record.payload_meta.payload_size_bytes,
                record.parsed.prompt_text,
                record.parsed.completion_text,
                record.parsed.reasoning_content,
                record.parsed.tool_calls_json,
                record.parsed.llm_model,
                record.parsed.prompt_tokens,
                record.parsed.completion_tokens,
                record.parsed.reasoning_tokens,
                record.parsed.total_tokens,
                record.parsed.parse_error,
                _now_iso(),
            ),
        )
        await self.conn.commit()
