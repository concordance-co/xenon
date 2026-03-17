from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Iterable

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from pipelines.ingest.full_log_parser import ParsedFullLog


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


@dataclass(slots=True)
class FullLogRecord:
    log_id: int
    vault_address: str | None
    parsed: ParsedFullLog
    raw_payload: dict | None = None


class IngestDatabase:
    def __init__(self) -> None:
        self.conn: AsyncConnection | None = None
        self._batch_depth: int = 0
        self._batch_lock: asyncio.Lock = asyncio.Lock()

    async def connect(self) -> None:
        from pipelines.db import ensure_schema, require_neon_dsn
        dsn = require_neon_dsn()
        # Run sync schema migrations (INT→BIGINT, etc.) before opening async conn
        import psycopg
        with psycopg.connect(dsn, row_factory=dict_row) as sync_conn:
            ensure_schema(sync_conn)
        self.conn = await AsyncConnection.connect(dsn, autocommit=False, row_factory=dict_row)

    async def close(self) -> None:
        if self.conn is not None:
            await self.conn.close()
            self.conn = None

    @asynccontextmanager
    async def batch(self) -> AsyncIterator[None]:
        """Group multiple writes into a single transaction.

        Acquires a lock so concurrent coroutines do not interleave
        transactions on the same connection.  Supports nesting: only
        the outermost batch issues BEGIN/COMMIT.  On exception the
        outermost batch issues ROLLBACK.
        """
        if self.conn is None:
            raise RuntimeError("IngestDatabase not connected — call connect() first")
        # The lock ensures that concurrent coroutines wait their turn
        # rather than interleaving BEGIN/COMMIT pairs.
        acquired = self._batch_depth > 0  # already inside a batch (nested)
        if not acquired:
            await self._batch_lock.acquire()
        try:
            if self._batch_depth == 0:
                await self.conn.execute("BEGIN")
            self._batch_depth += 1
            try:
                yield
            except BaseException:
                self._batch_depth -= 1
                if self._batch_depth == 0 and self.conn is not None:
                    await self.conn.execute("ROLLBACK")
                raise
            else:
                self._batch_depth -= 1
                if self._batch_depth == 0:
                    await self.conn.commit()
        finally:
            if not acquired:
                self._batch_lock.release()

    async def commit(self) -> None:
        """Commit only if we are not inside a batch context.

        When inside a ``batch()`` the outermost context manager handles
        the commit, so this is intentionally a no-op in that case.
        """
        if self.conn is None:
            raise RuntimeError("IngestDatabase not connected — call connect() first")
        if self._batch_depth == 0:
            await self.conn.commit()

    async def init_schema(self) -> None:
        from pipelines.db import TABLE_DDLS, INDEX_DDLS
        if self.conn is None:
            raise RuntimeError("IngestDatabase not connected — call connect() first")
        async with self.conn.cursor() as cur:
            for ddl in TABLE_DDLS:
                await cur.execute(ddl)
            for idx in INDEX_DDLS:
                await cur.execute(idx)
        await self.conn.commit()

    async def upsert_vault(self, leaderboard_item: dict[str, Any], vault_config: dict[str, Any]) -> None:
        if self.conn is None:
            raise RuntimeError("IngestDatabase not connected — call connect() first")
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
                %(vault_address)s, %(owner_address)s, %(nft_id)s, %(nft_name)s, %(persona_json)s,
                %(trade_size)s, %(trading_activity)s, %(holding_style)s, %(diversification)s,
                %(asset_risk_preference)s, %(max_trade_amount)s, %(slippage_bps)s, %(paused)s, %(state)s,
                %(leaderboard_rank)s, %(total_pnl_usd)s, %(realized_pnl_usd)s, %(unrealized_pnl_usd)s,
                %(created_block)s, %(updated_block)s, %(fetched_at)s
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
                "persona_json": json.dumps(vault_config.get("persona"), ensure_ascii=True) if vault_config.get("persona") is not None else None,
                "trade_size": vault_config.get("tradeSize"),
                "trading_activity": vault_config.get("tradingActivity"),
                "holding_style": vault_config.get("holdingStyle"),
                "diversification": vault_config.get("diversification"),
                "asset_risk_preference": vault_config.get("assetRiskPreference"),
                "max_trade_amount": vault_config.get("maxTradeAmount"),
                "slippage_bps": vault_config.get("slippageBps"),
                "paused": _as_bool(vault_config.get("paused")),
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
        await self.commit()

    async def upsert_strategies(self, vault_address: str, strategies: Iterable[dict[str, Any]]) -> None:
        if self.conn is None:
            raise RuntimeError("IngestDatabase not connected — call connect() first")
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
                    _as_bool(strategy.get("enabled")),
                    strategy.get("strategyPriority"),
                    strategy.get("createdBlock"),
                    strategy.get("updatedBlock"),
                    now,
                )
            )
        if rows:
            async with self.conn.cursor() as cur:
                for row in rows:
                    await cur.execute(
                        """
                        INSERT INTO strategies (
                            vault_address, strategy_id, vault_owner_address, content, expiry,
                            enabled, strategy_priority, created_block, updated_block, fetched_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        row,
                    )
            await self.commit()

    async def upsert_inference_logs(self, logs: Iterable[dict[str, Any]]) -> None:
        if self.conn is None:
            raise RuntimeError("IngestDatabase not connected — call connect() first")
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
                    json.dumps(log_item.get("tool_args"), ensure_ascii=True, separators=(",", ":")) if log_item.get("tool_args") is not None else None,
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
            async with self.conn.cursor() as cur:
                for row in rows:
                    await cur.execute(
                        """
                        INSERT INTO inference_logs (
                            id, cursor, vault_address, request_id, execution_key, tool, tool_args_json,
                            strategy_id, status, inference_duration_ms, error, transaction_hash,
                            created_at, completed_at, fetched_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        row,
                    )
            await self.commit()

    async def upsert_swaps(self, swaps: Iterable[dict[str, Any]]) -> None:
        if self.conn is None:
            raise RuntimeError("IngestDatabase not connected — call connect() first")
        now = _now_iso()
        rows = []
        for swap in swaps:
            rows.append(
                (
                    swap.get("transactionHash"),
                    swap.get("blockNumber"),
                    swap.get("logIndex"),
                    swap.get("timestamp"),
                    swap.get("poolId"),
                    swap.get("tokenAddress"),
                    swap.get("tokenName"),
                    swap.get("tokenSymbol"),
                    swap.get("vaultAddress"),
                    _as_bool(swap.get("isReapTwap")),
                    swap.get("side"),
                    swap.get("tokenAmount"),
                    swap.get("ethAmount"),
                    swap.get("ethPriceUsd"),
                    swap.get("effectivePriceEth"),
                    swap.get("effectivePriceUsd"),
                    swap.get("logId"),
                    swap.get("strategyId"),
                    now,
                )
            )
        if rows:
            async with self.conn.cursor() as cur:
                for row in rows:
                    await cur.execute(
                        """
                        INSERT INTO swaps (
                            transaction_hash, block_number, log_index, timestamp,
                            pool_id, token_address, token_name, token_symbol, vault_address,
                            is_reap_twap, side, token_amount, eth_amount, eth_price_usd,
                            effective_price_eth, effective_price_usd, log_id, strategy_id,
                            fetched_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT(transaction_hash, log_index) DO UPDATE SET
                            block_number=excluded.block_number,
                            timestamp=excluded.timestamp,
                            pool_id=excluded.pool_id,
                            token_address=excluded.token_address,
                            token_name=excluded.token_name,
                            token_symbol=excluded.token_symbol,
                            vault_address=excluded.vault_address,
                            is_reap_twap=excluded.is_reap_twap,
                            side=excluded.side,
                            token_amount=excluded.token_amount,
                            eth_amount=excluded.eth_amount,
                            eth_price_usd=excluded.eth_price_usd,
                            effective_price_eth=excluded.effective_price_eth,
                            effective_price_usd=excluded.effective_price_usd,
                            log_id=excluded.log_id,
                            strategy_id=excluded.strategy_id,
                            fetched_at=excluded.fetched_at
                        """,
                        row,
                    )
            await self.commit()

    async def get_last_cursor(self, vault_address: str, resource: str) -> str | None:
        """Get the last saved cursor for a vault+resource (logs, swaps)."""
        if self.conn is None:
            raise RuntimeError("IngestDatabase not connected — call connect() first")
        cursor = await self.conn.execute(
            "SELECT last_cursor FROM ingest_cursors WHERE vault_address = %s AND resource = %s",
            (vault_address, resource),
        )
        row = await cursor.fetchone()
        return row["last_cursor"] if row else None

    async def save_cursor(self, vault_address: str, resource: str, last_cursor: str) -> None:
        """Save the latest cursor for a vault+resource."""
        if self.conn is None:
            raise RuntimeError("IngestDatabase not connected — call connect() first")
        await self.conn.execute(
            """
            INSERT INTO ingest_cursors (vault_address, resource, last_cursor, updated_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT(vault_address, resource) DO UPDATE SET
                last_cursor=excluded.last_cursor,
                updated_at=excluded.updated_at
            """,
            (vault_address, resource, last_cursor, _now_iso()),
        )
        await self.commit()

    async def fetch_existing_full_log_ids(self, log_ids: list[int]) -> set[int]:
        if self.conn is None:
            raise RuntimeError("IngestDatabase not connected — call connect() first")
        if not log_ids:
            return set()
        cursor = await self.conn.execute(
            "SELECT log_id FROM full_logs WHERE log_id = ANY(%s)",
            (log_ids,),
        )
        rows = await cursor.fetchall()
        return {int(row["log_id"]) for row in rows}

    async def upsert_full_log(self, record: FullLogRecord) -> None:
        if self.conn is None:
            raise RuntimeError("IngestDatabase not connected — call connect() first")
        await self.conn.execute(
            """
            INSERT INTO full_logs (
                log_id, vault_address, raw_payload,
                prompt_text, completion_text, reasoning_content, tool_calls_json,
                llm_model, prompt_tokens, completion_tokens, reasoning_tokens, total_tokens,
                parse_error, fetched_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT(log_id) DO UPDATE SET
                vault_address=excluded.vault_address,
                raw_payload=excluded.raw_payload,
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
                Jsonb(record.raw_payload) if record.raw_payload is not None else None,
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
        await self.commit()
