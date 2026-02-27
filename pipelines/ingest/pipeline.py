from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from pipelines.ingest.api import RetryPolicy, TerminalApiError, TerminalMarketsApiClient
from pipelines.ingest.db import FullLogRecord, IngestDatabase
from pipelines.ingest.full_log_parser import parse_full_log
from pipelines.ingest.payload_store import RawPayloadStore


def _batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


@dataclass(slots=True)
class BackfillConfig:
    base_url: str = "https://api.terminal.markets/api/v1"
    db_path: Path = Path("data/terminal_ingest.db")
    raw_payload_dir: Path = Path("data/full_logs")
    top_n: int = 3
    leaderboard_sort_by: str = "total_pnl_usd"
    request_limit: int = 50
    request_concurrency: int = 10
    timeout_s: int = 30
    max_logs_per_vault: int | None = None
    max_full_logs_per_vault: int | None = None
    include_reasoning: bool = True
    retry_max_attempts: int = 6


@dataclass(slots=True)
class BackfillSummary:
    vaults_discovered: int = 0
    vaults_ingested: int = 0
    strategies_ingested: int = 0
    logs_ingested: int = 0
    full_logs_ingested: int = 0
    full_log_failures: int = 0


class TerminalBackfillIngestor:
    def __init__(self, config: BackfillConfig) -> None:
        self.config = config
        self.db = IngestDatabase(config.db_path)
        self.payload_store = RawPayloadStore(config.raw_payload_dir)
        self.summary = BackfillSummary()

    async def run(self) -> BackfillSummary:
        await self.db.connect()
        await self.db.init_schema()

        retry_policy = RetryPolicy(max_attempts=self.config.retry_max_attempts)

        try:
            async with TerminalMarketsApiClient(
                base_url=self.config.base_url,
                concurrency=self.config.request_concurrency,
                timeout_s=self.config.timeout_s,
                retry_policy=retry_policy,
            ) as api:
                leaderboard_items = await self._discover_top_vaults(api)
                self.summary.vaults_discovered = len(leaderboard_items)
                print(f"Discovered {len(leaderboard_items)} vaults from leaderboard")

                for index, item in enumerate(leaderboard_items, start=1):
                    vault_address = item.get("vaultAddress")
                    if not vault_address:
                        continue
                    print(f"[{index}/{len(leaderboard_items)}] Fetching vault + strategies: {vault_address}")
                    vault_config = await api.get_vault(vault_address)
                    strategies = await api.get_strategies(vault_address)
                    await self.db.upsert_vault(item, vault_config)
                    await self.db.upsert_strategies(vault_address, strategies)
                    self.summary.vaults_ingested += 1
                    self.summary.strategies_ingested += len(strategies)

                for index, item in enumerate(leaderboard_items, start=1):
                    vault_address = item.get("vaultAddress")
                    if not vault_address:
                        continue
                    print(f"[{index}/{len(leaderboard_items)}] Backfilling logs: {vault_address}")
                    await self._backfill_vault_logs(api, vault_address)
        finally:
            await self.db.close()

        return self.summary

    async def _discover_top_vaults(self, api: TerminalMarketsApiClient) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        cursor: str | None = None

        while len(collected) < self.config.top_n:
            page = await api.get_leaderboard_page(
                limit=self.config.request_limit,
                sort_by=self.config.leaderboard_sort_by,
                cursor=cursor,
            )
            items = page.get("items") or []
            if not items:
                break

            needed = self.config.top_n - len(collected)
            collected.extend(items[:needed])

            if len(collected) >= self.config.top_n or not page.get("hasMoreItems"):
                break

            cursor = items[-1].get("cursor")
            if not cursor:
                break

        return collected

    async def _backfill_vault_logs(self, api: TerminalMarketsApiClient, vault_address: str) -> None:
        cursor: str | None = None
        logs_seen = 0
        full_logs_seen = 0

        while True:
            if self.config.max_logs_per_vault is not None and logs_seen >= self.config.max_logs_per_vault:
                break
            if self.config.max_full_logs_per_vault is not None and full_logs_seen >= self.config.max_full_logs_per_vault:
                break

            page = await api.get_logs_page(
                vault_address,
                limit=self.config.request_limit,
                order="asc",
                cursor=cursor,
            )
            items = page.get("items") or []
            if not items:
                break

            if self.config.max_logs_per_vault is not None:
                remaining_logs = self.config.max_logs_per_vault - logs_seen
                if remaining_logs <= 0:
                    break
                items = items[:remaining_logs]

            await self.db.upsert_inference_logs(items)
            logs_seen += len(items)
            self.summary.logs_ingested += len(items)

            page_ids = [int(item["id"]) for item in items if item.get("id") is not None]
            existing_ids = await self.db.fetch_existing_full_log_ids(page_ids)
            missing_items = [item for item in items if int(item["id"]) not in existing_ids]

            if self.config.max_full_logs_per_vault is not None:
                remaining_full = self.config.max_full_logs_per_vault - full_logs_seen
                if remaining_full <= 0:
                    break
                missing_items = missing_items[:remaining_full]

            if missing_items:
                for batch in _batched(missing_items, self.config.request_concurrency):
                    results = await asyncio.gather(
                        *[self._fetch_and_store_full_log(api, item) for item in batch],
                        return_exceptions=True,
                    )
                    for result in results:
                        if isinstance(result, Exception):
                            self.summary.full_log_failures += 1
                            print(f"Failed full-log fetch: {result}")
                        elif result:
                            self.summary.full_logs_ingested += 1
                            full_logs_seen += 1

            if not page.get("hasMoreItems"):
                break

            next_cursor = items[-1].get("cursor")
            if not next_cursor:
                break
            cursor = next_cursor

    async def _fetch_and_store_full_log(
        self, api: TerminalMarketsApiClient, log_item: dict[str, Any]
    ) -> bool:
        log_id_raw = log_item.get("id")
        if log_id_raw is None:
            return False
        log_id = int(log_id_raw)
        payload = await api.get_full_log(log_id)
        payload_meta = self.payload_store.write(log_id, payload)
        parsed = parse_full_log(payload, include_reasoning=self.config.include_reasoning)
        record = FullLogRecord(
            log_id=log_id,
            vault_address=log_item.get("vault_address"),
            payload_meta=payload_meta,
            parsed=parsed,
        )
        await self.db.upsert_full_log(record)
        return True


async def run_backfill(config: BackfillConfig) -> BackfillSummary:
    ingestor = TerminalBackfillIngestor(config)
    try:
        return await ingestor.run()
    except TerminalApiError:
        raise
