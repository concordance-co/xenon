from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any, Iterable

from pipelines.ingest.api import RetryPolicy, TerminalApiDeferrable, TerminalApiError, TerminalMarketsApiClient
from pipelines.ingest.db import FullLogRecord, IngestDatabase
from pipelines.ingest.full_log_parser import parse_full_log


def _batched(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


@dataclass(slots=True)
class BackfillConfig:
    base_url: str = "https://api.terminal.markets/api/v1"
    top_n: int = 3
    leaderboard_sort_by: str = "total_pnl_usd"
    request_limit: int = 50
    request_concurrency: int = 3
    requests_per_second: float = 6.0
    timeout_s: int = 30
    max_logs_per_vault: int | None = None
    max_full_logs_per_vault: int | None = None
    max_swaps_per_vault: int | None = None
    include_reasoning: bool = True
    retry_max_attempts: int = 6
    selection: str = "top"  # "top" or "random"
    random_seed: int | None = None
    retry_deferred: bool = True


@dataclass(slots=True)
class BackfillSummary:
    vaults_discovered: int = 0
    vaults_ingested: int = 0
    strategies_ingested: int = 0
    logs_ingested: int = 0
    full_logs_ingested: int = 0
    full_log_failures: int = 0
    swaps_ingested: int = 0


class TerminalBackfillIngestor:
    def __init__(self, config: BackfillConfig) -> None:
        self.config = config
        self.db = IngestDatabase()
        self.summary = BackfillSummary()

    async def run(self) -> BackfillSummary:
        await self.db.connect()
        await self.db.init_schema()

        retry_policy = RetryPolicy(max_attempts=self.config.retry_max_attempts)

        try:
            async with TerminalMarketsApiClient(
                base_url=self.config.base_url,
                concurrency=self.config.request_concurrency,
                requests_per_second=self.config.requests_per_second,
                timeout_s=self.config.timeout_s,
                retry_policy=retry_policy,
            ) as api:
                if self.config.selection == "random":
                    leaderboard_items = await self._discover_random_vaults(api)
                else:
                    leaderboard_items = await self._discover_top_vaults(api)
                self.summary.vaults_discovered = len(leaderboard_items)
                print(f"Discovered {len(leaderboard_items)} vaults ({self.config.selection} selection)")

                # --- Phase 1: vault config + strategy fetches (parallel) ---
                valid_items = [
                    (idx, item) for idx, item in enumerate(leaderboard_items, start=1)
                    if item.get("vaultAddress")
                ]

                existing_vaults = await self.db.get_existing_vault_addresses()
                new_items = [(idx, item) for idx, item in valid_items if item["vaultAddress"] not in existing_vaults]
                if len(new_items) < len(valid_items):
                    print(f"Skipping {len(valid_items) - len(new_items)} vaults already in DB, fetching {len(new_items)} new")

                async def _ingest_vault_meta(index: int, item: dict[str, Any]) -> None:
                    vault_address = item["vaultAddress"]
                    print(f"[{index}/{len(leaderboard_items)}] Fetching vault + strategies: {vault_address}")
                    vault_config, strategies = await asyncio.gather(
                        api.get_vault(vault_address),
                        api.get_strategies(vault_address),
                    )
                    # DB writes are serialised through the batch context
                    async with self.db.batch():
                        await self.db.upsert_vault(item, vault_config)
                        await self.db.upsert_strategies(vault_address, strategies)
                    self.summary.vaults_ingested += 1
                    self.summary.strategies_ingested += len(strategies)

                await asyncio.gather(
                    *[_ingest_vault_meta(idx, item) for idx, item in new_items]
                )

                # --- Phase 2: log backfill ---
                async with self.db.batch():
                    deferred_ids = await self.db.get_deferred_log_ids()
                if deferred_ids:
                    print(f"Skipping {len(deferred_ids)} previously deferred logs")

                # Process vaults in bounded groups so each vault completes
                # all pages before we move on. This keeps cursors advancing
                # and gives visible progress. A small number of vaults run
                # concurrently so API calls overlap while DB writes stay fast.
                vault_concurrency = min(self.config.request_concurrency, len(valid_items))
                vault_sem = asyncio.Semaphore(vault_concurrency)
                log_progress = {"done": 0}

                async def _backfill_logs_task(index: int, item: dict[str, Any]) -> None:
                    async with vault_sem:
                        vault_address = item["vaultAddress"]
                        await self._backfill_vault_logs(api, vault_address, deferred_ids)
                        log_progress["done"] += 1
                        if log_progress["done"] % 25 == 0 or log_progress["done"] == len(valid_items):
                            print(f"  Log backfill progress: {log_progress['done']}/{len(valid_items)} vaults complete")

                await asyncio.gather(
                    *[_backfill_logs_task(idx, item) for idx, item in valid_items]
                )

                # --- Phase 3: swap backfill ---
                swap_progress = {"done": 0}

                async def _backfill_swaps_task(index: int, item: dict[str, Any]) -> None:
                    async with vault_sem:
                        vault_address = item["vaultAddress"]
                        await self._backfill_vault_swaps(api, vault_address)
                        swap_progress["done"] += 1
                        if swap_progress["done"] % 25 == 0 or swap_progress["done"] == len(valid_items):
                            print(f"  Swap backfill progress: {swap_progress['done']}/{len(valid_items)} vaults complete")

                await asyncio.gather(
                    *[_backfill_swaps_task(idx, item) for idx, item in valid_items]
                )

                # --- Phase 4: retry deferred logs (one attempt each) ---
                if not self.config.retry_deferred:
                    print("Skipping deferred log retry (disabled)")
                deferred_ids_final = await self.db.get_deferred_log_ids() if self.config.retry_deferred else set()
                if deferred_ids_final:
                    print(f"\nRetrying {len(deferred_ids_final)} deferred logs...")
                    recovered = 0
                    for batch in _batched(list(deferred_ids_final), self.config.request_concurrency):
                        items_to_retry = [{"id": lid, "vault_address": None} for lid in batch]
                        fetched = await asyncio.gather(
                            *[self._fetch_full_log(api, item) for item in items_to_retry],
                            return_exceptions=True,
                        )
                        async with self.db.batch():
                            for lid, result in zip(batch, fetched):
                                if isinstance(result, Exception):
                                    pass  # stays deferred
                                elif result is not None:
                                    await self.db.upsert_full_log(result)
                                    await self.db.remove_deferred_log(lid)
                                    recovered += 1
                                    self.summary.full_logs_ingested += 1
                    print(f"  Recovered {recovered}/{len(deferred_ids_final)} deferred logs")
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

    async def _discover_random_vaults(self, api: TerminalMarketsApiClient) -> list[dict[str, Any]]:
        """Page through entire leaderboard, then randomly sample top_n vaults."""
        all_vaults: list[dict[str, Any]] = []
        cursor: str | None = None

        print("Fetching all vaults from leaderboard for random sampling...")
        while True:
            page = await api.get_leaderboard_page(
                limit=self.config.request_limit,
                sort_by=self.config.leaderboard_sort_by,
                cursor=cursor,
            )
            items = page.get("items") or []
            if not items:
                break
            all_vaults.extend(items)
            print(f"  ...fetched {len(all_vaults)} vaults so far")

            if not page.get("hasMoreItems"):
                break
            cursor = items[-1].get("cursor")
            if not cursor:
                break

        if len(all_vaults) <= self.config.top_n:
            print(f"  Only {len(all_vaults)} vaults exist, using all of them")
            return all_vaults

        rng = random.Random(self.config.random_seed)
        sampled = rng.sample(all_vaults, self.config.top_n)
        print(f"  Randomly sampled {len(sampled)} from {len(all_vaults)} total vaults")
        return sampled

    async def _backfill_vault_logs(
        self, api: TerminalMarketsApiClient, vault_address: str, deferred_ids: set[int] | None = None,
    ) -> None:
        async with self.db.batch():
            cursor = await self.db.get_last_cursor(vault_address, "logs")
        if cursor:
            print(f"  Resuming logs from cursor {cursor[:20]}...")
        logs_seen = 0
        full_logs_seen = 0
        _deferred = deferred_ids or set()

        short_addr = vault_address[:8]
        page_num = 0

        while True:
            if self.config.max_logs_per_vault is not None and logs_seen >= self.config.max_logs_per_vault:
                break
            if self.config.max_full_logs_per_vault is not None and full_logs_seen >= self.config.max_full_logs_per_vault:
                break

            page_num += 1
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

            next_cursor = items[-1].get("cursor") if items else None
            hit_full_log_limit = False

            # Step 1: upsert inference logs (brief lock)
            async with self.db.batch():
                await self.db.upsert_inference_logs(items)
                logs_seen += len(items)
                self.summary.logs_ingested += len(items)

            # Step 1b: check which full logs we already have (separate txn so we
            # see commits from other coroutines)
            page_ids = [int(item["id"]) for item in items if item.get("id") is not None]
            async with self.db.batch():
                existing_ids = await self.db.fetch_existing_full_log_ids(page_ids)

            missing_items = [
                item for item in items
                if item.get("id") is not None
                and int(item["id"]) not in existing_ids
                and int(item["id"]) not in _deferred
            ]

            if self.config.max_full_logs_per_vault is not None:
                remaining_full = self.config.max_full_logs_per_vault - full_logs_seen
                if remaining_full <= 0:
                    hit_full_log_limit = True
                else:
                    missing_items = missing_items[:remaining_full]

            if missing_items:
                print(f"  [{short_addr}] page {page_num}: {len(items)} logs, {len(missing_items)} new full-logs to fetch")
            elif page_num == 1:
                print(f"  [{short_addr}] page {page_num}: {len(items)} logs, all full-logs already in DB")

            # Step 2: fetch full logs from API (NO lock held — parallel with other vaults)
            if not hit_full_log_limit and missing_items:
                for batch in _batched(missing_items, self.config.request_concurrency):
                    # Fetch from API (no DB lock)
                    fetched = await asyncio.gather(
                        *[self._fetch_full_log(api, item) for item in batch],
                        return_exceptions=True,
                    )
                    # Step 3: write results to DB (brief lock)
                    async with self.db.batch():
                        for fetch_result, item in zip(fetched, batch):
                            if isinstance(fetch_result, TerminalApiDeferrable):
                                log_id = int(item["id"])
                                await self.db.defer_log(log_id, vault_address, fetch_result.status)
                                _deferred.add(log_id)
                                self.summary.full_log_failures += 1
                                print(f"  [{short_addr}] deferred log {log_id} (status={fetch_result.status})")
                            elif isinstance(fetch_result, Exception):
                                self.summary.full_log_failures += 1
                                print(f"  [{short_addr}] failed full-log fetch: {fetch_result}")
                            elif fetch_result is not None:
                                await self.db.upsert_full_log(fetch_result)
                                self.summary.full_logs_ingested += 1
                                full_logs_seen += 1

            # Step 4: save cursor (brief lock)
            if next_cursor:
                async with self.db.batch():
                    await self.db.save_cursor(vault_address, "logs", next_cursor)

            if hit_full_log_limit or not page.get("hasMoreItems"):
                break
            if not next_cursor:
                break
            cursor = next_cursor

        if page_num > 1:
            print(f"  [{short_addr}] done: {page_num} pages, {logs_seen} logs, {full_logs_seen} new full-logs")

    async def _backfill_vault_swaps(self, api: TerminalMarketsApiClient, vault_address: str) -> None:
        async with self.db.batch():
            cursor = await self.db.get_last_cursor(vault_address, "swaps")
        if cursor:
            print(f"  Resuming swaps from cursor {cursor[:20]}...")
        swaps_seen = 0

        while True:
            if self.config.max_swaps_per_vault is not None and swaps_seen >= self.config.max_swaps_per_vault:
                break

            page = await api.get_swaps_page(
                vault_address,
                limit=self.config.request_limit,
                order="asc",
                cursor=cursor,
            )
            items = page.get("items") or []
            if not items:
                break

            if self.config.max_swaps_per_vault is not None:
                remaining = self.config.max_swaps_per_vault - swaps_seen
                if remaining <= 0:
                    break
                items = items[:remaining]

            next_cursor = items[-1].get("cursor") if items else None

            async with self.db.batch():
                await self.db.upsert_swaps(items)
                swaps_seen += len(items)
                self.summary.swaps_ingested += len(items)

                # Cursor save inside the batch — committed with the batch
                if next_cursor:
                    await self.db.save_cursor(vault_address, "swaps", next_cursor)

            if not page.get("hasMoreItems"):
                break
            if not next_cursor:
                break
            cursor = next_cursor

    async def _fetch_full_log(
        self, api: TerminalMarketsApiClient, log_item: dict[str, Any]
    ) -> FullLogRecord | None:
        """Fetch a full log from the API. Returns the record (no DB write)."""
        log_id_raw = log_item.get("id")
        if log_id_raw is None:
            return None
        log_id = int(log_id_raw)
        payload = await api.get_full_log(log_id)
        parsed = parse_full_log(payload, include_reasoning=self.config.include_reasoning)
        return FullLogRecord(
            log_id=log_id,
            vault_address=log_item.get("vault_address"),
            parsed=parsed,
            raw_payload=payload,
        )


async def run_backfill(config: BackfillConfig) -> BackfillSummary:
    ingestor = TerminalBackfillIngestor(config)
    try:
        return await ingestor.run()
    except TerminalApiError:
        raise
