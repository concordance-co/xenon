"""Trade outcome labeling — enriches swaps with forward-looking PnL from candle data."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg

logger = logging.getLogger(__name__)

from pipelines.db import connect_neon, ensure_schema
from pipelines.ingest.api import RetryPolicy, TerminalMarketsApiClient


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class OutcomesConfig:
    base_url: str = "https://api.terminal.markets/api/v1"
    concurrency: int = 5
    timeout_s: int = 30
    retry_max_attempts: int = 6
    limit: int | None = None


_MAX_CANDLE_DRIFT_S = 7200  # 2 candle periods (2h for 1h candles)

HORIZON_SECONDS = {
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def _find_unlabeled_swaps(
    conn: psycopg.Connection, limit: int | None
) -> list[dict[str, Any]]:
    sql = """
        SELECT
            s.log_id,
            s.transaction_hash,
            s.side,
            s.token_address,
            s.timestamp,
            s.effective_price_eth,
            s.effective_price_usd,
            s.eth_price_usd
        FROM swaps s
        LEFT JOIN trade_outcomes t ON t.log_id = s.log_id
        WHERE t.log_id IS NULL
          AND s.log_id IS NOT NULL
        ORDER BY s.timestamp ASC
    """
    if limit is not None:
        sql += " LIMIT %s"
        rows = conn.execute(sql, [int(limit)]).fetchall()
    else:
        rows = conn.execute(sql).fetchall()
    return list(rows)


def _pick_candle_close(
    candles: dict[str, Any],
    target_ts: int,
) -> float | None:
    timestamps = candles.get("t") or []
    closes = candles.get("c") or []
    if not timestamps or not closes or len(timestamps) != len(closes):
        return None

    best_idx = 0
    best_diff = abs(timestamps[0] - target_ts)
    for i in range(1, len(timestamps)):
        diff = abs(timestamps[i] - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    # Only accept if within 2 candle periods (2h for 1h candles)
    if best_diff > _MAX_CANDLE_DRIFT_S:
        return None

    return float(closes[best_idx])


def _compute_pnl(
    entry_price: float,
    exit_price: float | None,
    side: str,
) -> float | None:
    if exit_price is None or entry_price == 0:
        return None
    raw_pnl = (exit_price - entry_price) / entry_price
    # For sells: if price went up after selling, that's a loss (missed gain)
    if side == "sell":
        raw_pnl = -raw_pnl
    return raw_pnl


async def _fetch_candles_for_swap(
    api: TerminalMarketsApiClient,
    token_address: str,
    swap_timestamp: int,
) -> dict[str, Any]:
    return await api.get_candles(
        token_address,
        timeframe="1h",
        from_ts=swap_timestamp,
        to_ts=swap_timestamp + 90000,  # 25 hours
    )


async def _process_swap(
    api: TerminalMarketsApiClient,
    swap: dict[str, Any],
    cached_candles: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    token_address = swap["token_address"]
    swap_ts_raw = swap["timestamp"]
    side = swap["side"]

    if not token_address or swap_ts_raw is None:
        return None
    try:
        swap_ts = int(swap_ts_raw)
    except (TypeError, ValueError):
        return None

    try:
        entry_price_eth = (
            float(swap["effective_price_eth"]) if swap["effective_price_eth"] else None
        )
        entry_price_usd = (
            float(swap["effective_price_usd"]) if swap["effective_price_usd"] else None
        )
        eth_price_usd = (
            float(swap["eth_price_usd"]) if swap["eth_price_usd"] else None
        )
    except (TypeError, ValueError):
        return None

    if entry_price_eth is None or entry_price_eth == 0:
        return None

    if cached_candles is not None:
        candles = cached_candles
    else:
        try:
            candles = await _fetch_candles_for_swap(api, token_address, swap_ts)
        except Exception as exc:
            logger.debug("Failed fetching candles for %s: %s", token_address, exc)
            return None

    if candles.get("s") != "ok":
        return None

    price_1h = _pick_candle_close(candles, swap_ts + HORIZON_SECONDS["1h"])
    price_4h = _pick_candle_close(candles, swap_ts + HORIZON_SECONDS["4h"])
    price_1d = _pick_candle_close(candles, swap_ts + HORIZON_SECONDS["1d"])

    pnl_1h = _compute_pnl(entry_price_eth, price_1h, side)
    pnl_4h = _compute_pnl(entry_price_eth, price_4h, side)
    pnl_1d = _compute_pnl(entry_price_eth, price_1d, side)

    # USD prices: approximate using ETH/USD at entry time
    price_1h_usd = price_1h * eth_price_usd if price_1h and eth_price_usd else None
    price_4h_usd = price_4h * eth_price_usd if price_4h and eth_price_usd else None
    price_1d_usd = price_1d * eth_price_usd if price_1d and eth_price_usd else None

    was_profitable_1h = None
    if pnl_1h is not None:
        was_profitable_1h = True if pnl_1h > 0 else False

    return {
        "log_id": swap["log_id"],
        "transaction_hash": swap["transaction_hash"],
        "side": side,
        "token_address": token_address,
        "entry_price_eth": entry_price_eth,
        "entry_price_usd": entry_price_usd,
        "eth_price_usd_at_entry": eth_price_usd,
        "price_1h_eth": price_1h,
        "price_4h_eth": price_4h,
        "price_1d_eth": price_1d,
        "price_1h_usd": price_1h_usd,
        "price_4h_usd": price_4h_usd,
        "price_1d_usd": price_1d_usd,
        "pnl_1h_pct": pnl_1h,
        "pnl_4h_pct": pnl_4h,
        "pnl_1d_pct": pnl_1d,
        "was_profitable_1h": was_profitable_1h,
        "candle_data_json": json.dumps(candles, ensure_ascii=True, separators=(",", ":")),
    }


async def run_outcomes(config: OutcomesConfig) -> dict[str, int]:
    conn = connect_neon(autocommit=False)
    ensure_schema(conn)

    unlabeled = _find_unlabeled_swaps(conn, config.limit)
    print(f"Found {len(unlabeled)} unlabeled swaps")

    if not unlabeled:
        conn.close()
        return {"processed": 0, "labeled": 0, "failed": 0}

    labeled = 0
    failed = 0
    failure_reasons: Counter[str] = Counter()
    started_at = time.monotonic()
    retry_policy = RetryPolicy(max_attempts=config.retry_max_attempts)

    print(
        "Starting outcomes run "
        f"(concurrency={config.concurrency}, timeout_s={config.timeout_s}, "
        f"retry_max_attempts={config.retry_max_attempts})"
    )

    async with TerminalMarketsApiClient(
        base_url=config.base_url,
        concurrency=config.concurrency,
        timeout_s=config.timeout_s,
        retry_policy=retry_policy,
    ) as api:
        # Phase 1: Fetch candle data per token (batch by token_address)
        token_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for swap in unlabeled:
            if swap["token_address"]:
                token_groups[swap["token_address"]].append(swap)

        candle_cache: dict[str, dict[str, Any]] = {}
        print(f"Fetching candles for {len(token_groups)} unique tokens...")
        for token_addr, swaps_for_token in token_groups.items():
            timestamps = [int(s["timestamp"]) for s in swaps_for_token if s["timestamp"]]
            if not timestamps:
                continue
            min_ts = min(timestamps)
            max_ts = max(timestamps)
            try:
                candles = await api.get_candles(
                    token_addr, timeframe="1h", from_ts=min_ts, to_ts=max_ts + 90000
                )
                if candles.get("s") == "ok":
                    candle_cache[token_addr] = candles
            except Exception as exc:
                print(f"  Failed fetching candles for {token_addr}: {exc}")

        print(f"Cached candles for {len(candle_cache)} tokens")

        # Phase 2: Process swaps using cached candles
        batch_size = config.concurrency
        for i in range(0, len(unlabeled), batch_size):
            batch = unlabeled[i : i + batch_size]
            results = await asyncio.gather(
                *[
                    _process_swap(
                        api, swap, cached_candles=candle_cache.get(swap["token_address"])
                    )
                    for swap in batch
                ],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                    key = type(result).__name__
                    failure_reasons[key] += 1
                    if failure_reasons[key] <= 3:
                        print(f"  Failed ({key}): {result}")
                elif result is not None:
                    try:
                        conn.execute(
                            """
                            INSERT INTO trade_outcomes (
                                log_id, transaction_hash, side, token_address,
                                entry_price_eth, entry_price_usd, eth_price_usd_at_entry,
                                price_1h_eth, price_4h_eth, price_1d_eth,
                                price_1h_usd, price_4h_usd, price_1d_usd,
                                pnl_1h_pct, pnl_4h_pct, pnl_1d_pct,
                                was_profitable_1h, candle_data_json, computed_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT(log_id) DO UPDATE SET
                                price_1h_eth=excluded.price_1h_eth,
                                price_4h_eth=excluded.price_4h_eth,
                                price_1d_eth=excluded.price_1d_eth,
                                price_1h_usd=excluded.price_1h_usd,
                                price_4h_usd=excluded.price_4h_usd,
                                price_1d_usd=excluded.price_1d_usd,
                                pnl_1h_pct=excluded.pnl_1h_pct,
                                pnl_4h_pct=excluded.pnl_4h_pct,
                                pnl_1d_pct=excluded.pnl_1d_pct,
                                was_profitable_1h=excluded.was_profitable_1h,
                                candle_data_json=excluded.candle_data_json,
                                computed_at=excluded.computed_at
                            """,
                            (
                                result["log_id"],
                                result["transaction_hash"],
                                result["side"],
                                result["token_address"],
                                result["entry_price_eth"],
                                result["entry_price_usd"],
                                result["eth_price_usd_at_entry"],
                                result["price_1h_eth"],
                                result["price_4h_eth"],
                                result["price_1d_eth"],
                                result["price_1h_usd"],
                                result["price_4h_usd"],
                                result["price_1d_usd"],
                                result["pnl_1h_pct"],
                                result["pnl_4h_pct"],
                                result["pnl_1d_pct"],
                                result["was_profitable_1h"],
                                result["candle_data_json"],
                                _now_iso(),
                            ),
                        )
                        labeled += 1
                    except Exception as exc:
                        failed += 1
                        failure_reasons["db_write_error"] += 1
                        if failure_reasons["db_write_error"] <= 3:
                            print(f"  Failed (db_write_error): log_id={result.get('log_id')} err={exc}")
                else:
                    failed += 1
                    failure_reasons["not_labelable"] += 1
            conn.commit()

            processed = min(i + batch_size, len(unlabeled))
            if processed % max(100, batch_size * 20) == 0 or processed == len(unlabeled):
                elapsed_s = max(1e-6, time.monotonic() - started_at)
                rate = processed / elapsed_s
                print(
                    f"Progress: {processed}/{len(unlabeled)} "
                    f"({processed/len(unlabeled)*100:.1f}%) "
                    f"labeled={labeled} failed={failed} rate={rate:.1f}/s"
                )

    conn.close()
    if failure_reasons:
        top = sorted(failure_reasons.items(), key=lambda kv: kv[1], reverse=True)[:8]
        print(f"Failure breakdown: {top}")
    return {"processed": len(unlabeled), "labeled": labeled, "failed": failed}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute trade outcomes (forward PnL) from swap data and candles"
    )
    parser.add_argument("--base-url", default="https://api.terminal.markets/api/v1")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout-s", type=int, default=30)
    parser.add_argument("--retry-max-attempts", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = OutcomesConfig(
        base_url=args.base_url,
        concurrency=args.concurrency,
        timeout_s=args.timeout_s,
        retry_max_attempts=args.retry_max_attempts,
        limit=args.limit,
    )
    stats = asyncio.run(run_outcomes(config))
    print("Outcome labeling complete")
    for key, value in stats.items():
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
