"""Trade outcome labeling — enriches swaps with forward-looking PnL from candle data."""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pipelines.ingest.api import RetryPolicy, TerminalMarketsApiClient


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class OutcomesConfig:
    db_path: Path = Path("data/terminal_ingest.db")
    base_url: str = "https://api.terminal.markets/api/v1"
    concurrency: int = 5
    timeout_s: int = 30
    retry_max_attempts: int = 6
    limit: int | None = None


HORIZON_SECONDS = {
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def _init_outcomes_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS trade_outcomes (
            log_id INTEGER PRIMARY KEY,
            transaction_hash TEXT NOT NULL,
            side TEXT NOT NULL,
            token_address TEXT NOT NULL,
            entry_price_eth REAL,
            entry_price_usd REAL,
            eth_price_usd_at_entry REAL,
            price_1h_eth REAL,
            price_4h_eth REAL,
            price_1d_eth REAL,
            price_1h_usd REAL,
            price_4h_usd REAL,
            price_1d_usd REAL,
            pnl_1h_pct REAL,
            pnl_4h_pct REAL,
            pnl_1d_pct REAL,
            was_profitable_1h INTEGER,
            candle_data_json TEXT,
            computed_at TEXT NOT NULL
        );
        """
    )


def _find_unlabeled_swaps(
    conn: sqlite3.Connection, limit: int | None
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
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


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
    if best_diff > 7200:
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
) -> dict[str, Any] | None:
    token_address = swap["token_address"]
    swap_ts = swap["timestamp"]
    side = swap["side"]

    if not token_address or not swap_ts:
        return None

    entry_price_eth = (
        float(swap["effective_price_eth"]) if swap["effective_price_eth"] else None
    )
    entry_price_usd = (
        float(swap["effective_price_usd"]) if swap["effective_price_usd"] else None
    )
    eth_price_usd = (
        float(swap["eth_price_usd"]) if swap["eth_price_usd"] else None
    )

    if entry_price_eth is None or entry_price_eth == 0:
        return None

    try:
        candles = await _fetch_candles_for_swap(api, token_address, swap_ts)
    except Exception:
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
        was_profitable_1h = 1 if pnl_1h > 0 else 0

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
    conn = sqlite3.connect(config.db_path)
    conn.row_factory = sqlite3.Row
    _init_outcomes_schema(conn)

    unlabeled = _find_unlabeled_swaps(conn, config.limit)
    print(f"Found {len(unlabeled)} unlabeled swaps")

    if not unlabeled:
        conn.close()
        return {"processed": 0, "labeled": 0, "failed": 0}

    labeled = 0
    failed = 0
    retry_policy = RetryPolicy(max_attempts=config.retry_max_attempts)

    async with TerminalMarketsApiClient(
        base_url=config.base_url,
        concurrency=config.concurrency,
        timeout_s=config.timeout_s,
        retry_policy=retry_policy,
    ) as api:
        batch_size = config.concurrency
        for i in range(0, len(unlabeled), batch_size):
            batch = unlabeled[i : i + batch_size]
            results = await asyncio.gather(
                *[_process_swap(api, swap) for swap in batch],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    failed += 1
                    print(f"  Failed: {result}")
                elif result is not None:
                    conn.execute(
                        """
                        INSERT INTO trade_outcomes (
                            log_id, transaction_hash, side, token_address,
                            entry_price_eth, entry_price_usd, eth_price_usd_at_entry,
                            price_1h_eth, price_4h_eth, price_1d_eth,
                            price_1h_usd, price_4h_usd, price_1d_usd,
                            pnl_1h_pct, pnl_4h_pct, pnl_1d_pct,
                            was_profitable_1h, candle_data_json, computed_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                else:
                    failed += 1
            conn.commit()

    conn.close()
    return {"processed": len(unlabeled), "labeled": labeled, "failed": failed}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute trade outcomes (forward PnL) from swap data and candles"
    )
    parser.add_argument("--db-path", type=Path, default=Path("data/terminal_ingest.db"))
    parser.add_argument("--base-url", default="https://api.terminal.markets/api/v1")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout-s", type=int, default=30)
    parser.add_argument("--retry-max-attempts", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = OutcomesConfig(
        db_path=args.db_path,
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
