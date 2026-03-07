from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Sequence

from pipelines.ingest.pipeline import BackfillConfig, run_backfill


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one-shot Terminal Markets phase1+phase2 backfill"
    )
    parser.add_argument("--base-url", default="https://api.terminal.markets/api/v1")
    parser.add_argument("--db-path", type=Path, default=Path("data/terminal_ingest.db"))
    parser.add_argument("--raw-payload-dir", type=Path, default=Path("data/full_logs"))
    parser.add_argument("--top-n", type=int, default=3, help="Number of vaults to ingest")
    parser.add_argument("--selection", choices=["top", "random"], default="top",
                        help="'top' = top N by sort order, 'random' = random sample from all vaults")
    parser.add_argument("--random-seed", type=int, default=None, help="Seed for random vault selection")
    parser.add_argument("--leaderboard-sort-by", default="total_pnl_usd")
    parser.add_argument("--request-limit", type=int, default=50)
    parser.add_argument("--request-concurrency", type=int, default=10)
    parser.add_argument("--timeout-s", type=int, default=30)
    parser.add_argument("--retry-max-attempts", type=int, default=6)
    parser.add_argument("--max-logs-per-vault", type=int, default=None)
    parser.add_argument("--max-full-logs-per-vault", type=int, default=None)
    parser.add_argument("--max-swaps-per-vault", type=int, default=None)
    parser.add_argument(
        "--exclude-reasoning",
        action="store_true",
        help="Do not persist reasoning_content parsed from llm completion payload",
    )
    return parser


async def _run_from_args(args: argparse.Namespace) -> int:
    config = BackfillConfig(
        base_url=args.base_url,
        db_path=args.db_path,
        raw_payload_dir=args.raw_payload_dir,
        top_n=args.top_n,
        selection=args.selection,
        random_seed=args.random_seed,
        leaderboard_sort_by=args.leaderboard_sort_by,
        request_limit=args.request_limit,
        request_concurrency=args.request_concurrency,
        timeout_s=args.timeout_s,
        max_logs_per_vault=args.max_logs_per_vault,
        max_full_logs_per_vault=args.max_full_logs_per_vault,
        max_swaps_per_vault=args.max_swaps_per_vault,
        include_reasoning=not args.exclude_reasoning,
        retry_max_attempts=args.retry_max_attempts,
    )
    summary = await run_backfill(config)
    print("Backfill complete")
    print(f"vaults_discovered={summary.vaults_discovered}")
    print(f"vaults_ingested={summary.vaults_ingested}")
    print(f"strategies_ingested={summary.strategies_ingested}")
    print(f"logs_ingested={summary.logs_ingested}")
    print(f"full_logs_ingested={summary.full_logs_ingested}")
    print(f"full_log_failures={summary.full_log_failures}")
    print(f"swaps_ingested={summary.swaps_ingested}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_from_args(args))


if __name__ == "__main__":
    raise SystemExit(main())
