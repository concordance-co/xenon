"""Modal wrapper for ingest-side jobs.

This module runs ingest, prep, and outcome-labeling jobs on Modal while writing
all durable state to Neon. It is the ingest-side execution-plane entrypoint and
does not depend on the capture/analysis runtime in ``pipelines.interp``.
"""

import modal

app = modal.App("xenon-ingest")

neon_secret = modal.Secret.from_name("xenon-neon")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "aiohttp", "psycopg[binary]",
    )
    .add_local_python_source("pipelines")
)


@app.function(
    image=image,
    timeout=14400,
    cpu=2,
    secrets=[neon_secret],
)
def run_ingest(
    top_n: int = 3,
    selection: str = "top",
    random_seed: int = -1,
    concurrency: int = 5,
    requests_per_second: float = 6.0,
    leaderboard_sort_by: str = "total_pnl_usd",
    exclude_reasoning: bool = False,
    retry_deferred: bool = True,
    max_logs_per_vault: int = -1,
    max_full_logs_per_vault: int = -1,
    max_swaps_per_vault: int = -1,
) -> dict:
    """Run Terminal Markets ingest on Modal, writing to Neon Postgres."""
    import asyncio

    from pipelines.ingest.pipeline import BackfillConfig, run_backfill

    config = BackfillConfig(
        top_n=top_n,
        selection=selection,
        random_seed=random_seed if random_seed >= 0 else None,
        leaderboard_sort_by=leaderboard_sort_by,
        request_concurrency=concurrency,
        requests_per_second=requests_per_second,
        include_reasoning=not exclude_reasoning,
        retry_deferred=retry_deferred,
        max_logs_per_vault=max_logs_per_vault if max_logs_per_vault >= 0 else None,
        max_full_logs_per_vault=max_full_logs_per_vault if max_full_logs_per_vault >= 0 else None,
        max_swaps_per_vault=max_swaps_per_vault if max_swaps_per_vault >= 0 else None,
    )

    summary = asyncio.run(run_backfill(config))

    # Validate ingest results
    if summary.vaults_ingested == 0:
        print("WARNING: No vaults were ingested")
    if summary.full_logs_ingested == 0 and summary.logs_ingested > 0:
        print("WARNING: Logs ingested but no full logs — check API access")

    # Clean up cursors for vaults that no longer exist
    from pipelines.db import cleanup_stale_cursors, connect_neon
    conn = connect_neon()
    cleanup_stale_cursors(conn)
    _refresh_payload_stats(conn)
    conn.close()

    result = {
        "vaults_discovered": summary.vaults_discovered,
        "vaults_ingested": summary.vaults_ingested,
        "strategies_ingested": summary.strategies_ingested,
        "logs_ingested": summary.logs_ingested,
        "full_logs_ingested": summary.full_logs_ingested,
        "full_log_failures": summary.full_log_failures,
        "swaps_ingested": summary.swaps_ingested,
    }
    print(f"\nIngest complete: {result}")
    return result


@app.function(
    image=image,
    timeout=7200,
    cpu=2,
    secrets=[neon_secret],
)
def run_prep(
    limit: int = -1,
    include_all_decisions: bool = False,
    full_rebuild: bool = False,
    incremental_rebuild: bool = False,
) -> dict:
    """Run data prep on Modal, reading/writing Neon Postgres."""
    from pipelines.ingest.prepare import PrepareConfig, run_prepare

    config = PrepareConfig(
        limit=limit if limit >= 0 else 50_000,
        only_focus_decisions=not include_all_decisions,
        full_rebuild=full_rebuild,
        incremental_rebuild=incremental_rebuild,
    )

    stats = run_prepare(config)

    # Validate prep results
    if stats.get("rows_written", 0) == 0:
        print("WARNING: Prep produced 0 rows — check if full_logs have raw_payload data")
    if stats.get("row_errors", 0) > stats.get("rows_written", 0) * 0.1:
        print(f"WARNING: High error rate in prep: {stats['row_errors']} errors vs {stats['rows_written']} written")

    print(f"\nPrep complete: {stats}")
    return stats


@app.function(
    image=image,
    timeout=7200,
    cpu=2,
    secrets=[neon_secret],
)
def run_outcomes(
    concurrency: int = 5,
    timeout_s: int = 30,
    retry_max_attempts: int = 6,
    limit: int = -1,
) -> dict:
    """Compute forward-looking PnL for swaps using Terminal Markets candle data."""
    import asyncio

    from pipelines.ingest.outcomes import OutcomesConfig
    from pipelines.ingest.outcomes import run_outcomes as _run_outcomes

    config = OutcomesConfig(
        concurrency=concurrency,
        timeout_s=timeout_s,
        retry_max_attempts=retry_max_attempts,
        limit=limit if limit >= 0 else None,
    )

    stats = asyncio.run(_run_outcomes(config))

    # Validate outcomes results
    if stats.get("labeled", 0) == 0 and stats.get("processed", 0) > 0:
        print("WARNING: Processed swaps but labeled 0 — check candle API access")

    print(f"\nOutcomes complete: {stats}")
    return stats


@app.function(
    image=image,
    timeout=300,
    cpu=1,
    secrets=[neon_secret],
)
def reset_cursors() -> int:
    """Wipe all ingest_cursors so next run re-paginates from start."""
    from pipelines.db import connect_neon, reset_cursors as _reset

    conn = connect_neon()
    count = _reset(conn)
    conn.close()
    return count


def _refresh_payload_stats(conn) -> None:
    """Refresh the payload_stats materialized view if it exists."""
    try:
        row = conn.execute(
            "SELECT 1 FROM pg_matviews WHERE matviewname = 'payload_stats'"
        ).fetchone()
        if row:
            conn.execute("REFRESH MATERIALIZED VIEW payload_stats")
            print("Refreshed payload_stats materialized view")
    except Exception as e:
        print(f"Warning: could not refresh payload_stats: {e}")

@app.local_entrypoint()
def main(
    mode: str = "ingest",
    # Ingest args
    top_n: int = 3,
    selection: str = "top",
    random_seed: int = -1,
    concurrency: int = 5,
    rps: float = 6.0,
    exclude_reasoning: bool = False,
    skip_deferred: bool = False,
    max_logs_per_vault: int = -1,
    max_full_logs_per_vault: int = -1,
    max_swaps_per_vault: int = -1,
    # Prep args
    limit: int = -1,
    include_all_decisions: bool = False,
    full_rebuild: bool = False,
    incremental_rebuild: bool = False,
    # Outcomes args
    outcomes_limit: int = -1,
    timeout_s: int = 30,
    retry_max_attempts: int = 6,
):
    if mode == "ingest":
        result = run_ingest.remote(
            top_n=top_n,
            selection=selection,
            random_seed=random_seed,
            concurrency=concurrency,
            requests_per_second=rps,
            exclude_reasoning=exclude_reasoning,
            retry_deferred=not skip_deferred,
            max_logs_per_vault=max_logs_per_vault,
            max_full_logs_per_vault=max_full_logs_per_vault,
            max_swaps_per_vault=max_swaps_per_vault,
        )
        print(f"\nIngest result: {result}")

    elif mode == "prep":
        result = run_prep.remote(
            limit=limit,
            include_all_decisions=include_all_decisions,
            full_rebuild=full_rebuild,
            incremental_rebuild=incremental_rebuild,
        )
        print(f"\nPrep result: {result}")

    elif mode == "outcomes":
        result = run_outcomes.remote(
            concurrency=concurrency,
            timeout_s=timeout_s,
            retry_max_attempts=retry_max_attempts,
            limit=outcomes_limit,
        )
        print(f"\nOutcomes result: {result}")

    elif mode == "reset-cursors":
        count = reset_cursors.remote()
        print(f"\nReset {count} cursor(s)")

    else:
        print(f"Unknown mode: {mode}. Use 'ingest', 'prep', 'outcomes', or 'reset-cursors'.")
