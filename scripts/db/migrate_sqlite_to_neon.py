"""One-time migration: copy all data from SQLite on Modal volume to Neon Postgres.

Reads the SQLite database and gzipped payload files from the Modal volume,
writes everything to Neon Postgres using the new schema.

Usage:
    modal run scripts/db/migrate_sqlite_to_neon.py
    modal run scripts/db/migrate_sqlite_to_neon.py --dry-run
    modal run scripts/db/migrate_sqlite_to_neon.py --skip-payloads  # skip gzip→JSONB (slow)
"""

import modal

app = modal.App("xenon-migrate-sqlite-to-neon")
volume = modal.Volume.from_name("xenon-data", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("psycopg[binary]")
    .add_local_python_source("pipelines")
)


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=7200,
    cpu=2,
    memory=4096,
    secrets=[modal.Secret.from_name("xenon-neon")],
)
def migrate(dry_run: bool = False, skip_payloads: bool = False, batch_size: int = 500) -> dict:
    import gzip
    import json
    import sqlite3
    from pathlib import Path

    import psycopg
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb

    from pipelines.db import ensure_schema, require_neon_dsn

    # ---------------------------------------------------------------
    # Locate SQLite database on the volume
    # ---------------------------------------------------------------
    candidates = [
        Path("/data/terminal_ingest.db"),
        Path("/data/ingest/terminal_ingest.db"),
    ]
    sqlite_path = None
    for p in candidates:
        if p.exists():
            sqlite_path = p
            break

    if sqlite_path is None:
        print("ERROR: Could not find SQLite database on volume. Checked:")
        for p in candidates:
            print(f"  {p}")
        return {"error": "SQLite DB not found"}

    print(f"Found SQLite DB: {sqlite_path} ({sqlite_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # ---------------------------------------------------------------
    # Connect to both databases
    # ---------------------------------------------------------------
    src = sqlite3.connect(str(sqlite_path))
    src.row_factory = sqlite3.Row

    dst = psycopg.connect(require_neon_dsn(), autocommit=False, row_factory=dict_row)
    ensure_schema(dst)

    stats: dict[str, int] = {}

    # ---------------------------------------------------------------
    # Migration plan: table → (sqlite columns, neon columns, transform)
    # We map SQLite column names to Neon column names.
    # ---------------------------------------------------------------

    def _migrate_table(
        table: str,
        *,
        columns: list[str],
        pk_columns: list[str],
        bool_columns: list[str] | None = None,
    ) -> int:
        """Copy rows from SQLite to Neon via pipelined executemany.

        Commits every batch_size rows so the migration is resumable —
        ON CONFLICT means re-running after a crash just skips already-migrated rows.
        """
        bool_cols = set(bool_columns or [])

        count = src.execute(f"SELECT COUNT(*) AS n FROM [{table}]").fetchone()["n"]

        # Check how many already exist in Neon (for resume)
        existing = dst.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        if existing > 0:
            print(f"\n--- {table}: {count:,} rows (Neon already has {existing:,}) ---")
        else:
            print(f"\n--- {table}: {count:,} rows ---")

        if count == 0 or dry_run:
            return count

        if existing >= count:
            print(f"  Skipped (Neon already has {existing:,} >= SQLite {count:,})")
            return existing

        placeholders = ", ".join(["%s"] * len(columns))
        col_list = ", ".join(columns)
        pk_list = ", ".join(pk_columns)
        non_pk = [c for c in columns if c not in pk_columns]
        update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in non_pk) if non_pk else ""

        if update_clause:
            upsert_sql = (
                f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
                f"ON CONFLICT ({pk_list}) DO UPDATE SET {update_clause}"
            )
        else:
            upsert_sql = (
                f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
                f"ON CONFLICT ({pk_list}) DO NOTHING"
            )

        inserted = 0
        cursor = src.execute(f"SELECT * FROM [{table}]")

        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break

            values_batch = []
            for row in rows:
                vals = []
                for col in columns:
                    v = row[col]
                    if col in bool_cols and v is not None:
                        v = bool(v)
                    vals.append(v)
                values_batch.append(tuple(vals))

            # executemany uses psycopg3 pipelining — sends all rows in one
            # network round-trip instead of one per row
            with dst.cursor() as cur:
                cur.executemany(upsert_sql, values_batch)

            # Commit per batch so we can resume on failure
            dst.commit()
            inserted += len(values_batch)
            if inserted % 5000 < batch_size or inserted >= count:
                print(f"  {inserted:,}/{count:,}")

        print(f"  Done: {inserted:,} rows migrated")
        return inserted

    # ---------------------------------------------------------------
    # 1. Vaults
    # ---------------------------------------------------------------
    stats["vaults"] = _migrate_table(
        "vaults",
        columns=[
            "vault_address", "owner_address", "nft_id", "nft_name",
            "persona_json", "trade_size", "trading_activity", "holding_style",
            "diversification", "asset_risk_preference", "max_trade_amount",
            "slippage_bps", "paused", "state", "leaderboard_rank",
            "total_pnl_usd", "realized_pnl_usd", "unrealized_pnl_usd",
            "created_block", "updated_block", "fetched_at",
        ],
        pk_columns=["vault_address"],
        bool_columns=["paused"],
    )

    # ---------------------------------------------------------------
    # 2. Strategies
    # ---------------------------------------------------------------
    stats["strategies"] = _migrate_table(
        "strategies",
        columns=[
            "vault_address", "strategy_id", "vault_owner_address", "content",
            "expiry", "enabled", "strategy_priority", "created_block",
            "updated_block", "fetched_at",
        ],
        pk_columns=["vault_address", "strategy_id"],
        bool_columns=["enabled"],
    )

    # ---------------------------------------------------------------
    # 3. Inference logs
    # ---------------------------------------------------------------
    stats["inference_logs"] = _migrate_table(
        "inference_logs",
        columns=[
            "id", "cursor", "vault_address", "request_id", "execution_key",
            "tool", "tool_args_json", "strategy_id", "status",
            "inference_duration_ms", "error", "transaction_hash",
            "created_at", "completed_at", "fetched_at",
        ],
        pk_columns=["id"],
    )

    # ---------------------------------------------------------------
    # 4. Full logs (without raw_payload — that's step 6)
    # ---------------------------------------------------------------
    # SQLite full_logs may have columns that Neon doesn't (payload_path etc.)
    # We only copy the columns that exist in the Neon schema.
    neon_full_log_cols = [
        "log_id", "vault_address", "prompt_text", "completion_text",
        "reasoning_content", "tool_calls_json", "llm_model",
        "prompt_tokens", "completion_tokens", "reasoning_tokens",
        "total_tokens", "parse_error", "fetched_at",
    ]

    # Check which columns actually exist in SQLite
    sqlite_cols = {
        r["name"] for r in src.execute("PRAGMA table_info(full_logs)").fetchall()
    }
    full_log_cols = [c for c in neon_full_log_cols if c in sqlite_cols]

    stats["full_logs"] = _migrate_table(
        "full_logs",
        columns=full_log_cols,
        pk_columns=["log_id"],
    )

    # ---------------------------------------------------------------
    # 5. Swaps
    # ---------------------------------------------------------------
    stats["swaps"] = _migrate_table(
        "swaps",
        columns=[
            "transaction_hash", "block_number", "log_index", "timestamp",
            "pool_id", "token_address", "token_name", "token_symbol",
            "vault_address", "is_reap_twap", "side", "token_amount",
            "eth_amount", "eth_price_usd", "effective_price_eth",
            "effective_price_usd", "log_id", "strategy_id", "fetched_at",
        ],
        pk_columns=["transaction_hash", "log_index"],
        bool_columns=["is_reap_twap"],
    )

    # ---------------------------------------------------------------
    # 6. Ingest cursors
    # ---------------------------------------------------------------
    if "ingest_cursors" in {r["name"] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}:
        stats["ingest_cursors"] = _migrate_table(
            "ingest_cursors",
            columns=["vault_address", "resource", "last_cursor", "updated_at"],
            pk_columns=["vault_address", "resource"],
        )

    # ---------------------------------------------------------------
    # 7. Trade outcomes (if exists)
    # ---------------------------------------------------------------
    all_tables = {r["name"] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "trade_outcomes" in all_tables:
        stats["trade_outcomes"] = _migrate_table(
            "trade_outcomes",
            columns=[
                "log_id", "transaction_hash", "side", "token_address",
                "entry_price_eth", "entry_price_usd", "eth_price_usd_at_entry",
                "price_1h_eth", "price_4h_eth", "price_1d_eth",
                "price_1h_usd", "price_4h_usd", "price_1d_usd",
                "pnl_1h_pct", "pnl_4h_pct", "pnl_1d_pct",
                "was_profitable_1h", "candle_data_json", "computed_at",
            ],
            pk_columns=["log_id"],
            bool_columns=["was_profitable_1h"],
        )

    # ---------------------------------------------------------------
    # 8. Interp examples (if exists)
    # ---------------------------------------------------------------
    if "interp_examples_v0" in all_tables:
        interp_bool_cols = [
            "is_trade", "joined_swap", "parse_ok", "was_profitable_1h",
            "has_messages", "has_tools", "has_market", "has_portfolio",
            "has_strategy", "has_config", "has_memory", "context_complete",
        ]
        # Get actual columns from SQLite
        sqlite_interp_cols = [
            r["name"] for r in src.execute("PRAGMA table_info(interp_examples_v0)").fetchall()
        ]
        # Neon schema columns (from DDL)
        neon_interp_cols = [
            "example_id", "log_id", "vault_address", "created_at", "strategy_id",
            "transaction_hash", "is_trade", "prompt_messages_json", "system_text",
            "user_text", "tools_available_json", "market_snapshot_json",
            "portfolio_snapshot_json", "strategy_snapshot_json", "config_snapshot_json",
            "memory_snapshot_json", "model_source", "assistant_content",
            "reasoning_content", "tool_calls_json", "action_name", "decision_type",
            "trade_side", "asset", "size", "observation_text", "joined_swap",
            "swap_side", "swap_token_address", "swap_token_symbol", "swap_price_usd",
            "pnl_1h_pct", "pnl_4h_pct", "pnl_1d_pct", "was_profitable_1h",
            "entry_price_usd", "entry_price_eth", "vault_trade_size",
            "vault_trading_activity", "vault_holding_style", "vault_diversification",
            "vault_risk_preference", "parse_ok", "parse_error", "has_messages",
            "has_tools", "has_market", "has_portfolio", "has_strategy", "has_config",
            "has_memory", "context_complete", "missing_blocks_json", "label_quality",
            "label_confidence", "ingest_version", "transform_version", "built_at",
        ]
        # Only migrate columns that exist in both
        migrate_cols = [c for c in neon_interp_cols if c in sqlite_interp_cols]

        stats["interp_examples_v0"] = _migrate_table(
            "interp_examples_v0",
            columns=migrate_cols,
            pk_columns=["example_id"],
            bool_columns=interp_bool_cols,
        )

    # ---------------------------------------------------------------
    # 9. Backfill raw_payload JSONB from gzipped files
    # ---------------------------------------------------------------
    if not skip_payloads and not dry_run:
        print("\n--- Backfilling raw_payload from gzipped files ---")

        rows_needing = dst.execute(
            "SELECT log_id FROM full_logs WHERE raw_payload IS NULL ORDER BY log_id"
        ).fetchall()
        print(f"  {len(rows_needing)} full_logs rows need raw_payload")

        # Also check if SQLite has raw_payload already populated
        has_raw = "raw_payload" in sqlite_cols
        if has_raw:
            populated = src.execute(
                "SELECT COUNT(*) AS n FROM full_logs WHERE raw_payload IS NOT NULL"
            ).fetchone()["n"]
            print(f"  SQLite has {populated} rows with raw_payload already set")

        # First pass: copy raw_payload from SQLite if available
        if has_raw:
            updated_from_sqlite = 0
            for i in range(0, len(rows_needing), batch_size):
                batch_ids = [r["log_id"] for r in rows_needing[i:i + batch_size]]
                placeholders_list = ",".join("?" * len(batch_ids))
                src_rows = src.execute(
                    f"SELECT log_id, raw_payload FROM full_logs "
                    f"WHERE log_id IN ({placeholders_list}) AND raw_payload IS NOT NULL",
                    batch_ids,
                ).fetchall()

                update_batch = []
                for sr in src_rows:
                    try:
                        payload = json.loads(sr["raw_payload"])
                        update_batch.append((Jsonb(payload), sr["log_id"]))
                    except (json.JSONDecodeError, TypeError):
                        continue

                if update_batch:
                    with dst.cursor() as cur:
                        cur.executemany(
                            "UPDATE full_logs SET raw_payload = %s WHERE log_id = %s",
                            update_batch,
                        )
                    updated_from_sqlite += len(update_batch)

                dst.commit()
                if (i + batch_size) % 5000 < batch_size:
                    print(f"    Progress: {min(i + batch_size, len(rows_needing))}/{len(rows_needing)}")

            print(f"  Copied {updated_from_sqlite} raw_payload values from SQLite")

        # Second pass: read from gzipped files for any still missing
        # Parallelized with threads — the volume is network-attached so I/O
        # is the bottleneck, not CPU.
        still_missing = dst.execute(
            "SELECT log_id FROM full_logs WHERE raw_payload IS NULL ORDER BY log_id"
        ).fetchall()

        if still_missing and "payload_path" in sqlite_cols:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            print(f"  {len(still_missing)} still missing — trying gzipped files (parallel)...")
            updated_from_files = 0
            failed_files = 0

            # Build lookup: log_id → payload_path from SQLite
            missing_ids = [r["log_id"] for r in still_missing]
            id_to_path: dict[int, str] = {}
            for i in range(0, len(missing_ids), batch_size):
                batch_ids = missing_ids[i:i + batch_size]
                placeholders_list = ",".join("?" * len(batch_ids))
                path_rows = src.execute(
                    f"SELECT log_id, payload_path FROM full_logs "
                    f"WHERE log_id IN ({placeholders_list}) AND payload_path IS NOT NULL",
                    batch_ids,
                ).fetchall()
                for pr in path_rows:
                    id_to_path[pr["log_id"]] = pr["payload_path"]

            print(f"  {len(id_to_path)} have payload_path in SQLite")

            def _read_one_gz(log_id: int, payload_path: str):
                """Read a single gzipped payload. Runs in thread pool."""
                file_candidates = [Path(payload_path)]
                parts = payload_path.rsplit("full_logs/", 1)
                if len(parts) == 2:
                    file_candidates.append(Path("/data/ingest/full_logs") / parts[1])
                    file_candidates.append(Path("/data/full_logs") / parts[1])

                for fp in file_candidates:
                    if fp.exists():
                        try:
                            with gzip.open(fp, "rt", encoding="utf-8") as f:
                                return (log_id, json.load(f))
                        except Exception:
                            continue
                return (log_id, None)

            # Process in chunks: read files in parallel, then batch-write to Neon
            chunk_size = batch_size * 4  # larger chunks to amortize thread pool overhead
            items = list(id_to_path.items())
            workers = 16  # concurrent file reads

            for chunk_start in range(0, len(items), chunk_size):
                chunk = items[chunk_start:chunk_start + chunk_size]
                update_batch = []

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(_read_one_gz, lid, pp): lid
                        for lid, pp in chunk
                    }
                    for fut in as_completed(futures):
                        lid, payload = fut.result()
                        if payload is not None:
                            update_batch.append((Jsonb(payload), lid))
                        else:
                            failed_files += 1

                if update_batch:
                    with dst.cursor() as cur:
                        cur.executemany(
                            "UPDATE full_logs SET raw_payload = %s WHERE log_id = %s",
                            update_batch,
                        )
                    updated_from_files += len(update_batch)

                dst.commit()
                done = min(chunk_start + chunk_size, len(items))
                print(f"    Progress: {done}/{len(items)} ({updated_from_files} updated, {failed_files} failed)")

            print(f"  From files: {updated_from_files} updated, {failed_files} not found")
            stats["payload_backfill_from_files"] = updated_from_files
            stats["payload_backfill_failed"] = failed_files

        # Final count
        final_null = dst.execute(
            "SELECT COUNT(*) AS n FROM full_logs WHERE raw_payload IS NULL"
        ).fetchone()["n"]
        final_total = dst.execute("SELECT COUNT(*) AS n FROM full_logs").fetchone()["n"]
        print(f"  Final: {final_total - final_null}/{final_total} have raw_payload")
        stats["payload_coverage"] = f"{final_total - final_null}/{final_total}"

    # ---------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------
    src.close()
    dst.close()

    print("\n=== Migration Summary ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return stats


@app.local_entrypoint()
def main(dry_run: bool = False, skip_payloads: bool = False, batch_size: int = 500):
    result = migrate.remote(dry_run=dry_run, skip_payloads=skip_payloads, batch_size=batch_size)
    print(f"\nMigration result: {result}")
