"""Modal wrapper for running ingest, data prep, and outcomes on the cloud.

All operations run directly against the DB on the xenon-data Modal volume.

Usage (via wrapper script):
    ./scripts/modal_capture.sh modal-ingest --top-n 10 --selection random
    ./scripts/modal_capture.sh modal-prep --export-parquet
    ./scripts/modal_capture.sh modal-outcomes
"""

import modal

app = modal.App("xenon-ingest")

volume = modal.Volume.from_name("xenon-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "aiohttp", "aiosqlite", "pyarrow", "numpy",
    )
    .add_local_python_source("pipelines")
)


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=14400,
    cpu=2,
)
def run_ingest(
    top_n: int = 3,
    selection: str = "top",
    random_seed: int = -1,
    concurrency: int = 5,
    leaderboard_sort_by: str = "total_pnl_usd",
    exclude_reasoning: bool = False,
    max_logs_per_vault: int = -1,
    max_full_logs_per_vault: int = -1,
    max_swaps_per_vault: int = -1,
) -> dict:
    """Run Terminal Markets ingest on Modal with volume-mounted DB."""
    import asyncio
    from pathlib import Path

    from pipelines.ingest.pipeline import BackfillConfig, run_backfill

    db_path = Path("/data/ingest/terminal_ingest.db")
    raw_dir = Path("/data/ingest/full_logs")
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        # Create parent dir so the DB can be created fresh
        db_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"No existing DB at {db_path} — will create a fresh one")

    config = BackfillConfig(
        db_path=db_path,
        raw_payload_dir=raw_dir,
        top_n=top_n,
        selection=selection,
        random_seed=random_seed if random_seed >= 0 else None,
        leaderboard_sort_by=leaderboard_sort_by,
        request_concurrency=concurrency,
        include_reasoning=not exclude_reasoning,
        max_logs_per_vault=max_logs_per_vault if max_logs_per_vault >= 0 else None,
        max_full_logs_per_vault=max_full_logs_per_vault if max_full_logs_per_vault >= 0 else None,
        max_swaps_per_vault=max_swaps_per_vault if max_swaps_per_vault >= 0 else None,
    )

    summary = asyncio.run(run_backfill(config))

    # Write stats snapshot for the dashboard
    _write_stats_snapshot()
    volume.commit()

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
    volumes={"/data": volume},
    image=image,
    timeout=7200,
    cpu=2,
)
def run_prep(
    trade_sample_size: int = 150,
    observation_sample_size: int = 150,
    paired_sample_size: int = 100,
    include_all_decisions: bool = False,
    export_parquet: bool = True,
    export_jsonl: bool = False,
) -> dict:
    """Run data prep on Modal with volume-mounted DB."""
    from pathlib import Path

    from pipelines.interp.prepare import PrepareConfig, run_prepare

    db_path = Path("/data/ingest/terminal_ingest.db")
    export_dir = Path("/data/interp_exports")
    export_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise FileNotFoundError(
            f"DB not found at {db_path}. Upload it first with: "
            "./scripts/modal_capture.sh modal-ingest"
        )

    config = PrepareConfig(
        db_path=db_path,
        trade_sample_size=trade_sample_size,
        observation_sample_size=observation_sample_size,
        paired_sample_size=paired_sample_size,
        only_focus_decisions=not include_all_decisions,
        export_parquet=export_parquet,
        export_jsonl=export_jsonl,
        export_dir=export_dir,
    )

    stats = run_prepare(config)

    # Write stats snapshot for the dashboard
    _write_stats_snapshot()
    volume.commit()

    print(f"\nPrep complete: {stats}")
    return stats


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=7200,
    cpu=2,
)
def run_outcomes(
    concurrency: int = 5,
    timeout_s: int = 30,
    retry_max_attempts: int = 6,
    limit: int = -1,
) -> dict:
    """Compute forward-looking PnL for swaps using Terminal Markets candle data."""
    import asyncio
    from pathlib import Path

    from pipelines.interp.outcomes import OutcomesConfig, run_outcomes

    db_path = Path("/data/ingest/terminal_ingest.db")
    if not db_path.exists():
        raise FileNotFoundError(
            f"DB not found at {db_path}. Run ingest first with: "
            "./scripts/modal_capture.sh modal-ingest"
        )

    config = OutcomesConfig(
        db_path=db_path,
        concurrency=concurrency,
        timeout_s=timeout_s,
        retry_max_attempts=retry_max_attempts,
        limit=limit if limit >= 0 else None,
    )

    stats = asyncio.run(run_outcomes(config))

    _write_stats_snapshot()
    volume.commit()

    print(f"\nOutcomes complete: {stats}")
    return stats


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=7200,
    cpu=2,
)
def backfill_payload_gz() -> dict:
    """Migrate file-based payloads into the payload_gz column in the DB."""
    import gzip
    import sqlite3
    from pathlib import Path

    db_path = Path("/data/ingest/terminal_ingest.db")
    if not db_path.exists():
        return {"error": "DB not found"}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    # Add column if missing
    try:
        conn.execute("ALTER TABLE full_logs ADD COLUMN payload_gz BLOB")
        conn.commit()
    except Exception:
        pass

    # Find rows that have a payload_path but no payload_gz
    rows = conn.execute(
        "SELECT log_id, payload_path FROM full_logs WHERE payload_gz IS NULL AND payload_path != ''"
    ).fetchall()

    print(f"Backfilling {len(rows)} payloads into DB...")
    migrated = 0
    missing = 0

    for i, row in enumerate(rows):
        payload_path = row["payload_path"]
        path = Path(payload_path)

        # Remap local paths to Modal volume paths
        if not path.exists():
            parts = payload_path.split("full_logs/")
            if len(parts) == 2:
                path = Path("/data/ingest/full_logs/") / parts[1]

        if path.exists():
            # File is already gzipped, just read the raw bytes
            raw_gz = path.read_bytes()
            conn.execute(
                "UPDATE full_logs SET payload_gz = ? WHERE log_id = ?",
                (raw_gz, row["log_id"]),
            )
            migrated += 1
        else:
            missing += 1

        if (i + 1) % 1000 == 0:
            conn.commit()
            print(f"  {i + 1}/{len(rows)} processed ({migrated} migrated, {missing} missing)")

    conn.commit()
    conn.close()
    volume.commit()

    result = {"total": len(rows), "migrated": migrated, "missing": missing}
    print(f"Backfill complete: {result}")
    return result


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=600,
    cpu=2,
)
def write_stats_snapshot() -> None:
    """Remote-callable wrapper for _write_stats_snapshot."""
    _write_stats_snapshot()
    volume.commit()
    print("Stats snapshot written and committed.")


def _write_stats_snapshot() -> None:
    """Write a small JSON stats file to the volume after ingest/prep.

    The dashboard downloads this ~1KB file instead of the whole DB.
    """
    import json as _json
    import sqlite3
    from pathlib import Path

    db_path = Path("/data/ingest/terminal_ingest.db")
    exports_dir = Path("/data/interp_exports")
    out_path = Path("/data/dashboard_stats.json")

    result: dict = {
        "ingest": {
            "vault_count": 0, "strategy_count": 0, "log_count": 0,
            "full_log_count": 0, "full_log_coverage_pct": 0,
            "parse_error_count": 0, "tables": [],
        },
        "outcomes": {
            "total_outcomes": 0, "unlabeled_swaps": 0, "total_swaps": 0,
            "avg_pnl_1h": None, "avg_pnl_4h": None, "avg_pnl_1d": None,
            "win_rate_1h": None, "risk_breakdown": [],
        },
        "prep": {
            "total_examples": 0, "high_quality": 0, "medium_quality": 0, "low_quality": 0,
            "trade_count": 0, "observation_count": 0,
            "export_files": [], "label_distribution": [],
        },
    }

    if not db_path.exists():
        out_path.write_text(_json.dumps(result))
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    def table_exists(name: str) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", [name]
        ).fetchone()
        return row is not None

    def table_count(name: str) -> int:
        if not table_exists(name):
            return 0
        row = conn.execute(f"SELECT COUNT(*) AS n FROM [{name}]").fetchone()
        return int(row["n"]) if row else 0

    # --- Ingest stats ---
    table_names = ["vaults", "strategies", "inference_logs", "full_logs", "swaps",
                    "trade_outcomes", "interp_examples_v0"]
    tables = []
    for tn in table_names:
        if table_exists(tn):
            tables.append({"name": tn, "count": table_count(tn)})

    vc = table_count("vaults")
    sc = table_count("strategies")
    lc = table_count("inference_logs")
    flc = table_count("full_logs")
    cov = round((flc / lc) * 100, 1) if lc else 0

    pe = 0
    if table_exists("full_logs"):
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM full_logs WHERE parse_error IS NOT NULL AND parse_error != ''"
        ).fetchone()
        pe = int(row["n"]) if row else 0

    result["ingest"] = {
        "vault_count": vc, "strategy_count": sc, "log_count": lc,
        "full_log_count": flc, "full_log_coverage_pct": cov,
        "parse_error_count": pe, "tables": tables,
    }

    # --- Outcomes stats ---
    total_swaps = table_count("swaps")
    result["outcomes"]["total_swaps"] = total_swaps

    if table_exists("trade_outcomes"):
        oc = table_count("trade_outcomes")
        result["outcomes"]["total_outcomes"] = oc
        result["outcomes"]["unlabeled_swaps"] = total_swaps - oc

        if oc > 0:
            agg = conn.execute(
                """SELECT
                    AVG(pnl_1h_pct) AS avg_1h,
                    AVG(pnl_4h_pct) AS avg_4h,
                    AVG(pnl_1d_pct) AS avg_1d,
                    AVG(CASE WHEN was_profitable_1h = 1 THEN 1.0 ELSE 0.0 END) AS wr_1h
                FROM trade_outcomes
                WHERE pnl_1h_pct IS NOT NULL"""
            ).fetchone()
            if agg:
                result["outcomes"]["avg_pnl_1h"] = round(agg["avg_1h"], 4) if agg["avg_1h"] is not None else None
                result["outcomes"]["avg_pnl_4h"] = round(agg["avg_4h"], 4) if agg["avg_4h"] is not None else None
                result["outcomes"]["avg_pnl_1d"] = round(agg["avg_1d"], 4) if agg["avg_1d"] is not None else None
                result["outcomes"]["win_rate_1h"] = round(agg["wr_1h"], 4) if agg["wr_1h"] is not None else None

            risk_rows = conn.execute(
                """SELECT
                    COALESCE(v.asset_risk_preference, 0) AS risk_level,
                    COUNT(*) AS cnt,
                    AVG(t.pnl_1h_pct) AS avg_1h,
                    AVG(t.pnl_4h_pct) AS avg_4h,
                    AVG(t.pnl_1d_pct) AS avg_1d,
                    AVG(CASE WHEN t.was_profitable_1h = 1 THEN 1.0 ELSE 0.0 END) AS wr_1h
                FROM trade_outcomes t
                JOIN swaps s ON s.log_id = t.log_id
                JOIN vaults v ON v.vault_address = s.vault_address
                WHERE t.pnl_1h_pct IS NOT NULL
                GROUP BY risk_level
                ORDER BY risk_level"""
            ).fetchall()
            result["outcomes"]["risk_breakdown"] = [
                {
                    "risk_level": int(r["risk_level"]),
                    "count": r["cnt"],
                    "avg_pnl_1h": round(r["avg_1h"], 4) if r["avg_1h"] is not None else None,
                    "avg_pnl_4h": round(r["avg_4h"], 4) if r["avg_4h"] is not None else None,
                    "avg_pnl_1d": round(r["avg_1d"], 4) if r["avg_1d"] is not None else None,
                    "win_rate_1h": round(r["wr_1h"], 4) if r["wr_1h"] is not None else None,
                }
                for r in risk_rows
            ]
    else:
        result["outcomes"]["unlabeled_swaps"] = total_swaps

    # --- Prep stats ---
    if table_exists("interp_examples_v0"):
        row = conn.execute("SELECT COUNT(*) AS n FROM interp_examples_v0").fetchone()
        result["prep"]["total_examples"] = int(row["n"]) if row else 0

        for quality in ("high", "medium", "low"):
            qrow = conn.execute(
                "SELECT COUNT(*) AS n FROM interp_examples_v0 WHERE label_quality = ?",
                [quality],
            ).fetchone()
            result["prep"][f"{quality}_quality"] = int(qrow["n"]) if qrow else 0

        dt_rows = conn.execute(
            """SELECT decision_type, COUNT(*) AS count,
               GROUP_CONCAT(DISTINCT trade_side) AS trade_side,
               AVG(vault_risk_preference) AS avg_risk
               FROM interp_examples_v0 GROUP BY decision_type"""
        ).fetchall()
        result["prep"]["label_distribution"] = [
            {
                "decision_type": r["decision_type"],
                "count": r["count"],
                "trade_side": r["trade_side"],
                "avg_risk": float(r["avg_risk"]) if r["avg_risk"] is not None else None,
            }
            for r in dt_rows
        ]
        result["prep"]["trade_count"] = sum(
            r["count"] for r in result["prep"]["label_distribution"]
            if r["decision_type"] == "trade"
        )
        result["prep"]["observation_count"] = sum(
            r["count"] for r in result["prep"]["label_distribution"]
            if r["decision_type"] == "record_observation"
        )

    conn.close()

    # Export files on volume
    if exports_dir.exists():
        for f in sorted(exports_dir.iterdir()):
            if f.suffix in (".parquet", ".jsonl"):
                size = f.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / 1024 / 1024:.1f} MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f} KB"
                else:
                    size_str = f"{size} B"
                result["prep"]["export_files"].append({"name": f.name, "size": size_str})

    out_path.write_text(_json.dumps(result))
    print(f"Wrote dashboard stats snapshot to {out_path}")


@app.local_entrypoint()
def main(
    mode: str = "ingest",
    # Ingest args
    top_n: int = 3,
    selection: str = "top",
    random_seed: int = -1,
    concurrency: int = 5,
    exclude_reasoning: bool = False,
    max_logs_per_vault: int = -1,
    # Prep args
    trade_sample_size: int = 150,
    observation_sample_size: int = 150,
    paired_sample_size: int = 100,
    include_all_decisions: bool = False,
    export_parquet: bool = True,
    export_jsonl: bool = False,
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
            exclude_reasoning=exclude_reasoning,
            max_logs_per_vault=max_logs_per_vault,
        )
        print(f"\nIngest result: {result}")

    elif mode == "prep":
        result = run_prep.remote(
            trade_sample_size=trade_sample_size,
            observation_sample_size=observation_sample_size,
            paired_sample_size=paired_sample_size,
            include_all_decisions=include_all_decisions,
            export_parquet=export_parquet,
            export_jsonl=export_jsonl,
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

    elif mode == "snapshot":
        write_stats_snapshot.remote()

    elif mode == "backfill-payloads":
        result = backfill_payload_gz.remote()
        print(f"\nBackfill result: {result}")

    else:
        print(f"Unknown mode: {mode}. Use 'ingest', 'prep', 'outcomes', 'snapshot', or 'backfill-payloads'.")
