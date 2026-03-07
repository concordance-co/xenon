"""Modal wrapper for running ingest and data prep on the cloud.

Uploads the local SQLite DB to the xenon-data volume, then runs
ingest (fetching more data from Terminal Markets API) or prep
(building labeled examples from the DB) on Modal.

Usage (via wrapper script):
    ./scripts/modal_capture.sh upload-db
    ./scripts/modal_capture.sh modal-ingest --top-n 10 --selection random
    ./scripts/modal_capture.sh modal-prep --export-parquet
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
    timeout=3600,
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
    timeout=1800,
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
            "./scripts/modal_capture.sh upload-db"
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
    volume.commit()

    print(f"\nPrep complete: {stats}")
    return stats


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

    else:
        print(f"Unknown mode: {mode}. Use 'ingest' or 'prep'.")
