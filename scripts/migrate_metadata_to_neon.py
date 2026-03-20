"""Migrate capture metadata from Modal volume parquet files to Neon Postgres.

Reads metadata.parquet for both normal and counterfactual captures from the
Modal volume and inserts into the corresponding Neon tables.

Usage::

    # Run on Modal (needs volume access + Neon secret)
    modal run scripts/migrate_metadata_to_neon.py
"""

import modal

app = modal.App("xenon-migrate-metadata")

volume = modal.Volume.from_name("xenon-data", create_if_missing=True)
neon_secret = modal.Secret.from_name("xenon-neon")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("pyarrow", "psycopg[binary]")
)


@app.function(
    image=image,
    secrets=[neon_secret],
    volumes={"/data": volume},
    timeout=600,
)
def migrate_metadata():
    """Read parquet metadata from volume and insert into Neon."""
    from pathlib import Path

    import psycopg
    import pyarrow.parquet as pq

    from os import environ

    conn = psycopg.connect(
        environ["XENON_NEON_DATABASE_URL"],
        autocommit=False,
        row_factory=psycopg.rows.dict_row,
    )

    # ── Normal capture metadata ──
    normal_meta = Path("/data/activations/metadata.parquet")
    if normal_meta.exists():
        rows = pq.read_table(normal_meta).to_pylist()
        print(f"Normal captures: {len(rows)} rows in metadata.parquet")

        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS capture_metadata (
                    log_id            INT PRIMARY KEY,
                    seq_len           INT NOT NULL,
                    prompt_hash       TEXT,
                    capture_timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
                    file_size_bytes   BIGINT NOT NULL DEFAULT 0,
                    elapsed_s         REAL NOT NULL DEFAULT 0,
                    has_router        BOOLEAN NOT NULL DEFAULT false,
                    captured_layers   TEXT,
                    pooling           TEXT NOT NULL DEFAULT 'none',
                    num_layers_captured INT NOT NULL DEFAULT 0,
                    hidden_dim        INT NOT NULL DEFAULT 0,
                    num_experts       INT
                )
            """)
            inserted = 0
            for row in rows:
                cur.execute("""
                    INSERT INTO capture_metadata
                        (log_id, seq_len, prompt_hash, capture_timestamp,
                         file_size_bytes, elapsed_s, has_router, captured_layers,
                         pooling, num_layers_captured, hidden_dim, num_experts)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (log_id) DO NOTHING
                """, (
                    row["log_id"], row["seq_len"], row.get("prompt_hash"),
                    row.get("capture_timestamp"), row.get("file_size_bytes", 0),
                    row.get("elapsed_s", 0), row.get("has_router", False),
                    row.get("captured_layers"), row.get("pooling", "none"),
                    row.get("num_layers_captured", 0), row.get("hidden_dim", 0),
                    row.get("num_experts"),
                ))
                inserted += 1
        conn.commit()
        print(f"  Inserted/skipped {inserted} normal capture rows")
    else:
        print("No normal metadata.parquet found")

    # ── Counterfactual capture metadata ──
    cf_base = Path("/data/activations/counterfactual")
    cf_count = 0
    if cf_base.exists():
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS counterfactual_captures (
                    capture_id        TEXT PRIMARY KEY,
                    experiment_id     TEXT NOT NULL,
                    snapshot_id       TEXT NOT NULL,
                    dataset           TEXT NOT NULL DEFAULT 'a',
                    variant           TEXT NOT NULL,
                    seq_len           INT NOT NULL,
                    n_rows            INT NOT NULL,
                    n_residual_keys   INT NOT NULL DEFAULT 0,
                    n_router_keys    INT NOT NULL DEFAULT 0,
                    file_size_bytes   BIGINT NOT NULL DEFAULT 0,
                    elapsed_s         REAL NOT NULL DEFAULT 0,
                    capture_timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)

        for exp_dir in cf_base.iterdir():
            if not exp_dir.is_dir():
                continue
            experiment_id = exp_dir.name
            meta_path = exp_dir / "metadata.parquet"
            if not meta_path.exists():
                print(f"  {experiment_id}: no metadata.parquet")
                continue

            rows = pq.read_table(meta_path).to_pylist()
            print(f"  {experiment_id}: {len(rows)} rows")

            with conn.cursor() as cur:
                for row in rows:
                    # Infer dataset from variant name
                    variant = row.get("variant", "")
                    dataset = "a" if variant in ("low_raw", "high_raw", "low_pad", "high_pad") else "b"

                    cur.execute("""
                        INSERT INTO counterfactual_captures
                            (capture_id, experiment_id, snapshot_id, dataset, variant,
                             seq_len, n_rows, n_residual_keys, n_router_keys,
                             file_size_bytes, elapsed_s, capture_timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (capture_id) DO NOTHING
                    """, (
                        row["capture_id"], experiment_id,
                        row["snapshot_id"], dataset,
                        variant, row["seq_len"], row["n_rows"],
                        row.get("n_residual_keys", 0), row.get("n_router_keys", 0),
                        row.get("file_size_bytes", 0), row.get("elapsed_s", 0),
                        row.get("capture_timestamp"),
                    ))
                    cf_count += 1
            conn.commit()

    print(f"\nCounterfactual captures: {cf_count} rows migrated")
    conn.close()
    print("Migration complete.")
