"""Reconcile counterfactual captures on Modal volume with Neon DB.

Scans safetensors files, checks which are valid (>1 key, correct shapes),
deletes broken ones, and inserts metadata for valid ones into Neon.

Usage:
    modal run scripts/reconcile_captures.py
"""
import modal

app = modal.App("xenon-reconcile-captures")
volume = modal.Volume.from_name("xenon-data", create_if_missing=True)
neon_secret = modal.Secret.from_name("xenon-neon")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("safetensors", "torch", "packaging", "psycopg[binary]")
)


@app.function(
    volumes={"/data": volume},
    image=image,
    secrets=[neon_secret],
    timeout=600,
)
def reconcile(experiment_id: str = "init", dry_run: bool = False):
    from pathlib import Path
    from safetensors.torch import load_file
    import psycopg
    from psycopg.rows import dict_row
    from os import environ

    residual_dir = Path(f"/data/activations/counterfactual/{experiment_id}/residual")
    if not residual_dir.exists():
        print(f"No residual dir: {residual_dir}")
        return

    files = sorted(residual_dir.glob("*.safetensors"))
    print(f"Found {len(files)} safetensors files")

    good = []
    bad = []

    for f in files:
        try:
            tensors = load_file(str(f))
            n_keys = len(tensors)
            # Check for correct shape: good files have (48, 2048), bad have (4192, 2048)
            sample = next(iter(tensors.values()))
            shape = tuple(sample.shape)

            if n_keys > 1 and shape[0] == 48:
                good.append((f, n_keys, shape))
            else:
                bad.append((f, n_keys, shape))
        except Exception as e:
            bad.append((f, 0, str(e)))

    print(f"\nGood: {len(good)}, Bad: {len(bad)}")

    if bad:
        print(f"\nBad files (first 10):")
        for f, n_keys, shape in bad[:10]:
            print(f"  {f.name}: {n_keys} keys, shape={shape}")

    if good:
        print(f"\nGood files (first 5):")
        for f, n_keys, shape in good[:5]:
            print(f"  {f.name}: {n_keys} keys, shape={shape}")

    if dry_run:
        print("\nDry run — no changes made.")
        return

    # Delete bad files
    for f, _, _ in bad:
        f.unlink()
    print(f"\nDeleted {len(bad)} broken files")

    # Also delete corresponding router files for bad captures
    router_dir = Path(f"/data/activations/counterfactual/{experiment_id}/router")
    if router_dir.exists():
        for f, _, _ in bad:
            rf = router_dir / f.name
            if rf.exists():
                rf.unlink()

    # Insert good captures into Neon
    conn = psycopg.connect(
        environ["XENON_NEON_DATABASE_URL"],
        autocommit=False,
        row_factory=dict_row,
    )
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS counterfactual_captures (
                capture_id        TEXT PRIMARY KEY,
                experiment_id     TEXT NOT NULL,
                snapshot_id       TEXT NOT NULL,
                dataset           TEXT NOT NULL,
                variant           TEXT NOT NULL,
                seq_len           INT NOT NULL,
                n_rows            INT NOT NULL,
                n_residual_keys   INT NOT NULL DEFAULT 0,
                n_router_keys     INT NOT NULL DEFAULT 0,
                file_size_bytes   BIGINT NOT NULL DEFAULT 0,
                elapsed_s         REAL NOT NULL DEFAULT 0,
                capture_timestamp TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)

        inserted = 0
        for f, n_keys, shape in good:
            # Parse capture_id from filename: {capture_id}.safetensors
            capture_id = f.stem  # e.g. "0xABC:12345_low_raw"

            # Parse variant from capture_id
            # Format: {vault}:{log_id}_{variant}
            parts = capture_id.rsplit("_", 1)
            if len(parts) == 2:
                base, variant = parts
                # But variant could be multi-word like "settings_all1"
                # Try rsplit with maxsplit=2 for settings variants
                for v in ("low_raw", "high_raw", "low_pad", "high_pad",
                          "settings_all1", "settings_all5", "original"):
                    if capture_id.endswith(f"_{v}"):
                        variant = v
                        base = capture_id[:-(len(v) + 1)]
                        break
            else:
                variant = "unknown"
                base = capture_id

            # Determine dataset from variant
            if variant in ("low_raw", "high_raw", "low_pad", "high_pad"):
                dataset = "a"
            else:
                dataset = "b"

            # Parse snapshot_id from base: {vault}:{log_id}
            snapshot_id = base  # best we can do without DB lookup

            file_size = f.stat().st_size

            # Count rows from keys
            n_rows = sum(1 for k in load_file(str(f)).keys() if k.startswith("row_mean_"))

            cur.execute("""
                INSERT INTO counterfactual_captures
                    (capture_id, experiment_id, snapshot_id, dataset, variant,
                     seq_len, n_rows, n_residual_keys, n_router_keys,
                     file_size_bytes, elapsed_s)
                VALUES (%s, %s, %s, %s, %s, 0, %s, %s, 0, %s, 0)
                ON CONFLICT (capture_id) DO UPDATE SET
                    n_residual_keys = EXCLUDED.n_residual_keys,
                    file_size_bytes = EXCLUDED.file_size_bytes
            """, (
                capture_id, experiment_id, snapshot_id, dataset, variant,
                n_rows, n_keys, file_size,
            ))
            inserted += 1

    conn.commit()
    conn.close()

    volume.commit()
    print(f"Inserted {inserted} rows into Neon, deleted {len(bad)} broken files")
    print("Done.")


@app.local_entrypoint()
def main(experiment_id: str = "init", dry_run: bool = False):
    reconcile.remote(experiment_id=experiment_id, dry_run=dry_run)
