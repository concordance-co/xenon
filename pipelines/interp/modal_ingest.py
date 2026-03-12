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
    .apt_install("sqlite3")
    .pip_install(
        "aiohttp", "aiosqlite", "pyarrow", "numpy",
    )
    .add_local_python_source("pipelines")
)

DEFAULT_DB_BACKUP_RETAIN = 30


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


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=3600,
    cpu=2,
)
def repair_db() -> dict:
    """Attempt in-place SQLite salvage on Modal volume without local download."""
    import shlex
    import shutil
    import sqlite3
    import subprocess
    from datetime import UTC, datetime
    from pathlib import Path
    from shutil import which

    db_path = Path("/data/ingest/terminal_ingest.db")
    wal_path = Path("/data/ingest/terminal_ingest.db-wal")
    shm_path = Path("/data/ingest/terminal_ingest.db-shm")
    backup_dir = Path("/data/ingest/repair_backups")
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        return {"status": "error", "message": f"DB not found: {db_path}"}

    sqlite_cli = which("sqlite3")
    if not sqlite_cli:
        return {
            "status": "error",
            "message": "sqlite3 CLI not found in image. Redeploy with sqlite3 installed.",
        }

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_db = backup_dir / f"terminal_ingest.{ts}.db"
    backup_wal = backup_dir / f"terminal_ingest.{ts}.db-wal"
    backup_shm = backup_dir / f"terminal_ingest.{ts}.db-shm"

    shutil.copy2(db_path, backup_db)
    if wal_path.exists():
        shutil.copy2(wal_path, backup_wal)
    if shm_path.exists():
        shutil.copy2(shm_path, backup_shm)

    def _integrity_check(path: Path) -> str:
        try:
            conn = sqlite3.connect(str(path))
            out = conn.execute("PRAGMA integrity_check;").fetchone()
            conn.close()
            if out and out[0]:
                return str(out[0])
            return "unknown"
        except Exception as e:
            return f"error: {e}"

    def _table_names(path: Path) -> list[str]:
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            conn.close()
            return [str(r[0]) for r in rows]
        except Exception:
            return []

    before = _integrity_check(db_path)
    if before == "ok":
        volume.commit()
        return {
            "status": "ok",
            "message": "DB already healthy",
            "integrity_before": before,
            "backup_db": str(backup_db),
        }

    # First attempt: checkpoint any WAL state.
    checkpoint_error = None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        conn.close()
    except Exception as e:
        checkpoint_error = str(e)

    after_checkpoint = _integrity_check(db_path)
    if after_checkpoint == "ok":
        if wal_path.exists():
            wal_path.unlink()
        if shm_path.exists():
            shm_path.unlink()
        volume.commit()
        return {
            "status": "ok",
            "message": "Recovered after WAL checkpoint",
            "integrity_before": before,
            "integrity_after": after_checkpoint,
            "backup_db": str(backup_db),
            "checkpoint_error": checkpoint_error,
        }

    # Second attempt: sqlite3 .recover into a fresh file.
    recovered = Path(f"/data/ingest/terminal_ingest.recovered.{ts}.db")
    command = (
        f"{shlex.quote(sqlite_cli)} {shlex.quote(str(db_path))} '.recover' | "
        f"{shlex.quote(sqlite_cli)} {shlex.quote(str(recovered))}"
    )
    proc = subprocess.run(
        ["bash", "-lc", command],
        text=True,
        capture_output=True,
    )
    recovered_integrity = _integrity_check(recovered) if recovered.exists() else "missing"
    recovered_tables = _table_names(recovered) if recovered.exists() else []

    if proc.returncode != 0 or recovered_integrity != "ok":
        volume.commit()
        return {
            "status": "error",
            "message": "Recover step failed",
            "integrity_before": before,
            "integrity_after_checkpoint": after_checkpoint,
            "integrity_recovered": recovered_integrity,
            "recovered_table_count": len(recovered_tables),
            "recover_returncode": proc.returncode,
            "recover_stdout_tail": proc.stdout[-2000:],
            "recover_stderr_tail": proc.stderr[-2000:],
            "backup_db": str(backup_db),
            "checkpoint_error": checkpoint_error,
        }

    if len(recovered_tables) == 0:
        volume.commit()
        return {
            "status": "error",
            "message": "Recovered DB has zero tables; refusing to replace live DB",
            "integrity_before": before,
            "integrity_after_checkpoint": after_checkpoint,
            "integrity_recovered": recovered_integrity,
            "recovered_table_count": 0,
            "recover_returncode": proc.returncode,
            "recover_stdout_tail": proc.stdout[-2000:],
            "recover_stderr_tail": proc.stderr[-2000:],
            "backup_db": str(backup_db),
            "checkpoint_error": checkpoint_error,
        }

    quarantined = backup_dir / f"terminal_ingest.corrupt.{ts}.db"
    db_path.replace(quarantined)
    recovered.replace(db_path)
    if wal_path.exists():
        wal_path.unlink()
    if shm_path.exists():
        shm_path.unlink()

    # Normalize DB pragmas for next writes.
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.close()

    final = _integrity_check(db_path)
    volume.commit()
    return {
        "status": "ok" if final == "ok" else "warning",
        "message": "Recovered with sqlite .recover",
        "integrity_before": before,
        "integrity_after_checkpoint": after_checkpoint,
        "integrity_recovered": recovered_integrity,
        "integrity_final": final,
        "backup_db": str(backup_db),
        "quarantined_db": str(quarantined),
        "checkpoint_error": checkpoint_error,
    }


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=600,
    cpu=2,
)
def inspect_db_path(path: str = "ingest/terminal_ingest.db") -> dict:
    """Inspect a DB path on Modal volume without downloading it."""
    import sqlite3
    from pathlib import Path

    p = Path(path)
    if not p.is_absolute():
        p = Path("/data") / p

    result: dict[str, object] = {
        "path": str(p),
        "exists": p.exists(),
    }
    if not p.exists():
        return result

    result["size_mb"] = round(p.stat().st_size / 1024 / 1024, 2)

    try:
        conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        out = conn.execute("PRAGMA integrity_check;").fetchone()
        result["integrity_check"] = out[0] if out else "unknown"
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [str(r[0]) for r in tables]
        result["table_count"] = len(table_names)
        result["tables"] = table_names
        key_counts = {}
        for table in ("vaults", "inference_logs", "full_logs", "swaps", "trade_outcomes", "interp_examples_v0"):
            if table in table_names:
                try:
                    n = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
                    key_counts[table] = int(n)
                except Exception:
                    key_counts[table] = "error"
        result["key_counts"] = key_counts
        conn.close()
    except Exception as e:
        result["error"] = str(e)
    return result


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=600,
    cpu=2,
)
def inspect_full_logs(shard: str = "", sample_n: int = 8) -> dict:
    """Inspect full-log JSON payload coverage on Modal volume."""
    from pathlib import Path

    base = Path("/data/ingest/full_logs")
    result: dict[str, object] = {
        "base_path": str(base),
        "exists": base.exists(),
    }
    if not base.exists():
        return result

    target = base / shard if shard else base
    if shard and not target.exists():
        return {
            **result,
            "target_path": str(target),
            "target_exists": False,
        }

    files = sorted(target.rglob("*.json.gz")) if target.is_dir() else []
    total_files = len(files)
    total_bytes = sum(f.stat().st_size for f in files)

    sample: list[dict[str, object]] = []
    for f in files[: max(0, sample_n)]:
        name = f.name
        log_id = None
        if name.endswith(".json.gz"):
            raw = name[: -len(".json.gz")]
            if raw.isdigit():
                log_id = int(raw)
        sample.append(
            {
                "path": str(f),
                "log_id": log_id,
                "size_kb": round(f.stat().st_size / 1024, 2),
            }
        )

    shard_counts: list[dict[str, int]] = []
    if not shard:
        for sub in sorted(p for p in base.iterdir() if p.is_dir()):
            n = sum(1 for _ in sub.glob("*.json.gz"))
            if n > 0:
                shard_counts.append({"shard": sub.name, "count": n})

    result.update(
        {
            "target_path": str(target),
            "shard": shard or None,
            "total_json_gz_files": total_files,
            "total_size_gb": round(total_bytes / 1024 / 1024 / 1024, 3),
            "sample": sample,
            "shard_count": len(shard_counts),
            "top_shards": sorted(shard_counts, key=lambda x: x["count"], reverse=True)[:20],
        }
    )
    return result


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=600,
    cpu=2,
)
def backup_db_snapshot(reason: str = "manual", retain: int = DEFAULT_DB_BACKUP_RETAIN) -> dict:
    """Create a point-in-time DB backup on Modal volume and prune old backups."""
    import json
    import shutil
    import sqlite3
    from datetime import UTC, datetime
    from pathlib import Path

    db_path = Path("/data/ingest/terminal_ingest.db")
    wal_path = Path("/data/ingest/terminal_ingest.db-wal")
    shm_path = Path("/data/ingest/terminal_ingest.db-shm")
    backup_root = Path("/data/ingest/db_backups")
    backup_root.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        return {"status": "error", "message": f"DB not found: {db_path}"}

    checkpoint_error = None
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
        conn.close()
    except Exception as e:
        checkpoint_error = str(e)

    safe_reason = "".join(ch if (ch.isalnum() or ch in "._-") else "-" for ch in reason).strip("-")
    if not safe_reason:
        safe_reason = "manual"

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = backup_root / f"{ts}_{safe_reason}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for src, name in (
        (db_path, "terminal_ingest.db"),
        (wal_path, "terminal_ingest.db-wal"),
        (shm_path, "terminal_ingest.db-shm"),
    ):
        if src.exists():
            dst = backup_dir / name
            shutil.copy2(src, dst)
            copied.append(str(dst))

    manifest = {
        "timestamp": ts,
        "reason": reason,
        "safe_reason": safe_reason,
        "checkpoint_error": checkpoint_error,
        "files": copied,
    }
    (backup_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, separators=(",", ":")))

    removed: list[str] = []
    keep = max(1, retain)
    all_dirs = sorted([p for p in backup_root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for old in all_dirs[keep:]:
        shutil.rmtree(old, ignore_errors=True)
        removed.append(str(old))

    volume.commit()
    total_bytes = sum(Path(p).stat().st_size for p in copied if Path(p).exists())
    return {
        "status": "ok",
        "backup_dir": str(backup_dir),
        "files_copied": copied,
        "size_mb": round(total_bytes / 1024 / 1024, 2),
        "removed_old_backups": removed,
        "checkpoint_error": checkpoint_error,
    }


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=600,
    cpu=2,
)
def list_db_backups(limit: int = 30) -> dict:
    """List DB backup snapshots stored on Modal volume."""
    import json
    from pathlib import Path

    backup_root = Path("/data/ingest/db_backups")
    if not backup_root.exists():
        return {"backup_root": str(backup_root), "count": 0, "backups": []}

    dirs = sorted([p for p in backup_root.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    out = []
    for d in dirs[: max(1, limit)]:
        files = list(d.iterdir())
        size = sum(f.stat().st_size for f in files if f.is_file())
        manifest_path = d / "manifest.json"
        manifest = {}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except Exception:
                manifest = {}
        out.append(
            {
                "name": d.name,
                "path": str(d),
                "size_mb": round(size / 1024 / 1024, 2),
                "files": sorted([f.name for f in files if f.is_file()]),
                "reason": manifest.get("reason"),
            }
        )
    return {"backup_root": str(backup_root), "count": len(dirs), "backups": out}


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=900,
    cpu=2,
)
def restore_db_snapshot(
    backup_name: str = "",
    auto_pre_backup: bool = True,
    require_nonempty: bool = True,
) -> dict:
    """Restore live DB from a backup snapshot with safety checks and rollback."""
    import json
    import shutil
    import sqlite3
    from datetime import UTC, datetime
    from pathlib import Path

    db_path = Path("/data/ingest/terminal_ingest.db")
    wal_path = Path("/data/ingest/terminal_ingest.db-wal")
    shm_path = Path("/data/ingest/terminal_ingest.db-shm")
    backup_root = Path("/data/ingest/db_backups")

    def _inspect_db(path: Path) -> dict:
        out: dict[str, object] = {
            "path": str(path),
            "exists": path.exists(),
            "size_mb": round(path.stat().st_size / 1024 / 1024, 2) if path.exists() else 0.0,
            "integrity_check": "missing",
            "table_count": 0,
            "tables": [],
        }
        if not path.exists():
            return out
        conn = None
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            row = conn.execute("PRAGMA integrity_check;").fetchone()
            out["integrity_check"] = row[0] if row and row[0] is not None else "unknown"
            names = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            out["table_count"] = len(names)
            out["tables"] = [n[0] for n in names]
        except Exception as e:
            out["integrity_check"] = f"error: {e}"
        finally:
            if conn is not None:
                conn.close()
        return out

    if not backup_root.exists():
        return {"status": "error", "message": f"Backup root not found: {backup_root}"}

    backup_dirs = sorted(
        [p for p in backup_root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    if not backup_dirs:
        return {"status": "error", "message": f"No backups found in: {backup_root}"}

    selected = None
    if backup_name:
        selected = next((p for p in backup_dirs if p.name == backup_name), None)
        if selected is None:
            return {
                "status": "error",
                "message": f"Backup not found: {backup_name}",
                "available_latest": [p.name for p in backup_dirs[:10]],
            }
    else:
        selected = backup_dirs[0]

    src_db = selected / "terminal_ingest.db"
    if not src_db.exists():
        return {
            "status": "error",
            "message": f"Backup does not include terminal_ingest.db: {selected}",
        }

    source_info = _inspect_db(src_db)
    if source_info.get("integrity_check") != "ok":
        return {
            "status": "error",
            "message": "Refusing restore: backup DB integrity is not ok",
            "backup_name": selected.name,
            "source_info": source_info,
        }
    if require_nonempty and int(source_info.get("table_count", 0) or 0) == 0:
        return {
            "status": "error",
            "message": "Refusing restore: backup DB has zero tables",
            "backup_name": selected.name,
            "source_info": source_info,
        }

    pre_restore_backup_dir = None
    checkpoint_error = None
    if auto_pre_backup and db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
            conn.close()
        except Exception as e:
            checkpoint_error = str(e)
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        pre_restore_backup_dir = backup_root / f"{ts}_auto-pre-restore"
        pre_restore_backup_dir.mkdir(parents=True, exist_ok=True)
        copied = []
        for src, name in (
            (db_path, "terminal_ingest.db"),
            (wal_path, "terminal_ingest.db-wal"),
            (shm_path, "terminal_ingest.db-shm"),
        ):
            if src.exists():
                dst = pre_restore_backup_dir / name
                shutil.copy2(src, dst)
                copied.append(str(dst))
        (pre_restore_backup_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "timestamp": ts,
                    "reason": "auto-pre-restore",
                    "source_backup": selected.name,
                    "files": copied,
                    "checkpoint_error": checkpoint_error,
                },
                ensure_ascii=True,
                separators=(",", ":"),
            )
        )

    tmp_restore = db_path.parent / "terminal_ingest.restore.tmp.db"
    if tmp_restore.exists():
        tmp_restore.unlink()
    shutil.copy2(src_db, tmp_restore)

    tmp_info = _inspect_db(tmp_restore)
    if tmp_info.get("integrity_check") != "ok":
        try:
            tmp_restore.unlink()
        except Exception:
            pass
        return {
            "status": "error",
            "message": "Refusing restore: temp copy failed integrity check",
            "backup_name": selected.name,
            "temp_info": tmp_info,
        }

    if wal_path.exists():
        wal_path.unlink()
    if shm_path.exists():
        shm_path.unlink()
    shutil.move(str(tmp_restore), str(db_path))

    final_info = _inspect_db(db_path)
    final_ok = final_info.get("integrity_check") == "ok"
    final_nonempty = int(final_info.get("table_count", 0) or 0) > 0
    if (not final_ok) or (require_nonempty and not final_nonempty):
        rolled_back = False
        rollback_info = None
        if pre_restore_backup_dir is not None:
            prev_db = pre_restore_backup_dir / "terminal_ingest.db"
            if prev_db.exists():
                shutil.copy2(prev_db, db_path)
                if wal_path.exists():
                    wal_path.unlink()
                if shm_path.exists():
                    shm_path.unlink()
                rolled_back = True
                rollback_info = _inspect_db(db_path)
        volume.commit()
        return {
            "status": "error",
            "message": "Restore validation failed after promote",
            "backup_name": selected.name,
            "final_info": final_info,
            "rolled_back": rolled_back,
            "rollback_info": rollback_info,
            "pre_restore_backup_dir": str(pre_restore_backup_dir) if pre_restore_backup_dir else None,
        }

    _write_stats_snapshot()
    volume.commit()
    return {
        "status": "ok",
        "message": "Restore completed",
        "restored_from_backup": selected.name,
        "source_info": source_info,
        "final_info": final_info,
        "pre_restore_backup_dir": str(pre_restore_backup_dir) if pre_restore_backup_dir else None,
        "checkpoint_error": checkpoint_error,
    }


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=7200,
    cpu=2,
)
def rebuild_db_from_full_logs(
    reset_db: bool = True,
    limit: int = -1,
) -> dict:
    """Rebuild ingest tables from full_logs JSON.gz files without API requests."""
    import asyncio
    import gzip
    import hashlib
    import json
    import shutil
    import sqlite3
    from datetime import UTC, datetime
    from pathlib import Path

    from pipelines.ingest.db import IngestDatabase
    from pipelines.ingest.full_log_parser import parse_full_log

    db_path = Path("/data/ingest/terminal_ingest.db")
    wal_path = Path("/data/ingest/terminal_ingest.db-wal")
    shm_path = Path("/data/ingest/terminal_ingest.db-shm")
    raw_dir = Path("/data/ingest/full_logs")
    backup_dir = Path("/data/ingest/rebuild_backups")
    backup_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        return {"status": "error", "message": f"Missing payload dir: {raw_dir}"}

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if reset_db and db_path.exists():
        shutil.copy2(db_path, backup_dir / f"terminal_ingest.pre_rebuild.{ts}.db")
        if wal_path.exists():
            shutil.copy2(wal_path, backup_dir / f"terminal_ingest.pre_rebuild.{ts}.db-wal")
        if shm_path.exists():
            shutil.copy2(shm_path, backup_dir / f"terminal_ingest.pre_rebuild.{ts}.db-shm")
        db_path.unlink()
        if wal_path.exists():
            wal_path.unlink()
        if shm_path.exists():
            shm_path.unlink()
    elif not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)

    async def _init_schema() -> None:
        db = IngestDatabase(db_path)
        await db.connect()
        await db.init_schema()
        await db.close()

    asyncio.run(_init_schema())

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=OFF;")

    now_iso = datetime.now(UTC).isoformat()

    vault_sql = """
        INSERT INTO vaults (
            vault_address, owner_address, nft_id, nft_name, persona_json,
            trade_size, trading_activity, holding_style, diversification,
            asset_risk_preference, max_trade_amount, slippage_bps, paused, state,
            leaderboard_rank, total_pnl_usd, realized_pnl_usd, unrealized_pnl_usd,
            created_block, updated_block, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(vault_address) DO UPDATE SET
            owner_address=excluded.owner_address,
            nft_id=excluded.nft_id,
            persona_json=excluded.persona_json,
            trade_size=excluded.trade_size,
            trading_activity=excluded.trading_activity,
            holding_style=excluded.holding_style,
            diversification=excluded.diversification,
            asset_risk_preference=excluded.asset_risk_preference,
            max_trade_amount=excluded.max_trade_amount,
            slippage_bps=excluded.slippage_bps,
            paused=excluded.paused,
            state=excluded.state,
            fetched_at=excluded.fetched_at
    """

    strategy_sql = """
        INSERT INTO strategies (
            vault_address, strategy_id, vault_owner_address, content, expiry,
            enabled, strategy_priority, created_block, updated_block, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(vault_address, strategy_id) DO UPDATE SET
            vault_owner_address=excluded.vault_owner_address,
            content=excluded.content,
            expiry=excluded.expiry,
            enabled=excluded.enabled,
            strategy_priority=excluded.strategy_priority,
            created_block=excluded.created_block,
            updated_block=excluded.updated_block,
            fetched_at=excluded.fetched_at
    """

    inference_sql = """
        INSERT INTO inference_logs (
            id, cursor, vault_address, request_id, execution_key, tool, tool_args_json,
            strategy_id, status, inference_duration_ms, error, transaction_hash,
            created_at, completed_at, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            cursor=excluded.cursor,
            vault_address=excluded.vault_address,
            request_id=excluded.request_id,
            execution_key=excluded.execution_key,
            tool=excluded.tool,
            tool_args_json=excluded.tool_args_json,
            strategy_id=excluded.strategy_id,
            status=excluded.status,
            inference_duration_ms=excluded.inference_duration_ms,
            error=excluded.error,
            transaction_hash=excluded.transaction_hash,
            created_at=excluded.created_at,
            completed_at=excluded.completed_at,
            fetched_at=excluded.fetched_at
    """

    full_log_sql = """
        INSERT INTO full_logs (
            log_id, vault_address, payload_path, payload_sha256, payload_size_bytes,
            prompt_text, completion_text, reasoning_content, tool_calls_json,
            llm_model, prompt_tokens, completion_tokens, reasoning_tokens, total_tokens,
            parse_error, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(log_id) DO UPDATE SET
            vault_address=excluded.vault_address,
            payload_path=excluded.payload_path,
            payload_sha256=excluded.payload_sha256,
            payload_size_bytes=excluded.payload_size_bytes,
            prompt_text=excluded.prompt_text,
            completion_text=excluded.completion_text,
            reasoning_content=excluded.reasoning_content,
            tool_calls_json=excluded.tool_calls_json,
            llm_model=excluded.llm_model,
            prompt_tokens=excluded.prompt_tokens,
            completion_tokens=excluded.completion_tokens,
            reasoning_tokens=excluded.reasoning_tokens,
            total_tokens=excluded.total_tokens,
            parse_error=excluded.parse_error,
            fetched_at=excluded.fetched_at
    """

    files = sorted(raw_dir.rglob("*.json.gz"))
    if limit >= 0:
        files = files[:limit]

    vault_rows: dict[str, tuple] = {}
    strategy_rows: dict[tuple[str, str], tuple] = {}
    inference_rows: list[tuple] = []
    full_rows: list[tuple] = []

    processed = 0
    parse_failures = 0
    BATCH = 500

    def _flush() -> None:
        if vault_rows:
            conn.executemany(vault_sql, list(vault_rows.values()))
            vault_rows.clear()
        if strategy_rows:
            conn.executemany(strategy_sql, list(strategy_rows.values()))
            strategy_rows.clear()
        if inference_rows:
            conn.executemany(inference_sql, inference_rows)
            inference_rows.clear()
        if full_rows:
            conn.executemany(full_log_sql, full_rows)
            full_rows.clear()
        conn.commit()

    for path in files:
        processed += 1
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                parse_failures += 1
                continue

            log_id_raw = payload.get("id")
            if isinstance(log_id_raw, int):
                log_id = log_id_raw
            else:
                stem = path.name[:-len(".json.gz")] if path.name.endswith(".json.gz") else path.stem
                if not stem.isdigit():
                    parse_failures += 1
                    continue
                log_id = int(stem)

            vault_address = payload.get("vault_address")
            if not isinstance(vault_address, str) or not vault_address:
                snapshot = payload.get("snapshot")
                agent = snapshot.get("Agent", {}) if isinstance(snapshot, dict) else {}
                vault_address = agent.get("VaultAddress") if isinstance(agent, dict) else None
            if not isinstance(vault_address, str) or not vault_address:
                parse_failures += 1
                continue

            snapshot = payload.get("snapshot", {})
            agent = snapshot.get("Agent", {}) if isinstance(snapshot, dict) else {}
            options = agent.get("Options", {}) if isinstance(agent, dict) else {}
            persona = agent.get("Persona")
            owner = agent.get("OwnerAddress")
            nft_id = agent.get("CurrentNftId")
            state = agent.get("State")
            paused = agent.get("Paused")

            def _opt(*names: str):
                for n in names:
                    if isinstance(options, dict) and n in options:
                        return options.get(n)
                return None

            vault_rows[vault_address] = (
                vault_address,
                owner,
                str(nft_id) if nft_id is not None else None,
                None,
                json.dumps(persona, ensure_ascii=True) if persona is not None else None,
                _opt("tradeSize", "TradeSize"),
                _opt("tradingActivity", "TradingActivity"),
                _opt("holdingStyle", "HoldingStyle"),
                _opt("diversification", "Diversification"),
                _opt("assetRiskPreference", "AssetRiskPreference"),
                _opt("maxTradeAmount", "MaxTradeAmount"),
                _opt("slippageBps", "SlippageBps"),
                1 if paused else 0 if paused is not None else None,
                state,
                None, None, None, None, None, None,
                now_iso,
            )

            strategies = agent.get("Strategies") if isinstance(agent, dict) else None
            if isinstance(strategies, list):
                for idx, strategy in enumerate(strategies):
                    if not isinstance(strategy, dict):
                        continue
                    strategy_id = strategy.get("strategyId") or strategy.get("id") or f"recovered_{idx}"
                    sid = str(strategy_id)
                    strategy_rows[(vault_address, sid)] = (
                        vault_address,
                        sid,
                        owner,
                        strategy.get("content"),
                        strategy.get("expiry"),
                        1 if strategy.get("enabled") else 0 if strategy.get("enabled") is not None else None,
                        strategy.get("strategyPriority"),
                        strategy.get("createdBlock"),
                        strategy.get("updatedBlock"),
                        now_iso,
                    )

            tool_args = payload.get("tool_args")
            inference_rows.append(
                (
                    log_id,
                    payload.get("cursor"),
                    vault_address,
                    payload.get("request_id"),
                    payload.get("execution_key"),
                    payload.get("tool"),
                    json.dumps(tool_args, ensure_ascii=True, separators=(",", ":")) if tool_args is not None else None,
                    payload.get("strategy_id") or payload.get("strategyId"),
                    payload.get("status"),
                    payload.get("inference_duration_ms"),
                    payload.get("error"),
                    payload.get("transaction_hash") or payload.get("transactionHash"),
                    payload.get("created_at"),
                    payload.get("completed_at"),
                    now_iso,
                )
            )

            parsed = parse_full_log(payload, include_reasoning=True)
            raw_bytes = path.read_bytes()
            full_rows.append(
                (
                    log_id,
                    vault_address,
                    str(path),
                    hashlib.sha256(raw_bytes).hexdigest(),
                    len(raw_bytes),
                    parsed.prompt_text,
                    parsed.completion_text,
                    parsed.reasoning_content,
                    parsed.tool_calls_json,
                    parsed.llm_model,
                    parsed.prompt_tokens,
                    parsed.completion_tokens,
                    parsed.reasoning_tokens,
                    parsed.total_tokens,
                    parsed.parse_error,
                    now_iso,
                )
            )
        except Exception:
            parse_failures += 1

        if processed % BATCH == 0:
            _flush()
            print(f"  processed={processed} parse_failures={parse_failures}")

    _flush()
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.close()

    _write_stats_snapshot()
    volume.commit()

    summary_conn = sqlite3.connect(str(db_path))
    counts: dict[str, int] = {}
    for table in ("vaults", "strategies", "inference_logs", "full_logs"):
        row = summary_conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()
        counts[table] = int(row[0]) if row else 0
    summary_conn.close()

    return {
        "status": "ok",
        "message": "Rebuilt from full_logs files",
        "files_seen": len(files),
        "processed": processed,
        "parse_failures": parse_failures,
        "counts": counts,
    }


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
    inspect_path: str = "ingest/terminal_ingest.db",
    inspect_shard: str = "",
    inspect_sample_n: int = 8,
    rebuild_from_files_limit: int = -1,
    rebuild_from_files_no_reset: bool = False,
    backup_reason: str = "manual",
    backup_list_limit: int = 30,
    restore_backup_name: str = "",
    restore_skip_pre_backup: bool = False,
    restore_allow_empty: bool = False,
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

    elif mode == "repair-db":
        result = repair_db.remote()
        print(f"\nRepair result: {result}")

    elif mode == "inspect-db":
        result = inspect_db_path.remote(path=inspect_path)
        print(f"\nInspect result: {result}")

    elif mode == "inspect-full-logs":
        result = inspect_full_logs.remote(shard=inspect_shard, sample_n=inspect_sample_n)
        print(f"\nInspect result: {result}")

    elif mode == "rebuild-from-files":
        result = rebuild_db_from_full_logs.remote(
            reset_db=not rebuild_from_files_no_reset,
            limit=rebuild_from_files_limit,
        )
        print(f"\nRebuild result: {result}")

    elif mode == "backup-db":
        result = backup_db_snapshot.remote(reason=backup_reason, retain=DEFAULT_DB_BACKUP_RETAIN)
        print(f"\nBackup result: {result}")

    elif mode == "list-db-backups":
        result = list_db_backups.remote(limit=backup_list_limit)
        print(f"\nBackup list: {result}")

    elif mode == "restore-db":
        result = restore_db_snapshot.remote(
            backup_name=restore_backup_name,
            auto_pre_backup=not restore_skip_pre_backup,
            require_nonempty=not restore_allow_empty,
        )
        print(f"\nRestore result: {result}")

    else:
        print(f"Unknown mode: {mode}. Use 'ingest', 'prep', 'outcomes', 'snapshot', "
              "'backfill-payloads', 'repair-db', 'inspect-db', "
              "'inspect-full-logs', 'rebuild-from-files', 'backup-db', "
              "'list-db-backups', or 'restore-db'.")
