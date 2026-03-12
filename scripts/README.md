# Scripts Operational Reference

This document is the operational companion to `ARCHITECTURE.md`.

All commands are run from repo root.

## Script purposes

| Script | Purpose |
|---|---|
| `scripts/modal_capture.sh` | Main wrapper for Modal ingest/prep/outcomes/capture/analysis and DB utilities. |
| `scripts/modal_restore_db.sh` | Focused helper for listing/restoring DB backup snapshots on Modal with safe defaults. |
| `scripts/rebuild_db_from_full_logs_local.py` | Local heavy-lift rebuild from `data/full_logs/*.json.gz` into local SQLite DB. |
| `scripts/xenon_backend.sh` | Wrapper for backend deploy/dev and read-only backend queries. |

## Safety semantics

### Read-only commands

| Command | Reads |
|---|---|
| `./scripts/modal_capture.sh modal-inspect-db [path]` | DB file integrity/table counts on Modal volume. |
| `./scripts/modal_capture.sh modal-inspect-full-logs [shard] [sample_n]` | Full-log file coverage on Modal volume. |
| `./scripts/modal_capture.sh modal-list-db-backups [limit]` | Backup snapshot metadata. |
| `./scripts/modal_capture.sh modal-stats` | Existing `dashboard_stats.json` download only. |
| `./scripts/xenon_backend.sh query|tables|schema|sample|stats|parquet-*|activations|health` | Backend API reads. |

### Mutating commands

| Command | Mutates | Notes |
|---|---|---|
| `./scripts/modal_capture.sh upload-db` | `xenon-data:/ingest/terminal_ingest.db` | Overwrites only live DB file; does not touch backup dirs. |
| `./scripts/modal_capture.sh modal-ingest` | Modal DB tables + cursors + payload files | Idempotent upserts. |
| `./scripts/modal_capture.sh modal-prep` | Prep tables + exports + stats snapshot | Rebuilds prep outputs. |
| `./scripts/modal_capture.sh modal-outcomes` | `trade_outcomes` + stats snapshot | Processes unlabeled swaps only. |
| `./scripts/modal_capture.sh modal-repair-db` | Live DB replacement during salvage | Creates repair backups first. |
| `./scripts/modal_capture.sh modal-rebuild-from-files` | Live DB rebuild from volume full logs | Creates rebuild backups. |
| `./scripts/modal_capture.sh modal-backup-db [reason]` | `ingest/db_backups/*` | Retains 30 snapshots (oldest pruned). |
| `./scripts/modal_capture.sh modal-restore-db [backup_name]` | Live DB restore from backup | Integrity/non-empty guards + auto pre-restore backup + rollback on failed validation. |
| `./scripts/modal_capture.sh modal-snapshot` | Recomputes `dashboard_stats.json` on Modal + downloads local copy | Use when dashboard status is stale. |
| `./scripts/modal_capture.sh backfill-payloads` | Adds/fills `full_logs.payload_gz` | Increases DB size. |
| `python scripts/rebuild_db_from_full_logs_local.py ...` | Local DB file path given by `--db-path` | Fast local rebuild path. |

## Runbooks

### 1) Health check + backup

```bash
./scripts/modal_capture.sh modal-inspect-db ingest/terminal_ingest.db
./scripts/modal_capture.sh modal-backup-db pre-op
./scripts/modal_capture.sh modal-list-db-backups 10
```

### 2) Restore from backup

```bash
./scripts/modal_restore_db.sh --list 20
# Optional explicit snapshot first (restore also performs an automatic pre-restore backup by default)
./scripts/modal_capture.sh modal-backup-db pre-restore-manual
./scripts/modal_restore_db.sh 20260312T172948Z_abort-slow-rebuild
./scripts/modal_capture.sh modal-inspect-db ingest/terminal_ingest.db
```

### 3) Local rebuild + upload

```bash
# Pull logs once if needed
modal volume get xenon-data ingest/full_logs ./data/full_logs --force

# Rebuild local DB from full logs
uv run --extra interp python scripts/rebuild_db_from_full_logs_local.py \
  --input-dir data/full_logs \
  --db-path data/terminal_ingest.db \
  --batch-size 2000

# Verify local DB then upload
sqlite3 data/terminal_ingest.db "PRAGMA integrity_check;"
./scripts/modal_capture.sh modal-backup-db pre-upload-local-rebuild
./scripts/modal_capture.sh upload-db
./scripts/modal_capture.sh modal-inspect-db ingest/terminal_ingest.db
```

### 4) Swaps/outcomes continuation after rebuild

```bash
# Populate/continue outcomes on Modal
./scripts/modal_capture.sh modal-outcomes --outcomes-limit -1 --concurrency 5 --timeout-s 30 --retry-max-attempts 6

# Recompute + verify snapshot-backed status
./scripts/modal_capture.sh modal-snapshot
curl -s "$XENON_BACKEND_URL/stats" | jq '.ingest,.outcomes'
```

### 5) Snapshot refresh verification

```bash
# Recompute snapshot (not just download cached file)
./scripts/modal_capture.sh modal-snapshot

# Local dashboard forced refresh endpoint
curl -s "http://127.0.0.1:8800/api/status?refresh=1" | jq
```

## Troubleshooting quick checks

### Stale status in dashboard

Symptom:
- Dashboard shows old counts after successful run.

Checks/fix:
```bash
./scripts/modal_capture.sh modal-snapshot
curl -s "http://127.0.0.1:8800/api/status?refresh=1" | jq
```

Notes:
- `modal-stats` only downloads existing `dashboard_stats.json`.
- `modal-snapshot` recomputes it first, then downloads.

### `swaps` count seems too high vs inference logs

Reason:
- `swaps` can include rows where `log_id` is missing or not present in current `inference_logs` corpus.

Quick check:
```bash
sqlite3 data/terminal_ingest.db "
SELECT
  CASE
    WHEN s.log_id IS NULL THEN 'no_log_id'
    WHEN EXISTS (SELECT 1 FROM inference_logs l WHERE l.id=s.log_id) THEN 'joins_inference_logs'
    ELSE 'log_id_not_in_inference_logs'
  END AS category,
  COUNT(*) AS n
FROM swaps s
GROUP BY category
ORDER BY n DESC;
"
```

### Outcomes run stopped early

Expected behavior:
- Safe to stop/restart; outcomes resumes from unlabeled swaps only.

Quick check:
```bash
./scripts/modal_capture.sh modal-snapshot
curl -s "$XENON_BACKEND_URL/stats" | jq '.outcomes'
```

## Backup policy

- Canonical backup path: `ingest/db_backups/`
- Created by: `modal-backup-db`
- Retention: 30 snapshots (old snapshots pruned automatically)
