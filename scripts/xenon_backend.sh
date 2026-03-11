#!/usr/bin/env bash
# Xenon backend API — serve, deploy, query
set -euo pipefail

CMD="${1:-help}"
shift || true

case "$CMD" in
  serve)
    echo "Starting dev server (hot-reload)..."
    uv run --extra modal modal serve pipelines/backend/app.py
    ;;
  deploy)
    echo "Deploying backend..."
    uv run --extra modal modal deploy pipelines/backend/app.py
    echo ""
    echo "Set the URL with: export XENON_BACKEND_URL=<url-from-above>"
    echo "Or save it:       echo '<url>' > ~/.xenon_backend_url"
    ;;
  query|q)
    uv run -m pipelines.backend query "$@"
    ;;
  stats)
    uv run -m pipelines.backend stats
    ;;
  schema)
    uv run -m pipelines.backend schema "$@"
    ;;
  tables)
    uv run -m pipelines.backend tables
    ;;
  sample)
    uv run -m pipelines.backend sample "$@"
    ;;
  parquet-list|pql)
    uv run -m pipelines.backend parquet-list
    ;;
  parquet-info|pqi)
    uv run -m pipelines.backend parquet-info "$@"
    ;;
  parquet-sample|pqs)
    uv run -m pipelines.backend parquet-sample "$@"
    ;;
  activations|act)
    uv run -m pipelines.backend activations "$@"
    ;;
  health)
    uv run -m pipelines.backend health
    ;;
  reload)
    uv run -m pipelines.backend reload
    ;;
  *)
    cat <<'USAGE'
Usage: xenon_backend.sh <command> [args]

Server:
  serve                Start dev server with hot-reload
  deploy               Deploy persistent endpoint

Data:
  query|q "SQL"        Run read-only SQL query
  stats                Dashboard stats JSON
  schema [table]       List tables or show table schema
  tables               All tables with row counts
  sample TABLE [N]     Sample N rows from a table

Parquet:
  parquet-list|pql     List parquet files on volume
  parquet-info|pqi     Parquet file metadata
  parquet-sample|pqs   Sample rows from parquet

Activations:
  activations|act      Activation metadata summary

Misc:
  health               Health check
  reload               Force volume data refresh

Examples:
  ./scripts/xenon_backend.sh query "SELECT COUNT(*) FROM vaults"
  ./scripts/xenon_backend.sh sample interp_examples_v0 5
  ./scripts/xenon_backend.sh pqi interp_examples_v0_high_quality.parquet
  ./scripts/xenon_backend.sh pqs interp_sample_trade_v0.parquet 3

Config:
  Set XENON_BACKEND_URL env var or write URL to ~/.xenon_backend_url
USAGE
    ;;
esac
