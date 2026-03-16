# neon_query.py — Read-Only Neon DB Tool

Query the Xenon Neon database. All queries are validated read-only before execution.

## Setup

1. Make sure `XENON_NEON_DATABASE_URL` is in your `.env` (already there if you've run the pipeline)
2. `uv sync` to install dependencies

## Commands

```bash
# List all tables + row counts
uv run python scripts/neon_query.py tables

# Show schema for all tables (or one specific table)
uv run python scripts/neon_query.py schema
uv run python scripts/neon_query.py schema vaults

# Sample rows from a table (default 5)
uv run python scripts/neon_query.py sample inference_logs
uv run python scripts/neon_query.py sample swaps 20

# Run arbitrary read-only SQL
uv run python scripts/neon_query.py query "SELECT vault_address, total_pnl_usd FROM vaults ORDER BY total_pnl_usd DESC LIMIT 10"

# Run SQL from a file
uv run python scripts/neon_query.py query -f my_query.sql
```

## Output

- `tables` and `schema` output plain text tables
- `sample` and `query` output JSON (easy to paste into AI conversations)

## Safety

Only SELECT/EXPLAIN/WITH queries are allowed. Any mutating SQL (INSERT, UPDATE, DELETE, DROP, etc.) is rejected before it reaches the database.
