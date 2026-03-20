#!/usr/bin/env python3
"""Read-only query tool for the Xenon Neon database.

Usage (from repo root):
    uv run python scripts/neon_query.py tables              # list all tables + row counts
    uv run python scripts/neon_query.py schema              # all table schemas
    uv run python scripts/neon_query.py schema vaults       # one table's schema
    uv run python scripts/neon_query.py sample vaults       # 5 sample rows (default)
    uv run python scripts/neon_query.py sample vaults 20    # 20 sample rows
    uv run python scripts/neon_query.py query "SELECT ..."  # run arbitrary read-only SQL
    uv run python scripts/neon_query.py query -f query.sql  # run SQL from a file

All queries are validated read-only before execution. No writes possible.
Set XENON_NEON_DATABASE_URL in your environment or .env file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.db import require_neon_dsn, load_dotenv_if_present

# ---------------------------------------------------------------------------
# Read-only validation (mirrors pipelines/backend/app.py)
# ---------------------------------------------------------------------------

_READ_ONLY_PREFIXES = ("select", "pragma", "explain", "with")
_SQL_FRAGMENT_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|vacuum|reindex|truncate|grant|revoke)\b",
    re.IGNORECASE,
)


def validate_read_only(sql: str) -> str:
    normalized = sql.strip().rstrip(";")
    if not normalized:
        raise ValueError("Empty query")
    lowered = normalized.lower()
    if not lowered.startswith(_READ_ONLY_PREFIXES):
        raise ValueError(
            f"Only read-only queries allowed (SELECT/EXPLAIN/WITH). Got: {lowered[:40]}..."
        )
    if _SQL_FRAGMENT_FORBIDDEN.search(lowered):
        raise ValueError("Query contains mutating keywords — read-only queries only")
    return normalized


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect():
    import psycopg
    from psycopg.rows import dict_row

    load_dotenv_if_present()
    dsn = require_neon_dsn()
    return psycopg.connect(dsn, autocommit=True, row_factory=dict_row)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_tables(conn) -> None:
    rows = conn.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
    ).fetchall()

    print(f"{'Table':<40} {'Rows':>10}")
    print("-" * 52)
    for r in rows:
        name = r["table_name"]
        count = conn.execute(f'SELECT COUNT(*) AS n FROM "{name}"').fetchone()["n"]
        print(f"{name:<40} {count:>10,}")


def cmd_schema(conn, table: str | None) -> None:
    if table:
        _print_table_schema(conn, table)
    else:
        tables = [
            r["table_name"]
            for r in conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """
            ).fetchall()
        ]
        for t in tables:
            _print_table_schema(conn, t)
            print()


def _print_table_schema(conn, table: str) -> None:
    cols = conn.execute(
        """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """,
        [table],
    ).fetchall()

    if not cols:
        print(f"Table '{table}' not found.")
        return

    # Primary key columns
    pk_cols = conn.execute(
        """
        SELECT a.attname
        FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
        WHERE i.indrelid = %s::regclass AND i.indisprimary
        ORDER BY array_position(i.indkey, a.attnum)
        """,
        [table],
    ).fetchall()
    pk_set = {r["attname"] for r in pk_cols}

    print(f"## {table}")
    print(f"{'Column':<40} {'Type':<25} {'Nullable':>8}  PK")
    print("-" * 80)
    for c in cols:
        pk = " *" if c["column_name"] in pk_set else ""
        nullable = "YES" if c["is_nullable"] == "YES" else "NO"
        print(f"{c['column_name']:<40} {c['data_type']:<25} {nullable:>8}{pk}")


def cmd_sample(conn, table: str, n: int) -> None:
    # Validate table name
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", table):
        print(f"Invalid table name: {table}", file=sys.stderr)
        sys.exit(1)

    rows = conn.execute(f'SELECT * FROM "{table}" LIMIT %s', [n]).fetchall()
    if not rows:
        print(f"(no rows in {table})")
        return
    print(json.dumps([dict(r) for r in rows], indent=2, default=str))


def cmd_query(conn, sql: str) -> None:
    validated = validate_read_only(sql)
    rows = conn.execute(validated).fetchall()
    if not rows:
        print("(no results)")
        return
    print(json.dumps([dict(r) for r in rows], indent=2, default=str))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only query tool for Xenon Neon DB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("tables", help="List all tables with row counts")

    p_schema = sub.add_parser("schema", help="Show table schema(s)")
    p_schema.add_argument("table", nargs="?", help="Table name (omit for all)")

    p_sample = sub.add_parser("sample", help="Sample rows from a table")
    p_sample.add_argument("table", help="Table name")
    p_sample.add_argument("n", nargs="?", type=int, default=5, help="Number of rows (default: 5)")

    p_query = sub.add_parser("query", help="Run a read-only SQL query")
    p_query.add_argument("sql", nargs="?", help="SQL string")
    p_query.add_argument("-f", "--file", help="Read SQL from file instead")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    conn = connect()
    try:
        if args.command == "tables":
            cmd_tables(conn)
        elif args.command == "schema":
            cmd_schema(conn, args.table)
        elif args.command == "sample":
            cmd_sample(conn, args.table, args.n)
        elif args.command == "query":
            sql = args.sql
            if args.file:
                sql = Path(args.file).read_text()
            if not sql:
                print("Provide SQL as argument or via -f file", file=sys.stderr)
                sys.exit(1)
            cmd_query(conn, sql)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
