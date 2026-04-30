from __future__ import annotations

import argparse
import os
import re
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from projects.healthbench.shared.ask_vs_answer_smoke import build_smoke_records


DEFAULT_TABLE = "healthbench_ask_vs_answer_smoke_v1"
DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")

TABLE_COLUMNS = [
    "example_id",
    "version",
    "triple_id",
    "condition_id",
    "condition",
    "expected_behavior",
    "axis",
    "condition_index",
    "sample_index",
    "prompt_messages_json",
    "prompt_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the small HealthBench ask-vs-answer smoke set into Neon."
    )
    parser.add_argument("--dest-table", default=DEFAULT_TABLE)
    parser.add_argument("--database-url-env", default=DB_ENV_VAR)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--samples-per-condition", type=int, default=4)
    return parser.parse_args()


def require_identifier(value: str) -> str:
    if not IDENTIFIER_RE.match(value):
        raise SystemExit(f"Unsafe SQL identifier: {value!r}")
    return value


def ensure_table(conn: psycopg.Connection, table_name: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            example_id TEXT PRIMARY KEY,
            version TEXT NOT NULL,
            triple_id TEXT NOT NULL,
            condition_id TEXT NOT NULL,
            condition TEXT NOT NULL,
            expected_behavior TEXT NOT NULL,
            axis TEXT NOT NULL,
            condition_index INTEGER NOT NULL,
            sample_index INTEGER NOT NULL,
            prompt_messages_json JSONB NOT NULL,
            prompt_sha256 TEXT NOT NULL,
            uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {table_name}_condition_idx "
        f"ON {table_name} (condition)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {table_name}_triple_condition_idx "
        f"ON {table_name} (triple_id, condition)"
    )


def upload_rows(conn: psycopg.Connection, table_name: str, rows: list[dict[str, Any]]) -> None:
    ensure_table(conn, table_name)
    conn.execute(f"TRUNCATE TABLE {table_name}")
    placeholders = ", ".join(["%s"] * len(TABLE_COLUMNS))
    columns_sql = ", ".join(TABLE_COLUMNS)
    values = []
    for row in rows:
        values.append(
            [
                Jsonb(row[column]) if column == "prompt_messages_json" else row[column]
                for column in TABLE_COLUMNS
            ]
        )
    with conn.cursor() as cur:
        cur.executemany(
            f"INSERT INTO {table_name} ({columns_sql}) VALUES ({placeholders})",
            values,
        )


def main() -> None:
    args = parse_args()
    table_name = require_identifier(args.dest_table)
    database_url = os.environ.get(args.database_url_env)
    if not database_url:
        raise SystemExit(f"{args.database_url_env} is not set.")

    rows = build_smoke_records(
        version=args.version,
        samples_per_condition=args.samples_per_condition,
    )
    if not rows:
        raise SystemExit("No smoke rows generated.")

    with psycopg.connect(database_url, autocommit=True, row_factory=dict_row) as conn:
        upload_rows(conn, table_name, rows)

    print(f"Uploaded {len(rows)} HealthBench ask-vs-answer smoke rows to {table_name}.")


if __name__ == "__main__":
    main()
