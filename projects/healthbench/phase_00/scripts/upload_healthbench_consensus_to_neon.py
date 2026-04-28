from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from collections.abc import Iterable
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


DEFAULT_SOURCE_URL = (
    "https://openaipublic.blob.core.windows.net/simple-evals/healthbench/"
    "consensus_2025-05-09-20-00-46.jsonl"
)
DEFAULT_TABLE = "healthbench_consensus_v1"
DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_]*$")

TABLE_COLUMNS = [
    "prompt_id",
    "split_name",
    "source_url",
    "prompt_messages_json",
    "example_tags_json",
    "rubric_items_json",
    "rubric_count",
    "prompt_turn_count",
    "prompt_roles_json",
    "ideal_completions_data_json",
    "source_row_json",
    "source_row_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload HealthBench Consensus metadata into Neon."
    )
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--dest-table", default=DEFAULT_TABLE)
    parser.add_argument("--database-url-env", default=DB_ENV_VAR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row cap for smoke uploads.",
    )
    return parser.parse_args()


def require_identifier(value: str) -> str:
    if not IDENTIFIER_RE.match(value):
        raise SystemExit(f"Unsafe SQL identifier: {value!r}")
    return value


def iter_source_rows(source_url: str, *, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with urllib.request.urlopen(source_url) as response:
        for idx, raw_line in enumerate(response):
            if limit is not None and idx >= limit:
                break
            line = raw_line.decode("utf-8").strip()
            if line:
                yield json.loads(line)


def normalize_row(row: dict[str, Any], *, source_url: str) -> dict[str, Any]:
    prompt = row.get("prompt")
    rubrics = row.get("rubrics")
    if not isinstance(row.get("prompt_id"), str):
        raise ValueError("HealthBench row is missing string prompt_id.")
    if not isinstance(prompt, list):
        raise ValueError(f"HealthBench row {row['prompt_id']} has non-list prompt.")
    if not isinstance(rubrics, list):
        raise ValueError(f"HealthBench row {row['prompt_id']} has non-list rubrics.")

    raw_canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return {
        "prompt_id": row["prompt_id"],
        "split_name": "consensus",
        "source_url": source_url,
        "prompt_messages_json": Jsonb(prompt),
        "example_tags_json": Jsonb(row.get("example_tags", [])),
        "rubric_items_json": Jsonb(rubrics),
        "rubric_count": len(rubrics),
        "prompt_turn_count": len(prompt),
        "prompt_roles_json": Jsonb([message.get("role") for message in prompt]),
        "ideal_completions_data_json": Jsonb(row.get("ideal_completions_data")),
        "source_row_json": Jsonb(row),
        "source_row_sha256": hashlib.sha256(raw_canonical.encode("utf-8")).hexdigest(),
    }


def ensure_table(conn: psycopg.Connection, table_name: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            prompt_id TEXT PRIMARY KEY,
            split_name TEXT NOT NULL,
            source_url TEXT NOT NULL,
            prompt_messages_json JSONB NOT NULL,
            example_tags_json JSONB NOT NULL,
            rubric_items_json JSONB NOT NULL,
            rubric_count INTEGER NOT NULL,
            prompt_turn_count INTEGER NOT NULL,
            prompt_roles_json JSONB NOT NULL,
            ideal_completions_data_json JSONB,
            source_row_json JSONB NOT NULL,
            source_row_sha256 TEXT NOT NULL,
            uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {table_name}_example_tags_gin "
        f"ON {table_name} USING GIN (example_tags_json)"
    )
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS {table_name}_rubric_items_gin "
        f"ON {table_name} USING GIN (rubric_items_json)"
    )


def upload_rows(conn: psycopg.Connection, table_name: str, rows: list[dict[str, Any]]) -> None:
    ensure_table(conn, table_name)
    conn.execute(f"TRUNCATE TABLE {table_name}")
    placeholders = ", ".join(["%s"] * len(TABLE_COLUMNS))
    columns_sql = ", ".join(TABLE_COLUMNS)
    values = [[row[column] for column in TABLE_COLUMNS] for row in rows]
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

    rows = [
        normalize_row(row, source_url=args.source_url)
        for row in iter_source_rows(args.source_url, limit=args.limit)
    ]
    if not rows:
        raise SystemExit("No HealthBench rows loaded.")

    with psycopg.connect(database_url, autocommit=True, row_factory=dict_row) as conn:
        upload_rows(conn, table_name, rows)

    print(f"Uploaded {len(rows)} HealthBench Consensus rows to {table_name}.")


if __name__ == "__main__":
    main()
