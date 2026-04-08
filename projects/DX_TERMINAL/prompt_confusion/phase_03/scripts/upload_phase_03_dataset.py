from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipelines.db import connect_neon, ensure_schema


DEFAULT_INPUT = Path("projects/DX_TERMINAL/prompt_confusion/phase_03/outputs/phase_03_dataset/phase_03_dataset.jsonl")
DEFAULT_TABLE = "conflict_probe_examples_v2"

TABLE_COLUMNS = [
    "example_id",
    "matched_pair_id",
    "pair_member",
    "strategy_family",
    "strategy_variant_id",
    "setting_lexical_family_id",
    "setting_family",
    "setting_variant_id",
    "setting_value",
    "setting_bucket",
    "conflict_present",
    "conflict_strength",
    "environment_pressure_bucket",
    "context_family",
    "context_variant_id",
    "portfolio_state_family",
    "portfolio_variant_id",
    "lexical_split",
    "strategy_lexical_split",
    "setting_lexical_split",
    "system_text",
    "user_text",
    "prompt_messages_json",
    "strategy_snapshot_json",
    "settings_snapshot_json",
    "portfolio_snapshot_json",
    "market_snapshot_json",
    "market_expected_action",
    "market_expected_asset",
    "strategy_expected_action",
    "strategy_expected_asset",
    "strategy_expected_size",
    "setting_expected_action",
    "setting_expected_asset",
    "setting_expected_size",
    "expected_output_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload the prompt_confusion phase_03 dataset into Neon.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dest-table", default=DEFAULT_TABLE)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Dataset file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    if not rows:
        raise SystemExit(f"No rows loaded from {path}")
    return rows


def ensure_table(conn: Any, table_name: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            example_id TEXT PRIMARY KEY,
            matched_pair_id TEXT NOT NULL,
            pair_member TEXT NOT NULL,
            strategy_family TEXT NOT NULL,
            strategy_variant_id TEXT NOT NULL,
            setting_lexical_family_id TEXT NOT NULL,
            setting_family TEXT NOT NULL,
            setting_variant_id TEXT NOT NULL,
            setting_value INT NOT NULL,
            setting_bucket TEXT NOT NULL,
            conflict_present BOOLEAN NOT NULL,
            conflict_strength INT NOT NULL,
            environment_pressure_bucket TEXT NOT NULL,
            context_family TEXT NOT NULL,
            context_variant_id TEXT NOT NULL,
            portfolio_state_family TEXT NOT NULL,
            portfolio_variant_id TEXT NOT NULL,
            lexical_split TEXT NOT NULL,
            strategy_lexical_split TEXT NOT NULL,
            setting_lexical_split TEXT NOT NULL,
            system_text TEXT NOT NULL,
            user_text TEXT NOT NULL,
            prompt_messages_json JSONB NOT NULL,
            strategy_snapshot_json JSONB NOT NULL,
            settings_snapshot_json JSONB NOT NULL,
            portfolio_snapshot_json JSONB NOT NULL,
            market_snapshot_json JSONB NOT NULL,
            market_expected_action TEXT NOT NULL,
            market_expected_asset TEXT NOT NULL,
            strategy_expected_action TEXT NOT NULL,
            strategy_expected_asset TEXT NOT NULL,
            strategy_expected_size TEXT NOT NULL,
            setting_expected_action TEXT NOT NULL,
            setting_expected_asset TEXT NOT NULL,
            setting_expected_size TEXT NOT NULL,
            expected_output_json JSONB NOT NULL,
            built_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    conn.execute(f"TRUNCATE TABLE {table_name}")


def upload_rows(conn: Any, table_name: str, rows: list[dict[str, Any]]) -> None:
    ensure_table(conn, table_name)
    column_list = ", ".join(TABLE_COLUMNS)
    with conn.cursor().copy(f"COPY {table_name} ({column_list}) FROM STDIN") as copy:
        for row in rows:
            copy.write_row(
                tuple(
                    json.dumps(row[column], sort_keys=True) if isinstance(row[column], (dict, list)) else row[column]
                    for column in TABLE_COLUMNS
                )
            )


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        upload_rows(conn, args.dest_table, rows)
        row_count = conn.execute(f"SELECT COUNT(*) AS n FROM {args.dest_table}").fetchone()["n"]
    print(json.dumps({"dest_table": args.dest_table, "row_count": row_count}, indent=2))


if __name__ == "__main__":
    main()
