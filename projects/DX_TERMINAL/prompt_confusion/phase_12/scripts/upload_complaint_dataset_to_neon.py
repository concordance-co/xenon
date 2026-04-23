from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

DEFAULT_INPUT = Path(__file__).resolve().parents[3] / "dataset_exports" / "complaint_dataset_enriched.parquet"
DEFAULT_TABLE = "dx_terminal_complaint_dataset_enriched_v1"
REMOTE_INPUT_PATH = "/root/complaint_dataset_enriched.parquet"

TABLE_COLUMNS = [
    "trace_id",
    "person_id",
    "vault_address",
    "label",
    "fault",
    "root_cause",
    "agent_was_correct",
    "severity",
    "confidence",
    "urgency",
    "complaint_text",
    "complaint_type",
    "referenced_tokens",
    "slider_ta",
    "slider_arp",
    "slider_ts",
    "slider_hs",
    "slider_div",
    "has_strategy",
    "strategies_text",
    "n_relevant_ticks",
    "vault_summary_json",
    "agent_activity",
    "evidence_summary",
    "contributing_factors",
    "recommended_fix",
    "ticks",
    "n_ticks_attached",
    "raw_row_json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload the enriched DX Terminal complaint export to Neon.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dest-table", default=DEFAULT_TABLE)
    parser.add_argument("--if-exists", choices=("replace", "append"), default="replace")
    parser.add_argument("--mode", choices=("auto", "local", "modal"), default="auto")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def validate_table_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise SystemExit(f"Unsafe destination table name: {value}")
    return value


def load_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Dataset file not found: {path}")
    table = pq.read_table(path)
    rows = table.to_pylist()
    return rows[:limit] if limit is not None else rows


def _json_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except Exception:
            return value
    return value


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    normalized["referenced_tokens"] = _json_or_none(row.get("referenced_tokens"))
    normalized["contributing_factors"] = _json_or_none(row.get("contributing_factors"))
    normalized["vault_summary_json"] = _json_or_none(row.get("vault_summary_json"))
    normalized["ticks"] = _json_or_none(row.get("ticks"))
    normalized["raw_row_json"] = row
    return normalized


def ensure_table(conn: Any, table_name: str, *, replace: bool) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            trace_id TEXT PRIMARY KEY,
            person_id TEXT,
            vault_address TEXT,
            label TEXT,
            fault TEXT,
            root_cause TEXT,
            agent_was_correct BOOLEAN,
            severity BIGINT,
            confidence DOUBLE PRECISION,
            urgency BIGINT,
            complaint_text TEXT,
            complaint_type TEXT,
            referenced_tokens JSONB,
            slider_ta BIGINT,
            slider_arp BIGINT,
            slider_ts BIGINT,
            slider_hs BIGINT,
            slider_div BIGINT,
            has_strategy BOOLEAN,
            strategies_text TEXT,
            n_relevant_ticks BIGINT,
            vault_summary_json JSONB,
            agent_activity TEXT,
            evidence_summary TEXT,
            contributing_factors JSONB,
            recommended_fix TEXT,
            ticks JSONB,
            n_ticks_attached BIGINT,
            raw_row_json JSONB,
            uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    if replace:
        conn.execute(f"TRUNCATE TABLE {table_name}")


def upload_rows(conn: Any, table_name: str, rows: list[dict[str, Any]], *, replace: bool) -> int:
    ensure_table(conn, table_name, replace=replace)
    column_list = ", ".join(TABLE_COLUMNS)
    with conn.cursor().copy(f"COPY {table_name} ({column_list}) FROM STDIN") as copy:
        for raw_row in rows:
            row = normalize_row(raw_row)
            copy.write_row(
                tuple(
                    json.dumps(row[column], ensure_ascii=False, sort_keys=True)
                    if isinstance(row[column], (dict, list))
                    else row[column]
                    for column in TABLE_COLUMNS
                )
            )
    result = conn.execute(f"SELECT COUNT(*) AS n FROM {table_name}").fetchone()
    return int(result["n"])


def upload_local(input_path: Path, table_name: str, *, replace: bool, limit: int | None = None) -> dict[str, Any]:
    from projects.DX_TERMINAL.prompt_confusion.neon import connect_neon, ensure_schema

    rows = load_rows(input_path, limit=limit)
    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        row_count = upload_rows(conn, table_name, rows, replace=replace)
    return {
        "mode": "local",
        "input_path": str(input_path),
        "dest_table": table_name,
        "input_rows": len(rows),
        "table_rows": row_count,
    }


def upload_via_modal(input_path: Path, table_name: str, *, replace: bool, limit: int | None = None) -> dict[str, Any]:
    import modal

    app = modal.App("dx-terminal-complaint-neon-upload")
    image = (
        modal.Image.debian_slim(python_version="3.13")
        .pip_install("pyarrow", "psycopg[binary]")
        .add_local_file(str(input_path), REMOTE_INPUT_PATH)
    )
    neon_secret = modal.Secret.from_name("xenon-neon")

    @app.function(image=image, secrets=[neon_secret], timeout=60 * 60, serialized=True)
    def upload_remote(dest_table: str, replace_remote: bool, limit_remote: int | None) -> dict[str, Any]:
        import json
        import os
        from pathlib import Path
        from typing import Any

        import psycopg
        import pyarrow.parquet as pq
        from psycopg.rows import dict_row

        table_columns = TABLE_COLUMNS

        def json_or_none(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, str):
                text = value.strip()
                if not text:
                    return None
                try:
                    return json.loads(text)
                except Exception:
                    return value
            return value

        def normalize_remote_row(row: dict[str, Any]) -> dict[str, Any]:
            normalized = dict(row)
            normalized["referenced_tokens"] = json_or_none(row.get("referenced_tokens"))
            normalized["contributing_factors"] = json_or_none(row.get("contributing_factors"))
            normalized["vault_summary_json"] = json_or_none(row.get("vault_summary_json"))
            normalized["ticks"] = json_or_none(row.get("ticks"))
            normalized["raw_row_json"] = row
            return normalized

        def load_remote_rows(path: Path, limit_value: int | None = None) -> list[dict[str, Any]]:
            table = pq.read_table(path)
            rows = table.to_pylist()
            return rows[:limit_value] if limit_value is not None else rows

        def ensure_remote_table(conn: Any, table_name: str, *, replace_value: bool) -> None:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    trace_id TEXT PRIMARY KEY,
                    person_id TEXT,
                    vault_address TEXT,
                    label TEXT,
                    fault TEXT,
                    root_cause TEXT,
                    agent_was_correct BOOLEAN,
                    severity BIGINT,
                    confidence DOUBLE PRECISION,
                    urgency BIGINT,
                    complaint_text TEXT,
                    complaint_type TEXT,
                    referenced_tokens JSONB,
                    slider_ta BIGINT,
                    slider_arp BIGINT,
                    slider_ts BIGINT,
                    slider_hs BIGINT,
                    slider_div BIGINT,
                    has_strategy BOOLEAN,
                    strategies_text TEXT,
                    n_relevant_ticks BIGINT,
                    vault_summary_json JSONB,
                    agent_activity TEXT,
                    evidence_summary TEXT,
                    contributing_factors JSONB,
                    recommended_fix TEXT,
                    ticks JSONB,
                    n_ticks_attached BIGINT,
                    raw_row_json JSONB,
                    uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            if replace_value:
                conn.execute(f"TRUNCATE TABLE {table_name}")

        def upload_remote_rows(conn: Any, table_name: str, rows: list[dict[str, Any]], *, replace_value: bool) -> int:
            ensure_remote_table(conn, table_name, replace_value=replace_value)
            column_list = ", ".join(table_columns)
            with conn.cursor().copy(f"COPY {table_name} ({column_list}) FROM STDIN") as copy:
                for raw_row in rows:
                    row = normalize_remote_row(raw_row)
                    copy.write_row(
                        tuple(
                            json.dumps(row[column], ensure_ascii=False, sort_keys=True)
                            if isinstance(row[column], (dict, list))
                            else row[column]
                            for column in table_columns
                        )
                    )
            result = conn.execute(f"SELECT COUNT(*) AS n FROM {table_name}").fetchone()
            return int(result["n"])

        database_url = os.environ.get("XENON_NEON_DATABASE_URL")
        if not database_url:
            raise RuntimeError("XENON_NEON_DATABASE_URL is not available in the Modal environment.")

        rows = load_remote_rows(Path(REMOTE_INPUT_PATH), limit_value=limit_remote)
        with psycopg.connect(database_url, autocommit=True, row_factory=dict_row) as conn:
            row_count = upload_remote_rows(conn, dest_table, rows, replace_value=replace_remote)

        return {
            "mode": "modal",
            "input_path": REMOTE_INPUT_PATH,
            "dest_table": dest_table,
            "input_rows": len(rows),
            "table_rows": row_count,
        }

    with app.run():
        return upload_remote.remote(table_name, replace, limit)


def main() -> None:
    args = parse_args()
    table_name = validate_table_name(args.dest_table)
    replace = args.if_exists == "replace"

    mode = args.mode
    if mode == "auto":
        mode = "local"
        try:
            result = upload_local(args.input, table_name, replace=replace, limit=args.limit)
        except RuntimeError:
            mode = "modal"
            result = upload_via_modal(args.input, table_name, replace=replace, limit=args.limit)
    elif mode == "local":
        result = upload_local(args.input, table_name, replace=replace, limit=args.limit)
    else:
        result = upload_via_modal(args.input, table_name, replace=replace, limit=args.limit)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
