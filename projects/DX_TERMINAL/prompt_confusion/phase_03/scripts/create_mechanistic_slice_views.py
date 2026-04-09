from __future__ import annotations

import argparse
import json
from typing import Any

from pipelines.db import connect_neon, ensure_schema


DEFAULT_OUTPUT_TABLE = "capture_outputs_conflict_probe_v2"
DEFAULT_DATASET_RELATION = "workflow_dataset_conflict_probe_v2_v1"
DEFAULT_FAMILIES = ("trade_size_force_large", "activity_force_observe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Neon views for Modal-backed Phase 03 mechanistic slice analysis."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-table", default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--dataset-relation", default=DEFAULT_DATASET_RELATION)
    parser.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated strategy families to include in the slice.",
    )
    parser.add_argument(
        "--prefix",
        default="workflow_dataset_conflict_probe_v2_mechslice",
        help="Prefix for created public views.",
    )
    parser.add_argument(
        "--include-middle",
        action="store_true",
        help="Include middle rows in the conflict view. Source view remains strong-conflict only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview SQL and row counts without creating views.",
    )
    return parser.parse_args()


def _parse_families(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _build_family_sql(families: list[str]) -> str:
    return ", ".join(_quote_literal(family) for family in families)


def _build_base_cte(*, output_table: str, dataset_relation: str, run_id: str, families: list[str]) -> str:
    family_sql = _build_family_sql(families)
    return f"""
WITH base AS (
    SELECT
        d.log_id,
        d.workflow_row_key,
        d.workflow_prompt_hash,
        d.prompt_messages_json,
        d.example_id,
        d.matched_pair_id,
        d.strategy_family,
        d.environment_pressure_bucket,
        d.setting_bucket,
        d.conflict_present,
        d.conflict_strength,
        o.run_id,
        o.generated_text,
        o.valid_output,
        o.exact_expected,
        o.behavior_side,
        o.readout_side,
        o.action_label
    FROM {dataset_relation} d
    JOIN {output_table} o
      ON o.row_key = d.workflow_row_key
    WHERE o.run_id = {_quote_literal(run_id)}
      AND d.strategy_family IN ({family_sql})
)
""".strip()


def _conflict_view_sql(
    *,
    output_table: str,
    dataset_relation: str,
    run_id: str,
    families: list[str],
    include_middle: bool,
) -> str:
    setting_predicate = "('aligned', 'middle', 'strong_conflict')" if include_middle else "('aligned', 'strong_conflict')"
    base_cte = _build_base_cte(
        output_table=output_table,
        dataset_relation=dataset_relation,
        run_id=run_id,
        families=families,
    )
    return f"""
{base_cte}
SELECT
    log_id,
    workflow_row_key,
    workflow_prompt_hash,
    prompt_messages_json,
    example_id,
    matched_pair_id,
    strategy_family,
    environment_pressure_bucket,
    setting_bucket,
    conflict_present,
    conflict_strength,
    generated_text,
    valid_output,
    exact_expected,
    readout_side,
    behavior_side,
    CASE
        WHEN setting_bucket = 'aligned' THEN 'aligned'
        ELSE 'conflict'
    END AS workflow_label
FROM base
WHERE setting_bucket IN {setting_predicate}
""".strip()


def _source_view_sql(
    *,
    output_table: str,
    dataset_relation: str,
    run_id: str,
    families: list[str],
) -> str:
    base_cte = _build_base_cte(
        output_table=output_table,
        dataset_relation=dataset_relation,
        run_id=run_id,
        families=families,
    )
    return f"""
{base_cte}
SELECT
    log_id,
    workflow_row_key,
    workflow_prompt_hash,
    prompt_messages_json,
    example_id,
    matched_pair_id,
    strategy_family,
    environment_pressure_bucket,
    setting_bucket,
    conflict_present,
    conflict_strength,
    generated_text,
    valid_output,
    exact_expected,
    readout_side,
    behavior_side,
    readout_side AS workflow_label
FROM base
WHERE setting_bucket = 'strong_conflict'
  AND readout_side IN ('strategy', 'setting')
""".strip()


def _count_rows(conn: Any, sql: str) -> int:
    wrapped = f"SELECT count(*) AS n FROM ({sql}) slice"
    row = conn.execute(wrapped).fetchone()
    return int(row["n"]) if row else 0


def main() -> None:
    args = parse_args()
    families = _parse_families(args.families)
    conflict_view = f"{args.prefix}_conflict_v1"
    source_view = f"{args.prefix}_source_v1"

    conflict_sql = _conflict_view_sql(
        output_table=args.output_table,
        dataset_relation=args.dataset_relation,
        run_id=args.run_id,
        families=families,
        include_middle=args.include_middle,
    )
    source_sql = _source_view_sql(
        output_table=args.output_table,
        dataset_relation=args.dataset_relation,
        run_id=args.run_id,
        families=families,
    )

    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        conflict_count = _count_rows(conn, conflict_sql)
        source_count = _count_rows(conn, source_sql)

        if not args.dry_run:
            conn.execute(f"CREATE OR REPLACE VIEW {conflict_view} AS {conflict_sql}")
            conn.execute(f"CREATE OR REPLACE VIEW {source_view} AS {source_sql}")

    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "families": families,
                "conflict_view": conflict_view,
                "conflict_rows": conflict_count,
                "source_view": source_view,
                "source_rows": source_count,
                "include_middle": args.include_middle,
                "dry_run": args.dry_run,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
