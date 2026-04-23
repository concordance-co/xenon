from __future__ import annotations

import argparse
import re

from pipelines.db import connect_neon, ensure_schema


DEFAULT_DATASET_RELATION = "workflow_dataset_conflict_probe_v3_v1"
DEFAULT_OUTPUT_TABLE = "capture_outputs_conflict_probe_v3"
DEFAULT_VIEW_NAME = "workflow_dataset_conflict_probe_v3_conflict_readout_side_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a conflict-only publication view labeled by readout-side resolution."
    )
    parser.add_argument("--run-id", required=True, help="Capture run id to source generated outputs from.")
    parser.add_argument("--view-name", default=DEFAULT_VIEW_NAME)
    parser.add_argument("--dataset-relation", default=DEFAULT_DATASET_RELATION)
    parser.add_argument("--output-table", default=DEFAULT_OUTPUT_TABLE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError(f"Unsafe run_id: {args.run_id!r}")

    view_name = args.view_name
    dataset_relation = args.dataset_relation
    output_table = args.output_table

    sql = f"""
    DROP VIEW IF EXISTS {view_name};
    CREATE VIEW {view_name} AS
    WITH joined AS (
        SELECT
            d.log_id,
            d.example_id,
            d.matched_pair_id,
            d.workflow_row_key,
            d.workflow_prompt_hash,
            d.prompt_messages_json,
            d.strategy_family,
            d.strategy_variant_id,
            d.setting_lexical_family_id,
            d.environment_pressure_bucket,
            d.context_variant_id,
            d.conflict_present,
            d.strategy_expected_action,
            d.strategy_expected_size,
            d.setting_expected_action,
            d.setting_expected_size,
            o.generated_text::jsonb AS generated_json
        FROM {dataset_relation} d
        JOIN {output_table} o
          ON o.row_key = d.workflow_row_key
        WHERE o.run_id = '{args.run_id}'
          AND d.conflict_present = TRUE
    ),
    labeled AS (
        SELECT
            *,
            CASE
                WHEN strategy_family IN ('activity_force_trade', 'activity_force_observe') THEN
                    CASE
                        WHEN generated_json ->> 'action' = strategy_expected_action
                         AND generated_json ->> 'action' <> setting_expected_action THEN 'strategy'
                        WHEN generated_json ->> 'action' = setting_expected_action
                         AND generated_json ->> 'action' <> strategy_expected_action THEN 'setting'
                        ELSE NULL
                    END
                WHEN strategy_family IN ('trade_size_force_large', 'trade_size_force_small') THEN
                    CASE
                        WHEN generated_json ->> 'size' = strategy_expected_size
                         AND generated_json ->> 'size' <> setting_expected_size THEN 'strategy'
                        WHEN generated_json ->> 'size' = setting_expected_size
                         AND generated_json ->> 'size' <> strategy_expected_size THEN 'setting'
                        ELSE NULL
                    END
                ELSE NULL
            END AS workflow_label
        FROM joined
    )
    SELECT
        log_id,
        example_id,
        matched_pair_id,
        workflow_row_key,
        workflow_prompt_hash,
        prompt_messages_json,
        strategy_family,
        strategy_variant_id,
        setting_lexical_family_id,
        environment_pressure_bucket,
        context_variant_id,
        workflow_label,
        (
            strategy_family
            || '::' || strategy_variant_id
            || '::' || setting_lexical_family_id
            || '::' || environment_pressure_bucket
        ) AS arbitration_group_id
    FROM labeled
    WHERE workflow_label IN ('strategy', 'setting')
    ORDER BY log_id
    """

    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        conn.execute(sql)
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {view_name}").fetchone()
        count = int(row["n"])

    print(
        {
            "view_name": view_name,
            "run_id": args.run_id,
            "rows": count,
            "dataset_relation": dataset_relation,
            "output_table": output_table,
        }
    )


if __name__ == "__main__":
    main()
