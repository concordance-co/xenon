from __future__ import annotations

import argparse
import json
from typing import Any

from pipelines.db import connect_neon, ensure_schema


DEFAULT_OUTPUT_TABLE = "capture_outputs_conflict_probe_v2"
DEFAULT_DATASET_RELATION = "workflow_dataset_conflict_probe_v2_v1"
ALLOWED_ACTIONS = {"buy", "sell", "observe"}
ALLOWED_ASSETS = {"ALPHA", "BETA", "DELTA", "GAMMA", "NONE"}
ALLOWED_SIZES = {"small", "medium", "large", "none"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label phase 03 generated outputs in-place in Neon."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-table", default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--dataset-relation", default=DEFAULT_DATASET_RELATION)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview summary counts without updating rows.",
    )
    return parser.parse_args()


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def classify_generated_output(
    generated_text: str,
    *,
    strategy_family: str,
    expected_output: dict[str, Any],
    strategy_expected: dict[str, Any],
    setting_expected: dict[str, Any],
) -> dict[str, Any]:
    parsed = _parse_json_object(generated_text)
    valid_output = False
    action_label = "invalid"
    exact_expected = False
    behavior_side = "neither"
    readout_side = "neither"

    if parsed is not None and set(parsed.keys()) == {"action", "asset", "size"}:
        action = parsed.get("action")
        asset = parsed.get("asset")
        size = parsed.get("size")
        if (
            isinstance(action, str)
            and isinstance(asset, str)
            and isinstance(size, str)
            and action in ALLOWED_ACTIONS
            and asset in ALLOWED_ASSETS
            and size in ALLOWED_SIZES
        ):
            valid_output = True
            action_label = action
            generated_triplet = {
                "action": action,
                "asset": asset,
                "size": size,
            }
            exact_expected = generated_triplet == expected_output
            matches_strategy = generated_triplet == strategy_expected
            matches_setting = generated_triplet == setting_expected
            if matches_strategy and matches_setting:
                behavior_side = "both"
            elif matches_strategy:
                behavior_side = "strategy"
            elif matches_setting:
                behavior_side = "setting"

            readout_field = {
                "trade_size_force_large": "size",
                "trade_size_force_small": "size",
                "activity_force_trade": "action",
                "activity_force_observe": "action",
                "diversification_force_concentrate": "asset",
                "holding_force_exit": "action",
            }.get(strategy_family)
            if readout_field is not None:
                matches_strategy_readout = generated_triplet[readout_field] == strategy_expected[readout_field]
                matches_setting_readout = generated_triplet[readout_field] == setting_expected[readout_field]
                if matches_strategy_readout and matches_setting_readout:
                    readout_side = "both"
                elif matches_strategy_readout:
                    readout_side = "strategy"
                elif matches_setting_readout:
                    readout_side = "setting"

    return {
        "valid_output": valid_output,
        "behavior_side": behavior_side,
        "readout_side": readout_side,
        "action_label": action_label,
        "exact_expected": exact_expected,
    }


def ensure_label_columns(conn: Any, table_name: str) -> None:
    for ddl in (
        f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS valid_output BOOLEAN",
        f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS behavior_side TEXT",
        f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS readout_side TEXT",
        f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS action_label TEXT",
        f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS exact_expected BOOLEAN",
    ):
        conn.execute(ddl)


def load_rows(conn: Any, *, output_table: str, dataset_relation: str, run_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT
            o.run_id,
            o.log_id,
            o.generated_text,
            d.strategy_family,
            d.expected_output_json,
            d.strategy_expected_action,
            d.strategy_expected_asset,
            d.strategy_expected_size,
            d.setting_expected_action,
            d.setting_expected_asset,
            d.setting_expected_size
        FROM {output_table} o
        JOIN {dataset_relation} d
          ON o.row_key = d.workflow_row_key
        WHERE o.run_id = %s
        ORDER BY o.log_id
        """,
        [run_id],
    ).fetchall()
    return [dict(row) for row in rows]


def summarize(labels: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "rows": len(labels),
        "valid_output_true": 0,
        "exact_expected_true": 0,
        "behavior_side_counts": {"strategy": 0, "setting": 0, "both": 0, "neither": 0},
        "readout_side_counts": {"strategy": 0, "setting": 0, "both": 0, "neither": 0},
        "action_label_counts": {"buy": 0, "sell": 0, "observe": 0, "invalid": 0},
    }
    for label in labels:
        if label["valid_output"]:
            summary["valid_output_true"] += 1
        if label["exact_expected"]:
            summary["exact_expected_true"] += 1
        summary["behavior_side_counts"][str(label["behavior_side"])] += 1
        summary["readout_side_counts"][str(label["readout_side"])] += 1
        summary["action_label_counts"][str(label["action_label"])] += 1
    return summary


def main() -> None:
    args = parse_args()
    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        ensure_label_columns(conn, args.output_table)
        rows = load_rows(
            conn,
            output_table=args.output_table,
            dataset_relation=args.dataset_relation,
            run_id=args.run_id,
        )
        labels: list[dict[str, Any]] = []
        for row in rows:
            expected_output = dict(row["expected_output_json"])
            strategy_expected = {
                "action": row["strategy_expected_action"],
                "asset": row["strategy_expected_asset"],
                "size": row["strategy_expected_size"],
            }
            setting_expected = {
                "action": row["setting_expected_action"],
                "asset": row["setting_expected_asset"],
                "size": row["setting_expected_size"],
            }
            label = classify_generated_output(
                str(row.get("generated_text") or ""),
                strategy_family=str(row["strategy_family"]),
                expected_output=expected_output,
                strategy_expected=strategy_expected,
                setting_expected=setting_expected,
            )
            labels.append(
                {
                    "run_id": row["run_id"],
                    "log_id": row["log_id"],
                    **label,
                }
            )

        summary = summarize(labels)
        if args.dry_run:
            print(json.dumps(summary, indent=2))
            return

        for row in labels:
            conn.execute(
                f"""
                UPDATE {args.output_table}
                SET valid_output = %s,
                    behavior_side = %s,
                    readout_side = %s,
                    action_label = %s,
                    exact_expected = %s
                WHERE run_id = %s AND log_id = %s
                """,
                [
                    row["valid_output"],
                    row["behavior_side"],
                    row["readout_side"],
                    row["action_label"],
                    row["exact_expected"],
                    row["run_id"],
                    row["log_id"],
                ],
            )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
