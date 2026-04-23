from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipelines.db import connect_neon, ensure_schema


DEFAULT_OUTPUT_TABLE = "capture_outputs_conflict_probe_v3"
DEFAULT_DATASET_RELATION = "workflow_dataset_conflict_probe_v3_v1"
ALLOWED_ACTIONS = {"buy", "sell", "observe"}
ALLOWED_ASSETS = {"ALPHA", "BETA", "DELTA", "GAMMA", "NONE"}
ALLOWED_SIZES = {"small", "medium", "large", "none"}
READOUT_FIELD_BY_FAMILY = {
    "trade_size_force_large": "size",
    "trade_size_force_small": "size",
    "activity_force_trade": "action",
    "activity_force_observe": "action",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Phase 04 generated outputs and summarize behavioral splits."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-table", default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--dataset-relation", default=DEFAULT_DATASET_RELATION)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def classify_generated_output(
    generated_text: str,
    *,
    strategy_family: str,
    expected_output: dict[str, Any],
    strategy_expected: dict[str, Any],
    setting_expected: dict[str, Any],
) -> dict[str, Any]:
    parsed = _parse_json_object(generated_text)
    result = {
        "valid_output": False,
        "exact_expected": False,
        "behavior_side": "neither",
        "readout_side": "neither",
        "action_label": "invalid",
        "asset_label": "invalid",
        "size_label": "invalid",
    }
    if parsed is None or set(parsed.keys()) != {"action", "asset", "size"}:
        return result

    action = parsed.get("action")
    asset = parsed.get("asset")
    size = parsed.get("size")
    if not (
        isinstance(action, str)
        and isinstance(asset, str)
        and isinstance(size, str)
        and action in ALLOWED_ACTIONS
        and asset in ALLOWED_ASSETS
        and size in ALLOWED_SIZES
    ):
        return result

    generated_triplet = {"action": action, "asset": asset, "size": size}
    matches_strategy = generated_triplet == strategy_expected
    matches_setting = generated_triplet == setting_expected
    if matches_strategy and matches_setting:
        behavior_side = "both"
    elif matches_strategy:
        behavior_side = "strategy"
    elif matches_setting:
        behavior_side = "setting"
    else:
        behavior_side = "neither"

    readout_side = "neither"
    readout_field = READOUT_FIELD_BY_FAMILY.get(strategy_family)
    if readout_field is not None:
        matches_strategy_readout = generated_triplet[readout_field] == strategy_expected[readout_field]
        matches_setting_readout = generated_triplet[readout_field] == setting_expected[readout_field]
        if matches_strategy_readout and matches_setting_readout:
            readout_side = "both"
        elif matches_strategy_readout:
            readout_side = "strategy"
        elif matches_setting_readout:
            readout_side = "setting"

    result.update(
        {
            "valid_output": True,
            "exact_expected": generated_triplet == expected_output,
            "behavior_side": behavior_side,
            "readout_side": readout_side,
            "action_label": action,
            "asset_label": asset,
            "size_label": size,
        }
    )
    return result


def load_rows(
    conn: Any,
    *,
    output_table: str,
    dataset_relation: str,
    run_id: str,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT
            o.run_id,
            o.log_id,
            o.generated_text,
            o.finish_reason,
            o.reasoning_text,
            d.example_id,
            d.conflict_present,
            d.conflict_strength,
            d.strategy_family,
            d.setting_family,
            d.environment_pressure_bucket,
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


def _empty_counts(keys: list[str]) -> dict[str, int]:
    return {key: 0 for key in keys}


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    behavior_keys = ["strategy", "setting", "both", "neither"]
    action_keys = ["buy", "sell", "observe", "invalid"]

    overall = {
        "rows": len(rows),
        "valid_output_true": 0,
        "exact_expected_true": 0,
        "behavior_side_counts": _empty_counts(behavior_keys),
        "readout_side_counts": _empty_counts(behavior_keys),
        "action_label_counts": _empty_counts(action_keys),
        "finish_reason_counts": defaultdict(int),
    }
    by_split: dict[str, dict[str, Any]] = {}
    by_family: dict[str, dict[str, Any]] = {}
    by_family_pressure: dict[str, dict[str, Any]] = {}
    conflict_by_family: dict[str, dict[str, Any]] = {}
    conflict_by_family_pressure: dict[str, dict[str, Any]] = {}

    def _bucket(target: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
        if key not in target:
            target[key] = {
                "rows": 0,
                "valid_output_true": 0,
                "exact_expected_true": 0,
                "behavior_side_counts": _empty_counts(behavior_keys),
                "readout_side_counts": _empty_counts(behavior_keys),
                "action_label_counts": _empty_counts(action_keys),
            }
        return target[key]

    for row in rows:
        split_key = "conflict" if bool(row["conflict_present"]) else "aligned"
        family_key = f"{row['strategy_family']}::{split_key}"
        family_pressure_key = (
            f"{row['strategy_family']}::{split_key}::{row['environment_pressure_bucket']}"
        )

        overall["rows"] += 0
        if row["valid_output"]:
            overall["valid_output_true"] += 1
        if row["exact_expected"]:
            overall["exact_expected_true"] += 1
        overall["behavior_side_counts"][str(row["behavior_side"])] += 1
        overall["readout_side_counts"][str(row["readout_side"])] += 1
        overall["action_label_counts"][str(row["action_label"])] += 1
        overall["finish_reason_counts"][str(row.get("finish_reason") or "")] += 1

        buckets = [
            _bucket(by_split, split_key),
            _bucket(by_family, family_key),
            _bucket(by_family_pressure, family_pressure_key),
        ]
        if split_key == "conflict":
            buckets.extend(
                [
                    _bucket(conflict_by_family, str(row["strategy_family"])),
                    _bucket(
                        conflict_by_family_pressure,
                        f"{row['strategy_family']}::{row['environment_pressure_bucket']}",
                    ),
                ]
            )

        for bucket in buckets:
            bucket["rows"] += 1
            if row["valid_output"]:
                bucket["valid_output_true"] += 1
            if row["exact_expected"]:
                bucket["exact_expected_true"] += 1
            bucket["behavior_side_counts"][str(row["behavior_side"])] += 1
            bucket["readout_side_counts"][str(row["readout_side"])] += 1
            bucket["action_label_counts"][str(row["action_label"])] += 1

    def _finalize(bucket: dict[str, Any]) -> dict[str, Any]:
        rows_n = int(bucket["rows"])
        return {
            **bucket,
            "valid_output_rate": _rate(int(bucket["valid_output_true"]), rows_n),
            "exact_expected_rate": _rate(int(bucket["exact_expected_true"]), rows_n),
            "strategy_follow_rate": _rate(int(bucket["behavior_side_counts"]["strategy"]), rows_n),
            "setting_follow_rate": _rate(int(bucket["behavior_side_counts"]["setting"]), rows_n),
            "readout_strategy_rate": _rate(int(bucket["readout_side_counts"]["strategy"]), rows_n),
            "readout_setting_rate": _rate(int(bucket["readout_side_counts"]["setting"]), rows_n),
        }

    return {
        "overall": _finalize(overall),
        "by_split": {key: _finalize(value) for key, value in sorted(by_split.items())},
        "by_family": {key: _finalize(value) for key, value in sorted(by_family.items())},
        "by_family_pressure": {
            key: _finalize(value) for key, value in sorted(by_family_pressure.items())
        },
        "conflict_by_family": {
            key: _finalize(value) for key, value in sorted(conflict_by_family.items())
        },
        "conflict_by_family_pressure": {
            key: _finalize(value) for key, value in sorted(conflict_by_family_pressure.items())
        },
    }


def main() -> None:
    args = parse_args()
    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        raw_rows = load_rows(
            conn,
            output_table=args.output_table,
            dataset_relation=args.dataset_relation,
            run_id=args.run_id,
        )

    labeled_rows: list[dict[str, Any]] = []
    for row in raw_rows:
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
        labeled_rows.append({**row, **label})

    summary = {
        "run_id": args.run_id,
        "output_table": args.output_table,
        "dataset_relation": args.dataset_relation,
        "summary": build_summary(labeled_rows),
        "sample_rows": [
            {
                "log_id": row["log_id"],
                "example_id": row["example_id"],
                "strategy_family": row["strategy_family"],
                "setting_family": row["setting_family"],
                "conflict_present": row["conflict_present"],
                "strategy_expected_action": row["strategy_expected_action"],
                "setting_expected_action": row["setting_expected_action"],
                "behavior_side": row["behavior_side"],
                "readout_side": row["readout_side"],
                "valid_output": row["valid_output"],
                "generated_text": row["generated_text"],
            }
            for row in labeled_rows[:24]
        ],
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
