from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipelines.db import connect_neon, ensure_schema


DEFAULT_TABLE = "conflict_probe_examples_v2"

NEW_SYSTEM_TEXT = (
    "You are a trading agent. Choose exactly one action each turn. "
    'Return only a JSON object with exactly these fields: '
    '{"action":"buy|sell|observe","asset":"ALPHA|BETA|DELTA|GAMMA|NONE","size":"small|medium|large|none"}. '
    "Do not return any other keys or any other text. "
    'If no trade should be made, return {"action":"observe","asset":"NONE","size":"none"}.'
)

UPDATED_STRATEGY_TEXTS = {
    "activity_observe_v0": "Capital preservation comes first. Do not trade unless the edge is clearly exceptional.",
    "activity_observe_v1": "Default to no action unless one setup is unusually strong.",
    "activity_observe_v2": "Stay disciplined and avoid forcing trades on marginal signals.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch phase 03 prompt text in Neon without rebuilding the full dataset."
    )
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the number of rows that would change without writing updates.",
    )
    return parser.parse_args()


def _parse_json(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _maybe_patch_row(row: dict[str, Any]) -> dict[str, Any] | None:
    updated = dict(row)
    changed = False

    if updated.get("system_text") != NEW_SYSTEM_TEXT:
        updated["system_text"] = NEW_SYSTEM_TEXT
        changed = True

    strategy_variant_id = str(updated.get("strategy_variant_id") or "")
    new_strategy_text = UPDATED_STRATEGY_TEXTS.get(strategy_variant_id)
    if new_strategy_text:
        strategy_snapshot = dict(_parse_json(updated.get("strategy_snapshot_json")) or {})
        old_strategy_text = str(strategy_snapshot.get("strategy_text") or "")
        if old_strategy_text and old_strategy_text != new_strategy_text:
            strategy_snapshot["strategy_text"] = new_strategy_text
            updated["strategy_snapshot_json"] = strategy_snapshot
            user_text = str(updated.get("user_text") or "")
            if old_strategy_text in user_text:
                updated["user_text"] = user_text.replace(old_strategy_text, new_strategy_text, 1)
            changed = True

    prompt_messages = list(_parse_json(updated.get("prompt_messages_json")) or [])
    if prompt_messages:
        first = dict(prompt_messages[0])
        if first.get("role") == "system" and first.get("content") != NEW_SYSTEM_TEXT:
            first["content"] = NEW_SYSTEM_TEXT
            prompt_messages[0] = first
            changed = True
        if len(prompt_messages) > 1:
            second = dict(prompt_messages[1])
            if second.get("role") == "user":
                expected_user_text = str(updated.get("user_text") or "")
                if second.get("content") != expected_user_text:
                    second["content"] = expected_user_text
                    prompt_messages[1] = second
                    changed = True
        updated["prompt_messages_json"] = prompt_messages

    if not changed:
        return None
    return updated


def main() -> None:
    args = parse_args()
    table_name = args.table

    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        rows = conn.execute(
            f"""
            SELECT example_id, strategy_variant_id, system_text, user_text,
                   prompt_messages_json, strategy_snapshot_json
            FROM {table_name}
            """
        ).fetchall()

        updates: list[dict[str, Any]] = []
        for row in rows:
            patched = _maybe_patch_row(dict(row))
            if patched is not None:
                updates.append(patched)

        if args.dry_run:
            print(json.dumps({"table": table_name, "rows_total": len(rows), "rows_to_update": len(updates)}, indent=2))
            return

        for row in updates:
            conn.execute(
                f"""
                UPDATE {table_name}
                SET system_text = %s,
                    user_text = %s,
                    prompt_messages_json = %s::jsonb,
                    strategy_snapshot_json = %s::jsonb
                WHERE example_id = %s
                """,
                [
                    row["system_text"],
                    row["user_text"],
                    json.dumps(row["prompt_messages_json"], sort_keys=True),
                    json.dumps(row["strategy_snapshot_json"], sort_keys=True),
                    row["example_id"],
                ],
            )

    print(json.dumps({"table": table_name, "rows_updated": len(updates)}, indent=2))


if __name__ == "__main__":
    main()
