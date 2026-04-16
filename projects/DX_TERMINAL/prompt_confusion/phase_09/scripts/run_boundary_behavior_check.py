from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.DX_TERMINAL.prompt_confusion.phase_09.scripts.run_behavior_smoke import app, generate_rows


DEFAULT_INPUT = Path(
    "projects/DX_TERMINAL/prompt_confusion/phase_09/outputs/phase_09_dataset/phase_09_dataset.jsonl"
)
DEFAULT_OUTPUT = Path(
    "projects/DX_TERMINAL/prompt_confusion/phase_09/reports/boundary_behavior_check.json"
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    if not rows:
        raise SystemExit(f"No rows loaded from {path}")
    return rows


def _boundary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("target_dimension") == "trading_activity"
        and int(row.get("setting_value", -1)) == 1
        and any(tag in str(row.get("context_variant_id", "")) for tag in ("solid", "exceptional"))
    ]
    return sorted(selected, key=lambda row: str(row["example_id"]))


def _strict_parse(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except Exception:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None


def _analyze(rows: list[dict[str, Any]], generated_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": len(rows),
        "strict_valid": 0,
        "extract_valid": 0,
        "strict_exact": 0,
        "extract_exact": 0,
        "finish_reasons": Counter(),
        "by_cell": {},
        "failures": [],
    }
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["strategy_direction"]), str(row["market_snapshot_json"]["evidence_tier"]), int(row["setting_value"]))].append(
            row
        )

    for key, bucket in sorted(grouped.items()):
        cell_counter = Counter()
        cell_failures: list[dict[str, Any]] = []
        for row in bucket:
            output = generated_by_id[str(row["example_id"])]
            text = str(output.get("generated_text") or "")
            finish_reason = str(output.get("finish_reason") or "")
            strict = _strict_parse(text)
            extracted = strict or _extract_first_json_object(text)
            expected = dict(row["expected_output_json"])
            summary["finish_reasons"][finish_reason] += 1
            strict_exact = strict == expected if strict is not None else False
            extract_exact = extracted == expected if extracted is not None else False
            if strict is not None:
                summary["strict_valid"] += 1
                cell_counter["strict_valid"] += 1
            if extracted is not None:
                summary["extract_valid"] += 1
                cell_counter["extract_valid"] += 1
            if strict_exact:
                summary["strict_exact"] += 1
                cell_counter["strict_exact"] += 1
            if extract_exact:
                summary["extract_exact"] += 1
                cell_counter["extract_exact"] += 1
            parsed = extracted
            if parsed is not None:
                cell_counter[f"action:{parsed.get('action')}"] += 1
            else:
                cell_counter["invalid"] += 1
            if extracted is None or not extract_exact:
                cell_failures.append(
                    {
                        "example_id": row["example_id"],
                        "strategy_direction": row["strategy_direction"],
                        "evidence_tier": row["market_snapshot_json"]["evidence_tier"],
                        "expected_output_json": expected,
                        "generated_text": text,
                        "finish_reason": finish_reason,
                        "strict_parsed": strict,
                        "extracted_parsed": extracted,
                    }
                )
        summary["by_cell"]["::".join((key[0], key[1], f"value_{key[2]}"))] = {
            "rows": len(bucket),
            **dict(cell_counter),
        }
        summary["failures"].extend(cell_failures[:8])

    rows_count = max(1, len(rows))
    summary["strict_valid_rate"] = round(summary["strict_valid"] / rows_count, 4)
    summary["extract_valid_rate"] = round(summary["extract_valid"] / rows_count, 4)
    summary["strict_exact_rate"] = round(summary["strict_exact"] / rows_count, 4)
    summary["extract_exact_rate"] = round(summary["extract_exact"] / rows_count, 4)
    summary["finish_reasons"] = dict(summary["finish_reasons"])
    return summary


if __name__ == "__main__":
    rows = _boundary_rows(_load_rows(DEFAULT_INPUT))
    with app.run():
        generations = generate_rows.remote(rows, batch_size=8, max_tokens=256)
    generated_by_id = {str(item["example_id"]): item for item in generations}
    summary = _analyze(rows, generated_by_id)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
