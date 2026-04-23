from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipelines.db import connect_neon, ensure_schema


DEFAULT_OUTPUT_TABLE = "capture_outputs_conflict_probe_v2"
DEFAULT_DATASET_RELATION = "workflow_dataset_conflict_probe_v2_v1"
DEFAULT_OUTPUT_DIR = Path("projects/DX_TERMINAL/prompt_confusion/phase_03/reports")
DEFAULT_FAMILIES = ("trade_size_force_large", "activity_force_observe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a quality-filtered matched-pair slice for Phase 03 mechanistic analysis."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-table", default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--dataset-relation", default=DEFAULT_DATASET_RELATION)
    parser.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILIES),
        help="Comma-separated strategy families to include.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--require-output-change",
        action="store_true",
        help="Require the aligned and strong-conflict generations to differ.",
    )
    return parser.parse_args()


def parse_family_filter(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def load_rows(
    conn: Any,
    *,
    output_table: str,
    dataset_relation: str,
    run_id: str,
    families: list[str],
) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT
            o.run_id,
            o.log_id AS capture_log_id,
            o.row_key,
            d.strategy_family,
            d.setting_bucket,
            d.matched_pair_id,
            d.example_id,
            d.environment_pressure_bucket,
            d.context_family,
            d.context_variant_id,
            d.portfolio_variant_id,
            d.strategy_variant_id,
            d.setting_variant_id,
            d.user_text,
            d.expected_output_json,
            d.strategy_expected_action,
            d.strategy_expected_asset,
            d.strategy_expected_size,
            d.setting_expected_action,
            d.setting_expected_asset,
            d.setting_expected_size,
            o.generated_text,
            o.valid_output,
            o.exact_expected,
            o.behavior_side,
            o.readout_side,
            o.action_label
        FROM {output_table} o
        JOIN {dataset_relation} d
          ON o.row_key = d.workflow_row_key
        WHERE o.run_id = %s
          AND d.strategy_family = ANY(%s)
          AND d.setting_bucket IN ('aligned', 'strong_conflict')
        ORDER BY d.strategy_family, d.matched_pair_id, d.setting_bucket
        """,
        [run_id, families],
    ).fetchall()
    return [dict(row) for row in rows]


def normalize_output(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, sort_keys=True)


def build_pairs(rows: list[dict[str, Any]], *, require_output_change: bool) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    family_by_pair: dict[str, str] = {}
    for row in rows:
        grouped[str(row["matched_pair_id"])][str(row["setting_bucket"])] = row
        family_by_pair[str(row["matched_pair_id"])] = str(row["strategy_family"])

    pairs: list[dict[str, Any]] = []
    for matched_pair_id, pair_rows in grouped.items():
        if {"aligned", "strong_conflict"} - pair_rows.keys():
            continue
        aligned = pair_rows["aligned"]
        strong = pair_rows["strong_conflict"]
        aligned_text = normalize_output(aligned.get("generated_text") or "")
        strong_text = normalize_output(strong.get("generated_text") or "")
        changed_output = aligned_text != strong_text
        keep = bool(aligned.get("exact_expected")) and bool(strong.get("valid_output"))
        if require_output_change:
            keep = keep and changed_output
        if not keep:
            continue
        pairs.append(
            {
                "strategy_family": family_by_pair[matched_pair_id],
                "matched_pair_id": matched_pair_id,
                "aligned": aligned,
                "strong_conflict": strong,
                "changed_output": changed_output,
            }
        )
    pairs.sort(
        key=lambda pair: (
            str(pair["strategy_family"]),
            0 if pair["strong_conflict"].get("exact_expected") else 1,
            0 if pair["strong_conflict"].get("readout_side") in {"setting", "strategy"} else 1,
            str(pair["matched_pair_id"]),
        )
    )
    return pairs


def summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    per_family: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["strategy_family"])].append(pair)

    for family, family_pairs in grouped.items():
        strong_readout = Counter(str(pair["strong_conflict"].get("readout_side") or "unknown") for pair in family_pairs)
        strong_exact = sum(1 for pair in family_pairs if pair["strong_conflict"].get("exact_expected"))
        changed = sum(1 for pair in family_pairs if pair["changed_output"])
        per_family[family] = {
            "pairs": len(family_pairs),
            "strong_conflict_exact_expected": strong_exact,
            "strong_conflict_readout_counts": dict(strong_readout),
            "changed_output_pairs": changed,
        }

    return {
        "pair_count": len(pairs),
        "family_summaries": per_family,
    }


def render_pair_section(pair: dict[str, Any]) -> str:
    aligned = pair["aligned"]
    strong = pair["strong_conflict"]
    lines = [
        f"### `{pair['matched_pair_id']}`",
        "",
        f"- Family: `{pair['strategy_family']}`",
        f"- Pressure: `{aligned['environment_pressure_bucket']}`",
        f"- Context: `{aligned['context_variant_id']}`",
        f"- Changed output: `{str(pair['changed_output']).lower()}`",
        f"- Strong conflict readout side: `{strong['readout_side']}`",
        f"- Strong conflict exact expected: `{str(bool(strong['exact_expected'])).lower()}`",
        "",
        "**Aligned**",
        "",
        f"- `example_id`: `{aligned['example_id']}`",
        f"- generated: `{normalize_output(aligned['generated_text'])}`",
        f"- expected: `{normalize_output(aligned['expected_output_json'])}`",
        "",
        "**Strong Conflict**",
        "",
        f"- `example_id`: `{strong['example_id']}`",
        f"- generated: `{normalize_output(strong['generated_text'])}`",
        f"- expected: `{normalize_output(strong['expected_output_json'])}`",
        "",
        "**Prompt**",
        "",
        "```text",
        str(aligned["user_text"]).strip(),
        "```",
        "",
    ]
    return "\n".join(lines)


def render_markdown(
    *,
    run_id: str,
    families: list[str],
    summary: dict[str, Any],
    pairs: list[dict[str, Any]],
) -> str:
    lines = [
        "# Phase 03 Mechanistic Slice",
        "",
        f"Run: `{run_id}`",
        f"Families: `{', '.join(families)}`",
        "",
        "## Summary",
        "",
        f"- Filtered matched pairs: `{summary['pair_count']}`",
    ]

    for family in families:
        family_summary = summary["family_summaries"].get(family)
        if family_summary is None:
            continue
        lines.extend(
            [
                f"- `{family}`: `{family_summary['pairs']}` pairs, "
                f"`{family_summary['strong_conflict_exact_expected']}` strong-conflict exact matches, "
                f"readout counts `{json.dumps(family_summary['strong_conflict_readout_counts'], sort_keys=True)}`",
            ]
        )

    for family in families:
        family_pairs = [pair for pair in pairs if pair["strategy_family"] == family]
        if not family_pairs:
            continue
        lines.extend(["", f"## {family}", ""])
        for pair in family_pairs[:8]:
            lines.append(render_pair_section(pair))

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    families = parse_family_filter(args.families)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        rows = load_rows(
            conn,
            output_table=args.output_table,
            dataset_relation=args.dataset_relation,
            run_id=args.run_id,
            families=families,
        )

    pairs = build_pairs(rows, require_output_change=args.require_output_change)
    summary = summarize_pairs(pairs)

    stem = f"mechanistic_slice_{args.run_id}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    json_path.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "families": families,
                "require_output_change": args.require_output_change,
                "summary": summary,
                "pairs": pairs,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    md_path.write_text(
        render_markdown(
            run_id=args.run_id,
            families=families,
            summary=summary,
            pairs=pairs,
        )
    )

    print(
        json.dumps(
            {
                "json_path": str(json_path),
                "md_path": str(md_path),
                "pair_count": summary["pair_count"],
                "family_summaries": summary["family_summaries"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
