from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipelines.db import connect_neon


DEFAULT_OUTPUT = Path(
    "projects/DX_TERMINAL/prompt_confusion/phase_02/outputs/conflict_probe_v1_behavioral_audit_manifest.json"
)


def load_pairs(conn: Any, *, table_name: str, pairs_per_family: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        f"""
        SELECT *
        FROM {table_name}
        WHERE conflict_severity_bucket IN ('aligned', 'strong_conflict')
        ORDER BY strategy_family, md5(base_context_id || ':' || strategy_variant_id::text), conflict_severity_bucket
        """,
    ).fetchall()

    grouped: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["base_context_id"]),
            str(row["strategy_family"]),
            int(row["strategy_variant_id"]),
        )
        payload = grouped.setdefault(
            key,
            {
                "base_context_id": key[0],
                "strategy_family": key[1],
                "strategy_variant_id": key[2],
                "lexical_split": str(row["lexical_split"]),
            },
        )
        payload[str(row["conflict_severity_bucket"])] = str(row["example_id"])

    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    for payload in grouped.values():
        if "aligned" not in payload or "strong_conflict" not in payload:
            continue
        family = str(payload["strategy_family"])
        if family_counts.get(family, 0) >= pairs_per_family:
            continue
        selected.append(payload)
        family_counts[family] = family_counts.get(family, 0) + 1
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small aligned-vs-strong audit manifest for conflict_probe_v1.")
    parser.add_argument("--table-name", default="conflict_probe_examples_v1")
    parser.add_argument("--pairs-per-family", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with connect_neon() as conn:
        pairs = load_pairs(conn, table_name=args.table_name, pairs_per_family=args.pairs_per_family)

    family_counts: dict[str, int] = {}
    for pair in pairs:
        family = str(pair["strategy_family"])
        family_counts[family] = family_counts.get(family, 0) + 1

    payload = {
        "table_name": args.table_name,
        "pairs_per_family": args.pairs_per_family,
        "pair_count": len(pairs),
        "family_counts": family_counts,
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
