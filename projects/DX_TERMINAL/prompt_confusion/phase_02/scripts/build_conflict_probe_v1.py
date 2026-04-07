from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipelines.db import connect_neon
from projects.DX_TERMINAL.counterfactual.core import build_settings_edited_variant
from projects.DX_TERMINAL.research_rerun.core import (
    ACTIVE_STRATEGIES_HEADER,
    get_section_body,
    replace_section_body,
)


BASE_CONTEXTS = 24
DEST_TABLE = "conflict_probe_examples_v1"
OUTPUT_DIR = Path("projects/DX_TERMINAL/prompt_confusion/phase_02/outputs/conflict_probe_v1_dataset")

SLIDER_COLUMNS = [
    "trade_size",
    "trading_activity",
    "holding_style",
    "diversification",
    "risk_preference",
]

PROMPT_SLIDER_NAMES = {
    "trade_size": "Trade Size",
    "trading_activity": "Trading Activity",
    "holding_style": "Holding Style",
    "diversification": "Diversification",
    "risk_preference": "Asset Risk Preference",
}

MARKET_HEADER = "## MARKET SNAPSHOT"
ACTIVE_SETTINGS_HEADER = "## ACTIVE SETTINGS"
PORTFOLIO_HEADER = "## PORTFOLIO CONTEXT"
KEPT_SECTION_HEADERS = (
    MARKET_HEADER,
    ACTIVE_STRATEGIES_HEADER,
    ACTIVE_SETTINGS_HEADER,
    PORTFOLIO_HEADER,
)

SOURCE_SLIDER_COLUMNS = {
    "trade_size": "vault_trade_size",
    "trading_activity": "vault_trading_activity",
    "holding_style": "vault_holding_style",
    "diversification": "vault_diversification",
    "risk_preference": "vault_risk_preference",
}

CONFIG_SLIDER_KEYS = {
    "trade_size": "trade_size",
    "trading_activity": "trading_activity",
    "holding_style": "holding_style",
    "diversification": "diversification",
    "risk_preference": "asset_risk_preference",
}

SEVERITY_BUCKETS = {
    "aligned": 0,
    "edge_conflict": 1,
    "strong_conflict": 2,
}

LEXICAL_SPLIT_BY_VARIANT = {
    0: "train",
    1: "train",
    2: "val",
    3: "test",
}

TABLE_COLUMNS = [
    "example_id",
    "source_log_id",
    "source_example_id",
    "base_context_id",
    "block_id",
    "source_label_quality",
    "strategy_key",
    "strategy_family",
    "strategy_variant_id",
    "lexical_split",
    "strategy_text",
    "strategy_priority",
    "conflicting_slider",
    "swept_slider",
    "swept_value",
    "trade_size",
    "trading_activity",
    "holding_style",
    "diversification",
    "risk_preference",
    "conflict",
    "conflict_strength",
    "conflict_severity_bucket",
    "system_text",
    "user_text",
    "prompt_messages_json",
    "config_snapshot_json",
    "market_snapshot_json",
    "portfolio_snapshot_json",
    "strategy_snapshot_json",
    "market_context_hash",
    "built_at",
]


@dataclass(frozen=True)
class FamilySpec:
    family: str
    conflicting_slider: str
    aligned_values: tuple[int, ...]
    edge_values: tuple[int, ...]
    strong_values: tuple[int, ...]
    paraphrases: tuple[str, ...]


FAMILY_SPECS: tuple[FamilySpec, ...] = (
    FamilySpec(
        family="trade_size_force_large",
        conflicting_slider="trade_size",
        aligned_values=(4, 5),
        edge_values=(3,),
        strong_values=(1, 2),
        paraphrases=(
            "Deploy in size on the best setup. If you trade, make it a large position rather than a tester.",
            "When an opportunity is good enough, commit meaningful capital immediately instead of scaling in small.",
            "Favor a heavy entry on the strongest idea. Avoid nibbling with tiny clips.",
            "Do not treat good setups as half-measures. Size up aggressively when you decide to act.",
        ),
    ),
    FamilySpec(
        family="trade_size_force_small",
        conflicting_slider="trade_size",
        aligned_values=(1, 2),
        edge_values=(3,),
        strong_values=(4, 5),
        paraphrases=(
            "Keep every trade small and controlled. Use probing size rather than a big swing.",
            "Scale in with restraint. Prefer small clips over chunky entries.",
            "Stay light on risk per trade. A cautious starter size is the default.",
            "Treat positions as nibbles, not commitments. Avoid large deployments on any single tick.",
        ),
    ),
    FamilySpec(
        family="activity_force_trade",
        conflicting_slider="trading_activity",
        aligned_values=(4, 5),
        edge_values=(3,),
        strong_values=(1, 2),
        paraphrases=(
            "Stay active this tick. If there is any reasonable edge, take a trade instead of sitting out.",
            "Bias toward action. Fresh setups should be traded rather than watched.",
            "Keep capital moving when you have a plausible entry. Do not default to observe.",
            "Be proactive, not passive. Turn decent signals into trades instead of waiting for perfection.",
        ),
    ),
    FamilySpec(
        family="activity_force_observe",
        conflicting_slider="trading_activity",
        aligned_values=(1, 2),
        edge_values=(3,),
        strong_values=(4, 5),
        paraphrases=(
            "Stand down unless the edge is unusually clear. Observing is the default.",
            "Do not force a trade this tick. Wait for a genuinely exceptional setup.",
            "Stay patient and selective. If the signal is merely decent, observe.",
            "Treat this tick as a low-activity one. Prefer no trade over a marginal trade.",
        ),
    ),
    FamilySpec(
        family="holding_force_hold",
        conflicting_slider="holding_style",
        aligned_values=(4, 5),
        edge_values=(3,),
        strong_values=(1, 2),
        paraphrases=(
            "Once in a position, hold through noise and let the thesis play out.",
            "Favor patience after entry. Avoid quick exits unless the thesis is clearly broken.",
            "Strong hands here. Give winners and developing positions time to work.",
            "Treat positions as holds, not flips. Exits should come late, not early.",
        ),
    ),
    FamilySpec(
        family="holding_force_exit",
        conflicting_slider="holding_style",
        aligned_values=(1, 2),
        edge_values=(3,),
        strong_values=(4, 5),
        paraphrases=(
            "Keep holding periods short. If a trade works, take the exit promptly.",
            "Favor quick exits over sitting in positions for a long time.",
            "Treat positions as fast trades rather than long holds.",
            "Do not get married to positions. Lock gains or cut risk early.",
        ),
    ),
    FamilySpec(
        family="diversification_force_concentrate",
        conflicting_slider="diversification",
        aligned_values=(1, 2),
        edge_values=(3,),
        strong_values=(4, 5),
        paraphrases=(
            "Keep the book concentrated in one or two best ideas.",
            "Do not spread into many names. Focus capital into a tight, concentrated book.",
            "Concentrate exposure instead of diversifying broadly across tokens.",
            "Prioritize your top conviction names and avoid a wide portfolio.",
        ),
    ),
    FamilySpec(
        family="diversification_force_spread",
        conflicting_slider="diversification",
        aligned_values=(4, 5),
        edge_values=(3,),
        strong_values=(1, 2),
        paraphrases=(
            "Spread exposure across several names instead of concentrating into one bet.",
            "Diversify the book. Add breadth rather than loading everything into a single token.",
            "Maintain a wider portfolio with multiple positions instead of a narrow one.",
            "Prefer portfolio spread and balance over concentration in only one or two names.",
        ),
    ),
    FamilySpec(
        family="risk_force_safe",
        conflicting_slider="risk_preference",
        aligned_values=(1, 2),
        edge_values=(3,),
        strong_values=(4, 5),
        paraphrases=(
            "Stay in the safest available names. Favor lower-volatility tokens over high-beta punts.",
            "Keep risk conservative. Prefer stable, lower-vol names when you deploy capital.",
            "Do not chase the wildest movers. Lean toward safer assets.",
            "Treat capital preservation as the priority and choose the safer token set.",
        ),
    ),
    FamilySpec(
        family="risk_force_degen",
        conflicting_slider="risk_preference",
        aligned_values=(4, 5),
        edge_values=(3,),
        strong_values=(1, 2),
        paraphrases=(
            "Lean into the highest-beta names and embrace volatility for upside.",
            "Favor the more explosive tokens rather than the safest options.",
            "Take the aggressive route. Prioritize volatile upside over safety.",
            "Seek the degen trade, not the conservative one. Higher volatility is acceptable here.",
        ),
    ),
)


def stable_index(token: str, size: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % size


def market_context_hash(row: dict[str, Any]) -> str:
    payload = "||".join(
        [
            str(row.get("market_snapshot_json") or ""),
            str(row.get("portfolio_snapshot_json") or ""),
        ]
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def sanitize_example_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9:_-]+", "_", value)


def strategy_priority(base_context_id: str, family: str, variant_id: int) -> str:
    token = f"{base_context_id}|{family}|{variant_id}"
    return "high" if stable_index(token, 2) == 0 else "medium"


def pick_swept_value(base_context_id: str, family: str, variant_id: int, bucket: str, values: tuple[int, ...]) -> int:
    token = f"{base_context_id}|{family}|{variant_id}|{bucket}"
    return values[stable_index(token, len(values))]


def build_strategy_body(priority: str, strategy_text: str) -> str:
    return f"- [{priority}] {strategy_text}"


def build_trimmed_user_prompt(user_text: str) -> str:
    eol = "\r\n" if "\r\n" in user_text else "\n"
    separator = f"{eol}------------------------------{eol}{eol}"
    sections = []
    for header in KEPT_SECTION_HEADERS:
        body = get_section_body(user_text, header)
        sections.append(f"{header}{eol}{eol}{body.strip()}")
    return separator.join(sections)


def rewrite_prompt(
    source_row: dict[str, Any],
    *,
    priority: str,
    strategy_text: str,
    slider_values: dict[str, int],
) -> tuple[str, str]:
    user_text = build_trimmed_user_prompt(source_row["user_text"])
    user_text = replace_section_body(
        source_row["user_text"],
        ACTIVE_STRATEGIES_HEADER,
        build_strategy_body(priority, strategy_text),
    )
    user_text = build_trimmed_user_prompt(user_text)
    user_text = build_settings_edited_variant(
        user_text,
        {
            PROMPT_SLIDER_NAMES[key]: value
            for key, value in slider_values.items()
        },
    )
    messages = json.dumps(
        [
            {"role": "system", "content": source_row["system_text"]},
            {"role": "user", "content": user_text},
        ],
        separators=(",", ":"),
    )
    return user_text, messages


def rewrite_config_snapshot(config_snapshot_json: str, slider_values: dict[str, int]) -> str:
    payload = json.loads(config_snapshot_json or "{}")
    for slider, value in slider_values.items():
        payload[CONFIG_SLIDER_KEYS[slider]] = int(value)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def load_source_contexts(conn: Any, *, base_contexts: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            example_id,
            log_id,
            label_quality,
            system_text,
            user_text,
            prompt_messages_json,
            config_snapshot_json,
            market_snapshot_json,
            portfolio_snapshot_json,
            strategy_snapshot_json,
            vault_trade_size,
            vault_trading_activity,
            vault_holding_style,
            vault_diversification,
            vault_risk_preference
        FROM interp_examples_v0
        WHERE prompt_messages_json IS NOT NULL
          AND system_text IS NOT NULL
          AND user_text IS NOT NULL
          AND config_snapshot_json IS NOT NULL
          AND market_snapshot_json IS NOT NULL
          AND portfolio_snapshot_json IS NOT NULL
          AND label_quality IN ('high', 'medium')
          AND POSITION('## ACTIVE STRATEGIES' IN user_text) > 0
          AND POSITION('## ACTIVE SETTINGS' IN user_text) > 0
        ORDER BY
          CASE label_quality WHEN 'high' THEN 0 ELSE 1 END,
          md5(example_id || ':conflict_probe_v1')
        LIMIT 2000
        """
    ).fetchall()

    selected: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for row in rows:
        item = dict(row)
        item["market_context_hash"] = market_context_hash(item)
        if item["market_context_hash"] in seen_hashes:
            continue
        seen_hashes.add(item["market_context_hash"])
        selected.append(item)
        if len(selected) >= base_contexts:
            break

    if len(selected) < base_contexts:
        raise RuntimeError(f"Expected at least {base_contexts} distinct source contexts, found {len(selected)}")
    return selected


def generate_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    built_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []

    for source_row in source_rows:
        base_context_id = str(source_row["example_id"])
        base_sliders = {
            slider: int(source_row[SOURCE_SLIDER_COLUMNS[slider]])
            for slider in SLIDER_COLUMNS
        }
        for family_spec in FAMILY_SPECS:
            for variant_id, strategy_text in enumerate(family_spec.paraphrases):
                priority = strategy_priority(base_context_id, family_spec.family, variant_id)
                block_id = f"{base_context_id}:{family_spec.family}:v{variant_id}"
                lexical_split = LEXICAL_SPLIT_BY_VARIANT[variant_id]
                bucket_values = {
                    "aligned": family_spec.aligned_values,
                    "edge_conflict": family_spec.edge_values,
                    "strong_conflict": family_spec.strong_values,
                }
                for bucket, strength in SEVERITY_BUCKETS.items():
                    row_sliders = dict(base_sliders)
                    swept_value = pick_swept_value(
                        base_context_id,
                        family_spec.family,
                        variant_id,
                        bucket,
                        bucket_values[bucket],
                    )
                    row_sliders[family_spec.conflicting_slider] = swept_value
                    user_text, prompt_messages_json = rewrite_prompt(
                        source_row,
                        priority=priority,
                        strategy_text=strategy_text,
                        slider_values=row_sliders,
                    )
                    config_snapshot_json = rewrite_config_snapshot(
                        source_row["config_snapshot_json"],
                        row_sliders,
                    )
                    strategy_snapshot_json = json.dumps(
                        [
                            {
                                "priority": priority,
                                "content": strategy_text,
                                "strategy_family": family_spec.family,
                                "strategy_variant_id": variant_id,
                            }
                        ],
                        separators=(",", ":"),
                    )
                    example_id = sanitize_example_id(
                        f"cpv1:{base_context_id}:{family_spec.family}:v{variant_id}:s{strength}"
                    )
                    rows.append(
                        {
                            "example_id": example_id,
                            "source_log_id": int(source_row["log_id"]),
                            "source_example_id": str(source_row["example_id"]),
                            "base_context_id": base_context_id,
                            "block_id": block_id,
                            "source_label_quality": str(source_row["label_quality"]),
                            "strategy_key": family_spec.family,
                            "strategy_family": family_spec.family,
                            "strategy_variant_id": variant_id,
                            "lexical_split": lexical_split,
                            "strategy_text": strategy_text,
                            "strategy_priority": priority,
                            "conflicting_slider": family_spec.conflicting_slider,
                            "swept_slider": family_spec.conflicting_slider,
                            "swept_value": swept_value,
                            "trade_size": row_sliders["trade_size"],
                            "trading_activity": row_sliders["trading_activity"],
                            "holding_style": row_sliders["holding_style"],
                            "diversification": row_sliders["diversification"],
                            "risk_preference": row_sliders["risk_preference"],
                            "conflict": bucket != "aligned",
                            "conflict_strength": strength,
                            "conflict_severity_bucket": bucket,
                            "system_text": source_row["system_text"],
                            "user_text": user_text,
                            "prompt_messages_json": prompt_messages_json,
                            "config_snapshot_json": config_snapshot_json,
                            "market_snapshot_json": source_row["market_snapshot_json"],
                            "portfolio_snapshot_json": source_row["portfolio_snapshot_json"],
                            "strategy_snapshot_json": strategy_snapshot_json,
                            "market_context_hash": source_row["market_context_hash"],
                            "built_at": built_at,
                        }
                    )
    return rows


def validate_rows(rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_rows = len(source_rows) * len(FAMILY_SPECS) * 4 * 3
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, found {len(rows)}")

    example_ids = [row["example_id"] for row in rows]
    if len(example_ids) != len(set(example_ids)):
        raise RuntimeError("Duplicate example_id detected in generated rows")

    source_by_context = {str(row["example_id"]): row for row in source_rows}
    block_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    lexical_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    conflict_counts: Counter[bool] = Counter()

    for row in rows:
        block_counts[row["block_id"]] += 1
        severity_counts[row["conflict_severity_bucket"]] += 1
        lexical_counts[row["lexical_split"]] += 1
        family_counts[row["strategy_family"]] += 1
        conflict_counts[bool(row["conflict"])] += 1

        source_row = source_by_context[row["base_context_id"]]
        for slider in SLIDER_COLUMNS:
            expected = int(source_row[SOURCE_SLIDER_COLUMNS[slider]])
            actual = int(row[slider])
            if slider == row["conflicting_slider"]:
                continue
            if actual != expected:
                raise RuntimeError(
                    f"Non-target slider drift for {row['example_id']} on {slider}: expected {expected}, found {actual}"
                )

    bad_blocks = [block_id for block_id, count in block_counts.items() if count != 3]
    if bad_blocks:
        raise RuntimeError(f"Expected every block to have 3 rows, found mismatches for {len(bad_blocks)} blocks")

    expected_per_bucket = expected_rows // 3
    for bucket in SEVERITY_BUCKETS:
        if severity_counts[bucket] != expected_per_bucket:
            raise RuntimeError(f"Expected {expected_per_bucket} rows for {bucket}, found {severity_counts[bucket]}")

    if conflict_counts[False] != expected_per_bucket:
        raise RuntimeError("Aligned row count is not balanced")
    if conflict_counts[True] != expected_per_bucket * 2:
        raise RuntimeError("Conflict row count is not balanced")

    return {
        "row_count": len(rows),
        "expected_row_count": expected_rows,
        "base_context_count": len(source_rows),
        "family_count": len(FAMILY_SPECS),
        "paraphrases_per_family": 4,
        "severity_counts": dict(severity_counts),
        "lexical_split_counts": dict(lexical_counts),
        "family_counts": dict(family_counts),
        "conflict_counts": {
            "aligned": conflict_counts[False],
            "conflict": conflict_counts[True],
        },
    }


def ensure_table(conn: Any, table_name: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            example_id TEXT PRIMARY KEY,
            source_log_id BIGINT NOT NULL,
            source_example_id TEXT NOT NULL,
            base_context_id TEXT NOT NULL,
            block_id TEXT NOT NULL,
            source_label_quality TEXT NOT NULL,
            strategy_key TEXT NOT NULL,
            strategy_family TEXT NOT NULL,
            strategy_variant_id INT NOT NULL,
            lexical_split TEXT NOT NULL,
            strategy_text TEXT NOT NULL,
            strategy_priority TEXT NOT NULL,
            conflicting_slider TEXT NOT NULL,
            swept_slider TEXT NOT NULL,
            swept_value INT NOT NULL,
            trade_size INT NOT NULL,
            trading_activity INT NOT NULL,
            holding_style INT NOT NULL,
            diversification INT NOT NULL,
            risk_preference INT NOT NULL,
            conflict BOOLEAN NOT NULL,
            conflict_strength INT NOT NULL,
            conflict_severity_bucket TEXT NOT NULL,
            system_text TEXT NOT NULL,
            user_text TEXT NOT NULL,
            prompt_messages_json TEXT NOT NULL,
            config_snapshot_json TEXT NOT NULL,
            market_snapshot_json TEXT,
            portfolio_snapshot_json TEXT,
            strategy_snapshot_json TEXT,
            market_context_hash TEXT NOT NULL,
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
            copy.write_row(tuple(row[column] for column in TABLE_COLUMNS))


def write_bookkeeping(output_dir: Path, source_rows: list[dict[str, Any]], summary: dict[str, Any], table_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_contexts = [
        {
            "source_example_id": str(row["example_id"]),
            "source_log_id": int(row["log_id"]),
            "label_quality": str(row["label_quality"]),
            "market_context_hash": row["market_context_hash"],
            "sliders": {
                slider: int(row[SOURCE_SLIDER_COLUMNS[slider]])
                for slider in SLIDER_COLUMNS
            },
        }
        for row in source_rows
    ]
    summary_payload = {
        **summary,
        "dest_table": table_name,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "acceptance_checks": {
            "rows_exactly_2880": summary["row_count"] == 2880,
            "base_contexts_exactly_24": summary["base_context_count"] == 24,
            "families_exactly_10": summary["family_count"] == 10,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")
    (output_dir / "seed_contexts.json").write_text(json.dumps(seed_contexts, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and upload the prompt_confusion phase_02 v1 dataset.")
    parser.add_argument("--base-contexts", type=int, default=BASE_CONTEXTS)
    parser.add_argument("--dest-table", default=DEST_TABLE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.base_contexts != BASE_CONTEXTS:
        raise RuntimeError("phase_02 currently expects exactly 24 base contexts")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.dest_table):
        raise RuntimeError(f"Invalid destination table name: {args.dest_table}")

    with connect_neon(autocommit=False) as conn:
        source_rows = load_source_contexts(conn, base_contexts=args.base_contexts)
        rows = generate_rows(source_rows)
        summary = validate_rows(rows, source_rows)
        if not args.no_upload:
            upload_rows(conn, args.dest_table, rows)
            conn.commit()
        else:
            conn.rollback()

    write_bookkeeping(args.output_dir, source_rows, summary, args.dest_table)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
