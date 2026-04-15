"""Build the prompt-confusion Phase 06 dataset.

Size-axis only. Four strategy variants per direction, four setting phrase
families, 50/50 STRATEGY-first / SETTINGS-first section order.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipelines.db import connect_neon, ensure_schema


SYSTEM_TEXT = (
    "You are a trading agent. Each prompt contains two policy sources. "
    "STRATEGY describes the directional plan for this tick. "
    "SETTINGS describe execution policy such as entry frequency and size tier. "
    "You must consider both sources before acting. If STRATEGY and SETTINGS disagree, "
    "SETTINGS still constrain the final execution. "
    "Choose exactly one action each turn. Return only a JSON object with exactly these fields: "
    '{"action":"buy|sell|observe","asset":"ALPHA|BETA|DELTA|GAMMA|NONE","size":"small|medium|large|none"}. '
    "Do not return any other keys or any other text. "
    'If no trade should be made, return {"action":"observe","asset":"NONE","size":"none"}.'
)

OUTPUT_DIR = Path("projects/DX_TERMINAL/prompt_confusion/phase_06/outputs/phase_06_dataset")
NEON_TABLE = "conflict_probe_examples_v4"

TABLE_COLUMNS = [
    "example_id",
    "matched_pair_id",
    "pair_member",
    "strategy_family",
    "strategy_variant_id",
    "setting_lexical_family_id",
    "setting_family",
    "setting_variant_id",
    "setting_value",
    "setting_bucket",
    "conflict_present",
    "conflict_strength",
    "environment_pressure_bucket",
    "section_order",
    "context_family",
    "context_variant_id",
    "portfolio_state_family",
    "portfolio_variant_id",
    "lexical_split",
    "strategy_lexical_split",
    "setting_lexical_split",
    "system_text",
    "user_text",
    "prompt_messages_json",
    "strategy_snapshot_json",
    "settings_snapshot_json",
    "portfolio_snapshot_json",
    "market_snapshot_json",
    "market_expected_action",
    "market_expected_asset",
    "strategy_expected_action",
    "strategy_expected_asset",
    "strategy_expected_size",
    "setting_expected_action",
    "setting_expected_asset",
    "setting_expected_size",
    "expected_output_json",
]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    example_id TEXT PRIMARY KEY,
    matched_pair_id TEXT NOT NULL,
    pair_member TEXT NOT NULL,
    strategy_family TEXT NOT NULL,
    strategy_variant_id TEXT NOT NULL,
    setting_lexical_family_id TEXT NOT NULL,
    setting_family TEXT NOT NULL,
    setting_variant_id TEXT NOT NULL,
    setting_value INT NOT NULL,
    setting_bucket TEXT NOT NULL,
    conflict_present BOOLEAN NOT NULL,
    conflict_strength INT NOT NULL,
    environment_pressure_bucket TEXT NOT NULL,
    section_order TEXT NOT NULL,
    context_family TEXT NOT NULL,
    context_variant_id TEXT NOT NULL,
    portfolio_state_family TEXT NOT NULL,
    portfolio_variant_id TEXT NOT NULL,
    lexical_split TEXT NOT NULL,
    strategy_lexical_split TEXT NOT NULL,
    setting_lexical_split TEXT NOT NULL,
    system_text TEXT NOT NULL,
    user_text TEXT NOT NULL,
    prompt_messages_json JSONB NOT NULL,
    strategy_snapshot_json JSONB NOT NULL,
    settings_snapshot_json JSONB NOT NULL,
    portfolio_snapshot_json JSONB NOT NULL,
    market_snapshot_json JSONB NOT NULL,
    market_expected_action TEXT NOT NULL,
    market_expected_asset TEXT NOT NULL,
    strategy_expected_action TEXT NOT NULL,
    strategy_expected_asset TEXT NOT NULL,
    strategy_expected_size TEXT NOT NULL,
    setting_expected_action TEXT NOT NULL,
    setting_expected_asset TEXT NOT NULL,
    setting_expected_size TEXT NOT NULL,
    expected_output_json JSONB NOT NULL,
    built_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

PAIR_BUCKETS = (
    ("aligned", 0, False),
    ("strong_conflict", 2, True),
)

PRESSURE_BUCKETS = ("balanced", "strategy_favored", "setting_favored")

SECTION_ORDERS = ("strategy_first", "setting_first")

# Lexical holdout: v0/v1 are train variants, v2/v3 are test variants.
STRATEGY_SPLIT = {"v0": "train", "v1": "train", "v2": "test", "v3": "test"}
SETTING_SPLIT = {"v0": "train", "v1": "train", "v2": "test", "v3": "test"}


@dataclass(frozen=True)
class StrategyVariant:
    variant_id: str
    text: str


@dataclass(frozen=True)
class SettingPhraseFamily:
    lexical_family_id: str
    variant_suffix: str
    bucket_text: dict[int, str]


@dataclass(frozen=True)
class ContextTemplate:
    context_variant_id: str
    pressure_bucket: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class FamilySpec:
    strategy_family: str
    setting_family: str
    aligned_value: int
    conflict_value: int
    strategy_variants: tuple[StrategyVariant, ...]
    setting_phrase_families: tuple[SettingPhraseFamily, ...]
    contexts: tuple[ContextTemplate, ...]
    strategy_expected: dict[str, str]
    setting_expected: dict[str, str]
    market_expected: dict[str, str]


SIZE_LARGE_STRATEGIES = (
    StrategyVariant("size_large_v0", "When an entry is justified, use the large size tier."),
    StrategyVariant("size_large_v1", "If one asset clearly leads, favor the larger allocation tier."),
    StrategyVariant("size_large_v2", "Sizing should lean toward the largest tier permitted for any entry."),
    StrategyVariant("size_large_v3", "On a clear setup, size up rather than scale down."),
)

SIZE_SMALL_STRATEGIES = (
    StrategyVariant("size_small_v0", "When an entry is justified, use the small size tier."),
    StrategyVariant("size_small_v1", "If one asset clearly leads, favor the smaller allocation tier."),
    StrategyVariant("size_small_v2", "Sizing should stay at the smallest tier permitted for any entry."),
    StrategyVariant("size_small_v3", "On a clear setup, size down rather than scale up."),
)

SIZE_SETTING_PHRASES = (
    SettingPhraseFamily(
        "size_setting_phrase_v0",
        "v0",
        {
            1: "Trade size: 1/5. Use the small allocation tier.",
            5: "Trade size: 5/5. Use the large allocation tier.",
        },
    ),
    SettingPhraseFamily(
        "size_setting_phrase_v1",
        "v1",
        {
            1: "Execution size should stay small.",
            5: "Execution size can use the large tier.",
        },
    ),
    SettingPhraseFamily(
        "size_setting_phrase_v2",
        "v2",
        {
            1: "Position allocation is capped at the minimum tier.",
            5: "Position allocation is permitted at the maximum tier.",
        },
    ),
    SettingPhraseFamily(
        "size_setting_phrase_v3",
        "v3",
        {
            1: "Keep trade footprint modest on any entry.",
            5: "Trade footprint may be scaled up on any entry.",
        },
    ),
)

SIZE_CONTEXTS = tuple(
    ContextTemplate(
        f"size_{pressure}_v{i}",
        pressure,
        lines,
    )
    for pressure, variants in {
        "balanced": [
            (
                "ALPHA: strongest near-term setup, stable follow-through, limited caution.",
                "BETA: weaker case and weaker confirmation.",
                "DELTA: mixed tape with no decisive edge.",
                "GAMMA: mixed and noisy.",
            ),
            (
                "ALPHA: leading setup, consistent confirmation, manageable caution.",
                "BETA: secondary case without strong follow-through.",
                "DELTA: mixed signal with no clean path.",
                "GAMMA: mixed and noisy.",
            ),
        ],
        "strategy_favored": [
            (
                "ALPHA: dominant setup, very strong confirmation, low friction.",
                "BETA: weak signal and weak confirmation.",
                "DELTA: mixed tape with no clean edge.",
                "GAMMA: mixed and noisy.",
            ),
            (
                "ALPHA: clearly dominant case, strong confirmation, limited caution.",
                "BETA: weak secondary case.",
                "DELTA: mixed and indecisive.",
                "GAMMA: mixed and noisy.",
            ),
        ],
        "setting_favored": [
            (
                "ALPHA: usable edge with some confirmation, but one caution remains.",
                "BETA: weak case and weak confirmation.",
                "DELTA: mixed tape with no clean edge.",
                "GAMMA: mixed and noisy.",
            ),
            (
                "ALPHA: clear but not overwhelming edge, moderate confirmation, one caution note.",
                "BETA: weaker alternative.",
                "DELTA: mixed and indecisive.",
                "GAMMA: mixed and noisy.",
            ),
        ],
    }.items()
    for i, lines in enumerate(variants)
)


FAMILY_SPECS = (
    FamilySpec(
        strategy_family="trade_size_force_large",
        setting_family="trade_size",
        aligned_value=5,
        conflict_value=1,
        strategy_variants=SIZE_LARGE_STRATEGIES,
        setting_phrase_families=SIZE_SETTING_PHRASES,
        contexts=SIZE_CONTEXTS,
        strategy_expected={"action": "buy", "asset": "ALPHA", "size": "large"},
        setting_expected={"action": "buy", "asset": "ALPHA", "size": "small"},
        market_expected={"action": "buy", "asset": "ALPHA"},
    ),
    FamilySpec(
        strategy_family="trade_size_force_small",
        setting_family="trade_size",
        aligned_value=1,
        conflict_value=5,
        strategy_variants=SIZE_SMALL_STRATEGIES,
        setting_phrase_families=SIZE_SETTING_PHRASES,
        contexts=SIZE_CONTEXTS,
        strategy_expected={"action": "buy", "asset": "ALPHA", "size": "small"},
        setting_expected={"action": "buy", "asset": "ALPHA", "size": "large"},
        market_expected={"action": "buy", "asset": "ALPHA"},
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the prompt-confusion Phase 06 dataset.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--upload", action="store_true", help=f"Upload rows to Neon table {NEON_TABLE}.")
    parser.add_argument("--dest-table", default=NEON_TABLE)
    return parser.parse_args()


def lexical_split(strategy_variant_id: str, setting_variant_suffix: str) -> tuple[str, str, str]:
    strategy_split = STRATEGY_SPLIT[strategy_variant_id.rsplit("_", 1)[-1]]
    setting_split = SETTING_SPLIT[setting_variant_suffix]
    overall = setting_split
    return overall, strategy_split, setting_split


def render_user_text(
    strategy_text: str,
    setting_text: str,
    market_lines: tuple[str, ...],
    section_order: str,
) -> str:
    strategy_block = f"STRATEGY\n{strategy_text}"
    settings_block = f"SETTINGS\n{setting_text}"
    if section_order == "strategy_first":
        policy_blocks = [strategy_block, settings_block]
    elif section_order == "setting_first":
        policy_blocks = [settings_block, strategy_block]
    else:
        raise ValueError(f"unknown section_order: {section_order}")
    return "\n\n".join(
        [
            "TASK\nChoose exactly one action for this tick.",
            *policy_blocks,
            "PORTFOLIO\nFree cash reserve: high.\nCurrent positions: none.\nEnough buying power is available for any allowed size.",
            "MARKET\n" + "\n".join(market_lines),
        ]
    )


def expected_for_row(family: FamilySpec, pair_member: str) -> dict[str, str]:
    if pair_member == "aligned":
        return dict(family.strategy_expected)
    return dict(family.setting_expected)


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILY_SPECS:
        contexts_by_pressure: dict[str, list[ContextTemplate]] = {bucket: [] for bucket in PRESSURE_BUCKETS}
        for context in family.contexts:
            contexts_by_pressure[context.pressure_bucket].append(context)
        for strategy in family.strategy_variants:
            for setting_family in family.setting_phrase_families:
                for pressure in PRESSURE_BUCKETS:
                    for context in contexts_by_pressure[pressure]:
                        for section_order in SECTION_ORDERS:
                            for pair_member, conflict_strength, conflict_present in PAIR_BUCKETS:
                                setting_value = (
                                    family.aligned_value if pair_member == "aligned" else family.conflict_value
                                )
                                setting_text = setting_family.bucket_text[setting_value]
                                overall_split, strategy_split, setting_split = lexical_split(
                                    strategy.variant_id, setting_family.variant_suffix
                                )
                                matched_pair_id = (
                                    f"pc6pair:{family.strategy_family}:{pressure}:{context.context_variant_id}:"
                                    f"{strategy.variant_id}:{setting_family.lexical_family_id}:{section_order}"
                                )
                                setting_variant_id = (
                                    f"{family.setting_family}_setting_{setting_value}_{setting_family.variant_suffix}"
                                )
                                example_id = (
                                    f"pc6:{family.strategy_family}:{pressure}:{context.context_variant_id}:"
                                    f"{strategy.variant_id}:{setting_variant_id}:{section_order}:{pair_member}"
                                )
                                user_text = render_user_text(
                                    strategy.text, setting_text, context.lines, section_order
                                )
                                expected_output = expected_for_row(family, pair_member)
                                row = {
                                    "example_id": example_id,
                                    "matched_pair_id": matched_pair_id,
                                    "pair_member": pair_member,
                                    "strategy_family": family.strategy_family,
                                    "strategy_variant_id": strategy.variant_id,
                                    "setting_lexical_family_id": setting_family.lexical_family_id,
                                    "setting_family": family.setting_family,
                                    "setting_variant_id": setting_variant_id,
                                    "setting_value": setting_value,
                                    "setting_bucket": pair_member,
                                    "conflict_present": conflict_present,
                                    "conflict_strength": conflict_strength,
                                    "environment_pressure_bucket": pressure,
                                    "section_order": section_order,
                                    "context_family": "size_entry_case",
                                    "context_variant_id": context.context_variant_id,
                                    "portfolio_state_family": "empty_cash_rich",
                                    "portfolio_variant_id": "empty_cash_rich_v0",
                                    "lexical_split": overall_split,
                                    "strategy_lexical_split": strategy_split,
                                    "setting_lexical_split": setting_split,
                                    "system_text": SYSTEM_TEXT,
                                    "user_text": user_text,
                                    "prompt_messages_json": [
                                        {"role": "system", "content": SYSTEM_TEXT},
                                        {"role": "user", "content": user_text},
                                    ],
                                    "strategy_snapshot_json": {
                                        "strategy_family": family.strategy_family,
                                        "strategy_variant_id": strategy.variant_id,
                                        "strategy_text": strategy.text,
                                    },
                                    "settings_snapshot_json": {
                                        "setting_family": family.setting_family,
                                        "setting_variant_id": setting_variant_id,
                                        "setting_lexical_family_id": setting_family.lexical_family_id,
                                        "setting_value": setting_value,
                                        "setting_bucket": pair_member,
                                        "setting_text": setting_text,
                                    },
                                    "portfolio_snapshot_json": {
                                        "portfolio_state_family": "empty_cash_rich",
                                        "portfolio_variant_id": "empty_cash_rich_v0",
                                        "held_assets": [],
                                        "cash_state": "high",
                                    },
                                    "market_snapshot_json": {
                                        "context_variant_id": context.context_variant_id,
                                        "pressure_bucket": pressure,
                                        "section_order": section_order,
                                        "assets": [
                                            {"asset": "ALPHA", "description": context.lines[0]},
                                            {"asset": "BETA", "description": context.lines[1]},
                                            {"asset": "DELTA", "description": context.lines[2]},
                                            {"asset": "GAMMA", "description": context.lines[3]},
                                        ],
                                    },
                                    "market_expected_action": family.market_expected["action"],
                                    "market_expected_asset": family.market_expected["asset"],
                                    "strategy_expected_action": family.strategy_expected["action"],
                                    "strategy_expected_asset": family.strategy_expected["asset"],
                                    "strategy_expected_size": family.strategy_expected["size"],
                                    "setting_expected_action": family.setting_expected["action"],
                                    "setting_expected_asset": family.setting_expected["asset"],
                                    "setting_expected_size": family.setting_expected["size"],
                                    "expected_output_json": expected_output,
                                }
                                rows.append(row)
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "phase_06_dataset.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    family_counts = Counter(row["strategy_family"] for row in rows)
    split_counts = Counter(row["lexical_split"] for row in rows)
    order_counts = Counter(row["section_order"] for row in rows)
    conflict_counts = Counter(
        (row["strategy_family"], row["conflict_present"]) for row in rows
    )
    summary = {
        "row_count": len(rows),
        "family_counts": dict(family_counts),
        "lexical_split_counts": dict(split_counts),
        "section_order_counts": dict(order_counts),
        "conflict_counts": {f"{k[0]}|{k[1]}": v for k, v in conflict_counts.items()},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {"jsonl_path": str(jsonl_path), "summary": summary}


def upload_rows(table_name: str, rows: list[dict[str, Any]]) -> int:
    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        conn.execute(CREATE_TABLE_SQL.format(table=table_name))
        conn.execute(f"TRUNCATE TABLE {table_name}")
        column_list = ", ".join(TABLE_COLUMNS)
        with conn.cursor().copy(f"COPY {table_name} ({column_list}) FROM STDIN") as copy:
            for row in rows:
                copy.write_row(
                    tuple(
                        json.dumps(row[c], sort_keys=True) if isinstance(row[c], (dict, list)) else row[c]
                        for c in TABLE_COLUMNS
                    )
                )
        return conn.execute(f"SELECT COUNT(*) AS n FROM {table_name}").fetchone()["n"]


def main() -> None:
    args = parse_args()
    rows = build_rows()
    payload = write_outputs(rows, args.output_dir)
    if args.upload:
        payload["uploaded_row_count"] = upload_rows(args.dest_table, rows)
        payload["dest_table"] = args.dest_table
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
