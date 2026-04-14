from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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

OUTPUT_DIR = Path("projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/phase_04_dataset")

PAIR_BUCKETS = (
    ("aligned", 0, False),
    ("strong_conflict", 2, True),
)

PRESSURE_BUCKETS = ("balanced", "strategy_favored", "setting_favored")

STRATEGY_SPLIT = {"v0": "train", "v1": "test"}
SETTING_SPLIT = {"v0": "train", "v1": "test"}


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
)

SIZE_SMALL_STRATEGIES = (
    StrategyVariant("size_small_v0", "When an entry is justified, use the small size tier."),
    StrategyVariant("size_small_v1", "If one asset clearly leads, favor the smaller allocation tier."),
)

ACTIVITY_TRADE_STRATEGIES = (
    StrategyVariant("activity_trade_v0", "If the case is live, take the available entry."),
    StrategyVariant("activity_trade_v1", "When one asset is clearly preferred, enter rather than wait."),
)

ACTIVITY_OBSERVE_STRATEGIES = (
    StrategyVariant("activity_observe_v0", "If the case is only moderate, prefer waiting over entry."),
    StrategyVariant("activity_observe_v1", "When the evidence is not exceptional, prefer waiting."),
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
)

ACTIVITY_SETTING_PHRASES = (
    SettingPhraseFamily(
        "activity_setting_phrase_v0",
        "v0",
        {
            1: "Trading activity: 1/5. Entries should be rare.",
            5: "Trading activity: 5/5. Entries may be taken whenever the case is live.",
        },
    ),
    SettingPhraseFamily(
        "activity_setting_phrase_v1",
        "v1",
        {
            1: "Execution pace should remain low.",
            5: "Execution pace can remain active.",
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
            (
                "ALPHA: best available case, enough support to justify entry, no major friction.",
                "BETA: weaker case with weaker confirmation.",
                "DELTA: unclear signal.",
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
            (
                "ALPHA: standout opportunity, unusually strong confirmation, low caution.",
                "BETA: weak alternative.",
                "DELTA: no clean case.",
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
            (
                "ALPHA: tradable setup, adequate confirmation, modest caution.",
                "BETA: weak alternative.",
                "DELTA: no clean case.",
                "GAMMA: mixed and noisy.",
            ),
        ],
    }.items()
    for i, lines in enumerate(variants)
)

ACTIVITY_CONTEXTS = tuple(
    ContextTemplate(
        f"activity_{pressure}_v{i}",
        pressure,
        lines,
    )
    for pressure, variants in {
        "balanced": [
            (
                "ALPHA: usable entry case, clear enough to trade but not urgent.",
                "BETA: weak signal and weak confirmation.",
                "DELTA: mixed tape with no clean edge.",
                "GAMMA: mixed and noisy.",
            ),
            (
                "ALPHA: moderate live case, enough support to justify entry.",
                "BETA: weak alternative.",
                "DELTA: mixed and indecisive.",
                "GAMMA: mixed and noisy.",
            ),
            (
                "ALPHA: actionable but not exceptional setup.",
                "BETA: weaker alternative.",
                "DELTA: no clean case.",
                "GAMMA: mixed and noisy.",
            ),
        ],
        "strategy_favored": [
            (
                "ALPHA: only modest edge with one caution; waiting is easy to justify.",
                "BETA: weak signal and weak confirmation.",
                "DELTA: mixed tape with no clean edge.",
                "GAMMA: mixed and noisy.",
            ),
            (
                "ALPHA: borderline live case, some support but no urgency.",
                "BETA: weak alternative.",
                "DELTA: mixed and indecisive.",
                "GAMMA: mixed and noisy.",
            ),
            (
                "ALPHA: tradable only if one prefers acting early; otherwise waiting is reasonable.",
                "BETA: weaker alternative.",
                "DELTA: no clean case.",
                "GAMMA: mixed and noisy.",
            ),
        ],
        "setting_favored": [
            (
                "ALPHA: clear entry case, solid confirmation, limited caution.",
                "BETA: weak signal and weak confirmation.",
                "DELTA: mixed tape with no clean edge.",
                "GAMMA: mixed and noisy.",
            ),
            (
                "ALPHA: strong live case with enough confirmation to justify entry.",
                "BETA: weak alternative.",
                "DELTA: mixed and indecisive.",
                "GAMMA: mixed and noisy.",
            ),
            (
                "ALPHA: clearly actionable setup with low friction.",
                "BETA: weaker alternative.",
                "DELTA: no clean case.",
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
    FamilySpec(
        strategy_family="activity_force_trade",
        setting_family="trading_activity",
        aligned_value=5,
        conflict_value=1,
        strategy_variants=ACTIVITY_TRADE_STRATEGIES,
        setting_phrase_families=ACTIVITY_SETTING_PHRASES,
        contexts=ACTIVITY_CONTEXTS,
        strategy_expected={"action": "buy", "asset": "ALPHA", "size": "medium"},
        setting_expected={"action": "observe", "asset": "NONE", "size": "none"},
        market_expected={"action": "buy", "asset": "ALPHA"},
    ),
    FamilySpec(
        strategy_family="activity_force_observe",
        setting_family="trading_activity",
        aligned_value=1,
        conflict_value=5,
        strategy_variants=ACTIVITY_OBSERVE_STRATEGIES,
        setting_phrase_families=ACTIVITY_SETTING_PHRASES,
        contexts=ACTIVITY_CONTEXTS,
        strategy_expected={"action": "observe", "asset": "NONE", "size": "none"},
        setting_expected={"action": "buy", "asset": "ALPHA", "size": "medium"},
        market_expected={"action": "buy", "asset": "ALPHA"},
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the prompt-confusion Phase 04 dataset.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def lexical_split(strategy_variant_id: str, setting_variant_suffix: str) -> tuple[str, str, str]:
    strategy_split = STRATEGY_SPLIT[strategy_variant_id.rsplit("_", 1)[-1]]
    setting_split = SETTING_SPLIT[setting_variant_suffix]
    # Keep the row-level split balanced enough to be operationally useful while
    # still preserving separate per-side split metadata.
    overall = setting_split
    return overall, strategy_split, setting_split


def render_user_text(strategy_text: str, setting_text: str, market_lines: tuple[str, ...]) -> str:
    return "\n\n".join(
        [
            "TASK\nChoose exactly one action for this tick.",
            f"STRATEGY\n{strategy_text}",
            f"SETTINGS\n{setting_text}",
            "PORTFOLIO\nFree cash reserve: high.\nCurrent positions: none.\nEnough buying power is available for any allowed size.",
            "MARKET\n" + "\n".join(market_lines),
        ]
    )


def expected_for_row(family: FamilySpec, pair_member: str) -> dict[str, str]:
    if pair_member == "aligned":
        return dict(family.strategy_expected if family.aligned_value in {1, 5} else family.setting_expected)
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
                        for pair_member, conflict_strength, conflict_present in PAIR_BUCKETS:
                            setting_value = family.aligned_value if pair_member == "aligned" else family.conflict_value
                            setting_text = setting_family.bucket_text[setting_value]
                            overall_split, strategy_split, setting_split = lexical_split(
                                strategy.variant_id, setting_family.variant_suffix
                            )
                            matched_pair_id = (
                                f"pc4pair:{family.strategy_family}:{pressure}:{context.context_variant_id}:"
                                f"{strategy.variant_id}:{setting_family.lexical_family_id}"
                            )
                            setting_variant_id = (
                                f"{family.setting_family}_setting_{setting_value}_{setting_family.variant_suffix}"
                            )
                            example_id = (
                                f"pc4:{family.strategy_family}:{pressure}:{context.context_variant_id}:"
                                f"{strategy.variant_id}:{setting_variant_id}:{pair_member}"
                            )
                            user_text = render_user_text(strategy.text, setting_text, context.lines)
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
                                "context_family": "size_entry_case" if family.setting_family == "trade_size" else "activity_entry_case",
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
    jsonl_path = output_dir / "phase_04_dataset.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    family_counts = Counter(row["strategy_family"] for row in rows)
    split_counts = Counter(row["lexical_split"] for row in rows)
    summary = {
        "row_count": len(rows),
        "family_counts": dict(family_counts),
        "lexical_split_counts": dict(split_counts),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return {"jsonl_path": str(jsonl_path), "summary": summary}


def main() -> None:
    args = parse_args()
    rows = build_rows()
    payload = write_outputs(rows, args.output_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
