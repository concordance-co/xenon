from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SYSTEM_TEXT = (
    "You are a trading agent. Each prompt contains a STRATEGY and ACTIVE SETTINGS. "
    "ACTIVE SETTINGS are binding execution constraints. STRATEGY expresses a preference "
    "that applies only within what ACTIVE SETTINGS allow. "
    "Decide in this order: first whether ACTIVE SETTINGS permit entry, then which asset "
    "is best from MARKET, then what size ACTIVE SETTINGS allow. "
    "If ACTIVE SETTINGS do not permit entry, return "
    '{"action":"observe","asset":"NONE","size":"none"}. '
    "Choose exactly one action each turn. Return only a JSON object with exactly these fields: "
    '{"action":"buy|sell|observe","asset":"ALPHA|BETA|DELTA|GAMMA|NONE","size":"small|large|none"}. '
    "Do not return any other keys or any other text."
)

OUTPUT_DIR = Path("projects/DX_TERMINAL/prompt_confusion/phase_09/outputs/phase_09_dataset")
TARGET_SETTING_VALUES = (1, 5)

STRATEGY_TEMPLATE_SPLIT = {"policy_v0": "train", "policy_v1": "train", "policy_v2": "test", "policy_v3": "test"}
SETTINGS_TEMPLATE_SPLIT = {"settings_v0": "train", "settings_v1": "train", "settings_v2": "test", "settings_v3": "test"}


@dataclass(frozen=True)
class StrategyTemplate:
    variant_id: str
    activity_shell: str
    size_shell: str


@dataclass(frozen=True)
class SettingsTemplate:
    variant_id: str
    activity_gloss: str
    size_gloss: str
    risk_gloss: str
    holding_gloss: str
    diversification_gloss: str


@dataclass(frozen=True)
class MarketContext:
    context_variant_id: str
    evidence_tier: str
    evidence_score: int
    alpha_lines: tuple[str, str, str]


ACTIVITY_STRATEGY_TEXT = {
    "observe": (
        "Prefer entering only when evidence is unusually strong.",
        "Stay selective and avoid early entries unless the evidence is exceptional.",
        "Lean toward waiting unless confirmation becomes very strong.",
        "Keep entry activity conservative unless the setup is unusually convincing.",
    ),
    "trade": (
        "Prefer taking credible supported opportunities without waiting for exceptional confirmation.",
        "Be willing to enter once evidence is solid and coherent.",
        "Lean toward acting when the setup is credible and supported.",
        "Take supported opportunities rather than waiting for unusually strong confirmation.",
    ),
}

SIZE_STRATEGY_TEXT = {
    "small": (
        "Prefer smaller allocations when trading.",
        "Keep position size in the small tier when a trade is taken.",
        "Favor the smallest working size tier on entries.",
        "Use a smaller allocation band when entering.",
    ),
    "large": (
        "Prefer larger allocations when trading.",
        "Keep position size in the large tier when a trade is taken.",
        "Favor the largest working size tier on entries.",
        "Use a larger allocation band when entering.",
    ),
}

STRATEGY_TEMPLATES = (
    StrategyTemplate("policy_v0", "{activity_text}", "{size_text}"),
    StrategyTemplate("policy_v1", "For this tick, {activity_text}", "For sizing, {size_text}"),
    StrategyTemplate("policy_v2", "Working preference: {activity_text}", "Working size preference: {size_text}"),
    StrategyTemplate("policy_v3", "Current preference: {activity_text}", "Current size preference: {size_text}"),
)

SETTINGS_TEMPLATES = (
    SettingsTemplate(
        "settings_v0",
        "Trading Activity",
        "Trade Size",
        "Asset Risk Preference",
        "Holding Style",
        "Diversification",
    ),
    SettingsTemplate(
        "settings_v1",
        "Execution Activity",
        "Execution Size",
        "Risk Preference",
        "Holding Horizon",
        "Diversification Preference",
    ),
    SettingsTemplate(
        "settings_v2",
        "Activity Constraint",
        "Size Constraint",
        "Risk Constraint",
        "Hold Constraint",
        "Diversification Constraint",
    ),
    SettingsTemplate(
        "settings_v3",
        "Activity Setting",
        "Size Setting",
        "Risk Setting",
        "Hold Setting",
        "Diversification Setting",
    ),
)

MARKET_CONTEXTS = (
    MarketContext(
        "ctx_weak_v0",
        "weak",
        1,
        (
            "momentum has improved only intermittently",
            "confirmation remains narrow",
            "downside uncertainty is still elevated",
        ),
    ),
    MarketContext(
        "ctx_weak_v1",
        "weak",
        1,
        (
            "signal quality is noisy",
            "follow-through is inconsistent",
            "uncertainty remains high",
        ),
    ),
    MarketContext(
        "ctx_weak_v2",
        "weak",
        1,
        (
            "price action has some positive drift but weak persistence",
            "confirmation is limited",
            "risk remains difficult to bound",
        ),
    ),
    MarketContext(
        "ctx_solid_v0",
        "solid",
        3,
        (
            "momentum is stable",
            "confirmation is moderate and improving",
            "downside uncertainty is contained",
        ),
    ),
    MarketContext(
        "ctx_solid_v1",
        "solid",
        3,
        (
            "follow-through has been fairly consistent",
            "confirmation is broad enough to take seriously",
            "risk looks manageable",
        ),
    ),
    MarketContext(
        "ctx_solid_v2",
        "solid",
        3,
        (
            "signal quality is credible",
            "supporting evidence is reasonably coherent",
            "uncertainty is present but not dominant",
        ),
    ),
    MarketContext(
        "ctx_exceptional_v0",
        "exceptional",
        5,
        (
            "momentum is strong and persistent",
            "confirmation is broad",
            "downside uncertainty is low",
        ),
    ),
    MarketContext(
        "ctx_exceptional_v1",
        "exceptional",
        5,
        (
            "follow-through is unusually consistent",
            "supporting evidence is strong across multiple signals",
            "risk is tightly contained",
        ),
    ),
    MarketContext(
        "ctx_exceptional_v2",
        "exceptional",
        5,
        (
            "signal quality is unusually strong",
            "confirmation is both broad and deep",
            "uncertainty is minimal relative to the opportunity",
        ),
    ),
)


def _activity_threshold(value: int) -> int:
    return {1: 5, 5: 1}[value]


def _activity_implied_direction(value: int, evidence_score: int) -> str:
    return "trade" if evidence_score >= _activity_threshold(value) else "observe"


def _size_implied_direction(value: int) -> str:
    return "small" if value == 1 else "large"


def _size_from_value(value: int) -> str:
    return {1: "small", 5: "large"}[value]


def _activity_setting_text(value: int) -> str:
    mapping = {
        1: "treat a setup as actionable only when confirmation is very broad and instability is low",
        5: "treat any setup as actionable; do not require a minimum evidence bar",
    }
    return mapping[value]


def _size_setting_text(value: int) -> str:
    mapping = {
        1: "use the smallest size tier",
        2: "stay in the smaller size range",
        3: "use the standard size tier",
        4: "allow sizes above the standard tier",
        5: "use the largest size tier",
    }
    return mapping[value]


def _risk_setting_text(value: int) -> str:
    return {
        1: "favor the most stable risk profile",
        2: "stay near the conservative side of balanced risk",
        3: "use a balanced risk profile",
        4: "allow somewhat more aggressive risk when evidence supports it",
        5: "allow the most aggressive risk profile in the allowed set",
    }[value]


def _holding_setting_text(value: int) -> str:
    return {
        1: "prefer shorter holds",
        2: "lean toward shorter holds when uncertainty rises",
        3: "use a normal multi-tick hold horizon",
        4: "allow somewhat longer holds when follow-through stays stable",
        5: "allow the longest holding horizon in the allowed set",
    }[value]


def _diversification_setting_text(value: int) -> str:
    return {
        1: "keep diversification fairly tight",
        2: "allow limited diversification",
        3: "use a balanced diversification stance",
        4: "allow somewhat broader diversification",
        5: "allow the broadest diversification stance in the allowed set",
    }[value]


def _stable_index(parts: tuple[str, ...], n: int) -> int:
    total = 0
    for part in parts:
        for idx, ch in enumerate(part):
            total += (idx + 1) * ord(ch)
    return total % n


def _nuisance_setting_value(group_key: str, field_name: str) -> int:
    return (2, 3, 4)[_stable_index((group_key, field_name), 3)]


def _nuisance_size_value(group_key: str) -> int:
    return (1, 5)[_stable_index((group_key, "size_nuisance"), 2)]


def _strategy_phrase_variant(
    pool: tuple[str, ...],
    *,
    template_index: int,
    settings_index: int,
    context_index: int,
    direction_index: int,
) -> tuple[str, str]:
    phrase_index = (template_index + settings_index + context_index + direction_index) % len(pool)
    return f"phrase_v{phrase_index}", pool[phrase_index]


def _lexical_split(strategy_variant_id: str, settings_variant_id: str) -> str:
    strategy_split = STRATEGY_TEMPLATE_SPLIT[strategy_variant_id]
    settings_split = SETTINGS_TEMPLATE_SPLIT[settings_variant_id]
    return "train" if strategy_split == settings_split else "test"


def _render_system_text() -> str:
    return SYSTEM_TEXT


def _render_market_text(context: MarketContext) -> tuple[str, dict[str, Any]]:
    alpha = (
        f"ALPHA: {context.alpha_lines[0]}, {context.alpha_lines[1]}, and {context.alpha_lines[2]}.",
        "BETA: evidence is weaker, confirmation is thinner, and follow-through is less reliable.",
        "DELTA: the signal remains mixed and difficult to size confidently.",
        "GAMMA: behavior is noisy and lacks coherent confirmation.",
    )
    return "\n".join(alpha), {
        "evidence_tier": context.evidence_tier,
        "evidence_score": context.evidence_score,
        "alpha_lines": list(context.alpha_lines),
    }


def _render_user_text(
    *,
    strategy_template: StrategyTemplate,
    settings_template: SettingsTemplate,
    activity_phrase_id: str,
    activity_text: str,
    size_phrase_id: str,
    size_text: str,
    activity_value: int,
    size_value: int,
    risk_value: int,
    holding_value: int,
    diversification_value: int,
    context: MarketContext,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    market_text, market_snapshot = _render_market_text(context)
    strategy_block = (
        "TASK\n"
        "Choose exactly one action for this tick.\n\n"
        "STRATEGY\n"
        f"- {strategy_template.activity_shell.format(activity_text=activity_text)}\n"
        f"- {strategy_template.size_shell.format(size_text=size_text)}\n"
        "Apply the strategy only within what ACTIVE SETTINGS allow.\n\n"
        "ACTIVE SETTINGS\n"
        f"- {settings_template.activity_gloss}: {activity_value}/5 — {_activity_setting_text(activity_value)}.\n"
        f"- {settings_template.size_gloss}: {size_value}/5 — {_size_setting_text(size_value)}.\n"
        f"- {settings_template.risk_gloss}: {risk_value}/5 — {_risk_setting_text(risk_value)}.\n"
        f"- {settings_template.holding_gloss}: {holding_value}/5 — {_holding_setting_text(holding_value)}.\n"
        f"- {settings_template.diversification_gloss}: {diversification_value}/5 — {_diversification_setting_text(diversification_value)}.\n\n"
        "PORTFOLIO\n"
        "Free cash reserve: high.\n"
        "Current positions: none.\n"
        "Enough buying power is available for any allowed size.\n\n"
        "MARKET\n"
        f"{market_text}"
    )
    strategy_snapshot = {
        "activity_phrase_id": activity_phrase_id,
        "activity_text": activity_text,
        "size_phrase_id": size_phrase_id,
        "size_text": size_text,
        "strategy_variant_id": strategy_template.variant_id,
    }
    settings_snapshot = {
        "activity_value": activity_value,
        "size_value": size_value,
        "risk_value": risk_value,
        "holding_value": holding_value,
        "diversification_value": diversification_value,
        "settings_variant_id": settings_template.variant_id,
    }
    portfolio_snapshot = {
        "free_cash_reserve": "high",
        "current_positions": "none",
        "buying_power": "enough_for_any_allowed_size",
    }
    return strategy_block, strategy_snapshot, settings_snapshot, portfolio_snapshot | market_snapshot


def _expected_for_row(
    *,
    target_dimension: str,
    strategy_direction: str,
    setting_value: int,
    size_value: int,
    context: MarketContext,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], str, bool | None, bool]:
    if target_dimension == "trade_size":
        setting_dir = _size_implied_direction(setting_value)
        edge_conflict = False
        conflict_present = setting_dir != strategy_direction
        expected = {"action": "buy", "asset": "ALPHA", "size": _size_from_value(setting_value)}
        strategy_expected = {"action": "buy", "asset": "ALPHA", "size": strategy_direction}
        setting_expected = dict(expected)
        conflict_band = "strong_conflict" if conflict_present else "aligned"
        return strategy_expected, setting_expected, expected, conflict_band, conflict_present, edge_conflict

    setting_dir = _activity_implied_direction(setting_value, context.evidence_score)
    conflict_present = setting_dir != strategy_direction
    edge_conflict = False
    if setting_dir == "trade":
        expected = {"action": "buy", "asset": "ALPHA", "size": _size_from_value(size_value)}
    else:
        expected = {"action": "observe", "asset": "NONE", "size": "none"}
    strategy_expected = {"action": "buy", "asset": "ALPHA", "size": "large" if strategy_direction == "trade" else "none"}
    if strategy_direction == "observe":
        strategy_expected = {"action": "observe", "asset": "NONE", "size": "none"}
    setting_expected = dict(expected)
    if conflict_present:
        conflict_band = "strong_conflict"
    else:
        conflict_band = "aligned"
    return strategy_expected, setting_expected, expected, conflict_band, conflict_present, edge_conflict


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    strategy_lookup = {
        "trade_size": ("small", "large"),
        "trading_activity": ("observe", "trade"),
    }

    for target_dimension, directions in strategy_lookup.items():
        for direction_index, strategy_direction in enumerate(directions):
            activity_pool = ACTIVITY_STRATEGY_TEXT["observe" if strategy_direction == "observe" else "trade"]
            if target_dimension == "trade_size":
                activity_pool = ACTIVITY_STRATEGY_TEXT["trade"]
            size_pool = SIZE_STRATEGY_TEXT["small" if strategy_direction == "small" else "large"]
            if target_dimension == "trading_activity":
                size_pool = SIZE_STRATEGY_TEXT["large"]

            for template_index, strategy_template in enumerate(STRATEGY_TEMPLATES):
                for settings_index, settings_template in enumerate(SETTINGS_TEMPLATES):
                    for context_index, context in enumerate(MARKET_CONTEXTS):
                        activity_phrase_id, activity_text = _strategy_phrase_variant(
                            activity_pool,
                            template_index=template_index,
                            settings_index=settings_index,
                            context_index=context_index,
                            direction_index=direction_index,
                        )
                        size_phrase_id, size_text = _strategy_phrase_variant(
                            size_pool,
                            template_index=template_index,
                            settings_index=settings_index,
                            context_index=context_index,
                            direction_index=direction_index + 1,
                        )
                        for setting_value in TARGET_SETTING_VALUES:
                            if target_dimension == "trade_size":
                                activity_value = 5
                                size_value = setting_value
                            else:
                                activity_value = setting_value
                                size_value = _nuisance_size_value(
                                    f"{target_dimension}:{strategy_direction}:{strategy_template.variant_id}:{settings_template.variant_id}:{context.context_variant_id}"
                                )
                            group_key = (
                                f"{target_dimension}:{strategy_direction}:{strategy_template.variant_id}:"
                                f"{settings_template.variant_id}:{context.context_variant_id}"
                            )
                            risk_value = _nuisance_setting_value(group_key, "risk")
                            holding_value = _nuisance_setting_value(group_key, "holding")
                            diversification_value = _nuisance_setting_value(group_key, "diversification")
                            strategy_expected, setting_expected, expected_output, conflict_band, conflict_present, edge_conflict = _expected_for_row(
                                target_dimension=target_dimension,
                                strategy_direction=strategy_direction,
                                setting_value=setting_value,
                                size_value=size_value,
                                context=context,
                            )
                            if expected_output["action"] == "buy" and context.evidence_tier == "weak":
                                continue
                            user_text, strategy_snapshot, settings_snapshot, prompt_snapshot = _render_user_text(
                                strategy_template=strategy_template,
                                settings_template=settings_template,
                                activity_phrase_id=activity_phrase_id,
                                activity_text=activity_text,
                                size_phrase_id=size_phrase_id,
                                size_text=size_text,
                                activity_value=activity_value,
                                size_value=size_value,
                                risk_value=risk_value,
                                holding_value=holding_value,
                                diversification_value=diversification_value,
                                context=context,
                            )
                            matched_group_id = group_key
                            if target_dimension == "trade_size":
                                pair_value = 1 if strategy_direction == "large" else 5
                            else:
                                pair_value = 1 if strategy_direction == "trade" else 5
                            matched_pair_id = f"{matched_group_id}:pair_{pair_value}"
                            row = {
                                "example_id": (
                                    f"pc9:{target_dimension}:{strategy_direction}:{context.context_variant_id}:"
                                    f"{strategy_template.variant_id}:{settings_template.variant_id}:setting_{setting_value}"
                                ),
                                "matched_group_id": matched_group_id,
                                "matched_pair_id": matched_pair_id,
                                "target_dimension": target_dimension,
                                "strategy_direction": strategy_direction,
                                "setting_value": setting_value,
                                "setting_implied_direction": _size_implied_direction(setting_value) if target_dimension == "trade_size" else _activity_implied_direction(setting_value, context.evidence_score),
                                "conflict_present": conflict_present,
                                "edge_conflict": edge_conflict,
                                "conflict_band": conflict_band,
                                "main_benchmark_row": not (
                                    target_dimension == "trading_activity"
                                    and setting_value == 1
                                    and context.evidence_tier == "exceptional"
                                ),
                                "stress_test_slice": (
                                    "activity_v1_exceptional_boundary"
                                    if target_dimension == "trading_activity"
                                    and setting_value == 1
                                    and context.evidence_tier == "exceptional"
                                    else ""
                                ),
                                "conflict_strength": 0 if conflict_band == "aligned" else (1 if conflict_band == "edge" else 2),
                                "strategy_variant_id": strategy_template.variant_id,
                                "activity_phrase_id": activity_phrase_id,
                                "size_phrase_id": size_phrase_id,
                                "strategy_lexical_split": STRATEGY_TEMPLATE_SPLIT[strategy_template.variant_id],
                                "settings_variant_id": settings_template.variant_id,
                                "settings_lexical_split": SETTINGS_TEMPLATE_SPLIT[settings_template.variant_id],
                                "context_variant_id": context.context_variant_id,
                                "lexical_split": _lexical_split(strategy_template.variant_id, settings_template.variant_id),
                                "system_text": _render_system_text(),
                                "user_text": user_text,
                                "prompt_messages_json": [
                                    {"role": "system", "content": _render_system_text()},
                                    {"role": "user", "content": user_text},
                                ],
                                "strategy_snapshot_json": strategy_snapshot,
                                "settings_snapshot_json": settings_snapshot,
                                "portfolio_snapshot_json": {
                                    "free_cash_reserve": "high",
                                    "current_positions": "none",
                                    "buying_power": "enough_for_any_allowed_size",
                                },
                                "market_snapshot_json": prompt_snapshot,
                                "strategy_expected_action": strategy_expected["action"],
                                "strategy_expected_asset": strategy_expected["asset"],
                                "strategy_expected_size": strategy_expected["size"],
                                "setting_expected_action": setting_expected["action"],
                                "setting_expected_asset": setting_expected["asset"],
                                "setting_expected_size": setting_expected["size"],
                                "expected_output_json": expected_output,
                            }
                            rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the prompt-confusion Phase 09 dataset scaffold.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "phase_09_dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "rows": len(rows),
        "by_dimension": dict(Counter(row["target_dimension"] for row in rows)),
        "by_band": dict(Counter(row["conflict_band"] for row in rows)),
        "by_main_benchmark_row": dict(Counter(row["main_benchmark_row"] for row in rows)),
        "main_benchmark_rows": sum(1 for row in rows if bool(row["main_benchmark_row"])),
        "stress_test_rows": sum(1 for row in rows if str(row["stress_test_slice"])),
        "by_split": dict(Counter(row["lexical_split"] for row in rows)),
        "by_strategy_split": dict(Counter(row["strategy_lexical_split"] for row in rows)),
        "by_settings_split": dict(Counter(row["settings_lexical_split"] for row in rows)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows()
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
