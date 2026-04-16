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
    "best matches the allowed risk profile from MARKET, then what size ACTIVE SETTINGS allow. "
    "When multiple assets are viable, choose the asset whose risk profile best matches ACTIVE SETTINGS. "
    "If ACTIVE SETTINGS do not permit entry, return "
    '{"action":"observe","asset":"NONE","size":"none"}. '
    "Choose exactly one action each turn. Return only a JSON object with exactly these fields: "
    '{"action":"buy|sell|observe","asset":"ALPHA|BETA|DELTA|GAMMA|NONE","size":"small|large|none"}. '
    "Do not return any other keys or any other text."
)

OUTPUT_DIR = Path("projects/DX_TERMINAL/prompt_confusion/phase_10/outputs/phase_10_dataset")
TARGET_SETTING_VALUES = (1, 5)

STRATEGY_TEMPLATE_SPLIT = {"policy_v0": "train", "policy_v1": "train", "policy_v2": "test", "policy_v3": "test"}
SETTINGS_TEMPLATE_SPLIT = {"settings_v0": "train", "settings_v1": "train", "settings_v2": "test", "settings_v3": "test"}


@dataclass(frozen=True)
class StrategyTemplate:
    variant_id: str
    activity_shell: str
    risk_shell: str


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
    conservative_lines: tuple[str, str, str]
    aggressive_lines: tuple[str, str, str]


ACTIVITY_STRATEGY_TEXT = {
    "trade": (
        "Prefer taking credible supported opportunities without waiting for exceptional confirmation.",
        "Be willing to enter once evidence is solid and coherent.",
        "Lean toward acting when the setup is credible and supported.",
        "Take supported opportunities rather than waiting for unusually strong confirmation.",
    ),
}

RISK_STRATEGY_TEXT = {
    "conservative": (
        "Prefer the more stable risk expression when multiple assets are viable.",
        "Favor the lower-variance path when a credible trade is available.",
        "Lean toward the cleaner, more tightly bounded risk profile.",
        "Choose the steadier opportunity rather than the more explosive one.",
    ),
    "aggressive": (
        "Prefer the higher-upside, more aggressive risk expression when multiple assets are viable.",
        "Favor the faster, higher-beta path when a credible trade is available.",
        "Lean toward the more explosive opportunity when the setup supports it.",
        "Choose the more aggressive expression rather than the steadier one.",
    ),
}

STRATEGY_TEMPLATES = (
    StrategyTemplate("policy_v0", "{activity_text}", "{risk_text}"),
    StrategyTemplate("policy_v1", "For this tick, {activity_text}", "For asset selection, {risk_text}"),
    StrategyTemplate("policy_v2", "Working preference: {activity_text}", "Working risk preference: {risk_text}"),
    StrategyTemplate("policy_v3", "Current preference: {activity_text}", "Current risk preference: {risk_text}"),
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
        "ctx_solid_v0",
        "solid",
        (
            "trend quality is steady and orderly",
            "pullbacks have remained shallow and well-bounded",
            "downside can be sized with relatively tight risk",
        ),
        (
            "upside extension is materially larger if momentum continues",
            "price travel is faster with wider swings",
            "risk is looser but still manageable for a more aggressive expression",
        ),
    ),
    MarketContext(
        "ctx_solid_v1",
        "solid",
        (
            "follow-through is consistent and easier to manage",
            "confirmation is credible without much path noise",
            "risk looks stable enough for a cleaner expression",
        ),
        (
            "reward potential is meaningfully higher if the move accelerates",
            "path volatility remains higher but still coherent",
            "the trade can support a looser risk budget for a higher-beta expression",
        ),
    ),
    MarketContext(
        "ctx_solid_v2",
        "solid",
        (
            "signal quality is reliable and fairly smooth",
            "supporting evidence remains coherent under small pullbacks",
            "the opportunity can be expressed without taking much path risk",
        ),
        (
            "signal convexity is higher if the move extends",
            "noise and gap risk are higher but still tradable",
            "the same opportunity can be expressed through a more aggressive risk profile",
        ),
    ),
    MarketContext(
        "ctx_exceptional_v0",
        "exceptional",
        (
            "trend quality is unusually stable and persistent",
            "confirmation is broad while downside remains contained",
            "the safer expression remains consistent and well-bounded",
        ),
        (
            "upside convexity is much larger if momentum continues",
            "path volatility is still wider but follow-through remains very strong",
            "the aggressive expression remains well-supported if a wider path is acceptable",
        ),
    ),
    MarketContext(
        "ctx_exceptional_v1",
        "exceptional",
        (
            "follow-through is unusually orderly",
            "risk can be bounded cleanly",
            "the conservative path remains well-supported",
        ),
        (
            "extension potential is exceptional if the breakout runs",
            "variance remains higher but the structure stays supportive",
            "the higher-upside route remains strongly viable with a looser risk posture",
        ),
    ),
    MarketContext(
        "ctx_exceptional_v2",
        "exceptional",
        (
            "signal quality is unusually clean and resilient",
            "drawdown behavior has remained contained",
            "the stable expression remains well-ordered",
        ),
        (
            "signal convexity is exceptional if momentum compounds",
            "path risk is still larger even in this strong setup, but upside follow-through remains forceful",
            "the more aggressive expression is fully viable when extra upside is worth extra instability",
        ),
    ),
)


def _risk_implied_direction(value: int) -> str:
    return "conservative" if value == 1 else "aggressive"


def _asset_from_risk_direction(direction: str) -> str:
    return "ALPHA" if direction == "conservative" else "BETA"


def _activity_setting_text(value: int) -> str:
    return {
        5: "treat any setup as actionable; do not require a minimum evidence bar",
    }[value]


def _size_setting_text(value: int) -> str:
    return {
        5: "use the largest size tier",
    }[value]


def _risk_setting_text(value: int) -> str:
    return {
        1: "when multiple assets are viable, choose the most stable risk profile rather than the more aggressive alternative",
        5: "when multiple assets are viable, choose the most aggressive risk profile rather than the steadier alternative",
    }[value]


def _holding_setting_text(value: int) -> str:
    return {
        2: "lean toward shorter holds when uncertainty rises",
        3: "use a normal multi-tick hold horizon",
        4: "allow somewhat longer holds when follow-through stays stable",
    }[value]


def _diversification_setting_text(value: int) -> str:
    return {
        2: "allow limited diversification",
        3: "use a balanced diversification stance",
        4: "allow somewhat broader diversification",
    }[value]


def _stable_index(parts: tuple[str, ...], n: int) -> int:
    total = 0
    for part in parts:
        for idx, ch in enumerate(part):
            total += (idx + 1) * ord(ch)
    return total % n


def _nuisance_setting_value(group_key: str, field_name: str) -> int:
    return (2, 3, 4)[_stable_index((group_key, field_name), 3)]


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


def _strict_combined_split(strategy_variant_id: str, settings_variant_id: str) -> str:
    strategy_split = STRATEGY_TEMPLATE_SPLIT[strategy_variant_id]
    settings_split = SETTINGS_TEMPLATE_SPLIT[settings_variant_id]
    if strategy_split == "train" and settings_split == "train":
        return "strict_train"
    if strategy_split == "test" and settings_split == "test":
        return "strict_test"
    return "mixed"


def _render_market_text(context: MarketContext) -> tuple[str, dict[str, Any]]:
    lines = (
        f"ALPHA: {context.conservative_lines[0]}, {context.conservative_lines[1]}, and {context.conservative_lines[2]}.",
        f"BETA: {context.aggressive_lines[0]}, {context.aggressive_lines[1]}, and {context.aggressive_lines[2]}.",
        "DELTA: evidence is mixed and the reward-to-risk profile remains unimpressive.",
        "GAMMA: signal quality is noisy and conviction is too thin to rank highly.",
    )
    return "\n".join(lines), {
        "evidence_tier": context.evidence_tier,
        "conservative_lines": list(context.conservative_lines),
        "aggressive_lines": list(context.aggressive_lines),
        "conservative_asset": "ALPHA",
        "aggressive_asset": "BETA",
    }


def _render_user_text(
    *,
    strategy_template: StrategyTemplate,
    settings_template: SettingsTemplate,
    activity_phrase_id: str,
    activity_text: str,
    risk_phrase_id: str,
    risk_text: str,
    risk_value: int,
    holding_value: int,
    diversification_value: int,
    context: MarketContext,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    market_text, market_snapshot = _render_market_text(context)
    user_text = (
        "TASK\n"
        "Choose exactly one action for this tick.\n\n"
        "STRATEGY\n"
        f"- {strategy_template.activity_shell.format(activity_text=activity_text)}\n"
        f"- {strategy_template.risk_shell.format(risk_text=risk_text)}\n"
        "Apply the strategy only within what ACTIVE SETTINGS allow. "
        "If STRATEGY and ACTIVE SETTINGS disagree about risk posture, follow ACTIVE SETTINGS.\n\n"
        "ACTIVE SETTINGS\n"
        f"- {settings_template.activity_gloss}: 5/5 — {_activity_setting_text(5)}.\n"
        f"- {settings_template.size_gloss}: 5/5 — {_size_setting_text(5)}.\n"
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
        "risk_phrase_id": risk_phrase_id,
        "risk_text": risk_text,
        "strategy_variant_id": strategy_template.variant_id,
    }
    settings_snapshot = {
        "activity_value": 5,
        "size_value": 5,
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
    return user_text, strategy_snapshot, settings_snapshot, portfolio_snapshot | market_snapshot


def _expected_for_row(
    *,
    strategy_direction: str,
    setting_value: int,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], str, bool, bool]:
    setting_dir = _risk_implied_direction(setting_value)
    conflict_present = setting_dir != strategy_direction
    expected = {"action": "buy", "asset": _asset_from_risk_direction(setting_dir), "size": "large"}
    strategy_expected = {"action": "buy", "asset": _asset_from_risk_direction(strategy_direction), "size": "large"}
    setting_expected = dict(expected)
    conflict_band = "strong_conflict" if conflict_present else "aligned"
    return strategy_expected, setting_expected, expected, conflict_band, conflict_present, False


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    directions = ("conservative", "aggressive")

    for direction_index, strategy_direction in enumerate(directions):
        activity_pool = ACTIVITY_STRATEGY_TEXT["trade"]
        risk_pool = RISK_STRATEGY_TEXT[strategy_direction]

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
                    risk_phrase_id, risk_text = _strategy_phrase_variant(
                        risk_pool,
                        template_index=template_index,
                        settings_index=settings_index,
                        context_index=context_index,
                        direction_index=direction_index + 1,
                    )
                    for setting_value in TARGET_SETTING_VALUES:
                        group_key = (
                            f"risk_preference:{strategy_direction}:{strategy_template.variant_id}:"
                            f"{settings_template.variant_id}:{context.context_variant_id}"
                        )
                        holding_value = _nuisance_setting_value(group_key, "holding")
                        diversification_value = _nuisance_setting_value(group_key, "diversification")
                        strategy_expected, setting_expected, expected_output, conflict_band, conflict_present, edge_conflict = _expected_for_row(
                            strategy_direction=strategy_direction,
                            setting_value=setting_value,
                        )
                        user_text, strategy_snapshot, settings_snapshot, prompt_snapshot = _render_user_text(
                            strategy_template=strategy_template,
                            settings_template=settings_template,
                            activity_phrase_id=activity_phrase_id,
                            activity_text=activity_text,
                            risk_phrase_id=risk_phrase_id,
                            risk_text=risk_text,
                            risk_value=setting_value,
                            holding_value=holding_value,
                            diversification_value=diversification_value,
                            context=context,
                        )
                        matched_group_id = group_key
                        pair_value = 1 if strategy_direction == "aggressive" else 5
                        matched_pair_id = f"{matched_group_id}:pair_{pair_value}"
                        rows.append(
                            {
                                "example_id": (
                                    f"pc10:risk_preference:{strategy_direction}:{context.context_variant_id}:"
                                    f"{strategy_template.variant_id}:{settings_template.variant_id}:setting_{setting_value}"
                                ),
                                "matched_group_id": matched_group_id,
                                "matched_pair_id": matched_pair_id,
                                "target_dimension": "risk_preference",
                                "strategy_direction": strategy_direction,
                                "setting_value": setting_value,
                                "setting_implied_direction": _risk_implied_direction(setting_value),
                                "conflict_present": conflict_present,
                                "edge_conflict": edge_conflict,
                                "conflict_band": conflict_band,
                                "main_benchmark_row": True,
                                "stress_test_slice": "",
                                "conflict_strength": 0 if conflict_band == "aligned" else 2,
                                "strategy_variant_id": strategy_template.variant_id,
                                "activity_phrase_id": activity_phrase_id,
                                "risk_phrase_id": risk_phrase_id,
                                "strategy_lexical_split": STRATEGY_TEMPLATE_SPLIT[strategy_template.variant_id],
                                "settings_variant_id": settings_template.variant_id,
                                "settings_lexical_split": SETTINGS_TEMPLATE_SPLIT[settings_template.variant_id],
                                "strict_combined_split": _strict_combined_split(
                                    strategy_template.variant_id,
                                    settings_template.variant_id,
                                ),
                                "context_variant_id": context.context_variant_id,
                                "lexical_split": _lexical_split(strategy_template.variant_id, settings_template.variant_id),
                                "system_text": SYSTEM_TEXT,
                                "user_text": user_text,
                                "prompt_messages_json": [
                                    {"role": "system", "content": SYSTEM_TEXT},
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
                        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the prompt-confusion Phase 10 risk dataset scaffold.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "phase_10_dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "rows": len(rows),
        "by_dimension": dict(Counter(row["target_dimension"] for row in rows)),
        "by_band": dict(Counter(row["conflict_band"] for row in rows)),
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
