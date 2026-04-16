from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SYSTEM_TEXT = (
    "You are a trading agent. Each prompt contains a STRATEGY and ACTIVE SETTINGS. "
    "ACTIVE SETTINGS are binding execution constraints. STRATEGY expresses preferences "
    "that apply only within what ACTIVE SETTINGS allow. "
    "Decide in this order: first whether ACTIVE SETTINGS permit entry, then which asset "
    "best matches the allowed risk profile from MARKET, then what size ACTIVE SETTINGS allow. "
    "If STRATEGY and ACTIVE SETTINGS disagree about size or risk posture, follow ACTIVE SETTINGS. "
    'If ACTIVE SETTINGS do not permit entry, return {"action":"observe","asset":"NONE","size":"none"}. '
    "Choose exactly one action each turn. Return only a JSON object with exactly these fields: "
    '{"action":"buy|sell|observe","asset":"ALPHA|BETA|DELTA|GAMMA|NONE","size":"small|large|none"}. '
    "Do not return any other keys or any other text."
)

OUTPUT_DIR = Path("projects/DX_TERMINAL/prompt_confusion/phase_11/outputs/phase_11_dataset")
SIZE_SETTING_VALUES = (1, 5)
RISK_SETTING_VALUES = (1, 5)

STRATEGY_TEMPLATE_SPLIT = {"policy_v0": "train", "policy_v1": "train", "policy_v2": "test", "policy_v3": "test"}
SETTINGS_TEMPLATE_SPLIT = {"settings_v0": "train", "settings_v1": "train", "settings_v2": "test", "settings_v3": "test"}


@dataclass(frozen=True)
class StrategyTemplate:
    variant_id: str
    activity_shell: str
    size_shell: str
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

SIZE_STRATEGY_TEXT = {
    "small": (
        "Prefer the smaller size tier unless settings explicitly allow more size.",
        "Favor the more controlled position size when entering a credible trade.",
        "Lean toward the smaller expression rather than pressing size.",
        "Choose the smaller size unless ACTIVE SETTINGS explicitly call for larger exposure.",
    ),
    "large": (
        "Prefer the larger size tier when the setup is credible and settings allow it.",
        "Favor the larger expression rather than the smaller one when entry is supported.",
        "Lean toward pressing size when ACTIVE SETTINGS permit it.",
        "Choose the larger size rather than the smaller expression when allowed.",
    ),
}

RISK_STRATEGY_TEXT = {
    "conservative": (
        "Prefer the more stable risk expression when assets are both viable.",
        "Favor the lower-variance path when a credible trade is available.",
        "Lean toward the cleaner, more tightly bounded risk profile.",
        "Choose the steadier opportunity rather than the more explosive one.",
    ),
    "aggressive": (
        "Prefer the higher-upside, more aggressive risk expression when assets are both viable.",
        "Favor the faster, higher-beta path when a credible trade is available.",
        "Lean toward the more explosive opportunity when the setup supports it.",
        "Choose the more aggressive expression rather than the steadier one.",
    ),
}

STRATEGY_TEMPLATES = (
    StrategyTemplate("policy_v0", "{activity_text}", "{size_text}", "{risk_text}"),
    StrategyTemplate("policy_v1", "For this tick, {activity_text}", "For size, {size_text}", "For asset choice, {risk_text}"),
    StrategyTemplate("policy_v2", "Working preference: {activity_text}", "Working size preference: {size_text}", "Working risk preference: {risk_text}"),
    StrategyTemplate("policy_v3", "Current preference: {activity_text}", "Current size preference: {size_text}", "Current risk preference: {risk_text}"),
)

SETTINGS_TEMPLATES = (
    SettingsTemplate("settings_v0", "Trading Activity", "Trade Size", "Asset Risk Preference", "Holding Style", "Diversification"),
    SettingsTemplate("settings_v1", "Execution Activity", "Execution Size", "Risk Preference", "Holding Horizon", "Diversification Preference"),
    SettingsTemplate("settings_v2", "Activity Constraint", "Size Constraint", "Risk Constraint", "Hold Constraint", "Diversification Constraint"),
    SettingsTemplate("settings_v3", "Activity Setting", "Size Setting", "Risk Setting", "Hold Setting", "Diversification Setting"),
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


def _size_implied_direction(value: int) -> str:
    return "small" if value == 1 else "large"


def _risk_implied_direction(value: int) -> str:
    return "conservative" if value == 1 else "aggressive"


def _asset_from_risk_direction(direction: str) -> str:
    return "ALPHA" if direction == "conservative" else "BETA"


def _activity_setting_text(value: int) -> str:
    return {5: "treat any setup as actionable; do not require a minimum evidence bar"}[value]


def _size_setting_text(value: int) -> str:
    return {
        1: "use the smallest size tier rather than scaling up",
        5: "use the largest size tier rather than the smaller alternative",
    }[value]


def _risk_setting_text(value: int) -> str:
    return {
        1: "choose the most stable risk profile rather than the more aggressive alternative",
        5: "choose the most aggressive risk profile rather than the steadier alternative",
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
    return "train" if STRATEGY_TEMPLATE_SPLIT[strategy_variant_id] == SETTINGS_TEMPLATE_SPLIT[settings_variant_id] else "test"


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
    size_phrase_id: str,
    size_text: str,
    risk_phrase_id: str,
    risk_text: str,
    size_value: int,
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
        f"- {strategy_template.size_shell.format(size_text=size_text)}\n"
        f"- {strategy_template.risk_shell.format(risk_text=risk_text)}\n"
        "Apply the strategy only within what ACTIVE SETTINGS allow. "
        "If STRATEGY and ACTIVE SETTINGS disagree about size or risk posture, follow ACTIVE SETTINGS.\n\n"
        "ACTIVE SETTINGS\n"
        f"- {settings_template.activity_gloss}: 5/5 — {_activity_setting_text(5)}.\n"
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
        "risk_phrase_id": risk_phrase_id,
        "risk_text": risk_text,
        "strategy_variant_id": strategy_template.variant_id,
    }
    settings_snapshot = {
        "activity_value": 5,
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
    return user_text, strategy_snapshot, settings_snapshot, portfolio_snapshot | market_snapshot


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    size_directions = ("small", "large")
    risk_directions = ("conservative", "aggressive")

    for size_index, strategy_size_direction in enumerate(size_directions):
        for risk_index, strategy_risk_direction in enumerate(risk_directions):
            activity_pool = ACTIVITY_STRATEGY_TEXT["trade"]
            size_pool = SIZE_STRATEGY_TEXT[strategy_size_direction]
            risk_pool = RISK_STRATEGY_TEXT[strategy_risk_direction]

            for template_index, strategy_template in enumerate(STRATEGY_TEMPLATES):
                for settings_index, settings_template in enumerate(SETTINGS_TEMPLATES):
                    for context_index, context in enumerate(MARKET_CONTEXTS):
                        activity_phrase_id, activity_text = _strategy_phrase_variant(
                            activity_pool,
                            template_index=template_index,
                            settings_index=settings_index,
                            context_index=context_index,
                            direction_index=size_index + risk_index,
                        )
                        size_phrase_id, size_text = _strategy_phrase_variant(
                            size_pool,
                            template_index=template_index,
                            settings_index=settings_index,
                            context_index=context_index,
                            direction_index=size_index,
                        )
                        risk_phrase_id, risk_text = _strategy_phrase_variant(
                            risk_pool,
                            template_index=template_index,
                            settings_index=settings_index,
                            context_index=context_index,
                            direction_index=risk_index + 1,
                        )
                        for size_value in SIZE_SETTING_VALUES:
                            for risk_value in RISK_SETTING_VALUES:
                                resolved_size = _size_implied_direction(size_value)
                                resolved_risk = _risk_implied_direction(risk_value)
                                size_conflict = resolved_size != strategy_size_direction
                                risk_conflict = resolved_risk != strategy_risk_direction
                                conflict_count = int(size_conflict) + int(risk_conflict)
                                conflict_band = {
                                    0: "aligned",
                                    1: "single_conflict",
                                    2: "double_conflict",
                                }[conflict_count]
                                group_key = (
                                    f"multi_conflict:{strategy_size_direction}:{strategy_risk_direction}:"
                                    f"{strategy_template.variant_id}:{settings_template.variant_id}:{context.context_variant_id}"
                                )
                                holding_value = _nuisance_setting_value(group_key, "holding")
                                diversification_value = _nuisance_setting_value(group_key, "diversification")
                                user_text, strategy_snapshot, settings_snapshot, prompt_snapshot = _render_user_text(
                                    strategy_template=strategy_template,
                                    settings_template=settings_template,
                                    activity_phrase_id=activity_phrase_id,
                                    activity_text=activity_text,
                                    size_phrase_id=size_phrase_id,
                                    size_text=size_text,
                                    risk_phrase_id=risk_phrase_id,
                                    risk_text=risk_text,
                                    size_value=size_value,
                                    risk_value=risk_value,
                                    holding_value=holding_value,
                                    diversification_value=diversification_value,
                                    context=context,
                                )
                                expected_output = {
                                    "action": "buy",
                                    "asset": _asset_from_risk_direction(resolved_risk),
                                    "size": resolved_size,
                                }
                                rows.append(
                                    {
                                        "example_id": (
                                            f"pc11:multi_conflict:{strategy_size_direction}:{strategy_risk_direction}:"
                                            f"{context.context_variant_id}:{strategy_template.variant_id}:"
                                            f"{settings_template.variant_id}:size_{size_value}:risk_{risk_value}"
                                        ),
                                        "matched_group_id": group_key,
                                        "matched_pair_id": (
                                            f"{group_key}:pair_size_{size_value}:risk_{risk_value}"
                                        ),
                                        "target_dimension": "multi_conflict",
                                        "strategy_size_direction": strategy_size_direction,
                                        "strategy_risk_direction": strategy_risk_direction,
                                        "size_setting_value": size_value,
                                        "risk_setting_value": risk_value,
                                        "size_setting_implied_direction": resolved_size,
                                        "risk_setting_implied_direction": resolved_risk,
                                        "size_conflict_present": size_conflict,
                                        "risk_conflict_present": risk_conflict,
                                        "any_conflict_present": conflict_count > 0,
                                        "double_conflict_present": conflict_count == 2,
                                        "conflict_count": conflict_count,
                                        "edge_conflict": False,
                                        "conflict_band": conflict_band,
                                        "main_benchmark_row": True,
                                        "stress_test_slice": "",
                                        "conflict_strength": conflict_count,
                                        "strategy_variant_id": strategy_template.variant_id,
                                        "activity_phrase_id": activity_phrase_id,
                                        "size_phrase_id": size_phrase_id,
                                        "risk_phrase_id": risk_phrase_id,
                                        "strategy_lexical_split": STRATEGY_TEMPLATE_SPLIT[strategy_template.variant_id],
                                        "settings_variant_id": settings_template.variant_id,
                                        "settings_lexical_split": SETTINGS_TEMPLATE_SPLIT[settings_template.variant_id],
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
                                        "expected_output_json": expected_output,
                                    }
                                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the prompt-confusion Phase 11 multi-conflict dataset scaffold.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "phase_11_dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "rows": len(rows),
        "by_band": dict(Counter(row["conflict_band"] for row in rows)),
        "by_conflict_count": dict(Counter(row["conflict_count"] for row in rows)),
        "by_lexical_split": dict(Counter(row["lexical_split"] for row in rows)),
        "by_size_conflict": dict(Counter(row["size_conflict_present"] for row in rows)),
        "by_risk_conflict": dict(Counter(row["risk_conflict_present"] for row in rows)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows()
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
