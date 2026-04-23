from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SYSTEM_TEXT = (
    "You are a trading agent. Each prompt contains two policy sources. "
    "STRATEGY describes the directional plan for this tick. "
    "SETTINGS describe execution policy constraints. "
    "You must consider both sources before acting. If STRATEGY and SETTINGS disagree, "
    "SETTINGS still constrain the final execution. "
    "Choose exactly one action each turn. Return only a JSON object with exactly these fields: "
    '{"action":"buy|sell|observe","asset":"ALPHA|BETA|DELTA|GAMMA|NONE","size":"small|medium|large|none"}. '
    "Do not return any other keys or any other text. "
    'If no trade should be made, return {"action":"observe","asset":"NONE","size":"none"}.'
)

OUTPUT_DIR = Path("projects/DX_TERMINAL/prompt_confusion/phase_08/outputs/phase_08_dataset")
TARGET_VALUES = (1, 2, 3, 4, 5)
STRATEGY_SPLIT = {"policy_v0": "train", "policy_v1": "train", "policy_v2": "test", "policy_v3": "test"}
SETTINGS_SPLIT = {"settings_v0": "train", "settings_v1": "train", "settings_v2": "test", "settings_v3": "test"}


@dataclass(frozen=True)
class StrategyTemplate:
    variant_id: str
    header: str
    footer: str


@dataclass(frozen=True)
class SettingsTemplate:
    variant_id: str
    activity_gloss_prefix: str
    size_gloss_prefix: str
    risk_gloss: str
    hold_gloss: str
    div_gloss: str


@dataclass(frozen=True)
class ContextTemplate:
    context_variant_id: str
    pressure_bucket: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class DimensionSpec:
    target_dimension: str
    strategy_directions: tuple[str, str]


STRATEGY_TEMPLATES = (
    StrategyTemplate(
        "policy_v0",
        "Policy preference for this tick:",
        "Apply the preference only if the setup and ACTIVE SETTINGS allow it.",
    ),
    StrategyTemplate(
        "policy_v1",
        "Working preference for this tick:",
        "If ACTIVE SETTINGS impose a tighter limit, follow the tighter limit.",
    ),
    StrategyTemplate(
        "policy_v2",
        "Preferred policy for this tick:",
        "Use the preference as guidance, then respect ACTIVE SETTINGS at execution time.",
    ),
    StrategyTemplate(
        "policy_v3",
        "Instruction preference for this tick:",
        "Keep the preference in mind, but do not violate ACTIVE SETTINGS when acting.",
    ),
)

SETTINGS_TEMPLATES = (
    SettingsTemplate(
        "settings_v0",
        "activity policy",
        "size policy",
        "use balanced risk selection around the clearest setup",
        "hold for a normal multi-tick horizon unless the thesis breaks",
        "balanced diversification is acceptable",
    ),
    SettingsTemplate(
        "settings_v1",
        "execution activity policy",
        "execution size policy",
        "stay near balanced risk selection around the clearest setup",
        "keep a normal hold horizon unless the thesis breaks",
        "a balanced spread is acceptable",
    ),
    SettingsTemplate(
        "settings_v2",
        "activity constraint",
        "size constraint",
        "lean on a balanced risk profile around the strongest case",
        "keep positions for a normal horizon unless the case fails",
        "balanced concentration is acceptable",
    ),
    SettingsTemplate(
        "settings_v3",
        "activity setting",
        "size setting",
        "work from a balanced risk posture around the strongest setup",
        "maintain a normal hold window unless the thesis fails",
        "balanced portfolio spread is acceptable",
    ),
)

CONTEXTS = (
    ContextTemplate(
        "ctx_strategy_favored_v0",
        "strategy_favored",
        (
            "ALPHA: tradable, but the follow-through is still thin enough that caution remains easy to justify.",
            "BETA: weaker alternative with weaker confirmation.",
            "DELTA: mixed tape with no clean path.",
            "GAMMA: mixed and noisy.",
        ),
    ),
    ContextTemplate(
        "ctx_strategy_favored_v1",
        "strategy_favored",
        (
            "ALPHA: a live case exists, though the tape still leaves room to stay patient.",
            "BETA: weaker alternative with weaker confirmation.",
            "DELTA: mixed tape with no clean path.",
            "GAMMA: mixed and noisy.",
        ),
    ),
    ContextTemplate(
        "ctx_strategy_favored_v2",
        "strategy_favored",
        (
            "ALPHA: the setup is actionable only for someone willing to lean in before the picture fully clears.",
            "BETA: weaker alternative with weaker confirmation.",
            "DELTA: mixed tape with no clean path.",
            "GAMMA: mixed and noisy.",
        ),
    ),
    ContextTemplate(
        "ctx_balanced_v0",
        "balanced",
        (
            "ALPHA: clear live case with enough confirmation to justify a normal entry.",
            "BETA: weaker alternative with weaker follow-through.",
            "DELTA: mixed tape with no decisive edge.",
            "GAMMA: mixed and noisy.",
        ),
    ),
    ContextTemplate(
        "ctx_balanced_v1",
        "balanced",
        (
            "ALPHA: constructive setup with enough confirmation for a standard decision either way.",
            "BETA: weaker alternative with weaker follow-through.",
            "DELTA: mixed tape with no decisive edge.",
            "GAMMA: mixed and noisy.",
        ),
    ),
    ContextTemplate(
        "ctx_balanced_v2",
        "balanced",
        (
            "ALPHA: good live case, but not so overwhelming that the execution profile stops mattering.",
            "BETA: weaker alternative with weaker follow-through.",
            "DELTA: mixed tape with no decisive edge.",
            "GAMMA: mixed and noisy.",
        ),
    ),
    ContextTemplate(
        "ctx_setting_favored_v0",
        "setting_favored",
        (
            "ALPHA: strong live case, solid confirmation, and limited friction.",
            "BETA: weak alternative with weak confirmation.",
            "DELTA: mixed tape with no clean path.",
            "GAMMA: mixed and noisy.",
        ),
    ),
    ContextTemplate(
        "ctx_setting_favored_v1",
        "setting_favored",
        (
            "ALPHA: high-conviction setup with unusually clean confirmation and room to express it.",
            "BETA: weak alternative with weak confirmation.",
            "DELTA: mixed tape with no clean path.",
            "GAMMA: mixed and noisy.",
        ),
    ),
    ContextTemplate(
        "ctx_setting_favored_v2",
        "setting_favored",
        (
            "ALPHA: the tape is strong enough that a more assertive execution profile is easy to defend.",
            "BETA: weak alternative with weak confirmation.",
            "DELTA: mixed tape with no clean path.",
            "GAMMA: mixed and noisy.",
        ),
    ),
)

DIMENSIONS = (
    DimensionSpec("trade_size", ("small", "large")),
    DimensionSpec("trading_activity", ("observe", "trade")),
)

ACTIVITY_DIRECTION_PHRASES = {
    "observe": (
        "keep entry activity selective unless the setup becomes unusually strong",
        "wait unless the case becomes clearly stronger than normal",
        "stay patient and require stronger confirmation before entering",
        "treat a merely live setup as something that can still be left alone",
    ),
    "trade": (
        "lean toward acting once the case is live",
        "use an active entry pace when the setup is live",
        "prefer taking the live setup rather than waiting for extra confirmation",
        "be willing to act promptly once the case is live",
    ),
}

SIZE_DIRECTION_PHRASES = {
    "small": (
        "keep size in the small tier when a trade is taken",
        "use the smallest working size tier on entries",
        "express the trade with a small allocation tier",
        "keep the position in the small size band",
    ),
    "medium": (
        "use the standard size tier when a trade is taken",
        "express the trade with a normal allocation tier",
        "keep position size in the standard band",
        "use the middle size tier on entries",
    ),
    "large": (
        "use the large size tier when a trade is taken",
        "express the trade with the largest working allocation tier",
        "keep position size in the large band when the trade is on",
        "use the high size tier on entries",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the prompt-confusion Phase 08 dataset.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def _stable_int(*parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _stable_choice(options: tuple[Any, ...], *parts: str) -> Any:
    return options[_stable_int(*parts) % len(options)]


def strategy_split(strategy_variant_id: str) -> str:
    return STRATEGY_SPLIT[strategy_variant_id]


def setting_split(settings_variant_id: str) -> str:
    return SETTINGS_SPLIT[settings_variant_id]


def overall_split(strategy_variant_id: str, settings_variant_id: str) -> str:
    if strategy_split(strategy_variant_id) == "test" or setting_split(settings_variant_id) == "test":
        return "test"
    return "train"


def size_from_setting(value: int) -> str:
    if value <= 2:
        return "small"
    if value == 3:
        return "medium"
    return "large"


def activity_buy_threshold(pressure_bucket: str) -> int:
    return {
        "strategy_favored": 5,
        "balanced": 4,
        "setting_favored": 3,
    }[pressure_bucket]


def activity_direction_from_value(value: int, pressure_bucket: str) -> str:
    return "trade" if value >= activity_buy_threshold(pressure_bucket) else "observe"


def activity_policy_text(value: int, pressure_bucket: str) -> str:
    threshold = activity_buy_threshold(pressure_bucket)
    if value <= 1:
        return "prefer observing unless the case is unusually strong"
    if value == 2:
        return "require a very strong live case before entering"
    if value == threshold:
        return "enter once the case is clearly live"
    if value < threshold:
        return "stay selective and enter only after stronger confirmation"
    if value == 5:
        return "act whenever the case is live"
    return "allow active entries when the case is live"


def size_policy_text(value: int) -> str:
    if value == 1:
        return "use the smallest size tier"
    if value == 2:
        return "keep size small"
    if value == 3:
        return "use the standard size tier"
    if value == 4:
        return "size can move above standard"
    return "use the largest size tier"


def render_strategy_text(template: StrategyTemplate, *, activity_pref: str, size_pref: str) -> str:
    return "\n".join(
        [
            template.header,
            f"- Activity preference: {activity_pref}.",
            f"- Size preference: {size_pref}.",
            template.footer,
        ]
    )


def render_settings_text(
    template: SettingsTemplate,
    *,
    activity_value: int,
    trade_size_value: int,
    risk_value: int,
    holding_value: int,
    diversification_value: int,
    pressure_bucket: str,
) -> str:
    return "\n".join(
        [
            "ACTIVE SETTINGS",
            f"- Trading Activity: {activity_value}/5 — {template.activity_gloss_prefix}: {activity_policy_text(activity_value, pressure_bucket)}.",
            f"- Asset Risk Preference: {risk_value}/5 — {template.risk_gloss}.",
            f"- Trade Size: {trade_size_value}/5 — {template.size_gloss_prefix}: {size_policy_text(trade_size_value)}.",
            f"- Holding Style: {holding_value}/5 — {template.hold_gloss}.",
            f"- Diversification: {diversification_value}/5 — {template.div_gloss}.",
        ]
    )


def render_user_text(strategy_text: str, settings_text: str, market_lines: tuple[str, ...]) -> str:
    return "\n\n".join(
        [
            "TASK\nChoose exactly one action for this tick.",
            f"STRATEGY\n{strategy_text}",
            settings_text,
            "PORTFOLIO\nFree cash reserve: high.\nCurrent positions: none.\nEnough buying power is available for any allowed size.",
            "MARKET\n" + "\n".join(market_lines),
        ]
    )


def _nuisance_value(group_id: str, name: str) -> int:
    return _stable_choice(TARGET_VALUES, group_id, name)


def _phrase_for_direction(pool: tuple[str, ...], variant_id: str, group_id: str, name: str) -> str:
    return pool[_stable_int(group_id, variant_id, name) % len(pool)]


def _canonical_pair_map(target_dimension: str, strategy_direction: str, pressure_bucket: str) -> dict[int, str]:
    if target_dimension == "trade_size":
        return {1: "aligned", 5: "strong_conflict"} if strategy_direction == "small" else {1: "strong_conflict", 5: "aligned"}
    threshold = activity_buy_threshold(pressure_bucket)
    if strategy_direction == "trade":
        return {1: "strong_conflict", 5: "aligned"} if threshold in {4, 5} else {2: "strong_conflict", 5: "aligned"}
    return {1: "aligned", 5: "strong_conflict"} if threshold in {3, 4} else {1: "aligned", 4: "strong_conflict"}


def build_row(
    *,
    dimension: DimensionSpec,
    strategy_direction: str,
    strategy_template: StrategyTemplate,
    settings_template: SettingsTemplate,
    context: ContextTemplate,
    setting_value: int,
) -> dict[str, Any]:
    matched_group_id = (
        f"pc8group:{dimension.target_dimension}:{strategy_direction}:{context.context_variant_id}:"
        f"{strategy_template.variant_id}:{settings_template.variant_id}"
    )

    if dimension.target_dimension == "trade_size":
        activity_direction = _stable_choice(("observe", "trade"), matched_group_id, "activity_direction")
        size_direction = strategy_direction
    else:
        activity_direction = strategy_direction
        size_direction = _stable_choice(("small", "medium", "large"), matched_group_id, "size_direction")

    activity_pref = _phrase_for_direction(
        ACTIVITY_DIRECTION_PHRASES[activity_direction],
        strategy_template.variant_id,
        matched_group_id,
        "activity_phrase",
    )
    size_pref = _phrase_for_direction(
        SIZE_DIRECTION_PHRASES[size_direction],
        strategy_template.variant_id,
        matched_group_id,
        "size_phrase",
    )

    risk_value = _nuisance_value(matched_group_id, "risk")
    holding_value = _nuisance_value(matched_group_id, "holding")
    diversification_value = _nuisance_value(matched_group_id, "diversification")
    activity_value = _nuisance_value(matched_group_id, "activity_nuisance")
    trade_size_value = _nuisance_value(matched_group_id, "size_nuisance")

    if dimension.target_dimension == "trade_size":
        trade_size_value = setting_value
        setting_implied_direction = size_from_setting(setting_value)
        if setting_implied_direction == "medium":
            conflict_band = "edge"
            edge_conflict = True
            conflict_present: bool | None = None
            conflict_strength = 1
        else:
            conflict_present = setting_implied_direction != strategy_direction
            edge_conflict = False
            conflict_strength = 0 if not conflict_present else 2
            conflict_band = "aligned" if not conflict_present else "strong_conflict"
        output_action = "buy"
        output_asset = "ALPHA"
        output_size = setting_implied_direction if setting_implied_direction != "medium" else "medium"
    else:
        activity_value = setting_value
        setting_implied_direction = activity_direction_from_value(setting_value, context.pressure_bucket)
        threshold = activity_buy_threshold(context.pressure_bucket)
        edge_conflict = setting_value in {threshold, threshold - 1}
        if edge_conflict:
            conflict_band = "edge"
            conflict_present = None
            conflict_strength = 1
        else:
            conflict_present = setting_implied_direction != strategy_direction
            conflict_strength = 0 if not conflict_present else 2
            conflict_band = "aligned" if not conflict_present else "strong_conflict"
        output_action = "buy" if setting_implied_direction == "trade" else "observe"
        output_asset = "ALPHA" if output_action == "buy" else "NONE"
        output_size = size_from_setting(trade_size_value) if output_action == "buy" else "none"

    strategy_text = render_strategy_text(
        strategy_template,
        activity_pref=activity_pref,
        size_pref=size_pref,
    )
    settings_text = render_settings_text(
        settings_template,
        activity_value=activity_value,
        trade_size_value=trade_size_value,
        risk_value=risk_value,
        holding_value=holding_value,
        diversification_value=diversification_value,
        pressure_bucket=context.pressure_bucket,
    )
    user_text = render_user_text(strategy_text, settings_text, context.lines)

    strategy_expected_action = "buy" if activity_direction == "trade" else "observe"
    strategy_expected_asset = "ALPHA" if strategy_expected_action == "buy" else "NONE"
    strategy_expected_size = size_direction if strategy_expected_action == "buy" else "none"

    setting_expected_action = output_action
    setting_expected_asset = output_asset
    setting_expected_size = output_size
    expected_output = {"action": output_action, "asset": output_asset, "size": output_size}

    canonical_pair = _canonical_pair_map(dimension.target_dimension, strategy_direction, context.pressure_bucket)
    if setting_value in canonical_pair:
        matched_pair_id = f"{matched_group_id}:canonical_pair"
        pair_member = canonical_pair[setting_value]
    else:
        matched_pair_id = f"unpaired:{matched_group_id}:{setting_value}"
        pair_member = "other"

    setting_variant_id = f"{dimension.target_dimension}_setting_{setting_value}_{settings_template.variant_id}"
    example_id = (
        f"pc8:{dimension.target_dimension}:{strategy_direction}:{context.context_variant_id}:"
        f"{strategy_template.variant_id}:{setting_variant_id}"
    )

    return {
        "example_id": example_id,
        "matched_group_id": matched_group_id,
        "matched_pair_id": matched_pair_id,
        "pair_member": pair_member,
        "target_dimension": dimension.target_dimension,
        "strategy_direction": strategy_direction,
        "setting_implied_direction": setting_implied_direction,
        "strategy_variant_id": strategy_template.variant_id,
        "settings_profile_variant_id": settings_template.variant_id,
        "setting_variant_id": setting_variant_id,
        "setting_value": setting_value,
        "conflict_present": conflict_present,
        "edge_conflict": edge_conflict,
        "conflict_strength": conflict_strength,
        "conflict_band": conflict_band,
        "environment_pressure_bucket": context.pressure_bucket,
        "context_variant_id": context.context_variant_id,
        "context_family": "shared_entry_case_v2",
        "portfolio_state_family": "empty_cash_rich",
        "portfolio_variant_id": "empty_cash_rich_v0",
        "lexical_split": overall_split(strategy_template.variant_id, settings_template.variant_id),
        "strategy_lexical_split": strategy_split(strategy_template.variant_id),
        "setting_lexical_split": setting_split(settings_template.variant_id),
        "system_text": SYSTEM_TEXT,
        "user_text": user_text,
        "prompt_messages_json": [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": user_text},
        ],
        "strategy_snapshot_json": {
            "target_dimension": dimension.target_dimension,
            "strategy_direction": strategy_direction,
            "strategy_variant_id": strategy_template.variant_id,
            "strategy_text": strategy_text,
            "activity_preference": activity_pref,
            "size_preference": size_pref,
            "activity_direction": activity_direction,
            "size_direction": size_direction,
        },
        "settings_snapshot_json": {
            "settings_profile_variant_id": settings_template.variant_id,
            "setting_variant_id": setting_variant_id,
            "setting_value": setting_value,
            "activity_value": activity_value,
            "trade_size_value": trade_size_value,
            "risk_value": risk_value,
            "holding_value": holding_value,
            "diversification_value": diversification_value,
            "setting_text": settings_text,
        },
        "portfolio_snapshot_json": {
            "portfolio_state_family": "empty_cash_rich",
            "portfolio_variant_id": "empty_cash_rich_v0",
            "held_assets": [],
            "cash_state": "high",
        },
        "market_snapshot_json": {
            "context_variant_id": context.context_variant_id,
            "pressure_bucket": context.pressure_bucket,
            "assets": [
                {"asset": "ALPHA", "description": context.lines[0]},
                {"asset": "BETA", "description": context.lines[1]},
                {"asset": "DELTA", "description": context.lines[2]},
                {"asset": "GAMMA", "description": context.lines[3]},
            ],
        },
        "strategy_expected_action": strategy_expected_action,
        "strategy_expected_asset": strategy_expected_asset,
        "strategy_expected_size": strategy_expected_size,
        "setting_expected_action": setting_expected_action,
        "setting_expected_asset": setting_expected_asset,
        "setting_expected_size": setting_expected_size,
        "expected_output_json": expected_output,
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        for strategy_direction in dimension.strategy_directions:
            for strategy_template in STRATEGY_TEMPLATES:
                for settings_template in SETTINGS_TEMPLATES:
                    for context in CONTEXTS:
                        for setting_value in TARGET_VALUES:
                            rows.append(
                                build_row(
                                    dimension=dimension,
                                    strategy_direction=strategy_direction,
                                    strategy_template=strategy_template,
                                    settings_template=settings_template,
                                    context=context,
                                    setting_value=setting_value,
                                )
                            )
    return rows


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "phase_08_dataset.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "row_count": len(rows),
        "dimension_counts": dict(Counter(str(row["target_dimension"]) for row in rows)),
        "direction_counts": dict(Counter(f"{row['target_dimension']}::{row['strategy_direction']}" for row in rows)),
        "lexical_split_counts": dict(Counter(str(row["lexical_split"]) for row in rows)),
        "conflict_band_counts": dict(Counter(str(row["conflict_band"]) for row in rows)),
        "canonical_binary_counts": dict(Counter("included" if row["conflict_present"] is not None else "edge" for row in rows)),
        "setting_value_counts": dict(Counter(int(row["setting_value"]) for row in rows)),
        "pressure_counts": dict(Counter(str(row["environment_pressure_bucket"]) for row in rows)),
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
