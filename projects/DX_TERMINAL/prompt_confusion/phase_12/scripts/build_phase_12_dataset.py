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
    "best matches the allowed diversification posture given PORTFOLIO and MARKET, then what size ACTIVE SETTINGS allow. "
    "If STRATEGY and ACTIVE SETTINGS disagree about diversification posture, follow ACTIVE SETTINGS. "
    'If ACTIVE SETTINGS do not permit entry, return {"action":"observe","asset":"NONE","size":"none"}. '
    "Choose exactly one action each turn. Return only a JSON object with exactly these fields: "
    '{"action":"buy|sell|observe","asset":"ALPHA|BETA|DELTA|GAMMA|NONE","size":"small|large|none"}. '
    "Do not return any other keys or any other text."
)

OUTPUT_DIR = Path("projects/DX_TERMINAL/prompt_confusion/phase_12/outputs/phase_12_dataset")
TARGET_SETTING_VALUES = (1, 5)

STRATEGY_TEMPLATE_SPLIT = {
    "policy_v0": "train",
    "policy_v1": "train",
    "policy_v2": "test",
    "policy_v3": "test",
}
SETTINGS_TEMPLATE_SPLIT = {
    "settings_v0": "train",
    "settings_v1": "train",
    "settings_v2": "test",
    "settings_v3": "test",
}


@dataclass(frozen=True)
class StrategyTemplate:
    variant_id: str
    activity_shell: str
    diversification_shell: str


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
    concentrated_lines: tuple[str, str, str]
    diversified_lines: tuple[str, str, str]


ACTIVITY_STRATEGY_TEXT = {
    "trade": (
        "Prefer taking credible supported opportunities without waiting for exceptional confirmation.",
        "Be willing to enter once evidence is solid and coherent.",
        "Lean toward acting when the setup is credible and supported.",
        "Take supported opportunities rather than waiting for unusually strong confirmation.",
    ),
}

DIVERSIFICATION_STRATEGY_TEXT = {
    "diversified": (
        "Prefer broadening the book into a distinct sleeve when multiple trades are credible.",
        "Favor the viable setup that diversifies current exposure rather than stacking into the same sleeve.",
        "Lean toward the asset that adds differentiated exposure to the current book.",
        "Choose the opportunity that broadens the portfolio rather than adding to existing overlap.",
    ),
    "concentrated": (
        "Prefer pressing the strongest sleeve even if it overlaps the current book.",
        "Favor concentrating into the highest-conviction setup rather than broadening exposure.",
        "Lean toward adding to the existing leadership sleeve when the setup is strongest there.",
        "Choose the best overlapping opportunity rather than prioritizing diversification.",
    ),
}

STRATEGY_TEMPLATES = (
    StrategyTemplate("policy_v0", "{activity_text}", "{diversification_text}"),
    StrategyTemplate("policy_v1", "For this tick, {activity_text}", "For portfolio construction, {diversification_text}"),
    StrategyTemplate("policy_v2", "Working preference: {activity_text}", "Working diversification preference: {diversification_text}"),
    StrategyTemplate("policy_v3", "Current preference: {activity_text}", "Current portfolio preference: {diversification_text}"),
)

SETTINGS_TEMPLATES = (
    SettingsTemplate("settings_v0", "Trading Activity", "Trade Size", "Risk Preference", "Holding Style", "Diversification"),
    SettingsTemplate("settings_v1", "Execution Activity", "Execution Size", "Risk Posture", "Holding Horizon", "Portfolio Diversification"),
    SettingsTemplate("settings_v2", "Activity Constraint", "Size Constraint", "Risk Constraint", "Hold Constraint", "Diversification Constraint"),
    SettingsTemplate("settings_v3", "Activity Setting", "Size Setting", "Risk Setting", "Hold Setting", "Diversification Setting"),
)

MARKET_CONTEXTS = (
    MarketContext(
        "ctx_solid_v0",
        "solid",
        (
            "follow-through is the strongest in the same leadership sleeve the book already owns",
            "confirmation is especially clean and the opportunity ranks near the top of the current tape",
            "taking it would add directly to the existing OMEGA / SIGMA cluster",
        ),
        (
            "the setup is still credible and sufficiently strong to buy",
            "its catalyst path is more distinct from the current book",
            "taking it would broaden exposure into a separate sleeve rather than deepen the existing cluster",
        ),
    ),
    MarketContext(
        "ctx_solid_v1",
        "solid",
        (
            "signal quality is strongest in the sleeve the portfolio already leans on",
            "reward-to-risk looks cleaner than most alternatives on the board",
            "adding it would materially increase overlap with current leadership exposure",
        ),
        (
            "signal quality is a little less forceful but still fully actionable",
            "its driver set remains differentiated from existing positions",
            "adding it would diversify the book without dropping below the actionable bar",
        ),
    ),
    MarketContext(
        "ctx_solid_v2",
        "solid",
        (
            "trend quality is strongest in the same pocket of the market the book already emphasizes",
            "path quality is especially orderly and conviction is high",
            "selecting it would increase concentration in the current sleeve rather than broaden the book",
        ),
        (
            "trend quality is still solid enough to deserve capital",
            "its path is supported by a different sleeve than the current book",
            "selecting it would add differentiated exposure while remaining a real trade",
        ),
    ),
    MarketContext(
        "ctx_exceptional_v0",
        "exceptional",
        (
            "the highest-conviction continuation sits in the same sleeve the book already owns heavily",
            "confirmation is unusually broad and the raw setup is exceptional",
            "selecting it would materially increase concentration in the existing cluster",
        ),
        (
            "the differentiated sleeve is also strong enough to buy with confidence",
            "confirmation is broad even if the raw setup is slightly less dominant",
            "selecting it would broaden the book instead of pressing the current cluster harder",
        ),
    ),
    MarketContext(
        "ctx_exceptional_v1",
        "exceptional",
        (
            "the cleanest opportunity is again in the same sleeve already represented in the book",
            "follow-through quality is exceptional and easier to press",
            "adding it would stack more capital into the existing overlap pocket",
        ),
        (
            "the distinct sleeve also offers an unusually strong entry",
            "its structure is slightly less dominant but still clearly top-tier",
            "adding it would diversify the portfolio while keeping trade quality high",
        ),
    ),
    MarketContext(
        "ctx_exceptional_v2",
        "exceptional",
        (
            "the most forceful continuation remains in the same leadership sleeve the book already holds",
            "signal quality is unusually strong and resilient",
            "choosing it would deepen concentration in the current book rather than broaden it",
        ),
        (
            "the separate sleeve remains strong enough to rank as a real buy on its own merits",
            "its setup is robust even if raw momentum is slightly less dominant",
            "choosing it would add differentiated exposure without dropping below the exceptional bar",
        ),
    ),
)


def _diversification_implied_direction(value: int) -> str:
    return "diversified" if value == 1 else "concentrated"


def _asset_from_diversification(direction: str) -> str:
    return "BETA" if direction == "diversified" else "ALPHA"


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
        3: "use a normal risk posture rather than explicitly favoring the most stable or most aggressive profile",
    }[value]


def _holding_setting_text(value: int) -> str:
    return {
        3: "use a normal multi-tick hold horizon",
    }[value]


def _diversification_setting_text(value: int) -> str:
    return {
        1: "prioritize broadening the book into a distinct sleeve rather than adding more overlap to the current one",
        5: "prioritize concentrating into the strongest overlapping sleeve rather than broadening exposure",
    }[value]


def _stable_index(parts: tuple[str, ...], n: int) -> int:
    total = 0
    for part in parts:
        for idx, ch in enumerate(part):
            total += (idx + 1) * ord(ch)
    return total % n


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
        f"ALPHA: {context.concentrated_lines[0]}, {context.concentrated_lines[1]}, and {context.concentrated_lines[2]}.",
        f"BETA: {context.diversified_lines[0]}, {context.diversified_lines[1]}, and {context.diversified_lines[2]}.",
        "DELTA: evidence is mixed and the reward-to-risk profile remains unimpressive.",
        "GAMMA: signal quality is noisy and conviction is too thin to rank highly.",
    )
    return "\n".join(lines), {
        "evidence_tier": context.evidence_tier,
        "concentrated_lines": list(context.concentrated_lines),
        "diversified_lines": list(context.diversified_lines),
        "concentrated_asset": "ALPHA",
        "diversified_asset": "BETA",
    }


def _render_user_text(
    *,
    strategy_template: StrategyTemplate,
    settings_template: SettingsTemplate,
    activity_phrase_id: str,
    activity_text: str,
    diversification_phrase_id: str,
    diversification_text: str,
    diversification_value: int,
    context: MarketContext,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    market_text, market_snapshot = _render_market_text(context)
    portfolio_text = (
        "Free cash reserve: high.\n"
        "Current positions:\n"
        "- OMEGA: large position in the same leadership sleeve that overlaps heavily with ALPHA.\n"
        "- SIGMA: medium position with moderate overlap to the ALPHA sleeve.\n"
        "Adding ALPHA would deepen the current cluster.\n"
        "Adding BETA would broaden the book into a more distinct sleeve."
    )
    user_text = (
        "TASK\n"
        "Choose exactly one action for this tick.\n\n"
        "STRATEGY\n"
        f"- {strategy_template.activity_shell.format(activity_text=activity_text)}\n"
        f"- {strategy_template.diversification_shell.format(diversification_text=diversification_text)}\n"
        "Apply the strategy only within what ACTIVE SETTINGS allow. "
        "If STRATEGY and ACTIVE SETTINGS disagree about diversification posture, follow ACTIVE SETTINGS.\n\n"
        "ACTIVE SETTINGS\n"
        f"- {settings_template.activity_gloss}: 5/5 — {_activity_setting_text(5)}.\n"
        f"- {settings_template.size_gloss}: 5/5 — {_size_setting_text(5)}.\n"
        f"- {settings_template.risk_gloss}: 3/5 — {_risk_setting_text(3)}.\n"
        f"- {settings_template.holding_gloss}: 3/5 — {_holding_setting_text(3)}.\n"
        f"- {settings_template.diversification_gloss}: {diversification_value}/5 — {_diversification_setting_text(diversification_value)}.\n\n"
        "PORTFOLIO\n"
        f"{portfolio_text}\n\n"
        "MARKET\n"
        f"{market_text}"
    )
    strategy_snapshot = {
        "activity_phrase_id": activity_phrase_id,
        "activity_text": activity_text,
        "diversification_phrase_id": diversification_phrase_id,
        "diversification_text": diversification_text,
        "strategy_variant_id": strategy_template.variant_id,
    }
    settings_snapshot = {
        "activity_value": 5,
        "size_value": 5,
        "risk_value": 3,
        "holding_value": 3,
        "diversification_value": diversification_value,
        "settings_variant_id": settings_template.variant_id,
    }
    portfolio_snapshot = {
        "free_cash_reserve": "high",
        "current_positions": [
            {
                "asset": "OMEGA",
                "size": "large",
                "overlap_to_alpha": "high",
                "overlap_to_beta": "low",
            },
            {
                "asset": "SIGMA",
                "size": "medium",
                "overlap_to_alpha": "moderate",
                "overlap_to_beta": "low",
            },
        ],
        "alpha_effect": "deepen_existing_cluster",
        "beta_effect": "broaden_into_distinct_sleeve",
    }
    return user_text, strategy_snapshot, settings_snapshot, portfolio_snapshot, market_snapshot


def _expected_for_row(
    *,
    strategy_direction: str,
    setting_value: int,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], str, bool, bool]:
    setting_dir = _diversification_implied_direction(setting_value)
    conflict_present = setting_dir != strategy_direction
    expected = {"action": "buy", "asset": _asset_from_diversification(setting_dir), "size": "large"}
    strategy_expected = {"action": "buy", "asset": _asset_from_diversification(strategy_direction), "size": "large"}
    setting_expected = dict(expected)
    conflict_band = "strong_conflict" if conflict_present else "aligned"
    return strategy_expected, setting_expected, expected, conflict_band, conflict_present, False


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    directions = ("diversified", "concentrated")

    for direction_index, strategy_direction in enumerate(directions):
        activity_pool = ACTIVITY_STRATEGY_TEXT["trade"]
        diversification_pool = DIVERSIFICATION_STRATEGY_TEXT[strategy_direction]

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
                    diversification_phrase_id, diversification_text = _strategy_phrase_variant(
                        diversification_pool,
                        template_index=template_index,
                        settings_index=settings_index,
                        context_index=context_index,
                        direction_index=direction_index + 1,
                    )
                    for setting_value in TARGET_SETTING_VALUES:
                        group_key = (
                            f"diversification_preference:{strategy_direction}:{strategy_template.variant_id}:"
                            f"{settings_template.variant_id}:{context.context_variant_id}"
                        )
                        strategy_expected, setting_expected, expected_output, conflict_band, conflict_present, edge_conflict = _expected_for_row(
                            strategy_direction=strategy_direction,
                            setting_value=setting_value,
                        )
                        user_text, strategy_snapshot, settings_snapshot, portfolio_snapshot, market_snapshot = _render_user_text(
                            strategy_template=strategy_template,
                            settings_template=settings_template,
                            activity_phrase_id=activity_phrase_id,
                            activity_text=activity_text,
                            diversification_phrase_id=diversification_phrase_id,
                            diversification_text=diversification_text,
                            diversification_value=setting_value,
                            context=context,
                        )
                        matched_group_id = group_key
                        pair_value = 1 if strategy_direction == "concentrated" else 5
                        matched_pair_id = f"{matched_group_id}:pair_{pair_value}"
                        rows.append(
                            {
                                "example_id": (
                                    f"pc12:diversification_preference:{strategy_direction}:{context.context_variant_id}:"
                                    f"{strategy_template.variant_id}:{settings_template.variant_id}:setting_{setting_value}"
                                ),
                                "matched_group_id": matched_group_id,
                                "matched_pair_id": matched_pair_id,
                                "target_dimension": "diversification_preference",
                                "strategy_direction": strategy_direction,
                                "setting_value": setting_value,
                                "setting_implied_direction": _diversification_implied_direction(setting_value),
                                "conflict_present": conflict_present,
                                "edge_conflict": edge_conflict,
                                "conflict_band": conflict_band,
                                "main_benchmark_row": True,
                                "stress_test_slice": "",
                                "conflict_strength": 0 if conflict_band == "aligned" else 2,
                                "strategy_variant_id": strategy_template.variant_id,
                                "activity_phrase_id": activity_phrase_id,
                                "diversification_phrase_id": diversification_phrase_id,
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
                                "portfolio_snapshot_json": portfolio_snapshot,
                                "market_snapshot_json": market_snapshot,
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
    parser = argparse.ArgumentParser(description="Build the prompt-confusion Phase 12 diversification dataset scaffold.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "phase_12_dataset.jsonl"
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
        "by_setting_value": dict(Counter(row["setting_value"] for row in rows)),
        "by_strategy_direction": dict(Counter(row["strategy_direction"] for row in rows)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = build_rows()
    write_outputs(rows, args.output_dir)


if __name__ == "__main__":
    main()
