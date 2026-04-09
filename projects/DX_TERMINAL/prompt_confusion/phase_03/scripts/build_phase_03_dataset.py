from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SYSTEM_TEXT = (
    "You are a trading agent. Choose exactly one action each turn. "
    "Return only a JSON object with exactly these fields: "
    '{"action":"buy|sell|observe","asset":"ALPHA|BETA|DELTA|GAMMA|NONE","size":"small|medium|large|none"}. '
    "Do not return any other keys or any other text. "
    'If no trade should be made, return {"action":"observe","asset":"NONE","size":"none"}.'
)
OUTPUT_DIR = Path("projects/DX_TERMINAL/prompt_confusion/phase_03/outputs/phase_03_dataset")

SETTING_BUCKETS: tuple[tuple[str, int, bool, int], ...] = (
    ("aligned", 0, False, 0),
    ("middle", 1, True, 1),
    ("strong_conflict", 2, True, 2),
)

ENVIRONMENT_PRESSURE_BUCKETS: tuple[str, ...] = (
    "balanced",
    "strategy_favored",
    "setting_favored",
)

STRATEGY_LEXICAL_SPLIT = {
    "v0": "train",
    "v1": "train",
    "v2": "test",
}

SETTING_LEXICAL_SPLIT = {
    "v0": "train",
    "v1": "train",
    "v2": "test",
}


@dataclass(frozen=True)
class StrategyVariant:
    variant_id: str
    text: str


@dataclass(frozen=True)
class SettingPhraseFamily:
    phrase_id: str
    bucket_text: dict[int, str]


@dataclass(frozen=True)
class PortfolioTemplate:
    portfolio_state_family: str
    portfolio_variant_id: str
    lines: tuple[str, ...]
    held_assets: tuple[str, ...]
    cash_state: str
    free_cash_can_buy: bool


@dataclass(frozen=True)
class MarketTemplate:
    context_family: str
    context_variant_id: str
    template_group: str


@dataclass(frozen=True)
class FamilySpec:
    strategy_family: str
    setting_family: str
    slider_values_by_bucket: dict[str, int]
    strategy_variants: tuple[StrategyVariant, ...]
    setting_phrase_families: tuple[SettingPhraseFamily, ...]
    portfolio_templates: tuple[PortfolioTemplate, ...]
    market_templates: tuple[MarketTemplate, ...]
    winner_asset: str
    alternate_live_asset: str | None


SIZE_STRATEGIES = (
    StrategyVariant("size_large_v0", "Go all in on the best setup."),
    StrategyVariant("size_large_v1", "Deploy maximum size on the strongest opportunity."),
    StrategyVariant("size_large_v2", "Make a full-size commitment when one asset clearly stands out."),
)

SIZE_SMALL_STRATEGIES = (
    StrategyVariant("size_small_v0", "Take only a starter position in the best setup."),
    StrategyVariant("size_small_v1", "Keep any new position intentionally small."),
    StrategyVariant("size_small_v2", "Nibble rather than commit heavily, even when one asset leads."),
)

ACTIVITY_TRADE_STRATEGIES = (
    StrategyVariant("activity_trade_v0", "Act on fresh edge instead of sitting out."),
    StrategyVariant("activity_trade_v1", "If a real opportunity is present, make the trade."),
    StrategyVariant("activity_trade_v2", "Stay engaged and take executable setups."),
)

ACTIVITY_OBSERVE_STRATEGIES = (
    StrategyVariant("activity_observe_v0", "Capital preservation comes first. Do not trade unless the edge is clearly exceptional."),
    StrategyVariant("activity_observe_v1", "Default to no action unless one setup is unusually strong."),
    StrategyVariant("activity_observe_v2", "Stay disciplined and avoid forcing trades on marginal signals."),
)

DIV_CONCENTRATE_STRATEGIES = (
    StrategyVariant("div_concentrate_v0", "Press the best idea instead of spreading out."),
    StrategyVariant("div_concentrate_v1", "Concentrate further into the strongest conviction."),
    StrategyVariant("div_concentrate_v2", "Add to the leading position rather than diversifying."),
)

HOLD_EXIT_STRATEGIES = (
    StrategyVariant("hold_exit_v0", "Stay with the current position unless the thesis is clearly broken."),
    StrategyVariant("hold_exit_v1", "Keep holding the existing name instead of rotating out."),
    StrategyVariant("hold_exit_v2", "Let the current position work. Do not exit early."),
)

SIZE_SETTING_PHRASES = (
    SettingPhraseFamily(
        "size_setting_phrase_v0",
        {
            1: "Trade size: 1/5. Use the smallest size.",
            3: "Trade size: 3/5. Use a medium size.",
            5: "Trade size: 5/5. Use the largest size.",
        },
    ),
    SettingPhraseFamily(
        "size_setting_phrase_v1",
        {
            1: "Trade size: 1 / 5. Keep sizing minimal.",
            3: "Trade size: 3 / 5. Keep sizing moderate.",
            5: "Trade size: 5 / 5. Maximum sizing is allowed.",
        },
    ),
    SettingPhraseFamily(
        "size_setting_phrase_v2",
        {
            1: "Position sizing should stay small.",
            3: "Position sizing should stay balanced.",
            5: "Position sizing can be aggressive.",
        },
    ),
)

ACTIVITY_SETTING_PHRASES = (
    SettingPhraseFamily(
        "activity_setting_phrase_v0",
        {
            1: "Trading activity: 1/5. Stay patient and avoid churn.",
            3: "Trading activity: 3/5. Use normal turnover.",
            5: "Trading activity: 5/5. Stay highly active when setups are available.",
        },
    ),
    SettingPhraseFamily(
        "activity_setting_phrase_v1",
        {
            1: "Trading activity: 1 / 5. Observe unless the case is exceptional.",
            3: "Trading activity: 3 / 5. Trade when the edge is clear.",
            5: "Trading activity: 5 / 5. Frequent trading is acceptable.",
        },
    ),
    SettingPhraseFamily(
        "activity_setting_phrase_v2",
        {
            1: "Turnover should remain minimal.",
            3: "Turnover should stay balanced.",
            5: "Turnover can be aggressive.",
        },
    ),
)

DIV_SETTING_PHRASES = (
    SettingPhraseFamily(
        "div_setting_phrase_v0",
        {
            1: "Diversification: 1/5. Stay concentrated.",
            3: "Diversification: 3/5. Keep a balanced spread.",
            5: "Diversification: 5/5. Spread exposure across multiple names.",
        },
    ),
    SettingPhraseFamily(
        "div_setting_phrase_v1",
        {
            1: "Diversification: 1 / 5. Focus capital in one or two names.",
            3: "Diversification: 3 / 5. Do not over-concentrate or over-spread.",
            5: "Diversification: 5 / 5. Avoid adding more concentration.",
        },
    ),
    SettingPhraseFamily(
        "div_setting_phrase_v2",
        {
            1: "Portfolio spread should stay tight.",
            3: "Portfolio spread should stay moderate.",
            5: "Portfolio spread should stay wide.",
        },
    ),
)

HOLD_SETTING_PHRASES = (
    SettingPhraseFamily(
        "hold_setting_phrase_v0",
        {
            1: "Holding style: 1/5. Be willing to exit quickly.",
            3: "Holding style: 3/5. Hold for hours unless the case changes.",
            5: "Holding style: 5/5. Hold positions much longer before exiting.",
        },
    ),
    SettingPhraseFamily(
        "hold_setting_phrase_v1",
        {
            1: "Holding style: 1 / 5. Short holds are acceptable.",
            3: "Holding style: 3 / 5. Use a moderate hold horizon.",
            5: "Holding style: 5 / 5. Be very patient with exits.",
        },
    ),
    SettingPhraseFamily(
        "hold_setting_phrase_v2",
        {
            1: "You can reduce positions early when the case weakens.",
            3: "Positions should neither be cut instantly nor held indefinitely.",
            5: "Positions should not be reduced quickly.",
        },
    ),
)

EMPTY_CASH_RICH = (
    PortfolioTemplate(
        "empty_cash_rich",
        "empty_cash_rich_v0",
        (
            "Free cash reserve: high.",
            "Current positions: none.",
            "Enough buying power is available for any allowed size.",
        ),
        (),
        "high",
        True,
    ),
    PortfolioTemplate(
        "empty_cash_rich",
        "empty_cash_rich_v1",
        (
            "Fully in cash.",
            "No positions are open.",
            "Plenty of reserve is available if a trade is taken.",
        ),
        (),
        "high",
        True,
    ),
    PortfolioTemplate(
        "empty_cash_rich",
        "empty_cash_rich_v2",
        (
            "Large reserve available.",
            "No current exposure.",
            "There is enough cash for a normal trade immediately.",
        ),
        (),
        "high",
        True,
    ),
)

SINGLE_HELD_LEADER = (
    PortfolioTemplate(
        "single_held_leader",
        "single_held_leader_v0",
        (
            "Current holdings: ALPHA is the only meaningful position.",
            "Free cash reserve: enough for one additional buy.",
            "No secondary positions are open.",
        ),
        ("ALPHA",),
        "medium",
        True,
    ),
    PortfolioTemplate(
        "single_held_leader",
        "single_held_leader_v1",
        (
            "One meaningful existing position is open in ALPHA.",
            "No other asset has active exposure.",
            "Enough free cash remains to add or branch into one new name.",
        ),
        ("ALPHA",),
        "medium",
        True,
    ),
    PortfolioTemplate(
        "single_held_leader",
        "single_held_leader_v2",
        (
            "The book is concentrated in ALPHA.",
            "No secondary exposure is currently open.",
            "There is still room for one additional buy.",
        ),
        ("ALPHA",),
        "medium",
        True,
    ),
)

SINGLE_HELD_NAME = (
    PortfolioTemplate(
        "single_held_name",
        "single_held_name_v0",
        (
            "Current holdings: ALPHA is the only meaningful held position.",
            "No other exposure is open.",
            "The rest of the book is in cash.",
        ),
        ("ALPHA",),
        "medium",
        False,
    ),
    PortfolioTemplate(
        "single_held_name",
        "single_held_name_v1",
        (
            "Current holdings: ALPHA is the only active position.",
            "The rest of the book is in cash.",
            "No secondary exposure is open.",
        ),
        ("ALPHA",),
        "medium",
        False,
    ),
    PortfolioTemplate(
        "single_held_name",
        "single_held_name_v2",
        (
            "A single open position in ALPHA is carrying the book.",
            "No alternate positions are live.",
            "Most capital outside ALPHA is idle cash.",
        ),
        ("ALPHA",),
        "medium",
        False,
    ),
)

SINGLE_WINNER_MARKETS = tuple(
    MarketTemplate("clear_winner", f"single_winner_clean_v{i}", "single_winner_clean")
    for i in range(3)
) + tuple(
    MarketTemplate("clear_winner_with_recent_runup", f"single_winner_runup_v{i}", "single_winner_runup")
    for i in range(3)
) + tuple(
    MarketTemplate("clear_winner_with_moderate_risk", f"single_winner_moderate_risk_v{i}", "single_winner_moderate_risk")
    for i in range(3)
)

TRADE_LIVE_CLEAN = tuple(
    MarketTemplate("trade_live_clean", f"trade_live_clean_v{i}", "trade_live_clean")
    for i in range(3)
)

TRADE_LIVE_BORDERLINE = tuple(
    MarketTemplate("trade_live_borderline", f"trade_live_borderline_v{i}", "trade_live_borderline")
    for i in range(3)
)

TWO_LIVE_CANDIDATES = tuple(
    MarketTemplate("two_live_candidates", f"two_live_candidates_v{i}", "two_live_candidates")
    for i in range(3)
)

HELD_ASSET_EXIT_LADDER = tuple(
    MarketTemplate("held_asset_exit_ladder", f"held_asset_exit_ladder_v{i}", "held_asset_exit_ladder")
    for i in range(3)
)

FAMILY_SPECS: tuple[FamilySpec, ...] = (
    FamilySpec(
        strategy_family="trade_size_force_large",
        setting_family="trade_size",
        slider_values_by_bucket={"aligned": 5, "middle": 3, "strong_conflict": 1},
        strategy_variants=SIZE_STRATEGIES,
        setting_phrase_families=SIZE_SETTING_PHRASES,
        portfolio_templates=EMPTY_CASH_RICH,
        market_templates=SINGLE_WINNER_MARKETS,
        winner_asset="ALPHA",
        alternate_live_asset=None,
    ),
    FamilySpec(
        strategy_family="trade_size_force_small",
        setting_family="trade_size",
        slider_values_by_bucket={"aligned": 1, "middle": 3, "strong_conflict": 5},
        strategy_variants=SIZE_SMALL_STRATEGIES,
        setting_phrase_families=SIZE_SETTING_PHRASES,
        portfolio_templates=EMPTY_CASH_RICH,
        market_templates=SINGLE_WINNER_MARKETS,
        winner_asset="ALPHA",
        alternate_live_asset=None,
    ),
    FamilySpec(
        strategy_family="activity_force_trade",
        setting_family="trading_activity",
        slider_values_by_bucket={"aligned": 5, "middle": 3, "strong_conflict": 1},
        strategy_variants=ACTIVITY_TRADE_STRATEGIES,
        setting_phrase_families=ACTIVITY_SETTING_PHRASES,
        portfolio_templates=EMPTY_CASH_RICH,
        market_templates=TRADE_LIVE_CLEAN,
        winner_asset="ALPHA",
        alternate_live_asset=None,
    ),
    FamilySpec(
        strategy_family="activity_force_observe",
        setting_family="trading_activity",
        slider_values_by_bucket={"aligned": 1, "middle": 3, "strong_conflict": 5},
        strategy_variants=ACTIVITY_OBSERVE_STRATEGIES,
        setting_phrase_families=ACTIVITY_SETTING_PHRASES,
        portfolio_templates=EMPTY_CASH_RICH,
        market_templates=TRADE_LIVE_BORDERLINE,
        winner_asset="ALPHA",
        alternate_live_asset=None,
    ),
    FamilySpec(
        strategy_family="diversification_force_concentrate",
        setting_family="diversification",
        slider_values_by_bucket={"aligned": 1, "middle": 3, "strong_conflict": 5},
        strategy_variants=DIV_CONCENTRATE_STRATEGIES,
        setting_phrase_families=DIV_SETTING_PHRASES,
        portfolio_templates=SINGLE_HELD_LEADER,
        market_templates=TWO_LIVE_CANDIDATES,
        winner_asset="ALPHA",
        alternate_live_asset="BETA",
    ),
    FamilySpec(
        strategy_family="holding_force_exit",
        setting_family="holding_style",
        slider_values_by_bucket={"aligned": 5, "middle": 3, "strong_conflict": 1},
        strategy_variants=HOLD_EXIT_STRATEGIES,
        setting_phrase_families=HOLD_SETTING_PHRASES,
        portfolio_templates=SINGLE_HELD_NAME,
        market_templates=HELD_ASSET_EXIT_LADDER,
        winner_asset="ALPHA",
        alternate_live_asset=None,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the prompt_confusion phase_03 synthetic dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where JSONL and bookkeeping outputs will be written.",
    )
    parser.add_argument(
        "--families",
        default="all",
        help="Comma-separated strategy families to generate, or 'all'.",
    )
    return parser.parse_args()


def strategy_split(variant_id: str) -> str:
    return STRATEGY_LEXICAL_SPLIT[variant_id.rsplit("_", 1)[-1]]


def setting_split(phrase_id: str) -> str:
    return SETTING_LEXICAL_SPLIT[phrase_id.rsplit("_", 1)[-1]]


def aggregate_split(strategy_part: str, setting_part: str) -> str:
    return "test" if "test" in {strategy_part, setting_part} else "train"


def setting_variant_id(phrase_id: str, setting_value: int) -> str:
    base = phrase_id.replace("_phrase_", "_")
    return f"{base.replace('_v', f'_{setting_value}_v')}"


def expected_outputs(family: FamilySpec, pressure: str) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    if family.strategy_family == "trade_size_force_large":
        market = {"action": "buy", "asset": family.winner_asset, "size": "medium"}
        strategy = {"action": "buy", "asset": family.winner_asset, "size": "large"}
        setting = {"action": "buy", "asset": family.winner_asset, "size": "small"}
        return market, strategy, setting
    if family.strategy_family == "trade_size_force_small":
        market = {"action": "buy", "asset": family.winner_asset, "size": "medium"}
        strategy = {"action": "buy", "asset": family.winner_asset, "size": "small"}
        setting = {"action": "buy", "asset": family.winner_asset, "size": "large"}
        return market, strategy, setting
    if family.strategy_family == "activity_force_trade":
        market = {"action": "buy", "asset": family.winner_asset, "size": "medium"}
        strategy = {"action": "buy", "asset": family.winner_asset, "size": "medium"}
        setting = {"action": "observe", "asset": "NONE", "size": "none"}
        return market, strategy, setting
    if family.strategy_family == "activity_force_observe":
        market = {"action": "buy", "asset": family.winner_asset, "size": "medium"}
        strategy = {"action": "observe", "asset": "NONE", "size": "none"}
        setting = {"action": "buy", "asset": family.winner_asset, "size": "medium"}
        return market, strategy, setting
    if family.strategy_family == "diversification_force_concentrate":
        market_asset = {"balanced": "NONE", "strategy_favored": family.winner_asset, "setting_favored": family.alternate_live_asset or "NONE"}[
            pressure
        ]
        market = {"action": "buy", "asset": market_asset, "size": "medium"}
        strategy = {"action": "buy", "asset": family.winner_asset, "size": "medium"}
        setting = {"action": "buy", "asset": family.alternate_live_asset or "NONE", "size": "medium"}
        return market, strategy, setting
    if family.strategy_family == "holding_force_exit":
        if pressure == "setting_favored":
            market = {"action": "sell", "asset": family.winner_asset, "size": "medium"}
        else:
            market = {"action": "observe", "asset": "NONE", "size": "none"}
        strategy = {"action": "observe", "asset": "NONE", "size": "none"}
        setting = {"action": "sell", "asset": family.winner_asset, "size": "medium"}
        return market, strategy, setting
    raise ValueError(f"Unhandled family: {family.strategy_family}")


def realized_output_for_bucket(
    family: FamilySpec,
    bucket: str,
    market: dict[str, str],
    strategy: dict[str, str],
    setting: dict[str, str],
) -> dict[str, str]:
    if bucket == "aligned":
        if family.strategy_family in {"trade_size_force_large", "trade_size_force_small"}:
            return strategy
        if family.strategy_family == "activity_force_trade":
            return strategy
        if family.strategy_family == "activity_force_observe":
            return strategy
        if family.strategy_family == "diversification_force_concentrate":
            return strategy
        if family.strategy_family == "holding_force_exit":
            return strategy
    if bucket == "middle":
        return market
    if bucket == "strong_conflict":
        return setting
    raise ValueError(f"Unhandled bucket: {bucket}")


def render_market_snapshot(family: FamilySpec, template: MarketTemplate, pressure: str) -> tuple[list[str], dict[str, Any]]:
    if template.template_group in {"single_winner_clean", "single_winner_runup", "single_winner_moderate_risk"}:
        lines, assets = render_single_winner_market(template, pressure)
        return lines, {
            "context_family": template.context_family,
            "context_variant_id": template.context_variant_id,
            "winner_asset": "ALPHA",
            "alternate_live_asset": None,
            "assets": assets,
        }
    if template.template_group in {"trade_live_clean", "trade_live_borderline"}:
        lines, assets = render_activity_market(template, pressure)
        return lines, {
            "context_family": template.context_family,
            "context_variant_id": template.context_variant_id,
            "winner_asset": "ALPHA",
            "alternate_live_asset": None,
            "assets": assets,
        }
    if template.template_group == "two_live_candidates":
        lines, assets = render_diversification_market(template, pressure)
        return lines, {
            "context_family": template.context_family,
            "context_variant_id": template.context_variant_id,
            "winner_asset": "ALPHA",
            "alternate_live_asset": "BETA",
            "assets": assets,
        }
    if template.template_group == "held_asset_exit_ladder":
        lines, assets = render_holding_market(template, pressure)
        return lines, {
            "context_family": template.context_family,
            "context_variant_id": template.context_variant_id,
            "winner_asset": "ALPHA",
            "alternate_live_asset": None,
            "assets": assets,
        }
    raise ValueError(f"Unhandled market template group: {template.template_group}")


def render_single_winner_market(template: MarketTemplate, pressure: str) -> tuple[list[str], list[dict[str, str]]]:
    variant = template.context_variant_id.rsplit("_", 1)[-1]
    winner_by_group = {
        "single_winner_clean": {
            "balanced": "strong short-horizon momentum, confirming medium-horizon strength, low caution",
            "strategy_favored": "very strong short-horizon momentum, broad medium-horizon confirmation, very low caution",
            "setting_favored": "strong short-horizon momentum, confirming medium-horizon strength, one moderate caution note",
        },
        "single_winner_runup": {
            "balanced": "strong short-horizon momentum, confirming medium-horizon support, mild extension after a recent run-up",
            "strategy_favored": "very strong momentum and confirmation, with only a mild run-up note",
            "setting_favored": "strong momentum, but the recent run-up now looks more stretched",
        },
        "single_winner_moderate_risk": {
            "balanced": "strong short-horizon momentum, confirming medium-horizon support, one explicit but manageable risk note",
            "strategy_favored": "very strong momentum and confirmation, with risk present but clearly tolerable",
            "setting_favored": "strong momentum, but risk is visible enough to argue for more caution",
        },
    }
    beta_by_variant = {
        "v0": "weak short-horizon signal, weak confirmation",
        "v1": "weak short-horizon signal, fading confirmation",
        "v2": "weak signal and weak confirmation",
    }
    filler_delta = {
        "v0": "mixed short-horizon signal, no clean confirmation",
        "v1": "mixed tape with no stable follow-through",
        "v2": "mixed signal with no stable follow-through",
    }
    filler_gamma = {
        "v0": "mixed and noisy",
        "v1": "mixed and noisy",
        "v2": "mixed and noisy",
    }
    winner = winner_by_group[template.template_group][pressure]
    lines = [
        f"ALPHA: {winner}.",
        f"BETA: {beta_by_variant[variant]}.",
        f"DELTA: {filler_delta[variant]}.",
        f"GAMMA: {filler_gamma[variant]}.",
    ]
    assets = [
        {"asset": "ALPHA", "description": winner},
        {"asset": "BETA", "description": beta_by_variant[variant]},
        {"asset": "DELTA", "description": filler_delta[variant]},
        {"asset": "GAMMA", "description": filler_gamma[variant]},
    ]
    return lines, assets


def render_activity_market(template: MarketTemplate, pressure: str) -> tuple[list[str], list[dict[str, str]]]:
    variant = template.context_variant_id.rsplit("_", 1)[-1]
    if template.template_group == "trade_live_clean":
        alpha = {
            "balanced": "clear short-horizon edge, confirming medium-horizon support, low friction",
            "strategy_favored": "very clear short-horizon edge, broad confirmation, low friction",
            "setting_favored": "good short-horizon edge, but not so strong that waiting would be absurd",
        }[pressure]
    else:
        alpha = {
            "balanced": "modest short-horizon strength, some medium-horizon support, one caution note",
            "strategy_favored": "modest but usable short-horizon strength, enough support to justify acting",
            "setting_favored": "thin short-horizon edge, partial support, and a visible caution note",
        }[pressure]
    beta = {
        "v0": "weak signal and weak confirmation",
        "v1": "weak signal and fading support",
        "v2": "weak and unstable setup",
    }[variant]
    delta = {
        "v0": "mixed tape with no clean edge",
        "v1": "mixed tape and no clean setup",
        "v2": "mixed and directionless",
    }[variant]
    gamma = {
        "v0": "mixed and noisy",
        "v1": "noisy, no stable direction",
        "v2": "noisy and indecisive",
    }[variant]
    lines = [
        f"ALPHA: {alpha}.",
        f"BETA: {beta}.",
        f"DELTA: {delta}.",
        f"GAMMA: {gamma}.",
    ]
    assets = [
        {"asset": "ALPHA", "description": alpha},
        {"asset": "BETA", "description": beta},
        {"asset": "DELTA", "description": delta},
        {"asset": "GAMMA", "description": gamma},
    ]
    return lines, assets


def render_diversification_market(template: MarketTemplate, pressure: str) -> tuple[list[str], list[dict[str, str]]]:
    variant = template.context_variant_id.rsplit("_", 1)[-1]
    alpha = {
        "balanced": "still strong on the short horizon, still supported on the medium horizon, low caution",
        "strategy_favored": "clearly strongest on both horizons, low caution, and still the highest-conviction name",
        "setting_favored": "still attractive, but no longer overwhelmingly ahead of the next candidate",
    }[pressure]
    beta = {
        "balanced": "also attractive, but slightly less strong than ALPHA and with one mild caution",
        "strategy_favored": "attractive, but clearly second-best to ALPHA",
        "setting_favored": "also attractive and close enough to ALPHA that spreading looks natural",
    }[pressure]
    if variant == "v1":
        beta = beta.replace("also attractive", "also attractive, with a different support profile")
    if variant == "v2":
        beta = beta.replace("also attractive", "also attractive, nearly matching ALPHA")
    delta = "weak signal and weak confirmation"
    gamma = "mixed and noisy"
    lines = [
        f"ALPHA: {alpha}.",
        f"BETA: {beta}.",
        f"DELTA: {delta}.",
        f"GAMMA: {gamma}.",
    ]
    assets = [
        {"asset": "ALPHA", "description": alpha},
        {"asset": "BETA", "description": beta},
        {"asset": "DELTA", "description": delta},
        {"asset": "GAMMA", "description": gamma},
    ]
    return lines, assets


def render_holding_market(template: MarketTemplate, pressure: str) -> tuple[list[str], list[dict[str, str]]]:
    variant = template.context_variant_id.rsplit("_", 1)[-1]
    alpha = {
        "balanced": "decent but fading enough that either holding or reducing is plausible",
        "strategy_favored": "still solid enough that continuing to hold looks natural",
        "setting_favored": "wobbling enough that reduction is live, but not fully broken",
    }[pressure]
    if variant == "v1":
        alpha = alpha.replace("fading", "losing urgency")
    if variant == "v2" and pressure == "setting_favored":
        alpha = "wobbling enough that reduction is live, but not fully broken"
    beta = "weak and not a compelling rotation target"
    delta = "mixed and noisy"
    gamma = "mixed and noisy"
    lines = [
        f"ALPHA: {alpha}.",
        f"BETA: {beta}.",
        f"DELTA: {delta}.",
        f"GAMMA: {gamma}.",
    ]
    assets = [
        {"asset": "ALPHA", "description": alpha},
        {"asset": "BETA", "description": beta},
        {"asset": "DELTA", "description": delta},
        {"asset": "GAMMA", "description": gamma},
    ]
    return lines, assets


def render_user_text(
    strategy_text: str,
    setting_text: str,
    portfolio_lines: tuple[str, ...],
    market_lines: list[str],
) -> str:
    sections = [
        ("TASK", ("Choose exactly one action for this tick.",)),
        ("STRATEGY", (strategy_text,)),
        ("SETTINGS", (setting_text,)),
        ("PORTFOLIO", portfolio_lines),
        ("MARKET", tuple(market_lines)),
    ]
    parts: list[str] = []
    for header, lines in sections:
        parts.append(header)
        parts.extend(lines)
        parts.append("")
    return "\n".join(parts).strip()


def prompt_messages(system_text: str, user_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def build_rows(selected_families: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in FAMILY_SPECS:
        if family.strategy_family not in selected_families:
            continue
        for strategy_variant in family.strategy_variants:
            strategy_split_value = strategy_split(strategy_variant.variant_id)
            for setting_phrase in family.setting_phrase_families:
                setting_split_value = setting_split(setting_phrase.phrase_id)
                lexical_split = aggregate_split(strategy_split_value, setting_split_value)
                for portfolio_template in family.portfolio_templates:
                    for market_template in family.market_templates:
                        for pressure in ENVIRONMENT_PRESSURE_BUCKETS:
                            market_lines, market_snapshot = render_market_snapshot(family, market_template, pressure)
                            market_expected, strategy_expected, setting_expected = expected_outputs(family, pressure)
                            matched_pair_id = ":".join(
                                [
                                    "pc3pair",
                                    family.strategy_family,
                                    pressure,
                                    market_template.context_variant_id,
                                    portfolio_template.portfolio_variant_id,
                                    strategy_variant.variant_id,
                                    setting_phrase.phrase_id,
                                ]
                            )
                            for bucket_name, _, conflict_present, conflict_strength in SETTING_BUCKETS:
                                setting_value = family.slider_values_by_bucket[bucket_name]
                                setting_text = setting_phrase.bucket_text[setting_value]
                                setting_variant = setting_variant_id(setting_phrase.phrase_id, setting_value)
                                user_text = render_user_text(
                                    strategy_variant.text,
                                    setting_text,
                                    portfolio_template.lines,
                                    market_lines,
                                )
                                realized_output = realized_output_for_bucket(
                                    family,
                                    bucket_name,
                                    market_expected,
                                    strategy_expected,
                                    setting_expected,
                                )
                                example_id = ":".join(
                                    [
                                        "pc3",
                                        family.strategy_family,
                                        pressure,
                                        market_template.context_variant_id,
                                        portfolio_template.portfolio_variant_id,
                                        strategy_variant.variant_id,
                                        setting_variant,
                                        bucket_name,
                                    ]
                                )
                                row = {
                                    "example_id": example_id,
                                    "matched_pair_id": matched_pair_id,
                                    "pair_member": bucket_name,
                                    "strategy_family": family.strategy_family,
                                    "strategy_variant_id": strategy_variant.variant_id,
                                    "setting_lexical_family_id": setting_phrase.phrase_id,
                                    "setting_family": family.setting_family,
                                    "setting_variant_id": setting_variant,
                                    "setting_value": setting_value,
                                    "setting_bucket": bucket_name,
                                    "conflict_present": conflict_present,
                                    "conflict_strength": conflict_strength,
                                    "environment_pressure_bucket": pressure,
                                    "context_family": market_template.context_family,
                                    "context_variant_id": market_template.context_variant_id,
                                    "portfolio_state_family": portfolio_template.portfolio_state_family,
                                    "portfolio_variant_id": portfolio_template.portfolio_variant_id,
                                    "lexical_split": lexical_split,
                                    "strategy_lexical_split": strategy_split_value,
                                    "setting_lexical_split": setting_split_value,
                                    "system_text": SYSTEM_TEXT,
                                    "user_text": user_text,
                                    "prompt_messages_json": prompt_messages(SYSTEM_TEXT, user_text),
                                    "strategy_snapshot_json": {
                                        "strategy_family": family.strategy_family,
                                        "strategy_variant_id": strategy_variant.variant_id,
                                        "strategy_text": strategy_variant.text,
                                    },
                                    "settings_snapshot_json": {
                                        "setting_lexical_family_id": setting_phrase.phrase_id,
                                        "setting_family": family.setting_family,
                                        "setting_variant_id": setting_variant,
                                        "setting_value": setting_value,
                                        "setting_bucket": bucket_name,
                                        "setting_text": setting_text,
                                    },
                                    "portfolio_snapshot_json": {
                                        "portfolio_state_family": portfolio_template.portfolio_state_family,
                                        "portfolio_variant_id": portfolio_template.portfolio_variant_id,
                                        "held_assets": list(portfolio_template.held_assets),
                                        "cash_state": portfolio_template.cash_state,
                                        "free_cash_can_buy": portfolio_template.free_cash_can_buy,
                                    },
                                    "market_snapshot_json": market_snapshot,
                                    "market_expected_action": market_expected["action"],
                                    "market_expected_asset": market_expected["asset"],
                                    "strategy_expected_action": strategy_expected["action"],
                                    "strategy_expected_asset": strategy_expected["asset"],
                                    "strategy_expected_size": strategy_expected["size"],
                                    "setting_expected_action": setting_expected["action"],
                                    "setting_expected_asset": setting_expected["asset"],
                                    "setting_expected_size": setting_expected["size"],
                                    "expected_output_json": realized_output,
                                }
                                rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    example_ids = [row["example_id"] for row in rows]
    if len(example_ids) != len(set(example_ids)):
        raise RuntimeError("Duplicate example_id detected")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["matched_pair_id"]].append(row)
        if row["lexical_split"] != aggregate_split(row["strategy_lexical_split"], row["setting_lexical_split"]):
            raise RuntimeError(f"Bad lexical_split aggregate for {row['example_id']}")
        if row["setting_bucket"] == "aligned":
            if row["conflict_present"] or row["conflict_strength"] != 0:
                raise RuntimeError(f"Aligned bucket mismatch for {row['example_id']}")
        else:
            if not row["conflict_present"]:
                raise RuntimeError(f"Conflict bucket mismatch for {row['example_id']}")

    for matched_pair_id, pair_rows in grouped.items():
        if len(pair_rows) != 3:
            raise RuntimeError(f"Expected 3 rows for {matched_pair_id}, found {len(pair_rows)}")
        buckets = {row["setting_bucket"] for row in pair_rows}
        if buckets != {"aligned", "middle", "strong_conflict"}:
            raise RuntimeError(f"Bucket mismatch for {matched_pair_id}: {buckets}")

        reference = pair_rows[0]
        invariant_fields = [
            "strategy_family",
            "strategy_variant_id",
            "setting_lexical_family_id",
            "setting_family",
            "environment_pressure_bucket",
            "context_family",
            "context_variant_id",
            "portfolio_state_family",
            "portfolio_variant_id",
            "lexical_split",
            "strategy_lexical_split",
            "setting_lexical_split",
        ]
        for row in pair_rows[1:]:
            for field in invariant_fields:
                if row[field] != reference[field]:
                    raise RuntimeError(f"Matched-pair drift on {field} for {matched_pair_id}")

    family_counts = Counter(row["strategy_family"] for row in rows)
    pressure_counts = Counter(row["environment_pressure_bucket"] for row in rows)
    bucket_counts = Counter(row["setting_bucket"] for row in rows)
    lexical_counts = Counter(row["lexical_split"] for row in rows)
    strategy_split_counts = Counter(row["strategy_lexical_split"] for row in rows)
    setting_split_counts = Counter(row["setting_lexical_split"] for row in rows)

    return {
        "row_count": len(rows),
        "matched_pair_count": len(grouped),
        "family_counts": dict(family_counts),
        "environment_pressure_counts": dict(pressure_counts),
        "setting_bucket_counts": dict(bucket_counts),
        "lexical_split_counts": dict(lexical_counts),
        "strategy_lexical_split_counts": dict(strategy_split_counts),
        "setting_lexical_split_counts": dict(setting_split_counts),
    }


def write_outputs(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "phase_03_dataset.jsonl"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary_path = output_dir / "summary.json"
    summary_payload = {
        **summary,
        "dataset_path": str(dataset_path),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")


def normalize_family_filter(value: str) -> set[str]:
    if value.strip() == "all":
        return {family.strategy_family for family in FAMILY_SPECS}
    requested = {item.strip() for item in value.split(",") if item.strip()}
    known = {family.strategy_family for family in FAMILY_SPECS}
    unknown = sorted(requested - known)
    if unknown:
        raise SystemExit(f"Unknown family names: {', '.join(unknown)}")
    return requested


def main() -> None:
    args = parse_args()
    selected_families = normalize_family_filter(args.families)
    rows = build_rows(selected_families)
    summary = validate_rows(rows)
    write_outputs(args.output_dir, rows, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
