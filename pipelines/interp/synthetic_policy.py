"""Synthetic policy-algebra dataset for preference-vs-permission research.

This dataset keeps the market rows simple and controlled, then varies explicit
policy conditions:

- permission / affordance mode
- strategy override mode
- risk mode

The goal is to study whether the model preserves a stable market preference
while downstream policy text changes the final action choice.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pipelines.interp.synthetic_market import (
    SyntheticAsset,
    SyntheticMarketConfig,
    SyntheticMarketExample,
    _compute_labels,
    _make_asset,
    _score_asset,
    write_dataset,
)
from pipelines.interp.synthetic_market_db import upload_dataset


SYSTEM_PROMPT = (
    "You are an autonomous trading agent. On each tick, choose exactly one action: "
    "buy, sell, or observe. Determine the strongest market preference from the "
    "market rows, then apply explicit permissions, settings, and strategy rules."
)

SYMBOL_SETS: tuple[tuple[str, str, str, str], ...] = (
    ("A", "B", "C", "D"),
    ("B", "C", "D", "A"),
    ("C", "D", "A", "B"),
    ("D", "A", "B", "C"),
)

PERMISSION_VARIANTS: tuple[str, ...] = (
    "buy_and_sell",
    "buy_only",
    "sell_only",
    "observe_only",
)

STRATEGY_VARIANTS: tuple[str, ...] = (
    "none",
    "no_new_buys",
    "force_sell_held",
    "force_observe",
)

RISK_VARIANTS: tuple[str, ...] = (
    "low_risk",
    "high_risk",
)


@dataclass
class SyntheticPolicyConfig:
    seed: int = 42
    phase_name: str = "policy_algebra_v1"
    output_dir: Path = Path("data/interp_exports/synthetic_policy")
    scenario_seeds: int = 12
    upload: bool = False
    log_id_base: int = 2_145_000_000


def _format_asset_block(asset: SyntheticAsset) -> list[str]:
    return [
        f"- Asset {asset.symbol}",
        f"  - Archetype: {asset.archetype}",
        f"  - 5m change: {asset.pct_5m:+.1f}%",
        f"  - 1h change: {asset.pct_1h:+.1f}%",
        f"  - Net flow 5m: {asset.net_flow_5m:+.2f}",
        f"  - Volume 5m: {asset.vol_5m:.2f}",
        f"  - Volume 1h: {asset.vol_1h:.2f}",
        f"  - Unique traders 5m: {asset.unique_traders_5m}",
        f"  - Top 20 holder pct: {asset.top20_holder_pct:.1f}%",
        f"  - Age bucket: {asset.age_bucket}",
    ]


def _symbol_set(seed_idx: int) -> tuple[str, str, str, str]:
    return SYMBOL_SETS[seed_idx % len(SYMBOL_SETS)]


def _permission_description(permission_mode: str) -> str:
    return {
        "buy_and_sell": "Both buy and sell actions are available. ETH is funded and held positions are sellable.",
        "buy_only": "Buys are allowed. No held position is eligible for sell this tick.",
        "sell_only": "Zero ETH. Held positions may be sold, but new buys are unavailable.",
        "observe_only": "No action is currently executable. Observe if no higher-priority strategy overrides this.",
    }[permission_mode]


def _strategy_description(strategy_mode: str, held_symbol: str) -> str:
    return {
        "none": "No active strategies.",
        "no_new_buys": "HIGH strategy: no new buys are allowed this tick; only sells or observe are permitted.",
        "force_sell_held": f"HIGH strategy: immediately sell Asset {held_symbol} if any action is taken.",
        "force_observe": "HIGH strategy: record observation only; do not trade this tick.",
    }[strategy_mode]


def _risk_description(risk_mode: str) -> str:
    return {
        "neutral": "Asset Risk Preference: 3 / 5. Normal balance between stability and opportunity.",
        "low_risk": "Asset Risk Preference: 1 / 5. Prefer safer, more mature setups and avoid fragile spikes.",
        "high_risk": "Asset Risk Preference: 5 / 5. High-volatility setups are acceptable if edge is strong.",
    }[risk_mode]


def _portfolio_description(permission_mode: str, held_symbol: str) -> str:
    return {
        "buy_and_sell": f"ETH available. Held asset: {held_symbol}.",
        "buy_only": "ETH available. No held asset is currently sellable.",
        "sell_only": f"ETH balance is zero. Held asset: {held_symbol}.",
        "observe_only": "ETH balance is zero. No held asset is currently sellable.",
    }[permission_mode]


def _render_user_prompt(
    example_id: str,
    *,
    permission_mode: str,
    strategy_mode: str,
    risk_mode: str,
    held_symbol: str,
    assets: list[SyntheticAsset],
) -> str:
    lines = [
        f"## SYNTHETIC POLICY SCENARIO {example_id}",
        "",
        "These assets are neutral synthetic placeholders, not real tickers.",
        "",
        "## ACTIVE SETTINGS",
        f"- {_risk_description(risk_mode)}",
        f"- Permission mode: {permission_mode}",
        f"- {_permission_description(permission_mode)}",
        f"- Strategy mode: {strategy_mode}",
        f"- {_strategy_description(strategy_mode, held_symbol)}",
        f"- Portfolio context: {_portfolio_description(permission_mode, held_symbol)}",
        "- Policy note: first identify the market-preferred asset from the rows, then apply permission and strategy priority.",
        "",
        "## MARKET SNAPSHOT",
    ]
    for asset in assets:
        lines.extend(_format_asset_block(asset))
    lines.extend([
        "",
        "Respond with the single best action for this tick: buy, sell, or observe.",
    ])
    return "\n".join(lines)


def _neutral_market_assets(seed_idx: int) -> tuple[list[SyntheticAsset], str, str]:
    a, b, c, d = _symbol_set(seed_idx)
    assets = [
        _make_asset(a, "flow_backed_continuation", jitter_index=seed_idx * 7 + 1),
        _make_asset(b, "stable_winner", jitter_index=seed_idx * 7 + 2),
        _make_asset(c, "mean_reverter", jitter_index=seed_idx * 7 + 3),
        _make_asset(d, "crowded_risk", jitter_index=seed_idx * 7 + 4),
    ]
    return assets, a, c


def _risk_market_assets(seed_idx: int) -> tuple[list[SyntheticAsset], str, str]:
    a, b, c, d = _symbol_set(seed_idx)
    assets = [
        _make_asset(a, "noisy_pump", jitter_index=seed_idx * 11 + 1),
        _make_asset(b, "stable_winner", jitter_index=seed_idx * 11 + 2),
        _make_asset(c, "mean_reverter", jitter_index=seed_idx * 11 + 3),
        _make_asset(d, "flow_backed_continuation", jitter_index=seed_idx * 11 + 4),
    ]
    return assets, a, c


def _policy_best_asset(assets: list[SyntheticAsset], *, risk_mode: str) -> str | None:
    scores = [
        _score_asset(asset, {"neutral": "market_only", "low_risk": "low_risk", "high_risk": "high_risk"}[risk_mode])
        for asset in assets
    ]
    edge_after_fee = [row["edge_after_fee_score"] for row in scores]
    best_idx = max(range(len(assets)), key=lambda idx: edge_after_fee[idx])
    if edge_after_fee[best_idx] <= 0.0:
        return None
    return assets[best_idx].symbol


def _resolve_expected_action(
    *,
    permission_mode: str,
    strategy_mode: str,
    policy_best_asset: str | None,
    held_symbol: str,
) -> tuple[str, str | None]:
    allow_buy = permission_mode in {"buy_and_sell", "buy_only"}
    allow_sell = permission_mode in {"buy_and_sell", "sell_only"}

    if strategy_mode == "force_observe":
        return "observe", None
    if strategy_mode == "force_sell_held":
        return ("sell", held_symbol) if allow_sell else ("observe", None)
    if strategy_mode == "no_new_buys":
        return ("sell", held_symbol) if allow_sell else ("observe", None)

    if allow_buy and policy_best_asset is not None:
        return "buy", policy_best_asset
    if permission_mode == "sell_only" and allow_sell:
        return "sell", held_symbol
    return "observe", None


def _base_tick_row(
    *,
    example_id: str,
    family: str,
    family_variant: str,
    user_prompt: str,
    labels: dict[str, Any],
    assets: list[SyntheticAsset],
) -> SyntheticMarketExample:
    return SyntheticMarketExample(
        log_id=0,
        example_id=example_id,
        family=family,
        family_variant=family_variant,
        context_variant="market_only",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        prompt_messages=(
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ),
        labels=labels,
        assets=tuple(assets),
    )


def _build_policy_labels(
    *,
    example_id: str,
    family: str,
    family_variant: str,
    scenario_group: str,
    assets: list[SyntheticAsset],
    permission_mode: str,
    strategy_mode: str,
    risk_mode: str,
    held_symbol: str,
    policy_best_override: str | None = None,
) -> dict[str, Any]:
    market_labels = _compute_labels(example_id, family, family_variant, "market_only", assets)
    market_best_asset = market_labels["best_asset"]
    policy_best_asset = policy_best_override or _policy_best_asset(assets, risk_mode=risk_mode)
    expected_action_type, expected_action_asset = _resolve_expected_action(
        permission_mode=permission_mode,
        strategy_mode=strategy_mode,
        policy_best_asset=policy_best_asset,
        held_symbol=held_symbol,
    )
    return {
        **market_labels,
        "best_asset": policy_best_asset or market_best_asset,
        "buy_any": int(expected_action_type == "buy"),
        "observe_vs_act": "observe" if expected_action_type == "observe" else "act",
        "scenario_group": scenario_group,
        "permission_mode": permission_mode,
        "strategy_mode": strategy_mode,
        "risk_mode": risk_mode,
        "market_best_asset": market_best_asset,
        "policy_best_asset": policy_best_asset,
        "held_symbol": held_symbol,
        "expected_action_type": expected_action_type,
        "expected_action_asset": expected_action_asset,
        "policy_changes_best_asset": int(
            market_best_asset is not None
            and policy_best_asset is not None
            and market_best_asset != policy_best_asset
        ),
    }


def _generate_permission_grid(config: SyntheticPolicyConfig) -> list[SyntheticMarketExample]:
    examples: list[SyntheticMarketExample] = []
    for seed_idx in range(config.scenario_seeds):
        assets, _, held_symbol = _neutral_market_assets(seed_idx)
        scenario_group = f"permission:{seed_idx:02d}"
        for permission_mode in PERMISSION_VARIANTS:
            family = "permission_grid"
            family_variant = permission_mode
            example_id = f"{scenario_group}:{permission_mode}"
            labels = _build_policy_labels(
                example_id=example_id,
                family=family,
                family_variant=family_variant,
                scenario_group=scenario_group,
                assets=assets,
                permission_mode=permission_mode,
                strategy_mode="none",
                risk_mode="neutral",
                held_symbol=held_symbol,
            )
            user_prompt = _render_user_prompt(
                example_id,
                permission_mode=permission_mode,
                strategy_mode="none",
                risk_mode="neutral",
                held_symbol=held_symbol,
                assets=assets,
            )
            examples.append(
                _base_tick_row(
                    example_id=example_id,
                    family=family,
                    family_variant=family_variant,
                    user_prompt=user_prompt,
                    labels=labels,
                    assets=assets,
                )
            )
    return examples


def _generate_strategy_grid(config: SyntheticPolicyConfig) -> list[SyntheticMarketExample]:
    examples: list[SyntheticMarketExample] = []
    for seed_idx in range(config.scenario_seeds):
        assets, _, held_symbol = _neutral_market_assets(seed_idx + 100)
        scenario_group = f"strategy:{seed_idx:02d}"
        for strategy_mode in STRATEGY_VARIANTS:
            family = "strategy_override_grid"
            family_variant = strategy_mode
            example_id = f"{scenario_group}:{strategy_mode}"
            labels = _build_policy_labels(
                example_id=example_id,
                family=family,
                family_variant=family_variant,
                scenario_group=scenario_group,
                assets=assets,
                permission_mode="buy_and_sell",
                strategy_mode=strategy_mode,
                risk_mode="neutral",
                held_symbol=held_symbol,
            )
            user_prompt = _render_user_prompt(
                example_id,
                permission_mode="buy_and_sell",
                strategy_mode=strategy_mode,
                risk_mode="neutral",
                held_symbol=held_symbol,
                assets=assets,
            )
            examples.append(
                _base_tick_row(
                    example_id=example_id,
                    family=family,
                    family_variant=family_variant,
                    user_prompt=user_prompt,
                    labels=labels,
                    assets=assets,
                )
            )
    return examples


def _generate_risk_grid(config: SyntheticPolicyConfig) -> list[SyntheticMarketExample]:
    examples: list[SyntheticMarketExample] = []
    for seed_idx in range(config.scenario_seeds):
        assets, risky_symbol, held_symbol = _risk_market_assets(seed_idx + 200)
        safe_symbol = assets[1].symbol
        scenario_group = f"risk:{seed_idx:02d}"
        for risk_mode in RISK_VARIANTS:
            family = "risk_gate_grid"
            family_variant = risk_mode
            example_id = f"{scenario_group}:{risk_mode}"
            labels = _build_policy_labels(
                example_id=example_id,
                family=family,
                family_variant=family_variant,
                scenario_group=scenario_group,
                assets=assets,
                permission_mode="buy_and_sell",
                strategy_mode="none",
                risk_mode=risk_mode,
                held_symbol=held_symbol,
                policy_best_override=safe_symbol if risk_mode == "low_risk" else risky_symbol,
            )
            user_prompt = _render_user_prompt(
                example_id,
                permission_mode="buy_and_sell",
                strategy_mode="none",
                risk_mode=risk_mode,
                held_symbol=held_symbol,
                assets=assets,
            )
            examples.append(
                _base_tick_row(
                    example_id=example_id,
                    family=family,
                    family_variant=family_variant,
                    user_prompt=user_prompt,
                    labels=labels,
                    assets=assets,
                )
            )
    return examples


def _assign_log_ids(
    examples: list[SyntheticMarketExample],
    *,
    base_log_id: int,
) -> list[SyntheticMarketExample]:
    ordered = sorted(
        examples,
        key=lambda ex: (ex.family, ex.example_id),
    )
    assigned: list[SyntheticMarketExample] = []
    for offset, example in enumerate(ordered):
        assigned.append(
            SyntheticMarketExample(
                log_id=base_log_id + offset,
                example_id=example.example_id,
                family=example.family,
                family_variant=example.family_variant,
                context_variant=example.context_variant,
                system_prompt=example.system_prompt,
                user_prompt=example.user_prompt,
                prompt_messages=example.prompt_messages,
                labels=example.labels,
                assets=example.assets,
            )
        )
    return assigned


def generate_dataset(config: SyntheticPolicyConfig) -> list[SyntheticMarketExample]:
    examples: list[SyntheticMarketExample] = []
    examples.extend(_generate_permission_grid(config))
    examples.extend(_generate_strategy_grid(config))
    examples.extend(_generate_risk_grid(config))
    return _assign_log_ids(examples, base_log_id=config.log_id_base)


def build_dataset(config: SyntheticPolicyConfig) -> dict[str, Any]:
    examples = generate_dataset(config)
    result = write_dataset(examples, config.output_dir)
    if config.upload:
        upload_result = upload_dataset(
            config.output_dir,
            phase_name=config.phase_name,
            replace_phase=True,
        )
        result["upload"] = upload_result
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the synthetic policy-algebra dataset")
    parser.add_argument("--phase-name", default="policy_algebra_v1")
    parser.add_argument("--output-dir", type=Path, default=Path("data/interp_exports/synthetic_policy"))
    parser.add_argument("--scenario-seeds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-id-base", type=int, default=2_145_000_000)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args(argv)

    config = SyntheticPolicyConfig(
        seed=args.seed,
        phase_name=args.phase_name,
        output_dir=args.output_dir,
        scenario_seeds=args.scenario_seeds,
        upload=args.upload,
        log_id_base=args.log_id_base,
    )
    result = build_dataset(config)
    print(json.dumps(result["summary"], indent=2))
    if "upload" in result:
        print(json.dumps(result["upload"], indent=2))


if __name__ == "__main__":
    main()
