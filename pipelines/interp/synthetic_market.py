"""Synthetic market dataset generator for manifold-oriented interp work.

This module builds small, controlled market snapshots with neutral asset
identities and labels by construction. The goal is to isolate clean latent
variables such as attractiveness, pairwise preference, and risk-adjusted
acceptability before returning to noisy DX-style prompts.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


SYSTEM_PROMPT = (
    "You are an autonomous trading agent. On each tick, choose exactly one action: "
    "buy, sell, or observe. Only buy when the expected edge clearly justifies fees. "
    "When no asset has enough edge, observe."
)

RISK_SETTING_TEXT = {
    "market_only": "No explicit settings are provided.",
    "low_risk": "Asset Risk Preference: 1 / 5. Prefer lower-risk, more stable assets.",
    "high_risk": "Asset Risk Preference: 5 / 5. High-volatility and fresh setups are acceptable.",
}


@dataclass(frozen=True)
class SyntheticAsset:
    symbol: str
    archetype: str
    pct_5m: float
    pct_1h: float
    net_flow_5m: float
    vol_5m: float
    vol_1h: float
    unique_traders_5m: int
    top20_holder_pct: float
    age_bucket: str
    profile_id: str | None = None


@dataclass(frozen=True)
class SyntheticMarketExample:
    log_id: int
    example_id: str
    family: str
    family_variant: str
    context_variant: str
    system_prompt: str
    user_prompt: str
    prompt_messages: tuple[dict[str, str], ...]
    labels: dict[str, Any]
    assets: tuple[SyntheticAsset, ...]


@dataclass
class SyntheticMarketConfig:
    seed: int = 42
    scalar_steps: int = 9
    pairwise_variants: int = 5
    archetype_variants: int = 4
    coupled_grid_steps: int = 11
    coupled_background_variants: int = 2
    coupled_minimal_templates: int = 1
    representation_steps: int = 7
    representation_background_variants: int = 3
    permutation_variants: int = 6
    profile_surface_variants: int = 4
    relation_roster_variants: int = 4
    relation_scale_variants: int = 3
    include_settings_variants: bool = True
    dataset_preset: str = "phase1"
    scalar_background_variants: int = 1
    minimal_scalar_templates: int = 0
    log_id_base: int | None = None
    output_dir: Path = Path("data/interp_exports/synthetic_market")


ARCHETYPES: dict[str, dict[str, Any]] = {
    "stable_winner": {
        "pct_5m": 3.0,
        "pct_1h": 10.0,
        "net_flow_5m": 1.8,
        "vol_5m": 5.5,
        "vol_1h": 28.0,
        "unique_traders_5m": 18,
        "top20_holder_pct": 28.0,
        "age_bucket": "mature",
    },
    "momentum_burst": {
        "pct_5m": 8.5,
        "pct_1h": 18.0,
        "net_flow_5m": 1.2,
        "vol_5m": 6.0,
        "vol_1h": 22.0,
        "unique_traders_5m": 21,
        "top20_holder_pct": 36.0,
        "age_bucket": "mid",
    },
    "flow_backed_continuation": {
        "pct_5m": 4.0,
        "pct_1h": 12.0,
        "net_flow_5m": 3.0,
        "vol_5m": 5.2,
        "vol_1h": 24.0,
        "unique_traders_5m": 20,
        "top20_holder_pct": 31.0,
        "age_bucket": "mid",
    },
    "noisy_pump": {
        "pct_5m": 10.0,
        "pct_1h": 6.0,
        "net_flow_5m": 0.4,
        "vol_5m": 8.0,
        "vol_1h": 14.0,
        "unique_traders_5m": 24,
        "top20_holder_pct": 49.0,
        "age_bucket": "fresh",
    },
    "fading_leader": {
        "pct_5m": -3.5,
        "pct_1h": 8.0,
        "net_flow_5m": -1.4,
        "vol_5m": 4.6,
        "vol_1h": 18.0,
        "unique_traders_5m": 13,
        "top20_holder_pct": 39.0,
        "age_bucket": "mid",
    },
    "illiquid_spike": {
        "pct_5m": 7.0,
        "pct_1h": 4.0,
        "net_flow_5m": 0.3,
        "vol_5m": 1.4,
        "vol_1h": 4.2,
        "unique_traders_5m": 4,
        "top20_holder_pct": 58.0,
        "age_bucket": "fresh",
    },
    "crowded_risk": {
        "pct_5m": 2.5,
        "pct_1h": 7.0,
        "net_flow_5m": 0.8,
        "vol_5m": 3.8,
        "vol_1h": 16.0,
        "unique_traders_5m": 11,
        "top20_holder_pct": 67.0,
        "age_bucket": "mid",
    },
    "mean_reverter": {
        "pct_5m": -4.5,
        "pct_1h": -9.0,
        "net_flow_5m": 1.1,
        "vol_5m": 4.4,
        "vol_1h": 15.0,
        "unique_traders_5m": 14,
        "top20_holder_pct": 33.0,
        "age_bucket": "mature",
    },
}


SCALAR_ANCHOR_ARCHETYPES = {
    "pct_5m": "momentum_burst",
    "net_flow_5m": "flow_backed_continuation",
    "top20_holder_pct": "crowded_risk",
}


COUPLED_FACTOR_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "pct_5m__unique_traders_5m",
        "anchor_archetype": "momentum_burst",
        "metric_x": ("pct_5m", -10.0, 12.0),
        "metric_y": ("unique_traders_5m", 4.0, 28.0),
    },
    {
        "name": "pct_5m__top20_holder_pct",
        "anchor_archetype": "crowded_risk",
        "metric_x": ("pct_5m", -10.0, 12.0),
        "metric_y": ("top20_holder_pct", 18.0, 78.0),
    },
    {
        "name": "pct_5m__net_flow_5m",
        "anchor_archetype": "flow_backed_continuation",
        "metric_x": ("pct_5m", -10.0, 12.0),
        "metric_y": ("net_flow_5m", -2.8, 3.2),
    },
)


REPRESENTATION_BACKGROUND_ROSTERS: list[tuple[str, str]] = [
    ("mean_reverter", "illiquid_spike"),
    ("stable_winner", "flow_backed_continuation"),
    ("crowded_risk", "stable_winner"),
]


SYMBOL_PERMUTATION_LAYOUTS: list[dict[str, tuple[int, ...] | tuple[str, ...]]] = [
    {"order": (0, 1, 2, 3), "symbols": ("A", "B", "C", "D")},
    {"order": (1, 0, 3, 2), "symbols": ("C", "A", "D", "B")},
    {"order": (2, 3, 0, 1), "symbols": ("B", "D", "A", "C")},
    {"order": (3, 2, 1, 0), "symbols": ("D", "C", "B", "A")},
    {"order": (1, 3, 0, 2), "symbols": ("A", "D", "C", "B")},
    {"order": (2, 0, 3, 1), "symbols": ("B", "C", "D", "A")},
]


PROFILE_INVARIANCE_SURFACE_STYLES: list[dict[str, Any]] = [
    {"name": "canonical", "symbols": ("A", "B", "C", "D")},
    {"name": "reordered", "symbols": ("Alpha", "Beta", "Gamma", "Delta")},
    {"name": "compact", "symbols": ("North", "South", "East", "West")},
    {"name": "analyst", "symbols": ("One", "Two", "Three", "Four")},
]


RELATION_INVARIANCE_SURFACE_STYLES: list[dict[str, Any]] = [
    {"name": "canonical", "symbols": ("A", "B", "C", "D")},
    {"name": "analyst", "symbols": ("North", "South", "East", "West")},
]


RELATION_INVARIANCE_SCALE_FACTORS: tuple[tuple[str, float], ...] = (
    ("compressed", 0.82),
    ("baseline", 1.00),
    ("expanded", 1.18),
)


SCALAR_BACKGROUND_ROSTERS: list[tuple[str, str, str]] = [
    ("stable_winner", "flow_backed_continuation", "crowded_risk"),
    ("stable_winner", "mean_reverter", "illiquid_spike"),
    ("flow_backed_continuation", "stable_winner", "fading_leader"),
    ("crowded_risk", "mean_reverter", "stable_winner"),
    ("illiquid_spike", "stable_winner", "flow_backed_continuation"),
    ("mean_reverter", "crowded_risk", "stable_winner"),
]


MINIMAL_TEMPLATES: list[dict[str, Any]] = [
    {
        "archetype": "flat_neutral",
        "pct_5m": 0.6,
        "pct_1h": 1.1,
        "net_flow_5m": 0.05,
        "vol_5m": 2.4,
        "vol_1h": 9.6,
        "unique_traders_5m": 9,
        "top20_holder_pct": 31.0,
        "age_bucket": "mid",
    },
    {
        "archetype": "flat_cautious",
        "pct_5m": -0.4,
        "pct_1h": 0.5,
        "net_flow_5m": -0.02,
        "vol_5m": 2.1,
        "vol_1h": 8.3,
        "unique_traders_5m": 8,
        "top20_holder_pct": 33.5,
        "age_bucket": "mature",
    },
]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _age_score(age_bucket: str) -> float:
    return {"fresh": -0.45, "mid": 0.0, "mature": 0.25}[age_bucket]


def _age_risk_penalty(age_bucket: str) -> float:
    return {"fresh": 1.0, "mid": 0.45, "mature": 0.1}[age_bucket]


def _risk_multiplier(context_variant: str) -> float:
    if context_variant == "low_risk":
        return 1.0
    if context_variant == "high_risk":
        return 0.15
    return 0.55


def _score_asset(asset: SyntheticAsset, context_variant: str) -> dict[str, float]:
    momentum = 0.55 * asset.pct_5m + 0.45 * asset.pct_1h
    participation = 0.18 * asset.vol_5m + 0.06 * asset.vol_1h + 0.22 * asset.unique_traders_5m
    flow = 1.6 * asset.net_flow_5m
    concentration_penalty = 0.12 * max(0.0, asset.top20_holder_pct - 35.0)
    freshness = _age_score(asset.age_bucket)
    riskiness = (
        0.06 * abs(asset.pct_5m)
        + 0.03 * abs(asset.pct_1h)
        + 0.02 * max(0.0, asset.top20_holder_pct - 30.0)
        + _age_risk_penalty(asset.age_bucket)
    )
    attractiveness = 0.08 * momentum + 0.05 * participation + flow + freshness - concentration_penalty
    risk_adjusted = attractiveness - _risk_multiplier(context_variant) * riskiness
    edge_after_fee = risk_adjusted - 0.55
    return {
        "momentum_score": momentum,
        "participation_score": participation,
        "flow_score": flow,
        "concentration_penalty": concentration_penalty,
        "riskiness_score": riskiness,
        "attractiveness_score": attractiveness,
        "risk_adjusted_score": risk_adjusted,
        "edge_after_fee_score": edge_after_fee,
        "edge_gt_fee": float(edge_after_fee > 0.0),
    }


def _ordinal_ranks(values: list[float], reverse: bool = True) -> list[int]:
    ordered = sorted(range(len(values)), key=lambda idx: values[idx], reverse=reverse)
    ranks = [0] * len(values)
    for rank, idx in enumerate(ordered, start=1):
        ranks[idx] = rank
    return ranks


def _compute_labels(example_id: str, family: str, family_variant: str, context_variant: str, assets: list[SyntheticAsset]) -> dict[str, Any]:
    per_asset = [_score_asset(asset, context_variant) for asset in assets]
    attractiveness = [row["attractiveness_score"] for row in per_asset]
    risk_adjusted = [row["risk_adjusted_score"] for row in per_asset]
    edge_after_fee = [row["edge_after_fee_score"] for row in per_asset]
    best_idx = max(range(len(assets)), key=lambda idx: edge_after_fee[idx])
    buy_any = edge_after_fee[best_idx] > 0.0
    asset_labels: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    attractiveness_ranks = _ordinal_ranks(attractiveness)
    risk_ranks = _ordinal_ranks(risk_adjusted)
    for idx, asset in enumerate(assets):
        row = {
            "example_id": example_id,
            "family": family,
            "family_variant": family_variant,
            "context_variant": context_variant,
            "row_index": idx,
            "symbol": asset.symbol,
            "archetype": asset.archetype,
            **asdict(asset),
            **per_asset[idx],
            "attractiveness_rank": attractiveness_ranks[idx],
            "risk_adjusted_rank": risk_ranks[idx],
            "is_best_asset": int(idx == best_idx and buy_any),
            "buyable_if_unconstrained": int(per_asset[idx]["edge_after_fee_score"] > 0.0),
            "acceptable_under_risk_setting": int(per_asset[idx]["risk_adjusted_score"] > 0.0),
        }
        asset_labels.append(row)

    for i, asset_i in enumerate(assets):
        for j, asset_j in enumerate(assets):
            if i == j:
                continue
            pairwise_rows.append({
                "example_id": example_id,
                "family": family,
                "family_variant": family_variant,
                "context_variant": context_variant,
                "asset_a": asset_i.symbol,
                "asset_b": asset_j.symbol,
                "a_beats_b_on_attractiveness": int(attractiveness[i] > attractiveness[j]),
                "a_beats_b_on_risk_adjusted": int(risk_adjusted[i] > risk_adjusted[j]),
                "delta_pct_5m": asset_i.pct_5m - asset_j.pct_5m,
                "delta_pct_1h": asset_i.pct_1h - asset_j.pct_1h,
                "delta_net_flow_5m": asset_i.net_flow_5m - asset_j.net_flow_5m,
                "delta_vol_5m": asset_i.vol_5m - asset_j.vol_5m,
                "delta_unique_traders_5m": asset_i.unique_traders_5m - asset_j.unique_traders_5m,
                "delta_top20_holder_pct": asset_i.top20_holder_pct - asset_j.top20_holder_pct,
            })

    return {
        "example_id": example_id,
        "family": family,
        "family_variant": family_variant,
        "context_variant": context_variant,
        "best_asset": assets[best_idx].symbol if buy_any else None,
        "buy_any": int(buy_any),
        "observe_vs_act": "act" if buy_any else "observe",
        "market_only_best_asset": None,
        "settings_adjusted_best_asset": assets[best_idx].symbol if context_variant != "market_only" and buy_any else None,
        "asset_rows": asset_labels,
        "pairwise_rows": pairwise_rows,
    }


def _render_asset_lines(asset: SyntheticAsset, *, surface_style: str = "canonical") -> list[str]:
    if surface_style == "canonical":
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
    if surface_style == "reordered":
        return [
            f"- Asset {asset.symbol}",
            f"  - Holder concentration (top 20): {asset.top20_holder_pct:.1f}%",
            f"  - Active traders over 5m: {asset.unique_traders_5m}",
            f"  - Recent net flow (5m): {asset.net_flow_5m:+.2f}",
            f"  - 1h move: {asset.pct_1h:+.1f}%",
            f"  - 5m move: {asset.pct_5m:+.1f}%",
            f"  - Archetype family: {asset.archetype}",
            f"  - 1h volume: {asset.vol_1h:.2f}",
            f"  - 5m volume: {asset.vol_5m:.2f}",
            f"  - Age cohort: {asset.age_bucket}",
        ]
    if surface_style == "compact":
        return [
            f"- Asset {asset.symbol}",
            "  - Snapshot: "
            f"5m={asset.pct_5m:+.1f}%; "
            f"1h={asset.pct_1h:+.1f}%; "
            f"flow5m={asset.net_flow_5m:+.2f}; "
            f"traders5m={asset.unique_traders_5m}; "
            f"top20={asset.top20_holder_pct:.1f}%; "
            f"vol5m={asset.vol_5m:.2f}; "
            f"vol1h={asset.vol_1h:.2f}; "
            f"age={asset.age_bucket}; "
            f"archetype={asset.archetype}",
        ]
    if surface_style == "analyst":
        return [
            f"- Asset {asset.symbol}",
            f"  - Short-horizon price move: {asset.pct_5m:+.1f}% over 5m",
            f"  - Hourly continuation: {asset.pct_1h:+.1f}% over 1h",
            f"  - Participation pulse: {asset.unique_traders_5m} traders in 5m",
            f"  - Concentration check: top-20 holders own {asset.top20_holder_pct:.1f}%",
            f"  - Flow tape: {asset.net_flow_5m:+.2f} net over 5m",
            f"  - Liquidity tape: {asset.vol_5m:.2f} / {asset.vol_1h:.2f} volume",
            f"  - Lifecycle bucket: {asset.age_bucket}",
            f"  - Archetype read: {asset.archetype}",
        ]
    raise ValueError(f"Unsupported surface_style: {surface_style}")


def _render_user_prompt(
    example_id: str,
    context_variant: str,
    assets: list[SyntheticAsset],
    *,
    surface_style: str = "canonical",
) -> str:
    lines = [
        f"## SYNTHETIC MARKET SCENARIO {example_id}",
        "",
        "These assets are neutral synthetic placeholders, not real tickers.",
        "",
        "## ACTIVE SETTINGS",
        f"- {RISK_SETTING_TEXT[context_variant]}",
        "",
        "## MARKET SNAPSHOT",
    ]
    for asset in assets:
        lines.extend(_render_asset_lines(asset, surface_style=surface_style))
    lines.extend([
        "",
        "Respond with the single best action for this tick: buy, sell, or observe.",
    ])
    return "\n".join(lines)


def _make_asset(symbol: str, archetype: str, jitter_index: int = 0) -> SyntheticAsset:
    base = ARCHETYPES[archetype]
    # Small deterministic perturbation keeps archetype families from collapsing to a single point.
    delta = 0.35 * math.sin(jitter_index + len(symbol))
    return SyntheticAsset(
        symbol=symbol,
        archetype=archetype,
        pct_5m=round(base["pct_5m"] + delta, 2),
        pct_1h=round(base["pct_1h"] + 1.7 * delta, 2),
        net_flow_5m=round(base["net_flow_5m"] + 0.15 * delta, 3),
        vol_5m=round(_clamp(base["vol_5m"] + 0.35 * delta, 0.4, 12.0), 3),
        vol_1h=round(_clamp(base["vol_1h"] + 0.9 * delta, 1.0, 45.0), 3),
        unique_traders_5m=max(1, int(round(base["unique_traders_5m"] + 1.2 * delta))),
        top20_holder_pct=round(_clamp(base["top20_holder_pct"] + 0.7 * delta, 15.0, 85.0), 2),
        age_bucket=base["age_bucket"],
    )


def _override_metric(asset: SyntheticAsset, metric_name: str, value: float) -> SyntheticAsset:
    payload = asdict(asset)
    if metric_name == "pct_5m":
        payload["pct_5m"] = round(value, 2)
    elif metric_name == "net_flow_5m":
        payload["net_flow_5m"] = round(value, 3)
    elif metric_name == "unique_traders_5m":
        payload["unique_traders_5m"] = max(1, int(round(value)))
    elif metric_name == "top20_holder_pct":
        payload["top20_holder_pct"] = round(value, 2)
    else:
        raise ValueError(f"Unsupported metric_name: {metric_name}")
    return SyntheticAsset(**payload)


def _override_metrics(asset: SyntheticAsset, overrides: dict[str, float]) -> SyntheticAsset:
    updated = asset
    for metric_name, value in overrides.items():
        updated = _override_metric(updated, metric_name, value)
    return updated


def _override_display(asset: SyntheticAsset, *, symbol: str | None = None, profile_id: str | None = None) -> SyntheticAsset:
    payload = asdict(asset)
    if symbol is not None:
        payload["symbol"] = symbol
    if profile_id is not None:
        payload["profile_id"] = profile_id
    return SyntheticAsset(**payload)


def _make_custom_asset(
    symbol: str,
    *,
    archetype: str,
    pct_5m: float,
    pct_1h: float,
    net_flow_5m: float,
    vol_5m: float,
    vol_1h: float,
    unique_traders_5m: int,
    top20_holder_pct: float,
    age_bucket: str,
    profile_id: str | None = None,
) -> SyntheticAsset:
    return SyntheticAsset(
        symbol=symbol,
        archetype=archetype,
        pct_5m=round(pct_5m, 2),
        pct_1h=round(pct_1h, 2),
        net_flow_5m=round(net_flow_5m, 3),
        vol_5m=round(vol_5m, 3),
        vol_1h=round(vol_1h, 3),
        unique_traders_5m=max(1, int(unique_traders_5m)),
        top20_holder_pct=round(top20_holder_pct, 2),
        age_bucket=age_bucket,
        profile_id=profile_id,
    )


def _scale_market_magnitude(asset: SyntheticAsset, factor: float) -> SyntheticAsset:
    payload = asdict(asset)
    payload["pct_5m"] = round(payload["pct_5m"] * factor, 2)
    payload["pct_1h"] = round(payload["pct_1h"] * factor, 2)
    payload["net_flow_5m"] = round(payload["net_flow_5m"] * factor, 3)
    payload["vol_5m"] = round(_clamp(payload["vol_5m"] * factor, 0.4, 12.0), 3)
    payload["vol_1h"] = round(_clamp(payload["vol_1h"] * factor, 1.0, 45.0), 3)
    payload["unique_traders_5m"] = max(1, int(round(payload["unique_traders_5m"] * factor)))
    payload["top20_holder_pct"] = round(_clamp(35.0 + factor * (payload["top20_holder_pct"] - 35.0), 15.0, 85.0), 2)
    return SyntheticAsset(**payload)


def _make_minimal_asset(symbol: str, template_index: int, jitter_index: int = 0) -> SyntheticAsset:
    template = MINIMAL_TEMPLATES[template_index % len(MINIMAL_TEMPLATES)]
    delta = 0.08 * math.sin(jitter_index + len(symbol))
    return SyntheticAsset(
        symbol=symbol,
        archetype=str(template["archetype"]),
        pct_5m=round(float(template["pct_5m"]) + delta, 2),
        pct_1h=round(float(template["pct_1h"]) + 1.2 * delta, 2),
        net_flow_5m=round(float(template["net_flow_5m"]) + 0.04 * delta, 3),
        vol_5m=round(_clamp(float(template["vol_5m"]) + 0.12 * delta, 0.5, 12.0), 3),
        vol_1h=round(_clamp(float(template["vol_1h"]) + 0.25 * delta, 1.0, 45.0), 3),
        unique_traders_5m=max(1, int(round(float(template["unique_traders_5m"]) + 0.4 * delta))),
        top20_holder_pct=round(_clamp(float(template["top20_holder_pct"]) + 0.25 * delta, 15.0, 85.0), 2),
        age_bucket=str(template["age_bucket"]),
    )


def _apply_context_variants(
    example_id: str,
    family: str,
    family_variant: str,
    base_assets: list[SyntheticAsset],
    include_settings_variants: bool,
    surface_style: str = "canonical",
) -> list[SyntheticMarketExample]:
    variants = ["market_only"]
    if include_settings_variants:
        variants.extend(["low_risk", "high_risk"])

    examples: list[SyntheticMarketExample] = []
    for context_variant in variants:
        labels = _compute_labels(example_id, family, family_variant, context_variant, base_assets)
        user_prompt = _render_user_prompt(
            example_id,
            context_variant,
            base_assets,
            surface_style=surface_style,
        )
        examples.append(
            SyntheticMarketExample(
                log_id=-1,
                example_id=example_id,
                family=family,
                family_variant=family_variant,
                context_variant=context_variant,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                prompt_messages=(
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ),
                labels=labels,
                assets=tuple(base_assets),
            )
        )
    return examples


def generate_scalar_sweeps(config: SyntheticMarketConfig) -> list[SyntheticMarketExample]:
    steps = max(3, config.scalar_steps)
    base_steps = [(-1.0 + 2.0 * i / (steps - 1)) for i in range(steps)]
    examples: list[SyntheticMarketExample] = []

    families = (
        ("pct_5m", -9.0, 11.0),
        ("net_flow_5m", -2.4, 2.8),
        ("top20_holder_pct", 20.0, 76.0),
    )
    for metric_name, lower, upper in families:
        for step_idx, alpha in enumerate(base_steps):
            a = _make_asset("A", "momentum_burst", jitter_index=step_idx)
            value = lower + (upper - lower) * ((alpha + 1.0) / 2.0)
            if metric_name == "pct_5m":
                a = SyntheticAsset(**{**asdict(a), "pct_5m": round(value, 2)})
            elif metric_name == "net_flow_5m":
                a = SyntheticAsset(**{**asdict(a), "net_flow_5m": round(value, 3)})
            else:
                a = SyntheticAsset(**{**asdict(a), "top20_holder_pct": round(value, 2)})

            base_assets = [
                a,
                _make_asset("B", "stable_winner", jitter_index=step_idx + 10),
                _make_asset("C", "flow_backed_continuation", jitter_index=step_idx + 20),
                _make_asset("D", "crowded_risk", jitter_index=step_idx + 30),
            ]
            example_id = f"scalar_{metric_name}_{step_idx:02d}"
            examples.extend(
                _apply_context_variants(
                    example_id,
                    family="scalar_sweep",
                    family_variant=metric_name,
                    base_assets=base_assets,
                    include_settings_variants=config.include_settings_variants,
                )
            )
    return examples


def generate_dense_scalar_sweeps(config: SyntheticMarketConfig) -> list[SyntheticMarketExample]:
    steps = max(5, config.scalar_steps)
    background_variants = max(1, config.scalar_background_variants)
    base_steps = [(-1.0 + 2.0 * i / (steps - 1)) for i in range(steps)]
    families = (
        ("pct_5m", -10.0, 12.0),
        ("net_flow_5m", -2.8, 3.2),
        ("top20_holder_pct", 18.0, 78.0),
    )
    examples: list[SyntheticMarketExample] = []

    for metric_name, lower, upper in families:
        anchor_archetype = SCALAR_ANCHOR_ARCHETYPES[metric_name]
        for roster_index in range(background_variants):
            distractor_roster = SCALAR_BACKGROUND_ROSTERS[roster_index % len(SCALAR_BACKGROUND_ROSTERS)]
            for step_idx, alpha in enumerate(base_steps):
                value = lower + (upper - lower) * ((alpha + 1.0) / 2.0)
                anchor = _make_asset(
                    "A",
                    anchor_archetype,
                    jitter_index=10_000 + 100 * roster_index + step_idx,
                )
                anchor = _override_metric(anchor, metric_name, value)
                base_assets = [anchor]
                for offset, distractor in enumerate(distractor_roster, start=1):
                    base_assets.append(
                        _make_asset(
                            chr(ord("A") + offset),
                            distractor,
                            jitter_index=10_000 + 100 * roster_index + 11 * offset + step_idx,
                        )
                    )

                example_id = f"dense_{metric_name}_r{roster_index:02d}_s{step_idx:02d}"
                examples.extend(
                    _apply_context_variants(
                        example_id,
                        family="scalar_sweep_dense",
                        family_variant=metric_name,
                        base_assets=base_assets,
                        include_settings_variants=config.include_settings_variants,
                    )
                )
    return examples


def generate_minimal_scalar_sweeps(config: SyntheticMarketConfig) -> list[SyntheticMarketExample]:
    steps = max(5, config.scalar_steps)
    template_count = max(1, config.minimal_scalar_templates)
    base_steps = [(-1.0 + 2.0 * i / (steps - 1)) for i in range(steps)]
    families = (
        ("pct_5m", -10.0, 12.0),
        ("net_flow_5m", -2.8, 3.2),
        ("top20_holder_pct", 18.0, 78.0),
    )
    examples: list[SyntheticMarketExample] = []

    for metric_name, lower, upper in families:
        anchor_archetype = SCALAR_ANCHOR_ARCHETYPES[metric_name]
        for template_index in range(template_count):
            for step_idx, alpha in enumerate(base_steps):
                value = lower + (upper - lower) * ((alpha + 1.0) / 2.0)
                anchor = _make_asset(
                    "A",
                    anchor_archetype,
                    jitter_index=20_000 + 100 * template_index + step_idx,
                )
                anchor = _override_metric(anchor, metric_name, value)
                base_assets = [anchor]
                for offset in range(1, 4):
                    base_assets.append(
                        _make_minimal_asset(
                            chr(ord("A") + offset),
                            template_index=template_index,
                            jitter_index=20_000 + 100 * template_index + 17 * offset + step_idx,
                        )
                    )

                example_id = f"minimal_{metric_name}_t{template_index:02d}_s{step_idx:02d}"
                examples.extend(
                    _apply_context_variants(
                        example_id,
                        family="scalar_sweep_minimal",
                        family_variant=metric_name,
                        base_assets=base_assets,
                        include_settings_variants=config.include_settings_variants,
                    )
                )
    return examples


def _linspace(lower: float, upper: float, steps: int) -> list[float]:
    if steps <= 1:
        return [round((lower + upper) / 2.0, 3)]
    return [
        lower + (upper - lower) * (step_idx / (steps - 1))
        for step_idx in range(steps)
    ]


def generate_coupled_factor_dense_grids(config: SyntheticMarketConfig) -> list[SyntheticMarketExample]:
    steps = max(5, config.coupled_grid_steps)
    background_variants = max(1, config.coupled_background_variants)
    examples: list[SyntheticMarketExample] = []

    for spec in COUPLED_FACTOR_SPECS:
        metric_x, lower_x, upper_x = spec["metric_x"]
        metric_y, lower_y, upper_y = spec["metric_y"]
        values_x = _linspace(lower_x, upper_x, steps)
        values_y = _linspace(lower_y, upper_y, steps)
        for roster_index in range(background_variants):
            distractor_roster = SCALAR_BACKGROUND_ROSTERS[roster_index % len(SCALAR_BACKGROUND_ROSTERS)]
            for x_idx, value_x in enumerate(values_x):
                for y_idx, value_y in enumerate(values_y):
                    jitter = 30_000 + 400 * roster_index + 17 * x_idx + y_idx
                    anchor = _make_asset("A", spec["anchor_archetype"], jitter_index=jitter)
                    anchor = _override_metric(anchor, metric_x, value_x)
                    anchor = _override_metric(anchor, metric_y, value_y)
                    base_assets = [anchor]
                    for offset, distractor in enumerate(distractor_roster, start=1):
                        base_assets.append(
                            _make_asset(
                                chr(ord("A") + offset),
                                distractor,
                                jitter_index=jitter + 19 * offset,
                            )
                        )
                    variant = f"{spec['name']}__bg{roster_index:02d}"
                    example_id = f"coupled_dense_{spec['name']}_r{roster_index:02d}_x{x_idx:02d}_y{y_idx:02d}"
                    examples.extend(
                        _apply_context_variants(
                            example_id,
                            family="coupled_factor_dense",
                            family_variant=variant,
                            base_assets=base_assets,
                            include_settings_variants=config.include_settings_variants,
                        )
                    )
    return examples


def generate_coupled_factor_minimal_grids(config: SyntheticMarketConfig) -> list[SyntheticMarketExample]:
    steps = max(5, config.coupled_grid_steps)
    template_count = max(1, config.coupled_minimal_templates)
    examples: list[SyntheticMarketExample] = []

    for spec in COUPLED_FACTOR_SPECS:
        metric_x, lower_x, upper_x = spec["metric_x"]
        metric_y, lower_y, upper_y = spec["metric_y"]
        values_x = _linspace(lower_x, upper_x, steps)
        values_y = _linspace(lower_y, upper_y, steps)
        for template_index in range(template_count):
            for x_idx, value_x in enumerate(values_x):
                for y_idx, value_y in enumerate(values_y):
                    jitter = 40_000 + 400 * template_index + 17 * x_idx + y_idx
                    anchor = _make_asset("A", spec["anchor_archetype"], jitter_index=jitter)
                    anchor = _override_metric(anchor, metric_x, value_x)
                    anchor = _override_metric(anchor, metric_y, value_y)
                    base_assets = [anchor]
                    for offset in range(1, 4):
                        base_assets.append(
                            _make_minimal_asset(
                                chr(ord("A") + offset),
                                template_index=template_index,
                                jitter_index=jitter + 23 * offset,
                            )
                        )
                    variant = f"{spec['name']}__t{template_index:02d}"
                    example_id = f"coupled_minimal_{spec['name']}_t{template_index:02d}_x{x_idx:02d}_y{y_idx:02d}"
                    examples.extend(
                        _apply_context_variants(
                            example_id,
                            family="coupled_factor_minimal",
                            family_variant=variant,
                            base_assets=base_assets,
                            include_settings_variants=config.include_settings_variants,
                        )
                    )
    return examples


def generate_pairwise_tradeoff_grids(config: SyntheticMarketConfig) -> list[SyntheticMarketExample]:
    examples: list[SyntheticMarketExample] = []

    def _alpha(step_idx: int) -> float:
        steps = max(3, config.pairwise_variants)
        return -1.0 + (2.0 * step_idx / (steps - 1))

    steps = max(3, config.pairwise_variants)
    for step_idx in range(steps):
        alpha = _alpha(step_idx)

        scenarios = [
            (
                f"momentum_vs_flow_s{step_idx:02d}",
                [
                    SyntheticAsset(
                        "A",
                        "momentum_burst",
                        round(8.0 + 2.8 * alpha, 2),
                        round(15.0 + 3.2 * alpha, 2),
                        round(0.45 - 0.25 * alpha, 3),
                        round(6.4 + 0.4 * alpha, 3),
                        round(21.0 + 0.8 * alpha, 3),
                        max(3, int(round(22 + alpha))),
                        round(39.0 + 1.5 * alpha, 2),
                        "mid",
                    ),
                    SyntheticAsset(
                        "B",
                        "flow_backed_continuation",
                        round(3.2 - 1.1 * alpha, 2),
                        round(9.5 - 1.4 * alpha, 2),
                        round(2.4 + 0.9 * alpha, 3),
                        round(5.1 + 0.2 * alpha, 3),
                        round(23.5 + 0.6 * alpha, 3),
                        max(3, int(round(17 + 1.3 * alpha))),
                        round(29.0 - 0.7 * alpha, 2),
                        "mid",
                    ),
                    _make_asset("C", "stable_winner", step_idx + 1),
                    _make_asset("D", "fading_leader", step_idx + 2),
                ],
            ),
            (
                f"participation_vs_concentration_s{step_idx:02d}",
                [
                    SyntheticAsset(
                        "A",
                        "crowded_risk",
                        round(4.5 + 0.9 * alpha, 2),
                        round(8.8 + 1.4 * alpha, 2),
                        round(1.1 + 0.15 * alpha, 3),
                        round(5.8 + 0.8 * alpha, 3),
                        round(24.0 + 1.4 * alpha, 3),
                        max(3, int(round(24 + 2.0 * alpha))),
                        round(71.0 - 6.0 * alpha, 2),
                        "mid",
                    ),
                    SyntheticAsset(
                        "B",
                        "stable_winner",
                        round(3.7 - 0.4 * alpha, 2),
                        round(8.1 - 0.7 * alpha, 2),
                        round(1.0 - 0.1 * alpha, 3),
                        round(4.9 - 0.3 * alpha, 3),
                        round(21.5 - 0.9 * alpha, 3),
                        max(3, int(round(17 - 1.0 * alpha))),
                        round(27.0 + 2.0 * alpha, 2),
                        "mature",
                    ),
                    _make_asset("C", "illiquid_spike", step_idx + 3),
                    _make_asset("D", "mean_reverter", step_idx + 4),
                ],
            ),
            (
                f"fresh_vs_mature_s{step_idx:02d}",
                [
                    SyntheticAsset(
                        "A",
                        "noisy_pump",
                        round(7.2 + 1.8 * alpha, 2),
                        round(12.4 + 2.1 * alpha, 2),
                        round(1.2 + 0.5 * alpha, 3),
                        round(6.9 + 0.7 * alpha, 3),
                        round(17.0 + 0.6 * alpha, 3),
                        max(3, int(round(22 + 1.5 * alpha))),
                        round(46.0 - 2.5 * alpha, 2),
                        "fresh",
                    ),
                    SyntheticAsset(
                        "B",
                        "stable_winner",
                        round(4.0 - 0.5 * alpha, 2),
                        round(9.1 - 0.7 * alpha, 2),
                        round(1.35 - 0.15 * alpha, 3),
                        round(5.4 - 0.2 * alpha, 3),
                        round(21.0 - 0.6 * alpha, 3),
                        max(3, int(round(16 - 0.7 * alpha))),
                        round(27.5 + 1.0 * alpha, 2),
                        "mature",
                    ),
                    _make_asset("C", "flow_backed_continuation", step_idx + 5),
                    _make_asset("D", "crowded_risk", step_idx + 6),
                ],
            ),
        ]

        for scenario_idx, (variant, assets) in enumerate(scenarios):
            example_id = f"pairwise_{scenario_idx:02d}_{step_idx:02d}"
            examples.extend(
                _apply_context_variants(
                    example_id,
                    family="pairwise_tradeoff",
                    family_variant=variant,
                    base_assets=assets,
                    include_settings_variants=config.include_settings_variants,
                )
            )
    return examples


def generate_hard_pairwise_tradeoff_grids(config: SyntheticMarketConfig) -> list[SyntheticMarketExample]:
    steps = max(5, config.representation_steps)
    background_variants = max(1, config.representation_background_variants)
    alphas = [(-1.0 + 2.0 * i / (steps - 1)) for i in range(steps)]
    examples: list[SyntheticMarketExample] = []

    for roster_index in range(background_variants):
        distractors = REPRESENTATION_BACKGROUND_ROSTERS[roster_index % len(REPRESENTATION_BACKGROUND_ROSTERS)]
        for step_idx, alpha in enumerate(alphas):
            scenarios: list[tuple[str, list[SyntheticAsset]]] = []

            a = _override_metrics(
                _make_asset("A", "momentum_burst", jitter_index=50_000 + 100 * roster_index + step_idx),
                {
                    "pct_5m": 6.2 + 1.3 * alpha,
                    "net_flow_5m": 0.85 - 0.45 * alpha,
                    "top20_holder_pct": 34.0 + 1.4 * alpha,
                },
            )
            b = _override_metrics(
                _make_asset("B", "flow_backed_continuation", jitter_index=50_000 + 100 * roster_index + 30 + step_idx),
                {
                    "pct_5m": 4.9 - 0.8 * alpha,
                    "net_flow_5m": 1.55 + 0.40 * alpha,
                    "top20_holder_pct": 29.0 - 0.6 * alpha,
                },
            )
            scenarios.append((
                "momentum_vs_flow_near_tie",
                [
                    a,
                    b,
                    _make_asset("C", distractors[0], jitter_index=50_000 + 100 * roster_index + 61 + step_idx),
                    _make_asset("D", distractors[1], jitter_index=50_000 + 100 * roster_index + 79 + step_idx),
                ],
            ))

            a = _override_metrics(
                _make_asset("A", "crowded_risk", jitter_index=51_000 + 100 * roster_index + step_idx),
                {
                    "unique_traders_5m": 25 + 2.2 * alpha,
                    "top20_holder_pct": 60.0 - 8.5 * alpha,
                    "pct_5m": 4.4 + 0.7 * alpha,
                },
            )
            b = _override_metrics(
                _make_asset("B", "stable_winner", jitter_index=51_000 + 100 * roster_index + 31 + step_idx),
                {
                    "unique_traders_5m": 16 - 0.9 * alpha,
                    "top20_holder_pct": 27.0 + 1.8 * alpha,
                    "pct_5m": 4.1 - 0.3 * alpha,
                },
            )
            scenarios.append((
                "participation_vs_concentration_near_tie",
                [
                    a,
                    b,
                    _make_asset("C", distractors[0], jitter_index=51_000 + 100 * roster_index + 63 + step_idx),
                    _make_asset("D", distractors[1], jitter_index=51_000 + 100 * roster_index + 81 + step_idx),
                ],
            ))

            a = _override_metrics(
                _make_asset("A", "noisy_pump", jitter_index=52_000 + 100 * roster_index + step_idx),
                {
                    "pct_5m": 6.0 + 1.1 * alpha,
                    "net_flow_5m": 1.15 + 0.25 * alpha,
                    "top20_holder_pct": 42.0 - 1.5 * alpha,
                },
            )
            b = _override_metrics(
                _make_asset("B", "stable_winner", jitter_index=52_000 + 100 * roster_index + 29 + step_idx),
                {
                    "pct_5m": 4.2 - 0.4 * alpha,
                    "net_flow_5m": 1.20 - 0.15 * alpha,
                    "top20_holder_pct": 27.0 + 0.8 * alpha,
                },
            )
            scenarios.append((
                "fresh_vs_mature_near_tie",
                [
                    a,
                    b,
                    _make_asset("C", distractors[0], jitter_index=52_000 + 100 * roster_index + 65 + step_idx),
                    _make_asset("D", distractors[1], jitter_index=52_000 + 100 * roster_index + 83 + step_idx),
                ],
            ))

            for scenario_idx, (variant, assets) in enumerate(scenarios):
                example_id = f"hard_pairwise_r{roster_index:02d}_{scenario_idx:02d}_{step_idx:02d}"
                examples.extend(
                    _apply_context_variants(
                        example_id,
                        family="pairwise_tradeoff_hard",
                        family_variant=variant,
                        base_assets=assets,
                        include_settings_variants=config.include_settings_variants,
                    )
                )
    return examples


def generate_rank_context_tradeoffs(config: SyntheticMarketConfig) -> list[SyntheticMarketExample]:
    background_variants = max(2, config.representation_background_variants)
    examples: list[SyntheticMarketExample] = []

    focal_scenarios = [
        (
            "fixed_momentum_flow_pair",
            _override_metrics(
                _make_asset("A", "momentum_burst", jitter_index=60_001),
                {"pct_5m": 5.7, "net_flow_5m": 0.95, "top20_holder_pct": 33.5},
            ),
            _override_metrics(
                _make_asset("B", "flow_backed_continuation", jitter_index=60_002),
                {"pct_5m": 4.9, "net_flow_5m": 1.45, "top20_holder_pct": 29.0},
            ),
        ),
        (
            "fixed_participation_concentration_pair",
            _override_metrics(
                _make_asset("A", "crowded_risk", jitter_index=60_101),
                {"unique_traders_5m": 23.0, "top20_holder_pct": 55.0, "pct_5m": 4.3},
            ),
            _override_metrics(
                _make_asset("B", "stable_winner", jitter_index=60_102),
                {"unique_traders_5m": 16.0, "top20_holder_pct": 27.0, "pct_5m": 4.0},
            ),
        ),
    ]

    background_progressions = [
        [
            _make_asset("C", "mean_reverter", jitter_index=61_000),
            _make_asset("D", "illiquid_spike", jitter_index=61_001),
        ],
        [
            _make_asset("C", "stable_winner", jitter_index=61_010),
            _make_asset("D", "mean_reverter", jitter_index=61_011),
        ],
        [
            _make_asset("C", "flow_backed_continuation", jitter_index=61_020),
            _make_asset("D", "stable_winner", jitter_index=61_021),
        ],
    ]

    for scenario_idx, (variant, focal_a, focal_b) in enumerate(focal_scenarios):
        for bg_idx in range(background_variants):
            c, d = background_progressions[bg_idx % len(background_progressions)]
            assets = [focal_a, focal_b, c, d]
            example_id = f"rank_context_{scenario_idx:02d}_{bg_idx:02d}"
            examples.extend(
                _apply_context_variants(
                    example_id,
                    family="rank_context_tradeoff",
                    family_variant=f"{variant}__bg{bg_idx:02d}",
                    base_assets=assets,
                    include_settings_variants=config.include_settings_variants,
                )
            )
    return examples


def generate_symbol_permutation_controls(config: SyntheticMarketConfig) -> list[SyntheticMarketExample]:
    examples: list[SyntheticMarketExample] = []
    layouts = SYMBOL_PERMUTATION_LAYOUTS[: max(2, min(config.permutation_variants, len(SYMBOL_PERMUTATION_LAYOUTS)))]

    base_scenarios: list[tuple[str, list[SyntheticAsset]]] = [
        (
            "momentum_flow_permuted_market",
            [
                _override_display(
                    _override_metrics(
                        _make_asset("A", "momentum_burst", jitter_index=70_001),
                        {"pct_5m": 5.8, "net_flow_5m": 0.90, "top20_holder_pct": 33.0},
                    ),
                    profile_id="profile_momentum_anchor",
                ),
                _override_display(
                    _override_metrics(
                        _make_asset("B", "flow_backed_continuation", jitter_index=70_002),
                        {"pct_5m": 4.7, "net_flow_5m": 1.50, "top20_holder_pct": 28.5},
                    ),
                    profile_id="profile_flow_anchor",
                ),
                _override_display(
                    _make_asset("C", "stable_winner", jitter_index=70_003),
                    profile_id="profile_stable_distractor",
                ),
                _override_display(
                    _make_asset("D", "mean_reverter", jitter_index=70_004),
                    profile_id="profile_mean_reverter",
                ),
            ],
        ),
        (
            "participation_concentration_permuted_market",
            [
                _override_display(
                    _override_metrics(
                        _make_asset("A", "crowded_risk", jitter_index=71_001),
                        {"unique_traders_5m": 24.0, "top20_holder_pct": 58.0, "pct_5m": 4.2},
                    ),
                    profile_id="profile_crowded_high_participation",
                ),
                _override_display(
                    _override_metrics(
                        _make_asset("B", "stable_winner", jitter_index=71_002),
                        {"unique_traders_5m": 16.0, "top20_holder_pct": 26.5, "pct_5m": 4.0},
                    ),
                    profile_id="profile_stable_low_concentration",
                ),
                _override_display(
                    _make_asset("C", "illiquid_spike", jitter_index=71_003),
                    profile_id="profile_illiquid_spike",
                ),
                _override_display(
                    _make_asset("D", "flow_backed_continuation", jitter_index=71_004),
                    profile_id="profile_flow_distractor",
                ),
            ],
        ),
    ]

    for scenario_idx, (variant, base_assets) in enumerate(base_scenarios):
        for perm_idx, layout in enumerate(layouts):
            order = tuple(int(idx) for idx in layout["order"])
            symbols = tuple(str(symbol) for symbol in layout["symbols"])
            permuted_assets: list[SyntheticAsset] = []
            for row_index, asset_index in enumerate(order):
                permuted_assets.append(
                    _override_display(
                        base_assets[asset_index],
                        symbol=symbols[row_index],
                    )
                )
            example_id = f"symbol_perm_{scenario_idx:02d}_{perm_idx:02d}"
            examples.extend(
                _apply_context_variants(
                    example_id,
                    family="symbol_permutation_control",
                    family_variant=variant,
                    base_assets=permuted_assets,
                    include_settings_variants=False,
                )
            )
    return examples


def generate_profile_invariance_controls(config: SyntheticMarketConfig) -> list[SyntheticMarketExample]:
    examples: list[SyntheticMarketExample] = []
    layouts = SYMBOL_PERMUTATION_LAYOUTS[: max(2, min(config.permutation_variants, len(SYMBOL_PERMUTATION_LAYOUTS)))]
    surface_styles = PROFILE_INVARIANCE_SURFACE_STYLES[
        : max(2, min(config.profile_surface_variants, len(PROFILE_INVARIANCE_SURFACE_STYLES)))
    ]

    base_scenarios: list[tuple[str, list[SyntheticAsset]]] = [
        (
            "participation_concentration_tiebreak",
            [
                _override_display(
                    _override_metrics(
                        _make_asset("A", "stable_winner", jitter_index=72_001),
                        {"pct_5m": 4.4, "net_flow_5m": 1.10, "unique_traders_5m": 18.0, "top20_holder_pct": 24.0},
                    ),
                    profile_id="profile_broad_participation",
                ),
                _override_display(
                    _override_metrics(
                        _make_asset("B", "crowded_risk", jitter_index=72_002),
                        {"pct_5m": 4.5, "net_flow_5m": 1.05, "unique_traders_5m": 27.0, "top20_holder_pct": 61.0},
                    ),
                    profile_id="profile_crowded_participation",
                ),
                _override_display(
                    _override_metrics(
                        _make_asset("C", "mean_reverter", jitter_index=72_003),
                        {"pct_5m": -2.6, "net_flow_5m": 0.20, "unique_traders_5m": 10.0, "top20_holder_pct": 34.0},
                    ),
                    profile_id="profile_mean_reverter_distractor",
                ),
                _override_display(
                    _override_metrics(
                        _make_asset("D", "illiquid_spike", jitter_index=72_004),
                        {"pct_5m": 5.0, "net_flow_5m": 0.18, "unique_traders_5m": 4.0, "top20_holder_pct": 72.0},
                    ),
                    profile_id="profile_illiquid_spike_distractor",
                ),
            ],
        ),
        (
            "momentum_flow_tiebreak",
            [
                _override_display(
                    _override_metrics(
                        _make_asset("A", "momentum_burst", jitter_index=73_001),
                        {"pct_5m": 5.8, "net_flow_5m": 0.82, "unique_traders_5m": 18.0, "top20_holder_pct": 33.0},
                    ),
                    profile_id="profile_momentum_anchor",
                ),
                _override_display(
                    _override_metrics(
                        _make_asset("B", "flow_backed_continuation", jitter_index=73_002),
                        {"pct_5m": 4.8, "net_flow_5m": 1.58, "unique_traders_5m": 18.0, "top20_holder_pct": 33.5},
                    ),
                    profile_id="profile_flow_anchor",
                ),
                _override_display(
                    _override_metrics(
                        _make_asset("C", "stable_winner", jitter_index=73_003),
                        {"pct_5m": 3.2, "net_flow_5m": 0.78, "unique_traders_5m": 15.0, "top20_holder_pct": 27.0},
                    ),
                    profile_id="profile_stable_distractor",
                ),
                _override_display(
                    _override_metrics(
                        _make_asset("D", "crowded_risk", jitter_index=73_004),
                        {"pct_5m": 3.8, "net_flow_5m": 0.64, "unique_traders_5m": 16.0, "top20_holder_pct": 60.0},
                    ),
                    profile_id="profile_crowded_distractor",
                ),
            ],
        ),
    ]

    for scenario_idx, (variant, base_assets) in enumerate(base_scenarios):
        for style_idx, style in enumerate(surface_styles):
            symbols = tuple(str(symbol) for symbol in style["symbols"])
            style_name = str(style["name"])
            for perm_idx, layout in enumerate(layouts):
                order = tuple(int(idx) for idx in layout["order"])
                permuted_assets: list[SyntheticAsset] = []
                for row_index, asset_index in enumerate(order):
                    permuted_assets.append(
                        _override_display(
                            base_assets[asset_index],
                            symbol=symbols[row_index],
                        )
                    )
                example_id = f"profile_inv_{scenario_idx:02d}_{style_idx:02d}_{perm_idx:02d}"
                examples.extend(
                    _apply_context_variants(
                        example_id,
                        family="profile_invariance_control",
                        family_variant=variant,
                        base_assets=permuted_assets,
                        include_settings_variants=False,
                        surface_style=style_name,
                    )
                    )
    return examples


def generate_relation_invariance_controls(config: SyntheticMarketConfig) -> list[SyntheticMarketExample]:
    examples: list[SyntheticMarketExample] = []
    layouts = SYMBOL_PERMUTATION_LAYOUTS[: max(2, min(config.permutation_variants, len(SYMBOL_PERMUTATION_LAYOUTS)))]
    surface_styles = RELATION_INVARIANCE_SURFACE_STYLES[
        : max(2, min(config.profile_surface_variants, len(RELATION_INVARIANCE_SURFACE_STYLES)))
    ]
    scale_specs = RELATION_INVARIANCE_SCALE_FACTORS[
        : max(2, min(config.relation_scale_variants, len(RELATION_INVARIANCE_SCALE_FACTORS)))
    ]

    base_scenarios: list[tuple[str, list[SyntheticAsset]]] = [
        (
            "momentum_edge_near_tie",
            [
                _make_custom_asset(
                    "A",
                    archetype="momentum_edge_anchor",
                    pct_5m=5.4,
                    pct_1h=10.5,
                    net_flow_5m=1.20,
                    vol_5m=5.0,
                    vol_1h=20.0,
                    unique_traders_5m=18,
                    top20_holder_pct=32.0,
                    age_bucket="mid",
                    profile_id="anchor_left",
                ),
                _make_custom_asset(
                    "B",
                    archetype="flow_counterpart",
                    pct_5m=4.6,
                    pct_1h=11.0,
                    net_flow_5m=1.15,
                    vol_5m=5.0,
                    vol_1h=20.0,
                    unique_traders_5m=18,
                    top20_holder_pct=32.0,
                    age_bucket="mid",
                    profile_id="anchor_right",
                ),
            ],
        ),
        (
            "flow_edge_near_tie",
            [
                _make_custom_asset(
                    "A",
                    archetype="flow_edge_anchor",
                    pct_5m=4.4,
                    pct_1h=10.5,
                    net_flow_5m=1.20,
                    vol_5m=5.0,
                    vol_1h=20.0,
                    unique_traders_5m=18,
                    top20_holder_pct=32.0,
                    age_bucket="mid",
                    profile_id="anchor_left",
                ),
                _make_custom_asset(
                    "B",
                    archetype="momentum_counterpart",
                    pct_5m=5.4,
                    pct_1h=11.0,
                    net_flow_5m=1.10,
                    vol_5m=5.0,
                    vol_1h=20.0,
                    unique_traders_5m=18,
                    top20_holder_pct=32.0,
                    age_bucket="mid",
                    profile_id="anchor_right",
                ),
            ],
        ),
        (
            "broad_participation_edge",
            [
                _make_custom_asset(
                    "A",
                    archetype="broad_holder_anchor",
                    pct_5m=4.6,
                    pct_1h=9.8,
                    net_flow_5m=1.02,
                    vol_5m=5.0,
                    vol_1h=20.0,
                    unique_traders_5m=22,
                    top20_holder_pct=22.0,
                    age_bucket="mid",
                    profile_id="anchor_left",
                ),
                _make_custom_asset(
                    "B",
                    archetype="crowded_participation_counterpart",
                    pct_5m=4.6,
                    pct_1h=9.8,
                    net_flow_5m=1.02,
                    vol_5m=5.0,
                    vol_1h=20.0,
                    unique_traders_5m=24,
                    top20_holder_pct=36.0,
                    age_bucket="mid",
                    profile_id="anchor_right",
                ),
            ],
        ),
        (
            "concentration_penalty_edge",
            [
                _make_custom_asset(
                    "A",
                    archetype="concentration_penalty_anchor",
                    pct_5m=4.5,
                    pct_1h=9.8,
                    net_flow_5m=1.00,
                    vol_5m=5.0,
                    vol_1h=20.0,
                    unique_traders_5m=22,
                    top20_holder_pct=22.0,
                    age_bucket="mid",
                    profile_id="anchor_left",
                ),
                _make_custom_asset(
                    "B",
                    archetype="crowded_momentum_counterpart",
                    pct_5m=4.7,
                    pct_1h=9.8,
                    net_flow_5m=1.00,
                    vol_5m=5.0,
                    vol_1h=20.0,
                    unique_traders_5m=24,
                    top20_holder_pct=36.0,
                    age_bucket="mid",
                    profile_id="anchor_right",
                ),
            ],
        ),
    ]

    roster_variants: list[tuple[str, list[SyntheticAsset]]] = [
        (
            "weak_distractors",
            [
                _make_custom_asset(
                    "C",
                    archetype="weak_distractor",
                    pct_5m=2.0,
                    pct_1h=4.0,
                    net_flow_5m=0.40,
                    vol_5m=3.0,
                    vol_1h=12.0,
                    unique_traders_5m=10,
                    top20_holder_pct=38.0,
                    age_bucket="mid",
                    profile_id="context_alpha",
                ),
                _make_custom_asset(
                    "D",
                    archetype="cold_distractor",
                    pct_5m=1.0,
                    pct_1h=2.0,
                    net_flow_5m=0.20,
                    vol_5m=2.2,
                    vol_1h=8.0,
                    unique_traders_5m=8,
                    top20_holder_pct=42.0,
                    age_bucket="mid",
                    profile_id="context_beta",
                ),
            ],
        ),
        (
            "lead_distractor",
            [
                _make_custom_asset(
                    "C",
                    archetype="lead_distractor",
                    pct_5m=6.5,
                    pct_1h=12.0,
                    net_flow_5m=1.40,
                    vol_5m=5.6,
                    vol_1h=22.0,
                    unique_traders_5m=22,
                    top20_holder_pct=28.0,
                    age_bucket="mid",
                    profile_id="context_alpha",
                ),
                _make_custom_asset(
                    "D",
                    archetype="cold_distractor",
                    pct_5m=1.2,
                    pct_1h=2.5,
                    net_flow_5m=0.20,
                    vol_5m=2.3,
                    vol_1h=8.5,
                    unique_traders_5m=8,
                    top20_holder_pct=42.0,
                    age_bucket="mid",
                    profile_id="context_beta",
                ),
            ],
        ),
        (
            "double_lead_distractors",
            [
                _make_custom_asset(
                    "C",
                    archetype="lead_distractor",
                    pct_5m=6.5,
                    pct_1h=12.0,
                    net_flow_5m=1.40,
                    vol_5m=5.6,
                    vol_1h=22.0,
                    unique_traders_5m=22,
                    top20_holder_pct=28.0,
                    age_bucket="mid",
                    profile_id="context_alpha",
                ),
                _make_custom_asset(
                    "D",
                    archetype="secondary_lead_distractor",
                    pct_5m=5.9,
                    pct_1h=11.5,
                    net_flow_5m=1.30,
                    vol_5m=5.3,
                    vol_1h=21.0,
                    unique_traders_5m=20,
                    top20_holder_pct=30.0,
                    age_bucket="mid",
                    profile_id="context_beta",
                ),
            ],
        ),
        (
            "interleaved_distractor",
            [
                _make_custom_asset(
                    "C",
                    archetype="interleaved_distractor",
                    pct_5m=5.0,
                    pct_1h=10.0,
                    net_flow_5m=1.18,
                    vol_5m=5.0,
                    vol_1h=20.0,
                    unique_traders_5m=18,
                    top20_holder_pct=31.0,
                    age_bucket="mid",
                    profile_id="context_alpha",
                ),
                _make_custom_asset(
                    "D",
                    archetype="cold_distractor",
                    pct_5m=1.2,
                    pct_1h=2.5,
                    net_flow_5m=0.20,
                    vol_5m=2.3,
                    vol_1h=8.5,
                    unique_traders_5m=8,
                    top20_holder_pct=42.0,
                    age_bucket="mid",
                    profile_id="context_beta",
                ),
            ],
        ),
    ][: max(2, min(config.relation_roster_variants, 4))]

    for scenario_idx, (variant, anchors) in enumerate(base_scenarios):
        for style_idx, style in enumerate(surface_styles):
            symbols = tuple(str(symbol) for symbol in style["symbols"])
            style_name = str(style["name"])
            for perm_idx, layout in enumerate(layouts):
                order = tuple(int(idx) for idx in layout["order"])
                for roster_idx, (_, roster_assets) in enumerate(roster_variants):
                    for scale_idx, (_, scale_factor) in enumerate(scale_specs):
                        scaled_assets = [
                            _scale_market_magnitude(asset, scale_factor)
                            for asset in [*anchors, *roster_assets]
                        ]
                        permuted_assets: list[SyntheticAsset] = []
                        for row_index, asset_index in enumerate(order):
                            permuted_assets.append(
                                _override_display(
                                    scaled_assets[asset_index],
                                    symbol=symbols[row_index],
                                )
                            )
                        example_id = (
                            f"relation_inv_{scenario_idx:02d}_{style_idx:02d}_{perm_idx:02d}_"
                            f"{roster_idx:02d}_{scale_idx:02d}"
                        )
                        examples.extend(
                            _apply_context_variants(
                                example_id,
                                family="relation_invariance_control",
                                family_variant=variant,
                                base_assets=permuted_assets,
                                include_settings_variants=False,
                                surface_style=style_name,
                            )
                        )
    return examples


def generate_contextual_relation_controls(config: SyntheticMarketConfig) -> list[SyntheticMarketExample]:
    examples: list[SyntheticMarketExample] = []
    layouts = SYMBOL_PERMUTATION_LAYOUTS[: max(2, min(config.permutation_variants, len(SYMBOL_PERMUTATION_LAYOUTS)))]
    surface_styles = RELATION_INVARIANCE_SURFACE_STYLES[
        : max(2, min(config.profile_surface_variants, len(RELATION_INVARIANCE_SURFACE_STYLES)))
    ]
    scale_specs = RELATION_INVARIANCE_SCALE_FACTORS[
        : max(2, min(config.relation_scale_variants, len(RELATION_INVARIANCE_SCALE_FACTORS)))
    ]

    anchor_left = _make_custom_asset(
        "A",
        archetype="contextual_anchor_left",
        pct_5m=5.08,
        pct_1h=10.10,
        net_flow_5m=1.06,
        vol_5m=5.10,
        vol_1h=20.40,
        unique_traders_5m=18,
        top20_holder_pct=30.0,
        age_bucket="mid",
        profile_id="anchor_left",
    )
    anchor_right = _make_custom_asset(
        "B",
        archetype="contextual_anchor_right",
        pct_5m=4.98,
        pct_1h=10.00,
        net_flow_5m=1.05,
        vol_5m=5.00,
        vol_1h=20.00,
        unique_traders_5m=18,
        top20_holder_pct=30.0,
        age_bucket="mid",
        profile_id="anchor_right",
    )

    scenario_specs: list[tuple[str, list[list[SyntheticAsset]]]] = [
        (
            "generic_duel_context",
            [
                [
                    _make_custom_asset(
                        "C",
                        archetype="generic_weak_alpha",
                        pct_5m=2.8,
                        pct_1h=5.4,
                        net_flow_5m=0.44,
                        vol_5m=3.2,
                        vol_1h=12.0,
                        unique_traders_5m=12,
                        top20_holder_pct=34.0,
                        age_bucket="mid",
                        profile_id="context_alpha",
                    ),
                    _make_custom_asset(
                        "D",
                        archetype="generic_weak_beta",
                        pct_5m=1.3,
                        pct_1h=2.7,
                        net_flow_5m=0.18,
                        vol_5m=2.3,
                        vol_1h=8.3,
                        unique_traders_5m=8,
                        top20_holder_pct=40.0,
                        age_bucket="mid",
                        profile_id="context_beta",
                    ),
                ],
                [
                    _make_custom_asset(
                        "C",
                        archetype="generic_lead_alpha",
                        pct_5m=5.42,
                        pct_1h=10.55,
                        net_flow_5m=1.12,
                        vol_5m=5.10,
                        vol_1h=20.30,
                        unique_traders_5m=18,
                        top20_holder_pct=29.0,
                        age_bucket="mid",
                        profile_id="context_alpha",
                    ),
                    _make_custom_asset(
                        "D",
                        archetype="generic_weak_beta",
                        pct_5m=1.3,
                        pct_1h=2.7,
                        net_flow_5m=0.18,
                        vol_5m=2.3,
                        vol_1h=8.3,
                        unique_traders_5m=8,
                        top20_holder_pct=40.0,
                        age_bucket="mid",
                        profile_id="context_beta",
                    ),
                ],
                [
                    _make_custom_asset(
                        "C",
                        archetype="generic_lead_alpha",
                        pct_5m=5.50,
                        pct_1h=10.70,
                        net_flow_5m=1.14,
                        vol_5m=5.20,
                        vol_1h=20.60,
                        unique_traders_5m=18,
                        top20_holder_pct=29.0,
                        age_bucket="mid",
                        profile_id="context_alpha",
                    ),
                    _make_custom_asset(
                        "D",
                        archetype="generic_lead_beta",
                        pct_5m=5.18,
                        pct_1h=10.28,
                        net_flow_5m=1.09,
                        vol_5m=5.00,
                        vol_1h=20.10,
                        unique_traders_5m=18,
                        top20_holder_pct=29.0,
                        age_bucket="mid",
                        profile_id="context_beta",
                    ),
                ],
                [
                    _make_custom_asset(
                        "C",
                        archetype="generic_interleave_alpha",
                        pct_5m=5.03,
                        pct_1h=10.04,
                        net_flow_5m=1.055,
                        vol_5m=5.02,
                        vol_1h=20.05,
                        unique_traders_5m=18,
                        top20_holder_pct=30.0,
                        age_bucket="mid",
                        profile_id="context_alpha",
                    ),
                    _make_custom_asset(
                        "D",
                        archetype="generic_weak_beta",
                        pct_5m=1.3,
                        pct_1h=2.7,
                        net_flow_5m=0.18,
                        vol_5m=2.3,
                        vol_1h=8.3,
                        unique_traders_5m=8,
                        top20_holder_pct=40.0,
                        age_bucket="mid",
                        profile_id="context_beta",
                    ),
                ],
            ],
        ),
        (
            "momentum_shadow_context",
            [
                [
                    _make_custom_asset(
                        "C",
                        archetype="momentum_shadow_weak",
                        pct_5m=4.88,
                        pct_1h=9.72,
                        net_flow_5m=0.98,
                        vol_5m=4.90,
                        vol_1h=19.10,
                        unique_traders_5m=17,
                        top20_holder_pct=30.0,
                        age_bucket="mid",
                        profile_id="context_alpha",
                    ),
                    _make_custom_asset(
                        "D",
                        archetype="generic_weak_beta",
                        pct_5m=1.3,
                        pct_1h=2.7,
                        net_flow_5m=0.18,
                        vol_5m=2.3,
                        vol_1h=8.3,
                        unique_traders_5m=8,
                        top20_holder_pct=40.0,
                        age_bucket="mid",
                        profile_id="context_beta",
                    ),
                ],
                [
                    _make_custom_asset(
                        "C",
                        archetype="momentum_shadow_lead",
                        pct_5m=5.46,
                        pct_1h=10.62,
                        net_flow_5m=1.11,
                        vol_5m=5.10,
                        vol_1h=20.40,
                        unique_traders_5m=18,
                        top20_holder_pct=30.0,
                        age_bucket="mid",
                        profile_id="context_alpha",
                    ),
                    _make_custom_asset(
                        "D",
                        archetype="generic_weak_beta",
                        pct_5m=1.3,
                        pct_1h=2.7,
                        net_flow_5m=0.18,
                        vol_5m=2.3,
                        vol_1h=8.3,
                        unique_traders_5m=8,
                        top20_holder_pct=40.0,
                        age_bucket="mid",
                        profile_id="context_beta",
                    ),
                ],
                [
                    _make_custom_asset(
                        "C",
                        archetype="momentum_shadow_lead",
                        pct_5m=5.54,
                        pct_1h=10.80,
                        net_flow_5m=1.14,
                        vol_5m=5.20,
                        vol_1h=20.60,
                        unique_traders_5m=18,
                        top20_holder_pct=30.0,
                        age_bucket="mid",
                        profile_id="context_alpha",
                    ),
                    _make_custom_asset(
                        "D",
                        archetype="momentum_shadow_secondary",
                        pct_5m=5.18,
                        pct_1h=10.36,
                        net_flow_5m=1.08,
                        vol_5m=5.04,
                        vol_1h=20.10,
                        unique_traders_5m=18,
                        top20_holder_pct=31.0,
                        age_bucket="mid",
                        profile_id="context_beta",
                    ),
                ],
                [
                    _make_custom_asset(
                        "C",
                        archetype="momentum_shadow_interleave",
                        pct_5m=5.04,
                        pct_1h=10.08,
                        net_flow_5m=1.058,
                        vol_5m=5.05,
                        vol_1h=20.10,
                        unique_traders_5m=18,
                        top20_holder_pct=30.0,
                        age_bucket="mid",
                        profile_id="context_alpha",
                    ),
                    _make_custom_asset(
                        "D",
                        archetype="generic_weak_beta",
                        pct_5m=1.3,
                        pct_1h=2.7,
                        net_flow_5m=0.18,
                        vol_5m=2.3,
                        vol_1h=8.3,
                        unique_traders_5m=8,
                        top20_holder_pct=40.0,
                        age_bucket="mid",
                        profile_id="context_beta",
                    ),
                ],
            ],
        ),
        (
            "flow_shadow_context",
            [
                [
                    _make_custom_asset(
                        "C",
                        archetype="flow_shadow_weak",
                        pct_5m=4.78,
                        pct_1h=9.68,
                        net_flow_5m=0.99,
                        vol_5m=4.92,
                        vol_1h=19.10,
                        unique_traders_5m=17,
                        top20_holder_pct=30.0,
                        age_bucket="mid",
                        profile_id="context_alpha",
                    ),
                    _make_custom_asset(
                        "D",
                        archetype="generic_weak_beta",
                        pct_5m=1.3,
                        pct_1h=2.7,
                        net_flow_5m=0.18,
                        vol_5m=2.3,
                        vol_1h=8.3,
                        unique_traders_5m=8,
                        top20_holder_pct=40.0,
                        age_bucket="mid",
                        profile_id="context_beta",
                    ),
                ],
                [
                    _make_custom_asset(
                        "C",
                        archetype="flow_shadow_lead",
                        pct_5m=5.34,
                        pct_1h=10.34,
                        net_flow_5m=1.125,
                        vol_5m=5.04,
                        vol_1h=20.20,
                        unique_traders_5m=18,
                        top20_holder_pct=30.0,
                        age_bucket="mid",
                        profile_id="context_alpha",
                    ),
                    _make_custom_asset(
                        "D",
                        archetype="generic_weak_beta",
                        pct_5m=1.3,
                        pct_1h=2.7,
                        net_flow_5m=0.18,
                        vol_5m=2.3,
                        vol_1h=8.3,
                        unique_traders_5m=8,
                        top20_holder_pct=40.0,
                        age_bucket="mid",
                        profile_id="context_beta",
                    ),
                ],
                [
                    _make_custom_asset(
                        "C",
                        archetype="flow_shadow_lead",
                        pct_5m=5.40,
                        pct_1h=10.42,
                        net_flow_5m=1.135,
                        vol_5m=5.08,
                        vol_1h=20.30,
                        unique_traders_5m=18,
                        top20_holder_pct=30.0,
                        age_bucket="mid",
                        profile_id="context_alpha",
                    ),
                    _make_custom_asset(
                        "D",
                        archetype="flow_shadow_secondary",
                        pct_5m=5.06,
                        pct_1h=10.08,
                        net_flow_5m=1.085,
                        vol_5m=5.02,
                        vol_1h=20.10,
                        unique_traders_5m=18,
                        top20_holder_pct=31.0,
                        age_bucket="mid",
                        profile_id="context_beta",
                    ),
                ],
                [
                    _make_custom_asset(
                        "C",
                        archetype="flow_shadow_interleave",
                        pct_5m=4.99,
                        pct_1h=10.02,
                        net_flow_5m=1.046,
                        vol_5m=5.00,
                        vol_1h=20.02,
                        unique_traders_5m=18,
                        top20_holder_pct=30.0,
                        age_bucket="mid",
                        profile_id="context_alpha",
                    ),
                    _make_custom_asset(
                        "D",
                        archetype="generic_weak_beta",
                        pct_5m=1.3,
                        pct_1h=2.7,
                        net_flow_5m=0.18,
                        vol_5m=2.3,
                        vol_1h=8.3,
                        unique_traders_5m=8,
                        top20_holder_pct=40.0,
                        age_bucket="mid",
                        profile_id="context_beta",
                    ),
                ],
            ],
        ),
        (
            "paired_cluster_context",
            [
                [
                    _make_custom_asset(
                        "C",
                        archetype="cluster_weak_alpha",
                        pct_5m=4.78,
                        pct_1h=9.82,
                        net_flow_5m=1.00,
                        vol_5m=4.95,
                        vol_1h=19.20,
                        unique_traders_5m=18,
                        top20_holder_pct=30.0,
                        age_bucket="mid",
                        profile_id="context_alpha",
                    ),
                    _make_custom_asset(
                        "D",
                        archetype="cluster_weak_beta",
                        pct_5m=4.72,
                        pct_1h=9.72,
                        net_flow_5m=0.99,
                        vol_5m=4.90,
                        vol_1h=19.00,
                        unique_traders_5m=18,
                        top20_holder_pct=31.0,
                        age_bucket="mid",
                        profile_id="context_beta",
                    ),
                ],
                [
                    _make_custom_asset(
                        "C",
                        archetype="cluster_lead_alpha",
                        pct_5m=5.38,
                        pct_1h=10.36,
                        net_flow_5m=1.10,
                        vol_5m=5.06,
                        vol_1h=20.18,
                        unique_traders_5m=18,
                        top20_holder_pct=30.0,
                        age_bucket="mid",
                        profile_id="context_alpha",
                    ),
                    _make_custom_asset(
                        "D",
                        archetype="cluster_weak_beta",
                        pct_5m=4.72,
                        pct_1h=9.72,
                        net_flow_5m=0.99,
                        vol_5m=4.90,
                        vol_1h=19.00,
                        unique_traders_5m=18,
                        top20_holder_pct=31.0,
                        age_bucket="mid",
                        profile_id="context_beta",
                    ),
                ],
                [
                    _make_custom_asset(
                        "C",
                        archetype="cluster_lead_alpha",
                        pct_5m=5.46,
                        pct_1h=10.52,
                        net_flow_5m=1.12,
                        vol_5m=5.12,
                        vol_1h=20.30,
                        unique_traders_5m=18,
                        top20_holder_pct=30.0,
                        age_bucket="mid",
                        profile_id="context_alpha",
                    ),
                    _make_custom_asset(
                        "D",
                        archetype="cluster_lead_beta",
                        pct_5m=5.18,
                        pct_1h=10.24,
                        net_flow_5m=1.08,
                        vol_5m=5.02,
                        vol_1h=20.06,
                        unique_traders_5m=18,
                        top20_holder_pct=31.0,
                        age_bucket="mid",
                        profile_id="context_beta",
                    ),
                ],
                [
                    _make_custom_asset(
                        "C",
                        archetype="cluster_interleave_alpha",
                        pct_5m=5.00,
                        pct_1h=10.02,
                        net_flow_5m=1.051,
                        vol_5m=5.00,
                        vol_1h=20.02,
                        unique_traders_5m=18,
                        top20_holder_pct=30.0,
                        age_bucket="mid",
                        profile_id="context_alpha",
                    ),
                    _make_custom_asset(
                        "D",
                        archetype="cluster_weak_beta",
                        pct_5m=4.72,
                        pct_1h=9.72,
                        net_flow_5m=0.99,
                        vol_5m=4.90,
                        vol_1h=19.00,
                        unique_traders_5m=18,
                        top20_holder_pct=31.0,
                        age_bucket="mid",
                        profile_id="context_beta",
                    ),
                ],
            ],
        ),
    ]

    scenario_specs = [
        (variant, rosters[: max(2, min(config.relation_roster_variants, len(rosters)))])
        for variant, rosters in scenario_specs
    ]

    for scenario_idx, (variant, roster_variants) in enumerate(scenario_specs):
        for style_idx, style in enumerate(surface_styles):
            symbols = tuple(str(symbol) for symbol in style["symbols"])
            style_name = str(style["name"])
            for perm_idx, layout in enumerate(layouts):
                order = tuple(int(idx) for idx in layout["order"])
                for roster_idx, roster_assets in enumerate(roster_variants):
                    for scale_idx, (_, scale_factor) in enumerate(scale_specs):
                        scaled_assets = [
                            _scale_market_magnitude(asset, scale_factor)
                            for asset in [anchor_left, anchor_right, *roster_assets]
                        ]
                        permuted_assets: list[SyntheticAsset] = []
                        for row_index, asset_index in enumerate(order):
                            permuted_assets.append(
                                _override_display(
                                    scaled_assets[asset_index],
                                    symbol=symbols[row_index],
                                )
                            )
                        example_id = (
                            f"relation_inv_{scenario_idx:02d}_{style_idx:02d}_{perm_idx:02d}_"
                            f"{roster_idx:02d}_{scale_idx:02d}"
                        )
                        examples.extend(
                            _apply_context_variants(
                                example_id,
                                family="relation_invariance_control",
                                family_variant=variant,
                                base_assets=permuted_assets,
                                include_settings_variants=False,
                                surface_style=style_name,
                            )
                        )
    return examples


def generate_archetype_families(config: SyntheticMarketConfig) -> list[SyntheticMarketExample]:
    examples: list[SyntheticMarketExample] = []
    distractors = ["stable_winner", "flow_backed_continuation", "crowded_risk"]
    for archetype_idx, archetype in enumerate(ARCHETYPES):
        for variant_idx in range(config.archetype_variants):
            symbols = ["A", "B", "C", "D"]
            assets = [_make_asset(symbols[0], archetype, jitter_index=variant_idx + archetype_idx)]
            for offset, distractor in enumerate(distractors, start=1):
                assets.append(_make_asset(symbols[offset], distractor, jitter_index=variant_idx + 10 * offset + archetype_idx))
            example_id = f"archetype_{archetype}_{variant_idx:02d}"
            examples.extend(
                _apply_context_variants(
                    example_id,
                    family="archetype_family",
                    family_variant=archetype,
                    base_assets=assets,
                    include_settings_variants=config.include_settings_variants,
                )
            )
    return examples


def generate_dataset(config: SyntheticMarketConfig) -> list[SyntheticMarketExample]:
    if config.dataset_preset == "phase2_geometry":
        examples = []
        examples.extend(generate_dense_scalar_sweeps(config))
        examples.extend(generate_minimal_scalar_sweeps(config))
        return examples
    if config.dataset_preset == "phase3_coupled_geometry":
        examples = []
        examples.extend(generate_coupled_factor_dense_grids(config))
        examples.extend(generate_coupled_factor_minimal_grids(config))
        return examples
    if config.dataset_preset == "phase4_market_representation":
        examples = []
        examples.extend(generate_hard_pairwise_tradeoff_grids(config))
        examples.extend(generate_rank_context_tradeoffs(config))
        return examples
    if config.dataset_preset == "phase5_symbol_permutation":
        return generate_symbol_permutation_controls(config)
    if config.dataset_preset == "phase6_profile_invariance":
        return generate_profile_invariance_controls(config)
    if config.dataset_preset == "phase7_relation_invariance":
        return generate_relation_invariance_controls(config)
    if config.dataset_preset == "phase8_contextual_relation":
        return generate_contextual_relation_controls(config)

    examples = []
    examples.extend(generate_scalar_sweeps(config))
    examples.extend(generate_pairwise_tradeoff_grids(config))
    examples.extend(generate_archetype_families(config))
    return examples


def _default_log_id_base(dataset_preset: str) -> int:
    return {
        "phase1": 2_000_000_000,
        "phase2_geometry": 2_100_000_000,
        "phase3_coupled_geometry": 2_130_000_000,
        "phase4_market_representation": 2_147_300_000,
        "phase5_symbol_permutation": 2_146_900_000,
        "phase6_profile_invariance": 2_146_950_000,
        "phase7_relation_invariance": 2_147_000_000,
        "phase8_contextual_relation": 2_147_050_000,
    }.get(dataset_preset, 2_140_000_000)


def _assign_log_ids(
    examples: list[SyntheticMarketExample],
    *,
    dataset_preset: str,
    base_log_id: int | None = None,
) -> list[SyntheticMarketExample]:
    base_log_id = int(base_log_id if base_log_id is not None else _default_log_id_base(dataset_preset))
    ordered = sorted(
        examples,
        key=lambda ex: (
            ex.family,
            ex.family_variant,
            ex.example_id,
            ex.context_variant,
        ),
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


def write_dataset(examples: list[SyntheticMarketExample], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "synthetic_market_prompts.jsonl"
    tick_path = output_dir / "synthetic_market_tick_records.parquet"
    asset_path = output_dir / "synthetic_market_asset_records.parquet"
    pairwise_path = output_dir / "synthetic_market_pairwise_records.parquet"
    summary_path = output_dir / "synthetic_market_summary.json"

    tick_rows: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []

    with jsonl_path.open("w", encoding="utf-8") as f:
        for example in examples:
            prompt_messages_json = json.dumps(list(example.prompt_messages))
            payload = {
                "log_id": example.log_id,
                "example_id": example.example_id,
                "family": example.family,
                "family_variant": example.family_variant,
                "context_variant": example.context_variant,
                "system_prompt": example.system_prompt,
                "user_prompt": example.user_prompt,
                "prompt_messages_json": json.loads(prompt_messages_json),
                "labels": {
                    k: v for k, v in example.labels.items()
                    if k not in {"asset_rows", "pairwise_rows"}
                },
            }
            f.write(json.dumps(payload) + "\n")
            tick_rows.append({
                "log_id": example.log_id,
                "example_id": example.example_id,
                "family": example.family,
                "family_variant": example.family_variant,
                "context_variant": example.context_variant,
                "system_prompt": example.system_prompt,
                "user_prompt": example.user_prompt,
                "prompt_messages_json": prompt_messages_json,
                "labels_json": json.dumps({
                    k: v for k, v in example.labels.items()
                    if k not in {"asset_rows", "pairwise_rows"}
                }),
                "best_asset": example.labels["best_asset"],
                "buy_any": example.labels["buy_any"],
                "observe_vs_act": example.labels["observe_vs_act"],
                "num_assets": len(example.assets),
            })
            for asset_row in example.labels["asset_rows"]:
                asset_rows.append({"log_id": example.log_id, **asset_row})
            for pairwise_row in example.labels["pairwise_rows"]:
                pairwise_rows.append({"log_id": example.log_id, **pairwise_row})

    pq.write_table(pa.Table.from_pylist(tick_rows), tick_path)
    pq.write_table(pa.Table.from_pylist(asset_rows), asset_path)
    pq.write_table(pa.Table.from_pylist(pairwise_rows), pairwise_path)

    summary = {
        "n_examples": len(examples),
        "n_tick_rows": len(tick_rows),
        "n_asset_rows": len(asset_rows),
        "n_pairwise_rows": len(pairwise_rows),
        "families": sorted({example.family for example in examples}),
        "context_variants": sorted({example.context_variant for example in examples}),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return {
        "jsonl_path": str(jsonl_path),
        "tick_path": str(tick_path),
        "asset_path": str(asset_path),
        "pairwise_path": str(pairwise_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def build_synthetic_market_dataset(config: SyntheticMarketConfig) -> dict[str, Any]:
    examples = _assign_log_ids(
        generate_dataset(config),
        dataset_preset=config.dataset_preset,
        base_log_id=config.log_id_base,
    )
    return write_dataset(examples, config.output_dir)


def main(argv: list[str] | None = None) -> None:
    _ = argv
    result = build_synthetic_market_dataset(SyntheticMarketConfig())
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
