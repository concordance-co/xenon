from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = Path(
    "data/analysis_results/synthetic_market_context_order/phase16_context_order_v1/results.json"
)
PROMPTS_PATH = Path(
    "data/interp_exports/synthetic_market_phase16_context_order/synthetic_market_prompts.jsonl"
)
ASSET_DIR = Path("data/report_assets/synthetic_market_phase16_context_order")
RAW_PROMPT_DIR = Path("docs/reports/raw_prompts/phase16_context_order")
EXAMPLE_ID = "context_coupled_pct_5m__net_flow_5m_r00_x00_y00"


def _load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text())


def _load_prompts() -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    with PROMPTS_PATH.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[(row["example_id"], row["context_variant"])] = row
    return rows


def _sorted_layers(state_payload: dict[str, dict]) -> list[int]:
    return sorted(int(layer) for layer in state_payload.keys())


def _series(state_payload: dict[str, dict], key: str) -> tuple[list[int], list[float]]:
    layers = _sorted_layers(state_payload)
    return layers, [float(state_payload[str(layer)][key]) for layer in layers]


def _best_gap_layer(state_payload: dict[str, dict]) -> tuple[int, dict]:
    layer, row = max(state_payload.items(), key=lambda kv: kv[1]["perception_gap"])
    return int(layer), row


def _write_raw_prompts(prompt_rows: dict[tuple[str, str], dict]) -> None:
    RAW_PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    variant_order = [
        "market_only",
        "risk_5_after_market",
        "risk_5_before_market",
        "affordance_5_after_market",
        "affordance_5_before_market",
    ]
    system_prompt = prompt_rows[(EXAMPLE_ID, "market_only")]["system_prompt"]
    (RAW_PROMPT_DIR / "phase16_system_prompt.txt").write_text(system_prompt)
    for variant in variant_order:
        user_prompt = prompt_rows[(EXAMPLE_ID, variant)]["user_prompt"]
        (RAW_PROMPT_DIR / f"phase16_{variant}_user_prompt.txt").write_text(user_prompt)


def build_assets() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    results = _load_results()
    prompt_rows = _load_prompts()
    _write_raw_prompts(prompt_rows)

    plt.style.use("seaborn-v0_8-whitegrid")

    groups = results["groups"]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    perception_specs = [
        ("risk", "market_mean", "Risk: market_mean"),
        ("risk", "market_eos", "Risk: corrected market_eos"),
        ("affordance", "market_mean", "Affordance: market_mean"),
        ("affordance", "market_eos", "Affordance: corrected market_eos"),
    ]
    colors = {"ab": "#c87533", "ac": "#b33a2a"}
    for ax, (group_name, state_key, title) in zip(axes.flat, perception_specs, strict=True):
        payload = groups[group_name]["state_results"][state_key]
        layers, ab_cos = _series(payload, "ab_cosine_mean")
        _, ac_cos = _series(payload, "ac_cosine_mean")
        ax.plot(layers, ab_cos, color=colors["ab"], lw=2.3, label="A vs B (market first, later context)")
        ax.plot(layers, ac_cos, color=colors["ac"], lw=2.3, label="A vs C (context before market)")
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean cosine")
        ax.set_ylim(0.92, 1.001)
        if state_key == "market_eos":
            best_layer, best_row = _best_gap_layer(payload)
            ax.scatter([best_layer], [best_row["ac_cosine_mean"]], color="#222", s=28, zorder=4)
            ax.annotate(
                f"L{best_layer}\nGap {best_row['perception_gap']:.3f}",
                (best_layer, best_row["ac_cosine_mean"]),
                textcoords="offset points",
                xytext=(6, -22),
                fontsize=8,
            )
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(ASSET_DIR / "phase16_perception_curves.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    integration_states = [
        ("last_token", "Last token", "#b33a2a"),
        ("active_settings_eos", "Settings EOS", "#c87533"),
        ("portfolio_eos", "Portfolio EOS", "#4f6d7a"),
        ("constraints_eos", "Constraints EOS", "#7a8b5b"),
    ]
    for ax, group_name in zip(axes, ["risk", "affordance"], strict=True):
        for state_key, label, color in integration_states:
            payload = groups[group_name]["integration_results"][state_key]
            layers, bc_cos = _series(payload, "bc_cosine_mean")
            ax.plot(layers, bc_cos, lw=2.0, color=color, label=label)
        ax.set_title(f"{group_name.title()}: B vs C downstream convergence")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean cosine")
        ax.set_ylim(0.92, 1.001)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.savefig(ASSET_DIR / "phase16_integration_curves.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for ax, group_name in zip(axes, ["risk", "affordance"], strict=True):
        payload = groups[group_name]["state_results"]["market_eos"]
        best_layer, best_row = _best_gap_layer(payload)
        features = [row["feature"] or f"PC {idx + 1}" for idx, row in enumerate(best_row["pc_shift"])]
        ab_vals = [row["ab_abs_mean"] for row in best_row["pc_shift"]]
        ac_vals = [row["ac_abs_mean"] for row in best_row["pc_shift"]]
        xpos = np.arange(len(features))
        width = 0.34
        ax.bar(xpos - width / 2, ab_vals, width, color="#c87533", label="A -> B")
        ax.bar(xpos + width / 2, ac_vals, width, color="#b33a2a", label="A -> C")
        ax.set_xticks(xpos, [feature.replace("_", "\n") for feature in features])
        ax.set_ylabel("Mean abs shift in Phase 15 PC basis")
        ax.set_title(f"{group_name.title()}: corrected market_eos basis shift at L{best_layer}")
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(ASSET_DIR / "phase16_basis_shift.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    build_assets()
