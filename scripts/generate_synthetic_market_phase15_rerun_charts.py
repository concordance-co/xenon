from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASELINE_RESULTS = Path(
    "data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/results.json"
)
RERUN_RESULTS = Path(
    "data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/results.json"
)
ASSET_DIR = Path("data/report_assets/synthetic_market_phase15_discovery_rerun")


def _layer_best_margin_series(state_payload: dict) -> dict[str, list[float]]:
    layers = sorted(int(layer) for layer in state_payload.keys())
    best_margin = []
    best_market = []
    best_nuisance = []
    max_nuisance = []
    for layer in layers:
        entry = state_payload[str(layer)]
        best_row = None
        layer_max_nuisance = 0.0
        for pc in entry["pcs"]:
            market = pc["top_market_correlations"][0]["abs_spearman"] if pc["top_market_correlations"] else 0.0
            nuisance = (
                pc["top_nuisance_correlations"][0]["abs_spearman"] if pc["top_nuisance_correlations"] else 0.0
            )
            layer_max_nuisance = max(layer_max_nuisance, nuisance)
            row = {
                "margin": market - nuisance,
                "market": market,
                "nuisance": nuisance,
            }
            if best_row is None or row["margin"] > best_row["margin"]:
                best_row = row
        best_margin.append(float(best_row["margin"]))
        best_market.append(float(best_row["market"]))
        best_nuisance.append(float(best_row["nuisance"]))
        max_nuisance.append(float(layer_max_nuisance))
    return {
        "layers": layers,
        "best_margin": best_margin,
        "best_market": best_market,
        "best_nuisance": best_nuisance,
        "max_nuisance": max_nuisance,
    }


def _best_feature_counts(state_payload: dict) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for layer in state_payload.values():
        best_feature = ""
        best_margin = None
        for pc in layer["pcs"]:
            market = pc["top_market_correlations"][0]["abs_spearman"] if pc["top_market_correlations"] else 0.0
            nuisance = pc["top_nuisance_correlations"][0]["abs_spearman"] if pc["top_nuisance_correlations"] else 0.0
            margin = market - nuisance
            feature = pc["top_market_correlations"][0]["feature"] if pc["top_market_correlations"] else ""
            if best_margin is None or margin > best_margin:
                best_margin = margin
                best_feature = feature
        counts[best_feature] += 1
    return counts.most_common()


def build_charts() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(BASELINE_RESULTS.read_text())
    rerun = json.loads(RERUN_RESULTS.read_text())

    base_mm = _layer_best_margin_series(baseline["states"]["market_mean"])
    base_me = _layer_best_margin_series(baseline["states"]["market_eos"])
    rerun_mm = _layer_best_margin_series(rerun["states"]["market_mean"])
    rerun_me = _layer_best_margin_series(rerun["states"]["market_eos"])

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(base_mm["layers"], base_mm["best_margin"], color="#c87533", lw=2.0, label="baseline")
    ax.plot(rerun_mm["layers"], rerun_mm["best_margin"], color="#b33a2a", lw=2.2, label="residualized")
    ax.axhline(0.0, color="#666", lw=1.0, ls="--")
    ax.set_title("market_mean: best market-over-nuisance margin")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Margin")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(base_me["layers"], base_me["best_margin"], color="#89a0b0", lw=2.0, label="baseline")
    ax.plot(rerun_me["layers"], rerun_me["best_margin"], color="#4f6d7a", lw=2.2, label="residualized")
    ax.axhline(0.0, color="#666", lw=1.0, ls="--")
    ax.set_title("market_eos: best market-over-nuisance margin")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Margin")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(base_mm["layers"], base_mm["max_nuisance"], color="#c87533", lw=2.0, label="baseline")
    ax.plot(rerun_mm["layers"], rerun_mm["max_nuisance"], color="#b33a2a", lw=2.2, label="residualized")
    ax.axhline(0.15, color="#666", lw=1.0, ls="--", label="0.15 sanity line")
    ax.set_title("market_mean: worst nuisance correlation among top PCs")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max |Spearman|")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(base_me["layers"], base_me["max_nuisance"], color="#89a0b0", lw=2.0, label="baseline")
    ax.plot(rerun_me["layers"], rerun_me["max_nuisance"], color="#4f6d7a", lw=2.2, label="residualized")
    ax.axhline(0.15, color="#666", lw=1.0, ls="--", label="0.15 sanity line")
    ax.set_title("market_eos: worst nuisance correlation among top PCs")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Max |Spearman|")
    ax.legend(frameon=False)

    fig.savefig(ASSET_DIR / "phase15_rerun_compare.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    mm_counts = _best_feature_counts(rerun["states"]["market_mean"])[:4]
    me_counts = _best_feature_counts(rerun["states"]["market_eos"])[:4]
    for ax, title, counts, color in [
        (axes[0], "market_mean best feature by layer", mm_counts, "#b33a2a"),
        (axes[1], "market_eos best feature by layer", me_counts, "#4f6d7a"),
    ]:
        labels = [feature.replace("_", "\n") for feature, _ in counts]
        values = [count for _, count in counts]
        ax.bar(labels, values, color=color)
        ax.set_title(title)
        ax.set_ylabel("Layers")
        ax.tick_params(axis="x", labelsize=8)

    ax = axes[2]
    labels = ["mean\nbaseline", "mean\nrerun", "eos\nbaseline", "eos\nrerun"]
    values = [176, 1, 177, 21]
    colors = ["#c87533", "#b33a2a", "#89a0b0", "#4f6d7a"]
    ax.bar(labels, values, color=colors)
    ax.set_title("PCs with nuisance correlation > 0.15")
    ax.set_ylabel("Count across 48 layers × 5 PCs")

    fig.savefig(ASSET_DIR / "phase15_rerun_feature_shift.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    build_charts()
