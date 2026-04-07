from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = Path(
    "data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/results.json"
)
ASSET_DIR = Path("data/report_assets/synthetic_market_phase15_discovery")


def _state_series(state_payload: dict) -> dict[str, list[float]]:
    layers = sorted(int(layer) for layer in state_payload.keys())
    best_margin = []
    best_market = []
    best_nuisance = []
    best_feature = []
    pc1_var = []
    pc2_var = []
    pc3_var = []
    for layer in layers:
        entry = state_payload[str(layer)]
        best_row = None
        for pc in entry["pcs"]:
            market = pc["top_market_correlations"][0]["abs_spearman"] if pc["top_market_correlations"] else 0.0
            nuisance = (
                pc["top_nuisance_correlations"][0]["abs_spearman"] if pc["top_nuisance_correlations"] else 0.0
            )
            row = {
                "margin": market - nuisance,
                "market": market,
                "nuisance": nuisance,
                "feature": pc["top_market_correlations"][0]["feature"] if pc["top_market_correlations"] else "",
            }
            if best_row is None or row["margin"] > best_row["margin"]:
                best_row = row
        ratios = entry["explained_variance_ratio"]
        best_margin.append(float(best_row["margin"]))
        best_market.append(float(best_row["market"]))
        best_nuisance.append(float(best_row["nuisance"]))
        best_feature.append(str(best_row["feature"]))
        pc1_var.append(float(ratios[0]) if len(ratios) > 0 else 0.0)
        pc2_var.append(float(ratios[1]) if len(ratios) > 1 else 0.0)
        pc3_var.append(float(ratios[2]) if len(ratios) > 2 else 0.0)
    return {
        "layers": layers,
        "best_margin": best_margin,
        "best_market": best_market,
        "best_nuisance": best_nuisance,
        "best_feature": best_feature,
        "pc1_var": pc1_var,
        "pc2_var": pc2_var,
        "pc3_var": pc3_var,
    }


def _family_breakdown(results: dict) -> dict[str, int]:
    breakdown: dict[str, int] = {}
    for family, variant, count in (
        ("market_basis_coupled", "pct_5m__net_flow_5m", 50),
        ("market_basis_coupled", "unique_traders_5m__top20_holder_pct", 50),
        ("market_basis_scalar", "net_flow_5m", 21),
        ("market_basis_scalar", "pct_5m", 21),
        ("market_basis_scalar", "top20_holder_pct", 21),
        ("market_basis_scalar", "unique_traders_5m", 21),
    ):
        breakdown[f"{family}:{variant}"] = count
    return breakdown


def build_charts() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    results = json.loads(RESULTS_PATH.read_text())
    mm = _state_series(results["states"]["market_mean"])
    me = _state_series(results["states"]["market_eos"])

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    families = _family_breakdown(results)
    ax = axes[0, 0]
    labels = list(families.keys())
    values = list(families.values())
    short = [
        "coupled\npct_5m×flow",
        "coupled\ntraders×holders",
        "scalar\nflow",
        "scalar\npct_5m",
        "scalar\nholders",
        "scalar\ntraders",
    ]
    colors = ["#b33a2a", "#d97b36", "#4f6d7a", "#6f8f72", "#8d6e63", "#7b8fb2"]
    ax.bar(short, values, color=colors)
    ax.set_title("Phase 1 prompt families")
    ax.set_ylabel("Prompts")
    ax.set_ylim(0, max(values) + 10)

    ax = axes[0, 1]
    ax.plot(mm["layers"], mm["best_margin"], color="#b33a2a", label="market_mean")
    ax.plot(me["layers"], me["best_margin"], color="#4f6d7a", label="market_eos")
    ax.axhline(0.0, color="#666", lw=1.0, ls="--")
    ax.set_title("Best market-over-nuisance margin by layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Margin")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(mm["layers"], mm["best_market"], color="#b33a2a", label="market corr")
    ax.plot(mm["layers"], mm["best_nuisance"], color="#d9a441", label="nuisance corr")
    ax.set_title("market_mean: best PC correlations by layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("|Spearman|")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(me["layers"], me["best_market"], color="#4f6d7a", label="market corr")
    ax.plot(me["layers"], me["best_nuisance"], color="#7b8fb2", label="nuisance corr")
    ax.set_title("market_eos: best PC correlations by layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("|Spearman|")
    ax.legend(frameon=False)

    fig.savefig(ASSET_DIR / "phase15_discovery_summary.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    selected_mm = [1, 4, 30, 35]
    selected_me = [1, 2, 3, 30]
    mm_idx = [mm["layers"].index(layer) for layer in selected_mm]
    me_idx = [me["layers"].index(layer) for layer in selected_me]

    ax = axes[0]
    x = np.arange(len(selected_mm))
    ax.bar(x - 0.25, [mm["pc1_var"][i] for i in mm_idx], width=0.25, color="#b33a2a", label="PC1")
    ax.bar(x, [mm["pc2_var"][i] for i in mm_idx], width=0.25, color="#d97b36", label="PC2")
    ax.bar(x + 0.25, [mm["pc3_var"][i] for i in mm_idx], width=0.25, color="#f0c36d", label="PC3")
    ax.set_xticks(x, [f"L{layer}" for layer in selected_mm])
    ax.set_title("market_mean explained variance")
    ax.set_ylabel("Variance ratio")
    ax.legend(frameon=False)

    ax = axes[1]
    x = np.arange(len(selected_me))
    ax.bar(x - 0.25, [me["pc1_var"][i] for i in me_idx], width=0.25, color="#4f6d7a", label="PC1")
    ax.bar(x, [me["pc2_var"][i] for i in me_idx], width=0.25, color="#7b8fb2", label="PC2")
    ax.bar(x + 0.25, [me["pc3_var"][i] for i in me_idx], width=0.25, color="#a9b8c9", label="PC3")
    ax.set_xticks(x, [f"L{layer}" for layer in selected_me])
    ax.set_title("market_eos explained variance")
    ax.set_ylabel("Variance ratio")
    ax.legend(frameon=False)

    fig.savefig(ASSET_DIR / "phase15_explained_variance.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    build_charts()
