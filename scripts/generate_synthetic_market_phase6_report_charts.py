from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PHASE5_PATH = Path("data/analysis_results/synthetic_market_representation/phase5_symbol_permutation_v1/results.json")
PHASE6_PATH = Path("data/analysis_results/synthetic_market_representation/phase6_profile_invariance_v1/results.json")
OUTPUT_DIR = Path("data/report_assets/synthetic_market_phase6_profile_invariance")

NAVY = "#16324F"
TEAL = "#2E6A69"
GOLD = "#CA9440"
ROSE = "#B56662"
SLATE = "#5E6F82"
GRID = "#D6DEE3"
CHARCOAL = "#21313F"

SCENARIO_LABELS = {
    "momentum_flow_tiebreak": "Momentum × Flow",
    "participation_concentration_tiebreak": "Participation × Concentration",
}

PHASE5_SCENARIOS = {
    "momentum_flow_tiebreak": "momentum_flow_permuted_market",
    "participation_concentration_tiebreak": "participation_concentration_permuted_market",
}

PRIMITIVE_LABELS = {
    "pct_5m": "Momentum",
    "net_flow_5m": "Flow",
    "unique_traders_5m": "Participation",
    "top20_holder_pct": "Concentration",
    "attractiveness_score": "Attractiveness",
    "risk_adjusted_score": "Risk-adjusted",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _setup_axes(ax: plt.Axes, *, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.65)


def primitive_regression_chart(phase6: dict) -> Path:
    path = OUTPUT_DIR / "primitive_regression_phase6.png"
    keys = list(PRIMITIVE_LABELS.keys())
    values = [phase6["summary"]["primitive_regression"][key]["r2"] for key in keys]

    fig, ax = plt.subplots(figsize=(10.4, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    bars = ax.bar(np.arange(len(keys)), values, color=NAVY, edgecolor="none", width=0.62)
    _setup_axes(ax)
    ax.set_xticks(np.arange(len(keys)), [PRIMITIVE_LABELS[key] for key in keys])
    ax.set_ylabel("Best held-out R²")
    ax.set_ylim(0.97, 1.005)
    ax.set_title("Primitive market factors remain explicit under the harder invariance slice", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    for bar, val in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0005, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=SLATE, rotation=90)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def full_control_comparison_chart(phase5: dict, phase6: dict) -> Path:
    path = OUTPUT_DIR / "full_control_comparison.png"
    scenarios = list(SCENARIO_LABELS.keys())
    x = np.arange(len(scenarios))
    width = 0.34

    phase5_margins = [
        phase5["summary"]["symbol_permutation"][PHASE5_SCENARIOS[scenario]]["profile_control_margin"]
        for scenario in scenarios
    ]
    phase6_margins = [
        phase6["summary"]["symbol_permutation"][scenario]["profile_control_margin"]
        for scenario in scenarios
    ]
    phase5_acc = [
        phase5["summary"]["symbol_permutation"][PHASE5_SCENARIOS[scenario]]["profile_control_nn_accuracy"]
        for scenario in scenarios
    ]
    phase6_acc = [
        phase6["summary"]["symbol_permutation"][scenario]["profile_control_nn_accuracy"]
        for scenario in scenarios
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    axes[0].bar(x - width / 2, phase5_margins, width=width, color=GOLD, edgecolor="none", label="Phase 5")
    axes[0].bar(x + width / 2, phase6_margins, width=width, color=ROSE, edgecolor="none", label="Phase 6")
    _setup_axes(axes[0])
    axes[0].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[0].set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios])
    axes[0].set_ylabel("Best full-control margin")
    axes[0].set_title("All nuisances at once is a much harder control", loc="left", fontsize=13, fontweight="bold", color=CHARCOAL)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")

    axes[1].bar(x - width / 2, phase5_acc, width=width, color=GOLD, edgecolor="none", label="Phase 5")
    axes[1].bar(x + width / 2, phase6_acc, width=width, color=ROSE, edgecolor="none", label="Phase 6")
    _setup_axes(axes[1])
    axes[1].set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios])
    axes[1].set_ylabel("Best full-control NN accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Participation × Concentration still survives better", loc="left", fontsize=13, fontweight="bold", color=CHARCOAL)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def decomposition_best_chart(phase6: dict) -> Path:
    path = OUTPUT_DIR / "decomposition_best.png"
    scenarios = list(SCENARIO_LABELS.keys())
    x = np.arange(len(scenarios))
    width = 0.36

    style_margins = [
        phase6["summary"]["profile_invariance_decomposition"][scenario]["best_style_only"]["margin"]
        for scenario in scenarios
    ]
    layout_margins = [
        phase6["summary"]["profile_invariance_decomposition"][scenario]["best_layout_only"]["margin"]
        for scenario in scenarios
    ]
    style_acc = [
        phase6["summary"]["profile_invariance_decomposition"][scenario]["best_style_only"]["nn_accuracy"]
        for scenario in scenarios
    ]
    layout_acc = [
        phase6["summary"]["profile_invariance_decomposition"][scenario]["best_layout_only"]["nn_accuracy"]
        for scenario in scenarios
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    axes[0].bar(x - width / 2, style_margins, width=width, color=TEAL, edgecolor="none", label="Style-only")
    axes[0].bar(x + width / 2, layout_margins, width=width, color=NAVY, edgecolor="none", label="Layout-only")
    _setup_axes(axes[0])
    axes[0].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[0].set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios])
    axes[0].set_ylabel("Best decomposition margin")
    axes[0].set_title("Surface style is robust; layout is the bottleneck", loc="left", fontsize=13, fontweight="bold", color=CHARCOAL)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")

    axes[1].bar(x - width / 2, style_acc, width=width, color=TEAL, edgecolor="none", label="Style-only")
    axes[1].bar(x + width / 2, layout_acc, width=width, color=NAVY, edgecolor="none", label="Layout-only")
    _setup_axes(axes[1])
    axes[1].set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios])
    axes[1].set_ylabel("Best decomposition NN accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Layout-sensitive retrieval still favors Participation × Concentration", loc="left", fontsize=13, fontweight="bold", color=CHARCOAL)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def decomposition_curves_chart(phase6: dict) -> Path:
    path = OUTPUT_DIR / "decomposition_curves.png"
    scenarios = list(SCENARIO_LABELS.keys())
    colors = {"full": ROSE, "style": TEAL, "layout": NAVY}

    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.6), dpi=180, sharex=True)
    fig.patch.set_facecolor("white")

    for ax, scenario in zip(axes, scenarios, strict=True):
        full_series = phase6["symbol_permutation"][scenario]["row_eos"]
        dec_series = phase6["profile_invariance_decomposition"][scenario]["row_eos"]
        layers = [row["layer"] for row in full_series]
        full_margin = [row["profile_control_margin"] for row in full_series]
        style_margin = [row["style_only_margin"] for row in dec_series]
        layout_margin = [row["layout_only_margin"] for row in dec_series]

        ax.plot(layers, full_margin, color=colors["full"], linewidth=2.0, label="Full control")
        ax.plot(layers, style_margin, color=colors["style"], linewidth=2.0, label="Style-only")
        ax.plot(layers, layout_margin, color=colors["layout"], linewidth=2.0, label="Layout-only")
        _setup_axes(ax)
        ax.axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
        ax.set_ylabel("Margin")
        ax.set_title(SCENARIO_LABELS[scenario], loc="left", fontsize=12.5, fontweight="bold", color=CHARCOAL)

    axes[1].set_xlabel("Layer")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, ncol=3, loc="upper left")
    fig.suptitle("Decomposing the nuisance stack shows what actually breaks invariance", x=0.06, ha="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase5 = _load(PHASE5_PATH)
    phase6 = _load(PHASE6_PATH)
    outputs = {
        "primitive_regression_phase6": str(primitive_regression_chart(phase6)),
        "full_control_comparison": str(full_control_comparison_chart(phase5, phase6)),
        "decomposition_best": str(decomposition_best_chart(phase6)),
        "decomposition_curves": str(decomposition_curves_chart(phase6)),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
