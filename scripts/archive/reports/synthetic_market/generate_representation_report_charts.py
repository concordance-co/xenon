from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PHASE4_PATH = Path("data/analysis_results/synthetic_market_representation/phase4_market_representation_v1/results.json")
PHASE5_PATH = Path("data/analysis_results/synthetic_market_representation/phase5_symbol_permutation_v1/results.json")
OUTPUT_DIR = Path("data/report_assets/synthetic_market_representation")

NAVY = "#16324F"
TEAL = "#2E6A69"
GOLD = "#CA9440"
ROSE = "#B56662"
SLATE = "#5E6F82"
GRID = "#D6DEE3"
CHARCOAL = "#21313F"
PALE = "#EEF3F7"

PROFILE_CONTROL_RANDOM_BASELINE = 0.18787878787878787

PRIMITIVE_LABELS = {
    "pct_5m": "Momentum",
    "net_flow_5m": "Flow",
    "unique_traders_5m": "Participation",
    "top20_holder_pct": "Concentration",
    "attractiveness_score": "Attractiveness",
    "risk_adjusted_score": "Risk-adjusted",
}

RANK_CONTEXT_LABELS = {
    "fixed_momentum_flow_pair": "Momentum/Flow pair",
    "fixed_participation_concentration_pair": "Participation/Concentration pair",
}

SYMBOL_PERM_LABELS = {
    "momentum_flow_permuted_market": "Momentum/Flow permuted",
    "participation_concentration_permuted_market": "Participation/Concentration permuted",
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


def dataset_counts_chart(phase4: dict, phase5: dict) -> Path:
    path = OUTPUT_DIR / "dataset_counts.png"
    labels = ["Phase 4\nrank-context", "Phase 5\nsymbol control"]
    values = [phase4["n_market_ticks"], phase5["n_market_ticks"]]
    colors = [NAVY, TEAL]

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=180)
    fig.patch.set_facecolor("white")
    bars = ax.bar(labels, values, color=colors, width=0.58, edgecolor="none")
    _setup_axes(ax)
    ax.set_ylabel("Captured market-only prompts")
    ax.set_title("Representation-control dataset sizes", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    ax.set_ylim(0, max(values) * 1.28)
    for bar, val in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2, f"{val}", ha="center", va="bottom", fontsize=10, color=SLATE)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def primitive_regression_chart(phase4: dict, phase5: dict) -> Path:
    path = OUTPUT_DIR / "primitive_regression.png"
    keys = list(PRIMITIVE_LABELS.keys())
    x = np.arange(len(keys))
    width = 0.34

    phase4_vals = [phase4["summary"]["primitive_regression"][key]["r2"] for key in keys]
    phase5_vals = [phase5["summary"]["primitive_regression"][key]["r2"] for key in keys]

    fig, ax = plt.subplots(figsize=(10.8, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.bar(x - width / 2, phase4_vals, width=width, color=NAVY, edgecolor="none", label="Phase 4")
    ax.bar(x + width / 2, phase5_vals, width=width, color=TEAL, edgecolor="none", label="Phase 5")
    _setup_axes(ax)
    ax.set_xticks(x, [PRIMITIVE_LABELS[key] for key in keys])
    ax.set_ylabel("Best held-out R²")
    ax.set_ylim(0.94, 1.005)
    ax.set_title("Primitive market variables are almost perfectly decodable", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    for xi, val in zip(x - width / 2, phase4_vals, strict=True):
        ax.text(xi, val + 0.0006, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=SLATE, rotation=90)
    for xi, val in zip(x + width / 2, phase5_vals, strict=True):
        ax.text(xi, val + 0.0006, f"{val:.3f}", ha="center", va="bottom", fontsize=8, color=SLATE, rotation=90)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def rank_context_confound_chart(phase4: dict) -> Path:
    path = OUTPUT_DIR / "rank_context_confound.png"
    keys = list(RANK_CONTEXT_LABELS.keys())
    margins = [phase4["summary"]["rank_context"][key]["same_symbol_margin"] for key in keys]

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=180)
    fig.patch.set_facecolor("white")
    bars = ax.bar([RANK_CONTEXT_LABELS[key] for key in keys], margins, color=[GOLD, ROSE], edgecolor="none", width=0.58)
    _setup_axes(ax)
    ax.set_ylabel("Same-symbol cosine margin")
    ax.set_ylim(0.0, max(margins) * 1.25)
    ax.set_title("Phase 4 rank-context result was symbol-confounded", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    for bar, val in zip(bars, margins, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=10, color=SLATE)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def symbol_permutation_control_chart(phase5: dict) -> Path:
    path = OUTPUT_DIR / "symbol_permutation_control.png"
    scenarios = list(SYMBOL_PERM_LABELS.keys())
    reps = {scenario: phase5["summary"]["symbol_permutation"][scenario]["representation"] for scenario in scenarios}

    fig, axes = plt.subplots(2, 1, figsize=(10.4, 8.6), dpi=180, sharex=True)
    fig.patch.set_facecolor("white")

    for scenario, color in zip(scenarios, (NAVY, TEAL), strict=True):
        rep = reps[scenario]
        series = phase5["symbol_permutation"][scenario][rep]
        layers = [entry["layer"] for entry in series]
        margins = [entry.get("profile_control_margin") for entry in series]
        accuracies = [entry.get("profile_control_nn_accuracy") for entry in series]
        axes[0].plot(layers, margins, color=color, linewidth=2.3, label=SYMBOL_PERM_LABELS[scenario])
        axes[1].plot(layers, accuracies, color=color, linewidth=2.3, label=SYMBOL_PERM_LABELS[scenario])

        best = phase5["summary"]["symbol_permutation"][scenario]
        axes[0].scatter([best["layer"]], [best["profile_control_margin"]], color=color, s=32, zorder=3)
        axes[1].scatter([best["layer"]], [best["profile_control_nn_accuracy"]], color=color, s=32, zorder=3)

    _setup_axes(axes[0])
    axes[0].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[0].set_ylabel("Profile control margin")
    axes[0].set_title("Profile retrieval after removing symbol and row shortcuts", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)

    _setup_axes(axes[1])
    axes[1].axhline(PROFILE_CONTROL_RANDOM_BASELINE, color=ROSE, linewidth=1.5, linestyle="--", label="Random baseline")
    axes[1].set_ylabel("Profile control NN accuracy")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylim(0.0, 1.0)

    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def best_layer_similarity_chart(phase5: dict) -> Path:
    path = OUTPUT_DIR / "best_layer_similarity.png"
    scenarios = list(SYMBOL_PERM_LABELS.keys())
    metrics = [
        ("same_profile_cosine_mean", "Same profile"),
        ("same_symbol_cosine_mean", "Same symbol"),
        ("same_row_cosine_mean", "Same row"),
        ("profile_control_other_cosine_mean", "Best non-profile\n(row/symbol removed)"),
    ]
    x = np.arange(len(scenarios))
    width = 0.18
    colors = [NAVY, TEAL, GOLD, ROSE]

    fig, ax = plt.subplots(figsize=(10.4, 5.0), dpi=180)
    fig.patch.set_facecolor("white")
    for idx, ((metric_key, label), color) in enumerate(zip(metrics, colors, strict=True)):
        vals = []
        for scenario in scenarios:
            summary = phase5["summary"]["symbol_permutation"][scenario]
            rep = summary["representation"]
            layer = summary["layer"]
            row = next(entry for entry in phase5["symbol_permutation"][scenario][rep] if entry["layer"] == layer)
            vals.append(row.get(metric_key))
        ax.bar(x + (idx - 1.5) * width, vals, width=width, color=color, edgecolor="none", label=label)
    _setup_axes(ax)
    ax.set_xticks(x, [SYMBOL_PERM_LABELS[scenario] for scenario in scenarios])
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Best-layer similarity structure is not uniform across factor families", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phase4 = _load(PHASE4_PATH)
    phase5 = _load(PHASE5_PATH)
    outputs = {
        "dataset_counts": str(dataset_counts_chart(phase4, phase5)),
        "primitive_regression": str(primitive_regression_chart(phase4, phase5)),
        "rank_context_confound": str(rank_context_confound_chart(phase4)),
        "symbol_permutation_control": str(symbol_permutation_control_chart(phase5)),
        "best_layer_similarity": str(best_layer_similarity_chart(phase5)),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
