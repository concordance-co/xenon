from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PHASE6_PATH = Path("data/analysis_results/synthetic_market_representation/phase6_profile_invariance_v1/results.json")
OUTPUT_DIR = Path("data/report_assets/synthetic_market_relational")

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

MODE_LABELS = {
    "full": "Full",
    "style_only": "Style-only",
    "layout_only": "Layout-only",
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


def object_comparison_chart(data: dict) -> Path:
    path = OUTPUT_DIR / "object_comparison.png"
    scenarios = list(SCENARIO_LABELS.keys())
    x = np.arange(len(scenarios))
    width = 0.22

    row_vals = [
        data["summary"]["profile_invariance_decomposition"][scenario]["best_layout_only"]["margin"]
        for scenario in scenarios
    ]
    relation_vals = [
        data["summary"]["pairwise_relation_invariance"][scenario]["layout_only"]["margin"]
        for scenario in scenarios
    ]
    geometry_vals = [
        data["summary"]["snapshot_geometry"][scenario]["layout_only"]["margin"]
        for scenario in scenarios
    ]

    fig, ax = plt.subplots(figsize=(10.4, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.bar(x - width, row_vals, width=width, color=ROSE, edgecolor="none", label="Row identity")
    ax.bar(x, relation_vals, width=width, color=TEAL, edgecolor="none", label="Pairwise relations")
    ax.bar(x + width, geometry_vals, width=width, color=NAVY, edgecolor="none", label="Snapshot geometry")
    _setup_axes(ax)
    ax.axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    ax.set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios])
    ax.set_ylabel("Best layout-only margin")
    ax.set_title("Layout-sensitive market representation is much more relational than row-identitarian", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def relation_modes_chart(data: dict) -> Path:
    path = OUTPUT_DIR / "relation_modes.png"
    scenarios = list(SCENARIO_LABELS.keys())
    modes = ["full", "style_only", "layout_only"]
    colors = [ROSE, GOLD, TEAL]
    x = np.arange(len(scenarios))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    for idx, mode in enumerate(modes):
        margins = [data["summary"]["pairwise_relation_invariance"][scenario][mode]["margin"] for scenario in scenarios]
        accs = [data["summary"]["pairwise_relation_invariance"][scenario][mode]["nn_accuracy"] for scenario in scenarios]
        axes[0].bar(x + (idx - 1) * width, margins, width=width, color=colors[idx], edgecolor="none", label=MODE_LABELS[mode])
        axes[1].bar(x + (idx - 1) * width, accs, width=width, color=colors[idx], edgecolor="none", label=MODE_LABELS[mode])

    _setup_axes(axes[0])
    axes[0].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[0].set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios])
    axes[0].set_ylabel("Pairwise relation margin")
    axes[0].set_title("Pairwise relations remain stable across nuisance variation", loc="left", fontsize=13, fontweight="bold", color=CHARCOAL)
    axes[0].legend(frameon=False, ncol=3, loc="upper right")

    _setup_axes(axes[1])
    axes[1].set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios])
    axes[1].set_ylabel("Pairwise relation NN accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Relation retrieval is near-perfect under style changes and strong under layout changes", loc="left", fontsize=13, fontweight="bold", color=CHARCOAL)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def geometry_modes_chart(data: dict) -> Path:
    path = OUTPUT_DIR / "geometry_modes.png"
    scenarios = list(SCENARIO_LABELS.keys())
    modes = ["full", "style_only", "layout_only"]
    colors = [ROSE, GOLD, NAVY]
    x = np.arange(len(scenarios))
    width = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    for idx, mode in enumerate(modes):
        margins = [data["summary"]["snapshot_geometry"][scenario][mode]["margin"] for scenario in scenarios]
        accs = [data["summary"]["snapshot_geometry"][scenario][mode]["nn_accuracy"] for scenario in scenarios]
        axes[0].bar(x + (idx - 1) * width, margins, width=width, color=colors[idx], edgecolor="none", label=MODE_LABELS[mode])
        axes[1].bar(x + (idx - 1) * width, accs, width=width, color=colors[idx], edgecolor="none", label=MODE_LABELS[mode])

    _setup_axes(axes[0])
    axes[0].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[0].set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios])
    axes[0].set_ylabel("Snapshot geometry margin")
    axes[0].set_title("Whole-snapshot geometry is present, but weaker than pairwise relations", loc="left", fontsize=13, fontweight="bold", color=CHARCOAL)
    axes[0].legend(frameon=False, ncol=3, loc="upper right")

    _setup_axes(axes[1])
    axes[1].set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios])
    axes[1].set_ylabel("Snapshot geometry NN accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title("Geometry alignment survives style variation better than layout variation", loc="left", fontsize=13, fontweight="bold", color=CHARCOAL)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load(PHASE6_PATH)
    outputs = {
        "object_comparison": str(object_comparison_chart(data)),
        "relation_modes": str(relation_modes_chart(data)),
        "geometry_modes": str(geometry_modes_chart(data)),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
