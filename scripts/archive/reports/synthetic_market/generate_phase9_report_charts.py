from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.interp.synthetic.market import SET_GEOMETRY_SCENARIOS


RESULTS_PATH = Path("data/analysis_results/synthetic_market_representation/phase9_set_geometry_v1/results.json")
OUTPUT_DIR = Path("data/report_assets/synthetic_market_phase9_set_geometry")

NAVY = "#16324F"
TEAL = "#2E6A69"
GOLD = "#CA9440"
ROSE = "#B56662"
SLATE = "#5E6F82"
GRID = "#D6DEE3"
CHARCOAL = "#21313F"

SCENARIO_LABELS = {
    "even_ladder": "Even Ladder",
    "top_pair_cluster": "Top Pair Cluster",
    "dominant_outlier": "Dominant Outlier",
    "middle_gap": "Middle Gap",
}

MODE_LABELS = {
    "full": "Full",
    "style_only": "Style-only",
    "layout_only": "Layout-only",
    "magnitude_only": "Magnitude-only",
}


def _load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text())


def _setup_axes(ax: plt.Axes, *, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.65)


def latent_shape_chart() -> Path:
    scenario_rows = []
    pair_labels = []
    for scenario in SET_GEOMETRY_SCENARIOS:
        coords = {profile["profile_id"]: np.asarray(profile["coords"], dtype=np.float32) for profile in scenario["profiles"]}
        ordered_profiles = tuple(coords)
        distances = []
        labels = []
        for left_idx in range(len(ordered_profiles)):
            for right_idx in range(left_idx + 1, len(ordered_profiles)):
                left = ordered_profiles[left_idx]
                right = ordered_profiles[right_idx]
                distances.append(float(np.linalg.norm(coords[left] - coords[right])))
                labels.append(f"{left.split('_')[-1][0].upper()}-{right.split('_')[-1][0].upper()}")
        pair_labels = labels
        scenario_rows.append((scenario["name"], distances))

    fig, ax = plt.subplots(figsize=(11.5, 4.7), dpi=180)
    fig.patch.set_facecolor("white")
    colors = [NAVY, TEAL, GOLD, ROSE]
    x = np.arange(len(pair_labels))
    for idx, (scenario_name, distances) in enumerate(scenario_rows):
        ax.plot(
            x,
            distances,
            marker="o",
            linewidth=2.2,
            color=colors[idx % len(colors)],
            label=SCENARIO_LABELS.get(str(scenario_name), str(scenario_name)),
        )
    _setup_axes(ax)
    ax.set_xticks(x, pair_labels)
    ax.set_ylabel("Latent pair distance")
    ax.set_title(
        "Phase 9 keeps rank order fixed while changing the 4-asset latent shape",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=CHARCOAL,
    )
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    path = OUTPUT_DIR / "latent_shapes.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def coordinate_regression_chart(data: dict) -> Path:
    targets = ["latent_x", "latent_y"]
    labels = ["Latent X", "Latent Y"]
    values = [data["summary"]["set_geometry_coordinate_regression"][target]["r2"] for target in targets]
    reps = [
        f"{data['summary']['set_geometry_coordinate_regression'][target]['representation']} L{data['summary']['set_geometry_coordinate_regression'][target]['layer']}"
        for target in targets
    ]

    fig, ax = plt.subplots(figsize=(7.8, 4.4), dpi=180)
    fig.patch.set_facecolor("white")
    bars = ax.bar(labels, values, color=[NAVY, TEAL], edgecolor="none", width=0.56)
    _setup_axes(ax)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Best held-out R²")
    ax.set_title(
        "The latent 2D market axes are linearly explicit in individual row states",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=CHARCOAL,
    )
    for bar, value, rep in zip(bars, values, reps, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}\n{rep}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=CHARCOAL,
        )
    fig.tight_layout()
    path = OUTPUT_DIR / "coordinate_regression.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def alignment_chart(data: dict) -> Path:
    scenarios = list(SCENARIO_LABELS.keys())
    spearman_vals = [data["summary"]["set_geometry_alignment"][scenario]["margin"] for scenario in scenarios]
    closest_vals = [data["summary"]["set_geometry_alignment"][scenario]["nn_accuracy"] for scenario in scenarios]
    farthest_vals = []
    for scenario in scenarios:
        best_rep = data["summary"]["set_geometry_alignment"][scenario]["representation"]
        best_layer = data["summary"]["set_geometry_alignment"][scenario]["layer"]
        matching = next(
            row
            for row in data["set_geometry_alignment"][scenario][best_rep]
            if row["layer"] == best_layer
        )
        farthest_vals.append(matching["farthest_pair_accuracy"])
    layers = [data["summary"]["set_geometry_alignment"][scenario]["layer"] for scenario in scenarios]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    x = np.arange(len(scenarios))

    axes[0].bar(x, spearman_vals, color=[NAVY, TEAL, GOLD, ROSE], edgecolor="none")
    _setup_axes(axes[0])
    axes[0].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[0].set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios], rotation=0)
    axes[0].set_ylabel("Best distance Spearman")
    axes[0].set_title(
        "Within-snapshot row geometry partially tracks the latent 4-asset shape",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=CHARCOAL,
    )

    width = 0.34
    axes[1].bar(x - width / 2, closest_vals, width=width, color=TEAL, edgecolor="none", label="Closest pair")
    axes[1].bar(x + width / 2, farthest_vals, width=width, color=GOLD, edgecolor="none", label="Farthest pair")
    _setup_axes(axes[1])
    axes[1].set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios], rotation=0)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Exact pair accuracy")
    axes[1].set_title(
        "Best layer for each scenario",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=CHARCOAL,
    )
    axes[1].legend(frameon=False, loc="upper right")
    for idx, layer in enumerate(layers):
        peak = max(closest_vals[idx], farthest_vals[idx])
        axes[1].text(idx, peak + 0.03, f"L{layer}", ha="center", va="bottom", fontsize=8, color=CHARCOAL)

    fig.tight_layout()
    path = OUTPUT_DIR / "alignment.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def identity_modes_chart(data: dict) -> Path:
    scenarios = list(SCENARIO_LABELS.keys())
    modes = ["full", "style_only", "layout_only", "magnitude_only"]
    x = np.arange(len(scenarios))
    width = 0.18
    colors = [ROSE, NAVY, TEAL, GOLD]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    for idx, mode in enumerate(modes):
        margins = [data["summary"]["set_geometry_identity"][scenario][mode]["margin"] for scenario in scenarios]
        accs = [data["summary"]["set_geometry_identity"][scenario][mode]["nn_accuracy"] for scenario in scenarios]
        offset = (idx - (len(modes) - 1) / 2) * width
        axes[0].bar(x + offset, margins, width=width, color=colors[idx], edgecolor="none", label=MODE_LABELS[mode])
        axes[1].bar(x + offset, accs, width=width, color=colors[idx], edgecolor="none", label=MODE_LABELS[mode])

    _setup_axes(axes[0])
    axes[0].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[0].set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios], rotation=0)
    axes[0].set_ylabel("Best geometry-identity margin")
    axes[0].set_title(
        "Geometry-family retrieval against same-rank negatives",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=CHARCOAL,
    )
    axes[0].legend(frameon=False, ncol=2, loc="upper right")

    _setup_axes(axes[1])
    axes[1].set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios], rotation=0)
    axes[1].set_ylabel("Best NN accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title(
        "Identity remains much harder than latent-axis recovery",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=CHARCOAL,
    )

    fig.tight_layout()
    path = OUTPUT_DIR / "identity_modes.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_results()
    outputs = {
        "latent_shapes": str(latent_shape_chart()),
        "coordinate_regression": str(coordinate_regression_chart(data)),
        "alignment": str(alignment_chart(data)),
        "identity_modes": str(identity_modes_chart(data)),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
