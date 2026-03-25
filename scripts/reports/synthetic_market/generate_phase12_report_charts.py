from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path(
    "data/analysis_results/synthetic_market_transform/phase11_set_geometry_risk_ladder_v1/phase12_explicit_transforms_v1/results.json"
)
OUTPUT_DIR = Path("data/report_assets/synthetic_market_phase12_transforms")

NAVY = "#16324F"
TEAL = "#2E6A69"
GOLD = "#CA9440"
ROSE = "#B56662"
SLATE = "#5E6F82"
GRID = "#D6DEE3"
CHARCOAL = "#21313F"
CREAM = "#F5F1EA"

PAIR_ORDER = [
    "market_only_to_risk_1",
    "risk_1_to_risk_2",
    "risk_2_to_risk_3",
    "risk_3_to_risk_4",
    "risk_4_to_risk_5",
    "market_only_to_risk_5",
]
PAIR_LABELS = {
    "market_only_to_risk_1": "M→1",
    "risk_1_to_risk_2": "1→2",
    "risk_2_to_risk_3": "2→3",
    "risk_3_to_risk_4": "3→4",
    "risk_4_to_risk_5": "4→5",
    "market_only_to_risk_5": "M→5",
}
FAMILY_ORDER = ["identity", "orthogonal", "similarity", "diagonal", "linear"]
FAMILY_LABELS = {
    "identity": "Identity",
    "orthogonal": "Orthogonal",
    "similarity": "Similarity",
    "diagonal": "Diagonal",
    "linear": "Linear",
}
FAMILY_COLORS = {
    "identity": SLATE,
    "orthogonal": NAVY,
    "similarity": TEAL,
    "diagonal": GOLD,
    "linear": ROSE,
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


def family_heatmap_chart(data: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8), dpi=180)
    fig.patch.set_facecolor("white")
    for ax, state_name in zip(axes, ["early", "late"], strict=True):
        rows = []
        for pair in PAIR_ORDER:
            rows.append(
                [
                    data["states"][state_name]["pair_transforms"][pair][family]["coord_r2_mean"]
                    for family in FAMILY_ORDER
                ]
            )
        arr = np.asarray(rows, dtype=np.float32)
        im = ax.imshow(arr, aspect="auto", cmap="YlGnBu", vmin=0.65 if state_name == "late" else 0.99, vmax=1.0)
        ax.set_xticks(range(len(FAMILY_ORDER)), [FAMILY_LABELS[family] for family in FAMILY_ORDER], rotation=30, ha="right")
        ax.set_yticks(range(len(PAIR_ORDER)), [PAIR_LABELS[pair] for pair in PAIR_ORDER])
        ax.set_title(
            f"{state_name.title()} state · {data['states'][state_name]['row_key']} L{data['states'][state_name]['layer']}",
            fontsize=12,
            fontweight="bold",
            color=CHARCOAL,
        )
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=7.5, color=CHARCOAL)
    fig.suptitle(
        "Early risk-step transforms are almost rigid; late transforms are locally mixed",
        x=0.02,
        y=1.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=CHARCOAL,
    )
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, pad=0.02)
    cbar.set_label("Coordinate reconstruction R²", color=CHARCOAL)
    fig.tight_layout()
    path = OUTPUT_DIR / "family_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def late_winner_stats_chart(data: dict) -> Path:
    state = data["states"]["late"]
    best_families = state["summary"]["best_family_by_pair"]
    x = np.arange(len(PAIR_ORDER))
    angles = []
    anisotropy = []
    labels = []
    for pair in PAIR_ORDER:
        family = best_families[pair]["family"]
        labels.append(FAMILY_LABELS[family])
        metrics = state["pair_transforms"][pair][family]
        angles.append(metrics["rotation_angle_deg"])
        anisotropy.append(metrics["anisotropy_ratio"] or 1.0)

    fig, ax1 = plt.subplots(figsize=(10.8, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    bars = ax1.bar(x, anisotropy, color=[FAMILY_COLORS[best_families[p]["family"]] for p in PAIR_ORDER], edgecolor="none")
    _setup_axes(ax1)
    ax1.set_ylabel("Winner anisotropy ratio")
    ax1.set_xticks(x, [PAIR_LABELS[pair] for pair in PAIR_ORDER])
    ax1.set_title(
        "Late-step winners alternate between near-rigid maps and strongly anisotropic local fits",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=CHARCOAL,
    )
    ax2 = ax1.twinx()
    ax2.plot(x, angles, color=NAVY, linewidth=2.2, marker="o")
    ax2.set_ylabel("Rotation angle (deg)", color=NAVY)
    ax2.tick_params(axis="y", colors=NAVY)
    for idx, (bar, label) in enumerate(zip(bars, labels, strict=True)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.22,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color=CHARCOAL,
        )
    fig.tight_layout()
    path = OUTPUT_DIR / "late_winner_stats.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def composition_chart(data: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    for ax, state_name in zip(axes, ["early", "late"], strict=True):
        state = data["states"][state_name]
        x = np.arange(len(FAMILY_ORDER))
        direct = [state["composition"][family]["direct"]["coord_r2_mean"] for family in FAMILY_ORDER]
        composed = [state["composition"][family]["composed"]["coord_r2_mean"] for family in FAMILY_ORDER]
        cosine = [state["composition"][family]["matrix_cosine"] for family in FAMILY_ORDER]
        width = 0.33

        _setup_axes(ax)
        ax.bar(x - width / 2, direct, width=width, color=NAVY, edgecolor="none", label="Direct M→5 fit")
        ax.bar(x + width / 2, composed, width=width, color=TEAL, edgecolor="none", label="Composed adjacent maps")
        ax.set_xticks(x, [FAMILY_LABELS[family] for family in FAMILY_ORDER], rotation=25, ha="right")
        ax.set_ylabel("Coordinate R² on M→5")
        ax.set_ylim(0.55 if state_name == "late" else 0.98, 1.01)
        ax.set_title(
            f"{state_name.title()} state",
            fontsize=12,
            fontweight="bold",
            color=CHARCOAL,
        )

        ax2 = ax.twinx()
        ax2.plot(x, cosine, color=GOLD, linewidth=2.0, marker="o")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("Composed vs direct matrix cosine", color=GOLD)
        ax2.tick_params(axis="y", colors=GOLD)

    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle(
        "Late global behavior composes best under near-rigid maps, not under the flexible linear fits",
        x=0.02,
        y=1.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=CHARCOAL,
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "composition.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_results()
    family_heatmap_chart(data)
    late_winner_stats_chart(data)
    composition_chart(data)
    print(f"Wrote Phase 12 chart assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
