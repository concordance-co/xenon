from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RESULTS_PATH = Path("data/analysis_results/synthetic_market_representation/phase10_set_geometry_context_v1/results.json")
OUTPUT_DIR = Path("data/report_assets/synthetic_market_phase10_geometry_deformation")

NAVY = "#16324F"
TEAL = "#2E6A69"
GOLD = "#CA9440"
ROSE = "#B56662"
SLATE = "#5E6F82"
GRID = "#D6DEE3"
CHARCOAL = "#21313F"

CONTEXT_LABELS = {
    "market_only": "Market-only",
    "low_risk": "Low risk",
    "high_risk": "High risk",
}

TRANSFER_LABELS = {
    "market_only_to_market_only": "Market -> Market",
    "market_only_to_low_risk": "Market -> Low risk",
    "market_only_to_high_risk": "Market -> High risk",
}

DEFORMATION_LABELS = {
    "market_only_to_low_risk": "Market -> Low risk",
    "market_only_to_high_risk": "Market -> High risk",
    "low_risk_to_high_risk": "Low risk -> High risk",
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


def coordinate_transfer_chart(data: dict) -> Path:
    summary = data["summary"]["set_geometry_context_transfer"]
    transfer_keys = list(TRANSFER_LABELS)
    x = np.arange(len(transfer_keys))
    width = 0.33

    latent_x = [summary["latent_x"][key]["r2"] for key in transfer_keys]
    latent_y = [summary["latent_y"][key]["r2"] for key in transfer_keys]

    fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.bar(x - width / 2, latent_x, width=width, color=NAVY, edgecolor="none", label="Latent X")
    ax.bar(x + width / 2, latent_y, width=width, color=TEAL, edgecolor="none", label="Latent Y")
    _setup_axes(ax)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Held-out R²")
    ax.set_xticks(x, [TRANSFER_LABELS[key] for key in transfer_keys])
    ax.set_title(
        "Base market coordinates transfer almost unchanged across context variants",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=CHARCOAL,
    )
    ax.legend(frameon=False, loc="lower left")
    for idx, key in enumerate(transfer_keys):
        best_layer = summary["latent_x"][key]["layer"]
        ax.text(
            idx,
            max(latent_x[idx], latent_y[idx]) + 0.02,
            f"L{best_layer}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=CHARCOAL,
        )
    fig.tight_layout()
    path = OUTPUT_DIR / "coordinate_transfer.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _best_realignment_rep(data: dict, context: str) -> str:
    best = data["summary"]["set_geometry_context_realignment"][context]
    return str(best["representation"])


def context_realignment_chart(data: dict) -> Path:
    contexts = ["market_only", "low_risk", "high_risk"]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), dpi=180, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, context in zip(axes, contexts, strict=True):
        rep = _best_realignment_rep(data, context)
        rows = data["set_geometry_context_realignment"][context][rep]
        layers = [row["layer"] for row in rows]
        base_vals = [row["base_distance_spearman_mean"] for row in rows]
        score_vals = [row["score_distance_spearman_mean"] for row in rows]
        margin_vals = [row["score_over_base_margin"] for row in rows]
        best = data["summary"]["set_geometry_context_realignment"][context]

        _setup_axes(ax)
        ax.plot(layers, base_vals, color=SLATE, linewidth=2.0, label="Base geometry")
        ax.plot(layers, score_vals, color=ROSE, linewidth=2.0, label="Context score geometry")
        ax.plot(layers, margin_vals, color=GOLD, linewidth=1.8, linestyle="--", label="Score - base")
        ax.axhline(0.0, color=GRID, linewidth=1.0, linestyle=":")
        ax.set_title(
            f"{CONTEXT_LABELS[context]}\n({rep})",
            fontsize=11.5,
            fontweight="bold",
            color=CHARCOAL,
        )
        ax.set_xlabel("Layer")
        ax.scatter([best["layer"]], [best["margin"]], color=GOLD, s=28, zorder=4)
        ax.text(
            best["layer"],
            best["margin"] + 0.01,
            f"L{best['layer']}",
            fontsize=8,
            color=CHARCOAL,
            ha="center",
        )

    axes[0].set_ylabel("Mean Spearman / margin")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle(
        "Later states can realign toward context-adjusted geometry without losing the base coordinate system",
        x=0.02,
        y=1.02,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=CHARCOAL,
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "context_realignment.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def context_deformation_chart(data: dict) -> Path:
    pairs = list(DEFORMATION_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    x = np.arange(len(pairs))
    best_spearman = [data["summary"]["set_geometry_context_deformation"][pair]["margin"] for pair in pairs]
    best_cosine = [data["summary"]["set_geometry_context_deformation"][pair]["nn_accuracy"] for pair in pairs]
    colors = [TEAL, ROSE, GOLD]

    axes[0].bar(x, best_spearman, color=colors, edgecolor="none")
    _setup_axes(axes[0])
    axes[0].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[0].set_xticks(x, [DEFORMATION_LABELS[pair] for pair in pairs], rotation=0)
    axes[0].set_ylabel("Best deformation Spearman")
    axes[0].set_title(
        "Activation-space geometry changes track score-space changes",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=CHARCOAL,
    )

    axes[1].bar(x, best_cosine, color=colors, edgecolor="none")
    _setup_axes(axes[1])
    axes[1].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[1].set_xticks(x, [DEFORMATION_LABELS[pair] for pair in pairs], rotation=0)
    axes[1].set_ylabel("Best deformation cosine")
    axes[1].set_title(
        "Direction of change is structured, not random",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=CHARCOAL,
    )

    fig.tight_layout()
    path = OUTPUT_DIR / "context_deformation.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_results()
    outputs = {
        "coordinate_transfer": str(coordinate_transfer_chart(data)),
        "context_realignment": str(context_realignment_chart(data)),
        "context_deformation": str(context_deformation_chart(data)),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
