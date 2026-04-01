from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.synthetic_market.synthetic_market_representation_analysis import SET_GEOMETRY_COORDS_BY_SCENARIO

RESULTS_PATH = Path("data/analysis_results/synthetic_market_representation/phase11_set_geometry_risk_ladder_v1/results.json")
EXPORT_ASSETS_PATH = Path(
    "data/interp_exports/synthetic_market_phase11_set_geometry_risk_ladder/synthetic_market_asset_records.parquet"
)
OUTPUT_DIR = Path("data/report_assets/synthetic_market_phase11_risk_ladder")

NAVY = "#16324F"
TEAL = "#2E6A69"
GOLD = "#CA9440"
ROSE = "#B56662"
SLATE = "#5E6F82"
GRID = "#D6DEE3"
CHARCOAL = "#21313F"
RISK_COLORS = ["#16324F", "#275D73", "#377A7D", "#6A8F4E", "#A37A39", "#B56662"]

CONTEXTS = ["market_only", "risk_1", "risk_2", "risk_3", "risk_4", "risk_5"]
CONTEXT_LABELS = {
    "market_only": "Market",
    "risk_1": "Risk 1",
    "risk_2": "Risk 2",
    "risk_3": "Risk 3",
    "risk_4": "Risk 4",
    "risk_5": "Risk 5",
}
PAIR_LABELS = {
    "market_only_to_risk_1": "M→1",
    "risk_1_to_risk_2": "1→2",
    "risk_2_to_risk_3": "2→3",
    "risk_3_to_risk_4": "3→4",
    "risk_4_to_risk_5": "4→5",
    "market_only_to_risk_5": "M→5",
}
ASSET_COLORS = {
    "geo_alpha": NAVY,
    "geo_beta": TEAL,
    "geo_gamma": GOLD,
    "geo_delta": ROSE,
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
    x = np.arange(len(CONTEXTS))
    latent_x = []
    latent_y = []
    for context in CONTEXTS:
        key = "market_only_to_market_only" if context == "market_only" else f"market_only_to_{context}"
        latent_x.append(summary["latent_x"][key]["r2"])
        latent_y.append(summary["latent_y"][key]["r2"])

    fig, ax = plt.subplots(figsize=(9.8, 4.6), dpi=180)
    fig.patch.set_facecolor("white")
    _setup_axes(ax)
    ax.plot(x, latent_x, color=NAVY, linewidth=2.4, marker="o", label="Latent X")
    ax.plot(x, latent_y, color=TEAL, linewidth=2.4, marker="s", label="Latent Y")
    ax.set_ylim(0.992, 1.0005)
    ax.set_xticks(x, [CONTEXT_LABELS[context] for context in CONTEXTS])
    ax.set_ylabel("Held-out R²")
    ax.set_title(
        "The base 4-asset market axes survive the full DX-native risk ladder",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=CHARCOAL,
    )
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    path = OUTPUT_DIR / "coordinate_transfer.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def context_realignment_chart(data: dict) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.8), dpi=180, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, context, color in zip(axes.flat, CONTEXTS, RISK_COLORS, strict=True):
        rows = data["set_geometry_context_realignment"][context]["row_eos"]
        layers = [row["layer"] for row in rows]
        base_vals = [row["base_distance_spearman_mean"] for row in rows]
        score_vals = [row["score_distance_spearman_mean"] for row in rows]
        margin_vals = [row["score_over_base_margin"] for row in rows]
        best = data["summary"]["set_geometry_context_realignment"][context]

        _setup_axes(ax)
        ax.plot(layers, base_vals, color=SLATE, linewidth=1.9, label="Base geometry")
        ax.plot(layers, score_vals, color=color, linewidth=2.1, label="Score geometry")
        ax.plot(layers, margin_vals, color=GOLD, linewidth=1.6, linestyle="--", label="Score - base")
        ax.axhline(0.0, color=GRID, linewidth=1.0, linestyle=":")
        ax.scatter([best["layer"]], [best["margin"]], color=GOLD, s=26, zorder=4)
        ax.set_title(
            f"{CONTEXT_LABELS[context]} · row_eos",
            fontsize=11.2,
            fontweight="bold",
            color=CHARCOAL,
        )
        ax.text(
            best["layer"],
            best["margin"] + 0.004,
            f"L{best['layer']}",
            fontsize=7.5,
            ha="center",
            color=CHARCOAL,
        )
        ax.set_xlabel("Layer")

    axes[0, 0].set_ylabel("Mean Spearman / margin")
    axes[1, 0].set_ylabel("Mean Spearman / margin")
    axes[0, 0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle(
        "Late row_eos states realign toward score geometry at nearly the same depth across the full ladder",
        x=0.02,
        y=1.01,
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
    pairs = list(PAIR_LABELS)
    x = np.arange(len(pairs))
    spearman = [data["summary"]["set_geometry_context_deformation"][pair]["margin"] for pair in pairs]
    cosine = [data["summary"]["set_geometry_context_deformation"][pair]["nn_accuracy"] for pair in pairs]
    reps = [data["summary"]["set_geometry_context_deformation"][pair]["representation"] for pair in pairs]
    layers = [data["summary"]["set_geometry_context_deformation"][pair]["layer"] for pair in pairs]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    axes[0].bar(x, spearman, color=RISK_COLORS, edgecolor="none")
    _setup_axes(axes[0])
    axes[0].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[0].set_xticks(x, [PAIR_LABELS[pair] for pair in pairs])
    axes[0].set_ylabel("Best deformation Spearman")
    axes[0].set_title(
        "Adjacent risk steps produce structured, nonzero geometry changes",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=CHARCOAL,
    )
    for idx, (rep, layer) in enumerate(zip(reps, layers, strict=True)):
        axes[0].text(
            idx,
            spearman[idx] + 0.015,
            f"{rep}\nL{layer}",
            fontsize=7.5,
            ha="center",
            va="bottom",
            color=CHARCOAL,
        )

    axes[1].bar(x, cosine, color=RISK_COLORS, edgecolor="none")
    _setup_axes(axes[1])
    axes[1].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[1].set_xticks(x, [PAIR_LABELS[pair] for pair in pairs])
    axes[1].set_ylabel("Best deformation cosine")
    axes[1].set_title(
        "The ladder is not a perfectly smooth rotation: some steps are mixed",
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


def representative_score_geometry_chart() -> Path:
    rows = pq.read_table(EXPORT_ASSETS_PATH).to_pylist()
    scenario = "top_pair_cluster"
    example_rows = [row for row in rows if str(row["family_variant"]) == scenario]
    example_id = min(str(row["example_id"]) for row in example_rows)
    example_rows = [row for row in example_rows if str(row["example_id"]) == example_id]
    example_rows.sort(key=lambda row: (CONTEXTS.index(str(row["context_variant"])), int(row["row_index"])))

    latent_coords = SET_GEOMETRY_COORDS_BY_SCENARIO[scenario]
    by_context: dict[str, dict[str, tuple[float, float]]] = {}
    for row in example_rows:
        by_context.setdefault(str(row["context_variant"]), {})[str(row["profile_id"])] = (
            float(row["attractiveness_score"]),
            float(row["risk_adjusted_score"]),
        )

    fig, axes = plt.subplots(2, 4, figsize=(17.6, 8.6), dpi=180)
    fig.patch.set_facecolor("white")
    for ax in axes.flat:
        _setup_axes(ax, ygrid=False)
        ax.grid(color=GRID, linewidth=0.7, alpha=0.45)

    # Top-left: latent layout.
    ax0 = axes[0, 0]
    ax0.set_title("Latent layout", loc="left", fontsize=12, fontweight="bold", color=CHARCOAL)
    for profile_id, coords in latent_coords.items():
        ax0.scatter(coords[0], coords[1], s=72, color=ASSET_COLORS[profile_id], edgecolors="white", linewidth=0.8)
        ax0.text(coords[0] + 0.03, coords[1] + 0.03, profile_id.replace("geo_", "").title(), fontsize=8.5, color=CHARCOAL)
    ax0.set_xlabel("Latent x")
    ax0.set_ylabel("Latent y")

    # Remaining panels: score geometry across ladder.
    score_axes = list(axes.flat[1 : 1 + len(CONTEXTS)])
    for ax, context in zip(score_axes, CONTEXTS, strict=True):
        score_coords = by_context[context]
        ax.set_title(CONTEXT_LABELS[context], loc="left", fontsize=12, fontweight="bold", color=CHARCOAL)
        for profile_id, coords in score_coords.items():
            ax.scatter(coords[0], coords[1], s=72, color=ASSET_COLORS[profile_id], edgecolors="white", linewidth=0.8)
            ax.text(coords[0] + 0.015, coords[1] + 0.015, profile_id.replace("geo_", "").title(), fontsize=8.2, color=CHARCOAL)
        ax.set_xlabel("Attractiveness")
        ax.set_ylabel("Risk-adjusted")

    for ax in axes.flat[1 + len(CONTEXTS) :]:
        ax.axis("off")

    fig.suptitle(
        "Representative score geometry across the 1–5 risk ladder",
        x=0.02,
        y=1.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=CHARCOAL,
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "score_geometry_example.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_results()
    coordinate_transfer_chart(data)
    context_realignment_chart(data)
    context_deformation_chart(data)
    representative_score_geometry_chart()
    print(f"Wrote Phase 11 chart assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
