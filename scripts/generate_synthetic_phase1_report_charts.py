from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.db import connect_neon


RESULTS_PATH = Path("data/analysis_results/synthetic_manifold/phase1/results.json")
OUTPUT_DIR = Path("data/report_assets/synthetic_phase1")

NAVY = "#16324F"
TEAL = "#2E6A69"
GOLD = "#CA9440"
ROSE = "#B56662"
SLATE = "#5E6F82"
MIST = "#EAF2F2"
CREAM = "#F6EFE3"
GRID = "#D6DEE3"


def _load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text())


def _load_family_counts() -> list[tuple[str, int, int]]:
    conn = connect_neon()
    try:
        rows = conn.execute(
            """
            SELECT
              family,
              COUNT(*) FILTER (WHERE context_variant = 'market_only') AS market_only_n,
              COUNT(*) AS total_n
            FROM synthetic_market_examples_v0
            WHERE phase_name = 'phase1'
            GROUP BY family
            ORDER BY family
            """
        ).fetchall()
        return [
            (str(row["family"]), int(row["market_only_n"]), int(row["total_n"]))
            for row in rows
        ]
    finally:
        conn.close()


def _setup_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)


def dataset_composition_chart() -> Path:
    counts = _load_family_counts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "dataset_composition.png"

    labels = [family.replace("_", "\n") for family, _, _ in counts]
    market_only = [row[1] for row in counts]
    total = [row[2] for row in counts]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.bar(x - width / 2, total, width=width, color=MIST, edgecolor=NAVY, linewidth=1.0, label="All contexts")
    ax.bar(x + width / 2, market_only, width=width, color=TEAL, edgecolor=TEAL, linewidth=1.0, label="Market only")
    _setup_axes(ax)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Prompts")
    ax.set_title("Synthetic Phase-1 Dataset Composition", loc="left", fontsize=14, fontweight="bold", color=NAVY)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    for xi, v in zip(x - width / 2, total, strict=True):
        ax.text(xi, v + 4, str(v), ha="center", va="bottom", fontsize=8, color=SLATE)
    for xi, v in zip(x + width / 2, market_only, strict=True):
        ax.text(xi, v + 4, str(v), ha="center", va="bottom", fontsize=8, color=SLATE)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def latent_regression_chart(results: dict) -> Path:
    path = OUTPUT_DIR / "latent_regression.png"
    layers = results["layers"]
    targets = [
        ("attractiveness_score", "Attractiveness"),
        ("risk_adjusted_score", "Risk-adjusted"),
        ("edge_after_fee_score", "Edge after fee"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=180, sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (target, label) in zip(axes, targets, strict=True):
        row_mean = [entry.get("r2") for entry in results["regression"][target]["row_mean"]]
        row_eos = [entry.get("r2") for entry in results["regression"][target]["row_eos"]]
        ax.plot(layers, row_mean, color=NAVY, linewidth=2.2, label="row_mean")
        ax.plot(layers, row_eos, color=TEAL, linewidth=2.0, linestyle="--", label="row_eos")
        _setup_axes(ax)
        ax.set_title(label, fontsize=11, color=NAVY)
        ax.set_xlabel("Layer")
        ax.set_ylim(0.9, 1.005)
    axes[0].set_ylabel("Held-out R²")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Latent Score Decodability Peaks Early and Stays High", x=0.065, ha="left", fontsize=14, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def preference_probe_chart(results: dict) -> Path:
    path = OUTPUT_DIR / "preference_probe_curves.png"
    layers = results["layers"]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=180, sharey=True)
    fig.patch.set_facecolor("white")

    best_asset_mean = [entry.get("auroc") for entry in results["best_asset"]["row_mean"]]
    best_asset_eos = [entry.get("auroc") for entry in results["best_asset"]["row_eos"]]
    axes[0].plot(layers, best_asset_mean, color=NAVY, linewidth=2.2, label="row_mean")
    axes[0].plot(layers, best_asset_eos, color=TEAL, linewidth=2.0, linestyle="--", label="row_eos")
    axes[0].set_title("Best asset", fontsize=11, color=NAVY)

    for ax, target, title in [
        (axes[1], "a_beats_b_on_attractiveness", "Pairwise attractiveness"),
        (axes[2], "a_beats_b_on_risk_adjusted", "Pairwise risk-adjusted"),
    ]:
        diff_row_mean = [entry.get("auroc") for entry in results["pairwise"][target]["diff"]["row_mean"]]
        concat_row_mean = [entry.get("auroc") for entry in results["pairwise"][target]["concat"]["row_mean"]]
        ax.plot(layers, diff_row_mean, color=ROSE, linewidth=2.2, label="diff:row_mean")
        ax.plot(layers, concat_row_mean, color=GOLD, linewidth=2.0, linestyle="--", label="concat:row_mean")
        ax.set_title(title, fontsize=11, color=NAVY)

    for ax in axes:
        _setup_axes(ax)
        ax.set_xlabel("Layer")
        ax.set_ylim(0.5, 1.02)
    axes[0].set_ylabel("Held-out AUROC")
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].legend(frameon=False, loc="lower right")
    axes[2].legend(frameon=False, loc="lower right")
    fig.suptitle("Choice Signals Are Nearly Trivial on the Controlled Synthetic Slice", x=0.065, ha="left", fontsize=14, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def scalar_geometry_chart(results: dict) -> Path:
    path = OUTPUT_DIR / "scalar_geometry.png"
    layers = results["layers"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), dpi=180)
    fig.patch.set_facecolor("white")

    for family, color in [("pct_5m", NAVY), ("net_flow_5m", TEAL), ("top20_holder_pct", GOLD)]:
        values = [entry.get("distance_value_spearman") for entry in results["scalar_geometry"][family]["row_mean"]]
        axes[0].plot(layers, values, color=color, linewidth=2.2, label=family)
    _setup_axes(axes[0])
    axes[0].set_title("Distance ordering vs scalar value", fontsize=11, color=NAVY)
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Spearman(distance, value delta)")
    axes[0].set_ylim(0.0, 0.8)
    axes[0].legend(frameon=False, loc="upper left")

    best_vals = []
    labels = []
    for family in ["pct_5m", "net_flow_5m", "top20_holder_pct"]:
        series = results["scalar_geometry"][family]["row_mean"]
        best = max(series, key=lambda entry: entry.get("distance_value_spearman", -999))
        labels.append(family.replace("_", "\n"))
        best_vals.append(best["distance_value_spearman"])
    bars = axes[1].bar(labels, best_vals, color=[NAVY, TEAL, GOLD], edgecolor="none")
    _setup_axes(axes[1])
    axes[1].set_title("Best layer by scalar family", fontsize=11, color=NAVY)
    axes[1].set_ylabel("Best distance/value Spearman")
    axes[1].set_ylim(0.0, 0.8)
    for bar, val in zip(bars, best_vals, strict=True):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=9, color=SLATE)

    fig.suptitle("Scalar Sweeps Show Partial Geometry, Not a Crisp 1D Manifold Yet", x=0.065, ha="left", fontsize=14, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    results = _load_results()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "dataset_composition": str(dataset_composition_chart()),
        "latent_regression": str(latent_regression_chart(results)),
        "preference_probe_curves": str(preference_probe_chart(results)),
        "scalar_geometry": str(scalar_geometry_chart(results)),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
