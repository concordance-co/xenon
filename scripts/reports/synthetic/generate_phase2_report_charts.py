from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PHASE1_PATH = Path("data/analysis_results/synthetic_manifold/phase1/results.json")
DENSE_PATH = Path("data/analysis_results/synthetic_manifold/phase2_geometry/dense/results.json")
MINIMAL_PATH = Path("data/analysis_results/synthetic_manifold/phase2_geometry/minimal/results.json")
OUTPUT_DIR = Path("data/report_assets/synthetic_phase2")

NAVY = "#16324F"
TEAL = "#2E6A69"
GOLD = "#CA9440"
ROSE = "#B56662"
SLATE = "#5E6F82"
MIST = "#EAF2F2"
CREAM = "#F6EFE3"
GRID = "#D6DEE3"
CHARCOAL = "#21313F"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_all() -> dict[str, dict]:
    return {
        "Phase 1": _load(PHASE1_PATH),
        "Dense Sweep": _load(DENSE_PATH),
        "Minimal Sweep": _load(MINIMAL_PATH),
    }


def _setup_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)


def dataset_counts_chart(results: dict[str, dict]) -> Path:
    path = OUTPUT_DIR / "dataset_counts.png"
    labels = list(results.keys())
    values = [results[label]["n_market_ticks"] for label in labels]
    colors = [ROSE, NAVY, TEAL]

    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=180)
    fig.patch.set_facecolor("white")
    bars = ax.bar(labels, values, color=colors, edgecolor="none", width=0.62)
    _setup_axes(ax)
    ax.set_ylabel("Market-only prompts")
    ax.set_title("Synthetic dataset scale by phase", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    ax.set_ylim(0, max(values) * 1.22)
    for bar, val in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 12, str(val), ha="center", va="bottom", fontsize=10, color=SLATE)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def regression_comparison_chart(results: dict[str, dict]) -> Path:
    path = OUTPUT_DIR / "regression_comparison.png"
    targets = [
        ("attractiveness_score", "Attractiveness"),
        ("risk_adjusted_score", "Risk-adjusted"),
        ("edge_after_fee_score", "Edge after fee"),
    ]
    labels = list(results.keys())
    x = np.arange(len(targets))
    width = 0.22
    colors = [ROSE, NAVY, TEAL]

    fig, ax = plt.subplots(figsize=(10.2, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    for idx, (label, color) in enumerate(zip(labels, colors, strict=True)):
        vals = [results[label]["summary"]["regression"][target]["r2"] for target, _ in targets]
        ax.bar(x + (idx - 1) * width, vals, width=width, color=color, edgecolor="none", label=label)
        for xi, v in zip(x + (idx - 1) * width, vals, strict=True):
            ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=SLATE, rotation=90)
    _setup_axes(ax)
    ax.set_xticks(x, [label for _, label in targets])
    ax.set_ylabel("Best held-out R²")
    ax.set_ylim(0.9, 1.02)
    ax.set_title("Latent-score decodability remains extremely strong", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def scalar_geometry_comparison_chart(results: dict[str, dict]) -> Path:
    path = OUTPUT_DIR / "scalar_geometry_comparison.png"
    metrics = [
        ("pct_5m", "5m change"),
        ("net_flow_5m", "5m net flow"),
        ("top20_holder_pct", "Top20 concentration"),
    ]
    labels = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.22
    colors = [ROSE, NAVY, TEAL]

    fig, ax = plt.subplots(figsize=(10.2, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    for idx, (label, color) in enumerate(zip(labels, colors, strict=True)):
        vals = [results[label]["summary"]["scalar_geometry"][metric]["distance_value_spearman"] for metric, _ in metrics]
        ax.bar(x + (idx - 1) * width, vals, width=width, color=color, edgecolor="none", label=label)
        for xi, v in zip(x + (idx - 1) * width, vals, strict=True):
            ax.text(xi, v + 0.014, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=SLATE, rotation=90)
    _setup_axes(ax)
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylabel("Best distance/value Spearman")
    ax.set_ylim(0.0, 0.86)
    ax.set_title("Scalar geometry improved, but unevenly across variables", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def pct5m_layerwise_chart(results: dict[str, dict]) -> Path:
    path = OUTPUT_DIR / "pct5m_layerwise.png"
    fig, ax = plt.subplots(figsize=(10.2, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    colors = {"Phase 1": ROSE, "Dense Sweep": NAVY, "Minimal Sweep": TEAL}

    for label, result in results.items():
        layers = result["layers"]
        series = result["scalar_geometry"]["pct_5m"]["row_mean"]
        values = [entry.get("distance_value_spearman") for entry in series]
        color = colors[label]
        ax.plot(layers, values, color=color, linewidth=2.4, label=label)
        best = max(series, key=lambda entry: entry.get("distance_value_spearman", float("-inf")))
        ax.scatter([best["layer"]], [best["distance_value_spearman"]], color=color, s=42, zorder=3)

    _setup_axes(ax)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman(distance, scalar delta)")
    ax.set_ylim(0.0, 0.82)
    ax.set_title("5m-change ordering becomes cleaner with denser synthetic sweeps", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def best_asset_chart(results: dict[str, dict]) -> Path:
    path = OUTPUT_DIR / "best_asset.png"
    labels = list(results.keys())
    vals = [results[label]["summary"]["best_asset"]["auroc"] for label in labels]
    layers = [results[label]["summary"]["best_asset"]["layer"] for label in labels]
    colors = [ROSE, NAVY, TEAL]

    fig, ax = plt.subplots(figsize=(8.6, 4.6), dpi=180)
    fig.patch.set_facecolor("white")
    bars = ax.bar(labels, vals, color=colors, edgecolor="none", width=0.62)
    _setup_axes(ax)
    ax.set_ylabel("Best held-out AUROC")
    ax.set_ylim(0.85, 1.02)
    ax.set_title("Best-asset probes stay near ceiling across phases", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    for bar, val, layer in zip(bars, vals, layers, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.006, f"{val:.3f}\nL{layer}", ha="center", va="bottom", fontsize=9, color=SLATE)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = _load_all()
    outputs = {
        "dataset_counts": str(dataset_counts_chart(results)),
        "regression_comparison": str(regression_comparison_chart(results)),
        "scalar_geometry_comparison": str(scalar_geometry_comparison_chart(results)),
        "pct5m_layerwise": str(pct5m_layerwise_chart(results)),
        "best_asset": str(best_asset_chart(results)),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
