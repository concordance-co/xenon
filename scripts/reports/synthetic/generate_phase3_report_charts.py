from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DENSE_PATH = Path("data/analysis_results/synthetic_manifold/phase3_coupled_geometry/dense/results.json")
MINIMAL_PATH = Path("data/analysis_results/synthetic_manifold/phase3_coupled_geometry/minimal/results.json")
OUTPUT_DIR = Path("data/report_assets/synthetic_phase3")

NAVY = "#16324F"
TEAL = "#2E6A69"
GOLD = "#CA9440"
ROSE = "#B56662"
SLATE = "#5E6F82"
GRID = "#D6DEE3"
CHARCOAL = "#21313F"

PAIR_LABELS = {
    "pct_5m__unique_traders_5m": "Momentum x Participation",
    "pct_5m__top20_holder_pct": "Momentum x Concentration",
    "pct_5m__net_flow_5m": "Momentum x Flow",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_all() -> dict[str, dict]:
    return {
        "Dense": _load(DENSE_PATH),
        "Minimal": _load(MINIMAL_PATH),
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
    colors = [NAVY, TEAL]

    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=180)
    fig.patch.set_facecolor("white")
    bars = ax.bar(labels, values, color=colors, edgecolor="none", width=0.62)
    _setup_axes(ax)
    ax.set_ylabel("Market-only prompts")
    ax.set_title("Coupled-factor prompt counts by family", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    ax.set_ylim(0, max(values) * 1.22)
    for bar, val in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 8, str(val), ha="center", va="bottom", fontsize=10, color=SLATE)
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
    width = 0.28
    colors = [NAVY, TEAL]

    fig, ax = plt.subplots(figsize=(10.2, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    for idx, (label, color) in enumerate(zip(labels, colors, strict=True)):
        vals = [results[label]["summary"]["regression"][target]["r2"] for target, _ in targets]
        ax.bar(x + (idx - 0.5) * width, vals, width=width, color=color, edgecolor="none", label=label)
        for xi, v in zip(x + (idx - 0.5) * width, vals, strict=True):
            ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=SLATE, rotation=90)
    _setup_axes(ax)
    ax.set_xticks(x, [label for _, label in targets])
    ax.set_ylabel("Best held-out R²")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Latent score decodability on coupled sweeps", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def coupled_geometry_chart(results: dict[str, dict]) -> Path:
    path = OUTPUT_DIR / "coupled_geometry.png"
    pairs = list(PAIR_LABELS.keys())
    x = np.arange(len(pairs))
    width = 0.18
    series = [
        ("Dense overall", NAVY, "Dense", "distance_latent_spearman"),
        ("Dense within", GOLD, "Dense", "within_variant_distance_latent_spearman_mean"),
        ("Minimal overall", TEAL, "Minimal", "distance_latent_spearman"),
        ("Minimal within", ROSE, "Minimal", "within_variant_distance_latent_spearman_mean"),
    ]

    fig, ax = plt.subplots(figsize=(11.0, 5.2), dpi=180)
    fig.patch.set_facecolor("white")
    for idx, (legend_label, color, family_label, metric_name) in enumerate(series):
        vals = []
        for pair in pairs:
            best = results[family_label]["summary"]["coupled_geometry"][pair]
            rep = best["representation"]
            layer = best["layer"]
            layer_row = next(
                row
                for row in results[family_label]["coupled_geometry"][pair][rep]
                if row["layer"] == layer
            )
            vals.append(layer_row[metric_name])
        ax.bar(x + (idx - 1.5) * width, vals, width=width, color=color, edgecolor="none", label=legend_label)
    _setup_axes(ax)
    ax.set_xticks(x, [PAIR_LABELS[pair] for pair in pairs])
    ax.set_ylabel("Best distance/latent Spearman")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Overall vs within-template coupled geometry", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def coupled_layerwise_chart(results: dict[str, dict]) -> Path:
    path = OUTPUT_DIR / "coupled_layerwise.png"
    pairs = list(PAIR_LABELS.keys())
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 11.2), dpi=180, sharex=True)
    fig.patch.set_facecolor("white")

    for ax, pair in zip(axes, pairs, strict=True):
        for label, color in (("Dense", NAVY), ("Minimal", TEAL)):
            best = results[label]["summary"]["coupled_geometry"][pair]
            rep = best["representation"]
            series = results[label]["coupled_geometry"][pair][rep]
            layers = [entry["layer"] for entry in series]
            values = [entry.get("distance_latent_spearman") for entry in series]
            within = [entry.get("within_variant_distance_latent_spearman_mean") for entry in series]
            ax.plot(layers, values, color=color, linewidth=2.2, label=f"{label} overall")
            ax.plot(layers, within, color=color, linewidth=1.6, linestyle="--", alpha=0.85, label=f"{label} within")
            ax.scatter([best["layer"]], [best["distance_latent_spearman"]], color=color, s=34, zorder=3)
        _setup_axes(ax)
        ax.set_ylabel("Spearman")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(PAIR_LABELS[pair], loc="left", fontsize=12, fontweight="bold", color=CHARCOAL)

    axes[-1].set_xlabel("Layer")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, ncol=2, loc="upper left")
    fig.suptitle("Layerwise coupled-geometry ordering", x=0.06, y=0.995, ha="left", fontsize=15, fontweight="bold", color=CHARCOAL)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def axis_fidelity_chart(results: dict[str, dict]) -> Path:
    path = OUTPUT_DIR / "axis_fidelity.png"
    pairs = list(PAIR_LABELS.keys())
    x = np.arange(len(pairs))
    width = 0.28
    colors = [NAVY, TEAL]

    fig, ax = plt.subplots(figsize=(10.2, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    for idx, (label, color) in enumerate((("Dense", NAVY), ("Minimal", TEAL))):
        vals = []
        for pair in pairs:
            best = results[label]["summary"]["coupled_geometry"][pair]
            rep = best["representation"]
            layer = best["layer"]
            layer_row = next(
                row
                for row in results[label]["coupled_geometry"][pair][rep]
                if row["layer"] == layer
            )
            axis_r2 = [value for value in layer_row.get("pc2_axis_r2", []) if value is not None]
            vals.append(float(np.mean(axis_r2)) if axis_r2 else 0.0)
        ax.bar(x + (idx - 0.5) * width, vals, width=width, color=color, edgecolor="none", label=label)
        for xi, v in zip(x + (idx - 0.5) * width, vals, strict=True):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=SLATE)
    _setup_axes(ax)
    ax.set_xticks(x, [PAIR_LABELS[pair] for pair in pairs])
    ax.set_ylabel("Mean axis R² from top-2 PCs")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Top-2 PC axis fidelity at each pair's best layer", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    ax.legend(frameon=False, ncol=2, loc="upper left")
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
        "coupled_geometry": str(coupled_geometry_chart(results)),
        "coupled_layerwise": str(coupled_layerwise_chart(results)),
        "axis_fidelity": str(axis_fidelity_chart(results)),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
