from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.colors import LinearSegmentedColormap


TARGET_LABELS = {
    "is_target_asset": "Target Asset",
    "is_buy_target": "Buy Target",
    "is_sell_target": "Sell Target",
}

REPRESENTATION_LABELS = {
    "row_mean": "row_mean",
    "row_eos": "row_eos",
    "row_mean+active_settings_eos": "+ settings",
    "row_mean+portfolio_eos": "+ portfolio",
    "row_mean+constraints_eos": "+ constraints",
    "row_mean+prev_decisions_eos": "+ prev decisions",
    "row_mean+last_token": "+ last token",
}

PALETTE = {
    "ink": "#16202A",
    "muted": "#5A6B7D",
    "navy": "#16324F",
    "teal": "#2D6A6A",
    "sand": "#F4EEE2",
    "mist": "#EEF4F6",
    "rose": "#D97777",
    "line": "#D6DEE3",
    "gold": "#D9A441",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate chart assets for the decision-structure findings report."
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("data/analysis_results/decision_structure"),
        help="Path to the decision structure JSON result file.",
    )
    parser.add_argument(
        "--tick-labels-path",
        type=Path,
        default=Path("data/activations/decision_structure/tick_labels.parquet"),
        help="Path to pooled tick labels parquet.",
    )
    parser.add_argument(
        "--asset-labels-path",
        type=Path,
        default=Path("data/activations/decision_structure/asset_labels.parquet"),
        help="Path to pooled asset labels parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/report_assets/decision_structure"),
        help="Directory where PNG figures should be written.",
    )
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PALETTE["line"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["navy"],
            "axes.grid": True,
            "grid.color": PALETTE["line"],
            "grid.alpha": 0.45,
            "grid.linewidth": 0.7,
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "legend.frameon": False,
        }
    )


def load_results(path: Path) -> dict:
    return json.loads(path.read_text())


def series_for(results: dict, target: str, representation: str) -> np.ndarray:
    layer_to_value = {
        int(entry["layer"]): float(entry["auroc"])
        for entry in results["targets"][target][representation]
    }
    return np.asarray([layer_to_value[layer] for layer in results["layers"]], dtype=float)


def count_column_values(path: Path, column: str) -> Counter:
    table = pq.read_table(path, columns=[column])
    values = table.to_pydict()[column]
    return Counter(values)


def load_decision_mix(tick_labels_path: Path) -> dict[str, int]:
    counts = count_column_values(tick_labels_path, "trade_side")
    return {
        "Observe": counts.get(None, 0),
        "Buy": counts.get("buy", 0),
        "Sell": counts.get("sell", 0),
    }


def load_target_symbol_mix(asset_labels_path: Path) -> dict[str, int]:
    table = pq.read_table(asset_labels_path, columns=["symbol", "is_target_asset"])
    data = table.to_pydict()
    counts: Counter[str] = Counter()
    for symbol, is_target in zip(data["symbol"], data["is_target_asset"], strict=True):
        if is_target:
            counts[str(symbol)] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_composition(
    decision_mix: dict[str, int], target_symbol_mix: dict[str, int], output_dir: Path
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.2, 3.9),
        gridspec_kw={"width_ratios": [1.05, 1]},
    )

    ax = axes[0]
    labels = list(decision_mix.keys())
    values = [decision_mix[label] for label in labels]
    colors = [PALETTE["mist"], PALETTE["teal"], PALETTE["rose"]]
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor=PALETTE["line"], height=0.58)
    ax.set_yticks(y, labels)
    ax.set_title("Tick action mix")
    ax.set_xlabel("Count")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(values) + 10)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + 1.1,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            ha="left",
            color=PALETTE["ink"],
            fontsize=9,
            weight="bold",
        )

    ax = axes[1]
    symbols = list(target_symbol_mix.keys())
    values = [target_symbol_mix[symbol] for symbol in symbols]
    y = np.arange(len(symbols))
    bars = ax.barh(y, values, color=PALETTE["navy"], edgecolor=PALETTE["line"], height=0.58)
    ax.set_yticks(y, symbols)
    ax.set_title("Target-asset symbol mix")
    ax.set_xlabel("Positive rows")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(values) + 5 if values else 1)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + 0.6,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            ha="left",
            color=PALETTE["ink"],
            fontsize=9,
            weight="bold",
        )

    fig.suptitle("Current sample composition", fontsize=13, fontweight="bold", color=PALETTE["navy"])
    fig.tight_layout()
    save(fig, output_dir / "dataset_composition.png")


def plot_best_pre_post(results: dict, output_dir: Path) -> None:
    summary = results["summary"]
    targets = list(TARGET_LABELS)
    pre_scores = [summary[target]["best_pre"]["auroc"] for target in targets]
    post_scores = [summary[target]["best_post"]["auroc"] for target in targets]
    deltas = [summary[target]["best_post_minus_best_pre"] for target in targets]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(targets))
    width = 0.34
    pre_bars = ax.bar(
        x - width / 2,
        pre_scores,
        width,
        label="Best pre-row state",
        color=PALETTE["navy"],
        edgecolor=PALETTE["line"],
    )
    post_bars = ax.bar(
        x + width / 2,
        post_scores,
        width,
        label="Best post state",
        color=PALETTE["rose"],
        edgecolor=PALETTE["line"],
    )
    ax.set_xticks(x, [TARGET_LABELS[target] for target in targets])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Best AUROC")
    ax.set_title("Best pre vs. post AUROC by target")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")

    for bars in (pre_bars, post_bars):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=PALETTE["ink"],
            )
    for idx, delta in enumerate(deltas):
        ax.text(
            x[idx],
            max(pre_scores[idx], post_scores[idx]) + 0.045,
            f"{delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=PALETTE["muted"],
            fontweight="bold",
        )

    fig.tight_layout()
    save(fig, output_dir / "best_pre_post.png")


def plot_layerwise_auroc(results: dict, output_dir: Path) -> None:
    layers = np.asarray(results["layers"], dtype=int)
    summary = results["summary"]
    targets = list(TARGET_LABELS)

    fig, axes = plt.subplots(len(targets), 1, figsize=(9.6, 9.4), sharex=True, sharey=True)

    for ax, target in zip(axes, targets, strict=True):
        row_mean = series_for(results, target, "row_mean")
        row_eos = series_for(results, target, "row_eos")
        best_post_rep = summary[target]["best_post"]["representation"]
        best_post = series_for(results, target, best_post_rep)

        ax.plot(layers, row_mean, color=PALETTE["navy"], linewidth=2.2, label="row_mean")
        ax.plot(layers, row_eos, color=PALETTE["teal"], linewidth=2.0, linestyle="--", label="row_eos")
        ax.plot(
            layers,
            best_post,
            color=PALETTE["rose"],
            linewidth=2.0,
            label=f"best post ({REPRESENTATION_LABELS[best_post_rep]})",
        )

        best_pre = summary[target]["best_pre"]
        best_post_summary = summary[target]["best_post"]
        ax.scatter(
            best_pre["layer"],
            best_pre["auroc"],
            s=36,
            color=PALETTE["gold"],
            zorder=4,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.scatter(
            best_post_summary["layer"],
            best_post_summary["auroc"],
            s=36,
            color=PALETTE["rose"],
            zorder=4,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_title(TARGET_LABELS[target], loc="left", fontsize=12, pad=8, fontweight="bold")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.5, 1.0)
        ax.set_xlim(layers.min(), layers.max())
        ax.text(
            0.99,
            0.06,
            (
                f"best pre: {best_pre['representation']} @ L{best_pre['layer']} = {best_pre['auroc']:.3f}\n"
                f"best post: {REPRESENTATION_LABELS[best_post_rep]} @ L{best_post_summary['layer']} = {best_post_summary['auroc']:.3f}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.2,
            color=PALETTE["muted"],
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#FBFDFE",
                "edgecolor": PALETTE["line"],
                "linewidth": 0.8,
            },
        )

    axes[-1].set_xlabel("Layer")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Layerwise target-binding readout", fontsize=14, fontweight="bold", color=PALETTE["navy"], y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, output_dir / "layerwise_auroc.png")


def plot_representation_heatmap(results: dict, output_dir: Path) -> None:
    targets = list(TARGET_LABELS)
    representations = list(REPRESENTATION_LABELS)
    matrix = np.asarray(
        [
            [max(float(entry["auroc"]) for entry in results["targets"][target][representation]) for target in targets]
            for representation in representations
        ],
        dtype=float,
    )

    cmap = LinearSegmentedColormap.from_list(
        "xenon_report",
        [PALETTE["mist"], PALETTE["sand"], "#F1C8C8", PALETTE["rose"], PALETTE["navy"]],
    )

    fig, ax = plt.subplots(figsize=(8.9, 4.8))
    im = ax.imshow(matrix, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(targets)), [TARGET_LABELS[target] for target in targets])
    ax.set_yticks(np.arange(len(representations)), [REPRESENTATION_LABELS[rep] for rep in representations])
    ax.set_title("Max AUROC by representation and target", fontsize=13, fontweight="bold", pad=10)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text_color = "white" if value >= 0.82 else PALETTE["ink"]
            ax.text(col, row, f"{value:.3f}", ha="center", va="center", fontsize=8.5, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Max AUROC")
    fig.tight_layout()
    save(fig, output_dir / "representation_heatmap.png")


def main() -> None:
    args = parse_args()
    setup_style()
    ensure_output_dir(args.output_dir)

    results = load_results(args.results_path)
    decision_mix = load_decision_mix(args.tick_labels_path)
    target_symbol_mix = load_target_symbol_mix(args.asset_labels_path)

    plot_dataset_composition(decision_mix, target_symbol_mix, args.output_dir)
    plot_best_pre_post(results, args.output_dir)
    plot_layerwise_auroc(results, args.output_dir)
    plot_representation_heatmap(results, args.output_dir)

    print(f"Wrote chart assets to {args.output_dir}")


if __name__ == "__main__":
    main()
