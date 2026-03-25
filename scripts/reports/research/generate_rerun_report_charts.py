from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


RESULTS_DIR = Path("data/analysis_results/research_rerun_kickoff_v2")
OUTPUT_DIR = Path("data/report_assets/research_rerun_kickoff_v2")

NAVY = "#16324F"
TEAL = "#2E6A69"
GOLD = "#CA9440"
ROSE = "#B56662"
SLATE = "#5E6F82"
GRID = "#D6DEE3"
CHARCOAL = "#21313F"
MIST = "#EAF2F2"
CREAM = "#F6EFE3"
SAND = "#D9C3A5"


def _load_json() -> dict:
    return json.loads((RESULTS_DIR / "results.json").read_text())


def _load_rows(name: str) -> list[dict]:
    return pq.read_table(RESULTS_DIR / name).to_pylist()


def _setup_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)


def experiment_design_chart(prompt_rows: list[dict]) -> Path:
    path = OUTPUT_DIR / "experiment_design.png"

    group_variant_counts: dict[str, Counter[str]] = {
        "blocked_valence": Counter(),
        "settings_twist": Counter(),
    }
    for row in prompt_rows:
        group_variant_counts[row["experiment_group"]][row["variant"]] += 1

    labels = ["Blocked valence", "Settings twist"]
    x = np.arange(len(labels))
    width = 0.58

    blocked_bottom = np.zeros(len(labels))
    stacks = [
        ("original", NAVY, "Original"),
        ("clear_strategies", ROSE, "Clear strategies"),
        ("settings_all1", GOLD, "All sliders = 1"),
        ("settings_all5", TEAL, "All sliders = 5"),
    ]

    fig, ax = plt.subplots(figsize=(9.4, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    for key, color, label in stacks:
        vals = [
            group_variant_counts["blocked_valence"].get(key, 0),
            group_variant_counts["settings_twist"].get(key, 0),
        ]
        ax.bar(x, vals, bottom=blocked_bottom, color=color, edgecolor="none", width=width, label=label)
        blocked_bottom += np.array(vals)

    _setup_axes(ax, grid_axis="y")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Captured prompt variants")
    ax.set_title("Kickoff rerun design", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    ax.set_ylim(0, blocked_bottom.max() * 1.16)
    for xi, total in zip(x, blocked_bottom, strict=True):
        ax.text(xi, total + blocked_bottom.max() * 0.02, f"{int(total)}", ha="center", va="bottom", fontsize=9, color=SLATE)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def blocked_valence_chart(summary: dict, blocked_rows: list[dict]) -> Path:
    path = OUTPUT_DIR / "blocked_valence.png"
    blocked = summary["blocked_valence"]
    labels = ["Neutral", "Bullish", "Bearish"]
    original = [
        blocked["original_valence_counts"].get("neutral", 0),
        blocked["original_valence_counts"].get("bullish", 0),
        blocked["original_valence_counts"].get("bearish", 0),
    ]
    cleared = [
        blocked["clear_valence_counts"].get("neutral", 0),
        blocked["clear_valence_counts"].get("bullish", 0),
        blocked["clear_valence_counts"].get("bearish", 0),
    ]

    block_reason_counts = Counter(row["block_reason"] for row in blocked_rows)
    reason_labels = [label.replace("_", " ") for label, _ in block_reason_counts.most_common()]
    reason_values = [value for _, value in block_reason_counts.most_common()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    x = np.arange(len(labels))
    width = 0.33
    ax1.bar(x - width / 2, original, width=width, color=NAVY, edgecolor="none", label="Original")
    ax1.bar(x + width / 2, cleared, width=width, color=ROSE, edgecolor="none", label="Clear strategies")
    _setup_axes(ax1, grid_axis="y")
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Blocked pairs")
    ax1.set_title("Blocked-valence labels stay mostly neutral", loc="left", fontsize=13, fontweight="bold", color=CHARCOAL)
    ax1.legend(frameon=False, loc="upper right")
    for xi, orig, clr in zip(x, original, cleared, strict=True):
        ax1.text(xi - width / 2, orig + 0.5, str(orig), ha="center", va="bottom", fontsize=8, color=SLATE)
        ax1.text(xi + width / 2, clr + 0.5, str(clr), ha="center", va="bottom", fontsize=8, color=SLATE)

    y = np.arange(len(reason_labels))
    ax2.barh(y, reason_values, color=ROSE, edgecolor="none", height=0.62)
    _setup_axes(ax2, grid_axis="x")
    ax2.set_yticks(y, reason_labels)
    ax2.set_xlabel("Pairs")
    ax2.set_title("The blocked pool is still dominated by strategy cases", loc="left", fontsize=13, fontweight="bold", color=CHARCOAL)
    for yi, val in zip(y, reason_values, strict=True):
        ax2.text(val + 0.35, yi, str(val), va="center", ha="left", fontsize=8, color=SLATE)
    ax2.invert_yaxis()

    fig.text(
        0.5,
        0.01,
        "Only 3 of 34 cleared cases reveal directional valence, and all 3 keep the same top buy/sell symbols as the original prompt.",
        ha="center",
        fontsize=8.5,
        color=SLATE,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def settings_shift_chart(summary: dict, triplets: list[dict]) -> Path:
    path = OUTPUT_DIR / "settings_shift.png"
    settings = summary["settings_twist"]
    labels = ["Neutral", "Bullish", "Bearish"]
    original = [
        settings["original_valence_counts"].get("neutral", 0),
        settings["original_valence_counts"].get("bullish", 0),
        settings["original_valence_counts"].get("bearish", 0),
    ]
    all1 = [
        settings["all1_valence_counts"].get("neutral", 0),
        settings["all1_valence_counts"].get("bullish", 0),
        settings["all1_valence_counts"].get("bearish", 0),
    ]
    all5 = [
        settings["all5_valence_counts"].get("neutral", 0),
        settings["all5_valence_counts"].get("bullish", 0),
        settings["all5_valence_counts"].get("bearish", 0),
    ]

    deltas = [row["delta_trade_probability_all5_minus_all1"] for row in triplets]
    buckets = [
        ("<= -0.5", sum(delta <= -0.5 for delta in deltas)),
        ("-0.5 .. 0", sum(-0.5 < delta < -1e-9 for delta in deltas)),
        ("~ 0", sum(abs(delta) <= 1e-9 for delta in deltas)),
        ("0 .. 0.5", sum(1e-9 < delta < 0.5 for delta in deltas)),
        (">= 0.5", sum(delta >= 0.5 for delta in deltas)),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    x = np.arange(len(labels))
    width = 0.24
    ax1.bar(x - width, original, width=width, color=NAVY, edgecolor="none", label="Original")
    ax1.bar(x, all1, width=width, color=GOLD, edgecolor="none", label="All sliders = 1")
    ax1.bar(x + width, all5, width=width, color=TEAL, edgecolor="none", label="All sliders = 5")
    _setup_axes(ax1, grid_axis="y")
    ax1.set_xticks(x, labels)
    ax1.set_ylabel("Triplets")
    ax1.set_title("Settings change valence on a minority subset", loc="left", fontsize=13, fontweight="bold", color=CHARCOAL)
    ax1.legend(frameon=False, loc="upper right")

    bx = np.arange(len(buckets))
    bvals = [val for _, val in buckets]
    bcols = [ROSE, SAND, SLATE, CREAM, TEAL]
    ax2.bar(bx, bvals, color=bcols, edgecolor="none", width=0.64)
    _setup_axes(ax2, grid_axis="y")
    ax2.set_xticks(bx, [label for label, _ in buckets])
    ax2.set_ylabel("Triplets")
    ax2.set_title("Most settings pairs stay near zero, with a small set of hard flips", loc="left", fontsize=13, fontweight="bold", color=CHARCOAL)
    ax2.set_ylim(0, max(bvals) * 1.18)
    for xi, val in zip(bx, bvals, strict=True):
        ax2.text(xi, val + max(bvals) * 0.02, str(val), ha="center", va="bottom", fontsize=8, color=SLATE)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def settings_cohort_effects_chart(triplets: list[dict]) -> Path:
    path = OUTPUT_DIR / "settings_cohort_effects.png"
    cohort_labels = ["policy_tension_observe", "buy", "sell"]
    label_map = {
        "policy_tension_observe": "Policy tension",
        "buy": "Buy",
        "sell": "Sell",
    }

    strong_counts = []
    flip_counts = []
    for cohort in cohort_labels:
        rows = [row for row in triplets if row["cohort_label"] == cohort]
        strong_counts.append(sum(abs(row["delta_trade_probability_all5_minus_all1"]) > 0.5 for row in rows))
        flip_counts.append(sum(row["all1_predicted_valence"] != row["all5_predicted_valence"] for row in rows))

    x = np.arange(len(cohort_labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.bar(x - width / 2, strong_counts, width=width, color=TEAL, edgecolor="none", label="Strong |Δtrade prob| > 0.5")
    ax.bar(x + width / 2, flip_counts, width=width, color=NAVY, edgecolor="none", label="Valence flips")
    _setup_axes(ax, grid_axis="y")
    ax.set_xticks(x, [label_map[key] for key in cohort_labels])
    ax.set_ylabel("Triplets")
    ax.set_title("Settings sensitivity is concentrated in policy-tension and buy cohorts", loc="left", fontsize=14, fontweight="bold", color=CHARCOAL)
    ax.legend(frameon=False, loc="upper right")
    ymax = max(max(strong_counts), max(flip_counts))
    ax.set_ylim(0, ymax * 1.22)
    for xi, sval, fval in zip(x, strong_counts, flip_counts, strict=True):
        ax.text(xi - width / 2, sval + ymax * 0.03, str(sval), ha="center", va="bottom", fontsize=8, color=SLATE)
        ax.text(xi + width / 2, fval + ymax * 0.03, str(fval), ha="center", va="bottom", fontsize=8, color=SLATE)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def settings_layerwise_chart(layer_rows: list[dict]) -> Path:
    path = OUTPUT_DIR / "settings_layerwise.png"
    layers = [row["layer"] for row in layer_rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    ax1.plot(layers, [row["row_mean_cka_original_all1"] for row in layer_rows], color=NAVY, linewidth=2.0, label="Row mean: original vs all1")
    ax1.plot(layers, [row["row_mean_cka_original_all5"] for row in layer_rows], color=TEAL, linewidth=2.0, label="Row mean: original vs all5")
    ax1.plot(layers, [row["last_token_cka_all1_all5"] for row in layer_rows], color=GOLD, linewidth=2.0, label="Last token: all1 vs all5")
    ax1.plot(layers, [row["active_settings_cka_all1_all5"] for row in layer_rows], color=ROSE, linewidth=2.0, label="Settings section: all1 vs all5")
    _setup_axes(ax1, grid_axis="y")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("CKA")
    ax1.set_ylim(0.7, 1.02)
    ax1.set_title("Row states stay fixed while downstream states move", loc="left", fontsize=13, fontweight="bold", color=CHARCOAL)
    ax1.legend(frameon=False, fontsize=8, loc="lower right")

    ax2.plot(layers, [row["last_token_parallel_fraction_mean"] for row in layer_rows], color=GOLD, linewidth=2.2, label="Last token parallel fraction")
    ax2.plot(layers, [row["active_settings_parallel_fraction_mean"] for row in layer_rows], color=ROSE, linewidth=2.2, label="Settings section parallel fraction")
    _setup_axes(ax2, grid_axis="y")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Mean parallel fraction")
    ax2.set_ylim(0.0, 0.09)
    ax2.set_title("Settings changes are not mostly simple parallel rescaling", loc="left", fontsize=13, fontweight="bold", color=CHARCOAL)
    ax2.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = _load_json()
    blocked_rows = _load_rows("blocked_pairs.parquet")
    triplets = _load_rows("settings_triplets.parquet")
    layer_rows = _load_rows("settings_layer_metrics.parquet")
    prompt_rows = _load_rows("prompt_scores.parquet")

    outputs = {
        "experiment_design": str(experiment_design_chart(prompt_rows)),
        "blocked_valence": str(blocked_valence_chart(results["summary"], blocked_rows)),
        "settings_shift": str(settings_shift_chart(results["summary"], triplets)),
        "settings_cohort_effects": str(settings_cohort_effects_chart(triplets)),
        "settings_layerwise": str(settings_layerwise_chart(layer_rows)),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
