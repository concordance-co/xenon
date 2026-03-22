from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


RESULTS_PATH = Path("data/analysis_results/synthetic_market_representation/phase8_contextual_relation_v1/results.json")
ASSET_EXPORT_PATH = Path("data/interp_exports/synthetic_market_phase8_contextual_relation/synthetic_market_asset_records.parquet")
OUTPUT_DIR = Path("data/report_assets/synthetic_market_phase8_contextual_relation")

NAVY = "#16324F"
TEAL = "#2E6A69"
GOLD = "#CA9440"
ROSE = "#B56662"
SLATE = "#5E6F82"
GRID = "#D6DEE3"
CHARCOAL = "#21313F"

SCENARIO_LABELS = {
    "generic_duel_context": "Generic Duel",
    "momentum_shadow_context": "Momentum Shadow",
    "flow_shadow_context": "Flow Shadow",
    "paired_cluster_context": "Paired Cluster",
}

MODE_LABELS = {
    "style_only": "Style-only",
    "layout_only": "Layout-only",
    "roster_only": "Roster-only",
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


def rank_bucket_chart() -> Path:
    rows = pq.read_table(ASSET_EXPORT_PATH).to_pylist()
    by_example: dict[str, dict[str, dict]] = defaultdict(dict)
    scenario_by_example: dict[str, str] = {}
    for row in rows:
        by_example[str(row["example_id"])][str(row["profile_id"])] = row
        scenario_by_example[str(row["example_id"])] = str(row["family_variant"])

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for example_id, assets in by_example.items():
        if "anchor_left" not in assets or "anchor_right" not in assets:
            continue
        left_rank = int(assets["anchor_left"]["attractiveness_rank"])
        right_rank = int(assets["anchor_right"]["attractiveness_rank"])
        bucket = f"{min(left_rank, right_rank)}v{max(left_rank, right_rank)}"
        counts[scenario_by_example[example_id]][bucket] += 1

    scenarios = list(SCENARIO_LABELS.keys())
    bucket_labels = sorted({bucket for counter in counts.values() for bucket in counter})
    x = np.arange(len(scenarios))
    width = 0.18
    colors = [NAVY, TEAL, GOLD, ROSE]

    fig, ax = plt.subplots(figsize=(11.2, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    for idx, bucket in enumerate(bucket_labels):
        values = [counts[scenario].get(bucket, 0) for scenario in scenarios]
        ax.bar(
            x + (idx - (len(bucket_labels) - 1) / 2) * width,
            values,
            width=width,
            color=colors[idx % len(colors)],
            edgecolor="none",
            label=bucket,
        )
    _setup_axes(ax)
    ax.set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios])
    ax.set_ylabel("Prompt count")
    ax.set_title(
        "Phase 8 keeps the anchor pair fixed while moving it through harder contextual rank regimes",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=CHARCOAL,
    )
    ax.legend(frameon=False, title="Anchor rank bucket")
    fig.tight_layout()
    path = OUTPUT_DIR / "rank_buckets.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def relation_modes_chart(data: dict) -> Path:
    scenarios = list(SCENARIO_LABELS.keys())
    modes = ["style_only", "layout_only", "roster_only", "magnitude_only"]
    x = np.arange(len(scenarios))
    width = 0.18
    colors = [ROSE, NAVY, TEAL, GOLD]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    for idx, mode in enumerate(modes):
        margins = [data["summary"]["relation_invariance"][scenario][mode]["margin"] for scenario in scenarios]
        accs = [data["summary"]["relation_invariance"][scenario][mode]["nn_accuracy"] for scenario in scenarios]
        offset = (idx - (len(modes) - 1) / 2) * width
        axes[0].bar(x + offset, margins, width=width, color=colors[idx], edgecolor="none", label=MODE_LABELS[mode])
        axes[1].bar(x + offset, accs, width=width, color=colors[idx], edgecolor="none", label=MODE_LABELS[mode])

    _setup_axes(axes[0])
    axes[0].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[0].set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios], rotation=0)
    axes[0].set_ylabel("Best relation margin")
    axes[0].set_title(
        "Phase 8 tests whether relation identity survives contextual roster pressure",
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
        "Nearest-neighbor relation retrieval under each nuisance mode",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=CHARCOAL,
    )

    fig.tight_layout()
    path = OUTPUT_DIR / "relation_modes.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def control_chart(data: dict) -> Path:
    scenarios = list(SCENARIO_LABELS.keys())
    x = np.arange(len(scenarios))
    width = 0.32

    rank_margins = [data["summary"]["relation_rank_control"][scenario]["margin"] for scenario in scenarios]
    scale_margins = [data["summary"]["relation_scale_control"][scenario]["margin"] for scenario in scenarios]
    rank_accs = [data["summary"]["relation_rank_control"][scenario]["nn_accuracy"] for scenario in scenarios]
    scale_accs = [data["summary"]["relation_scale_control"][scenario]["nn_accuracy"] for scenario in scenarios]

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    axes[0].bar(x - width / 2, rank_margins, width=width, color=TEAL, edgecolor="none", label="Relation over rank")
    axes[0].bar(x + width / 2, scale_margins, width=width, color=GOLD, edgecolor="none", label="Relation over scale")
    _setup_axes(axes[0])
    axes[0].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[0].set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios])
    axes[0].set_ylabel("Control margin")
    axes[0].set_title(
        "Relation identity against explicit rank-bucket and scale-bucket confounds",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=CHARCOAL,
    )
    axes[0].legend(frameon=False, ncol=2, loc="upper right")

    axes[1].bar(x - width / 2, rank_accs, width=width, color=TEAL, edgecolor="none", label="Relation over rank")
    axes[1].bar(x + width / 2, scale_accs, width=width, color=GOLD, edgecolor="none", label="Relation over scale")
    _setup_axes(axes[1])
    axes[1].set_xticks(x, [SCENARIO_LABELS[s] for s in scenarios])
    axes[1].set_ylabel("Control NN accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title(
        "The comparison is what the relation representation chooses to preserve",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=CHARCOAL,
    )

    fig.tight_layout()
    path = OUTPUT_DIR / "relation_controls.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_results()
    outputs = {
        "rank_buckets": str(rank_bucket_chart()),
        "relation_modes": str(relation_modes_chart(data)),
        "relation_controls": str(control_chart(data)),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
