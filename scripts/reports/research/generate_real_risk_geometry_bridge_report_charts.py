from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle


RESULTS_PATH = Path("data/analysis_results/research_risk_geometry/real_risk_geometry_bridge_v1/results.json")
MANIFEST_PATH = Path("data/analysis_results/real_risk_geometry_bridge/real_risk_geometry_bridge_v1_manifest.json")
OUTPUT_DIR = Path("data/report_assets/real_risk_geometry_bridge")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 170,
        }
    )


def _mean_curve(series_groups: list[list[dict]], metric_key: str) -> np.ndarray:
    curves = []
    for series in series_groups:
        vals = [row.get(metric_key) for row in series]
        curves.append([np.nan if value is None else float(value) for value in vals])
    return np.nanmean(np.asarray(curves, dtype=np.float64), axis=0)


def experiment_design_chart(summary: dict) -> Path:
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95], width_ratios=[1.0, 1.15], hspace=0.35, wspace=0.25)

    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])
    ax_bottom = fig.add_subplot(gs[1, :])

    ax_left.axis("off")
    ax_left.text(0.0, 1.02, "What Data Was Used", fontsize=14, fontweight="bold", va="bottom")
    bullets = [
        f"{summary['base_examples']} matched real base examples",
        f"{summary['prompts']} total prompts across risk_1..risk_5",
        f"{summary['top_rosters']} roster families x {summary['per_roster']} examples each",
        "4-asset slice chosen inside each larger real roster",
        "Only Asset Risk Preference is edited across variants",
    ]
    for idx, bullet in enumerate(bullets, start=1):
        ax_left.text(0.02, 0.9 - 0.12 * idx, f"{idx}.  {bullet}", fontsize=11, color="#263238")

    counts_text = (
        f"Base examples: {summary['base_examples']}\n"
        f"Prompts: {summary['prompts']}\n"
        f"Contexts: {len(summary['contexts'])}\n"
        f"Rosters: {len(summary['roster_counts'])}"
    )
    ax_left.text(
        0.02,
        0.06,
        counts_text,
        fontsize=10.5,
        color="#37474f",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f4f1eb", ec="#d3c8b8"),
    )

    contexts = ["market snapshot", "risk_1", "risk_2", "risk_3", "risk_4", "risk_5"]
    pressures = [0.0, 1, 2, 3, 4, 5]
    ax_right.set_title("What The Real Risk Ladder Means", loc="left", fontweight="bold")
    bars = ax_right.bar(range(len(contexts)), pressures, color=["#d7e3f4"] + ["#4f6d7a"] * 5, width=0.68)
    ax_right.set_xticks(range(len(contexts)))
    ax_right.set_xticklabels(["base", "R1", "R2", "R3", "R4", "R5"])
    ax_right.set_ylabel("Risk instruction level")
    ax_right.set_ylim(0, 5.8)
    for bar, label in zip(bars, contexts, strict=False):
        ax_right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12, label, ha="center", va="bottom", fontsize=9)
    ax_right.text(
        0.02,
        0.97,
        "Same market rows, same 4-asset slice,\nonly the risk preference setting changes.",
        transform=ax_right.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#455a64",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d8dde3"),
    )

    ax_bottom.axis("off")
    ax_bottom.text(0.0, 1.02, "What Is Being Tested", fontsize=14, fontweight="bold", va="bottom")
    box_y = 0.28
    box_w = 0.21
    gap = 0.04
    labels = [
        ("Real base market", "Take one real HQ observation prompt,\nthen select a stable 4-asset slice."),
        ("Row-local states", "Read row states across the matched\nrisk_1..risk_5 prompt family."),
        ("Geometry bridge", "Ask whether row-local market geometry\nmoves with the real risk ladder."),
        ("Interpretation", "If geometry stays fixed, risk is likely\napplied somewhere beyond row-local rows."),
    ]
    for idx, (title, body) in enumerate(labels):
        x0 = idx * (box_w + gap)
        rect = Rectangle((x0, box_y), box_w, 0.45, facecolor="#faf7f2", edgecolor="#d7cbbb", linewidth=1.2)
        ax_bottom.add_patch(rect)
        ax_bottom.text(x0 + 0.015, box_y + 0.34, title, fontsize=12, fontweight="bold", color="#263238")
        ax_bottom.text(x0 + 0.015, box_y + 0.18, body, fontsize=10, color="#546e7a")
        if idx < len(labels) - 1:
            arrow = FancyArrowPatch(
                (x0 + box_w, box_y + 0.225),
                (x0 + box_w + gap - 0.008, box_y + 0.225),
                arrowstyle="->",
                mutation_scale=16,
                linewidth=1.4,
                color="#c48f36",
            )
            ax_bottom.add_patch(arrow)

    out = OUTPUT_DIR / "experiment_design.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def coordinate_transfer_chart(results: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    colors = {"row_mean": "#2f6c8f", "row_eos": "#b55d3d"}
    axis_labels = {"base_x": "Base X Coordinate", "base_y": "Base Y Coordinate"}

    for ax, axis_key in zip(axes, ("base_x", "base_y"), strict=False):
        for row_key in ("row_mean", "row_eos"):
            groups = [per_row[row_key] for per_row in results["context_transfer"][axis_key].values()]
            curve = _mean_curve(groups, "r2")
            ax.plot(results["layers"], curve, label=row_key, linewidth=2.2, color=colors[row_key])
            best_idx = int(np.nanargmax(curve))
            ax.scatter(results["layers"][best_idx], curve[best_idx], color=colors[row_key], s=28, zorder=3)
        ax.set_title(axis_labels[axis_key], loc="left", fontweight="bold")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean transfer R²")
        ax.axhline(0.0, color="#cfd8dc", linewidth=1.0)
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Cross-Example Coordinate Transfer Across The Real Risk Ladder", fontsize=15, fontweight="bold", y=1.02)
    out = OUTPUT_DIR / "coordinate_transfer.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def realignment_chart(results: dict) -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    colors = {"row_mean": "#2f6c8f", "row_eos": "#b55d3d"}
    for row_key in ("row_mean", "row_eos"):
        groups = [per_row[row_key] for per_row in results["context_realignment"].values()]
        curve = _mean_curve(groups, "score_over_base_margin")
        ax.plot(results["layers"], curve, label=row_key, linewidth=2.2, color=colors[row_key])
        best_idx = int(np.nanargmax(curve))
        ax.scatter(results["layers"][best_idx], curve[best_idx], color=colors[row_key], s=28, zorder=3)
    ax.axhline(0.0, color="#455a64", linestyle="--", linewidth=1.1)
    ax.set_title("Realignment Margin By Layer", loc="left", fontweight="bold")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean score-over-base margin")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.01,
        0.05,
        "Positive would mean the row-local geometry moves toward risk-adjusted score geometry.\nIn the real bridge it stays slightly negative at every layer.",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#455a64",
        bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#d8dde3"),
    )
    out = OUTPUT_DIR / "realignment_margin.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def invariance_chart(results: dict) -> Path:
    state = results["states"]["early"]
    pairs = list(state["pair_transforms"].keys())
    coord_scores = [state["pair_transforms"][pair]["identity"]["coord_r2_mean"] for pair in pairs]
    score_spearman = [state["pair_transforms"][pair]["identity"]["score_distance_spearman_mean"] for pair in pairs]

    fig, ax = plt.subplots(figsize=(11.4, 4.6))
    x = np.arange(len(pairs))
    width = 0.34
    ax.bar(x - width / 2, coord_scores, width=width, color="#2e7d32", label="Decoded geometry self-fit (identity)")
    ax.bar(x + width / 2, score_spearman, width=width, color="#c48f36", label="Risk-score distance alignment")
    ax.set_xticks(x)
    ax.set_xticklabels([pair.replace("risk_", "R").replace("_to_", "→") for pair in pairs])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Metric value")
    ax.set_title("At The Selected Row-Local State, Risk Steps Behave Like Identity Maps", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22, linewidth=0.6)
    ax.legend(frameon=False, loc="upper left")
    for xpos, score in zip(x - width / 2, coord_scores, strict=False):
        ax.text(xpos, score + 0.02, f"{score:.2f}", ha="center", va="bottom", fontsize=9)
    for xpos, score in zip(x + width / 2, score_spearman, strict=False):
        ax.text(xpos, score + 0.02, f"{score:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(
        0.99,
        0.03,
        "Decoded row-local geometry stays fixed across the full real risk ladder.\nThe score geometry implied by the edited setting does change, but the row-local state does not follow it.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#455a64",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d8dde3"),
    )
    out = OUTPUT_DIR / "invariance_identity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    _setup()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = _load_json(RESULTS_PATH)
    summary = _load_json(MANIFEST_PATH)
    experiment_design_chart(summary)
    coordinate_transfer_chart(results)
    realignment_chart(results)
    invariance_chart(results)
    print(f"Wrote real risk geometry bridge chart assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
