from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

RESULTS_PATH = Path("data/analysis_results/research_postmarket_geometry/real_postmarket_geometry_bridge_v2_full24_results.json")
MANIFEST_PATH = Path("data/analysis_results/real_postmarket_geometry_bridge/real_postmarket_geometry_bridge_v2_full24_manifest.json")
OUTPUT_DIR = Path("data/report_assets/real_postmarket_geometry_bridge_v2")

CHARCOAL = "#21313F"
NAVY = "#16324F"
TEAL = "#2E6A69"
GOLD = "#CA9440"
ROSE = "#B56662"
SLATE = "#5E6F82"
GRID = "#D6DEE3"
SAND = "#F7F4EF"
LIGHT = "#FAF7F2"
RISK_COLOR = NAVY
AFFORDANCE_COLOR = ROSE
STATE_COLORS = {
    "market_mean": NAVY,
    "market_eos": "#275D73",
    "active_settings_eos": "#4C7B6A",
    "portfolio_eos": "#7C8E47",
    "constraints_eos": GOLD,
    "last_token": ROSE,
}
STATE_LABELS = {
    "market_mean": "Market mean",
    "market_eos": "Market EOS",
    "active_settings_eos": "Active settings EOS",
    "portfolio_eos": "Portfolio EOS",
    "constraints_eos": "Constraints EOS",
    "last_token": "Last token",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _setup_axes(ax: plt.Axes, *, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7)


def _contexts(group_name: str, data: dict) -> list[str]:
    if group_name == "risk_postmarket_geometry":
        return ["risk_1", "risk_2", "risk_3", "risk_4", "risk_5"]
    return ["market_only", "affordance_1", "affordance_2", "affordance_3", "affordance_4", "affordance_5"]


def _context_label(context: str) -> str:
    if context.startswith("risk_"):
        return f"Risk {context.split('_')[-1]}"
    if context == "market_only":
        return "Market"
    if context.startswith("affordance_"):
        return f"Aff {context.split('_')[-1]}"
    return context.replace("_", " ").title()


def _best_transfer_by_context(group: dict, group_name: str) -> list[float]:
    values: list[float] = []
    for context in _contexts(group_name, group):
        best = None
        for state_key, rows in group["coordinate_transfer"][context].items():
            for row in rows:
                value = row.get("coord_r2_mean")
                if value is None:
                    continue
                if best is None or float(value) > best:
                    best = float(value)
        values.append(best if best is not None else np.nan)
    return values


def _best_realignment_by_context(group: dict, group_name: str) -> list[float]:
    values: list[float] = []
    for context in _contexts(group_name, group):
        best = None
        for state_key, rows in group["realignment"][context].items():
            for row in rows:
                value = row.get("score_over_base_margin")
                if value is None:
                    continue
                if best is None or float(value) > best:
                    best = float(value)
        values.append(best if best is not None else np.nan)
    return values


def _mean_state_context_matrix(group: dict, group_name: str) -> tuple[list[str], list[str], np.ndarray]:
    states = list(group["state_keys"])
    contexts = _contexts(group_name, group)
    matrix = np.full((len(states), len(contexts)), np.nan, dtype=np.float32)
    for s_idx, state_key in enumerate(states):
        for c_idx, context in enumerate(contexts):
            rows = group["realignment"][context][state_key]
            vals = [float(row["score_over_base_margin"]) for row in rows if row.get("score_over_base_margin") is not None]
            if vals:
                matrix[s_idx, c_idx] = float(np.mean(vals))
    return states, contexts, matrix


def _selected_layer_curves(group: dict, context: str, state_keys: list[str]) -> dict[str, list[float]]:
    curves: dict[str, list[float]] = {}
    for state_key in state_keys:
        rows = group["realignment"][context][state_key]
        curves[state_key] = [
            float(row["score_over_base_margin"]) if row.get("score_over_base_margin") is not None else np.nan
            for row in rows
        ]
    return curves


def experiment_design_chart(results: dict, manifest: dict) -> Path:
    fig = plt.figure(figsize=(14.8, 8.0), dpi=180)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.34, wspace=0.30)

    risk_rows = [row for row in manifest["manifest_rows"] if row["experiment_group"] == "risk_postmarket_geometry"]
    afford_rows = [row for row in manifest["manifest_rows"] if row["experiment_group"] == "affordance_postmarket_geometry"]
    unique_rosters = len({tuple(row["roster_key"]) for row in manifest["manifest_rows"]})
    unique_vaults = len({row["vault_address"] for row in manifest["manifest_rows"]})

    ax_data = fig.add_subplot(gs[0, 0])
    ax_data.axis("off")
    ax_data.text(0.0, 1.02, "What data was used", fontsize=13, fontweight="bold", color=CHARCOAL, transform=ax_data.transAxes)
    ax_data.text(
        0.0,
        0.81,
        "Real HQ DX observation prompts,\nrerendered into matched risk and\naffordance ladders over the same\n6-asset rosters.",
        fontsize=9.5,
        color=SLATE,
        transform=ax_data.transAxes,
    )
    data_lines = [
        f"{manifest['base_examples']} total base examples",
        f"{manifest['prompts']} total prompts",
        f"{manifest['risk_base_examples']} risk bases / {manifest['affordance_base_examples']} affordance bases",
        f"{unique_rosters} roster families / {unique_vaults} vaults",
        "6 selected assets per base example",
    ]
    y = 0.56
    for idx, line in enumerate(data_lines, start=1):
        ax_data.text(0.02, y, f"{idx}.", fontsize=10, weight="bold", color=NAVY, transform=ax_data.transAxes)
        ax_data.text(0.10, y, line, fontsize=10, color=CHARCOAL, transform=ax_data.transAxes)
        y -= 0.10
    roster_counter = Counter(tuple(row["roster_key"]) for row in manifest["manifest_rows"])
    top_roster, top_count = roster_counter.most_common(1)[0]
    ax_data.text(
        0.02,
        0.02,
        "Most repeated roster:\n"
        + ", ".join(top_roster)
        + f"\nshared by {top_count} base examples",
        fontsize=9.2,
        color=CHARCOAL,
        transform=ax_data.transAxes,
        bbox=dict(boxstyle="round,pad=0.35", fc=SAND, ec=GRID),
    )

    ax_ladder = fig.add_subplot(gs[0, 1:])
    _setup_axes(ax_ladder, ygrid=False)
    risk_levels = np.arange(5)
    aff_levels = np.arange(6) + 6.2
    risk_strength = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    afford_strength = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)
    ax_ladder.bar(risk_levels, risk_strength, width=0.72, color=RISK_COLOR, alpha=0.9, label="Risk ladder")
    ax_ladder.bar(aff_levels, afford_strength, width=0.72, color=AFFORDANCE_COLOR, alpha=0.9, label="Affordance ladder")
    for x, yval in zip(risk_levels, risk_strength, strict=True):
        ax_ladder.text(x, yval + 0.08, f"{int(yval)}", ha="center", va="bottom", fontsize=8.3, color=CHARCOAL)
    for x, yval in zip(aff_levels, afford_strength, strict=True):
        ax_ladder.text(x, yval + 0.08, f"{int(yval)}", ha="center", va="bottom", fontsize=8.3, color=CHARCOAL)
    ax_ladder.axvline(5.2, color=GRID, linewidth=1.2)
    ax_ladder.set_xticks(
        list(risk_levels) + list(aff_levels),
        ["R1", "R2", "R3", "R4", "R5", "M", "A1", "A2", "A3", "A4", "A5"],
    )
    ax_ladder.set_ylim(0, 5.8)
    ax_ladder.set_ylabel("Context pressure")
    ax_ladder.set_title("The two matched ladders in this dataset", loc="left", fontsize=13, fontweight="bold", color=CHARCOAL)
    ax_ladder.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax_ladder.text(0.02, 0.92, "Risk keeps routes available but reweights preference.", transform=ax_ladder.transAxes, fontsize=9, color=SLATE)
    ax_ladder.text(0.55, 0.92, "Affordance progressively closes or caps routes.", transform=ax_ladder.transAxes, fontsize=9, color=SLATE)

    ax_states = fig.add_subplot(gs[1, :])
    ax_states.axis("off")
    ax_states.text(0.0, 1.02, "What is being tested", fontsize=13, fontweight="bold", color=CHARCOAL, transform=ax_states.transAxes)
    boxes = [
        ("Market block", "The same 6-asset market\nsnapshot is held fixed\nacross the ladder."),
        ("Post-market states", "Read pooled activations at\nmarket_mean, market_eos,\nsettings/portfolio/constraints EOS,\nand the last token."),
        ("Cross-example frame", "Can one shared coordinate\nsystem transfer across different\nreal examples?"),
        ("Within-example realignment", "Even if the global frame is weak,\ndoes the recovered geometry move\ntoward context-adjusted scores?"),
    ]
    x_positions = [0.00, 0.255, 0.51, 0.765]
    for x0, (title, body) in zip(x_positions, boxes, strict=True):
        rect = FancyBboxPatch((x0, 0.16), 0.22, 0.58, boxstyle="round,pad=0.012,rounding_size=0.014", facecolor=LIGHT, edgecolor=GRID, linewidth=1.1, transform=ax_states.transAxes)
        ax_states.add_patch(rect)
        ax_states.text(x0 + 0.015, 0.66, title, fontsize=10.6, fontweight="bold", color=CHARCOAL, transform=ax_states.transAxes)
        ax_states.text(x0 + 0.015, 0.56, body, fontsize=9.0, color=SLATE, transform=ax_states.transAxes, va="top")
        if x0 < x_positions[-1]:
            ax_states.annotate("", xy=(x0 + 0.247, 0.45), xytext=(x0 + 0.225, 0.45), arrowprops=dict(arrowstyle="->", lw=1.5, color=GOLD), xycoords=ax_states.transAxes)

    fig.suptitle("Real post-market bridge design and target", x=0.02, y=0.99, ha="left", fontsize=15, fontweight="bold", color=CHARCOAL)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "experiment_design.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def transfer_summary_chart(results: dict) -> Path:
    risk_vals = _best_transfer_by_context(results["groups"]["risk_postmarket_geometry"], "risk_postmarket_geometry")
    aff_vals = _best_transfer_by_context(results["groups"]["affordance_postmarket_geometry"], "affordance_postmarket_geometry")
    risk_ctx = [_context_label(c) for c in _contexts("risk_postmarket_geometry", results["groups"]["risk_postmarket_geometry"])]
    aff_ctx = [_context_label(c) for c in _contexts("affordance_postmarket_geometry", results["groups"]["affordance_postmarket_geometry"])]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    for ax, vals, labels, title, color in [
        (axes[0], risk_vals, risk_ctx, "Risk: best cross-example coordinate transfer", RISK_COLOR),
        (axes[1], aff_vals, aff_ctx, "Affordance: best cross-example coordinate transfer", AFFORDANCE_COLOR),
    ]:
        _setup_axes(ax)
        x = np.arange(len(labels))
        ax.bar(x, vals, color=color, alpha=0.92)
        ax.axhline(0.0, color=GRID, linewidth=1.0)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Best coord R²")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=CHARCOAL)
        ax.set_ylim(min(vals) - 0.08, 0.10)
        for xi, yi in zip(x, vals, strict=True):
            ax.text(xi, yi + (0.01 if yi >= 0 else -0.015), f"{yi:.3f}", ha="center", va="bottom" if yi >= 0 else "top", fontsize=8.2, color=CHARCOAL)

    fig.suptitle("There is no clean shared cross-example coordinate frame on real DX prompts", x=0.02, y=0.99, ha="left", fontsize=15, fontweight="bold", color=CHARCOAL)
    path = OUTPUT_DIR / "transfer_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def realignment_context_chart(results: dict) -> Path:
    risk_group = results["groups"]["risk_postmarket_geometry"]
    aff_group = results["groups"]["affordance_postmarket_geometry"]
    risk_contexts = _contexts("risk_postmarket_geometry", risk_group)
    aff_contexts = _contexts("affordance_postmarket_geometry", aff_group)
    risk_vals = _best_realignment_by_context(risk_group, "risk_postmarket_geometry")
    aff_vals = _best_realignment_by_context(aff_group, "affordance_postmarket_geometry")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    for ax, labels, vals, title, color in [
        (axes[0], [_context_label(c) for c in risk_contexts], risk_vals, "Risk ladder: best realignment margin by context", RISK_COLOR),
        (axes[1], [_context_label(c) for c in aff_contexts], aff_vals, "Affordance ladder: best realignment margin by context", AFFORDANCE_COLOR),
    ]:
        _setup_axes(ax)
        x = np.arange(len(labels))
        ax.plot(x, vals, color=color, marker="o", linewidth=2.2, markersize=5)
        ax.axhline(0.0, color=GRID, linewidth=1.0)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Best score-over-base margin")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=CHARCOAL)
        ymin = min(-0.05, min(vals) - 0.03)
        ymax = max(0.45, max(vals) + 0.05)
        ax.set_ylim(ymin, ymax)
        for xi, yi in zip(x, vals, strict=True):
            ax.text(xi, yi + 0.015, f"{yi:.3f}", ha="center", va="bottom", fontsize=8.2, color=CHARCOAL)

    fig.suptitle("Within-example geometry starts to move only after the market block, and affordance is much cleaner than risk", x=0.02, y=0.99, ha="left", fontsize=15, fontweight="bold", color=CHARCOAL)
    path = OUTPUT_DIR / "realignment_contexts.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def state_heatmaps_chart(results: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), dpi=180)
    fig.patch.set_facecolor("white")
    for ax, group_name, title in [
        (axes[0], "risk_postmarket_geometry", "Risk"),
        (axes[1], "affordance_postmarket_geometry", "Affordance"),
    ]:
        group = results["groups"][group_name]
        states, contexts, matrix = _mean_state_context_matrix(group, group_name)
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-0.20, vmax=0.20 if group_name == "risk_postmarket_geometry" else 0.42)
        ax.set_xticks(np.arange(len(contexts)), [_context_label(c) for c in contexts], rotation=0)
        ax.set_yticks(np.arange(len(states)), [STATE_LABELS[s] for s in states])
        ax.set_title(f"{title}: mean score-over-base margin", loc="left", fontsize=12, fontweight="bold", color=CHARCOAL)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                label = "NA" if np.isnan(value) else f"{value:.2f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=7.6, color="white" if not np.isnan(value) and abs(value) > 0.12 else CHARCOAL)
        ax.spines[:].set_visible(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.9, location="right")
    cbar.set_label("Mean score-over-base margin")
    fig.suptitle("The real context effect lives in post-market section states, not in a reusable global coordinate frame", x=0.02, y=0.99, ha="left", fontsize=15, fontweight="bold", color=CHARCOAL)
    path = OUTPUT_DIR / "state_heatmaps.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def selected_layerwise_chart(results: dict) -> Path:
    risk_group = results["groups"]["risk_postmarket_geometry"]
    aff_group = results["groups"]["affordance_postmarket_geometry"]
    risk_curves = _selected_layer_curves(risk_group, "risk_5", ["market_eos", "constraints_eos", "last_token"])
    aff_curves = _selected_layer_curves(aff_group, "affordance_5", ["market_mean", "constraints_eos", "active_settings_eos"])
    layers = np.arange(len(next(iter(risk_curves.values()))))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    for ax, curves, title in [
        (axes[0], risk_curves, "Risk 5: layerwise realignment at selected states"),
        (axes[1], aff_curves, "Affordance 5: layerwise realignment at selected states"),
    ]:
        _setup_axes(ax)
        for state_key, values in curves.items():
            ax.plot(layers, values, linewidth=2.0, color=STATE_COLORS[state_key], label=STATE_LABELS[state_key])
        ax.axhline(0.0, color=GRID, linewidth=1.0)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Score-over-base margin")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=CHARCOAL)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Risk only becomes weakly positive at the extreme end of the ladder, while affordance is positive much earlier and more strongly", x=0.02, y=0.99, ha="left", fontsize=15, fontweight="bold", color=CHARCOAL)
    path = OUTPUT_DIR / "selected_layerwise.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    results = _load_json(RESULTS_PATH)
    manifest = _load_json(MANIFEST_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_design_chart(results, manifest)
    transfer_summary_chart(results)
    realignment_context_chart(results)
    state_heatmaps_chart(results)
    selected_layerwise_chart(results)
    print(f"Wrote report assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
