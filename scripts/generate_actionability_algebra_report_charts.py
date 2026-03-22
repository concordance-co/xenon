from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/Users/brockelmore/concordance/xenon")
OUT_DIR = ROOT / "docs" / "reports" / "assets" / "actionability_algebra"
PHASE_PATHS = {
    "v1": ROOT / "data" / "analysis_results" / "synthetic_policy_actionability_algebra_v1_results.json",
    "v2": ROOT / "data" / "analysis_results" / "synthetic_policy_actionability_algebra_v2_results.json",
    "v3": ROOT / "data" / "analysis_results" / "synthetic_policy_actionability_algebra_v3_results.json",
}
PHASE_LABELS = {"v1": "V1 Leaked", "v2": "V2 Corrected", "v3": "V3 Paraphrased"}
PHASE_COLORS = {"v1": "#CA9440", "v2": "#16324F", "v3": "#B56662"}
GRID = "#D6DEE3"
CHARCOAL = "#21313F"
SLATE = "#5E6F82"
PAPER = "#FCFBF8"
SECTIONS = [
    "active_settings_eos",
    "portfolio_eos",
    "active_strategies_eos",
    "constraints_eos",
    "last_token",
]
SECTION_LABELS = {
    "active_settings_eos": "settings",
    "portfolio_eos": "portfolio",
    "active_strategies_eos": "strategies",
    "constraints_eos": "constraints",
    "last_token": "last token",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": PAPER,
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.titleweight": "bold",
        }
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _all_results() -> dict[str, dict]:
    return {phase: _load_json(path) for phase, path in PHASE_PATHS.items()}


def _save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    path = OUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _section_best(rows: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for row in rows:
        key = row["section_key"]
        if key not in best or row["balanced_accuracy"] > best[key]["balanced_accuracy"]:
            best[key] = row
    return best


def metric_shift_chart(results: dict[str, dict]) -> Path:
    metrics = [
        ("market_best_asset_probe", "Market-best asset", "hit_at_1"),
        ("expected_action_type_classifier", "Action type", "balanced_accuracy"),
        ("permission_mode_classifier", "Permission mode", "balanced_accuracy"),
        ("policy_best_asset_classifier", "Policy-best asset", "balanced_accuracy"),
    ]
    x = np.arange(len(metrics))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    for idx, phase in enumerate(("v1", "v2", "v3")):
        vals = [results[phase]["summary"][metric][metric_key] for metric, _, metric_key in metrics]
        pos = x + (idx - 1) * width
        bars = ax.bar(pos, vals, width=width, color=PHASE_COLORS[phase], label=PHASE_LABELS[phase], edgecolor="#444444", linewidth=0.4)
        for bar, val in zip(bars, vals, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=SLATE)

    ax.set_xticks(x, [label for _, label, _ in metrics])
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel("Held-out score")
    ax.set_title("Actionability metrics across V1, V2, and V3", loc="left", fontsize=14, color=CHARCOAL)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    return _save(fig, "metric_shift.png")


def phase_design_matrix_chart() -> Path:
    features = [
        "Header leak",
        "Composed cash/age rules",
        "Fixed wording",
        "Paraphrased sections",
        "Shuffled bullets",
    ]
    matrix = np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 1, 1],
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(features)), features, rotation=18, ha="right")
    ax.set_yticks(np.arange(3), [PHASE_LABELS["v1"], PHASE_LABELS["v2"], PHASE_LABELS["v3"]])
    ax.set_title("What changed between the three phases", loc="left", fontsize=14, color=CHARCOAL)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "yes" if matrix[i, j] > 0.5 else "no", ha="center", va="center", fontsize=9, color=CHARCOAL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["absent", "present"])
    return _save(fig, "phase_design_matrix.png")


def section_heatmaps_chart(results: dict[str, dict]) -> Path:
    targets = [
        ("expected_action_type", "Action type"),
        ("permission_mode", "Permission mode"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.2))

    for ax, (target_key, title) in zip(axes, targets, strict=True):
        matrix = np.zeros((len(SECTIONS), 3), dtype=float)
        for col, phase in enumerate(("v1", "v2", "v3")):
            best = _section_best(results[phase]["tick_classification"][target_key])
            for row, section in enumerate(SECTIONS):
                matrix[row, col] = best[section]["balanced_accuracy"]

        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.2, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(3), [PHASE_LABELS[p] for p in ("v1", "v2", "v3")])
        ax.set_yticks(np.arange(len(SECTIONS)), [SECTION_LABELS[s] for s in SECTIONS])
        ax.set_title(title, fontsize=12, color=CHARCOAL)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8, color=CHARCOAL)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Best balanced accuracy by section", x=0.06, y=1.02, ha="left", fontsize=14, color=CHARCOAL, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Balanced accuracy")
    return _save(fig, "section_heatmaps.png")


def layerwise_comparison_chart(results: dict[str, dict]) -> Path:
    targets = [
        ("expected_action_type", "Action type"),
        ("permission_mode", "Permission mode"),
    ]
    colors = {
        "active_settings_eos": "#B56662",
        "portfolio_eos": "#2E6A69",
        "active_strategies_eos": "#CA9440",
        "constraints_eos": "#16324F",
        "last_token": "#5E6F82",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.6), sharex=True, sharey=True)
    for row_idx, (target_key, title) in enumerate(targets):
        for col_idx, phase in enumerate(("v2", "v3")):
            ax = axes[row_idx, col_idx]
            rows = results[phase]["tick_classification"][target_key]
            by_section: dict[str, list[dict]] = {}
            for entry in rows:
                by_section.setdefault(entry["section_key"], []).append(entry)
            for section in SECTIONS:
                section_rows = sorted(by_section[section], key=lambda r: r["layer"])
                ax.plot(
                    [r["layer"] for r in section_rows],
                    [r["balanced_accuracy"] for r in section_rows],
                    label=SECTION_LABELS[section],
                    color=colors[section],
                    linewidth=2.0,
                )
            ax.set_title(f"{title} · {PHASE_LABELS[phase]}", fontsize=11, color=CHARCOAL)
            ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
            ax.set_axisbelow(True)
            ax.set_ylim(0.2, 1.05)
            if row_idx == 1:
                ax.set_xlabel("Layer")
            if col_idx == 0:
                ax.set_ylabel("Balanced accuracy")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Layerwise section-local decode after the leak fix", x=0.06, y=1.06, ha="left", fontsize=14, color=CHARCOAL, fontweight="bold")
    return _save(fig, "layerwise_comparison.png")


def invariance_chart(results: dict[str, dict]) -> Path:
    phases = ("v1", "v2", "v3")
    x = np.arange(len(phases))
    width = 0.32
    market_vals = [results[p]["summary"]["market_best_asset_probe"]["hit_at_1"] for p in phases]
    inv_vals = [results[p]["summary"]["invariance"]["permission_top_symbol_invariance"] for p in phases]

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.bar(x - width / 2, market_vals, width=width, color="#16324F", edgecolor="#444444", linewidth=0.4, label="Market-best hit@1")
    ax.bar(x + width / 2, inv_vals, width=width, color="#2E6A69", edgecolor="#444444", linewidth=0.4, label="Permission top-symbol invariance")
    ax.set_xticks(x, [PHASE_LABELS[p] for p in phases])
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    ax.set_ylabel("Score")
    ax.set_title("What stayed stable across all three phases", loc="left", fontsize=14, color=CHARCOAL)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    for pos, val in zip(x - width / 2, market_vals, strict=True):
        ax.text(pos, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=SLATE)
    for pos, val in zip(x + width / 2, inv_vals, strict=True):
        ax.text(pos, val + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=8, color=SLATE)
    return _save(fig, "invariance.png")


def main() -> None:
    _style()
    results = _all_results()
    outputs = {
        "metric_shift": str(metric_shift_chart(results)),
        "phase_design_matrix": str(phase_design_matrix_chart()),
        "section_heatmaps": str(section_heatmaps_chart(results)),
        "layerwise_comparison": str(layerwise_comparison_chart(results)),
        "invariance": str(invariance_chart(results)),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
