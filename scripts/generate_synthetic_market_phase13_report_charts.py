from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.interp.synthetic_market_representation_analysis import (  # noqa: E402
    SET_GEOMETRY_COORDS_BY_SCENARIO,
    _ordered_set_geometry_context_variants,
    _set_geometry_context_deformation_pairs,
)

REPRESENTATION_RESULTS_PATH = Path(
    "data/analysis_results/synthetic_market_representation/phase13_set_geometry_portfolio_ladder_v1"
)
TRANSFORM_RESULTS_PATH = Path(
    "data/analysis_results/synthetic_market_transform/phase13_set_geometry_portfolio_ladder_v1/phase13_explicit_transforms_v1"
)
EXPORT_ASSETS_PATH = Path(
    "data/interp_exports/synthetic_market_phase13_set_geometry_portfolio_ladder/synthetic_market_asset_records.parquet"
)
OUTPUT_DIR = Path("data/report_assets/synthetic_market_phase13_portfolio_ladder")

NAVY = "#16324F"
TEAL = "#2E6A69"
GOLD = "#CA9440"
ROSE = "#B56662"
SLATE = "#5E6F82"
GRID = "#D6DEE3"
CHARCOAL = "#21313F"
PORTFOLIO_COLORS = ["#16324F", "#275D73", "#377A7D", "#6A8F4E", "#A37A39", "#B56662"]
ASSET_COLORS = {
    "geo_alpha": NAVY,
    "geo_beta": TEAL,
    "geo_gamma": GOLD,
    "geo_delta": ROSE,
}
FAMILY_ORDER = ["identity", "orthogonal", "similarity", "diagonal", "linear"]
FAMILY_LABELS = {
    "identity": "Identity",
    "orthogonal": "Orthogonal",
    "similarity": "Similarity",
    "diagonal": "Diagonal",
    "linear": "Linear",
}
FAMILY_COLORS = {
    "identity": SLATE,
    "orthogonal": NAVY,
    "similarity": TEAL,
    "diagonal": GOLD,
    "linear": ROSE,
}


def _resolve_json_path(path: Path) -> Path:
    return path / "results.json" if path.is_dir() else path


def _load_json(path: Path) -> dict:
    return json.loads(_resolve_json_path(path).read_text())


def _setup_axes(ax: plt.Axes, *, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.65)


def _contexts(data: dict) -> list[str]:
    return _ordered_set_geometry_context_variants(list(data["set_geometry_context_realignment"].keys()))


def _context_label(context: str) -> str:
    if context == "market_only":
        return "Market"
    if context.startswith("portfolio_"):
        return f"Portfolio {context.split('_')[-1]}"
    return context.replace("_", " ").title()


def _pair_label(pair_key: str) -> str:
    left, right = pair_key.split("_to_")
    short = {
        "market_only": "M",
        "portfolio_1": "P1",
        "portfolio_2": "P2",
        "portfolio_3": "P3",
        "portfolio_4": "P4",
        "portfolio_5": "P5",
    }
    return f"{short.get(left, left)}→{short.get(right, right)}"


def coordinate_transfer_chart(data: dict) -> Path:
    contexts = _contexts(data)
    summary = data["summary"]["set_geometry_context_transfer"]
    x = np.arange(len(contexts))
    latent_x = []
    latent_y = []
    for context in contexts:
        if context == contexts[0]:
            latent_x.append(1.0)
            latent_y.append(1.0)
            continue
        key = f"{contexts[0]}_to_{context}"
        latent_x.append(summary["latent_x"][key]["r2"])
        latent_y.append(summary["latent_y"][key]["r2"])

    fig, ax = plt.subplots(figsize=(9.8, 4.6), dpi=180)
    fig.patch.set_facecolor("white")
    _setup_axes(ax)
    ax.plot(x, latent_x, color=NAVY, linewidth=2.4, marker="o", label="Latent X")
    ax.plot(x, latent_y, color=TEAL, linewidth=2.4, marker="s", label="Latent Y")
    ax.set_ylim(0.992, 1.0005)
    ax.set_xticks(x, [_context_label(context) for context in contexts])
    ax.set_ylabel("Held-out R²")
    ax.set_title(
        "The shared 4-asset market axes survive the full portfolio ladder",
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
    contexts = _contexts(data)
    ncols = 3
    nrows = int(np.ceil(len(contexts) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 4.0 + 3.7 * nrows), dpi=180, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, context, color in zip(axes_arr.flat, contexts, PORTFOLIO_COLORS, strict=False):
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
        if best is not None:
            ax.scatter([best["layer"]], [best["margin"]], color=GOLD, s=26, zorder=4)
            ax.text(best["layer"], best["margin"] + 0.004, f"L{best['layer']}", fontsize=7.5, ha="center", color=CHARCOAL)
        ax.set_title(
            f"{_context_label(context)} · row_eos",
            fontsize=11.2,
            fontweight="bold",
            color=CHARCOAL,
        )
        ax.set_xlabel("Layer")

    for ax in axes_arr.flat[len(contexts):]:
        ax.axis("off")

    axes_arr[0, 0].set_ylabel("Mean Spearman / margin")
    if nrows > 1:
        axes_arr[1, 0].set_ylabel("Mean Spearman / margin")
    axes_arr[0, 0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle(
        "Later row_eos states shift toward context-adjusted score geometry across the portfolio ladder",
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
    pairs = [f"{left}_to_{right}" for left, right in _set_geometry_context_deformation_pairs(_contexts(data))]
    x = np.arange(len(pairs))
    spearman = [data["summary"]["set_geometry_context_deformation"][pair]["margin"] for pair in pairs]
    cosine = [data["summary"]["set_geometry_context_deformation"][pair]["nn_accuracy"] for pair in pairs]
    reps = [data["summary"]["set_geometry_context_deformation"][pair]["representation"] for pair in pairs]
    layers = [data["summary"]["set_geometry_context_deformation"][pair]["layer"] for pair in pairs]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    axes[0].bar(x, spearman, color=PORTFOLIO_COLORS[: len(pairs)], edgecolor="none")
    _setup_axes(axes[0])
    axes[0].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[0].set_xticks(x, [_pair_label(pair) for pair in pairs])
    axes[0].set_ylabel("Best deformation Spearman")
    axes[0].set_title(
        "Adjacent portfolio steps induce structured geometry changes",
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

    axes[1].bar(x, cosine, color=PORTFOLIO_COLORS[: len(pairs)], edgecolor="none")
    _setup_axes(axes[1])
    axes[1].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[1].set_xticks(x, [_pair_label(pair) for pair in pairs])
    axes[1].set_ylabel("Best deformation cosine")
    axes[1].set_title(
        "The ladder may be coherent without being globally rigid",
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
    contexts = _ordered_set_geometry_context_variants(sorted({str(row["context_variant"]) for row in example_rows}))
    example_rows.sort(key=lambda row: (contexts.index(str(row["context_variant"])), int(row["row_index"])))

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

    ax0 = axes[0, 0]
    ax0.set_title("Latent layout", loc="left", fontsize=12, fontweight="bold", color=CHARCOAL)
    for profile_id, coords in latent_coords.items():
        ax0.scatter(coords[0], coords[1], s=72, color=ASSET_COLORS[profile_id], edgecolors="white", linewidth=0.8)
        ax0.text(coords[0] + 0.03, coords[1] + 0.03, profile_id.replace("geo_", "").title(), fontsize=8.5, color=CHARCOAL)
    ax0.set_xlabel("Latent x")
    ax0.set_ylabel("Latent y")

    score_axes = list(axes.flat[1 : 1 + len(contexts)])
    for ax, context in zip(score_axes, contexts, strict=True):
        score_coords = by_context[context]
        ax.set_title(_context_label(context), loc="left", fontsize=12, fontweight="bold", color=CHARCOAL)
        for profile_id, coords in score_coords.items():
            ax.scatter(coords[0], coords[1], s=72, color=ASSET_COLORS[profile_id], edgecolors="white", linewidth=0.8)
            ax.text(coords[0] + 0.015, coords[1] + 0.015, profile_id.replace("geo_", "").title(), fontsize=8.2, color=CHARCOAL)
        ax.set_xlabel("Attractiveness")
        ax.set_ylabel("Risk-adjusted")

    for ax in axes.flat[1 + len(contexts):]:
        ax.axis("off")

    fig.suptitle(
        "Representative score geometry across the portfolio ladder",
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


def family_heatmap_chart(data: dict) -> Path:
    pair_order = list(data["transform_pairs"])
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8), dpi=180)
    fig.patch.set_facecolor("white")
    for ax, state_name in zip(axes, ["early", "late"], strict=True):
        rows = []
        for pair in pair_order:
            rows.append(
                [
                    data["states"][state_name]["pair_transforms"][pair][family]["coord_r2_mean"]
                    for family in FAMILY_ORDER
                ]
            )
        arr = np.asarray(rows, dtype=np.float32)
        vmin = 0.65 if state_name == "late" else 0.99
        im = ax.imshow(arr, aspect="auto", cmap="YlGnBu", vmin=vmin, vmax=1.0)
        ax.set_xticks(range(len(FAMILY_ORDER)), [FAMILY_LABELS[family] for family in FAMILY_ORDER], rotation=30, ha="right")
        ax.set_yticks(range(len(pair_order)), [_pair_label(pair) for pair in pair_order])
        ax.set_title(
            f"{state_name.title()} state · {data['states'][state_name]['row_key']} L{data['states'][state_name]['layer']}",
            fontsize=12,
            fontweight="bold",
            color=CHARCOAL,
        )
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=7.5, color=CHARCOAL)
    fig.suptitle(
        "Portfolio-step transforms reveal which local maps stay near-rigid and which do not",
        x=0.02,
        y=1.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=CHARCOAL,
    )
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, pad=0.02)
    cbar.set_label("Coordinate reconstruction R²", color=CHARCOAL)
    fig.tight_layout()
    path = OUTPUT_DIR / "family_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def composition_chart(data: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    for ax, state_name in zip(axes, ["early", "late"], strict=True):
        state = data["states"][state_name]
        x = np.arange(len(FAMILY_ORDER))
        direct = [state["composition"][family]["direct"]["coord_r2_mean"] for family in FAMILY_ORDER]
        composed = [state["composition"][family]["composed"]["coord_r2_mean"] for family in FAMILY_ORDER]
        cosine = [state["composition"][family]["matrix_cosine"] for family in FAMILY_ORDER]
        width = 0.33

        _setup_axes(ax)
        ax.bar(x - width / 2, direct, width=width, color=NAVY, edgecolor="none", label="Direct end-to-end fit")
        ax.bar(x + width / 2, composed, width=width, color=TEAL, edgecolor="none", label="Composed adjacent maps")
        ax.set_xticks(x, [FAMILY_LABELS[family] for family in FAMILY_ORDER], rotation=25, ha="right")
        ax.set_ylabel("Coordinate R²")
        ax.set_ylim(0.55 if state_name == "late" else 0.98, 1.01)
        ax.set_title(
            f"{state_name.title()} state",
            fontsize=12,
            fontweight="bold",
            color=CHARCOAL,
        )

        ax2 = ax.twinx()
        ax2.plot(x, cosine, color=GOLD, linewidth=2.0, marker="o")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("Composed vs direct matrix cosine", color=GOLD)
        ax2.tick_params(axis="y", colors=GOLD)

    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle(
        "End-to-end portfolio behavior can be compared directly against the composition of local steps",
        x=0.02,
        y=1.01,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=CHARCOAL,
    )
    fig.tight_layout()
    path = OUTPUT_DIR / "composition.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rep = _load_json(REPRESENTATION_RESULTS_PATH)
    tr = _load_json(TRANSFORM_RESULTS_PATH)
    coordinate_transfer_chart(rep)
    context_realignment_chart(rep)
    context_deformation_chart(rep)
    representative_score_geometry_chart()
    family_heatmap_chart(tr)
    composition_chart(tr)
    print(f"Wrote Phase 13 chart assets to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
