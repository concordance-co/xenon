from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from safetensors.numpy import load_file
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.interp.synthetic_market_representation_analysis import SET_GEOMETRY_COORDS_BY_SCENARIO

RESULTS_PATH = Path("data/analysis_results/synthetic_market_representation/phase10_set_geometry_context_v1/results.json")
OUTPUT_DIR = Path("data/report_assets/synthetic_market_phase10_geometry_deformation")
ASSET_LABELS_PATH = Path("data/activations/synthetic_structure/phase10_set_geometry_context_v1/asset_labels.parquet")
RESIDUAL_DIR = Path("data/activations/synthetic_structure/phase10_set_geometry_context_v1/residual")

NAVY = "#16324F"
TEAL = "#2E6A69"
GOLD = "#CA9440"
ROSE = "#B56662"
SLATE = "#5E6F82"
GRID = "#D6DEE3"
CHARCOAL = "#21313F"

CONTEXT_LABELS = {
    "market_only": "Market-only",
    "low_risk": "Low risk",
    "high_risk": "High risk",
}

TRANSFER_LABELS = {
    "market_only_to_market_only": "Market -> Market",
    "market_only_to_low_risk": "Market -> Low risk",
    "market_only_to_high_risk": "Market -> High risk",
}

DEFORMATION_LABELS = {
    "market_only_to_low_risk": "Market -> Low risk",
    "market_only_to_high_risk": "Market -> High risk",
    "low_risk_to_high_risk": "Low risk -> High risk",
}

ASSET_COLORS = {
    "geo_alpha": NAVY,
    "geo_beta": TEAL,
    "geo_gamma": GOLD,
    "geo_delta": ROSE,
}

CONTEXT_MARKERS = {
    "market_only": "o",
    "low_risk": "^",
    "high_risk": "s",
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


def coordinate_transfer_chart(data: dict) -> Path:
    summary = data["summary"]["set_geometry_context_transfer"]
    transfer_keys = list(TRANSFER_LABELS)
    x = np.arange(len(transfer_keys))
    width = 0.33

    latent_x = [summary["latent_x"][key]["r2"] for key in transfer_keys]
    latent_y = [summary["latent_y"][key]["r2"] for key in transfer_keys]

    fig, ax = plt.subplots(figsize=(9.6, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.bar(x - width / 2, latent_x, width=width, color=NAVY, edgecolor="none", label="Latent X")
    ax.bar(x + width / 2, latent_y, width=width, color=TEAL, edgecolor="none", label="Latent Y")
    _setup_axes(ax)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Held-out R²")
    ax.set_xticks(x, [TRANSFER_LABELS[key] for key in transfer_keys])
    ax.set_title(
        "Base market coordinates transfer almost unchanged across context variants",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=CHARCOAL,
    )
    ax.legend(frameon=False, loc="lower left")
    for idx, key in enumerate(transfer_keys):
        best_layer = summary["latent_x"][key]["layer"]
        ax.text(
            idx,
            max(latent_x[idx], latent_y[idx]) + 0.02,
            f"L{best_layer}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=CHARCOAL,
        )
    fig.tight_layout()
    path = OUTPUT_DIR / "coordinate_transfer.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _best_realignment_rep(data: dict, context: str) -> str:
    best = data["summary"]["set_geometry_context_realignment"][context]
    return str(best["representation"])


def context_realignment_chart(data: dict) -> Path:
    contexts = ["market_only", "low_risk", "high_risk"]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), dpi=180, sharey=True)
    fig.patch.set_facecolor("white")
    for ax, context in zip(axes, contexts, strict=True):
        rep = _best_realignment_rep(data, context)
        rows = data["set_geometry_context_realignment"][context][rep]
        layers = [row["layer"] for row in rows]
        base_vals = [row["base_distance_spearman_mean"] for row in rows]
        score_vals = [row["score_distance_spearman_mean"] for row in rows]
        margin_vals = [row["score_over_base_margin"] for row in rows]
        best = data["summary"]["set_geometry_context_realignment"][context]

        _setup_axes(ax)
        ax.plot(layers, base_vals, color=SLATE, linewidth=2.0, label="Base geometry")
        ax.plot(layers, score_vals, color=ROSE, linewidth=2.0, label="Context score geometry")
        ax.plot(layers, margin_vals, color=GOLD, linewidth=1.8, linestyle="--", label="Score - base")
        ax.axhline(0.0, color=GRID, linewidth=1.0, linestyle=":")
        ax.set_title(
            f"{CONTEXT_LABELS[context]}\n({rep})",
            fontsize=11.5,
            fontweight="bold",
            color=CHARCOAL,
        )
        ax.set_xlabel("Layer")
        ax.scatter([best["layer"]], [best["margin"]], color=GOLD, s=28, zorder=4)
        ax.text(
            best["layer"],
            best["margin"] + 0.01,
            f"L{best['layer']}",
            fontsize=8,
            color=CHARCOAL,
            ha="center",
        )

    axes[0].set_ylabel("Mean Spearman / margin")
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle(
        "Later states can realign toward context-adjusted geometry without losing the base coordinate system",
        x=0.02,
        y=1.02,
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
    pairs = list(DEFORMATION_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=180)
    fig.patch.set_facecolor("white")

    x = np.arange(len(pairs))
    best_spearman = [data["summary"]["set_geometry_context_deformation"][pair]["margin"] for pair in pairs]
    best_cosine = [data["summary"]["set_geometry_context_deformation"][pair]["nn_accuracy"] for pair in pairs]
    colors = [TEAL, ROSE, GOLD]

    axes[0].bar(x, best_spearman, color=colors, edgecolor="none")
    _setup_axes(axes[0])
    axes[0].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[0].set_xticks(x, [DEFORMATION_LABELS[pair] for pair in pairs], rotation=0)
    axes[0].set_ylabel("Best deformation Spearman")
    axes[0].set_title(
        "Activation-space geometry changes track score-space changes",
        loc="left",
        fontsize=13,
        fontweight="bold",
        color=CHARCOAL,
    )

    axes[1].bar(x, best_cosine, color=colors, edgecolor="none")
    _setup_axes(axes[1])
    axes[1].axhline(0.0, color=GRID, linewidth=1.0, linestyle="--")
    axes[1].set_xticks(x, [DEFORMATION_LABELS[pair] for pair in pairs], rotation=0)
    axes[1].set_ylabel("Best deformation cosine")
    axes[1].set_title(
        "Direction of change is structured, not random",
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


def _load_example_rows(example_id: str) -> list[dict]:
    rows = pq.read_table(ASSET_LABELS_PATH).to_pylist()
    selected = [row for row in rows if str(row["example_id"]) == example_id]
    return sorted(selected, key=lambda row: (str(row["context_variant"]), int(row["row_index"])))


def _load_activation_points(example_rows: list[dict], *, key_prefix: str, layer: int) -> dict[tuple[str, str], np.ndarray]:
    by_log: dict[int, dict[str, np.ndarray]] = {}
    for log_id in sorted({int(row["log_id"]) for row in example_rows}):
        by_log[log_id] = load_file(str(RESIDUAL_DIR / f"{log_id}.safetensors"))

    points: dict[tuple[str, str], np.ndarray] = {}
    for row in example_rows:
        log_id = int(row["log_id"])
        row_index = int(row["row_index"])
        key = f"{key_prefix}_{row_index}"
        points[(str(row["profile_id"]), str(row["context_variant"]))] = by_log[log_id][key][layer].astype(np.float32)
    return points


def _pca_project(points: dict[tuple[str, str], np.ndarray]) -> dict[tuple[str, str], np.ndarray]:
    ordered_keys = sorted(points)
    X = np.stack([points[key] for key in ordered_keys], axis=0)
    proj = PCA(n_components=3).fit_transform(X)
    return {key: proj[idx] for idx, key in enumerate(ordered_keys)}


def geometry_example_chart() -> Path:
    example_id = "set_geom_01_00_00_00"
    scenario = "top_pair_cluster"
    example_rows = _load_example_rows(example_id)
    early_proj = _pca_project(_load_activation_points(example_rows, key_prefix="row_mean", layer=2))
    late_proj = _pca_project(_load_activation_points(example_rows, key_prefix="row_eos", layer=12))

    latent_coords = SET_GEOMETRY_COORDS_BY_SCENARIO[scenario]
    score_coords: dict[tuple[str, str], tuple[float, float]] = {}
    for row in example_rows:
        score_coords[(str(row["profile_id"]), str(row["context_variant"]))] = (
            float(row["attractiveness_score"]),
            float(row["risk_adjusted_score"]),
        )

    fig = plt.figure(figsize=(13.5, 10.2), dpi=180)
    fig.patch.set_facecolor("white")
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")

    # Latent 2D layout
    _setup_axes(ax1, ygrid=False)
    ax1.grid(color=GRID, linewidth=0.7, alpha=0.45)
    ax1.set_title("Latent 4-asset layout", loc="left", fontsize=12.5, fontweight="bold", color=CHARCOAL)
    for profile_id, coords in latent_coords.items():
        ax1.scatter(coords[0], coords[1], s=70, color=ASSET_COLORS[profile_id], edgecolors="white", linewidth=0.8)
        ax1.text(coords[0] + 0.03, coords[1] + 0.03, profile_id.replace("geo_", "").title(), fontsize=8.5, color=CHARCOAL)
    ax1.set_xlabel("Latent x")
    ax1.set_ylabel("Latent y")

    # Context-adjusted score geometry
    _setup_axes(ax2, ygrid=False)
    ax2.grid(color=GRID, linewidth=0.7, alpha=0.45)
    ax2.set_title("Score-space targets across contexts", loc="left", fontsize=12.5, fontweight="bold", color=CHARCOAL)
    for profile_id in latent_coords:
        context_points = []
        for context in ["market_only", "low_risk", "high_risk"]:
            x, y = score_coords[(profile_id, context)]
            context_points.append((x, y))
            ax2.scatter(
                x,
                y,
                s=66,
                color=ASSET_COLORS[profile_id],
                marker=CONTEXT_MARKERS[context],
                edgecolors="white",
                linewidth=0.8,
            )
        xs = [pt[0] for pt in context_points]
        ys = [pt[1] for pt in context_points]
        ax2.plot(xs, ys, color=ASSET_COLORS[profile_id], linewidth=1.2, alpha=0.7)
    ax2.set_xlabel("Attractiveness score")
    ax2.set_ylabel("Risk-adjusted score")

    def plot_3d(ax: plt.Axes, proj: dict[tuple[str, str], np.ndarray], *, title: str) -> None:
        ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=CHARCOAL, pad=8)
        for profile_id in latent_coords:
            line_points = []
            for context in ["market_only", "low_risk", "high_risk"]:
                vec = proj[(profile_id, context)]
                line_points.append(vec)
                ax.scatter(
                    vec[0], vec[1], vec[2],
                    s=44,
                    color=ASSET_COLORS[profile_id],
                    marker=CONTEXT_MARKERS[context],
                    depthshade=False,
                )
            line_points = np.asarray(line_points)
            ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], color=ASSET_COLORS[profile_id], linewidth=1.4, alpha=0.8)
            market_vec = proj[(profile_id, "market_only")]
            ax.text(market_vec[0], market_vec[1], market_vec[2], profile_id.replace("geo_", "").title(), fontsize=7.5, color=CHARCOAL)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.xaxis.pane.set_alpha(0.04)
        ax.yaxis.pane.set_alpha(0.04)
        ax.zaxis.pane.set_alpha(0.04)
        ax.grid(True, color=GRID, alpha=0.35)
        ax.view_init(elev=22, azim=38)

    plot_3d(ax3, early_proj, title="Early geometry: row_mean @ L2")
    plot_3d(ax4, late_proj, title="Later geometry: row_eos @ L12")

    handles = []
    for context in ["market_only", "low_risk", "high_risk"]:
        handle = plt.Line2D(
            [0], [0],
            marker=CONTEXT_MARKERS[context],
            color="none",
            markerfacecolor=SLATE,
            markeredgecolor="none",
            markersize=7,
            label=CONTEXT_LABELS[context],
        )
        handles.append(handle)
    fig.legend(handles=handles, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        "Representative market: early states preserve the shared frame, later states warp it toward context-adjusted geometry",
        x=0.02,
        y=0.98,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=CHARCOAL,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    path = OUTPUT_DIR / "geometry_example.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load_results()
    outputs = {
        "coordinate_transfer": str(coordinate_transfer_chart(data)),
        "context_realignment": str(context_realignment_chart(data)),
        "context_deformation": str(context_deformation_chart(data)),
        "geometry_example": str(geometry_example_chart()),
    }
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
