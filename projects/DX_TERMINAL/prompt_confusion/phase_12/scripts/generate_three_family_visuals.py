from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pipelines_v2.operations.execution.common import feature_matrices, ordered_values
from pipelines_v2.storage.artifacts import ArtifactManifest, CaptureArtifact
from pipelines_v2.storage.modal import ModalVolumeStore
from projects.DX_TERMINAL.prompt_confusion.phase_12.specs.three_family_geometry_workflow import build_dataset


DEFAULT_CAPTURE_ARTIFACT_ID = "capture_1_0257379d"
DEFAULT_OUTPUT_DIR = Path(
    "projects/DX_TERMINAL/prompt_confusion/phase_12/reports/three_family_visuals"
)
LAYERS = (28, 36, 40)
FAMILIES = ("trade_size", "risk_preference", "diversification_preference")
FAMILY_COLORS = {
    "trade_size": "#3b6cff",
    "risk_preference": "#ff6b35",
    "diversification_preference": "#0f9d58",
}
CONFLICT_MARKERS = {
    False: "o",
    True: "^",
}


def _load_capture_artifact(artifact_id: str) -> CaptureArtifact:
    store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/prompt_confusion_three_family_geometry",
    )
    artifact_root = store.localize(artifact_id)
    manifest = ArtifactManifest.from_dict(json.loads((artifact_root / "manifest.json").read_text()))
    return CaptureArtifact(_manifest=manifest, store=store)


def _family_directions(
    X: np.ndarray,
    families: np.ndarray,
    conflicts: np.ndarray,
) -> dict[str, np.ndarray]:
    directions: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        family_mask = families == family
        pos = X[family_mask & (conflicts == 1)]
        neg = X[family_mask & (conflicts == 0)]
        vector = pos.mean(axis=0) - neg.mean(axis=0)
        norm = np.linalg.norm(vector)
        unit = vector / norm if norm > 0 else vector
        scores = X @ unit
        if scores[family_mask & (conflicts == 1)].mean() < scores[family_mask & (conflicts == 0)].mean():
            unit = -unit
        directions[family] = unit.astype(np.float32)
    return directions


def _subspace_components(directions: dict[str, np.ndarray]) -> np.ndarray:
    stacked = np.stack([directions[family] for family in FAMILIES], axis=0)
    _, _, vh = np.linalg.svd(stacked, full_matrices=False)
    return vh[:2].astype(np.float32)


def _shared_axis(directions: dict[str, np.ndarray]) -> np.ndarray:
    stacked = np.stack([directions[family] for family in FAMILIES], axis=0)
    axis = stacked.mean(axis=0)
    norm = np.linalg.norm(axis)
    return (axis / norm if norm > 0 else axis).astype(np.float32)


def _save_directed_scatter(
    *,
    projections_by_layer: dict[int, np.ndarray],
    families: np.ndarray,
    conflicts: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(16, 5), constrained_layout=False)
    if len(LAYERS) == 1:
        axes = [axes]
    for ax, layer in zip(axes, LAYERS, strict=False):
        Z = projections_by_layer[layer]
        for family in FAMILIES:
            family_mask = families == family
            for conflict in (False, True):
                mask = family_mask & (conflicts == int(conflict))
                ax.scatter(
                    Z[mask, 0],
                    Z[mask, 1],
                    s=18,
                    alpha=0.65,
                    c=FAMILY_COLORS[family],
                    marker=CONFLICT_MARKERS[conflict],
                    linewidths=0,
                )
        ax.set_title(f"L{layer}")
        ax.set_xlabel("Directed PC1")
        ax.set_ylabel("Directed PC2")
        ax.axhline(0.0, color="#dddddd", linewidth=0.8, zorder=0)
        ax.axvline(0.0, color="#dddddd", linewidth=0.8, zorder=0)

    family_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=FAMILY_COLORS[family],
            markersize=8,
            linestyle="None",
            label=family,
        )
        for family in FAMILIES
    ]
    conflict_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=CONFLICT_MARKERS[False],
            color="#555555",
            markersize=8,
            linestyle="None",
            label="aligned",
        ),
        plt.Line2D(
            [0],
            [0],
            marker=CONFLICT_MARKERS[True],
            color="#555555",
            markersize=8,
            linestyle="None",
            label="conflict",
        ),
    ]
    fig.subplots_adjust(bottom=0.28)
    fig.legend(
        handles=family_handles + conflict_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.2,
        handletextpad=0.6,
    )
    fig.suptitle("Three-family directed subspace projection", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_conflict_only_scatter(
    *,
    projections_by_layer: dict[int, np.ndarray],
    families: np.ndarray,
    conflicts: np.ndarray,
    output_path: Path,
) -> None:
    color_map = {0: "#8e8e8e", 1: "#111111"}
    family_markers = {
        "trade_size": "o",
        "risk_preference": "s",
        "diversification_preference": "^",
    }
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(16, 5), constrained_layout=True)
    if len(LAYERS) == 1:
        axes = [axes]
    for ax, layer in zip(axes, LAYERS, strict=False):
        Z = projections_by_layer[layer]
        for family in FAMILIES:
            family_mask = families == family
            for conflict in (0, 1):
                mask = family_mask & (conflicts == conflict)
                ax.scatter(
                    Z[mask, 0],
                    Z[mask, 1],
                    s=18,
                    alpha=0.65,
                    c=color_map[conflict],
                    marker=family_markers[family],
                    linewidths=0,
                )
        ax.set_title(f"L{layer}")
        ax.set_xlabel("Directed PC1")
        ax.set_ylabel("Directed PC2")
        ax.axhline(0.0, color="#dddddd", linewidth=0.8, zorder=0)
        ax.axvline(0.0, color="#dddddd", linewidth=0.8, zorder=0)

    handles = [
        plt.Line2D([0], [0], marker="o", color="#8e8e8e", linestyle="None", markersize=8, label="aligned"),
        plt.Line2D([0], [0], marker="o", color="#111111", linestyle="None", markersize=8, label="conflict"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#444444", linestyle="None", markersize=8, label="trade_size"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#444444", linestyle="None", markersize=8, label="risk_preference"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#444444", linestyle="None", markersize=8, label="diversification_preference"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Conflict label gradient inside the directed subspace", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_shared_axis_distribution(
    *,
    axis_scores_by_layer: dict[int, np.ndarray],
    families: np.ndarray,
    conflicts: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(LAYERS), 1, figsize=(14, 10), constrained_layout=True)
    if len(LAYERS) == 1:
        axes = [axes]
    for ax, layer in zip(axes, LAYERS, strict=False):
        scores = axis_scores_by_layer[layer]
        y_base = {
            ("trade_size", 0): 5,
            ("trade_size", 1): 4,
            ("risk_preference", 0): 3,
            ("risk_preference", 1): 2,
            ("diversification_preference", 0): 1,
            ("diversification_preference", 1): 0,
        }
        for family in FAMILIES:
            for conflict in (0, 1):
                mask = (families == family) & (conflicts == conflict)
                jitter = np.random.default_rng(0).normal(scale=0.06, size=int(mask.sum()))
                y = np.full(int(mask.sum()), y_base[(family, conflict)], dtype=np.float32) + jitter
                ax.scatter(
                    scores[mask],
                    y,
                    s=12,
                    alpha=0.45,
                    c=FAMILY_COLORS[family],
                    marker=CONFLICT_MARKERS[bool(conflict)],
                    linewidths=0,
                )
                if mask.any():
                    mean = float(scores[mask].mean())
                    ax.axvline(mean, color=FAMILY_COLORS[family], alpha=0.22, linewidth=1.2)
        ax.set_yticks([5, 4, 3, 2, 1, 0])
        ax.set_yticklabels(
            [
                "size aligned",
                "size conflict",
                "risk aligned",
                "risk conflict",
                "div aligned",
                "div conflict",
            ]
        )
        ax.set_title(f"L{layer}")
        ax.set_xlabel("Shared conflict axis score")
        ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    fig.suptitle("Family-specific score offsets on the shared conflict axis", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    *,
    output_dir: Path,
    direction_similarity: dict[int, dict[str, float]],
) -> None:
    lines = [
        "# Three-Family Visualization Summary",
        "",
        "Figures:",
        "",
        "- `directed_subspace_scatter_by_family_conflict_v2.png`",
        "- `directed_subspace_scatter_by_conflict.png`",
        "- `shared_axis_distributions.png`",
        "",
        "Representative same-capture mean-diff cosine values:",
        "",
    ]
    for layer in LAYERS:
        sims = direction_similarity[layer]
        lines.append(f"- `L{layer}`")
        for key, value in sims.items():
            lines.append(f"  - `{key}`: `{value:.4f}`")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate three-family geometry visuals.")
    parser.add_argument("--capture-artifact-id", default=DEFAULT_CAPTURE_ARTIFACT_ID)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset()
    capture = _load_capture_artifact(args.capture_artifact_id)
    matrices, example_keys = feature_matrices(
        capture.feature("residual_prompt_eos"),
        layers=LAYERS,
    )
    families = np.asarray(ordered_values(dataset.labels("target_dimension"), example_keys, label="target_dimension"), dtype=object)
    conflicts = np.asarray(
        [1 if bool(value) else 0 for value in ordered_values(dataset.labels("conflict_present"), example_keys, label="conflict_present")],
        dtype=np.int64,
    )

    projections_by_layer: dict[int, np.ndarray] = {}
    axis_scores_by_layer: dict[int, np.ndarray] = {}
    direction_similarity: dict[int, dict[str, float]] = {}

    for layer in LAYERS:
        X = matrices[layer]
        directions = _family_directions(X, families, conflicts)
        components = _subspace_components(directions)
        shared_axis = _shared_axis(directions)

        projections = X @ components.T
        axis_scores = X @ shared_axis
        if axis_scores[conflicts == 1].mean() < axis_scores[conflicts == 0].mean():
            shared_axis = -shared_axis
            axis_scores = -axis_scores

        projections_by_layer[layer] = projections
        axis_scores_by_layer[layer] = axis_scores
        direction_similarity[layer] = {
            "risk_vs_size": float(np.dot(directions["risk_preference"], directions["trade_size"])),
            "div_vs_risk": float(np.dot(directions["diversification_preference"], directions["risk_preference"])),
            "div_vs_size": float(np.dot(directions["diversification_preference"], directions["trade_size"])),
        }

    _save_directed_scatter(
        projections_by_layer=projections_by_layer,
        families=families,
        conflicts=conflicts,
        output_path=output_dir / "directed_subspace_scatter_by_family_conflict_v2.png",
    )
    _save_conflict_only_scatter(
        projections_by_layer=projections_by_layer,
        families=families,
        conflicts=conflicts,
        output_path=output_dir / "directed_subspace_scatter_by_conflict.png",
    )
    _save_shared_axis_distribution(
        axis_scores_by_layer=axis_scores_by_layer,
        families=families,
        conflicts=conflicts,
        output_path=output_dir / "shared_axis_distributions.png",
    )
    _write_summary(output_dir=output_dir, direction_similarity=direction_similarity)

    summary = {
        "capture_artifact_id": args.capture_artifact_id,
        "layers": list(LAYERS),
        "direction_similarity": direction_similarity,
        "example_count": len(example_keys),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
