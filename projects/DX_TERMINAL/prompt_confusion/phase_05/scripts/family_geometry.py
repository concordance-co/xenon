"""Phase 05 Directions 2a + 2c: Branching geometry.

2a: PCA on conflict-only and full-dataset residual activations, RMS-normalized.
2c: 4-class LDA on conflict-only residual activations, family as label.

Produces plots and a summary.json. LDA results are visualization-quality
only -- 36 rows per class is near the floor for stable LDA.

Usage:
    uv run --extra interp --extra modal modal run \\
        projects/DX_TERMINAL/prompt_confusion/phase_05/scripts/family_geometry.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "xenon-prompt-confusion-phase5-family-geometry"

DEFAULT_CAPTURE_RUN_ID = "16474bceae4e"
DEFAULT_ACTIVATIONS_SUBDIR = f"workflows/conflict_probe_v3/{DEFAULT_CAPTURE_RUN_ID}"
DEFAULT_OUTPUT_SUBDIR = "prompt_confusion/phase_05/family_geometry"
DEFAULT_BASE_RELATION = "workflow_dataset_conflict_probe_v3_v1"

# Focus layers for geometry. Layer 36 is Phase 04 detection peak.
# Layers 24 and 28 are where arbitration-related structure may emerge.
DEFAULT_LAYERS = [20, 24, 28, 36]
FAMILY_LABELS = (
    "activity_force_observe",
    "activity_force_trade",
    "trade_size_force_large",
    "trade_size_force_small",
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("xenon-data", create_if_missing=True)
neon_secret = modal.Secret.from_name("xenon-neon")

base_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("matplotlib", "numpy", "pyarrow", "psycopg[binary]", "safetensors", "scikit-learn")
    .add_local_python_source("pipelines")
)


def _load_rows(base_relation: str) -> list[dict[str, Any]]:
    from pipelines.db import connect_neon, ensure_schema

    sql = f"""
    SELECT
        log_id,
        strategy_family,
        conflict_present,
        matched_pair_id
    FROM {base_relation}
    ORDER BY log_id
    """
    with connect_neon(autocommit=True) as conn:
        ensure_schema(conn)
        rows = conn.execute(sql).fetchall()
    return [dict(row) for row in rows]


def _load_compact_features(
    activations_dir: Path,
    data_source: str,
    layer: int,
) -> dict[int, Any]:
    from safetensors import safe_open

    path = activations_dir / "compact" / f"{data_source}_prompt_eos_layer{layer}.safetensors"
    with safe_open(str(path), framework="numpy") as f:
        features = f.get_tensor("features")
        log_ids = f.get_tensor("log_ids")
    return {int(lid): features[i] for i, lid in enumerate(log_ids)}


def _rms_normalize(X: Any) -> Any:
    """Row-wise RMS normalization. Makes PCA norm-invariant."""
    import numpy as np

    X = np.asarray(X, dtype=np.float32)
    rms = np.sqrt(np.mean(X ** 2, axis=1, keepdims=True))
    rms = np.maximum(rms, 1e-8)
    return X / rms


def _run_pca(X: Any, n_components: int = 2) -> tuple[Any, Any]:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(X)
    return projected, pca.explained_variance_ratio_


def _run_lda(X: Any, y: Any, n_components: int = 2) -> tuple[Any, dict[str, Any]]:
    import numpy as np
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    n_classes = len(np.unique(y))
    max_components = min(n_components, n_classes - 1, X.shape[1])
    lda = LinearDiscriminantAnalysis(n_components=max_components)
    projected = lda.fit_transform(X, y)
    return projected, {
        "n_components_used": int(max_components),
        "explained_variance_ratio": [float(v) for v in lda.explained_variance_ratio_.tolist()]
        if hasattr(lda, "explained_variance_ratio_") else None,
    }


def _plot_scatter(
    projected: Any,
    labels: list[str],
    *,
    title: str,
    output_path: Path,
    color_by: str = "family",
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, color in zip(unique_labels, colors):
        mask = np.array([l == label for l in labels])
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            c=[color],
            label=label,
            alpha=0.7,
            s=40,
            edgecolors="white",
            linewidths=0.5,
        )
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


@app.function(
    volumes={"/data": data_volume},
    image=base_image,
    timeout=1800,
    cpu=4,
    secrets=[neon_secret],
)
def run_family_geometry(
    *,
    activations_subdir: str = DEFAULT_ACTIVATIONS_SUBDIR,
    output_subdir: str = DEFAULT_OUTPUT_SUBDIR,
    base_relation: str = DEFAULT_BASE_RELATION,
    layers: list[int] | None = None,
    data_source: str = "residual",
) -> dict[str, Any]:
    import numpy as np

    activations_dir = Path("/data/activations") / activations_subdir
    output_dir = Path("/data/analysis_results") / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = layers or DEFAULT_LAYERS
    rows = _load_rows(base_relation)

    per_layer: list[dict[str, Any]] = []

    for layer in layers:
        features = _load_compact_features(activations_dir, data_source, layer)

        X_all, y_family_all, y_conflict_all = [], [], []
        X_conflict, y_family_conflict = [], []
        for row in rows:
            log_id = int(row["log_id"])
            if log_id not in features:
                continue
            fam = str(row["strategy_family"])
            is_conflict = bool(row["conflict_present"])
            vec = features[log_id]
            X_all.append(vec)
            y_family_all.append(fam)
            y_conflict_all.append("conflict" if is_conflict else "aligned")
            if is_conflict:
                X_conflict.append(vec)
                y_family_conflict.append(fam)

        X_all = _rms_normalize(np.stack(X_all))
        X_conflict = _rms_normalize(np.stack(X_conflict))

        # 2a: PCA on full dataset (color by alignment status AND by family)
        pca_all, pca_all_var = _run_pca(X_all, n_components=2)
        _plot_scatter(
            pca_all, y_conflict_all,
            title=f"PCA (all 288 rows) by alignment -- {data_source} layer {layer}",
            output_path=output_dir / f"pca_all_by_alignment_layer{layer}.png",
        )
        _plot_scatter(
            pca_all, y_family_all,
            title=f"PCA (all 288 rows) by family -- {data_source} layer {layer}",
            output_path=output_dir / f"pca_all_by_family_layer{layer}.png",
        )

        # 2a: PCA on conflict-only (color by family)
        pca_conflict, pca_conflict_var = _run_pca(X_conflict, n_components=2)
        _plot_scatter(
            pca_conflict, y_family_conflict,
            title=f"PCA (conflict only, 144 rows) by family -- {data_source} layer {layer}",
            output_path=output_dir / f"pca_conflict_by_family_layer{layer}.png",
        )

        # 2c: LDA on conflict-only, family label
        family_to_idx = {fam: i for i, fam in enumerate(FAMILY_LABELS)}
        y_family_idx = np.asarray([family_to_idx[f] for f in y_family_conflict], dtype=np.int64)
        lda_proj, lda_meta = _run_lda(X_conflict, y_family_idx, n_components=2)
        _plot_scatter(
            lda_proj, y_family_conflict,
            title=f"LDA (conflict only) by family -- {data_source} layer {layer}",
            output_path=output_dir / f"lda_conflict_by_family_layer{layer}.png",
        )

        per_layer.append({
            "layer": int(layer),
            "data_source": data_source,
            "n_all": int(len(y_family_all)),
            "n_conflict": int(len(y_family_conflict)),
            "pca_all_explained_variance_ratio": [float(v) for v in pca_all_var.tolist()],
            "pca_conflict_explained_variance_ratio": [float(v) for v in pca_conflict_var.tolist()],
            "lda_meta": lda_meta,
        })

    summary = {
        "capture_run_id": activations_subdir.split("/")[-1],
        "base_relation": base_relation,
        "layers": list(layers),
        "data_source": data_source,
        "normalization": "rms_per_row",
        "caveat_lda": (
            "LDA on ~36 rows/class is near the floor for stability. "
            "Discriminant axes are visualization-quality, not quantitative claims."
        ),
        "per_layer": per_layer,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


@app.local_entrypoint()
def main() -> None:
    result = run_family_geometry.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
