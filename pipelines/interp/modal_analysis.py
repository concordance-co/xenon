"""Canonical Modal analysis orchestrator for generic workflow-driven analysis."""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("xenon-analysis")

volume = modal.Volume.from_name("xenon-data", create_if_missing=True)
neon_secret = modal.Secret.from_name("xenon-neon")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch",
        "transformers",
        "safetensors",
        "scikit-learn",
        "matplotlib",
        "numpy",
        "pyarrow",
        "psycopg[binary]",
    )
    .add_local_python_source("pipelines")
)


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=1800,
    cpu=4,
    secrets=[neon_secret],
)
def run_analysis(
    mode: str,
    target: str,
    data_source: str = "router",
    pooling: str = "last_token",
    layers_csv: str = "",
    n_folds: int = 5,
    seed: int = 42,
    limit: int = 0,
    group_column: str = "",
    relation_name: str = "",
    activations_subdir: str = "",
    output_subdir: str = "",
    labels_subdir: str = "",
) -> dict:
    """Run generic analysis on Modal with volume-mounted activations."""
    from pipelines.interp.analysis import AnalysisConfig, dispatch
    from pipelines.db import connect_neon, ensure_schema
    from pipelines.workflows import export_publication_labels

    parsed_layers: list[int] | None = None
    if layers_csv:
        parsed_layers = [int(token.strip()) for token in layers_csv.split(",") if token.strip()]

    activations_dir = Path("/data/activations")
    if activations_subdir:
        activations_dir = activations_dir / activations_subdir

    output_dir = Path("/data/analysis_results")
    if output_subdir:
        output_dir = output_dir / output_subdir

    labels_path: Path | None = None
    if relation_name:
        labels_path = Path("/data/workflow_labels")
        if labels_subdir:
            labels_path = labels_path / labels_subdir
        labels_path.mkdir(parents=True, exist_ok=True)
        labels_path = labels_path / f"{relation_name}.parquet"
        with connect_neon(autocommit=True) as conn:
            ensure_schema(conn)
            export_publication_labels(
                conn,
                relation_name=relation_name,
                output_path=labels_path,
                group_column=group_column or None,
            )

    config = AnalysisConfig(
        activations_dir=activations_dir,
        labels_path=labels_path,
        output_dir=output_dir,
        mode=mode,
        target=target,
        data_source=data_source,
        pooling=pooling,
        n_folds=n_folds,
        layers=parsed_layers,
        limit=limit if limit > 0 else None,
        group_column=group_column or None,
        seed=seed,
    )

    results = dispatch(config)
    volume.commit()
    return results


@app.local_entrypoint()
def main(
    mode: str = "probe",
    target: str = "decision_type",
    data_source: str = "router",
    pooling: str = "last_token",
    layers: str = "",
    n_folds: int = 5,
    seed: int = 42,
    limit: int = 0,
    group_column: str = "",
    relation_name: str = "",
    activations_subdir: str = "",
    output_subdir: str = "",
    labels_subdir: str = "",
):
    results = run_analysis.remote(
        mode=mode,
        target=target,
        data_source=data_source,
        pooling=pooling,
        layers_csv=layers,
        n_folds=n_folds,
        seed=seed,
        limit=limit,
        group_column=group_column,
        relation_name=relation_name,
        activations_subdir=activations_subdir,
        output_subdir=output_subdir,
        labels_subdir=labels_subdir,
    )
    print(f"\nAnalysis complete. Results: {results}")
