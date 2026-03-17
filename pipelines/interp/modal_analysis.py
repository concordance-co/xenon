"""Modal wrapper for running analysis on the xenon-data volume.

Mounts the volume read-only, runs probes/expert analysis/PCA on cheap CPU instances.
Results are saved to /data/analysis_results/ on the volume.

Usage (via wrapper script):
    ./scripts/modal_capture.sh analyze --mode probe --target decision_type
    ./scripts/modal_capture.sh analyze --mode all --target decision_type

Or directly:
    uv run --extra analysis --extra modal modal run pipelines/interp/modal_analysis.py \
        --mode probe --target decision_type
"""

import modal

app = modal.App("xenon-analysis")

volume = modal.Volume.from_name("xenon-data", create_if_missing=True)

neon_secret = modal.Secret.from_name("xenon-neon")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "torch", "safetensors", "scikit-learn", "matplotlib",
        "numpy", "pyarrow", "psycopg[binary]",
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
) -> dict:
    """Run analysis on Modal with volume-mounted activations."""
    from pathlib import Path

    from pipelines.interp.analysis import AnalysisConfig, dispatch

    parsed_layers: list[int] | None = None
    if layers_csv:
        parsed_layers = [int(x.strip()) for x in layers_csv.split(",")]

    config = AnalysisConfig(
        activations_dir=Path("/data/activations"),
        output_dir=Path("/data/analysis_results"),
        mode=mode,
        target=target,
        data_source=data_source,
        pooling=pooling,
        n_folds=n_folds,
        layers=parsed_layers,
        limit=limit if limit > 0 else None,
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
    )

    print(f"\nAnalysis complete. Results: {results}")
    print("\nTo download results:")
    print("  modal volume get xenon-data analysis_results/ ./data/analysis_results/ --force")
