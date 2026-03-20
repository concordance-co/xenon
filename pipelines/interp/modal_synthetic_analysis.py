"""Modal CPU workflows for synthetic market structure pooling and analysis."""

from __future__ import annotations

import modal

app = modal.App("xenon-synthetic-analysis")

synthetic_volume = modal.Volume.from_name("xenon-synthetic-data", create_if_missing=True)
model_volume = modal.Volume.from_name("xenon-models", create_if_missing=True)
neon_secret = modal.Secret.from_name("xenon-neon")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "transformers",
        "tokenizers",
        "safetensors",
        "scikit-learn",
        "numpy",
        "pyarrow",
        "jinja2",
        "psycopg[binary]",
    )
    .add_local_python_source("pipelines")
)


@app.function(
    volumes={"/data": synthetic_volume, "/models": model_volume},
    image=image,
    timeout=12 * 3600,
    cpu=8,
    memory=32 * 1024,
    secrets=[neon_secret],
)
def run_synthetic_structure_pooling_modal(
    phase_name: str = "phase1",
    model_id: str = "Qwen/Qwen3-30B-A3B",
    limit: int = 0,
    skip_existing: bool = True,
    cohort_view: str = "synthetic_market_phase1_capture_v0",
    order_mode: str = "selection_rank_asc",
    num_workers: int = 8,
) -> dict:
    from pathlib import Path

    from pipelines.interp.synthetic_structure import (
        SyntheticStructureConfig,
        run_synthetic_structure_pooling,
    )

    config = SyntheticStructureConfig(
        activations_dir=Path(f"/data/activations/{phase_name}"),
        output_dir=Path(f"/data/activations/synthetic_structure/{phase_name}"),
        model_id=f"/models/{model_id}",
        limit=limit if limit > 0 else None,
        skip_existing=skip_existing,
        cohort_view=cohort_view,
        order_mode=order_mode,
        num_workers=num_workers,
    )
    result = run_synthetic_structure_pooling(config)
    synthetic_volume.commit()
    return result


@app.function(
    volumes={"/data": synthetic_volume},
    image=image,
    timeout=12 * 3600,
    cpu=8,
    memory=32 * 1024,
    secrets=[neon_secret],
)
def run_synthetic_manifold_analysis_modal(
    phase_name: str = "phase1",
    context_variant: str = "market_only",
    family_allowlist: str = "",
    scalar_family_name: str = "scalar_sweep",
    layers: str = "",
    num_workers: int = 8,
) -> dict:
    from pathlib import Path

    from pipelines.interp.synthetic_manifold_analysis import (
        SyntheticManifoldAnalysisConfig,
        run_synthetic_manifold_analysis,
    )

    parsed_layers = [int(token) for token in layers.split(",") if token.strip()] or None
    config = SyntheticManifoldAnalysisConfig(
        structure_dir=Path(f"/data/activations/synthetic_structure/{phase_name}"),
        output_dir=Path(f"/data/analysis_results/synthetic_manifold/{phase_name}"),
        phase_name=phase_name,
        context_variant=context_variant,
        family_allowlist=tuple(token.strip() for token in family_allowlist.split(",") if token.strip()),
        scalar_family_name=scalar_family_name,
        layers=parsed_layers,
        num_workers=num_workers,
    )
    result = run_synthetic_manifold_analysis(config)
    synthetic_volume.commit()
    return result


@app.local_entrypoint()
def main(
    mode: str = "synthetic-structure",
    phase_name: str = "phase1",
    model_id: str = "Qwen/Qwen3-30B-A3B",
    limit: int = 0,
    skip_existing: bool = True,
    cohort_view: str = "synthetic_market_phase1_capture_v0",
    order_mode: str = "selection_rank_asc",
    num_workers: int = 8,
    layers: str = "",
    context_variant: str = "market_only",
    family_allowlist: str = "",
    scalar_family_name: str = "scalar_sweep",
):
    if mode == "synthetic-structure":
        result = run_synthetic_structure_pooling_modal.remote(
            phase_name=phase_name,
            model_id=model_id,
            limit=limit,
            skip_existing=skip_existing,
            cohort_view=cohort_view,
            order_mode=order_mode,
            num_workers=num_workers,
        )
        print(result)
    elif mode == "synthetic-manifold":
        result = run_synthetic_manifold_analysis_modal.remote(
            phase_name=phase_name,
            context_variant=context_variant,
            family_allowlist=family_allowlist,
            scalar_family_name=scalar_family_name,
            layers=layers,
            num_workers=num_workers,
        )
        print(result)
    else:
        print(f"Unknown mode: {mode}")
