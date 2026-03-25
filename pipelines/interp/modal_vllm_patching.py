"""Modal entrypoints for synthetic market patching experiments."""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("xenon-vllm-patching")

synthetic_volume = modal.Volume.from_name("xenon-synthetic-data", create_if_missing=True)
model_volume = modal.Volume.from_name("xenon-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
neon_secret = modal.Secret.from_name("xenon-neon")

BASIS_NPZ_LOCAL = "data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/pca_basis.npz"
BASIS_RESULTS_LOCAL = "data/analysis_results/synthetic_market_axis_decomposition/phase17_market_axis_decomposition_v1/results.json"
BASIS_NPZ_REMOTE = "/root/phase15_pca_basis.npz"
BASIS_RESULTS_REMOTE = "/root/phase17_axis_results.json"

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "vllm",
        "torch",
        "transformers",
        "safetensors",
        "pyarrow",
        "huggingface_hub",
        "psycopg[binary]",
    )
    .env({"VLLM_ALLOW_INSECURE_SERIALIZATION": "1"})
    .add_local_file(BASIS_NPZ_LOCAL, BASIS_NPZ_REMOTE)
    .add_local_file(BASIS_RESULTS_LOCAL, BASIS_RESULTS_REMOTE)
    .add_local_python_source("pipelines")
)


@app.function(
    volumes={"/models": model_volume},
    image=image,
    secrets=[hf_secret],
    timeout=1800,
)
def download_model(model_id: str = "Qwen/Qwen3-30B-A3B") -> str:
    from huggingface_hub import snapshot_download

    local_dir = f"/models/{model_id}"
    snapshot_download(model_id, local_dir=local_dir)
    model_volume.commit()
    return local_dir


@app.function(
    gpu="H200",
    volumes={"/data": synthetic_volume, "/models": model_volume},
    image=image,
    timeout=12 * 3600,
    memory=64 * 1024,
    secrets=[hf_secret, neon_secret],
)
def run_synthetic_market_patching_modal(
    phase_name: str = "phase15_market_basis_discovery_v1",
    run_name: str = "phase18_market_patching_v1",
    model_id: str = "Qwen/Qwen3-30B-A3B",
    context_variant: str = "market_only",
    order_mode: str = "selection_rank_asc",
    limit: int = 0,
    family_allowlist: str = "",
    patch_mode: str = "project_out",
    target_layers: str = "4",
    components_per_layer: int = 4,
    direction_name: str = "",
    strength: float = 1.0,
    random_seed: int = 42,
    gpu_memory_utilization: float = 0.85,
) -> dict:
    from pipelines.interp.synthetic_market_patching_runner import (
        SyntheticMarketPatchingConfig,
        run_synthetic_market_patching,
    )

    target_layer_tuple = tuple(int(token) for token in target_layers.split(",") if token.strip())
    families = tuple(token.strip() for token in family_allowlist.split(",") if token.strip())
    result = run_synthetic_market_patching(
        SyntheticMarketPatchingConfig(
            phase_name=phase_name,
            output_dir=Path(f"/data/activations/synthetic_market_patching/{run_name}"),
            model_id=f"/models/{model_id}",
            context_variant=context_variant,
            order_mode=order_mode,
            limit=limit if limit > 0 else None,
            family_allowlist=families,
            patch_mode=patch_mode,
            target_layers=target_layer_tuple or (4,),
            components_per_layer=components_per_layer,
            direction_name=direction_name,
            strength=strength,
            random_seed=random_seed,
            gpu_memory_utilization=gpu_memory_utilization,
            basis_npz_path=Path(BASIS_NPZ_REMOTE),
            basis_results_path=Path(BASIS_RESULTS_REMOTE),
        )
    )
    synthetic_volume.commit()
    return result


@app.function(
    volumes={"/data": synthetic_volume},
    image=image,
    timeout=2 * 3600,
    memory=16 * 1024,
)
def analyze_synthetic_market_patching_modal(
    intervention_run_name: str,
    control_run_name: str = "",
    baseline_phase_name: str = "phase15_market_basis_discovery_v1",
    output_name: str = "",
    min_layer: int = 0,
    top_k: int = 20,
    basis_state_key: str = "market_mean",
    basis_components: int = 4,
) -> dict:
    from pipelines.interp.synthetic_market_patching_analysis import (
        SyntheticMarketPatchingAnalysisConfig,
        run_synthetic_market_patching_analysis,
    )

    result = run_synthetic_market_patching_analysis(
        SyntheticMarketPatchingAnalysisConfig(
            baseline_dir=Path(f"/data/activations/synthetic_structure/{baseline_phase_name}"),
            intervention_dir=Path(f"/data/activations/synthetic_market_patching/{intervention_run_name}"),
            control_dir=(
                Path(f"/data/activations/synthetic_market_patching/{control_run_name}")
                if control_run_name
                else None
            ),
            output_dir=Path(
                f"/data/analysis_results/synthetic_market_patching/{output_name or intervention_run_name}"
            ),
            min_layer=int(min_layer),
            top_k=int(top_k),
            basis_npz_path=Path(BASIS_NPZ_REMOTE),
            basis_state_key=basis_state_key,
            basis_components=int(basis_components),
        )
    )
    synthetic_volume.commit()
    return result
