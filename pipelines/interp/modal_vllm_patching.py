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
    selection_strategy: str = "ordered",
    limit: int = 0,
    family_allowlist: str = "",
    patch_mode: str = "project_out",
    target_layers: str = "4",
    components_per_layer: int = 4,
    component_indices: str = "",
    direction_name: str = "",
    strength: float = 1.0,
    random_seed: int = 42,
    gpu_memory_utilization: float = 0.85,
) -> dict:
    from pipelines.interp.synthetic_market_patching_runner import (
        SyntheticMarketPatchingConfig,
        _parse_component_indices_spec,
        run_synthetic_market_patching,
    )

    target_layer_tuple = tuple(int(token) for token in target_layers.split(",") if token.strip())
    component_indices_by_layer = _parse_component_indices_spec(
        component_indices,
        target_layers=target_layer_tuple or (4,),
    )
    families = tuple(token.strip() for token in family_allowlist.split(",") if token.strip())
    result = run_synthetic_market_patching(
        SyntheticMarketPatchingConfig(
            phase_name=phase_name,
            output_dir=Path(f"/data/activations/synthetic_market_patching/{run_name}"),
            model_id=f"/models/{model_id}",
            context_variant=context_variant,
            order_mode=order_mode,
            selection_strategy=selection_strategy,
            limit=limit if limit > 0 else None,
            family_allowlist=families,
            patch_mode=patch_mode,
            target_layers=target_layer_tuple or (4,),
            components_per_layer=components_per_layer,
            component_indices_by_layer=component_indices_by_layer,
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
    gpu="H200",
    volumes={"/data": synthetic_volume, "/models": model_volume},
    image=image,
    timeout=12 * 3600,
    memory=64 * 1024,
    secrets=[hf_secret, neon_secret],
)
def prepare_synthetic_market_behavior_donors_modal(
    phase_name: str = "phase15_market_basis_discovery_v1",
    run_name: str = "phase19_market_behavior_donors_v1",
    model_id: str = "Qwen/Qwen3-30B-A3B",
    context_variant: str = "market_only",
    order_mode: str = "selection_rank_asc",
    selection_strategy: str = "ordered",
    limit: int = 0,
    family_allowlist: str = "",
    pair_metric: str = "",
    pair_mode: str = "",
    min_pair_gap: float = 0.0,
    target_layers: str = "4",
    tool_schema_mode: str = "",
    tool_choice: str = "",
    add_generation_prompt: bool = True,
    gpu_memory_utilization: float = 0.85,
) -> dict:
    from pipelines.interp.synthetic_market_behavior_runner import (
        SyntheticMarketBehaviorConfig,
        prepare_synthetic_market_behavior_donors,
    )

    target_layer_tuple = tuple(int(token) for token in target_layers.split(",") if token.strip())
    families = tuple(token.strip() for token in family_allowlist.split(",") if token.strip())
    output_dir = Path(f"/data/analysis_results/synthetic_market_behavior/{run_name}")
    result = prepare_synthetic_market_behavior_donors(
        SyntheticMarketBehaviorConfig(
            phase_name=phase_name,
            output_dir=output_dir,
            model_id=f"/models/{model_id}",
            context_variant=context_variant,
            order_mode=order_mode,
            selection_strategy=selection_strategy,
            limit=limit if limit > 0 else None,
            family_allowlist=families,
            pair_metric=pair_metric,
            pair_mode=pair_mode,
            min_pair_gap=float(min_pair_gap),
            target_layers=target_layer_tuple or (4,),
            tool_schema_mode=tool_schema_mode,
            tool_choice=tool_choice,
            add_generation_prompt=bool(add_generation_prompt),
            gpu_memory_utilization=gpu_memory_utilization,
            donor_means_path=output_dir / "donor_means.npz",
            basis_npz_path=Path(BASIS_NPZ_REMOTE),
            basis_results_path=Path(BASIS_RESULTS_REMOTE),
        )
    )
    synthetic_volume.commit()
    return result


@app.function(
    gpu="H200",
    volumes={"/data": synthetic_volume, "/models": model_volume},
    image=image,
    timeout=12 * 3600,
    memory=64 * 1024,
    secrets=[hf_secret, neon_secret],
)
def run_synthetic_market_behavior_modal(
    phase_name: str = "phase15_market_basis_discovery_v1",
    run_name: str = "phase18_market_behavior_v1",
    model_id: str = "Qwen/Qwen3-30B-A3B",
    context_variant: str = "market_only",
    order_mode: str = "selection_rank_asc",
    selection_strategy: str = "ordered",
    limit: int = 0,
    family_allowlist: str = "",
    pair_metric: str = "",
    pair_mode: str = "",
    min_pair_gap: float = 0.0,
    generate_source_behavior: bool = False,
    batch_size: int = 1,
    patch_mode: str = "",
    target_layers: str = "4",
    components_per_layer: int = 4,
    component_indices: str = "",
    direction_name: str = "",
    strength: float = 1.0,
    random_seed: int = 42,
    max_tokens: int = 32,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    tool_schema_mode: str = "",
    tool_choice: str = "",
    add_generation_prompt: bool = True,
    gpu_memory_utilization: float = 0.85,
    donor_means_run_name: str = "",
    enable_chunked_prefill: bool = False,
) -> dict:
    from pipelines.interp.synthetic_market_behavior_runner import (
        SyntheticMarketBehaviorConfig,
        run_synthetic_market_behavior,
    )
    from pipelines.interp.synthetic_market_patching_runner import _parse_component_indices_spec

    target_layer_tuple = tuple(int(token) for token in target_layers.split(",") if token.strip())
    component_indices_by_layer = _parse_component_indices_spec(
        component_indices,
        target_layers=target_layer_tuple or (4,),
    )
    families = tuple(token.strip() for token in family_allowlist.split(",") if token.strip())
    donor_means_path = (
        Path(f"/data/analysis_results/synthetic_market_behavior/{donor_means_run_name}/donor_means.npz")
        if donor_means_run_name
        else None
    )
    result = run_synthetic_market_behavior(
        SyntheticMarketBehaviorConfig(
            phase_name=phase_name,
            output_dir=Path(f"/data/analysis_results/synthetic_market_behavior/{run_name}"),
            model_id=f"/models/{model_id}",
            context_variant=context_variant,
            order_mode=order_mode,
            selection_strategy=selection_strategy,
            limit=limit if limit > 0 else None,
            family_allowlist=families,
            pair_metric=pair_metric,
            pair_mode=pair_mode,
            min_pair_gap=float(min_pair_gap),
            generate_source_behavior=bool(generate_source_behavior),
            batch_size=max(1, int(batch_size)),
            patch_mode=patch_mode,
            target_layers=target_layer_tuple or (4,),
            components_per_layer=components_per_layer,
            component_indices_by_layer=component_indices_by_layer,
            direction_name=direction_name,
            strength=strength,
            random_seed=random_seed,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            tool_schema_mode=tool_schema_mode,
            tool_choice=tool_choice,
            add_generation_prompt=bool(add_generation_prompt),
            gpu_memory_utilization=gpu_memory_utilization,
            donor_means_path=donor_means_path,
            enable_chunked_prefill=bool(enable_chunked_prefill),
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


@app.function(
    volumes={"/data": synthetic_volume},
    image=image,
    timeout=2 * 3600,
    memory=16 * 1024,
)
def analyze_synthetic_market_behavior_modal(
    baseline_run_name: str,
    intervention_run_name: str,
    output_name: str = "",
) -> dict:
    from pipelines.interp.synthetic_market_behavior_analysis import (
        SyntheticMarketBehaviorAnalysisConfig,
        run_synthetic_market_behavior_analysis,
    )

    result = run_synthetic_market_behavior_analysis(
        SyntheticMarketBehaviorAnalysisConfig(
            baseline_dir=Path(f"/data/analysis_results/synthetic_market_behavior/{baseline_run_name}"),
            intervention_dir=Path(f"/data/analysis_results/synthetic_market_behavior/{intervention_run_name}"),
            output_dir=Path(
                f"/data/analysis_results/synthetic_market_behavior_compare/{output_name or intervention_run_name}"
            ),
        )
    )
    synthetic_volume.commit()
    return result
