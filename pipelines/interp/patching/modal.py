"""Modal entrypoints for synthetic market patching experiments."""

import json
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
    .add_local_python_source("projects")
)

GPU_WORKER_CPU = 8
GPU_WORKER_MEMORY_MB = 8 * 1024


@app.cls(
    gpu="H200",
    volumes={"/data": synthetic_volume, "/models": model_volume},
    image=image,
    timeout=12 * 3600,
    scaledown_window=300,
    cpu=GPU_WORKER_CPU,
    memory=GPU_WORKER_MEMORY_MB,
    secrets=[hf_secret],
)
class SyntheticMarketBehaviorMatrixWorker:
    model_id: str = modal.parameter(default="Qwen/Qwen3-30B-A3B")
    tool_schema_mode: str = modal.parameter(default="")
    tool_choice: str = modal.parameter(default="")
    add_generation_prompt: bool = modal.parameter(default=True)
    batch_size: int = modal.parameter(default=1)
    max_tokens: int = modal.parameter(default=32)
    temperature: str = modal.parameter(default="0.0")
    top_p: str = modal.parameter(default="1.0")
    top_k: int = modal.parameter(default=-1)
    tensor_parallel_size: int = modal.parameter(default=1)
    gpu_memory_utilization: str = modal.parameter(default="0.85")
    enforce_eager: bool = modal.parameter(default=True)
    enable_chunked_prefill: bool = modal.parameter(default=False)
    enable_logging_iteration_details: bool = modal.parameter(default=False)
    enable_mfu_metrics: bool = modal.parameter(default=False)
    basis_state_key: str = modal.parameter(default="market_mean")
    basis_layers: str = modal.parameter(default="")
    basis_components_per_layer: int = modal.parameter(default=0)

    @modal.enter()
    def setup(self) -> None:
        from pathlib import Path

        from transformers import AutoTokenizer

        from pipelines.interp.patching.basis import default_phase17_market_patch_basis
        from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_runner import (
            SyntheticMarketBehaviorConfig,
            _build_generation_config,
        )
        from pipelines.interp.modal_vllm_engine import (
            _create_llm,
            _init_market_patching_on_model,
            _register_market_patch_basis_on_model,
        )

        local_model_path = f"/models/{self.model_id}"
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        self.tool_choice_value = self.tool_choice.strip() or None

        worker_config = SyntheticMarketBehaviorConfig(
            output_dir=Path("/tmp/synthetic_market_behavior_matrix_worker"),
            model_id=local_model_path,
            batch_size=max(1, int(self.batch_size)),
            max_tokens=int(self.max_tokens),
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            top_k=int(self.top_k),
            tool_schema_mode=self.tool_schema_mode,
            tool_choice=self.tool_choice,
            add_generation_prompt=bool(self.add_generation_prompt),
            tensor_parallel_size=int(self.tensor_parallel_size),
            gpu_memory_utilization=float(self.gpu_memory_utilization),
            enforce_eager=bool(self.enforce_eager),
            enable_chunked_prefill=bool(self.enable_chunked_prefill),
            enable_logging_iteration_details=bool(self.enable_logging_iteration_details),
            enable_mfu_metrics=bool(self.enable_mfu_metrics),
            basis_state_key=self.basis_state_key,
            basis_npz_path=Path(BASIS_NPZ_REMOTE),
            basis_results_path=Path(BASIS_RESULTS_REMOTE),
        )
        self.generate_cfg = _build_generation_config(worker_config)
        self.llm = _create_llm(self.generate_cfg)

        parsed_layers = tuple(
            int(token.strip()) for token in self.basis_layers.split(",") if token.strip()
        )
        if parsed_layers and int(self.basis_components_per_layer) > 0:
            basis = default_phase17_market_patch_basis(
                basis_npz_path=Path(BASIS_NPZ_REMOTE),
                results_json_path=Path(BASIS_RESULTS_REMOTE),
                state_key=self.basis_state_key,
                layers=parsed_layers,
                components_per_layer=int(self.basis_components_per_layer),
            )
            _init_market_patching_on_model(self.llm)
            _register_market_patch_basis_on_model(self.llm, basis.to_payload())

    @modal.method()
    def generate_batch(self, requests: list[dict]) -> list[dict]:
        from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_runner import (
            _run_generation_batch,
        )

        from pipelines.interp.tool_schemas import resolve_tool_schema_mode

        tools = resolve_tool_schema_mode(self.tool_schema_mode)
        return _run_generation_batch(
            llm=self.llm,
            tokenizer=self.tokenizer,
            requests=requests,
            config=self.generate_cfg,
            max_tokens=int(self.max_tokens),
            temperature=float(self.temperature),
            top_p=float(self.top_p),
            top_k=int(self.top_k),
            tools=tools,
            tool_choice=self.tool_choice_value,
        )

    @modal.exit()
    def teardown(self) -> None:
        from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_runner import _destroy_llm

        _destroy_llm(getattr(self, "llm", None))


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
    cpu=GPU_WORKER_CPU,
    memory=GPU_WORKER_MEMORY_MB,
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
    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_patching_runner import (
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
    cpu=GPU_WORKER_CPU,
    memory=GPU_WORKER_MEMORY_MB,
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
    batch_size: int = 32,
    family_allowlist: str = "",
    pair_metric: str = "",
    pair_mode: str = "",
    min_pair_gap: float = 0.0,
    target_layers: str = "4",
    secondary_patch_mode: str = "",
    secondary_target_layers: str = "",
    tool_schema_mode: str = "",
    tool_choice: str = "",
    add_generation_prompt: bool = True,
    gpu_memory_utilization: float = 0.85,
    basis_state_key: str = "market_mean",
) -> dict:
    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_runner import (
        SyntheticMarketBehaviorConfig,
        prepare_synthetic_market_behavior_donors,
    )

    target_layer_tuple = tuple(int(token) for token in target_layers.split(",") if token.strip())
    secondary_target_layer_tuple = tuple(
        int(token) for token in secondary_target_layers.split(",") if token.strip()
    )
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
            batch_size=max(1, int(batch_size)),
            family_allowlist=families,
            pair_metric=pair_metric,
            pair_mode=pair_mode,
            min_pair_gap=float(min_pair_gap),
            target_layers=target_layer_tuple or (4,),
            secondary_patch_mode=secondary_patch_mode,
            secondary_target_layers=secondary_target_layer_tuple,
            tool_schema_mode=tool_schema_mode,
            tool_choice=tool_choice,
            add_generation_prompt=bool(add_generation_prompt),
            gpu_memory_utilization=gpu_memory_utilization,
            donor_means_path=output_dir / "donor_means.npz",
            basis_state_key=basis_state_key,
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
    max_containers=1,
    cpu=GPU_WORKER_CPU,
    memory=GPU_WORKER_MEMORY_MB,
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
    example_id_allowlist: str = "",
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
    secondary_patch_mode: str = "",
    secondary_target_layers: str = "",
    secondary_components_per_layer: int = 4,
    secondary_component_indices: str = "",
    secondary_direction_name: str = "",
    secondary_strength: float = 1.0,
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
    enforce_eager: bool = True,
    enable_chunked_prefill: bool = False,
    enable_logging_iteration_details: bool = False,
    enable_mfu_metrics: bool = False,
    basis_state_key: str = "market_mean",
) -> dict:
    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_runner import (
        SyntheticMarketBehaviorConfig,
        run_synthetic_market_behavior,
    )
    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_patching_runner import _parse_component_indices_spec

    target_layer_tuple = tuple(int(token) for token in target_layers.split(",") if token.strip())
    secondary_target_layer_tuple = tuple(
        int(token) for token in secondary_target_layers.split(",") if token.strip()
    )
    component_indices_by_layer = _parse_component_indices_spec(
        component_indices,
        target_layers=target_layer_tuple or (4,),
    )
    secondary_component_indices_by_layer = _parse_component_indices_spec(
        secondary_component_indices,
        target_layers=secondary_target_layer_tuple or (4,),
    )
    families = tuple(token.strip() for token in family_allowlist.split(",") if token.strip())
    example_ids = tuple(token.strip() for token in example_id_allowlist.split(",") if token.strip())
    normalized_patch_mode = patch_mode.strip().lower()
    normalized_secondary_patch_mode = secondary_patch_mode.strip().lower()
    if (
        normalized_patch_mode in {"swap_mean", "swap_components"}
        or normalized_secondary_patch_mode in {"swap_mean", "swap_components"}
    ) and not donor_means_run_name.strip():
        raise ValueError(
            "swap_mean/swap_components runs on Modal require a precomputed donor_means_run_name. "
            "Run prepare_synthetic_market_behavior_donors_modal first, then pass --donor-means-run-name."
        )
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
            example_id_allowlist=example_ids,
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
            secondary_patch_mode=secondary_patch_mode,
            secondary_target_layers=secondary_target_layer_tuple,
            secondary_components_per_layer=secondary_components_per_layer,
            secondary_component_indices_by_layer=secondary_component_indices_by_layer,
            secondary_direction_name=secondary_direction_name,
            secondary_strength=secondary_strength,
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
            enforce_eager=bool(enforce_eager),
            enable_chunked_prefill=bool(enable_chunked_prefill),
            enable_logging_iteration_details=bool(enable_logging_iteration_details),
            enable_mfu_metrics=bool(enable_mfu_metrics),
            basis_state_key=basis_state_key,
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
    memory=8 * 1024,
)
def plan_synthetic_market_behavior_battery_modal(
    run_name_prefix: str,
    output_name: str = "",
    phase_name: str = "phase15_market_basis_discovery_v1",
    model_id: str = "Qwen/Qwen3-30B-A3B",
    context_variant: str = "market_only",
    order_mode: str = "selection_rank_asc",
    selection_strategy: str = "ordered",
    limit: int = 0,
    family_allowlist: str = "",
    pair_metric: str = "",
    pair_mode: str = "",
    pair_modes: str = "",
    min_pair_gap: float = 0.0,
    generate_source_behavior: bool = False,
    batch_size: int = 1,
    patch_mode: str = "project_out",
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
    enforce_eager: bool = True,
    enable_chunked_prefill: bool = False,
    basis_state_key: str = "market_mean",
    lambda_sweep: str = "0.0,0.25,0.5,0.75,1.0,1.5",
    neighboring_component_offsets: str = "0",
    subspace_sizes: str = "",
    random_control_seeds: str = "11,17,23,29,31",
    include_baselines: bool = True,
) -> dict:
    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_battery import (
        build_behavior_robustness_payload,
    )
    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_runner import SyntheticMarketBehaviorConfig
    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_patching_runner import _parse_component_indices_spec

    def _parse_float_csv(text: str) -> tuple[float, ...]:
        return tuple(float(token.strip()) for token in text.split(",") if token.strip())

    def _parse_int_csv(text: str) -> tuple[int, ...]:
        return tuple(int(token.strip()) for token in text.split(",") if token.strip())

    def _parse_str_csv(text: str) -> tuple[str, ...]:
        return tuple(token.strip() for token in text.split(",") if token.strip())

    target_layer_tuple = tuple(int(token) for token in target_layers.split(",") if token.strip())
    component_indices_by_layer = _parse_component_indices_spec(
        component_indices,
        target_layers=target_layer_tuple or (4,),
    )
    families = tuple(token.strip() for token in family_allowlist.split(",") if token.strip())
    parsed_pair_modes = _parse_str_csv(pair_modes)
    if not parsed_pair_modes and pair_mode.strip():
        parsed_pair_modes = (pair_mode.strip(),)
    base_config = SyntheticMarketBehaviorConfig(
        phase_name=phase_name,
        output_dir=Path(f"/data/analysis_results/synthetic_market_behavior/{output_name or run_name_prefix}"),
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
        strength=float(strength),
        random_seed=int(random_seed),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        tool_schema_mode=tool_schema_mode,
        tool_choice=tool_choice,
        add_generation_prompt=bool(add_generation_prompt),
        enforce_eager=bool(enforce_eager),
        enable_chunked_prefill=bool(enable_chunked_prefill),
        basis_state_key=basis_state_key,
        basis_npz_path=Path(BASIS_NPZ_REMOTE),
        basis_results_path=Path(BASIS_RESULTS_REMOTE),
    )
    payload = build_behavior_robustness_payload(
        base_config,
        run_name_prefix=run_name_prefix,
        lambda_sweep=_parse_float_csv(lambda_sweep),
        neighboring_component_offsets=_parse_int_csv(neighboring_component_offsets),
        subspace_sizes=_parse_int_csv(subspace_sizes),
        random_control_seeds=_parse_int_csv(random_control_seeds),
        pair_modes=parsed_pair_modes,
        include_baselines=bool(include_baselines),
    )
    plan_dir = Path(f"/data/analysis_results/synthetic_market_behavior_batteries/{output_name or run_name_prefix}")
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / "plan.json").write_text(json.dumps(payload, indent=2))
    synthetic_volume.commit()
    return payload


@app.function(
    volumes={"/data": synthetic_volume, "/models": model_volume},
    image=image,
    timeout=12 * 3600,
    cpu=4,
    memory=8 * 1024,
    secrets=[hf_secret, neon_secret],
)
def run_synthetic_market_behavior_matrix_modal(
    run_name_prefix: str,
    run_name: str = "",
    phase_name: str = "phase15_market_basis_discovery_v1",
    model_id: str = "Qwen/Qwen3-30B-A3B",
    context_variant: str = "market_only",
    order_mode: str = "selection_rank_asc",
    selection_strategy: str = "ordered",
    limit: int = 0,
    family_allowlist: str = "",
    pair_metric: str = "",
    pair_mode: str = "",
    pair_modes: str = "",
    min_pair_gap: float = 0.0,
    generate_source_behavior: bool = False,
    batch_size: int = 1,
    patch_mode: str = "project_out",
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
    enforce_eager: bool = True,
    enable_chunked_prefill: bool = False,
    basis_state_key: str = "market_mean",
    lambda_sweep: str = "0.0,0.25,0.5,0.75,1.0,1.5",
    neighboring_component_offsets: str = "0",
    subspace_sizes: str = "",
    random_control_seeds: str = "11,17,23,29,31",
    include_baselines: bool = True,
) -> dict:
    from collections import defaultdict
    from dataclasses import replace
    from datetime import UTC, datetime

    from transformers import AutoTokenizer

    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_battery import (
        build_behavior_robustness_matrix,
    )
    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_matrix_runner import (
        _build_union_basis_payload,
        _cell_component_count,
        _prep_group_key,
        _requires_eager_runtime,
    )
    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_runner import (
        SyntheticMarketBehaviorConfig,
        _chunk_list,
        _decode_first_token,
        _extract_first_tool_call_fields,
        _flush_table,
        _prepare_behavior_rows,
    )
    from pipelines.interp.tool_schemas import resolve_tool_schema_mode
    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_patching_runner import _parse_component_indices_spec
    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_patching_runner import _build_patch_spec

    def _parse_float_csv(text: str) -> tuple[float, ...]:
        return tuple(float(token.strip()) for token in text.split(",") if token.strip())

    def _parse_int_csv(text: str) -> tuple[int, ...]:
        return tuple(int(token.strip()) for token in text.split(",") if token.strip())

    def _parse_str_csv(text: str) -> tuple[str, ...]:
        return tuple(token.strip() for token in text.split(",") if token.strip())

    target_layer_tuple = tuple(int(token) for token in target_layers.split(",") if token.strip())
    component_indices_by_layer = _parse_component_indices_spec(
        component_indices,
        target_layers=target_layer_tuple or (4,),
    )
    families = tuple(token.strip() for token in family_allowlist.split(",") if token.strip())
    parsed_pair_modes = _parse_str_csv(pair_modes)
    if not parsed_pair_modes and pair_mode.strip():
        parsed_pair_modes = (pair_mode.strip(),)

    output_dir = Path(f"/data/analysis_results/synthetic_market_behavior/{run_name or run_name_prefix}")
    base_config = SyntheticMarketBehaviorConfig(
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
        generate_source_behavior=bool(generate_source_behavior),
        batch_size=max(1, int(batch_size)),
        patch_mode=patch_mode,
        target_layers=target_layer_tuple or (4,),
        components_per_layer=components_per_layer,
        component_indices_by_layer=component_indices_by_layer,
        direction_name=direction_name,
        strength=float(strength),
        random_seed=int(random_seed),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        tool_schema_mode=tool_schema_mode,
        tool_choice=tool_choice,
        add_generation_prompt=bool(add_generation_prompt),
        gpu_memory_utilization=float(gpu_memory_utilization),
        enforce_eager=bool(enforce_eager),
        enable_chunked_prefill=bool(enable_chunked_prefill),
        basis_state_key=basis_state_key,
        basis_npz_path=Path(BASIS_NPZ_REMOTE),
        basis_results_path=Path(BASIS_RESULTS_REMOTE),
    )
    cells = tuple(
        build_behavior_robustness_matrix(
            base_config,
            run_name_prefix=run_name_prefix,
            lambda_sweep=_parse_float_csv(lambda_sweep),
            neighboring_component_offsets=_parse_int_csv(neighboring_component_offsets),
            subspace_sizes=_parse_int_csv(subspace_sizes),
            random_control_seeds=_parse_int_csv(random_control_seeds),
            pair_modes=parsed_pair_modes,
            include_baselines=bool(include_baselines),
        )
    )

    if any(cell.config.patch_mode in {"swap_mean", "swap_components"} for cell in cells):
        raise NotImplementedError(
            "Modal matrix orchestration currently supports baseline/project_out/random_control cells only"
        )

    tokenizer = AutoTokenizer.from_pretrained(base_config.model_id)
    tools = resolve_tool_schema_mode(base_config.tool_schema_mode)
    tool_choice_value = base_config.tool_choice.strip() or None
    chat_template_kwargs = {"tool_choice": tool_choice_value} if tool_choice_value is not None else None

    cells_by_group: dict[tuple[object, ...], list] = defaultdict(list)
    for cell in cells:
        cells_by_group[_prep_group_key(cell)].append(cell)

    prepared_rows_by_group: dict[tuple[object, ...], list[dict]] = {}
    skipped_by_group: dict[tuple[object, ...], int] = {}
    for group_key, group_cells in cells_by_group.items():
        prep_config = replace(base_config, **{
            "phase_name": group_cells[0].config.phase_name,
            "context_variant": group_cells[0].config.context_variant,
            "order_mode": group_cells[0].config.order_mode,
            "selection_strategy": group_cells[0].config.selection_strategy,
            "limit": group_cells[0].config.limit,
            "family_allowlist": group_cells[0].config.family_allowlist,
            "pair_metric": group_cells[0].config.pair_metric,
            "pair_mode": group_cells[0].config.pair_mode,
            "min_pair_gap": group_cells[0].config.min_pair_gap,
            "output_dir": output_dir,
        })
        prepared_rows, skipped = _prepare_behavior_rows(
            config=prep_config,
            tokenizer=tokenizer,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        prepared_rows_by_group[group_key] = prepared_rows
        skipped_by_group[group_key] = skipped

    if not any(prepared_rows_by_group.values()):
        result = {"error": "no_valid_examples"}
        (output_dir / "results.json").write_text(json.dumps(result, indent=2))
        synthetic_volume.commit()
        return result

    engine_cells_by_group: dict[bool, dict[tuple[object, ...], list]] = defaultdict(lambda: defaultdict(list))
    for group_key, group_cells in cells_by_group.items():
        for cell in group_cells:
            runtime_enforce_eager = _requires_eager_runtime(base_config, cell)
            engine_cells_by_group[bool(runtime_enforce_eager)][group_key].append(cell)

    metadata_rows: list[dict] = []
    source_behavior_cache: dict[int, dict] = {}
    processed = 0

    WorkerCls = SyntheticMarketBehaviorMatrixWorker.with_options(max_containers=1)

    for runtime_enforce_eager in sorted(engine_cells_by_group):
        grouped_cells = engine_cells_by_group[runtime_enforce_eager]
        if not grouped_cells:
            continue

        shard_cells = [cell for shard in grouped_cells.values() for cell in shard]
        basis_payload = _build_union_basis_payload(shard_cells, base_config)
        basis_layers = ",".join(str(int(layer)) for layer in sorted(basis_payload))
        basis_components_per_layer = (
            max(_cell_component_count(cell) for cell in shard_cells if cell.config.patch_enabled)
            if basis_payload
            else 0
        )

        worker = WorkerCls(
            model_id=model_id,
            tool_schema_mode=tool_schema_mode,
            tool_choice=tool_choice,
            add_generation_prompt=bool(add_generation_prompt),
            batch_size=max(1, int(batch_size)),
            max_tokens=int(max_tokens),
            temperature=str(float(temperature)),
            top_p=str(float(top_p)),
            top_k=int(top_k),
            tensor_parallel_size=1,
            gpu_memory_utilization=str(float(gpu_memory_utilization)),
            enforce_eager=bool(runtime_enforce_eager),
            enable_chunked_prefill=bool(enable_chunked_prefill),
            basis_state_key=base_config.basis_state_key,
            basis_layers=basis_layers,
            basis_components_per_layer=int(basis_components_per_layer),
        )

        if any(cell.config.generate_source_behavior for cells_group in grouped_cells.values() for cell in cells_group):
            source_request_batches: list[list[dict]] = []
            source_batch_contexts: list[list[int]] = []
            seen_source_log_ids: set[int] = set(source_behavior_cache)
            unique_source_requests: list[dict] = []
            source_log_ids: list[int] = []
            for group_key, prepared_rows in prepared_rows_by_group.items():
                if not any(cell.config.generate_source_behavior for cell in grouped_cells.get(group_key, [])):
                    continue
                for prepared in prepared_rows:
                    source_log_id = prepared["source_log_id"]
                    source_row_messages = prepared["source_row_messages"]
                    if source_log_id is None or source_row_messages is None or int(source_log_id) in seen_source_log_ids:
                        continue
                    seen_source_log_ids.add(int(source_log_id))
                    unique_source_requests.append({"messages": source_row_messages})
                    source_log_ids.append(int(source_log_id))
            for request_chunk, id_chunk in zip(
                _chunk_list(unique_source_requests, int(base_config.batch_size)),
                _chunk_list(source_log_ids, int(base_config.batch_size)),
                strict=False,
            ):
                source_request_batches.append(request_chunk)
                source_batch_contexts.append(id_chunk)
            mapped_source_outputs = (
                list(worker.generate_batch.map(source_request_batches))
                if source_request_batches
                else []
            )
            for source_id_chunk, outputs in zip(
                source_batch_contexts,
                mapped_source_outputs,
                strict=False,
            ):
                for source_log_id, source_output in zip(source_id_chunk, outputs, strict=False):
                    source_generated_token_ids = [int(token) for token in source_output["generated_token_ids"]]
                    source_tool_fields = _extract_first_tool_call_fields(str(source_output["generated_text"]))
                    source_behavior_cache[int(source_log_id)] = {
                        "source_generated_text": source_output["generated_text"],
                        "source_generated_token_count": len(source_generated_token_ids),
                        "source_first_generated_token_id": (
                            source_generated_token_ids[0] if source_generated_token_ids else None
                        ),
                        "source_first_generated_token_text": _decode_first_token(tokenizer, source_generated_token_ids),
                        **{f"source_{key}": value for key, value in source_tool_fields.items()},
                    }

        request_batches: list[list[dict]] = []
        batch_contexts: list[list[dict]] = []
        for group_key, prepared_rows in prepared_rows_by_group.items():
            group_cells = grouped_cells.get(group_key, [])
            if not group_cells:
                continue
            request_items: list[dict] = []
            for cell in group_cells:
                for prepared in prepared_rows:
                    request_items.append({"cell": cell, "prepared": prepared})

            for request_chunk in _chunk_list(request_items, int(base_config.batch_size)):
                batch_requests: list[dict] = []
                chunk_context: list[dict] = []
                for item in request_chunk:
                    cell = item["cell"]
                    prepared = item["prepared"]
                    row = prepared["row"]
                    market_span = prepared["market_span"]
                    source_log_id = prepared["source_log_id"]
                    cell_cfg = cell.config
                    source_behavior_fields: dict[str, object] = {}
                    if cell_cfg.generate_source_behavior and source_log_id is not None:
                        source_behavior_fields = dict(source_behavior_cache.get(int(source_log_id), {}))

                    patch_spec = None
                    if cell_cfg.patch_enabled:
                        patch_spec = _build_patch_spec(
                            config=cell_cfg,  # type: ignore[arg-type]
                            market_span=market_span,
                            basis_payload=basis_payload,
                        ).to_payload()

                    batch_requests.append({"messages": prepared["messages"], "patch_spec": patch_spec})
                    chunk_context.append(
                        {
                            "cell": cell,
                            "row": row,
                            "source_log_id": source_log_id,
                            "source_behavior_fields": source_behavior_fields,
                        }
                    )
                request_batches.append(batch_requests)
                batch_contexts.append(chunk_context)

        mapped_outputs = list(worker.generate_batch.map(request_batches)) if request_batches else []
        for chunk_context, outputs in zip(
            batch_contexts,
            mapped_outputs,
            strict=False,
        ):
            for context, output in zip(chunk_context, outputs, strict=False):
                cell = context["cell"]
                row = context["row"]
                cell_cfg = cell.config
                generated_token_ids = [int(token) for token in output["generated_token_ids"]]
                tool_fields = _extract_first_tool_call_fields(str(output["generated_text"]))
                metadata_rows.append(
                    {
                        "matrix_cell_id": cell.run_name,
                        "matrix_sweep_kind": cell.sweep_kind,
                        "matrix_sweep_value": cell.sweep_value,
                        "matrix_cell_description": cell.description,
                        "runtime_enforce_eager": bool(runtime_enforce_eager),
                        "log_id": int(row["log_id"]),
                        "phase_name": row["phase_name"],
                        "example_id": row["example_id"],
                        "family": row["family"],
                        "family_variant": row["family_variant"],
                        "context_variant": row["context_variant"],
                        "roster_key": row.get("roster_key") or "",
                        "pair_metric_name": row.get("pair_metric_name") or None,
                        "pair_mode": row.get("pair_mode") or None,
                        "pair_id": row.get("pair_id"),
                        "pair_metric_value": (
                            float(row.get("base_pair_metric_value", row.get("pair_metric_value")))
                            if row.get("base_pair_metric_value", row.get("pair_metric_value")) is not None
                            else None
                        ),
                        "source_pair_metric_value": (
                            float(row.get("source_pair_metric_value"))
                            if row.get("source_pair_metric_value") is not None
                            else None
                        ),
                        "pair_metric_gap": (
                            float(row.get("pair_metric_gap")) if row.get("pair_metric_gap") is not None else None
                        ),
                        "source_log_id": context["source_log_id"],
                        "source_example_id": row.get("source_example_id"),
                        "source_family_variant": row.get("source_family_variant"),
                        "source_roster_key": row.get("source_roster_key"),
                        "capture_timestamp": datetime.now(UTC).isoformat(),
                        "seq_len": int(len(output["input_ids"])),
                        "max_tokens": int(base_config.max_tokens),
                        "temperature": float(base_config.temperature),
                        "top_p": float(base_config.top_p),
                        "top_k": int(base_config.top_k),
                        "tool_schema_mode": base_config.tool_schema_mode,
                        "tool_choice": base_config.tool_choice,
                        "tool_count": len(tools) if tools is not None else 0,
                        "patch_mode": cell_cfg.patch_mode,
                        "target_layers": ",".join(str(int(layer)) for layer in cell_cfg.target_layers),
                        "components_per_layer": int(cell_cfg.components_per_layer),
                        "component_indices_by_layer_json": json.dumps(
                            {
                                str(layer): [int(index) for index in indices]
                                for layer, indices in cell_cfg.component_indices_by_layer.items()
                            }
                        ),
                        "direction_name": cell_cfg.direction_name,
                        "selection_strategy": cell_cfg.selection_strategy,
                        "strength": float(cell_cfg.strength),
                        "random_seed": int(cell_cfg.random_seed),
                        "generated_token_ids_json": json.dumps(generated_token_ids),
                        "generated_token_count": len(generated_token_ids),
                        "first_generated_token_id": generated_token_ids[0] if generated_token_ids else None,
                        "first_generated_token_text": _decode_first_token(tokenizer, generated_token_ids),
                        "generated_text": output["generated_text"],
                        "finish_reason": output["finish_reason"],
                        "request_id": output.get("request_id") or None,
                        **tool_fields,
                        **context["source_behavior_fields"],
                        "patch_stats_json": json.dumps(output["patch_stats"]),
                        "all_patch_stats_json": json.dumps(output.get("all_patch_stats", {})),
                    }
                )
                processed += 1

    metadata_path = output_dir / "metadata.parquet"
    _flush_table(metadata_path, metadata_rows)
    plan_payload = {
        "count": len(cells),
        "runs": [cell.to_dict() for cell in cells],
    }
    (output_dir / "plan.json").write_text(json.dumps(plan_payload, indent=2))

    rows_by_cell: dict[str, int] = defaultdict(int)
    for row in metadata_rows:
        rows_by_cell[str(row["matrix_cell_id"])] += 1

    result = {
        "phase_name": base_config.phase_name,
        "processed": processed,
        "output_dir": str(output_dir),
        "cell_count": len(cells),
        "rows_per_cell": dict(sorted(rows_by_cell.items())),
        "counts_by_sweep_kind": {
            sweep_kind: sum(1 for cell in cells if cell.sweep_kind == sweep_kind)
            for sweep_kind in sorted({cell.sweep_kind for cell in cells})
        },
        "pair_modes": sorted({str(cell.config.pair_mode) for cell in cells if cell.config.pair_mode}),
        "runtime_groups": {
            ("eager" if runtime_enforce_eager else "compiled"): sum(
                len(cells_group) for cells_group in grouped.values()
            )
            for runtime_enforce_eager, grouped in engine_cells_by_group.items()
        },
        "skipped_by_group": {json.dumps(list(key)): int(value) for key, value in skipped_by_group.items()},
        "batch_size": int(base_config.batch_size),
        "max_tokens": int(base_config.max_tokens),
    }
    (output_dir / "results.json").write_text(json.dumps(result, indent=2))
    synthetic_volume.commit()
    return result


@app.function(
    gpu="H200",
    volumes={"/data": synthetic_volume, "/models": model_volume},
    image=image,
    timeout=4 * 3600,
    cpu=GPU_WORKER_CPU,
    memory=GPU_WORKER_MEMORY_MB,
    secrets=[hf_secret],
)
def benchmark_standard_vllm_serve_modal(
    run_name: str = "standard_vllm_serve_benchmark_v1",
    model_id: str = "Qwen/Qwen3-30B-A3B",
    served_model_name: str = "benchmark-model",
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 32,
    max_model_len: int = 0,
    enable_chunked_prefill: bool = True,
    benchmark_max_tokens: int = 256,
    warmup_requests: int = 1,
    measured_requests: int = 3,
    fixed_prompt_num_prompts: int = 32,
    fixed_prompt_max_concurrency: int = 32,
    bench_input_len: int = 2048,
    bench_output_len: int = 256,
    bench_num_prompts: int = 64,
    bench_max_concurrency: int = 32,
    benchmark_messages_json: str = "",
) -> dict:
    from pipelines.interp.patching.serve_benchmark import (
        DEFAULT_BENCHMARK_MESSAGES,
        run_standard_vllm_serve_benchmark,
    )

    benchmark_messages = DEFAULT_BENCHMARK_MESSAGES
    if benchmark_messages_json.strip():
        parsed = json.loads(benchmark_messages_json)
        if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
            raise ValueError("benchmark_messages_json must decode to a list[dict]")
        benchmark_messages = parsed

    result = run_standard_vllm_serve_benchmark(
        output_dir=Path(f"/data/analysis_results/vllm_benchmarks/{run_name}"),
        model_id=f"/models/{model_id}",
        served_model_name=served_model_name,
        gpu_memory_utilization=float(gpu_memory_utilization),
        max_num_seqs=int(max_num_seqs),
        max_model_len=(None if int(max_model_len) <= 0 else int(max_model_len)),
        enable_chunked_prefill=bool(enable_chunked_prefill),
        benchmark_messages=benchmark_messages,
        benchmark_max_tokens=int(benchmark_max_tokens),
        warmup_requests=int(warmup_requests),
        measured_requests=int(measured_requests),
        fixed_prompt_num_prompts=int(fixed_prompt_num_prompts),
        fixed_prompt_max_concurrency=int(fixed_prompt_max_concurrency),
        bench_input_len=int(bench_input_len),
        bench_output_len=int(bench_output_len),
        bench_num_prompts=int(bench_num_prompts),
        bench_max_concurrency=int(bench_max_concurrency),
    )
    synthetic_volume.commit()
    return result


@app.function(
    gpu="H200",
    volumes={"/data": synthetic_volume, "/models": model_volume},
    image=image,
    timeout=6 * 3600,
    cpu=GPU_WORKER_CPU,
    memory=GPU_WORKER_MEMORY_MB,
    secrets=[hf_secret],
)
def benchmark_customop_vs_stock_vllm_modal(
    run_name: str = "customop_vs_stock_vllm_benchmark_v1",
    model_id: str = "Qwen/Qwen3-30B-A3B",
    served_model_name: str = "benchmark-model",
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 32,
    max_model_len: int = 0,
    enable_chunked_prefill: bool = True,
    benchmark_max_tokens: int = 256,
    warmup_requests: int = 1,
    measured_requests: int = 3,
    fixed_prompt_num_prompts: int = 32,
    fixed_prompt_max_concurrency: int = 32,
    measured_batch_runs: int = 3,
    bench_input_len: int = 2048,
    bench_output_len: int = 256,
    bench_num_prompts: int = 64,
    bench_max_concurrency: int = 32,
    patch_mode: str = "project_out",
    target_layers: str = "4,35",
    components_per_layer: int = 4,
    strength: float = 1.0,
    benchmark_messages_json: str = "",
) -> dict:
    from pipelines.interp.patching.serve_benchmark import (
        DEFAULT_BENCHMARK_MESSAGES,
        run_customop_vs_stock_vllm_benchmark,
    )

    benchmark_messages = DEFAULT_BENCHMARK_MESSAGES
    if benchmark_messages_json.strip():
        parsed = json.loads(benchmark_messages_json)
        if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
            raise ValueError("benchmark_messages_json must decode to a list[dict]")
        benchmark_messages = parsed

    layer_tuple = tuple(int(token) for token in target_layers.split(",") if token.strip())
    result = run_customop_vs_stock_vllm_benchmark(
        output_dir=Path(f"/data/analysis_results/vllm_benchmarks/{run_name}"),
        model_id=f"/models/{model_id}",
        served_model_name=served_model_name,
        gpu_memory_utilization=float(gpu_memory_utilization),
        max_num_seqs=int(max_num_seqs),
        max_model_len=(None if int(max_model_len) <= 0 else int(max_model_len)),
        enable_chunked_prefill=bool(enable_chunked_prefill),
        benchmark_messages=benchmark_messages,
        benchmark_max_tokens=int(benchmark_max_tokens),
        warmup_requests=int(warmup_requests),
        measured_requests=int(measured_requests),
        fixed_prompt_num_prompts=int(fixed_prompt_num_prompts),
        fixed_prompt_max_concurrency=int(fixed_prompt_max_concurrency),
        measured_batch_runs=int(measured_batch_runs),
        bench_input_len=int(bench_input_len),
        bench_output_len=int(bench_output_len),
        bench_num_prompts=int(bench_num_prompts),
        bench_max_concurrency=int(bench_max_concurrency),
        patch_mode=patch_mode,
        target_layers=layer_tuple or (4, 35),
        components_per_layer=int(components_per_layer),
        strength=float(strength),
        basis_npz_path=Path(BASIS_NPZ_REMOTE),
        results_json_path=Path(BASIS_RESULTS_REMOTE),
    )
    synthetic_volume.commit()
    return result


@app.function(
    gpu="H200",
    volumes={"/data": synthetic_volume, "/models": model_volume},
    image=image,
    timeout=4 * 3600,
    cpu=GPU_WORKER_CPU,
    memory=GPU_WORKER_MEMORY_MB,
    secrets=[hf_secret],
)
def benchmark_direct_vllm_variant_modal(
    run_name: str = "direct_vllm_variant_benchmark_v1",
    model_id: str = "Qwen/Qwen3-30B-A3B",
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 32,
    enable_chunked_prefill: bool = True,
    enable_prefix_caching: bool = False,
    use_custom_worker: bool = True,
    benchmark_max_tokens: int = 256,
    warmup_requests: int = 1,
    measured_requests: int = 3,
    fixed_prompt_num_prompts: int = 32,
    measured_batch_runs: int = 3,
    patch_mode: str = "",
    target_layers: str = "4,35",
    components_per_layer: int = 4,
    strength: float = 1.0,
    benchmark_messages_json: str = "",
) -> dict:
    from pipelines.interp.patching.serve_benchmark import (
        DEFAULT_BENCHMARK_MESSAGES,
        run_customop_vllm_benchmark,
    )

    benchmark_messages = DEFAULT_BENCHMARK_MESSAGES
    if benchmark_messages_json.strip():
        parsed = json.loads(benchmark_messages_json)
        if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
            raise ValueError("benchmark_messages_json must decode to a list[dict]")
        benchmark_messages = parsed

    layer_tuple = tuple(int(token) for token in target_layers.split(",") if token.strip())
    result = run_customop_vllm_benchmark(
        output_dir=Path(f"/data/analysis_results/vllm_benchmarks/{run_name}"),
        model_id=f"/models/{model_id}",
        benchmark_messages=benchmark_messages,
        benchmark_max_tokens=int(benchmark_max_tokens),
        warmup_requests=int(warmup_requests),
        measured_requests=int(measured_requests),
        fixed_prompt_num_prompts=int(fixed_prompt_num_prompts),
        measured_batch_runs=int(measured_batch_runs),
        max_num_seqs=int(max_num_seqs),
        gpu_memory_utilization=float(gpu_memory_utilization),
        enable_chunked_prefill=bool(enable_chunked_prefill),
        enable_prefix_caching=bool(enable_prefix_caching),
        use_custom_worker=bool(use_custom_worker),
        patch_mode=patch_mode,
        target_layers=layer_tuple or (4, 35),
        components_per_layer=int(components_per_layer),
        strength=float(strength),
        basis_npz_path=Path(BASIS_NPZ_REMOTE),
        results_json_path=Path(BASIS_RESULTS_REMOTE),
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
    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_patching_analysis import (
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
    from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_analysis import (
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
