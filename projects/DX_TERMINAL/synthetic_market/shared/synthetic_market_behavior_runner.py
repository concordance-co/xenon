from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipelines.interp.pooling import _extract_system_user, _parse_messages
from pipelines.interp.tool_schemas import resolve_tool_schema_mode
from projects.DX_TERMINAL.synthetic_market.shared.patch_basis import load_phase17_activation_patch_basis
from pipelines.datasets.synthetic.pairing import build_matched_metric_examples
from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_patching_runner import (
    _build_patch_spec,
    _load_examples,
    _parse_component_indices_spec,
)
from pipelines.datasets.synthetic.structure import find_synthetic_section_boundaries
from pipelines.interp.modal_vllm_engine import (
    VLLMCaptureConfig,
    _build_chat_template_kwargs,
    _capture_batch_vllm,
    _capture_one_vllm,
    _create_llm,
    _generate_batch_vllm,
    _generate_one_vllm,
    _init_activation_patching_on_model,
    _register_activation_patch_basis_on_model,
)


@dataclass(slots=True)
class SyntheticMarketBehaviorConfig:
    phase_name: str = "phase15_market_basis_discovery_v1"
    output_dir: Path = field(
        default_factory=lambda: Path("data/analysis_results/synthetic_market_behavior/phase18_market_behavior_v1")
    )
    model_id: str = "Qwen/Qwen3-30B-A3B"
    context_variant: str = "market_only"
    order_mode: str = "selection_rank_asc"
    selection_strategy: str = "ordered"
    limit: int | None = None
    family_allowlist: tuple[str, ...] = ()
    example_id_allowlist: tuple[str, ...] = ()
    pair_metric: str = ""
    pair_mode: str = ""
    min_pair_gap: float = 0.0
    generate_source_behavior: bool = False
    donor_means_path: Path | None = None
    batch_size: int = 1
    patch_mode: str = ""
    target_layers: tuple[int, ...] = (4,)
    components_per_layer: int = 4
    component_indices_by_layer: dict[int, tuple[int, ...]] = field(default_factory=dict)
    direction_name: str = ""
    strength: float = 1.0
    secondary_patch_mode: str = ""
    secondary_target_layers: tuple[int, ...] = ()
    secondary_components_per_layer: int = 4
    secondary_component_indices_by_layer: dict[int, tuple[int, ...]] = field(default_factory=dict)
    secondary_direction_name: str = ""
    secondary_strength: float = 1.0
    random_seed: int = 42
    max_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    tool_schema_mode: str = ""
    tool_choice: str = ""
    add_generation_prompt: bool = True
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    enforce_eager: bool = True
    enable_chunked_prefill: bool = False
    enable_logging_iteration_details: bool = False
    enable_mfu_metrics: bool = False
    basis_state_key: str = "market_mean"
    basis_npz_path: Path = Path(
        "data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/pca_basis.npz"
    )
    basis_results_path: Path = Path(
        "data/analysis_results/synthetic_market_axis_decomposition/phase17_market_axis_decomposition_v1/results.json"
    )

    @property
    def patch_enabled(self) -> bool:
        return bool(self.patch_mode and self.patch_mode.lower() != "none")

    @property
    def paired_mode_enabled(self) -> bool:
        return bool(self.pair_metric.strip() and self.pair_mode.strip())

    @property
    def secondary_patch_enabled(self) -> bool:
        return bool(self.secondary_patch_mode and self.secondary_patch_mode.lower() != "none")


def _requested_patch_layers(config: SyntheticMarketBehaviorConfig) -> tuple[int, ...]:
    layers: list[int] = []
    if config.patch_enabled:
        layers.extend(int(layer) for layer in config.target_layers)
    if config.secondary_patch_enabled:
        layers.extend(int(layer) for layer in config.secondary_target_layers)
    if not layers:
        return tuple(int(layer) for layer in config.target_layers)
    return tuple(sorted(set(layers)))


def _requested_basis_components(config: SyntheticMarketBehaviorConfig) -> int:
    values = [int(config.components_per_layer)]
    if config.secondary_patch_enabled:
        values.append(int(config.secondary_components_per_layer))
    return max(1, max(values))


def _build_behavior_patch_spec(
    *,
    patch_mode: str,
    target_layers: tuple[int, ...],
    components_per_layer: int,
    component_indices_by_layer: dict[int, tuple[int, ...]],
    direction_name: str,
    strength: float,
    random_seed: int,
    market_span: tuple[int, int],
    basis_payload: dict[int, dict[str, Any]],
    donor_mean_by_layer: dict[int, Any] | None = None,
):
    spec_config = SimpleNamespace(
        patch_mode=patch_mode,
        target_layers=target_layers,
        components_per_layer=components_per_layer,
        component_indices_by_layer=component_indices_by_layer,
        direction_name=direction_name,
        strength=strength,
        random_seed=random_seed,
    )
    return _build_patch_spec(
        config=spec_config,
        market_span=market_span,
        basis_payload=basis_payload,
        donor_mean_by_layer=donor_mean_by_layer,
    )


def _flush_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    size = max(1, int(chunk_size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _decode_first_token(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    try:
        return str(tokenizer.decode([int(token_ids[0])]))
    except Exception:
        return ""


_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _normalize_tool_call_payload(payload: Any, *, raw_json: str) -> dict[str, Any]:
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    arguments = payload.get("arguments", {}) if isinstance(payload, dict) else {}
    return {
        "has_tool_call": payload is not None,
        "tool_call_parse_ok": isinstance(payload, dict),
        "first_tool_name": payload.get("name") if isinstance(payload, dict) else None,
        "first_tool_token": arguments.get("token") if isinstance(arguments, dict) else None,
        "first_tool_spend_pct": arguments.get("spend_pct") if isinstance(arguments, dict) else None,
        "first_tool_strategy": arguments.get("strategy") if isinstance(arguments, dict) else None,
        "first_tool_content": arguments.get("content") if isinstance(arguments, dict) else None,
        "first_tool_raw_json": raw_json,
    }


def _extract_first_tool_call_fields(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    json_candidates: list[str] = []
    if stripped.startswith("{") or stripped.startswith("["):
        json_candidates.append(stripped)
    if "</think>" in stripped:
        suffix = stripped.rsplit("</think>", 1)[1].strip()
        if suffix.startswith("{") or suffix.startswith("["):
            json_candidates.append(suffix)
    for marker in ("\n{", "\n["):
        idx = stripped.rfind(marker)
        if idx >= 0:
            suffix = stripped[idx + 1 :].strip()
            if suffix.startswith("{") or suffix.startswith("["):
                json_candidates.append(suffix)

    for candidate in json_candidates:
        try:
            payload = json.loads(candidate)
            return _normalize_tool_call_payload(payload, raw_json=candidate)
        except json.JSONDecodeError:
            continue

    match = _TOOL_CALL_PATTERN.search(text or "")
    if match is None:
        return {
            "has_tool_call": False,
            "tool_call_parse_ok": False,
            "first_tool_name": None,
            "first_tool_token": None,
            "first_tool_spend_pct": None,
            "first_tool_strategy": None,
            "first_tool_content": None,
            "first_tool_raw_json": None,
        }
    raw_json = match.group(1)
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        return {
            "has_tool_call": True,
            "tool_call_parse_ok": False,
            "first_tool_name": None,
            "first_tool_token": None,
            "first_tool_spend_pct": None,
            "first_tool_strategy": None,
            "first_tool_content": None,
            "first_tool_raw_json": raw_json,
        }
    return _normalize_tool_call_payload(payload, raw_json=raw_json)


def _compute_donor_mean_by_layer(
    *,
    llm_capture: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    capture_cfg: VLLMCaptureConfig,
    source_log_id: int,
    market_span: tuple[int, int],
    target_layers: tuple[int, ...],
) -> dict[int, Any]:
    residual, _, _, _ = _capture_one_vllm(
        llm=llm_capture,
        tokenizer=tokenizer,
        messages=messages,
        config=capture_cfg,
        log_id=f"swap_mean_src_{source_log_id}",
        patch_spec=None,
        skip_residual_save=True,
    )
    if residual is None:
        raise RuntimeError(f"Residual capture returned None for source log_id={source_log_id}")

    residual_np = residual.detach().cpu().numpy() if hasattr(residual, "detach") else residual
    start, end = market_span
    donor_mean_by_layer: dict[int, Any] = {}
    for layer in target_layers:
        donor_mean_by_layer[int(layer)] = residual_np[int(layer), int(start):int(end)].mean(axis=0).astype("float32")
    return donor_mean_by_layer


def _compute_batched_donor_means(
    *,
    llm_capture: Any,
    tokenizer: Any,
    prepared_rows: list[dict[str, Any]],
    capture_cfg: VLLMCaptureConfig,
    target_layers: tuple[int, ...],
    batch_size: int,
) -> dict[int, dict[int, Any]]:
    unique_sources: list[dict[str, Any]] = []
    seen_source_log_ids: set[int] = set()
    for prepared in prepared_rows:
        source_log_id = prepared["source_log_id"]
        source_row_messages = prepared["source_row_messages"]
        source_market_span = prepared["source_market_span"]
        if source_log_id is None or source_row_messages is None or source_market_span is None:
            continue
        source_log_id = int(source_log_id)
        if source_log_id in seen_source_log_ids:
            continue
        seen_source_log_ids.add(source_log_id)
        unique_sources.append(
            {
                "id": str(source_log_id),
                "source_log_id": source_log_id,
                "messages": source_row_messages,
                "market_span": source_market_span,
            }
        )

    donor_mean_cache: dict[int, dict[int, Any]] = {}
    for chunk in _chunk_list(unique_sources, max(1, int(batch_size))):
        captured = _capture_batch_vllm(
            llm=llm_capture,
            tokenizer=tokenizer,
            prompts=[{"id": item["id"], "messages": item["messages"]} for item in chunk],
            config=capture_cfg,
        )
        captured_by_id = {str(prompt_id): residual for residual, _input_ids, prompt_id in captured}
        for item in chunk:
            residual = captured_by_id.get(str(item["source_log_id"]))
            if residual is None:
                raise RuntimeError(
                    f"Residual capture returned None for source log_id={item['source_log_id']}"
                )
            residual_np = residual.detach().cpu().numpy() if hasattr(residual, "detach") else residual
            start, end = item["market_span"]
            donor_mean_by_layer: dict[int, Any] = {}
            for layer in target_layers:
                donor_mean_by_layer[int(layer)] = (
                    residual_np[int(layer), int(start):int(end)].mean(axis=0).astype("float32")
                )
            donor_mean_cache[int(item["source_log_id"])] = donor_mean_by_layer
    return donor_mean_cache


def _destroy_llm(llm: Any | None) -> None:
    if llm is None:
        return
    try:
        llm_engine = getattr(llm, "llm_engine", None)
        shutdown = getattr(llm_engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        pass


def _cleanup_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _prepare_behavior_rows(
    *,
    config: SyntheticMarketBehaviorConfig,
    tokenizer: Any,
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], int]:
    load_limit = None if config.paired_mode_enabled else config.limit
    examples = _load_examples(
        phase_name=config.phase_name,
        context_variant=config.context_variant,
        order_mode=config.order_mode,
        selection_strategy=config.selection_strategy,
        limit=load_limit,
        family_allowlist=config.family_allowlist,
    )
    if config.paired_mode_enabled:
        examples = build_matched_metric_examples(
            examples,
            phase_name=config.phase_name,
            pair_metric=config.pair_metric,
            pair_mode=config.pair_mode,
            min_metric_gap=float(config.min_pair_gap),
            limit=config.limit,
        )
    if config.example_id_allowlist:
        allowed = set(config.example_id_allowlist)
        examples = [row for row in examples if str(row.get("example_id") or "") in allowed]

    prepared_rows: list[dict[str, Any]] = []
    skipped = 0
    for row in examples:
        messages = _parse_messages(row["prompt_messages_json"])
        system_user = _extract_system_user(messages)
        if system_user is None:
            skipped += 1
            continue

        system_text, user_text = system_user
        market_span = find_synthetic_section_boundaries(
            tokenizer,
            system_text,
            user_text,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        ).get("market")
        if market_span is None:
            skipped += 1
            continue

        source_log_id = None
        source_row_messages = None
        source_market_span = None
        if row.get("source_prompt_messages_json"):
            source_log_id = int(row["source_log_id"])
            source_row_messages = _parse_messages(row["source_prompt_messages_json"])
            source_system_user = _extract_system_user(source_row_messages)
            if source_system_user is None:
                skipped += 1
                continue
            source_system_text, source_user_text = source_system_user
            source_market_span = find_synthetic_section_boundaries(
                tokenizer,
                source_system_text,
                source_user_text,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
            ).get("market")
            if source_market_span is None:
                skipped += 1
                continue

        prepared_rows.append(
            {
                "row": row,
                "messages": messages,
                "market_span": market_span,
                "source_log_id": source_log_id,
                "source_row_messages": source_row_messages,
                "source_market_span": source_market_span,
            }
        )
    return prepared_rows, skipped


def _save_donor_mean_cache(path: Path, donor_mean_cache: dict[int, dict[int, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    for source_log_id, layer_map in donor_mean_cache.items():
        for layer, donor_mean in layer_map.items():
            arrays[f"src_{int(source_log_id)}_layer_{int(layer)}"] = np.asarray(donor_mean, dtype=np.float32)
    np.savez(path, **arrays)


def _load_donor_mean_cache(path: Path) -> dict[int, dict[int, Any]]:
    donor_mean_cache: dict[int, dict[int, Any]] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            match = re.fullmatch(r"src_(\d+)_layer_(\d+)", key)
            if match is None:
                continue
            source_log_id = int(match.group(1))
            layer = int(match.group(2))
            donor_mean_cache.setdefault(source_log_id, {})[layer] = np.asarray(data[key], dtype=np.float32)
    return donor_mean_cache


def prepare_synthetic_market_behavior_donors(config: SyntheticMarketBehaviorConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config.output_dir.mkdir(parents=True, exist_ok=True)
    donor_path = config.donor_means_path or (config.output_dir / "donor_means.npz")

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tools = resolve_tool_schema_mode(config.tool_schema_mode)
    tool_choice = config.tool_choice.strip() or None
    chat_template_kwargs = _build_chat_template_kwargs(tools=tools, tool_choice=tool_choice)

    prepared_rows, skipped = _prepare_behavior_rows(
        config=config,
        tokenizer=tokenizer,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )
    if not prepared_rows:
        result = {"error": "no_valid_examples"}
        (config.output_dir / "donor_results.json").write_text(json.dumps(result, indent=2))
        return result

    capture_cfg = VLLMCaptureConfig(
        output_dir=config.output_dir / "_tmp_capture",
        model_id=config.model_id,
        capture_router=False,
        capture_residual=True,
        add_generation_prompt=bool(config.add_generation_prompt),
        tensor_parallel_size=int(config.tensor_parallel_size),
        gpu_memory_utilization=float(config.gpu_memory_utilization),
        max_num_seqs=max(1, int(config.batch_size)),
        max_num_batched_tokens=max(40960, max(1, int(config.batch_size)) * 4096),
        enable_prefix_caching=False,
    )
    donor_target_layers = _requested_patch_layers(config)
    llm_capture = _create_llm(capture_cfg)
    try:
        donor_mean_cache = _compute_batched_donor_means(
            llm_capture=llm_capture,
            tokenizer=tokenizer,
            prepared_rows=prepared_rows,
            capture_cfg=capture_cfg,
            target_layers=donor_target_layers,
            batch_size=max(1, int(config.batch_size)),
        )
    finally:
        _destroy_llm(llm_capture)
        _cleanup_cuda_memory()

    _save_donor_mean_cache(donor_path, donor_mean_cache)
    result = {
        "phase_name": config.phase_name,
        "prepared_examples": len(prepared_rows),
        "skipped": skipped,
        "donor_source_count": len(donor_mean_cache),
        "target_layers": [int(layer) for layer in donor_target_layers],
        "pair_metric": config.pair_metric,
        "pair_mode": config.pair_mode,
        "donor_means_path": str(donor_path),
    }
    (config.output_dir / "donor_results.json").write_text(json.dumps(result, indent=2))
    return result


def _build_generation_config(config: SyntheticMarketBehaviorConfig) -> VLLMCaptureConfig:
    batch_size = max(1, int(config.batch_size))
    # Keep batched behavior runs on the same vLLM worker path across baseline,
    # project-out, and random-control conditions. Patched requests carry
    # `patch_spec` in SamplingParams.extra_args; baseline requests leave
    # it empty but still benefit from the same scheduler / runner codepath.
    use_request_scoped_patching = batch_size > 1 or bool(config.secondary_patch_enabled)
    max_num_batched_tokens = None
    if batch_size > 1:
        max_num_batched_tokens = max(40960, batch_size * 4096)
    async_scheduling = None
    if batch_size > 1:
        # Keep batched experiment conditions aligned across baseline and
        # patched runs. vLLM 0.17 can auto-enable async scheduling when left
        # unset, which changes scheduler behavior and throughput.
        async_scheduling = False
    return VLLMCaptureConfig(
        output_dir=config.output_dir / "_tmp_capture",
        model_id=config.model_id,
        capture_router=False,
        capture_residual=False,
        add_generation_prompt=bool(config.add_generation_prompt),
        tensor_parallel_size=int(config.tensor_parallel_size),
        gpu_memory_utilization=float(config.gpu_memory_utilization),
        enforce_eager=bool(config.enforce_eager),
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=batch_size,
        enable_prefix_caching=False,
        enable_chunked_prefill=bool(config.enable_chunked_prefill),
        async_scheduling=async_scheduling,
        worker_cls=(
            "pipelines.interp.patching.activation_patch_request_worker.ActivationPatchGPUWorker"
            if use_request_scoped_patching
            else ""
        ),
        request_scoped_patching=use_request_scoped_patching,
        enable_logging_iteration_details=bool(config.enable_logging_iteration_details),
        enable_mfu_metrics=bool(config.enable_mfu_metrics),
    )


def _run_generation_batch(
    *,
    llm: Any,
    tokenizer: Any,
    requests: list[dict[str, Any]],
    config: VLLMCaptureConfig,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not requests:
        return []
    if len(requests) == 1:
        request = requests[0]
        return [
            _generate_one_vllm(
                llm=llm,
                tokenizer=tokenizer,
                messages=request["messages"],
                config=config,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                patch_spec=request.get("patch_spec"),
                tools=tools,
                tool_choice=tool_choice,
                chat_template_kwargs=chat_template_kwargs,
            )
        ]
    return _generate_batch_vllm(
        llm=llm,
        tokenizer=tokenizer,
        batch_requests=requests,
        config=config,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        tools=tools,
        tool_choice=tool_choice,
        chat_template_kwargs=chat_template_kwargs,
    )


def run_synthetic_market_behavior(config: SyntheticMarketBehaviorConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = config.output_dir / "metadata.parquet"

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tools = resolve_tool_schema_mode(config.tool_schema_mode)
    tool_choice = config.tool_choice.strip() or None
    chat_template_kwargs = _build_chat_template_kwargs(tools=tools, tool_choice=tool_choice)

    prepared_rows, skipped = _prepare_behavior_rows(
        config=config,
        tokenizer=tokenizer,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )

    if not prepared_rows:
        result = {"error": "no_valid_examples"}
        (config.output_dir / "results.json").write_text(json.dumps(result, indent=2))
        return result

    needs_donor_capture = bool(
        config.paired_mode_enabled
        and (
            (config.patch_enabled and config.patch_mode in {"swap_mean", "swap_components"})
            or (config.secondary_patch_enabled and config.secondary_patch_mode in {"swap_mean", "swap_components"})
        )
    )
    donor_mean_cache: dict[int, dict[int, Any]] = {}
    requested_patch_layers = _requested_patch_layers(config)
    if config.donor_means_path is not None:
        donor_mean_cache = _load_donor_mean_cache(config.donor_means_path)
    elif needs_donor_capture:
        capture_cfg = VLLMCaptureConfig(
            output_dir=config.output_dir / "_tmp_capture",
            model_id=config.model_id,
            capture_router=False,
            capture_residual=True,
            add_generation_prompt=bool(config.add_generation_prompt),
            tensor_parallel_size=int(config.tensor_parallel_size),
            gpu_memory_utilization=float(config.gpu_memory_utilization),
            max_num_seqs=max(1, int(config.batch_size)),
            max_num_batched_tokens=max(40960, max(1, int(config.batch_size)) * 4096),
            enable_prefix_caching=False,
        )
        llm_capture = _create_llm(capture_cfg)
        try:
            donor_mean_cache = _compute_batched_donor_means(
                llm_capture=llm_capture,
                tokenizer=tokenizer,
                prepared_rows=prepared_rows,
                capture_cfg=capture_cfg,
                target_layers=requested_patch_layers,
                batch_size=max(1, int(config.batch_size)),
            )
        finally:
            _destroy_llm(llm_capture)
            _cleanup_cuda_memory()

    generate_cfg = _build_generation_config(config)
    llm_generate = _create_llm(generate_cfg)

    basis_payload: dict[int, dict[str, Any]] = {}
    if config.patch_enabled or config.secondary_patch_enabled:
        basis = load_phase17_activation_patch_basis(
            basis_npz_path=config.basis_npz_path,
            results_json_path=config.basis_results_path,
            state_key=config.basis_state_key,
            layers=requested_patch_layers,
            components_per_layer=_requested_basis_components(config),
        )
        basis_payload = basis.to_payload()
        _init_activation_patching_on_model(llm_generate)
        _register_activation_patch_basis_on_model(llm_generate, basis_payload)

    source_behavior_cache: dict[int, dict[str, Any]] = {}
    metadata_rows: list[dict[str, Any]] = []
    processed = 0
    try:
        if config.generate_source_behavior:
            unique_source_requests: list[dict[str, Any]] = []
            seen_source_log_ids: set[int] = set()
            for prepared in prepared_rows:
                source_log_id = prepared["source_log_id"]
                source_row_messages = prepared["source_row_messages"]
                if source_log_id is None or source_row_messages is None or source_log_id in seen_source_log_ids:
                    continue
                seen_source_log_ids.add(int(source_log_id))
                unique_source_requests.append(
                    {
                        "source_log_id": int(source_log_id),
                        "messages": source_row_messages,
                    }
                )

            for source_chunk in _chunk_list(unique_source_requests, int(config.batch_size)):
                requests = [{"messages": item["messages"]} for item in source_chunk]
                outputs = _run_generation_batch(
                    llm=llm_generate,
                    tokenizer=tokenizer,
                    requests=requests,
                    config=generate_cfg,
                    max_tokens=int(config.max_tokens),
                    temperature=float(config.temperature),
                    top_p=float(config.top_p),
                    top_k=int(config.top_k),
                    tools=tools,
                    tool_choice=tool_choice,
                    chat_template_kwargs=chat_template_kwargs,
                )
                for item, source_output in zip(source_chunk, outputs, strict=False):
                    source_generated_token_ids = [int(token) for token in source_output["generated_token_ids"]]
                    source_tool_fields = _extract_first_tool_call_fields(str(source_output["generated_text"]))
                    source_behavior_cache[int(item["source_log_id"])] = {
                        "source_generated_text": source_output["generated_text"],
                        "source_generated_token_count": len(source_generated_token_ids),
                        "source_first_generated_token_id": (
                            source_generated_token_ids[0] if source_generated_token_ids else None
                        ),
                        "source_first_generated_token_text": _decode_first_token(tokenizer, source_generated_token_ids),
                        **{f"source_{key}": value for key, value in source_tool_fields.items()},
                    }

        for prepared_chunk in _chunk_list(prepared_rows, int(config.batch_size)):
            batch_requests: list[dict[str, Any]] = []
            chunk_context: list[dict[str, Any]] = []
            for prepared in prepared_chunk:
                row = prepared["row"]
                messages = prepared["messages"]
                market_span = prepared["market_span"]
                source_log_id = prepared["source_log_id"]
                source_row_messages = prepared["source_row_messages"]
                source_market_span = prepared["source_market_span"]

                pair_metric_name = row.get("pair_metric_name")
                pair_metric_value = row.get("base_pair_metric_value", row.get("pair_metric_value"))
                source_pair_metric_value = row.get("source_pair_metric_value")
                pair_metric_gap = row.get("pair_metric_gap")
                source_behavior_fields: dict[str, Any] = {}
                if config.generate_source_behavior and source_log_id is not None:
                    source_behavior_fields = dict(source_behavior_cache.get(int(source_log_id), {}))

                patch_specs: list[dict[str, Any]] = []
                if config.patch_enabled:
                    if config.patch_mode in {"swap_mean", "swap_components"}:
                        if (
                            not config.paired_mode_enabled
                            or source_row_messages is None
                            or source_market_span is None
                            or source_log_id is None
                        ):
                            raise ValueError(
                                f"{config.patch_mode} behavior runs require paired mode with a valid source row"
                            )
                        donor_mean_by_layer = donor_mean_cache.get(int(source_log_id))
                        if donor_mean_by_layer is None:
                            raise RuntimeError(f"Missing donor mean cache for source log_id={source_log_id}")
                        patch_specs.append(_build_behavior_patch_spec(
                            patch_mode=config.patch_mode,
                            target_layers=tuple(int(layer) for layer in config.target_layers),
                            components_per_layer=int(config.components_per_layer),
                            component_indices_by_layer=config.component_indices_by_layer,
                            direction_name=config.direction_name,
                            strength=float(config.strength),
                            random_seed=int(config.random_seed),
                            market_span=market_span,
                            basis_payload=basis_payload,
                            donor_mean_by_layer=donor_mean_by_layer,
                        ).to_payload())
                    else:
                        patch_specs.append(_build_behavior_patch_spec(
                            patch_mode=config.patch_mode,
                            target_layers=tuple(int(layer) for layer in config.target_layers),
                            components_per_layer=int(config.components_per_layer),
                            component_indices_by_layer=config.component_indices_by_layer,
                            direction_name=config.direction_name,
                            strength=float(config.strength),
                            random_seed=int(config.random_seed),
                            market_span=market_span,
                            basis_payload=basis_payload,
                        ).to_payload())
                if config.secondary_patch_enabled:
                    donor_mean_by_layer = None
                    if config.secondary_patch_mode in {"swap_mean", "swap_components"}:
                        if (
                            not config.paired_mode_enabled
                            or source_row_messages is None
                            or source_market_span is None
                            or source_log_id is None
                        ):
                            raise ValueError(
                                f"{config.secondary_patch_mode} behavior runs require paired mode with a valid source row"
                            )
                        donor_mean_by_layer = donor_mean_cache.get(int(source_log_id))
                        if donor_mean_by_layer is None:
                            raise RuntimeError(f"Missing donor mean cache for source log_id={source_log_id}")
                    patch_specs.append(_build_behavior_patch_spec(
                        patch_mode=config.secondary_patch_mode,
                        target_layers=tuple(int(layer) for layer in config.secondary_target_layers),
                        components_per_layer=int(config.secondary_components_per_layer),
                        component_indices_by_layer=config.secondary_component_indices_by_layer,
                        direction_name=config.secondary_direction_name,
                        strength=float(config.secondary_strength),
                        random_seed=int(config.random_seed),
                        market_span=market_span,
                        basis_payload=basis_payload,
                        donor_mean_by_layer=donor_mean_by_layer,
                    ).to_payload())

                batch_requests.append(
                    {
                        "messages": messages,
                        "patch_spec": patch_specs[0] if len(patch_specs) == 1 else None,
                        "patch_specs": patch_specs if len(patch_specs) > 1 else None,
                    }
                )
                chunk_context.append(
                    {
                        "row": row,
                        "pair_metric_name": pair_metric_name,
                        "pair_metric_value": pair_metric_value,
                        "source_pair_metric_value": source_pair_metric_value,
                        "pair_metric_gap": pair_metric_gap,
                        "source_log_id": source_log_id,
                        "source_behavior_fields": source_behavior_fields,
                    }
                )

            outputs = _run_generation_batch(
                llm=llm_generate,
                tokenizer=tokenizer,
                requests=batch_requests,
                config=generate_cfg,
                max_tokens=int(config.max_tokens),
                temperature=float(config.temperature),
                top_p=float(config.top_p),
                top_k=int(config.top_k),
                tools=tools,
                tool_choice=tool_choice,
                chat_template_kwargs=chat_template_kwargs,
            )

            for context, output in zip(chunk_context, outputs, strict=False):
                row = context["row"]
                generated_token_ids = [int(token) for token in output["generated_token_ids"]]
                tool_fields = _extract_first_tool_call_fields(str(output["generated_text"]))
                metadata_rows.append(
                    {
                        "log_id": int(row["log_id"]),
                        "phase_name": row["phase_name"],
                        "example_id": row["example_id"],
                        "family": row["family"],
                        "family_variant": row["family_variant"],
                        "context_variant": row["context_variant"],
                        "roster_key": row.get("roster_key") or "",
                        "pair_metric_name": context["pair_metric_name"] or None,
                        "pair_mode": row.get("pair_mode") or None,
                        "pair_id": row.get("pair_id"),
                        "pair_metric_value": (
                            float(context["pair_metric_value"])
                            if context["pair_metric_value"] is not None
                            else None
                        ),
                        "source_pair_metric_value": (
                            float(context["source_pair_metric_value"])
                            if context["source_pair_metric_value"] is not None
                            else None
                        ),
                        "pair_metric_gap": (
                            float(context["pair_metric_gap"]) if context["pair_metric_gap"] is not None else None
                        ),
                        "source_log_id": context["source_log_id"],
                        "source_example_id": row.get("source_example_id"),
                        "source_family_variant": row.get("source_family_variant"),
                        "source_roster_key": row.get("source_roster_key"),
                        "capture_timestamp": datetime.now(UTC).isoformat(),
                        "seq_len": int(len(output["input_ids"])),
                        "max_tokens": int(config.max_tokens),
                        "temperature": float(config.temperature),
                        "top_p": float(config.top_p),
                        "top_k": int(config.top_k),
                        "tool_schema_mode": config.tool_schema_mode,
                        "tool_choice": config.tool_choice,
                        "tool_count": len(tools) if tools is not None else 0,
                        "patch_mode": (
                            config.patch_mode
                            if not config.secondary_patch_enabled
                            else f"{config.patch_mode}+{config.secondary_patch_mode}"
                        ),
                        "primary_patch_mode": config.patch_mode,
                        "secondary_patch_mode": config.secondary_patch_mode or None,
                        "target_layers": ",".join(str(int(layer)) for layer in config.target_layers),
                        "secondary_target_layers": ",".join(
                            str(int(layer)) for layer in config.secondary_target_layers
                        ),
                        "components_per_layer": int(config.components_per_layer),
                        "component_indices_by_layer_json": json.dumps(
                            {
                                str(layer): [int(index) for index in indices]
                                for layer, indices in config.component_indices_by_layer.items()
                            }
                        ),
                        "secondary_components_per_layer": int(config.secondary_components_per_layer),
                        "secondary_component_indices_by_layer_json": json.dumps(
                            {
                                str(layer): [int(index) for index in indices]
                                for layer, indices in config.secondary_component_indices_by_layer.items()
                            }
                        ),
                        "basis_state_key": config.basis_state_key,
                        "direction_name": config.direction_name,
                        "secondary_direction_name": config.secondary_direction_name or None,
                        "selection_strategy": config.selection_strategy,
                        "strength": float(config.strength),
                        "secondary_strength": float(config.secondary_strength),
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
    finally:
        _destroy_llm(llm_generate)
        _cleanup_cuda_memory()

    _flush_table(metadata_path, metadata_rows)

    result = {
        "phase_name": config.phase_name,
        "processed": processed,
        "skipped": skipped,
        "output_dir": str(config.output_dir),
        "patch_mode": (
            config.patch_mode
            if not config.secondary_patch_enabled
            else f"{config.patch_mode}+{config.secondary_patch_mode}"
        ),
        "patch_enabled": bool(config.patch_enabled or config.secondary_patch_enabled),
        "primary_patch_mode": config.patch_mode,
        "secondary_patch_mode": config.secondary_patch_mode or None,
        "pair_metric": config.pair_metric,
        "pair_mode": config.pair_mode,
        "generate_source_behavior": bool(config.generate_source_behavior),
        "batch_size": max(1, int(config.batch_size)),
        "target_layers": [int(layer) for layer in config.target_layers],
        "secondary_target_layers": [int(layer) for layer in config.secondary_target_layers],
        "components_per_layer": int(config.components_per_layer),
        "component_indices_by_layer": {
            str(layer): [int(index) for index in indices]
            for layer, indices in config.component_indices_by_layer.items()
        },
        "secondary_components_per_layer": int(config.secondary_components_per_layer),
        "secondary_component_indices_by_layer": {
            str(layer): [int(index) for index in indices]
            for layer, indices in config.secondary_component_indices_by_layer.items()
        },
        "max_tokens": int(config.max_tokens),
        "selection_strategy": config.selection_strategy,
        "basis_state_key": config.basis_state_key,
        "donor_means_path": str(config.donor_means_path) if config.donor_means_path is not None else None,
    }
    (config.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic market behavior patching on the current vLLM stack.")
    parser.add_argument("--phase-name", default="phase15_market_basis_discovery_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis_results/synthetic_market_behavior/phase18_market_behavior_v1"),
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--context-variant", default="market_only")
    parser.add_argument("--order-mode", default="selection_rank_asc")
    parser.add_argument("--selection-strategy", default="ordered")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--family-allowlist", default="")
    parser.add_argument("--example-id-allowlist", default="")
    parser.add_argument("--pair-metric", default="")
    parser.add_argument("--pair-mode", default="")
    parser.add_argument("--min-pair-gap", type=float, default=0.0)
    parser.add_argument("--generate-source-behavior", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-mode", default="")
    parser.add_argument("--target-layers", default="4")
    parser.add_argument("--components-per-layer", type=int, default=4)
    parser.add_argument("--component-indices", default="")
    parser.add_argument("--direction-name", default="")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--secondary-patch-mode", default="")
    parser.add_argument("--secondary-target-layers", default="")
    parser.add_argument("--secondary-components-per-layer", type=int, default=4)
    parser.add_argument("--secondary-component-indices", default="")
    parser.add_argument("--secondary-direction-name", default="")
    parser.add_argument("--secondary-strength", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--tool-schema-mode", default="")
    parser.add_argument("--tool-choice", default="")
    parser.add_argument("--add-generation-prompt", action="store_true")
    parser.add_argument("--donor-means-path", type=Path, default=None)
    parser.add_argument("--no-enforce-eager", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--enable-logging-iteration-details", action="store_true")
    parser.add_argument("--enable-mfu-metrics", action="store_true")
    parser.add_argument("--basis-state-key", default="market_mean")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    target_layers = tuple(int(token) for token in args.target_layers.split(",") if token.strip())
    component_indices_by_layer = _parse_component_indices_spec(
        args.component_indices,
        target_layers=target_layers or (4,),
    )
    secondary_target_layers = tuple(int(token) for token in args.secondary_target_layers.split(",") if token.strip())
    secondary_component_indices_by_layer = _parse_component_indices_spec(
        args.secondary_component_indices,
        target_layers=secondary_target_layers or (4,),
    )
    family_allowlist = tuple(token.strip() for token in args.family_allowlist.split(",") if token.strip())
    example_id_allowlist = tuple(token.strip() for token in args.example_id_allowlist.split(",") if token.strip())
    result = run_synthetic_market_behavior(
        SyntheticMarketBehaviorConfig(
            phase_name=args.phase_name,
            output_dir=args.output_dir,
            model_id=args.model_id,
            context_variant=args.context_variant,
            order_mode=args.order_mode,
            selection_strategy=args.selection_strategy,
            limit=args.limit if args.limit > 0 else None,
            family_allowlist=family_allowlist,
            example_id_allowlist=example_id_allowlist,
            pair_metric=args.pair_metric,
            pair_mode=args.pair_mode,
            min_pair_gap=args.min_pair_gap,
            generate_source_behavior=bool(args.generate_source_behavior),
            batch_size=max(1, int(args.batch_size)),
            patch_mode=args.patch_mode,
            target_layers=target_layers or (4,),
            components_per_layer=args.components_per_layer,
            component_indices_by_layer=component_indices_by_layer,
            direction_name=args.direction_name,
            strength=args.strength,
            secondary_patch_mode=args.secondary_patch_mode,
            secondary_target_layers=secondary_target_layers,
            secondary_components_per_layer=args.secondary_components_per_layer,
            secondary_component_indices_by_layer=secondary_component_indices_by_layer,
            secondary_direction_name=args.secondary_direction_name,
            secondary_strength=args.secondary_strength,
            random_seed=args.random_seed,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            tool_schema_mode=args.tool_schema_mode,
            tool_choice=args.tool_choice,
            add_generation_prompt=bool(args.add_generation_prompt),
            donor_means_path=args.donor_means_path,
            enforce_eager=not bool(args.no_enforce_eager),
            enable_chunked_prefill=bool(args.enable_chunked_prefill),
            enable_logging_iteration_details=bool(args.enable_logging_iteration_details),
            enable_mfu_metrics=bool(args.enable_mfu_metrics),
            basis_state_key=args.basis_state_key,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
