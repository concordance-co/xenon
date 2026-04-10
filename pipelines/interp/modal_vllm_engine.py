"""Shared vLLM capture engine.

This module owns the actual vLLM-side activation extraction logic: request
execution, residual capture, router capture, and artifact writing. It is used
by the canonical Modal orchestrator and by a few specialized research paths
that still share the same engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq  # used for metadata.parquet I/O
from pipelines.interp.tool_schemas import build_structured_tool_call_schema


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class VLLMCaptureConfig:
    """Configuration for the vLLM-based capture pipeline."""

    output_dir: Path = field(default_factory=lambda: Path("data/activations"))
    model_id: str = "Qwen/Qwen3-30B-A3B"
    limit: int | None = None
    layers: list[int] | None = None
    skip_existing: bool = False
    add_generation_prompt: bool = False
    capture_router: bool = True
    capture_residual: bool = True
    pool_on_capture: str | None = None  # None = full sequence, "last_token"|"prompt_eos"|"mean_pool"

    # vLLM engine knobs
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = True
    max_model_len: int | None = None
    max_num_batched_tokens: int | None = None
    max_num_seqs: int = 1  # >1 enables batched prefill (router capture requires 1)
    max_tokens_buffer: int = 8192  # pre-allocated router buffer size
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = True
    async_scheduling: bool | None = None
    worker_cls: str = ""
    request_scoped_patching: bool = False
    enable_logging_iteration_details: bool = False
    enable_mfu_metrics: bool = False
    reasoning_parser: str = ""

    # Router capture knobs
    router_top_k: int = 8  # top-k indices to save alongside full logits
    router_dtype: str = "float16"  # storage dtype for router logits
    capture_generation: bool = False
    capture_reasoning: bool = True
    reasoning_parser: str = ""
    generation_max_tokens: int = 256
    generation_temperature: float = 0.0
    generation_top_p: float = 1.0

    # Metadata
    metadata_flush_interval: int = 10  # flush metadata every N examples


# ---------------------------------------------------------------------------
# Helpers reused from the local fallback capture module
# ---------------------------------------------------------------------------

# Re-export helpers from the local fallback module so the vLLM engine can emit
# the same artifact format without duplicating save/parsing logic.
from pipelines.interp.local_capture import (  # noqa: E402
    _apply_pooling,
    _artifact_basename_for_row,
    _build_existing_capture_index,
    _can_reuse_capture,
    _load_examples_from_neon,
    _parse_messages,
    _row_identity_token,
    _save_activations,
    _save_router,
)


# ---------------------------------------------------------------------------
# vLLM model access
# ---------------------------------------------------------------------------

def _apply_to_model(llm: Any, func: Any) -> Any:
    """Run a function on the model inside each vLLM worker.

    Uses ``LLM.apply_model()`` which is the official API for accessing
    the underlying ``nn.Module`` across all vLLM versions (v0.17+).

    Args:
        llm: The vLLM ``LLM`` instance.
        func: A callable that takes an ``nn.Module`` and returns a result.

    Returns:
        The result from the first (driver) worker.
    """
    results = llm.apply_model(func)
    return results[0]


def _setup_router_capture(model: Any, *, max_tokens: int = 8192) -> bool:
    """Top-level function for apply_model — must be picklable (no closures)."""
    from pipelines.interp.modal_qwen_moe_hooks import (
        enable_router_capture,
        find_moe_blocks,
        init_router_capture,
    )

    blocks = find_moe_blocks(model)
    if not blocks:
        return False
    init_router_capture(model, max_tokens=max_tokens)
    enable_router_capture(model)
    return True


def _init_router_capture_on_model(llm: Any, max_tokens: int = 8192) -> bool:
    """Initialise router capture on the model inside vLLM workers.

    Returns True if MoE blocks were found and patched.
    """
    from functools import partial

    func = partial(_setup_router_capture, max_tokens=max_tokens)
    return _apply_to_model(llm, func)


def _collect_router_logits_from_model(llm: Any) -> dict[int, Any]:
    """Collect captured router logits from the model inside vLLM workers."""
    from pipelines.interp.modal_qwen_moe_hooks import collect_router_logits

    return _apply_to_model(llm, collect_router_logits)


def _setup_activation_patching(model: Any) -> bool:
    """Top-level function for apply_model — initialise residual patch wrappers."""
    from pipelines.interp.patching.activation_patch_core import init_activation_patching

    init_activation_patching(model)
    return True


def _register_activation_patch_basis(model: Any, basis_payload: dict[int, dict[str, Any]]) -> dict[str, Any]:
    from pipelines.interp.patching.activation_patch_core import register_patch_basis

    return register_patch_basis(model, basis_payload)


def _set_activation_patch_spec(model: Any, patch_spec: dict[str, Any]) -> dict[str, Any]:
    from pipelines.interp.patching.activation_patch_core import set_patch_spec

    return set_patch_spec(model, patch_spec)


def _clear_activation_patch_spec(model: Any) -> None:
    from pipelines.interp.patching.activation_patch_core import clear_patch_spec

    clear_patch_spec(model)


def _collect_activation_patch_stats(
    model: Any,
    req_id: str | None = None,
) -> dict[int, dict[str, Any]]:
    from pipelines.interp.patching.activation_patch_core import collect_patch_stats

    return collect_patch_stats(model, req_id=req_id)


def _init_activation_patching_on_model(llm: Any) -> bool:
    """Initialise activation-patching wrappers on the model inside vLLM workers."""
    return _apply_to_model(llm, _setup_activation_patching)


def _register_activation_patch_basis_on_model(
    llm: Any,
    basis_payload: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Register activation-patch basis tensors on the model inside vLLM workers."""
    from functools import partial

    func = partial(_register_activation_patch_basis, basis_payload=basis_payload)
    return _apply_to_model(llm, func)


def _set_activation_patch_spec_on_model(llm: Any, patch_spec: dict[str, Any]) -> dict[str, Any]:
    """Activate a patch spec for the next request on the worker model."""
    from functools import partial

    func = partial(_set_activation_patch_spec, patch_spec=patch_spec)
    return _apply_to_model(llm, func)


def _clear_activation_patch_spec_on_model(llm: Any) -> None:
    """Disable any active patch spec on the worker model."""
    _apply_to_model(llm, _clear_activation_patch_spec)


def _collect_activation_patch_stats_from_model(
    llm: Any,
    req_id: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Collect patch statistics from the worker-side model."""
    from functools import partial

    func = partial(_collect_activation_patch_stats, req_id=req_id)
    return _apply_to_model(llm, func)


def _reset_router_buffers_on_model(llm: Any) -> None:
    """Reset router capture buffers on the model inside vLLM workers."""
    from pipelines.interp.modal_qwen_moe_hooks import reset_router_buffers

    _apply_to_model(llm, reset_router_buffers)


def _resolve_num_layers(model_id: str) -> int:
    """Determine the number of hidden layers from the model config.

    Loads only the HF config (not the full model) to read
    ``num_hidden_layers``.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return cfg.num_hidden_layers


def _destroy_llm(llm: Any | None) -> None:
    """Best-effort shutdown for a vLLM engine held by a long-lived worker."""
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
    """Best-effort GPU memory cleanup after destroying a vLLM engine."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _create_llm(config: VLLMCaptureConfig) -> Any:
    """Create a vLLM ``LLM`` instance configured for activation capture.

    The engine is configured with:
    - ``extract_hidden_states`` speculative decode method for residual stream
      capture via the stock ``ExampleHiddenStatesConnector``.
    - ``enforce_eager=True`` to avoid CUDA graph issues with patched forwards.
    - ``max_num_seqs=1`` so router buffers contain exactly one request's data.
    """
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": config.model_id,
        "enforce_eager": bool(config.enforce_eager),
        "max_num_seqs": config.max_num_seqs,
        "enable_chunked_prefill": bool(config.enable_chunked_prefill),
        "enable_prefix_caching": config.enable_prefix_caching,
        "tensor_parallel_size": config.tensor_parallel_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
    }
    reasoning_parser = (config.reasoning_parser or "").strip()
    if config.capture_reasoning and not reasoning_parser and "qwen3" in config.model_id.lower():
        reasoning_parser = "qwen3"
    if config.capture_reasoning and reasoning_parser:
        kwargs["structured_outputs_config"] = {
            "reasoning_parser": reasoning_parser,
        }
    if config.worker_cls:
        kwargs["worker_cls"] = config.worker_cls
        if (
            not config.enforce_eager
            and (
                "ActivationPatchGPUWorker" in str(config.worker_cls)
                or "vllm_request_patch_worker" in str(config.worker_cls)
            )
        ):
            kwargs["compilation_config"] = {
                "custom_ops": ["none", "+activation_patch_hidden_states"],
            }
    if config.async_scheduling is not None:
        kwargs["async_scheduling"] = bool(config.async_scheduling)
    if config.enable_logging_iteration_details:
        kwargs["enable_logging_iteration_details"] = True
    if config.enable_mfu_metrics:
        kwargs["enable_mfu_metrics"] = True

    if config.max_model_len is not None:
        kwargs["max_model_len"] = config.max_model_len
    if config.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = int(config.max_num_batched_tokens)

    # Set up extract_hidden_states for residual capture
    if config.capture_residual:
        storage_path = str(config.output_dir / "residual_stream")
        Path(storage_path).mkdir(parents=True, exist_ok=True)

        # Determine which layers to capture
        if config.layers is not None:
            layer_ids = config.layers
        else:
            num_layers = _resolve_num_layers(config.model_id)
            layer_ids = list(range(num_layers))

        kwargs["speculative_config"] = {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": layer_ids,
                }
            },
        }
        kwargs["kv_transfer_config"] = {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": storage_path,
            },
        }
        print(f"Creating vLLM engine: {config.model_id}")
        print(
            f"  enforce_eager={config.enforce_eager}, "
            f"max_num_seqs={config.max_num_seqs}, "
            f"enable_chunked_prefill={config.enable_chunked_prefill}, "
            f"enable_prefix_caching={config.enable_prefix_caching}"
        )
        print(f"  Residual capture: {len(layer_ids)} layers -> {storage_path}")
    else:
        print(f"Creating vLLM engine: {config.model_id}")
        print(
            f"  enforce_eager={config.enforce_eager}, "
            f"max_num_seqs={config.max_num_seqs}, "
            f"enable_chunked_prefill={config.enable_chunked_prefill}, "
            f"enable_prefix_caching={config.enable_prefix_caching}"
        )
        print(f"  Residual capture: disabled")

    llm = LLM(**kwargs)
    return llm


# ---------------------------------------------------------------------------
# Single-prompt capture
# ---------------------------------------------------------------------------

def _tokenize_chat_messages(
    *,
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    """Tokenize chat messages through the HF chat template and normalize to a list[int]."""
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if chat_template_kwargs:
        kwargs.update(chat_template_kwargs)
    input_ids = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(input_ids, list):
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.squeeze().tolist()
        elif hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.squeeze().tolist()
    return [int(token) for token in input_ids]


def _build_chat_template_kwargs(
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    del tools
    kwargs: dict[str, Any] = {}
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    return kwargs or None


def _apply_structured_tool_constraint(
    *,
    sampling_params: Any,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
) -> None:
    schema = build_structured_tool_call_schema(tools, tool_choice)
    if schema is None:
        return

    from vllm.sampling_params import StructuredOutputsParams

    sampling_params.structured_outputs = StructuredOutputsParams(json=schema)


def _generate_one_vllm(
    *,
    llm: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    config: VLLMCaptureConfig,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    patch_spec: dict[str, Any] | None = None,
    patch_specs: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a single prompt through vLLM generation, optionally under a patch spec."""
    from vllm import SamplingParams

    chat_template_kwargs = {
        **(_build_chat_template_kwargs(tools=tools, tool_choice=tool_choice) or {}),
        **(chat_template_kwargs or {}),
    } or None

    input_ids = _tokenize_chat_messages(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=config.add_generation_prompt,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )

    sampling_params = SamplingParams(
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
    )
    _apply_structured_tool_constraint(
        sampling_params=sampling_params,
        tools=tools,
        tool_choice=tool_choice,
    )
    if patch_spec is not None and patch_specs:
        raise ValueError("Provide either patch_spec or patch_specs, not both")
    has_patch = bool(patch_spec is not None or patch_specs)
    use_request_scoped_patch = bool(config.request_scoped_patching and has_patch)
    if use_request_scoped_patch:
        extra_args = getattr(sampling_params, "extra_args", None)
        if not isinstance(extra_args, dict):
            extra_args = {}
        if patch_specs:
            extra_args["patch_specs"] = [dict(spec) for spec in patch_specs]
        elif patch_spec is not None:
            extra_args["patch_spec"] = dict(patch_spec)
        sampling_params.extra_args = extra_args
    elif patch_specs:
        raise ValueError("Multiple patch specs require request_scoped_patching=True")
    elif patch_spec is not None:
        _set_activation_patch_spec_on_model(llm, patch_spec)
    try:
        outputs = llm.generate(
            prompts=[{"prompt_token_ids": input_ids}],
            sampling_params=sampling_params,
        )
    finally:
        if patch_spec is not None and not use_request_scoped_patch:
            _clear_activation_patch_spec_on_model(llm)

    request_output = outputs[0]
    completion = request_output.outputs[0] if getattr(request_output, "outputs", None) else None
    generated_token_ids = [int(token) for token in getattr(completion, "token_ids", [])] if completion is not None else []
    generated_text = str(getattr(completion, "text", "")) if completion is not None else ""
    finish_reason = getattr(completion, "finish_reason", None) if completion is not None else None
    request_id = getattr(request_output, "request_id", None)
    reasoning_text = ""
    for candidate in (
        getattr(completion, "reasoning_content", None),
        getattr(completion, "reasoning", None),
        getattr(request_output, "reasoning_content", None),
        getattr(request_output, "reasoning", None),
    ):
        if candidate is None:
            continue
        if isinstance(candidate, str) and candidate.strip():
            reasoning_text = candidate
            break
        if isinstance(candidate, dict):
            text = candidate.get("text") or candidate.get("content")
            if isinstance(text, str) and text.strip():
                reasoning_text = text
                break

    return {
        "input_ids": input_ids,
        "generated_token_ids": generated_token_ids,
        "generated_text": generated_text,
        "finish_reason": str(finish_reason) if finish_reason is not None else "",
        "reasoning_text": reasoning_text,
        "request_id": str(request_id) if request_id is not None else "",
        "patch_stats": (
            _collect_activation_patch_stats_from_model(llm, req_id=str(request_id) if request_id is not None else None)
            if has_patch
            else {}
        ),
        "all_patch_stats": (
            _collect_activation_patch_stats_from_model(llm)
            if has_patch
            else {}
        ),
    }


def _generate_batch_vllm(
    *,
    llm: Any,
    tokenizer: Any,
    batch_requests: list[dict[str, Any]],
    config: VLLMCaptureConfig,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = -1,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    chat_template_kwargs = {
        **(_build_chat_template_kwargs(tools=tools, tool_choice=tool_choice) or {}),
        **(chat_template_kwargs or {}),
    } or None

    prompts: list[dict[str, Any]] = []
    sampling_params_list: list[Any] = []
    input_ids_list: list[list[int]] = []

    for item in batch_requests:
        input_ids = _tokenize_chat_messages(
            tokenizer=tokenizer,
            messages=item["messages"],
            add_generation_prompt=config.add_generation_prompt,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        sampling_params = SamplingParams(
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
        )
        _apply_structured_tool_constraint(
            sampling_params=sampling_params,
            tools=tools,
            tool_choice=tool_choice,
        )
        patch_spec = item.get("patch_spec")
        patch_specs = item.get("patch_specs")
        if patch_spec is not None and patch_specs:
            raise ValueError("Provide either patch_spec or patch_specs per request, not both")
        if patch_spec is not None or patch_specs:
            if not config.request_scoped_patching:
                raise ValueError(
                    "Batched generation requires request_scoped_patching=True when patch specs are present"
                )
            extra_args = getattr(sampling_params, "extra_args", None)
            if not isinstance(extra_args, dict):
                extra_args = {}
            if patch_specs:
                extra_args["patch_specs"] = [dict(spec) for spec in patch_specs]
            else:
                extra_args["patch_spec"] = dict(patch_spec)
            sampling_params.extra_args = extra_args
        prompts.append({"prompt_token_ids": input_ids})
        sampling_params_list.append(sampling_params)
        input_ids_list.append(input_ids)

    # Match the older stable path: apply the HF chat template locally, then
    # send pretokenized prompts through llm.generate even when tools are used.
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params_list)

    results: list[dict[str, Any]] = []
    for input_ids, item, request_output in zip(input_ids_list, batch_requests, outputs, strict=False):
        completion = request_output.outputs[0] if getattr(request_output, "outputs", None) else None
        generated_token_ids = [int(token) for token in getattr(completion, "token_ids", [])] if completion is not None else []
        generated_text = str(getattr(completion, "text", "")) if completion is not None else ""
        finish_reason = getattr(completion, "finish_reason", None) if completion is not None else None
        request_id = getattr(request_output, "request_id", None)
        patch_spec = item.get("patch_spec")
        patch_specs = item.get("patch_specs")
        has_patch = bool(patch_spec is not None or patch_specs)
        results.append(
            {
                "input_ids": input_ids,
                "generated_token_ids": generated_token_ids,
                "generated_text": generated_text,
                "finish_reason": str(finish_reason) if finish_reason is not None else "",
                "request_id": str(request_id) if request_id is not None else "",
                "patch_stats": (
                    _collect_activation_patch_stats_from_model(
                        llm,
                        req_id=str(request_id) if request_id is not None else None,
                    )
                    if has_patch
                    else {}
                ),
                "all_patch_stats": (
                    _collect_activation_patch_stats_from_model(llm)
                    if has_patch
                    else {}
                ),
            }
        )
    return results


def _capture_one_vllm(
    *,
    llm: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    config: VLLMCaptureConfig,
    log_id: str | int,
    patch_spec: dict[str, Any] | None = None,
    skip_residual_save: bool = False,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Run a single prompt through vLLM and capture activations.

    Returns ``(residual, router_logits, router_indices, input_ids_list, generation_result)``
    where:
    - ``residual``: ``(num_layers, seq_len, hidden_dim)`` float16 tensor or None
    - ``router_logits``: ``(num_moe_layers, seq_len, num_experts)`` float16 tensor or None
    - ``router_indices``: ``(num_moe_layers, seq_len, top_k)`` int16 tensor or None
    - ``input_ids_list``: list[int] of token IDs
    """
    import torch
    from vllm import SamplingParams

    input_ids = _tokenize_chat_messages(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=config.add_generation_prompt,
        tools=None,
    )

    # Reset router buffers before each request
    if config.capture_router:
        _reset_router_buffers_on_model(llm)

    request_output = None
    captured_router_data = None
    generation_result = {
        "generated_token_ids": [],
        "generated_text": "",
        "finish_reason": "",
        "reasoning_text": "",
        "request_id": "",
    }

    should_run_capture_pass = bool(config.capture_residual or config.capture_router or not config.capture_generation)
    if should_run_capture_pass:
        # Keep prompt-side activation capture on the prefill-only path. The
        # stock ExampleHiddenStatesConnector is not stable when we also ask the
        # same request to decode multiple generated tokens.
        sampling_params = SamplingParams(max_tokens=1)
        if patch_spec is not None:
            _set_activation_patch_spec_on_model(llm, patch_spec)
        try:
            outputs = llm.generate(
                prompts=[{"prompt_token_ids": input_ids}],
                sampling_params=sampling_params,
            )
        finally:
            if patch_spec is not None:
                _clear_activation_patch_spec_on_model(llm)

        request_output = outputs[0]
        completion = request_output.outputs[0] if getattr(request_output, "outputs", None) else None
        generation_result = {
            "generated_token_ids": [
                int(token) for token in getattr(completion, "token_ids", [])
            ] if completion is not None else [],
            "generated_text": str(getattr(completion, "text", "")) if completion is not None else "",
            "finish_reason": (
                str(getattr(completion, "finish_reason", ""))
                if completion is not None and getattr(completion, "finish_reason", None) is not None
                else ""
            ),
            "reasoning_text": "",
            "request_id": str(getattr(request_output, "request_id", "") or ""),
        }

        if config.capture_router and config.capture_generation:
            captured_router_data = _collect_router_logits_from_model(llm)

    if config.capture_generation:
        generation_result = _generate_one_vllm(
            llm=llm,
            tokenizer=tokenizer,
            messages=messages,
            config=config,
            max_tokens=int(config.generation_max_tokens),
            temperature=float(config.generation_temperature),
            top_p=float(config.generation_top_p),
            top_k=-1,
            patch_spec=patch_spec,
        )
        if not config.capture_reasoning:
            generation_result["reasoning_text"] = ""

    seq_len = len(input_ids)

    # --- Collect residual stream ---
    residual = None
    if config.capture_residual:
        residual_dir = config.output_dir / "residual_stream"
        output = request_output

        # The ExampleHiddenStatesConnector returns the path via
        # kv_transfer_params["hidden_states_path"] on request completion.
        hidden_states_path = None
        if hasattr(output, "kv_transfer_params") and output.kv_transfer_params:
            hidden_states_path = output.kv_transfer_params.get("hidden_states_path")

        if hidden_states_path and Path(hidden_states_path).exists():
            connector_file = Path(hidden_states_path)
            target_file = residual_dir / f"{log_id}.safetensors"

            from safetensors.torch import load_file

            tensors = load_file(str(connector_file))
            # The connector saves with key "hidden_states" and "token_ids".
            # ExampleHiddenStatesConnector outputs (seq_len, num_layers, hidden_dim)
            # but our pipeline expects (num_layers, seq_len, hidden_dim).
            hs = tensors["hidden_states"] if "hidden_states" in tensors else next(iter(tensors.values()))
            if hs.dim() == 3:
                hs = hs.permute(1, 0, 2)  # (seq, layers, dim) -> (layers, seq, dim)
            hs = hs[:, :seq_len, :]
            residual = hs.to(torch.float16)

            if skip_residual_save:
                # Just load into memory, delete connector file, don't re-save
                connector_file.unlink(missing_ok=True)
            else:
                # Re-save in our canonical format with the "residual_stream" key
                # and rename from vLLM's req_id to our log_id.
                _save_activations(residual, target_file)
                if connector_file != target_file:
                    connector_file.unlink(missing_ok=True)
        else:
            print(
                f"  WARNING: Residual file not found for request. "
                f"kv_transfer_params={getattr(output, 'kv_transfer_params', None)}"
            )

    # --- Collect router logits ---
    router_logits_tensor = None
    router_indices_tensor = None
    if config.capture_router:
        router_data = captured_router_data if captured_router_data is not None else _collect_router_logits_from_model(llm)
        if router_data:
            sorted_layers = sorted(router_data.keys())
            # Stack: {layer_idx: (seq_len, num_experts)} -> (num_layers, seq_len, num_experts)
            stacked = torch.stack(
                [router_data[i] for i in sorted_layers], dim=0
            )
            stacked = stacked[:, :seq_len, :]

            # Compute top-k indices
            _, topk_indices = torch.topk(stacked, k=config.router_top_k, dim=-1)

            # Cast to storage dtypes
            target_dtype = (
                torch.float16
                if config.router_dtype == "float16"
                else torch.float32
            )
            router_logits_tensor = stacked.to(target_dtype)
            router_indices_tensor = topk_indices.to(torch.int16)

    return residual, router_logits_tensor, router_indices_tensor, input_ids, generation_result


# ---------------------------------------------------------------------------
# Batched capture (residual only — router requires max_num_seqs=1)
# ---------------------------------------------------------------------------

def _capture_batch_vllm(
    *,
    llm: Any,
    tokenizer: Any,
    prompts: list[dict[str, Any]],
    config: VLLMCaptureConfig,
) -> list[tuple[Any, list[int], str]]:
    """Run multiple prompts through vLLM in a single generate() call.

    Each prompt dict must have "messages" (list of role/content dicts) and "id".

    Returns list of (residual_tensor, input_ids, prompt_id) in the same order
    as the input prompts.  Router capture is NOT supported in batch mode.
    """
    import torch
    from safetensors.torch import load_file
    from vllm import SamplingParams

    # Pre-tokenize all prompts
    all_inputs: list[dict[str, Any]] = []
    id_order: list[str] = []
    input_ids_map: dict[str, list[int]] = {}

    for p in prompts:
        input_ids = tokenizer.apply_chat_template(
            p["messages"],
            tokenize=True,
            add_generation_prompt=config.add_generation_prompt,
        )
        if not isinstance(input_ids, list):
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.squeeze().tolist()
        pid = str(p["id"])
        all_inputs.append({"prompt_token_ids": input_ids})
        id_order.append(pid)
        input_ids_map[pid] = input_ids

    # Single batched generate — vLLM schedules concurrently up to max_num_seqs
    sampling_params = SamplingParams(max_tokens=1)
    outputs = llm.generate(all_inputs, sampling_params)

    # Map outputs back by request_id → our prompt order
    # vLLM returns outputs in the same order as inputs
    results: list[tuple[Any, list[int], str]] = []
    storage_path = str(config.output_dir / "residual_stream")

    for i, output in enumerate(outputs):
        pid = id_order[i]
        residual = None

        if config.capture_residual:
            hidden_states_path = None
            if hasattr(output, "kv_transfer_params") and output.kv_transfer_params:
                hidden_states_path = output.kv_transfer_params.get("hidden_states_path")

            if hidden_states_path and Path(hidden_states_path).exists():
                connector_file = Path(hidden_states_path)
                tensors = load_file(str(connector_file))
                hs = tensors["hidden_states"] if "hidden_states" in tensors else next(iter(tensors.values()))
                if hs.dim() == 3:
                    hs = hs.permute(1, 0, 2)  # (seq, layers, dim) -> (layers, seq, dim)
                residual = hs.to(torch.float16)
                connector_file.unlink(missing_ok=True)
            else:
                print(
                    f"  WARNING: Residual file not found for {pid}. "
                    f"kv_transfer_params={getattr(output, 'kv_transfer_params', None)}"
                )

        results.append((residual, input_ids_map[pid], pid))

    return results


# ---------------------------------------------------------------------------
# Metadata I/O
# ---------------------------------------------------------------------------

def _load_existing_metadata(meta_path: Path) -> list[dict[str, Any]]:
    """Load existing metadata rows from parquet, or return empty list."""
    if meta_path.exists():
        table = pq.read_table(meta_path)
        rows = table.to_pylist()
        print(f"Loaded {len(rows)} existing metadata rows from {meta_path}")
        return rows
    return []


def _flush_metadata(
    meta_path: Path, metadata_rows: list[dict[str, Any]]
) -> None:
    """Write metadata rows to parquet (overwrites)."""
    if not metadata_rows:
        return
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(metadata_rows)
    pq.write_table(table, meta_path, compression="snappy")


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------

def run_vllm_capture(config: VLLMCaptureConfig) -> dict[str, Any]:
    """Run the vLLM capture pipeline.

    Processes each example one at a time (required by ``max_num_seqs=1``
    constraint for router capture correctness).  Produces the same output
    directory structure and metadata format as :func:`capture.run_capture`.

    Returns:
        Dict with ``processed``, ``skipped``, ``errors`` counts.
    """
    import torch
    from transformers import AutoTokenizer

    examples = _load_examples_from_neon(limit=config.limit)
    if not examples:
        print("No examples to process.")
        return {"processed": 0, "skipped": 0, "errors": 0}

    # --- Create vLLM engine ---
    llm = _create_llm(config)

    # --- Load tokenizer separately for chat template application ---
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    # --- Set up router capture via apply_model ---
    is_moe = False
    if config.capture_router:
        is_moe = _init_router_capture_on_model(
            llm, max_tokens=config.max_tokens_buffer
        )
        if is_moe:
            print("Router capture enabled on MoE blocks")
        else:
            print(
                "WARNING: capture_router=True but model has no MoE layers; "
                "skipping router capture"
            )

    # Get layer count for metadata
    num_model_layers = _resolve_num_layers(config.model_id)

    # --- Prepare output directories ---
    residual_dir = config.output_dir / "residual_stream"
    router_dir = config.output_dir / "router_logits"
    if config.capture_residual:
        residual_dir.mkdir(parents=True, exist_ok=True)
    if config.capture_router and is_moe:
        router_dir.mkdir(parents=True, exist_ok=True)

    # --- Load or initialise metadata ---
    meta_path = config.output_dir / "metadata.parquet"
    metadata_rows = _load_existing_metadata(meta_path)
    existing_capture_index = _build_existing_capture_index(metadata_rows)

    processed = 0
    skipped = 0
    errors = 0
    flush_interval = config.metadata_flush_interval

    for idx, row in enumerate(examples):
        log_id = row.get("log_id")
        if log_id is None:
            print(f"  [{idx + 1}/{len(examples)}] Skipping row with no log_id")
            skipped += 1
            continue
        artifact_id = _artifact_basename_for_row(row)

        # Skip if already captured
        if config.skip_existing:
            current_token = _row_identity_token(row)
            existing_row = existing_capture_index.get(current_token) if current_token is not None else None
            residual_exists = (
                not config.capture_residual
                or (residual_dir / f"{artifact_id}.safetensors").exists()
            )
            router_exists = (
                not (config.capture_router and is_moe)
                or (router_dir / f"{artifact_id}.safetensors").exists()
            )
            if _can_reuse_capture(row, existing_row) and residual_exists and router_exists:
                print(
                    f"  [{idx + 1}/{len(examples)}] Skipping existing: {log_id}"
                )
                skipped += 1
                continue

        messages = _parse_messages(row)
        if not messages:
            print(f"  [{idx + 1}/{len(examples)}] Skipping {log_id}: no messages")
            skipped += 1
            continue

        try:
            t0 = time.monotonic()
            residual, router_logits, router_indices, input_ids, _generation_result = (
                _capture_one_vllm(
                    llm=llm,
                    tokenizer=tokenizer,
                    messages=messages,
                    config=config,
                    log_id=artifact_id,
                )
            )
            elapsed = time.monotonic() - t0

            seq_len = len(input_ids)

            # Pool before saving if requested
            if config.pool_on_capture:
                residual, router_logits, router_indices = _apply_pooling(
                    residual,
                    router_logits,
                    router_indices,
                    config.pool_on_capture,
                    input_ids=input_ids,
                    eos_token_id=getattr(tokenizer, "eos_token_id", None),
                )

            file_size = 0

            # Residual is already saved by the connector + rename in
            # _capture_one_vllm, but we may need to re-save after pooling.
            if residual is not None and config.pool_on_capture:
                file_size += _save_activations(
                    residual, residual_dir / f"{artifact_id}.safetensors"
                )
            elif residual is not None:
                # Already saved; compute file size
                p = residual_dir / f"{artifact_id}.safetensors"
                if p.exists():
                    file_size += p.stat().st_size

            if router_logits is not None and router_indices is not None:
                file_size += _save_router(
                    router_logits,
                    router_indices,
                    router_dir / f"{artifact_id}.safetensors",
                    router_dtype=config.router_dtype,
                )

            # Build metadata row (same schema as HF capture)
            prompt_hash = hashlib.sha256(
                bytes(json.dumps(input_ids), "utf-8")
            ).hexdigest()

            captured_layers = (
                sorted(config.layers)
                if config.layers is not None
                else list(range(num_model_layers))
            )

            meta_row: dict[str, Any] = {
                "log_id": int(log_id),
                "row_key": row.get("row_key"),
                "artifact_id": artifact_id,
                "source_prompt_hash": row.get("source_prompt_hash"),
                "source_relation": row.get("source_relation"),
                "workflow_spec_id": row.get("workflow_spec_id"),
                "workflow_spec_version": row.get("workflow_spec_version"),
                "seq_len": seq_len,
                "prompt_hash": prompt_hash,
                "capture_timestamp": datetime.now(UTC).isoformat(),
                "file_size_bytes": file_size,
                "elapsed_s": round(elapsed, 2),
                "has_router": router_logits is not None,
                "captured_layers": json.dumps(captured_layers),
                "pooling": config.pool_on_capture or "none",
            }
            if residual is not None:
                meta_row["num_layers_captured"] = int(residual.shape[0])
                meta_row["hidden_dim"] = int(residual.shape[-1])
            else:
                meta_row["num_layers_captured"] = (
                    int(router_logits.shape[0]) if router_logits is not None else 0
                )
                meta_row["hidden_dim"] = 0
            if router_logits is not None:
                meta_row["num_experts"] = int(router_logits.shape[-1])

            metadata_rows.append(meta_row)
            token = _row_identity_token(meta_row)
            if token is not None:
                existing_capture_index[token] = meta_row
            processed += 1

            shape_parts = []
            if residual is not None:
                shape_parts.append(f"residual={tuple(residual.shape)}")
            if router_logits is not None:
                shape_parts.append(f"router={tuple(router_logits.shape)}")
            print(
                f"  [{idx + 1}/{len(examples)}] {log_id}: "
                f"{', '.join(shape_parts)}, "
                f"{file_size / 1024 / 1024:.1f}MB, "
                f"{elapsed:.1f}s"
            )

            # Periodic metadata flush
            if processed % flush_interval == 0:
                _flush_metadata(meta_path, metadata_rows)

        except Exception as exc:
            import traceback

            errors += 1
            print(f"  [{idx + 1}/{len(examples)}] ERROR {log_id}: {exc}")
            traceback.print_exc()

    # Final metadata flush
    if metadata_rows:
        _flush_metadata(meta_path, metadata_rows)
        print(f"\nWrote metadata: {meta_path} ({len(metadata_rows)} rows)")

    print(f"\nDone: {processed} captured, {skipped} skipped, {errors} errors")
    return {"processed": processed, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture activations (residual stream and/or MoE router logits) "
            "using vLLM for efficient GPU inference"
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/activations"),
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (e.g. 0,12,24,35). Default: all layers.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--add-generation-prompt",
        action="store_true",
        help="Append assistant turn start tokens to the chat template",
    )
    parser.add_argument(
        "--capture-router",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture MoE router logits (default: True, --no-capture-router to disable)",
    )
    parser.add_argument(
        "--capture-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture residual stream (default: True, --no-capture-residual to disable)",
    )
    parser.add_argument(
        "--pool-on-capture",
        choices=["last_token", "prompt_eos", "mean_pool"],
        default=None,
        help=(
            "Pool the sequence dimension during capture to reduce file size. "
            "Stores (layers, dim) instead of (layers, seq_len, dim)."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory for vLLM (default: 0.90)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length. Default: model config value.",
    )
    parser.add_argument(
        "--max-tokens-buffer",
        type=int,
        default=8192,
        help="Pre-allocated router buffer size (default: 8192)",
    )
    parser.add_argument(
        "--router-top-k",
        type=int,
        default=8,
        help="Number of top-k router indices to save (default: 8)",
    )
    parser.add_argument(
        "--router-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Storage dtype for router logits (default: float16)",
    )
    parser.add_argument(
        "--metadata-flush-interval",
        type=int,
        default=10,
        help="Flush metadata every N examples (default: 10)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    layers: list[int] | None = None
    if args.layers is not None:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    cfg = VLLMCaptureConfig(
        output_dir=args.output_dir,
        model_id=args.model_id,
        limit=args.limit,
        layers=layers,
        skip_existing=args.skip_existing,
        add_generation_prompt=args.add_generation_prompt,
        capture_router=args.capture_router,
        capture_residual=args.capture_residual,
        pool_on_capture=args.pool_on_capture,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_tokens_buffer=args.max_tokens_buffer,
        router_top_k=args.router_top_k,
        router_dtype=args.router_dtype,
        metadata_flush_interval=args.metadata_flush_interval,
    )
    run_vllm_capture(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
