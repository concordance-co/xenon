"""vLLM capture implementation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pipelines_v2.data.datasets import Example
from pipelines_v2.engine.base import EngineCaptureResult
from pipelines_v2.engine.prompt_metadata import rebase_section_records, rebase_token_sections
from pipelines_v2.engine.prompt_metadata import resolve_prompt_metadata, section_records_from_metadata, token_sections_from_metadata
from pipelines_v2.engine.routing import requested_topk_from_gate_k, routing_record_payload
from pipelines_v2.operations.specs import CaptureSpec, MoERoutingSite, ResidualSite, RoutingRecord

if TYPE_CHECKING:
    from pipelines_v2.engine.vllm.engine import VLLMEngine


def run_vllm_capture(
    *,
    engine: VLLMEngine,
    spec: CaptureSpec,
    batch_callback: Callable[[list[Example], list[dict[str, Any]], list[dict[str, Any]]], None] | None = None,
) -> EngineCaptureResult:
    """Run a capture spec with vLLM in the current process."""

    from transformers import AutoTokenizer
    from vllm import LLM

    examples = list(spec.dataset.examples)
    residual_sites = [site for site in spec.sites if isinstance(site, ResidualSite)]
    routing_sites = [site for site in spec.sites if isinstance(site, MoERoutingSite)]
    residual_layers = sorted({int(layer) for site in residual_sites for layer in site.layers})
    wants_residual = bool(residual_sites)
    wants_routing = bool(routing_sites)
    wants_generation = bool(spec.generation.enabled)
    capture_generated_tokens = bool(spec.generation.capture_generated_tokens)
    batch_size = max(1, int(engine.max_num_seqs or 1))
    section_names = [
        str(getattr(site.tokens, "value", ""))
        for site in spec.sites
        if getattr(site.tokens, "kind", None) == "section"
    ]
    wants_sections = bool(section_names)
    requires_prompt_metadata_sections = wants_sections and not (
        capture_generated_tokens and all(name in {"prompt", "generated", "full"} for name in section_names)
    )

    if wants_routing and bool(engine.enable_prefix_caching):
        raise ValueError(
            "MoE routing capture currently requires enable_prefix_caching=False in the current vLLM implementation"
        )

    model_path = engine.resolved_model_path()
    llm_kwargs: dict[str, Any] = {
        "model": model_path,
        "enforce_eager": bool(engine.enforce_eager),
        "max_num_seqs": int(engine.max_num_seqs or 1),
        "enable_chunked_prefill": bool(engine.enable_chunked_prefill),
        "enable_prefix_caching": bool(engine.enable_prefix_caching),
        "tensor_parallel_size": int(engine.tensor_parallel_size or 1),
        "pipeline_parallel_size": int(engine.pipeline_parallel_size or 1),
        "gpu_memory_utilization": float(engine.gpu_memory_utilization or 0.90),
    }
    if engine.distributed_executor_backend:
        llm_kwargs["distributed_executor_backend"] = str(engine.distributed_executor_backend)
    if model_path != engine.canonical_model_name():
        llm_kwargs["served_model_name"] = engine.canonical_model_name()
    if engine.max_model_len:
        llm_kwargs["max_model_len"] = int(engine.max_model_len)
    llm_kwargs.update(engine.extra_llm_kwargs())
    reasoning_parser = (engine.reasoning_parser or "").strip()
    if spec.generation.capture_reasoning and not reasoning_parser and "qwen3" in str(engine.model_id).lower():
        reasoning_parser = "qwen3"
    if reasoning_parser:
        llm_kwargs["structured_outputs_config"] = {"reasoning_parser": reasoning_parser}

    with tempfile.TemporaryDirectory(prefix="pipelines_v2_vllm_") as tmpdir:
        connector_dir = Path(tmpdir) / "hidden_states"
        if wants_residual:
            connector_dir.mkdir(parents=True, exist_ok=True)
            llm_kwargs["speculative_config"] = {
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": residual_layers,
                    }
                },
            }
            llm_kwargs["kv_transfer_config"] = {
                "kv_connector": (
                    "PipelinesV2HiddenStatesConnector"
                    if capture_generated_tokens
                    else "ExampleHiddenStatesConnector"
                ),
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": str(connector_dir),
                },
            }
            if capture_generated_tokens:
                llm_kwargs["kv_transfer_config"]["kv_connector_module_path"] = (
                    "pipelines_v2.engine.vllm.hidden_states_connector"
                )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = LLM(**llm_kwargs)
        reasoning_parser = _build_reasoning_parser(
            tokenizer=tokenizer,
            parser_name=reasoning_parser,
            enable_thinking=engine.enable_thinking,
        )
        return _run_vllm_capture_loaded(
            engine=engine,
            spec=spec,
            llm=llm,
            tokenizer=tokenizer,
            reasoning_parser=reasoning_parser,
            batch_callback=batch_callback,
        )


def run_vllm_capture_with_runtime(
    *,
    runtime: Any,
    spec: CaptureSpec,
    batch_callback: Callable[[list[Example], list[dict[str, Any]], list[dict[str, Any]]], None] | None = None,
) -> EngineCaptureResult:
    """Run capture against an already-loaded reusable vLLM runtime."""

    return _run_vllm_capture_loaded(
        engine=runtime.engine,
        spec=spec,
        llm=runtime.llm,
        tokenizer=runtime.tokenizer,
        reasoning_parser=runtime.reasoning_parser_instance,
        batch_callback=batch_callback,
    )


def _run_vllm_capture_loaded(
    *,
    engine: VLLMEngine,
    spec: CaptureSpec,
    llm: Any,
    tokenizer: Any,
    reasoning_parser: Any | None,
    batch_callback: Callable[[list[Example], list[dict[str, Any]], list[dict[str, Any]]], None] | None = None,
) -> EngineCaptureResult:
    from functools import partial

    examples = list(spec.dataset.examples)
    residual_sites = [site for site in spec.sites if isinstance(site, ResidualSite)]
    routing_sites = [site for site in spec.sites if isinstance(site, MoERoutingSite)]
    residual_layers = sorted({int(layer) for site in residual_sites for layer in site.layers})
    wants_residual = bool(residual_sites)
    wants_routing = bool(routing_sites)
    wants_generation = bool(spec.generation.enabled)
    capture_generated_tokens = bool(spec.generation.capture_generated_tokens)
    batch_size = max(1, int(engine.max_num_seqs or 1))
    section_names = [
        str(getattr(site.tokens, "value", ""))
        for site in spec.sites
        if getattr(site.tokens, "kind", None) == "section"
    ]
    wants_sections = bool(section_names)
    requires_prompt_metadata_sections = wants_sections and not (
        capture_generated_tokens and all(name in {"prompt", "generated", "full"} for name in section_names)
    )

    if wants_routing and bool(engine.enable_prefix_caching):
        raise ValueError(
            "MoE routing capture currently requires enable_prefix_caching=False in the current vLLM implementation"
        )

    router_enabled = False
    discovered_router_layers: list[int] = []
    if wants_routing:
        buffer_size = int(engine.max_model_len or 32768)
        router_enabled = bool(_apply_to_model(llm, partial(_setup_router_capture_on_model, max_tokens=buffer_size)))
        if not router_enabled:
            raise RuntimeError("MoE routing capture was requested, but no compatible MoE blocks were found")
        discovered_router_layers = sorted(
            int(layer) for layer in (_apply_to_model(llm, _discover_router_layers_on_model) or [])
        )

    feature_payloads: dict[str, dict[str, Any]] = {site.name: _empty_feature(site) for site in spec.sites}
    generations: list[dict[str, Any]] = []
    example_metadata: list[dict[str, Any]] = []

    for batch in _iter_batches(examples, batch_size):
        batch_records = _capture_prompt_batch(
            llm=llm,
            tokenizer=tokenizer,
            reasoning_parser=reasoning_parser,
            examples=batch,
            add_generation_prompt=bool(engine.add_generation_prompt),
            require_sections=requires_prompt_metadata_sections,
            prompt_metadata_builder=spec.prompt_metadata_builder,
            wants_residual=wants_residual,
            wants_routing=wants_routing,
            wants_generation=wants_generation,
            generation_max_tokens=spec.generation.max_tokens,
            generation_temperature=float(spec.generation.temperature or 0.0),
            generation_top_p=float(spec.generation.top_p),
            generation_top_k=int(spec.generation.top_k),
            generation_chat_tools=spec.generation.chat_tools,
            generation_tool_choice=spec.generation.tool_choice,
            generation_structured_output=spec.generation.structured_output,
            capture_reasoning=bool(spec.generation.capture_reasoning),
            capture_generated_tokens=capture_generated_tokens,
            enable_thinking=engine.enable_thinking,
            chat_template_kwargs=dict(engine.extra.get("chat_template_kwargs", {})),
        )
        batch_generations: list[dict[str, Any]] = []
        batch_example_metadata: list[dict[str, Any]] = []
        for record in batch_records:
            example = record["example"]
            prompt_token_ids = record["prompt_token_ids"]
            prompt_token_sections = record["prompt_token_sections"]
            prompt_section_records = record["prompt_section_records"]
            residual_token_count = record["residual_token_count"]
            residual_token_sections = record["residual_token_sections"]
            residual_section_records = record["residual_section_records"]
            residual = record["residual"]
            router_data = record["router_data"]
            actual_router_layers = record["actual_router_layers"]
            generation_result = record["generation_result"]

            _fill_residual_features(
                feature_payloads=feature_payloads,
                residual_sites=residual_sites,
                residual_layers=residual_layers,
                residual=residual,
                example=example,
                token_count=residual_token_count,
                token_sections=residual_token_sections,
                section_records=residual_section_records,
            )
            _fill_router_features(
                feature_payloads=feature_payloads,
                routing_sites=routing_sites,
                router_data=router_data,
                example=example,
                token_count=len(prompt_token_ids),
                token_sections=prompt_token_sections,
                section_records=prompt_section_records,
                discovered_router_layers=discovered_router_layers,
            )

            if generation_result is not None:
                generation_row = {"example_key": example.key, **generation_result}
                generations.append(generation_row)
                batch_generations.append(generation_row)

            metadata_row = {
                "example_key": example.key,
                "prompt_hash": example.prompt_hash,
                "token_count": residual_token_count,
                "prompt_token_count": len(prompt_token_ids),
                "generated_token_count": int(record["generated_token_count"]),
                "captured_generated_token_count": int(record["captured_generated_token_count"]),
                "generated": generation_result is not None,
                "capture_generated_tokens": capture_generated_tokens,
                "capture_mode": "batched_prompt_capture" if len(batch) > 1 else "single_request",
                "actual_router_layers": actual_router_layers,
            }
            example_metadata.append(metadata_row)
            batch_example_metadata.append(metadata_row)

        if batch_callback is not None:
            batch_callback(
                list(batch),
                list(batch_generations),
                list(batch_example_metadata),
            )

    return EngineCaptureResult(
        features=feature_payloads,
        generations=generations,
        metadata={
            "backend": "vllm",
            "example_metadata": example_metadata,
            "router_enabled": router_enabled,
            "requested_router_layers": sorted({int(layer) for site in routing_sites for layer in site.layers}),
            "discovered_router_layers": discovered_router_layers,
            "residual_layers": residual_layers,
            "batch_size": batch_size,
            "spec_hash": spec.spec_hash(),
        },
    )


def _setup_router_capture_on_model(model: Any, *, max_tokens: int) -> bool:
    from pipelines_v2.engine.vllm.moe_hooks import enable_router_capture, init_router_capture

    ok = init_router_capture(model, max_tokens=max_tokens)
    if ok:
        enable_router_capture(model)
    return ok


def _reset_router_buffers_on_model(model: Any) -> None:
    from pipelines_v2.engine.vllm.moe_hooks import reset_router_buffers

    reset_router_buffers(model)


def _collect_router_capture_from_model(model: Any) -> dict[int, Any]:
    from pipelines_v2.engine.vllm.moe_hooks import collect_router_capture

    return collect_router_capture(model)


def _discover_router_layers_on_model(model: Any) -> list[int]:
    from pipelines_v2.engine.vllm.moe_hooks import find_moe_blocks

    return sorted(int(layer) for layer in find_moe_blocks(model).keys())


def _apply_to_model(llm: Any, fn: Any) -> Any:
    results = llm.apply_model(fn)
    return results[0]


def _empty_feature(site: ResidualSite | MoERoutingSite) -> dict[str, Any]:
    if isinstance(site, ResidualSite):
        return {
            "kind": "residual",
            "site": site.site,
            "storage": {"dtype": site.storage.dtype, "format": site.storage.format},
            "layers": {str(layer): {} for layer in site.layers},
        }
    return {
        "kind": "moe_routing",
        "routing_policy": {
            "source": "vllm_gate_logits",
            "observed_routing_decisions": True,
        },
        "layers": {str(layer): {} for layer in site.layers},
    }


def _prompt_token_ids(
    *,
    tokenizer: Any,
    example: Example,
    add_generation_prompt: bool,
    require_sections: bool,
    prompt_metadata_builder: Any | None,
    tools: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    enable_thinking: bool | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    token_ids = _tokenize_prompt(
        tokenizer=tokenizer,
        prompt=example.prompt,
        add_generation_prompt=add_generation_prompt,
        tools=tools,
        tool_choice=tool_choice,
        enable_thinking=enable_thinking,
        chat_template_kwargs=chat_template_kwargs,
    )
    token_sections: dict[str, list[int]] = {}
    section_records: list[dict[str, Any]] = []
    needs_rendered_metadata = require_sections or prompt_metadata_builder is not None or bool(example.metadata)
    if needs_rendered_metadata:
        rendered = _render_prompt(
            tokenizer=tokenizer,
            prompt=example.prompt,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            tool_choice=tool_choice,
            enable_thinking=enable_thinking,
            chat_template_kwargs=chat_template_kwargs,
        )
        metadata = resolve_prompt_metadata(
            metadata=example.metadata,
            rendered_prompt=rendered,
            builder=prompt_metadata_builder,
            prompt=example.prompt,
        )
        encoding = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        rendered_token_ids = _normalize_token_ids(encoding)
        offsets = _normalize_offsets(encoding)
        token_sections = token_sections_from_metadata(
            metadata=metadata,
            offsets=offsets,
            require_sections=require_sections,
            allow_char_spans=True,
        )
        section_records = section_records_from_metadata(
            metadata=metadata,
            offsets=offsets,
            token_sections=token_sections,
            allow_char_spans=True,
        )
        if (token_sections or section_records) and token_ids != rendered_token_ids:
            index_map = _align_source_positions_to_target(
                source_token_ids=rendered_token_ids,
                target_token_ids=token_ids,
            )
            token_sections = _remap_token_sections_with_index_map(
                token_sections=token_sections,
                index_map=index_map,
            )
            section_records = _remap_section_records(
                section_records=section_records,
                index_map=index_map,
            )
    return {
        "token_ids": token_ids,
        "token_sections": token_sections,
        "section_records": section_records,
    }


def _tokenize_prompt(
    *,
    tokenizer: Any,
    prompt: Any,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    enable_thinking: bool | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    if isinstance(prompt, list):
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools:
            kwargs["tools"] = [dict(tool) for tool in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        if chat_template_kwargs:
            kwargs.update(chat_template_kwargs)
        token_ids = tokenizer.apply_chat_template(prompt, **kwargs)
        return _normalize_token_ids(token_ids)
    if isinstance(prompt, str):
        encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        return _normalize_token_ids(encoding)
    raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")
def _remap_token_sections(
    *,
    token_sections: dict[str, list[int]],
    source_token_ids: list[int],
    target_token_ids: list[int],
) -> dict[str, list[int]]:
    index_map = _align_source_positions_to_target(
        source_token_ids=source_token_ids,
        target_token_ids=target_token_ids,
    )
    return _remap_token_sections_with_index_map(
        token_sections=token_sections,
        index_map=index_map,
    )


def _remap_token_sections_with_index_map(
    *,
    token_sections: dict[str, list[int]],
    index_map: dict[int, int],
) -> dict[str, list[int]]:
    remapped: dict[str, list[int]] = {}
    for name, positions in token_sections.items():
        mapped = [index_map[int(position)] for position in positions if int(position) in index_map]
        if mapped:
            remapped[str(name)] = mapped
    return remapped


def _remap_section_records(
    *,
    section_records: list[dict[str, Any]],
    index_map: dict[int, int],
) -> list[dict[str, Any]]:
    remapped: list[dict[str, Any]] = []
    for raw_record in section_records:
        positions = raw_record.get("token_positions")
        if not isinstance(positions, list):
            continue
        mapped = [index_map[int(position)] for position in positions if int(position) in index_map]
        if not mapped:
            continue
        record = dict(raw_record)
        record["token_positions"] = mapped
        remapped.append(record)
    return remapped


def _align_source_positions_to_target(
    *,
    source_token_ids: list[int],
    target_token_ids: list[int],
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    target_index = 0
    for source_index, token_id in enumerate(source_token_ids):
        while target_index < len(target_token_ids) and int(target_token_ids[target_index]) != int(token_id):
            target_index += 1
        if target_index >= len(target_token_ids):
            raise RuntimeError(
                "Rendered prompt tokenization could not be aligned to chat-template tokenization. "
                "This usually means the chat template emitted non-textual control tokens that need explicit handling."
            )
        mapping[int(source_index)] = int(target_index)
        target_index += 1
    return mapping


def _render_prompt(
    *,
    tokenizer: Any,
    prompt: Any,
    add_generation_prompt: bool,
    tools: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    enable_thinking: bool | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    if isinstance(prompt, list):
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools:
            template_kwargs["tools"] = [dict(tool) for tool in tools]
        if tool_choice is not None:
            template_kwargs["tool_choice"] = tool_choice
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = enable_thinking
        if chat_template_kwargs:
            template_kwargs.update(chat_template_kwargs)
        rendered = tokenizer.apply_chat_template(prompt, **template_kwargs)
        return str(rendered)
    if isinstance(prompt, str):
        return prompt
    raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")


def _normalize_token_ids(encoding: Any) -> list[int]:
    token_ids = getattr(encoding, "input_ids", encoding)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(token) for token in token_ids]


def _normalize_offsets(encoding: Any) -> list[tuple[int, int]]:
    offsets = getattr(encoding, "offset_mapping", None)
    if offsets is None:
        return []
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], list | tuple):
        offsets = offsets[0]
    return [(int(start), int(end)) for start, end in offsets]


def _hidden_states_path(request_output: Any) -> str | None:
    params = getattr(request_output, "kv_transfer_params", None)
    if isinstance(params, dict):
        value = params.get("hidden_states_path")
        return str(value) if value else None
    return None


def _fill_residual_features(
    *,
    feature_payloads: dict[str, dict[str, Any]],
    residual_sites: list[ResidualSite],
    residual_layers: list[int],
    residual: Any,
    example: Example,
    token_count: int,
    token_sections: dict[str, list[int]],
    section_records: list[dict[str, Any]],
) -> None:
    if not residual_sites:
        return
    if residual is None:
        raise RuntimeError("Residual capture requested but no residual tensor was produced")
    layer_index = {layer: idx for idx, layer in enumerate(residual_layers)}
    for site in residual_sites:
        positions = site.tokens.resolve(token_count, token_sections=token_sections)
        feature_token_sections = rebase_token_sections(
            token_sections=token_sections,
            selected_positions=positions,
        )
        feature_section_records = rebase_section_records(
            section_records=section_records,
            selected_positions=positions,
        )
        for layer in site.layers:
            idx = layer_index[int(layer)]
            values = residual[idx, positions, :]
            feature_payloads[site.name]["layers"][str(layer)][example.key] = {
                "tokens": positions,
                "values": values,
                "prompt_hash": example.prompt_hash,
                "token_sections": feature_token_sections,
                "section_records": feature_section_records,
            }


def _capture_prompt_batch(
    *,
    llm: Any,
    tokenizer: Any,
    examples: list[Example],
    add_generation_prompt: bool,
    require_sections: bool,
    prompt_metadata_builder: Any | None,
    wants_residual: bool,
    wants_routing: bool,
    wants_generation: bool,
    generation_max_tokens: int | None,
    generation_temperature: float,
    capture_reasoning: bool = False,
    generation_top_p: float = 1.0,
    generation_top_k: int = -1,
    generation_chat_tools: tuple[dict[str, Any], ...] = (),
    generation_tool_choice: str | dict[str, Any] | None = None,
    generation_structured_output: dict[str, Any] | None = None,
    reasoning_parser: Any | None = None,
    enable_thinking: bool | None = None,
    capture_generated_tokens: bool = False,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    import torch
    from safetensors.torch import load_file
    from vllm import SamplingParams

    prompts: list[dict[str, Any]] = []
    tokenized_by_key: dict[str, dict[str, Any]] = {}
    for example in examples:
        tokenized_prompt = _prompt_token_ids(
            tokenizer=tokenizer,
            example=example,
            add_generation_prompt=add_generation_prompt,
            require_sections=require_sections,
            prompt_metadata_builder=prompt_metadata_builder,
            tools=generation_chat_tools,
            tool_choice=generation_tool_choice,
            enable_thinking=enable_thinking,
            chat_template_kwargs=chat_template_kwargs,
        )
        prompt_token_ids = tokenized_prompt["token_ids"]
        prompts.append({"prompt_token_ids": prompt_token_ids})
        tokenized_by_key[example.key] = tokenized_prompt

    if wants_routing:
        _apply_to_model(llm, _reset_router_buffers_on_model)

    sampling_params = SamplingParams(
        max_tokens=generation_max_tokens if wants_generation else 1,
        temperature=float(generation_temperature if wants_generation else 0.0),
        top_p=float(generation_top_p if wants_generation else 1.0),
        top_k=int(generation_top_k if wants_generation else -1),
    )
    _apply_structured_output_constraint(sampling_params, generation_structured_output if wants_generation else None)
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )
    if len(outputs) != len(examples):
        raise RuntimeError(
            "vLLM returned a different number of request outputs than prompts: "
            f"got {len(outputs)}, expected {len(examples)}"
        )

    router_by_example: dict[str, dict[int, dict[str, Any]]] = {}
    actual_router_layers: list[int] = []
    if wants_routing:
        raw_router = _apply_to_model(llm, _collect_router_capture_from_model)
        actual_router_layers = sorted(int(layer) for layer in raw_router.keys())
        prompt_lengths = [len(tokenized_by_key[example.key]["token_ids"]) for example in examples]
        router_by_example = _split_router_capture_batch(
            raw_router=raw_router,
            examples=examples,
            prompt_lengths=prompt_lengths,
            allow_trailing_rows=wants_generation,
        )

    results: list[dict[str, Any]] = []
    for example, request_output in zip(examples, outputs, strict=False):
        tokenized_prompt = tokenized_by_key[example.key]
        prompt_token_ids = tokenized_prompt["token_ids"]
        generation_result = (
            _generation_result_from_output(
                request_output,
                capture_reasoning=capture_reasoning,
                reasoning_parser=reasoning_parser,
            )
            if wants_generation
            else None
        )
        generated_token_ids = (
            list(generation_result.get("generated_token_ids") or ())
            if isinstance(generation_result, dict)
            else []
        )
        generated_token_count = len(generated_token_ids)
        residual_token_count = len(prompt_token_ids)
        captured_generated_token_count = 0
        residual_token_sections = dict(tokenized_prompt["token_sections"])
        residual_section_records = [dict(record) for record in tokenized_prompt.get("section_records", ())]
        if capture_generated_tokens and not wants_generation:
            raise RuntimeError("capture_generated_tokens=True requires generation.enabled=True")
        residual = None
        if wants_residual:
            hidden_states_path = _hidden_states_path(request_output)
            if not hidden_states_path:
                raise RuntimeError(f"vLLM did not return hidden_states_path for example {example.key!r}")
            connector_file = Path(hidden_states_path)
            if not connector_file.exists():
                raise RuntimeError(
                    f"Hidden-state connector file missing for example {example.key!r}: {hidden_states_path}"
                )
            tensors = load_file(str(connector_file))
            hs = tensors["hidden_states"] if "hidden_states" in tensors else next(iter(tensors.values()))
            if hs.dim() == 3:
                hs = hs.permute(1, 0, 2)
            available_token_count = int(hs.shape[1])
            if available_token_count < len(prompt_token_ids):
                raise RuntimeError(
                    "vLLM returned fewer hidden-state rows than prompt tokens "
                    f"for example {example.key!r}: got {available_token_count}, "
                    f"expected at least {len(prompt_token_ids)}."
                )
            if capture_generated_tokens:
                residual_token_count = min(
                    available_token_count,
                    len(prompt_token_ids) + generated_token_count,
                )
                captured_generated_token_count = max(0, residual_token_count - len(prompt_token_ids))
                residual_token_sections = _with_generation_token_sections(
                    token_sections=residual_token_sections,
                    prompt_token_count=len(prompt_token_ids),
                    captured_generated_token_count=captured_generated_token_count,
                )
                residual_section_records = _with_generation_section_records(
                    section_records=residual_section_records,
                    prompt_token_count=len(prompt_token_ids),
                    captured_generated_token_count=captured_generated_token_count,
                )
            residual = hs[:, :residual_token_count, :].detach().cpu().to(torch.float32).numpy()
            connector_file.unlink(missing_ok=True)
        results.append(
            {
                "example": example,
                "prompt_token_ids": prompt_token_ids,
                "prompt_token_sections": tokenized_prompt["token_sections"],
                "prompt_section_records": tokenized_prompt.get("section_records", []),
                "residual_token_count": residual_token_count,
                "residual_token_sections": residual_token_sections,
                "residual_section_records": residual_section_records,
                "generated_token_count": generated_token_count,
                "captured_generated_token_count": captured_generated_token_count,
                "residual": residual,
                "router_data": router_by_example.get(example.key, {}),
                "actual_router_layers": actual_router_layers,
                "generation_result": generation_result,
            }
        )
    return results


def _with_generation_token_sections(
    *,
    token_sections: dict[str, list[int]],
    prompt_token_count: int,
    captured_generated_token_count: int,
) -> dict[str, list[int]]:
    token_count = int(prompt_token_count) + int(captured_generated_token_count)
    sections = {str(name): [int(position) for position in positions] for name, positions in token_sections.items()}
    sections["prompt"] = list(range(int(prompt_token_count)))
    sections["generated"] = list(range(int(prompt_token_count), token_count))
    sections["full"] = list(range(token_count))
    return sections


def _with_generation_section_records(
    *,
    section_records: list[dict[str, Any]],
    prompt_token_count: int,
    captured_generated_token_count: int,
) -> list[dict[str, Any]]:
    token_count = int(prompt_token_count) + int(captured_generated_token_count)
    records = [dict(record) for record in section_records]
    existing_names = {str(record.get("name") or "") for record in records}
    extras = (
        {
            "name": "prompt",
            "unit": "segment",
            "index": 0,
            "token_positions": list(range(int(prompt_token_count))),
        },
        {
            "name": "generated",
            "unit": "segment",
            "index": 1,
            "token_positions": list(range(int(prompt_token_count), token_count)),
        },
        {
            "name": "full",
            "unit": "segment",
            "index": 2,
            "token_positions": list(range(token_count)),
        },
    )
    for record in extras:
        if record["name"] in existing_names:
            continue
        if not record["token_positions"]:
            continue
        records.append(record)
    return records


def _split_router_capture_batch(
    *,
    raw_router: dict[int, dict[str, Any]],
    examples: list[Example],
    prompt_lengths: list[int],
    allow_trailing_rows: bool = False,
) -> dict[str, dict[int, dict[str, Any]]]:
    import numpy as np

    total_tokens = int(sum(int(length) for length in prompt_lengths))
    result: dict[str, dict[int, dict[str, Any]]] = {example.key: {} for example in examples}
    offsets: list[tuple[int, int]] = []
    start = 0
    for length in prompt_lengths:
        end = start + int(length)
        offsets.append((start, end))
        start = end

    for layer, payload in raw_router.items():
        logits = np.asarray(payload["logits"], dtype=np.float32)
        topk_ids = np.asarray(payload["topk_ids"], dtype=np.int64)
        topk_weights = np.asarray(payload["topk_weights"], dtype=np.float32)
        captured_tokens = int(logits.shape[0])
        if captured_tokens < total_tokens:
            raise RuntimeError(
                "Captured MoE router rows are shorter than the prompt token count for the batch. "
                f"Layer {int(layer)} captured {captured_tokens} rows, expected at least {total_tokens}. "
                "This usually means prefix caching skipped prompt execution or vLLM changed token packing order."
            )
        if captured_tokens > total_tokens and not allow_trailing_rows:
            raise RuntimeError(
                "Captured MoE router rows exceed the prompt token count for the batch. "
                f"Layer {int(layer)} captured {captured_tokens} rows, expected {total_tokens}. "
                "This usually means decode-token router rows were included when only prompt rows were expected."
            )
        logits = logits[:total_tokens]
        topk_ids = topk_ids[:total_tokens]
        topk_weights = topk_weights[:total_tokens]
        for example, (item_start, item_end) in zip(examples, offsets, strict=False):
            result[example.key][int(layer)] = {
                "logits": logits[item_start:item_end],
                "topk_ids": topk_ids[item_start:item_end],
                "topk_weights": topk_weights[item_start:item_end],
            }
    return result


def _generation_result_from_output(
    request_output: Any,
    *,
    capture_reasoning: bool,
    reasoning_parser: Any | None = None,
) -> dict[str, Any]:
    completion = request_output.outputs[0] if getattr(request_output, "outputs", None) else None
    raw_text = str(getattr(completion, "text", "")) if completion is not None else ""
    token_ids = [int(token) for token in getattr(completion, "token_ids", [])] if completion is not None else []
    reasoning_text = _reasoning_text(request_output, completion)
    output_text = raw_text

    if reasoning_parser is not None and _text_contains_reasoning_markers(reasoning_parser, raw_text):
        parsed_reasoning, parsed_output = _extract_reasoning_from_text(reasoning_parser, raw_text)
        if parsed_reasoning:
            reasoning_text = parsed_reasoning
        output_text = parsed_output

    payload = {
        "generated_token_ids": list(token_ids),
        "text": output_text,
        "finish_reason": (
            str(getattr(completion, "finish_reason"))
            if completion is not None and getattr(completion, "finish_reason", None) is not None
            else ""
        ),
        "request_id": str(getattr(request_output, "request_id", "") or ""),
    }
    if capture_reasoning:
        payload["reasoning_text"] = reasoning_text
    return payload


def _apply_structured_output_constraint(
    sampling_params: Any,
    schema: dict[str, Any] | None,
) -> None:
    if schema is None:
        return
    from vllm.sampling_params import StructuredOutputsParams

    sampling_params.structured_outputs = StructuredOutputsParams(json=dict(schema))


def _fill_router_features(
    *,
    feature_payloads: dict[str, dict[str, Any]],
    routing_sites: list[MoERoutingSite],
    router_data: dict[int, dict[str, Any]],
    example: Example,
    token_count: int,
    token_sections: dict[str, list[int]],
    section_records: list[dict[str, Any]] | None = None,
    discovered_router_layers: list[int] | None = None,
) -> None:
    if not routing_sites:
        return
    available_layers = sorted(int(layer) for layer in router_data.keys())
    discovered_layers = sorted(int(layer) for layer in (discovered_router_layers or []))
    for site in routing_sites:
        positions = site.tokens.resolve(token_count, token_sections=token_sections)
        feature_token_sections = rebase_token_sections(
            token_sections=token_sections,
            selected_positions=positions,
        )
        feature_section_records = rebase_section_records(
            section_records=section_records or [],
            selected_positions=positions,
        )
        for layer in site.layers:
            layer_int = int(layer)
            if layer_int not in router_data:
                raise RuntimeError(
                    "Requested MoE routing layer "
                    f"{layer_int}, but vLLM did not capture it for example {example.key!r}. "
                    f"Requested router layers={sorted(int(item) for item in site.layers)}; "
                    f"captured router layers={available_layers}; "
                    f"discovered MoE layers={discovered_layers}"
                )
            layer_payload = router_data[layer_int]
            logits = layer_payload["logits"]
            topk_ids = layer_payload.get("topk_ids")
            topk_weights = layer_payload.get("topk_weights")
            records_by_token: dict[str, Any] = {}
            for pos in positions:
                if int(pos) >= int(logits.shape[0]):
                    raise RuntimeError(
                        "Requested router token position "
                        f"{pos} for layer {layer_int} on example {example.key!r}, "
                        f"but captured router logits only have length {int(logits.shape[0])}. "
                        f"Requested positions={positions}; "
                        f"token_count={token_count}; "
                        f"captured router layers={available_layers}; "
                        f"discovered MoE layers={discovered_layers}"
                    )
                token_logits = logits[pos]
                token_topk_ids = topk_ids[pos] if topk_ids is not None else None
                token_topk_weights = topk_weights[pos] if topk_weights is not None else None
                records_by_token[str(pos)] = _routing_records(
                    site.record,
                    token_logits,
                    observed_topk_ids=token_topk_ids,
                    observed_topk_weights=token_topk_weights,
                )
            feature_payloads[site.name]["layers"][str(layer)][example.key] = {
                "tokens": positions,
                "records": records_by_token,
                "prompt_hash": example.prompt_hash,
                "token_sections": feature_token_sections,
                "section_records": feature_section_records,
            }


def _routing_records(
    requested: list[RoutingRecord] | tuple[RoutingRecord, ...],
    logits: Any,
    *,
    observed_topk_ids: Any | None = None,
    observed_topk_weights: Any | None = None,
) -> dict[str, Any]:
    token_records: dict[str, Any] = {}
    topk_from_gate_k = requested_topk_from_gate_k(requested, fallback=8)
    for record in requested:
        token_records.update(
            routing_record_payload(
                record,
                logits,
                topk_from_gate_k=topk_from_gate_k,
                fallback_top_k=8,
                observed_topk_ids=observed_topk_ids,
                observed_topk_weights=observed_topk_weights,
            )
        )
    return token_records


def _reasoning_text(request_output: Any, completion: Any) -> str:
    for candidate in (
        getattr(completion, "reasoning_content", None) if completion is not None else None,
        getattr(completion, "reasoning", None) if completion is not None else None,
        getattr(request_output, "reasoning_content", None),
        getattr(request_output, "reasoning", None),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate
        if isinstance(candidate, dict):
            text = candidate.get("text") or candidate.get("content")
            if isinstance(text, str) and text.strip():
                return text
    return ""


def _build_reasoning_parser(
    *,
    tokenizer: Any,
    parser_name: str | None,
    enable_thinking: bool | None,
) -> Any | None:
    normalized = str(parser_name or "").strip()
    if not normalized:
        return None
    try:
        from vllm.reasoning import ReasoningParserManager
    except Exception:
        return None
    parser_cls = ReasoningParserManager.get_reasoning_parser(normalized)
    parser_kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        parser_kwargs["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
    try:
        return parser_cls(tokenizer, **parser_kwargs)
    except TypeError:
        return parser_cls(tokenizer)


def _text_contains_reasoning_markers(reasoning_parser: Any, text: str) -> bool:
    start_token = getattr(reasoning_parser, "start_token", "")
    end_token = getattr(reasoning_parser, "end_token", "")
    return bool((start_token and start_token in text) or (end_token and end_token in text))


def _extract_reasoning_from_text(reasoning_parser: Any, raw_text: str) -> tuple[str, str]:
    try:
        reasoning, content = reasoning_parser.extract_reasoning(raw_text, request=None)
    except Exception:
        return "", raw_text
    reasoning_text = str(reasoning or "")
    output_text = str(content or "")
    if not reasoning_text:
        return "", raw_text
    return reasoning_text, output_text


def _iter_batches(items: list[Example], batch_size: int) -> list[list[Example]]:
    return [items[start:start + batch_size] for start in range(0, len(items), batch_size)]
