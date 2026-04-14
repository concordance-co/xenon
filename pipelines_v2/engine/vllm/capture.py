"""vLLM capture implementation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pipelines_v2.data.datasets import Example
from pipelines_v2.engine.base import EngineCaptureResult
from pipelines_v2.engine.prompt_metadata import rebase_token_sections
from pipelines_v2.engine.prompt_metadata import resolve_prompt_metadata, token_sections_from_metadata
from pipelines_v2.operations.specs import CaptureSpec, MoERoutingSite, ResidualSite, RoutingRecord

if TYPE_CHECKING:
    from pipelines_v2.engine.vllm.engine import VLLMEngine


def run_vllm_capture(*, engine: VLLMEngine, spec: CaptureSpec) -> EngineCaptureResult:
    """Run a capture spec with vLLM in the current process."""

    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    examples = list(spec.dataset.examples)
    residual_sites = [site for site in spec.sites if isinstance(site, ResidualSite)]
    routing_sites = [site for site in spec.sites if isinstance(site, MoERoutingSite)]
    residual_layers = sorted({int(layer) for site in residual_sites for layer in site.layers})
    wants_residual = bool(residual_sites)
    wants_routing = bool(routing_sites)
    wants_generation = bool(spec.generation.enabled)
    batch_size = max(1, int(engine.max_num_seqs or 1))
    wants_sections = any(getattr(site.tokens, "kind", None) == "section" for site in spec.sites)

    if wants_routing and batch_size > 1:
        raise ValueError("MoE routing capture requires max_num_seqs == 1 in the current vLLM implementation")

    llm_kwargs: dict[str, Any] = {
        "model": engine.model_id,
        "enforce_eager": bool(engine.enforce_eager),
        "max_num_seqs": int(engine.max_num_seqs or 1),
        "enable_chunked_prefill": bool(engine.enable_chunked_prefill),
        "enable_prefix_caching": bool(engine.enable_prefix_caching),
        "tensor_parallel_size": int(engine.tensor_parallel_size or 1),
        "gpu_memory_utilization": float(engine.gpu_memory_utilization or 0.90),
    }
    if engine.max_model_len:
        llm_kwargs["max_model_len"] = int(engine.max_model_len)
    reasoning_parser = (engine.reasoning_parser or "").strip()
    if not reasoning_parser and "qwen3" in str(engine.model_id).lower():
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
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": str(connector_dir),
                },
            }

        tokenizer = AutoTokenizer.from_pretrained(engine.model_id, trust_remote_code=True)
        llm = LLM(**llm_kwargs)

        router_enabled = False
        if wants_routing:
            from functools import partial

            buffer_size = int(engine.max_model_len or 32768)
            router_enabled = bool(_apply_to_model(llm, partial(_setup_router_capture_on_model, max_tokens=buffer_size)))
            if not router_enabled:
                raise RuntimeError("MoE routing capture was requested, but no compatible MoE blocks were found")

        feature_payloads: dict[str, dict[str, Any]] = {site.name: _empty_feature(site) for site in spec.sites}
        generations: list[dict[str, Any]] = []
        example_metadata: list[dict[str, Any]] = []

        if wants_residual and not wants_routing and batch_size > 1 and len(examples) > 1:
            for batch in _iter_batches(examples, batch_size):
                batch_records = _capture_residual_batch(
                    llm=llm,
                    tokenizer=tokenizer,
                    examples=batch,
                    add_generation_prompt=bool(engine.add_generation_prompt),
                    require_sections=wants_sections,
                    prompt_metadata_builder=spec.prompt_metadata_builder,
                )
                for record in batch_records:
                    example = record["example"]
                    prompt_token_ids = record["prompt_token_ids"]
                    token_sections = record["token_sections"]
                    residual = record["residual"]

                    _fill_residual_features(
                        feature_payloads=feature_payloads,
                        residual_sites=residual_sites,
                        residual_layers=residual_layers,
                        residual=residual,
                        example=example,
                        token_count=len(prompt_token_ids),
                        token_sections=token_sections,
                    )

                    generation_result = None
                    if wants_generation:
                        generation_result = _generate_one(
                            llm=llm,
                            prompt_token_ids=prompt_token_ids,
                            max_tokens=int(spec.generation.max_tokens or 1),
                            temperature=float(spec.generation.temperature or 0.0),
                        )
                        if not spec.generation.capture_reasoning:
                            generation_result["reasoning_text"] = ""
                        generations.append({"example_key": example.key, **generation_result})

                    example_metadata.append(
                        {
                            "example_key": example.key,
                            "prompt_hash": example.prompt_hash,
                            "token_count": len(prompt_token_ids),
                            "generated": generation_result is not None,
                            "capture_mode": "batched_residual",
                        }
                    )
        else:
            for example in examples:
                prompt_token_ids = _prompt_token_ids(
                    tokenizer=tokenizer,
                    example=example,
                    add_generation_prompt=bool(engine.add_generation_prompt),
                    require_sections=wants_sections,
                    prompt_metadata_builder=spec.prompt_metadata_builder,
                )
                token_sections = prompt_token_ids["token_sections"]
                prompt_token_ids = prompt_token_ids["token_ids"]
                if wants_routing:
                    _apply_to_model(llm, _reset_router_buffers_on_model)

                residual = None
                if wants_residual or wants_routing:
                    outputs = llm.generate(
                        prompts=[{"prompt_token_ids": prompt_token_ids}],
                        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
                    )
                    request_output = outputs[0]
                    if wants_residual:
                        hidden_states_path = _hidden_states_path(request_output)
                        if not hidden_states_path:
                            raise RuntimeError("vLLM did not return hidden_states_path in kv_transfer_params")
                        tensors = load_file(hidden_states_path)
                        hs = tensors["hidden_states"] if "hidden_states" in tensors else next(iter(tensors.values()))
                        if hs.dim() == 3:
                            hs = hs.permute(1, 0, 2)
                        residual = hs[:, : len(prompt_token_ids), :].detach().cpu().to(torch.float32).numpy()
                        Path(hidden_states_path).unlink(missing_ok=True)

                router_data: dict[int, Any] = {}
                if wants_routing:
                    raw_router = _apply_to_model(llm, _collect_router_logits_from_model)
                    router_data = {
                        int(layer): tensor.detach().cpu().to(torch.float32).numpy()
                        for layer, tensor in raw_router.items()
                    }

                _fill_residual_features(
                    feature_payloads=feature_payloads,
                    residual_sites=residual_sites,
                    residual_layers=residual_layers,
                    residual=residual,
                    example=example,
                    token_count=len(prompt_token_ids),
                    token_sections=token_sections,
                )
                _fill_router_features(
                    feature_payloads=feature_payloads,
                    routing_sites=routing_sites,
                    router_data=router_data,
                    example=example,
                    token_count=len(prompt_token_ids),
                    token_sections=token_sections,
                )

                generation_result = None
                if wants_generation:
                    generation_result = _generate_one(
                        llm=llm,
                        prompt_token_ids=prompt_token_ids,
                        max_tokens=int(spec.generation.max_tokens or 1),
                        temperature=float(spec.generation.temperature or 0.0),
                    )
                    if not spec.generation.capture_reasoning:
                        generation_result["reasoning_text"] = ""
                    generations.append({"example_key": example.key, **generation_result})

                example_metadata.append(
                    {
                        "example_key": example.key,
                        "prompt_hash": example.prompt_hash,
                        "token_count": len(prompt_token_ids),
                        "generated": generation_result is not None,
                        "capture_mode": "single_request",
                    }
                )

    return EngineCaptureResult(
        features=feature_payloads,
        generations=generations,
        metadata={
            "backend": "vllm",
            "example_metadata": example_metadata,
            "router_enabled": router_enabled,
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


def _collect_router_logits_from_model(model: Any) -> dict[int, Any]:
    from pipelines_v2.engine.vllm.moe_hooks import collect_router_logits

    return collect_router_logits(model)


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
            "observed_routing_decisions": False,
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
) -> dict[str, Any]:
    rendered = _render_prompt(tokenizer=tokenizer, prompt=example.prompt, add_generation_prompt=add_generation_prompt)
    metadata = resolve_prompt_metadata(
        metadata=example.metadata,
        rendered_prompt=rendered,
        builder=prompt_metadata_builder,
    )
    encoding = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = _normalize_token_ids(encoding)
    offsets = _normalize_offsets(encoding)
    token_sections = token_sections_from_metadata(
        metadata=metadata,
        offsets=offsets,
        require_sections=require_sections,
        allow_char_spans=True,
    )
    return {
        "token_ids": token_ids,
        "token_sections": token_sections,
    }


def _render_prompt(*, tokenizer: Any, prompt: Any, add_generation_prompt: bool) -> str:
    if isinstance(prompt, list):
        rendered = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
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
        for layer in site.layers:
            idx = layer_index[int(layer)]
            values = residual[idx, positions, :]
            feature_payloads[site.name]["layers"][str(layer)][example.key] = {
                "tokens": positions,
                "values": values,
                "prompt_hash": example.prompt_hash,
                "token_sections": feature_token_sections,
            }


def _capture_residual_batch(
    *,
    llm: Any,
    tokenizer: Any,
    examples: list[Example],
    add_generation_prompt: bool,
    require_sections: bool,
    prompt_metadata_builder: Any | None,
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
        )
        prompt_token_ids = tokenized_prompt["token_ids"]
        prompts.append({"prompt_token_ids": prompt_token_ids})
        tokenized_by_key[example.key] = tokenized_prompt

    outputs = llm.generate(
        prompts=prompts,
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
    )

    results: list[dict[str, Any]] = []
    for example, request_output in zip(examples, outputs, strict=False):
        hidden_states_path = _hidden_states_path(request_output)
        if not hidden_states_path:
            raise RuntimeError(f"vLLM did not return hidden_states_path for batched example {example.key!r}")
        connector_file = Path(hidden_states_path)
        if not connector_file.exists():
            raise RuntimeError(
                f"Hidden-state connector file missing for batched example {example.key!r}: {hidden_states_path}"
            )
        tensors = load_file(str(connector_file))
        hs = tensors["hidden_states"] if "hidden_states" in tensors else next(iter(tensors.values()))
        if hs.dim() == 3:
            hs = hs.permute(1, 0, 2)
        tokenized_prompt = tokenized_by_key[example.key]
        prompt_token_ids = tokenized_prompt["token_ids"]
        residual = hs[:, : len(prompt_token_ids), :].detach().cpu().to(torch.float32).numpy()
        connector_file.unlink(missing_ok=True)
        results.append(
            {
                "example": example,
                "prompt_token_ids": prompt_token_ids,
                "token_sections": tokenized_prompt["token_sections"],
                "residual": residual,
            }
        )
    return results


def _fill_router_features(
    *,
    feature_payloads: dict[str, dict[str, Any]],
    routing_sites: list[MoERoutingSite],
    router_data: dict[int, Any],
    example: Example,
    token_count: int,
    token_sections: dict[str, list[int]],
) -> None:
    if not routing_sites:
        return
    for site in routing_sites:
        positions = site.tokens.resolve(token_count, token_sections=token_sections)
        feature_token_sections = rebase_token_sections(
            token_sections=token_sections,
            selected_positions=positions,
        )
        for layer in site.layers:
            layer_int = int(layer)
            if layer_int not in router_data:
                raise RuntimeError(f"Requested MoE routing layer {layer_int}, but vLLM did not capture it")
            logits = router_data[layer_int]
            records_by_token: dict[str, Any] = {}
            for pos in positions:
                token_logits = logits[pos]
                records_by_token[str(pos)] = _routing_records(site.record, token_logits)
            feature_payloads[site.name]["layers"][str(layer)][example.key] = {
                "tokens": positions,
                "records": records_by_token,
                "prompt_hash": example.prompt_hash,
                "token_sections": feature_token_sections,
            }


def _routing_records(requested: list[RoutingRecord] | tuple[RoutingRecord, ...], logits: Any) -> dict[str, Any]:
    token_records: dict[str, Any] = {}
    topk_from_gate_k = _requested_topk_from_gate_k(requested, fallback=8)
    for record in requested:
        token_records.update(_routing_record(record, logits, topk_from_gate_k=topk_from_gate_k))
    return token_records


def _routing_record(record: RoutingRecord, logits: Any, *, topk_from_gate_k: int) -> dict[str, Any]:
    import numpy as np

    if record.kind == "gate_logits":
        return {"gate_logits": logits.astype(_float_dtype(record.params.get("dtype", "float16")))}
    if record.kind == "gate_probs":
        return {"gate_probs": _softmax(logits).astype(_float_dtype(record.params.get("dtype", "float16")))}
    if record.kind == "routing_decisions":
        if record.params.get("required", True):
            raise RuntimeError("Observed routing decisions are not exposed by the current vLLM MoE hook")
        return {"routing_decisions": {"source": "not_observed", "expert_ids": [], "weights": []}}
    if record.kind == "topk_from_gate":
        k = int(record.params["k"])
        indices = np.argsort(logits)[::-1][:k].astype(np.int64)
        values = logits[indices].astype(np.float32)
        payload: dict[str, Any] = {
            "source": "derived_from_gate_logits",
            "expert_ids": indices,
        }
        if record.params.get("include_weights", True):
            payload["weights"] = _normalize(values)
        return {"topk_from_gate": payload}
    if record.kind == "expert_load":
        source = str(record.params.get("source") or "topk_from_gate")
        topk = np.argsort(logits)[::-1][:topk_from_gate_k if source == "topk_from_gate" else 8]
        return {"expert_load": {"source": source, "counts": {str(int(idx)): 1 for idx in topk}}}
    raise ValueError(f"Unsupported routing record: {record.kind}")


def _generate_one(
    *,
    llm: Any,
    prompt_token_ids: list[int],
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    from vllm import SamplingParams

    outputs = llm.generate(
        prompts=[{"prompt_token_ids": prompt_token_ids}],
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=temperature),
    )
    request_output = outputs[0]
    completion = request_output.outputs[0] if getattr(request_output, "outputs", None) else None
    return {
        "generated_token_ids": [
            int(token) for token in getattr(completion, "token_ids", [])
        ] if completion is not None else [],
        "text": str(getattr(completion, "text", "")) if completion is not None else "",
        "finish_reason": (
            str(getattr(completion, "finish_reason"))
            if completion is not None and getattr(completion, "finish_reason", None) is not None
            else ""
        ),
        "reasoning_text": _reasoning_text(request_output, completion),
        "request_id": str(getattr(request_output, "request_id", "") or ""),
    }


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


def _softmax(values: Any) -> Any:
    import numpy as np

    shifted = values.astype(np.float32) - np.max(values)
    exps = np.exp(shifted)
    return (exps / np.sum(exps)).astype(np.float32)


def _normalize(values: Any) -> Any:
    import numpy as np

    if values.size == 0:
        return np.asarray([], dtype=np.float32)
    shifted = values.astype(np.float32) - np.min(values) + np.float32(1e-6)
    return (shifted / np.sum(shifted)).astype(np.float32)


def _float_dtype(name: str) -> Any:
    import numpy as np

    normalized = str(name).lower()
    if normalized == "float16":
        return np.float16
    if normalized in {"float32", "bfloat16"}:
        return np.float32
    raise ValueError(f"Unsupported routing dtype: {name}")


def _requested_topk_from_gate_k(
    requested: list[RoutingRecord] | tuple[RoutingRecord, ...],
    *,
    fallback: int,
) -> int:
    for record in requested:
        if record.kind == "topk_from_gate":
            return int(record.params["k"])
    return int(fallback)


def _iter_batches(items: list[Example], batch_size: int) -> list[list[Example]]:
    return [items[start:start + batch_size] for start in range(0, len(items), batch_size)]
