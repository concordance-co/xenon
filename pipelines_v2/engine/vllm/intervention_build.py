"""Shared payload-building helpers for patched vLLM generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pipelines_v2.data.datasets import Example
from pipelines_v2.operations.interventions import (
    AddDirectionPatch,
    InterchangePatch,
    PatchedGenerationSpec,
    ResidualPathPatch,
    SwapComponentsPatch,
    SwapMeanPatch,
)
from pipelines_v2.operations.interventions.runtime import load_path_mask_source

from .capture import _apply_structured_output_constraint

if TYPE_CHECKING:
    from pipelines_v2.engine.vllm.engine import VLLMEngine


_PATCH_WORKER_CLS = "pipelines_v2.engine.vllm.activation_patch_request_worker.ActivationPatchGPUWorker"


def target_policy_payload(spec: PatchedGenerationSpec) -> dict[str, Any]:
    application = getattr(spec.patch, "application", None)
    to_dict = getattr(application, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())
    return {"kind": "static", "include_prompt": True, "include_decode": False, "config": {}}


def build_llm_kwargs(
    engine: "VLLMEngine",
    *,
    compiled_operator_hint: str | None = None,
) -> tuple[dict[str, Any], str]:
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
        "worker_cls": _PATCH_WORKER_CLS,
    }
    if engine.distributed_executor_backend:
        llm_kwargs["distributed_executor_backend"] = str(engine.distributed_executor_backend)
    if model_path != engine.canonical_model_name():
        llm_kwargs["served_model_name"] = engine.canonical_model_name()
    if not bool(engine.enforce_eager):
        llm_kwargs["compilation_config"] = {
            "custom_ops": ["none", "+activation_patch_hidden_states"],
        }
        additional_config = {
            "xenon_activation_patch_worker_cls": _PATCH_WORKER_CLS,
        }
        hint = str(compiled_operator_hint or "").strip()
        if hint:
            additional_config["xenon_activation_patch_compiled_operator"] = hint
        llm_kwargs["additional_config"] = additional_config
    if engine.max_model_len:
        llm_kwargs["max_model_len"] = int(engine.max_model_len)
    if engine.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = int(engine.max_num_batched_tokens)
    if bool(engine.async_scheduling):
        llm_kwargs["async_scheduling"] = True
    reasoning_parser = (engine.reasoning_parser or "").strip()
    if not reasoning_parser and "qwen3" in str(engine.model_id).lower():
        reasoning_parser = "qwen3"
    if reasoning_parser:
        llm_kwargs["structured_outputs_config"] = {"reasoning_parser": reasoning_parser}
    return llm_kwargs, reasoning_parser


def build_activation_bank_runtime_payload(
    spec: PatchedGenerationSpec,
    activation_bank: Mapping[str, Any],
    resolved_cases: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    layers_payload = activation_bank["layers"]
    keys: set[str] = set()
    keys.update(item["donor"].key for item in resolved_cases)
    if not isinstance(spec.patch, InterchangePatch):
        keys.update(item["target"].key for item in resolved_cases)

    source_layers: set[int]
    if isinstance(spec.patch, InterchangePatch):
        source_layers = {int(spec.patch.source_layer_for(int(write_layer))) for write_layer in spec.patch.write_site.layers}
    else:
        assert isinstance(spec.patch, ResidualPathPatch)
        path_mask = load_path_mask_source(spec.patch)
        source_layers = {int(edge["source_layer"]) for edge in path_mask["edges"]}

    payload: dict[int, dict[str, Any]] = {}
    for source_layer in sorted(source_layers):
        layer_items: dict[str, Any] = {}
        layer_payload = dict(layers_payload[str(int(source_layer))])
        for key in sorted(keys):
            if key not in layer_payload:
                continue
            example_payload = dict(layer_payload[key])
            layer_items[str(key)] = {
                "values": example_payload["values"],
                "token_count": len(example_payload.get("values", ())),
                "token_sections": dict(example_payload.get("token_sections", {})),
            }
        payload[int(source_layer)] = layer_items
    return payload


def build_subspace_runtime_payload(patch: Any, source_payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    layers_payload = source_payload.get("layers")
    if not isinstance(layers_payload, Mapping):
        return {}
    required_source_layers = {
        int(patch.source_layer_for(int(write_layer)))
        for write_layer in patch.write_site.layers
    }
    payload: dict[int, dict[str, Any]] = {}
    for source_layer in required_source_layers:
        layer_payload = layers_payload.get(str(int(source_layer)))
        if isinstance(layer_payload, Mapping):
            payload[int(source_layer)] = dict(layer_payload)
    return payload


def build_direction_runtime_payload(patch: AddDirectionPatch, source_payload: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    layers_payload = source_payload.get("layers")
    if not isinstance(layers_payload, Mapping):
        return {}
    payload: dict[int, dict[str, Any]] = {}
    for source_layer in {int(patch.source_layer_for(int(write_layer))) for write_layer in patch.write_site.layers}:
        layer_payload = layers_payload.get(str(int(source_layer)))
        if isinstance(layer_payload, Mapping):
            payload[int(source_layer)] = dict(layer_payload)
    return payload


def build_centroid_runtime_payload(
    patch: SwapMeanPatch | SwapComponentsPatch,
    source_payload: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    layers_payload = source_payload.get("layers")
    if not isinstance(layers_payload, Mapping):
        return {}
    payload: dict[int, dict[str, Any]] = {}
    for source_layer in {int(patch.source_layer_for(int(write_layer))) for write_layer in patch.write_site.layers}:
        layer_payload = layers_payload.get(str(int(source_layer)))
        if isinstance(layer_payload, Mapping):
            payload[int(source_layer)] = dict(layer_payload)
    return payload


def paired_request_payload(
    *,
    spec: PatchedGenerationSpec,
    activation_bank: Mapping[str, Any],
    path_mask_payload: Mapping[str, Any] | None,
    target: Example,
    donor: Example,
    case_key: str,
    tokenized: Mapping[str, Any],
    target_positions: list[int],
) -> dict[str, Any] | str:
    common = {
        "operator": spec.patch.operator,
        "target_layers": [int(layer) for layer in spec.patch.write_site.layers],
        "target_positions": [int(pos) for pos in target_positions],
        "target_policy": target_policy_payload(spec),
        "source_layer_map": {
            str(int(write_layer)): int(spec.patch.source_layer_for(int(write_layer)))
            for write_layer in spec.patch.write_site.layers
        },
        "strength": float(spec.patch.strength),
        "example_key": target.key,
        "case_key": str(case_key),
    }

    if isinstance(spec.patch, InterchangePatch):
        donor_selector = spec.patch.donor_tokens or spec.patch.target_tokens
        donor_positions, issue = resolve_bank_positions(
            activation_bank=activation_bank,
            source_layer=int(spec.patch.source_layer_for(int(spec.patch.write_site.layers[0]))),
            example_key=donor.key,
            selector=donor_selector,
            expected_count=len(target_positions),
        )
        if issue is not None:
            return issue
        common["donor_example_key"] = donor.key
        common["donor_positions"] = [int(pos) for pos in donor_positions]
        return common

    assert isinstance(spec.patch, ResidualPathPatch)
    read_selector = spec.patch.read_tokens or spec.patch.target_tokens
    first_source_layer = first_path_source_layer(path_mask_payload)
    donor_positions, issue = resolve_bank_positions(
        activation_bank=activation_bank,
        source_layer=first_source_layer,
        example_key=donor.key,
        selector=read_selector,
        expected_count=len(target_positions),
    )
    if issue is not None:
        return issue
    target_read_positions, issue = resolve_bank_positions(
        activation_bank=activation_bank,
        source_layer=first_source_layer,
        example_key=target.key,
        selector=read_selector,
        expected_count=len(donor_positions),
    )
    if issue is not None:
        return issue
    common["donor_example_key"] = donor.key
    common["donor_positions"] = [int(pos) for pos in donor_positions]
    common["target_read_positions"] = [int(pos) for pos in target_read_positions]
    common["transport"] = spec.patch.transport
    common["path_edges"] = request_path_edges(
        path_mask_payload=path_mask_payload,
        write_layers=tuple(int(layer) for layer in spec.patch.write_site.layers),
    )
    return common


def unpaired_request_payload(
    *,
    spec: PatchedGenerationSpec,
    target: Example,
    target_positions: list[int],
) -> dict[str, Any]:
    payload = {
        "operator": spec.patch.operator,
        "target_layers": [int(layer) for layer in spec.patch.write_site.layers],
        "target_positions": [int(pos) for pos in target_positions],
        "target_policy": target_policy_payload(spec),
        "source_layer_map": {
            str(int(write_layer)): int(spec.patch.source_layer_for(int(write_layer)))
            for write_layer in spec.patch.write_site.layers
        },
        "strength": float(spec.patch.strength),
        "example_key": target.key,
    }
    component_map = getattr(spec.patch, "component_indices_by_layer", None)
    if component_map is not None:
        payload["component_indices_by_layer"] = {
            str(int(write_layer)): [int(index) for index in indices]
            for write_layer, indices in dict(component_map).items()
        }
    if hasattr(spec.patch, "random_seed"):
        payload["random_seed"] = int(spec.patch.random_seed)
    if hasattr(spec.patch, "match_projected_norm"):
        payload["match_projected_norm"] = bool(spec.patch.match_projected_norm)
    if hasattr(spec.patch, "centroid_name"):
        payload["centroid_name"] = str(spec.patch.centroid_name)
    return payload


def resolve_bank_positions(
    *,
    activation_bank: Mapping[str, Any],
    source_layer: int,
    example_key: str,
    selector: Any,
    expected_count: int,
) -> tuple[list[int], str | None]:
    first_layer_payload = activation_bank["layers"].get(str(int(source_layer)))
    if not isinstance(first_layer_payload, Mapping):
        return [], f"activation_bank is missing source layer {int(source_layer)}"
    donor_record = dict(first_layer_payload).get(example_key)
    if not isinstance(donor_record, Mapping):
        return [], "activation_bank is missing donor activation rows"
    donor_values = donor_record.get("values")
    donor_sections = donor_record.get("token_sections")
    donor_positions = selector.resolve(
        len(donor_values),
        token_sections=donor_sections,
    )
    if len(donor_positions) != int(expected_count):
        return list(donor_positions), "target and donor token selections must have equal length"
    return list(donor_positions), None


def first_path_source_layer(path_mask_payload: Mapping[str, Any] | None) -> int:
    edges = list(path_mask_payload.get("edges", ())) if isinstance(path_mask_payload, Mapping) else []
    if not edges:
        raise RuntimeError("ResidualPathPatch requires at least one path-mask edge")
    return int(edges[0]["source_layer"])


def request_path_edges(
    *,
    path_mask_payload: Mapping[str, Any] | None,
    write_layers: tuple[int, ...],
) -> list[dict[str, Any]]:
    if not isinstance(path_mask_payload, Mapping):
        return []
    allowed_write_layers = {int(layer) for layer in write_layers}
    return [
        {
            "source_layer": int(edge["source_layer"]),
            "write_layer": int(edge["write_layer"]),
            "weight": float(edge.get("weight", 1.0)),
        }
        for edge in path_mask_payload.get("edges", ())
        if int(edge["write_layer"]) in allowed_write_layers
    ]


def patched_sampling_params(
    *,
    max_tokens: int | None,
    temperature: float,
    top_p: float,
    top_k: int,
    structured_output: dict[str, Any] | None,
    extra_args: dict[str, Any],
) -> Any:
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        max_tokens=None if max_tokens is None else int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        extra_args=dict(extra_args),
    )
    _apply_structured_output_constraint(sampling_params, structured_output)
    return sampling_params


__all__ = [
    "build_activation_bank_runtime_payload",
    "build_centroid_runtime_payload",
    "build_direction_runtime_payload",
    "build_llm_kwargs",
    "build_subspace_runtime_payload",
    "first_path_source_layer",
    "paired_request_payload",
    "patched_sampling_params",
    "request_path_edges",
    "resolve_bank_positions",
    "unpaired_request_payload",
]
