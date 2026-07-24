"""Reusable vLLM session runtime for model-bound operation batches."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from pipelines_v2.core.types import OperationSpec, stable_hash
from pipelines_v2.operations.specs import (
    CaptureSpec,
    GenerationRunSpec,
    PatchedGenerationSpec,
    ResidualSite,
)

from .intervention_build import build_llm_kwargs

if TYPE_CHECKING:
    from pipelines_v2.engine.base import (
        EngineCaptureResult,
        EngineGenerationResult,
        EngineInterventionResult,
    )
    from pipelines_v2.engine.vllm.engine import VLLMEngine


_SUBSPACE_PATCH_OPERATORS = frozenset(
    {"project_out", "add_direction", "swap_mean", "swap_components", "random_control"}
)
_SESSION_COMPATIBLE_PATCH_FAMILIES = frozenset({"subspace", "paired"})


@dataclass(slots=True)
class VLLMSessionRuntime:
    """One loaded vLLM instance reused across compatible operation specs."""

    engine: "VLLMEngine"
    llm: Any
    tokenizer: Any
    reasoning_parser_instance: Any | None
    batch_size: int
    session_key: str
    _tempdir: tempfile.TemporaryDirectory[str] | None = None

    def capture(
        self,
        spec: CaptureSpec,
        *,
        batch_callback: Any | None = None,
        progress_callback: Any | None = None,
    ) -> "EngineCaptureResult":
        from .capture import run_vllm_capture_with_runtime

        return run_vllm_capture_with_runtime(
            runtime=self,
            spec=spec,
            batch_callback=batch_callback,
            progress_callback=progress_callback,
        )

    def generate(
        self,
        spec: GenerationRunSpec,
        *,
        batch_callback: Any | None = None,
        progress_callback: Any | None = None,
    ) -> "EngineGenerationResult":
        from .generate import run_vllm_generation_with_runtime

        return run_vllm_generation_with_runtime(
            runtime=self,
            spec=spec,
            batch_callback=batch_callback,
            progress_callback=progress_callback,
        )

    def intervene(self, spec: PatchedGenerationSpec) -> "EngineInterventionResult":
        from .intervene import run_vllm_intervention_with_runtime

        return run_vllm_intervention_with_runtime(runtime=self, spec=spec)

    def close(self) -> None:
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None


def build_vllm_session_runtime(
    *,
    engine: "VLLMEngine",
    specs: Sequence[OperationSpec],
    progress_callback: Any | None = None,
) -> VLLMSessionRuntime:
    """Construct one loaded vLLM runtime for a compatible operation group."""

    from transformers import AutoTokenizer
    from vllm import LLM

    specs_tuple = tuple(specs)
    llm_kwargs, reasoning_parser, tempdir = build_vllm_session_llm_kwargs(
        engine=engine,
        specs=specs_tuple,
    )
    from .model_load_progress import (
        enable_model_load_progress,
        model_load_progress_monitor,
    )

    enable_model_load_progress(llm_kwargs, progress_callback)
    tokenizer = AutoTokenizer.from_pretrained(engine.resolved_model_path(), trust_remote_code=True)
    try:
        with model_load_progress_monitor(progress_callback):
            llm = LLM(**llm_kwargs)
    except Exception:
        if tempdir is not None:
            tempdir.cleanup()
        raise

    from .capture import _build_reasoning_parser

    reasoning_parser_instance = _build_reasoning_parser(
        tokenizer=tokenizer,
        parser_name=reasoning_parser,
        enable_thinking=engine.enable_thinking,
    )
    return VLLMSessionRuntime(
        engine=engine,
        llm=llm,
        tokenizer=tokenizer,
        reasoning_parser_instance=reasoning_parser_instance,
        batch_size=max(1, int(engine.max_num_seqs or 1)),
        session_key=vllm_session_key(engine=engine, specs=specs_tuple),
        _tempdir=tempdir,
    )


def build_vllm_session_llm_kwargs(
    *,
    engine: "VLLMEngine",
    specs: Sequence[OperationSpec],
) -> tuple[dict[str, Any], str, tempfile.TemporaryDirectory[str] | None]:
    """Return vLLM construction kwargs for the superset needed by ``specs``."""

    specs_tuple = tuple(specs)
    patch_specs = [spec for spec in specs_tuple if isinstance(spec, PatchedGenerationSpec)]
    _validate_patch_families(patch_specs)
    compiled_operator_hint = _compiled_operator_hint_for_specs(patch_specs)
    if patch_specs:
        llm_kwargs, reasoning_parser = build_llm_kwargs(
            engine,
            compiled_operator_hint=compiled_operator_hint,
        )
    else:
        llm_kwargs = _base_llm_kwargs(engine)
        reasoning_parser = _reasoning_parser_for_specs(engine=engine, specs=specs_tuple)
        if reasoning_parser:
            llm_kwargs["structured_outputs_config"] = {"reasoning_parser": reasoning_parser}

    residual_layers = _residual_layers_for_specs(specs_tuple)
    tempdir: tempfile.TemporaryDirectory[str] | None = None
    if residual_layers:
        tempdir = tempfile.TemporaryDirectory(prefix="pipelines_v2_vllm_")
        connector_dir = Path(tempdir.name) / "hidden_states"
        connector_dir.mkdir(parents=True, exist_ok=True)
        capture_generated_tokens = any(
            isinstance(spec, CaptureSpec) and bool(spec.generation.capture_generated_tokens)
            for spec in specs_tuple
        )
        llm_kwargs["speculative_config"] = {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": residual_layers,
                }
            },
        }
        connector_config: dict[str, Any] = {
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
            connector_config["kv_connector_module_path"] = "pipelines_v2.engine.vllm.hidden_states_connector"
        llm_kwargs["kv_transfer_config"] = connector_config
    return llm_kwargs, reasoning_parser, tempdir


def vllm_session_key(
    *,
    engine: "VLLMEngine",
    specs: Sequence[OperationSpec],
) -> str:
    """Return a stable key for specs that can share one loaded vLLM instance."""

    specs_tuple = tuple(specs)
    patch_specs = [spec for spec in specs_tuple if isinstance(spec, PatchedGenerationSpec)]
    _validate_patch_families(patch_specs)
    llm_kwargs, reasoning_parser, tempdir = build_vllm_session_llm_kwargs(
        engine=engine,
        specs=specs_tuple,
    )
    if tempdir is not None:
        tempdir.cleanup()
    normalized_kwargs = _normalize_session_kwargs(llm_kwargs)
    return stable_hash(
        {
            "kind": "vllm_session_runtime",
            "engine": engine.identity(),
            "llm_kwargs": normalized_kwargs,
            "reasoning_parser": reasoning_parser,
            "patch_family": sorted({_patch_family(spec) for spec in patch_specs}),
            "spec_kinds": sorted({spec.kind for spec in specs_tuple}),
        }
    )


def vllm_modal_batch_family(spec: OperationSpec) -> str | None:
    """Return the broad vLLM Modal batch family for one model-bound spec."""

    if isinstance(spec, PatchedGenerationSpec):
        return _patch_family(spec)
    if isinstance(spec, (CaptureSpec, GenerationRunSpec)):
        return "generation"
    return None


def _base_llm_kwargs(engine: "VLLMEngine") -> dict[str, Any]:
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
    if engine.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = int(engine.max_num_batched_tokens)
    if bool(engine.async_scheduling):
        llm_kwargs["async_scheduling"] = True
    llm_kwargs.update(engine.extra_llm_kwargs())
    return llm_kwargs


def _residual_layers_for_specs(specs: Sequence[OperationSpec]) -> list[int]:
    return sorted(
        {
            int(layer)
            for spec in specs
            if isinstance(spec, CaptureSpec)
            for site in spec.sites
            if isinstance(site, ResidualSite)
            for layer in site.layers
        }
    )


def _reasoning_parser_for_specs(
    *,
    engine: "VLLMEngine",
    specs: Sequence[OperationSpec],
) -> str:
    reasoning_parser = (engine.reasoning_parser or "").strip()
    if reasoning_parser:
        return reasoning_parser
    wants_reasoning = any(
        isinstance(spec, (CaptureSpec, GenerationRunSpec, PatchedGenerationSpec))
        and bool(spec.generation.capture_reasoning)
        for spec in specs
    )
    if wants_reasoning and "qwen3" in str(engine.model_id).lower():
        return "qwen3"
    return ""


def _compiled_operator_hint_for_specs(specs: Sequence[PatchedGenerationSpec]) -> str | None:
    hints = {
        "subspace"
        for spec in specs
        if spec.patch is not None and spec.patch.operator in _SUBSPACE_PATCH_OPERATORS
    }
    if len(hints) > 1:
        raise ValueError(f"Incompatible vLLM compiled operator hints in one session: {sorted(hints)}")
    return next(iter(hints), None)


def _patch_family(spec: PatchedGenerationSpec) -> str:
    if spec.patch is not None and spec.patch.operator in _SUBSPACE_PATCH_OPERATORS:
        return "subspace"
    if spec.patch is not None and spec.patch.requires_pairing():
        return "paired"
    return str(spec.patch.operator if spec.patch is not None else "unknown")


def _validate_patch_families(specs: Sequence[PatchedGenerationSpec]) -> None:
    families = {_patch_family(spec) for spec in specs}
    incompatible = families - _SESSION_COMPATIBLE_PATCH_FAMILIES
    if incompatible and len(families) > 1:
        raise ValueError(f"Cannot share one loaded vLLM session across patch families: {sorted(families)}")
    if len(incompatible) > 1:
        raise ValueError(f"Cannot share one loaded vLLM session across patch families: {sorted(families)}")


def _normalize_session_kwargs(llm_kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(llm_kwargs)
    kv_transfer_config = normalized.get("kv_transfer_config")
    if isinstance(kv_transfer_config, dict):
        kv_transfer_config = dict(kv_transfer_config)
        extra = kv_transfer_config.get("kv_connector_extra_config")
        if isinstance(extra, dict) and "shared_storage_path" in extra:
            extra = dict(extra)
            extra["shared_storage_path"] = "<session-tempdir>"
            kv_transfer_config["kv_connector_extra_config"] = extra
        normalized["kv_transfer_config"] = kv_transfer_config
    return normalized


__all__ = [
    "VLLMSessionRuntime",
    "build_vllm_session_llm_kwargs",
    "build_vllm_session_runtime",
    "vllm_modal_batch_family",
    "vllm_session_key",
]
