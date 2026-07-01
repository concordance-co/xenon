"""vLLM engine descriptor."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import PurePosixPath
from typing import Any

from pipelines_v2.core.types import EngineCapability
from pipelines_v2.engine.base import (
    EngineCaptureResult,
    EngineGenerationResult,
    EngineInterventionResult,
    PythonRuntimeSpec,
)
from pipelines_v2.operations.interventions import GenerationRunSpec, PatchedGenerationSpec
from pipelines_v2.operations.specs import CaptureSpec, MoERoutingSite


@dataclass(frozen=True, slots=True)
class VLLMEngine:
    """vLLM engine configuration and execution surface."""

    model_id: str
    model_path_root: str | None = None
    max_model_len: int | None = None
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    distributed_executor_backend: str | None = None
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = True
    max_num_seqs: int = 1
    max_num_batched_tokens: int | None = None
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = False
    async_scheduling: bool = False
    add_generation_prompt: bool = False
    reasoning_parser: str | None = None
    enable_thinking: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VLLMEngine":
        data = dict(payload)
        data.pop("kind", None)
        field_names = {f.name for f in fields(cls)}
        known = {key: value for key, value in data.items() if key in field_names}
        unknown = {key: value for key, value in data.items() if key not in field_names}
        if unknown:
            extra = dict(known.get("extra") or {})
            extra.update(unknown)
            known["extra"] = extra
        return cls(**known)

    @classmethod
    def from_file(cls, path: str) -> "VLLMEngine":
        import json
        from pathlib import Path

        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)

    def identity(self) -> dict[str, Any]:
        return {
            "kind": "vllm",
            "model_id": self.model_id,
            "model_path_root": self.model_path_root,
            "max_model_len": self.max_model_len,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "distributed_executor_backend": self.distributed_executor_backend,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "async_scheduling": self.async_scheduling,
            "add_generation_prompt": self.add_generation_prompt,
            "reasoning_parser": self.reasoning_parser,
            "enable_thinking": self.enable_thinking,
            "extra": self.extra,
        }

    def semantic_identity(self) -> dict[str, Any]:
        return {
            "kind": "vllm",
            "model_id": self.model_id,
            "max_model_len": self.max_model_len,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "add_generation_prompt": self.add_generation_prompt,
            "reasoning_parser": self.reasoning_parser,
            "enable_thinking": self.enable_thinking,
            "extra": self.extra,
        }

    def extra_llm_kwargs(self) -> dict[str, Any]:
        """Return explicitly allowed passthrough kwargs for ``vllm.LLM``.

        Keep this narrow: ``extra`` is also used for tokenizer/chat-template
        settings, and letting it override core engine fields would make run
        identity and safety checks unreliable.
        """
        payload = dict(self.extra.get("llm_kwargs") or {})
        for key in ("attention_backend", "attention_config"):
            if key in self.extra:
                payload[key] = self.extra[key]
        allowed = {"attention_backend", "attention_config"}
        unsupported = sorted(str(key) for key in payload if key not in allowed)
        if unsupported:
            raise ValueError(
                "Unsupported VLLMEngine.extra llm_kwargs keys: "
                + ", ".join(unsupported)
                + ". Supported keys: "
                + ", ".join(sorted(allowed))
            )
        return payload

    def resolved_model_path(self) -> str:
        """Return the concrete model path/name vLLM should load."""
        model_id = str(self.model_id)
        root = str(self.model_path_root or "").strip()
        if not root or model_id.startswith(("/", ".")):
            return model_id
        return str(PurePosixPath(root) / model_id)

    def canonical_model_name(self) -> str:
        """Return the semantic model id to expose in serving/logging metadata."""
        return str(self.model_id)

    def capabilities(self) -> set[EngineCapability]:
        return {
            EngineCapability.GENERATION,
            EngineCapability.LOGPROBS,
            EngineCapability.RESIDUAL_CAPTURE,
            EngineCapability.MOE_ROUTING_CAPTURE,
            EngineCapability.STRUCTURED_OUTPUT,
            EngineCapability.ACTIVATION_PATCHING,
            EngineCapability.REQUEST_SCOPED_INTERVENTIONS,
        }

    def runtime_spec(self) -> PythonRuntimeSpec:
        env = {
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "VLLM_COMPILE_CACHE_SAVE_FORMAT": "binary",
        }
        debug_value = str(os.getenv("XENON_ACTIVATION_PATCH_DEBUG", "") or "").strip()
        if debug_value:
            env["XENON_ACTIVATION_PATCH_DEBUG"] = debug_value
        # vllm is pinned because residual capture relies on the
        # `extract_hidden_states` speculative-decoding path, which has changed
        # shape across minor releases (0.19.1 shipped a regression in the v1
        # engine's input validator). Unpinned installs mean any Modal image
        # rebuild can silently change engine behavior; bump deliberately and
        # re-verify capture before raising the pin. Keep in sync with
        # yora/modal_base.py.
        return PythonRuntimeSpec(
            pip_packages=(
                "matplotlib",
                "vllm==0.19.0",
                "torch",
                "transformers",
                "safetensors",
                "numpy",
                "huggingface_hub",
            ),
            env=env,
            local_python_sources=("pipelines_v2",),
        )

    def planning_errors(self, spec: Any) -> tuple[str, ...]:
        errors: list[str] = []
        wants_routing = isinstance(spec, CaptureSpec) and any(isinstance(site, MoERoutingSite) for site in spec.sites)
        if wants_routing and bool(self.enable_prefix_caching):
            errors.append(
                "VLLMEngine currently requires enable_prefix_caching=False for MoE routing capture; "
                "prefix caching can skip execution on shared prompt prefixes and make router token positions incomplete."
            )
        if isinstance(spec, PatchedGenerationSpec):
            if bool(self.enable_prefix_caching):
                errors.append(
                    "VLLMEngine currently requires enable_prefix_caching=False for PatchedGenerationSpec; "
                    "prefix caching can skip prompt execution and bypass prompt-side patch writes."
                )
        return tuple(errors)

    def capture(self, spec: CaptureSpec) -> EngineCaptureResult:
        from .capture import run_vllm_capture

        return run_vllm_capture(engine=self, spec=spec)

    def generate(self, spec: GenerationRunSpec) -> EngineGenerationResult:
        from .generate import run_vllm_generation

        return run_vllm_generation(engine=self, spec=spec)

    def generate_incremental(self, spec: GenerationRunSpec, *, batch_callback: Any) -> EngineGenerationResult:
        from .generate import run_vllm_generation

        return run_vllm_generation(engine=self, spec=spec, batch_callback=batch_callback)

    def intervene(self, spec: PatchedGenerationSpec) -> EngineInterventionResult:
        from .intervene import run_vllm_intervention

        return run_vllm_intervention(engine=self, spec=spec)
