"""vLLM engine descriptor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pipelines_v2.core.types import EngineCapability
from pipelines_v2.engine.base import EngineCaptureResult, PythonRuntimeSpec
from pipelines_v2.operations.specs import CaptureSpec, MoERoutingSite


@dataclass(frozen=True, slots=True)
class VLLMEngine:
    """vLLM engine configuration and execution surface."""

    model_id: str
    max_model_len: int | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = True
    max_num_seqs: int = 1
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = False
    add_generation_prompt: bool = False
    reasoning_parser: str | None = None
    enable_thinking: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VLLMEngine":
        data = dict(payload)
        data.pop("kind", None)
        return cls(**data)

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
            "max_model_len": self.max_model_len,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "enforce_eager": self.enforce_eager,
            "max_num_seqs": self.max_num_seqs,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_chunked_prefill": self.enable_chunked_prefill,
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
            "add_generation_prompt": self.add_generation_prompt,
            "reasoning_parser": self.reasoning_parser,
            "enable_thinking": self.enable_thinking,
            "extra": self.extra,
        }

    def capabilities(self) -> set[EngineCapability]:
        return {
            EngineCapability.GENERATION,
            EngineCapability.LOGPROBS,
            EngineCapability.RESIDUAL_CAPTURE,
            EngineCapability.MOE_ROUTING_CAPTURE,
        }

    def runtime_spec(self) -> PythonRuntimeSpec:
        return PythonRuntimeSpec(
            pip_packages=(
                "matplotlib",
                "vllm",
                "torch",
                "transformers",
                "safetensors",
                "numpy",
                "huggingface_hub",
            ),
            env={"VLLM_ALLOW_INSECURE_SERIALIZATION": "1"},
            local_python_sources=("pipelines_v2",),
        )

    def planning_errors(self, spec: CaptureSpec) -> tuple[str, ...]:
        errors: list[str] = []
        wants_routing = any(isinstance(site, MoERoutingSite) for site in spec.sites)
        if wants_routing and bool(self.enable_prefix_caching):
            errors.append(
                "VLLMEngine currently requires enable_prefix_caching=False for MoE routing capture; "
                "prefix caching can skip execution on shared prompt prefixes and make router token positions incomplete."
            )
        return tuple(errors)

    def capture(self, spec: CaptureSpec) -> EngineCaptureResult:
        from .capture import run_vllm_capture

        return run_vllm_capture(engine=self, spec=spec)
