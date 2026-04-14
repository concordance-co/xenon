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
        if any(isinstance(site, MoERoutingSite) for site in spec.sites) and int(self.max_num_seqs or 1) > 1:
            errors.append(
                "VLLMEngine does not currently support MoE routing capture with max_num_seqs > 1; "
                "split router capture into its own serial step or set max_num_seqs=1."
            )
        return tuple(errors)

    def capture(self, spec: CaptureSpec) -> EngineCaptureResult:
        from .capture import run_vllm_capture

        return run_vllm_capture(engine=self, spec=spec)
