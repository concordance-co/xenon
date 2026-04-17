"""vLLM raw generation execution."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from pipelines_v2.data.datasets import Dataset
from pipelines_v2.engine.base import EngineGenerationResult
from pipelines_v2.operations.capture import CaptureSpec
from pipelines_v2.operations.interventions.runtime import resolve_generation_examples

from .capture import run_vllm_capture

if TYPE_CHECKING:
    from pipelines_v2.engine.vllm.engine import VLLMEngine
    from pipelines_v2.operations.interventions import GenerationRunSpec


def run_vllm_generation(*, engine: "VLLMEngine", spec: "GenerationRunSpec") -> EngineGenerationResult:
    selected_examples = resolve_generation_examples(spec)
    capture_spec = CaptureSpec(
        engine=engine,
        dataset=Dataset.from_examples(
            selected_examples,
            id=spec.dataset.id,
            name=spec.dataset.name,
        ),
        sites=(),
        generation=replace(spec.generation, enabled=True),
    )
    capture_result = run_vllm_capture(engine=engine, spec=capture_spec)
    rows = [
        {
            "example_key": example.key,
            "example": example.to_dict(),
            "generated_text": str(item.get("text") or ""),
            "generated_token_ids": list(item.get("generated_token_ids") or ()),
            "finish_reason": str(item.get("finish_reason") or ""),
            "request_id": str(item.get("request_id") or ""),
            **(
                {"reasoning_text": str(item.get("reasoning_text") or "")}
                if "reasoning_text" in item
                else {}
            ),
            **(
                {"structured_output": item.get("structured_output")}
                if "structured_output" in item
                else {}
            ),
        }
        for example, item in zip(selected_examples, capture_result.generations, strict=False)
    ]
    return EngineGenerationResult(
        rows=rows,
        metadata=dict(capture_result.metadata),
    )
