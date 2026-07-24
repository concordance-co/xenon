"""vLLM raw generation execution."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable

from pipelines_v2.data.datasets import Dataset
from pipelines_v2.engine.base import EngineGenerationResult
from pipelines_v2.operations.capture import CaptureSpec
from pipelines_v2.operations.interventions.runtime import resolve_generation_examples

from .capture import run_vllm_capture, run_vllm_capture_with_runtime

if TYPE_CHECKING:
    from pipelines_v2.engine.vllm.engine import VLLMEngine
    from pipelines_v2.operations.interventions import GenerationRunSpec


def run_vllm_generation(
    *,
    engine: "VLLMEngine",
    spec: "GenerationRunSpec",
    batch_callback: Callable[[list[dict[str, Any]], dict[str, Any]], None] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> EngineGenerationResult:
    return _run_vllm_generation(
        engine=engine,
        spec=spec,
        batch_callback=batch_callback,
        capture_runner=lambda capture_spec, capture_callback, capture_progress: run_vllm_capture(
            engine=engine,
            spec=capture_spec,
            batch_callback=capture_callback,
            **(
                {"progress_callback": capture_progress}
                if capture_progress is not None
                else {}
            ),
        ),
        progress_callback=progress_callback,
    )


def run_vllm_generation_with_runtime(
    *,
    runtime: Any,
    spec: "GenerationRunSpec",
    batch_callback: Callable[[list[dict[str, Any]], dict[str, Any]], None] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> EngineGenerationResult:
    """Run generation through an already-loaded reusable vLLM runtime."""

    return _run_vllm_generation(
        engine=runtime.engine,
        spec=spec,
        batch_callback=batch_callback,
        capture_runner=lambda capture_spec, capture_callback, capture_progress: run_vllm_capture_with_runtime(
            runtime=runtime,
            spec=capture_spec,
            batch_callback=capture_callback,
            **(
                {"progress_callback": capture_progress}
                if capture_progress is not None
                else {}
            ),
        ),
        progress_callback=progress_callback,
    )


def _run_vllm_generation(
    *,
    engine: "VLLMEngine",
    spec: "GenerationRunSpec",
    batch_callback: Callable[[list[dict[str, Any]], dict[str, Any]], None] | None,
    capture_runner: Callable[[CaptureSpec, Any, Any], Any],
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> EngineGenerationResult:
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

    def _on_capture_batch(
        batch_examples: list[Any],
        batch_generations: list[dict[str, Any]],
        batch_metadata: list[dict[str, Any]],
    ) -> None:
        if batch_callback is None:
            return
        batch_rows = _generation_rows_from_outputs(batch_examples, batch_generations)
        batch_callback(
            batch_rows,
            {
                "backend": "vllm",
                "example_metadata": list(batch_metadata),
                "batch_example_count": len(batch_examples),
                "batch_row_count": len(batch_rows),
            },
        )

    capture_result = capture_runner(
        capture_spec,
        _on_capture_batch if batch_callback is not None else None,
        progress_callback,
    )
    rows = _generation_rows_from_outputs(selected_examples, capture_result.generations)
    return EngineGenerationResult(
        rows=rows,
        metadata=dict(capture_result.metadata),
    )


def _generation_rows_from_outputs(
    examples: list[Any],
    generations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(examples) != len(generations):
        raise RuntimeError(
            "Generation output count does not match selected example count: "
            f"got {len(generations)}, expected {len(examples)}"
        )
    return [
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
        for example, item in zip(examples, generations, strict=False)
    ]
