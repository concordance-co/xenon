"""Full generation + residual capture for ethical-vs-self-advantage v2."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pipelines_v2.api import (
    ArtifactDatasetSource,
    CaptureSpec,
    Dataset,
    Example,
    GenerationRunSpec,
    GenerationSpec,
    ModalResources,
    ModalRunnerSpec,
    ReportSpec,
    ResidualSite,
    StepRef,
    TensorStorage,
    TokenSelector,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)

from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base
from projects.MOREBENCH.ethical_advantage_vectors.phase_01.specs import (
    behavior_smoke_workflow as smoke,
)


WORKFLOW_NAME = "morebench_ethical_advantage_vectors_phase01_v2_full_capture"
PHASE_ROOT = Path("projects/MOREBENCH/ethical_advantage_vectors/phase_01")
REPORT_OUTPUT_DIR = PHASE_ROOT / "reports" / "v2_full_capture"

CAPTURED_LAYERS = (16, 24, 32, 40)
GENERATION_MAX_TOKENS = smoke.GENERATION_MAX_TOKENS
GENERATION_TEMPERATURE = smoke.GENERATION_TEMPERATURE
GENERATION_TOP_P = smoke.GENERATION_TOP_P


def _last_non_whitespace_span(text: str, start: int, end: int) -> dict[str, int]:
    index = int(end) - 1
    while index >= int(start) and text[index].isspace():
        index -= 1
    if index < int(start):
        index = max(int(start), int(end) - 1)
    return {"char_start": index, "char_end": index + 1}


def _render_prompt_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, (list, tuple)):
        parts: list[str] = []
        for message in prompt:
            if isinstance(message, Mapping):
                role = str(message.get("role") or "").strip()
                content = message.get("content") or ""
                if isinstance(content, list):
                    content = " ".join(
                        str(item.get("text") or "") if isinstance(item, Mapping) else str(item)
                        for item in content
                    )
                label = role.capitalize() if role else "Message"
                parts.append(f"{label}:\n{content}")
            else:
                parts.append(str(message))
        return "\n\n".join(part for part in parts if part.strip())
    return str(prompt)


def _combined_prompt_and_sections(*, source_prompt: str, generated_text: str) -> tuple[str, dict[str, Any]]:
    separator = "\n\nAssistant response:\n"
    prompt_end = len(source_prompt)
    generated_start = prompt_end + len(separator)
    combined = f"{source_prompt}{separator}{generated_text}"
    generated_end = len(combined)
    return combined, {
        "prompt": {"char_start": 0, "char_end": prompt_end},
        "prompt_end": _last_non_whitespace_span(combined, 0, prompt_end),
        "generated": {"char_start": generated_start, "char_end": generated_end},
        "full": {"char_start": 0, "char_end": generated_end},
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_dataset() -> Dataset:
    return smoke.build_dataset()


def build_capture_dataset(*, generation: Any) -> dict[str, Any]:
    payload = generation.result()
    if not isinstance(payload, Mapping):
        raise TypeError("generation artifact result must be a mapping")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError("generation artifact must contain a rows list")

    examples: list[Example] = []
    skipped_length = 0
    skipped_empty = 0
    finish_reasons: Counter[str] = Counter()
    generated_texts: dict[str, str] = {}
    response_lengths: dict[str, int] = {}
    condition_labels: dict[str, str] = {}
    pole_labels: dict[str, str] = {}

    for row in raw_rows:
        if not isinstance(row, Mapping):
            continue
        source_example = _mapping(row.get("example"))
        key = str(row.get("example_key") or source_example.get("key") or "").strip()
        if not key:
            continue

        prompt_labels = dict(_mapping(source_example.get("labels")))
        finish_reason = str(row.get("finish_reason") or "")
        finish_reasons[finish_reason] += 1
        generated_text = str(row.get("generated_text") or row.get("text") or "")
        source_prompt = _render_prompt_text(source_example.get("prompt") or "")

        if finish_reason == "length":
            skipped_length += 1
            continue
        if not generated_text.strip() or not source_prompt.strip():
            skipped_empty += 1
            continue

        combined_prompt, token_sections = _combined_prompt_and_sections(
            source_prompt=source_prompt,
            generated_text=generated_text,
        )
        token_ids = row.get("generated_token_ids")
        labels = {
            **prompt_labels,
            "generated_text": generated_text,
            "generation_finish_reason": finish_reason,
            "generated_token_count": len(token_ids) if isinstance(token_ids, list) else 0,
            "response_char_length": len(generated_text),
        }
        metadata = {
            **_mapping(source_example.get("metadata")),
            "source_generation_artifact_id": getattr(generation, "id", ""),
            "token_sections": token_sections,
        }
        examples.append(
            Example(
                key=key,
                prompt=combined_prompt,
                labels=labels,
                metadata=metadata,
                cases={"dilemma_id": str(prompt_labels.get("dilemma_id") or key)},
                case_key=str(prompt_labels.get("dilemma_id") or key),
            )
        )
        generated_texts[key] = generated_text
        response_lengths[key] = len(generated_text)
        condition_labels[key] = str(prompt_labels.get("condition_id") or "")
        pole_labels[key] = str(prompt_labels.get("pole") or "")

    dataset = Dataset.from_examples(
        examples,
        name="morebench_ethical_advantage_vectors_phase01_v2_capture_dataset",
    )
    return {
        "payload": {
            "kind": "ethical_advantage_v2_capture_dataset",
            "dataset": dataset.to_dict(),
            "summary": {
                "source_generation_artifact_id": getattr(generation, "id", ""),
                "input_row_count": len(raw_rows),
                "kept_capture_example_count": len(examples),
                "skipped_length": skipped_length,
                "skipped_empty": skipped_empty,
                "finish_reason_counts": dict(sorted(finish_reasons.items())),
            },
        },
        "labels": {
            "generated_text": generated_texts,
            "response_char_length": response_lengths,
            "condition_id": condition_labels,
            "pole": pole_labels,
        },
        "metadata": {
            "source": "GenerationRunSpec result rows",
            "unit": "dilemma_id x condition_id x sample_index",
        },
        "example_keys": sorted(generated_texts),
    }


def summarize_capture(*, capture_result: Any, capture_dataset: Any) -> TransformResult:
    payload = capture_dataset.result() if hasattr(capture_dataset, "result") else {}
    summary = _mapping(_mapping(payload.get("payload")).get("summary"))
    return TransformResult(
        payload={
            "workflow": WORKFLOW_NAME,
            "capture_feature_artifact_id": getattr(capture_result, "id", ""),
            "captured_layers": list(CAPTURED_LAYERS),
            "capture_token_sections": ["prompt_end", "generated"],
            "kept_capture_example_count": summary.get("kept_capture_example_count"),
            "skipped_length": summary.get("skipped_length"),
            "skipped_empty": summary.get("skipped_empty"),
            "finish_reason_counts": summary.get("finish_reason_counts"),
            "source_generation_artifact_id": summary.get("source_generation_artifact_id"),
        }
    )


def _artifact_capture_dataset() -> Dataset:
    return Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef("build_capture_dataset"),
        result_key="dataset",
        provides_token_sections=True,
        name="morebench_ethical_advantage_vectors_phase01_v2_capture_dataset",
    )


def build_runner_specs() -> dict[str, object]:
    specs = smoke.build_runner_specs()
    generate_gpu = specs["generate_gpu"]
    specs["analysis_cpu"] = ModalRunnerSpec(
        resources=ModalResources(
            cpu=4,
            memory_mb=12 * 1024,
            timeout_seconds=60 * 60,
            secrets=generate_gpu.resources.secrets,
        ),
        artifacts=generate_gpu.artifacts,
        catalog=generate_gpu.catalog,
    )
    return specs


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="generate_v2_responses",
                runner="generate_gpu",
                description=(
                    "Generate full v2 ethical-vs-short-term-self-advantage responses with two "
                    "question-suffix variants per condition."
                ),
                spec=GenerationRunSpec(
                    engine=base._engine(),
                    dataset=dataset,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=GENERATION_MAX_TOKENS,
                        temperature=GENERATION_TEMPERATURE,
                        top_p=GENERATION_TOP_P,
                    ),
                ),
            ),
            WorkflowStep(
                name="build_capture_dataset",
                runner="analysis_cpu",
                description="Convert generated responses into prompt-end/generated token-section capture rows.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_capture_dataset,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_v2_responses")},
                ),
            ),
            WorkflowStep(
                name="capture_residuals",
                runner="generate_gpu",
                description="Capture prompt-end and generated-sequence residuals at L16/L24/L32/L40.",
                spec=CaptureSpec(
                    engine=base._engine(max_num_seqs=8),
                    dataset=_artifact_capture_dataset(),
                    sites=[
                        ResidualSite(
                            name="prompt_end_residual",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.section("prompt_end"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                        ResidualSite(
                            name="generated_sequence_residual",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.section("generated"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ],
                    generation=GenerationSpec(enabled=False),
                ),
            ),
            WorkflowStep(
                name="summarize_capture",
                runner="analysis_cpu",
                description="Compact post-capture summary.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_capture,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={
                        "capture_result": StepRef("capture_residuals"),
                        "capture_dataset": StepRef("build_capture_dataset"),
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                description="Package v2 generation and capture artifacts for local review.",
                spec=ReportSpec(
                    inputs=(
                        StepRef("generate_v2_responses"),
                        StepRef("build_capture_dataset"),
                        StepRef("capture_residuals"),
                        StepRef("summarize_capture"),
                    ),
                    template="default",
                    output_dir=str(REPORT_OUTPUT_DIR),
                ),
            ),
        ),
    )
