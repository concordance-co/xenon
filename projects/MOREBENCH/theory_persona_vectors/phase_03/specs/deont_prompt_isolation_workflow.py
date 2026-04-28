"""Controlled deontology prompt-isolation workflow for phase 03.

This mini-workflow is for a confound-isolation retry after the natural-response
deontology transfer test showed substantial text transfer. It narrows the
surface variation by:

- restricting to deontology + anchors only
- forcing a fixed three-line output schema
- banning both deont lexical families from the response text

Goal: make `P_deont_iso_01` vs `P_deont_iso_02` differ mainly at the prompt
level, so we can re-test whether activation transfer outperforms text transfer.
"""

from __future__ import annotations

import json
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
    ReportSpec,
    ResidualSite,
    StepRef,
    TensorStorage,
    TokenSelector,
    TransformBuilder,
    TransformSpec,
    TransformResult,
    WorkflowSpec,
    WorkflowStep,
)

from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base
from projects.MOREBENCH.theory_persona_vectors.phase_01.specs import deontology_pole_pilot_workflow as p1
from projects.MOREBENCH.theory_persona_vectors.phase_03.specs import all_theories_natural_prompt_workflow as natural


WORKFLOW_NAME = "morebench_theory_persona_vectors_phase03_deont_prompt_isolation"
PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
CONDITIONS_PATH = PHASE_ROOT / "specs" / "deont_prompt_isolation_conditions.json"
REPORT_OUTPUT_DIR = PHASE_ROOT / "reports" / "deont_prompt_isolation_report"
BEHAVIOR_REPORT_OUTPUT_DIR = PHASE_ROOT / "reports" / "deont_prompt_isolation_behavior_smoke"

BEHAVIOR_SMOKE_DILEMMA_COUNT = 8
SAMPLES_PER_CONDITION = 1
GENERATION_MAX_TOKENS = 160
GENERATION_TEMPERATURE = 0.7
GENERATION_TOP_P = 0.95
CAPTURED_LAYERS = (0, 4, 16, 24, 32, 40)
SYSTEM_PROMPT = p1.SYSTEM_PROMPT
QUESTION_SUFFIX = (
    "Respond using exactly these three lines:\n"
    "Recommendation: <one sentence>\n"
    "Reason: <one sentence>\n"
    "Caveat: <one sentence>\n\n"
    "Keep the whole answer between 45 and 60 words. Use plain language. "
    "Do not mention the instruction, moral theory names, or any of these words: "
    "duty, duties, right, rights, promise, promises, obligation, obligations, "
    "constraint, constraints, commitment, commitments, boundary, boundaries, forbidden."
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_dataset(*, dilemma_limit: int | None = None) -> Dataset:
    dilemmas = natural._read_jsonl(natural.DILEMMAS_PATH)
    if dilemma_limit is not None:
        dilemmas = dilemmas[:dilemma_limit]

    conditions_payload = _read_json(CONDITIONS_PATH)
    conditions: list[dict[str, Any]] = list(conditions_payload["conditions"])

    examples: list[Example] = []
    for dilemma in dilemmas:
        dilemma_id = str(dilemma["dilemma_id"])
        dilemma_text = str(dilemma["dilemma"])
        domain = str(dilemma.get("domain") or "")
        conflict_axis = str(dilemma.get("conflict_axis") or "")
        for condition in conditions:
            condition_id = str(condition["condition_id"])
            instruction = str(condition.get("instruction") or "")
            role = str(condition.get("role") or "")
            theory = str(condition.get("theory") or "")
            user_message = natural._user_message(instruction=instruction, dilemma=dilemma_text).replace(
                natural.QUESTION_SUFFIX,
                QUESTION_SUFFIX,
            )
            for sample_index in range(SAMPLES_PER_CONDITION):
                example_key = f"{dilemma_id}__{condition_id}__sample_{sample_index:02d}"
                examples.append(
                    Example(
                        key=example_key,
                        prompt=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_message},
                        ],
                        labels={
                            "dilemma_id": dilemma_id,
                            "condition_id": condition_id,
                            "condition_role": role,
                            "condition_theory": theory,
                            "sample_index": sample_index,
                            "domain": domain,
                            "conflict_axis": conflict_axis,
                            "is_positive": role.startswith("positive"),
                            "is_positive_variant": role == "positive_variant",
                            "is_neutral_negative": role.startswith("neutral_negative"),
                            "is_generic_moral_anchor": role == "generic_moral_anchor",
                            "is_anti_diagnostic": role == "anti_theory_diagnostic",
                        },
                        metadata={
                            "instruction": instruction,
                            "dilemma_text": dilemma_text,
                            "dilemma_text_without_embedded_question": natural._strip_embedded_question(dilemma_text),
                            "question_suffix": QUESTION_SUFFIX,
                            "prompt_regime": "deont_prompt_isolation",
                        },
                        cases={"dilemma_id": dilemma_id, "condition_id": condition_id},
                        case_key=dilemma_id,
                    )
                )

    suffix = "behavior_smoke" if dilemma_limit is not None else "full"
    return Dataset.from_examples(
        examples,
        name=f"{WORKFLOW_NAME}_{suffix}_dataset",
    )


def build_behavior_smoke_dataset() -> Dataset:
    return _build_dataset(dilemma_limit=BEHAVIOR_SMOKE_DILEMMA_COUNT)


def build_dataset() -> Dataset:
    return _build_dataset()


def _artifact_capture_dataset() -> Dataset:
    return Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef("build_capture_dataset"),
        result_key="dataset",
        provides_token_sections=True,
        name=f"{WORKFLOW_NAME}_capture_dataset",
    )


def build_capture_dataset(*, generation: Any) -> dict[str, Any]:
    return natural.build_capture_dataset(generation=generation)


def summarize_pilot(*, capture_result: Any, capture_dataset: Any) -> TransformResult:
    capture_payload = capture_dataset.result() if hasattr(capture_dataset, "result") else {}
    summary = {}
    if isinstance(capture_payload, Mapping):
        payload = capture_payload.get("payload")
        if isinstance(payload, Mapping):
            summary = dict(payload.get("summary") or {})
    return TransformResult(
        payload={
            "workflow": WORKFLOW_NAME,
            "prompt_regime": "deont_prompt_isolation",
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


def _generation_step(dataset: Dataset) -> WorkflowStep:
    return WorkflowStep(
        name="generate_natural_responses",
        runner="capture_gpu",
        description=(
            "Generate controlled deontology-anchor responses under a fixed three-line "
            "output schema with banned cue-word families."
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
    )


def build_behavior_smoke_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_behavior_smoke_dataset()
    return WorkflowSpec(
        name=f"{WORKFLOW_NAME}_behavior_only",
        steps=(
            _generation_step(dataset),
            WorkflowStep(
                name="report",
                runner="report_local",
                description="Package the controlled deontology behavior-only smoke for local inspection.",
                spec=ReportSpec(
                    inputs=(StepRef("generate_natural_responses"),),
                    template="default",
                    output_dir=str(BEHAVIOR_REPORT_OUTPUT_DIR),
                ),
            ),
        ),
    )


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            _generation_step(dataset),
            WorkflowStep(
                name="build_capture_dataset",
                runner="analysis_cpu",
                description=(
                    "Filter the controlled-response batch into a capture dataset, dropping empty rows "
                    "but keeping length-finished rows with labels."
                ),
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_capture_dataset,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_natural_responses")},
                ),
            ),
            WorkflowStep(
                name="capture_residuals",
                runner="capture_gpu",
                description=(
                    "Capture prompt-end and generated-sequence residuals on the controlled deontology "
                    "prompt-isolation dataset."
                ),
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
                name="summarize_pilot",
                runner="analysis_cpu",
                description="Compact post-capture summary for the controlled deontology isolation smoke.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_pilot,
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
                description="Package controlled deontology generation+capture artifacts for local browsing.",
                spec=ReportSpec(
                    inputs=(
                        StepRef("generate_natural_responses"),
                        StepRef("build_capture_dataset"),
                        StepRef("capture_residuals"),
                        StepRef("summarize_pilot"),
                    ),
                    template="default",
                    output_dir=str(REPORT_OUTPUT_DIR),
                ),
            ),
        ),
    )


def build_runner_specs() -> dict[str, object]:
    return natural.build_runner_specs()
