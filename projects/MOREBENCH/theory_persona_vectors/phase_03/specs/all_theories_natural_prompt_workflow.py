"""Natural-prompt all-theories persona-vector smoke workflow (phase 03).

Phase 01/02 over-constrained the model with "Give only the recommendation.
Do not explain." This workflow removes that constraint and asks only:
"What would you recommend?"

Two entry points are intentionally exposed:

- build_behavior_smoke_dataset + build_behavior_smoke_workflow:
  small generation-only read to verify the prompt produces signal-bearing text.
- build_dataset + build_workflow:
  full 30-dilemma generation plus prompt-end/generated residual capture.

Phase: theory_persona_vectors / phase_03.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pipelines_v2.api import (
    ArtifactDatasetSource,
    CaptureSpec,
    Dataset,
    Example,
    FileCatalog,
    GenerationRunSpec,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PostgresCatalog,
    PostgresSource,
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
from projects.MOREBENCH.theory_persona_vectors.phase_01.specs import (
    deontology_pole_pilot_workflow as p1,
)


WORKFLOW_NAME = "morebench_theory_persona_vectors_phase03_brief_recommendation_smoke"
PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_03")
DILEMMAS_PATH = PHASE_ROOT / "outputs" / "all_theories_natural_prompt_synth_dilemmas.jsonl"
CONDITIONS_PATH = PHASE_ROOT / "specs" / "all_theories_natural_prompt_conditions.json"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "morebench_theory_persona_vectors_phase03"
LOCAL_CATALOG_ROOT = Path("artifacts") / "morebench_theory_persona_vectors_phase03_catalog"
MODAL_ARTIFACT_ROOT = "/data/artifacts/morebench_theory_persona_vectors_phase03"
REPORT_OUTPUT_DIR = PHASE_ROOT / "reports" / "all_theories_brief_recommendation_report"
BEHAVIOR_REPORT_OUTPUT_DIR = PHASE_ROOT / "reports" / "all_theories_brief_recommendation_behavior_smoke"

BEHAVIOR_SMOKE_DILEMMA_COUNT = 8
SAMPLES_PER_CONDITION = 1
GENERATION_MAX_TOKENS = 384
GENERATION_TEMPERATURE = 0.7
GENERATION_TOP_P = 0.95
CAPTURED_LAYERS = (0, 4, 16, 24, 32, 40)
SYSTEM_PROMPT = p1.SYSTEM_PROMPT
DB_ENV_VAR = p1.DB_ENV_VAR

QUESTION_SUFFIX = "Briefly describe your recommendation to the scenario."


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _condition_filter() -> set[str] | None:
    raw = os.getenv("MOREBENCH_PHASE03_CONDITION_IDS", "").strip()
    if not raw:
        return None
    return {item.strip() for item in raw.split(",") if item.strip()}


def _workflow_name() -> str:
    filtered = _condition_filter()
    if not filtered:
        return WORKFLOW_NAME
    suffix = "_".join(sorted(item.replace("N_anti_", "anti_").replace("P_", "").replace("_01", "").replace("_02", "") for item in filtered))
    suffix = re.sub(r"[^A-Za-z0-9_]+", "_", suffix).strip("_")
    return f"{WORKFLOW_NAME}_{suffix}"


def _report_output_dir() -> Path:
    filtered = _condition_filter()
    if not filtered:
        return REPORT_OUTPUT_DIR
    return REPORT_OUTPUT_DIR / _workflow_name()


def _strip_embedded_question(dilemma: str) -> str:
    """Remove the old forced-choice question so phase 03 controls the ask."""
    text = dilemma.strip()
    text = re.sub(r"\s+What should [^?]+\?\s*$", "", text)
    return text.strip()


def _user_message(*, instruction: str, dilemma: str) -> str:
    case_text = _strip_embedded_question(dilemma)
    parts: list[str] = []
    if instruction.strip():
        parts.append(instruction.strip())
    parts.append(f"Dilemma: {case_text}")
    parts.append(QUESTION_SUFFIX)
    return "\n\n".join(parts)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _build_dataset(*, dilemma_limit: int | None = None) -> Dataset:
    dilemmas = _read_jsonl(DILEMMAS_PATH)
    if dilemma_limit is not None:
        dilemmas = dilemmas[:dilemma_limit]

    conditions_payload = _read_json(CONDITIONS_PATH)
    conditions: list[dict[str, Any]] = list(conditions_payload["conditions"])
    condition_filter = _condition_filter()
    if condition_filter:
        conditions = [condition for condition in conditions if str(condition["condition_id"]) in condition_filter]
        found = {str(condition["condition_id"]) for condition in conditions}
        missing = sorted(condition_filter - found)
        if missing:
            raise KeyError(f"condition filter referenced unknown condition ids: {missing}")

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
            user_message = _user_message(instruction=instruction, dilemma=dilemma_text)
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
                            "dilemma_text_without_embedded_question": _strip_embedded_question(dilemma_text),
                            "question_suffix": QUESTION_SUFFIX,
                            "prompt_regime": "natural_recommendation",
                        },
                        cases={"dilemma_id": dilemma_id, "condition_id": condition_id},
                        case_key=dilemma_id,
                    )
                )

    suffix = "behavior_smoke" if dilemma_limit is not None else "full"
    return Dataset.from_examples(
        examples,
        name=f"{_workflow_name()}_{suffix}_dataset",
    )


def build_behavior_smoke_dataset() -> Dataset:
    """Small generation-only smoke: 8 dilemmas x 15 conditions = 120 examples."""
    return _build_dataset(dilemma_limit=BEHAVIOR_SMOKE_DILEMMA_COUNT)


def build_dataset() -> Dataset:
    """Full phase 03 dataset: 30 dilemmas x 15 conditions = 450 examples."""
    return _build_dataset()


def _artifact_capture_dataset() -> Dataset:
    return Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef("build_capture_dataset"),
        result_key="dataset",
        provides_token_sections=True,
        name=f"{_workflow_name()}_capture_dataset",
    )


def build_capture_dataset(*, generation: Any) -> dict[str, Any]:
    """Filter generation results into a capture dataset.

    Unlike phase_01/02, phase_03 keeps length-finished generations. Natural
    recommendation responses can be long, and dropping length rows would select
    against exactly the persona-expressive tail we are testing.
    """
    payload = generation.result()
    if not isinstance(payload, Mapping):
        raise TypeError("generation artifact result must be a mapping")
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raise TypeError("generation artifact must contain a rows list")

    examples: list[Example] = []
    skipped_empty = 0
    finish_reasons: Counter[str] = Counter()
    response_lengths: dict[str, int] = {}
    generated_texts: dict[str, str] = {}
    finish_reason_labels: dict[str, str] = {}
    condition_role_labels: dict[str, str] = {}

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
        source_prompt = p1._render_prompt_text(source_example.get("prompt") or "")

        if not generated_text.strip() or not source_prompt.strip():
            skipped_empty += 1
            continue

        combined_prompt, token_sections = p1._combined_prompt_and_sections(
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
            "kept_length_finished": finish_reason == "length",
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
        response_lengths[key] = len(generated_text)
        generated_texts[key] = generated_text
        finish_reason_labels[key] = finish_reason
        condition_role_labels[key] = str(prompt_labels.get("condition_role") or "")

    dataset = Dataset.from_examples(
        examples,
        name="morebench_theory_persona_vectors_phase03_natural_prompt_capture_dataset",
    )

    return {
        "payload": {
            "kind": "natural_prompt_capture_dataset",
            "dataset": dataset.to_dict(),
            "summary": {
                "source_generation_artifact_id": getattr(generation, "id", ""),
                "input_row_count": len(raw_rows),
                "kept_capture_example_count": len(examples),
                "kept_length_finished": int(finish_reasons.get("length", 0)),
                "skipped_length": 0,
                "skipped_empty": skipped_empty,
                "finish_reason_counts": dict(sorted(finish_reasons.items())),
            },
        },
        "labels": {
            "generated_text": generated_texts,
            "response_char_length": response_lengths,
            "generation_finish_reason": finish_reason_labels,
            "condition_role": condition_role_labels,
        },
        "metadata": {
            "source": "GenerationRunSpec result rows",
            "unit": "dilemma_id x condition_id x sample_index",
            "length_finished_policy": "kept_for_phase03_natural_prompt_smoke",
        },
        "example_keys": sorted(generated_texts),
    }


def build_runner_specs() -> dict[str, object]:
    import os

    if os.getenv(DB_ENV_VAR):
        db = PostgresSource.from_env(DB_ENV_VAR)
        catalog = PostgresCatalog(source=db)
        modal_secrets = (ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon"),)
    else:
        catalog = FileCatalog(root=LOCAL_CATALOG_ROOT)
        modal_secrets = ()

    modal_store = ModalVolumeStore(name="xenon-data", root=MODAL_ARTIFACT_ROOT)

    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 4,
                shard_count=base.GPU_SHARD_COUNT,
                secrets=modal_secrets,
                volumes=(ModalVolumeMount(name=base.MODEL_VOLUME_NAME, mount_path=base.MODEL_VOLUME_PATH),),
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=4,
                memory_mb=12 * 1024,
                timeout_seconds=60 * 60,
                secrets=modal_secrets,
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
            catalog=catalog,
        ),
    }


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
            "workflow_instance": _workflow_name(),
            "prompt_regime": "brief_recommendation",
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
            "Generate natural recommendation responses under all phase_03 pole hypotheses. "
            "The prompt asks 'Briefly describe your recommendation to the scenario.' and does "
            "not require terse or multi-sentence reasoning."
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
        name=f"{_workflow_name()}_behavior_only",
        steps=(
            _generation_step(dataset),
            WorkflowStep(
                name="report",
                runner="report_local",
                description="Package the behavior-only generation smoke for local inspection.",
                spec=ReportSpec(
                    inputs=(StepRef("generate_natural_responses"),),
                    template="default",
                    output_dir=str(_report_output_dir() / "behavior_only"),
                ),
            ),
        ),
    )


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()

    return WorkflowSpec(
        name=_workflow_name(),
        steps=(
            _generation_step(dataset),
            WorkflowStep(
                name="build_capture_dataset",
                runner="analysis_cpu",
                description=(
                    "Filter the natural-response batch into a capture dataset, dropping empty rows "
                    "but keeping length-finished rows with labels. Records prompt-end and generated "
                    "section spans."
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
                    "Capture prompt-end and generated-sequence residuals on the filtered natural-response "
                    "dataset. Layers: 0, 4, 16, 24, 32, 40."
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
                description="Compact post-capture summary for the natural-prompt all-theories smoke.",
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
                description="Package natural-prompt generation+capture artifacts for local browsing.",
                spec=ReportSpec(
                    inputs=(
                        StepRef("generate_natural_responses"),
                        StepRef("build_capture_dataset"),
                        StepRef("capture_residuals"),
                        StepRef("summarize_pilot"),
                    ),
                    template="default",
                    output_dir=str(_report_output_dir()),
                ),
            ),
        ),
    )
