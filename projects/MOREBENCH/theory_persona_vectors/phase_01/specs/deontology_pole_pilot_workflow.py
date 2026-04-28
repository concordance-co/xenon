"""Deontology persona-vector pole pilot workflow.

Generates terse 'just-recommend' responses on 30 synthetic dilemmas under 6
prompt conditions, with 3 samples per (dilemma, condition), then captures
prompt-end and generated-sequence residuals for L0-L40.

Output substrate is used to compute deontology persona-vector directions under
multiple negative-pole constructions and to evaluate transfer to existing
MoReBench captures.

Phase: theory_persona_vectors / phase_01.
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


WORKFLOW_NAME = "morebench_theory_persona_vectors_phase01_deontology_pole_pilot"
PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_01")
DILEMMAS_PATH = PHASE_ROOT / "outputs" / "deontology_pole_pilot_synth_dilemmas.jsonl"
CONDITIONS_PATH = PHASE_ROOT / "specs" / "deontology_pole_pilot_prompt_conditions.json"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "morebench_theory_persona_vectors_phase01"
LOCAL_CATALOG_ROOT = Path("artifacts") / "morebench_theory_persona_vectors_phase01_catalog"
MODAL_ARTIFACT_ROOT = "/data/artifacts/morebench_theory_persona_vectors_phase01"
REPORT_OUTPUT_DIR = PHASE_ROOT / "reports" / "deontology_pole_pilot_report"

SAMPLES_PER_CONDITION = 3
GENERATION_MAX_TOKENS = 96
GENERATION_TEMPERATURE = 0.7
GENERATION_TOP_P = 0.95

# Layer subset for the pilot. Includes a low-layer control (0/4), early
# representational layer (16), the prior MoReBench L32 prompt-side winner, and
# late-layer diagnostics (40). Avoids capturing all 48 layers to keep the
# pilot artifact small.
CAPTURED_LAYERS = (0, 4, 16, 24, 32, 40)

# System prompt aligned with prior MoReBench moral-decision generation. Kept
# minimal so the persona prompt itself dominates the priming signal.
SYSTEM_PROMPT = "You are answering a moral-decision question. Follow the user's instruction precisely."

DB_ENV_VAR = "XENON_NEON_DATABASE_URL"


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


def _user_message(*, instruction: str, dilemma: str) -> str:
    """Compose the user message: instruction, then the dilemma. The instruction
    contains the persona/neutral framing and the answer-only constraint.
    """
    return f"{instruction}\n\nDilemma: {dilemma}"


def _last_non_whitespace_span(text: str, start: int, end: int) -> dict[str, int]:
    index = int(end) - 1
    while index >= int(start) and text[index].isspace():
        index -= 1
    if index < int(start):
        index = max(int(start), int(end) - 1)
    return {"char_start": index, "char_end": index + 1}


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


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def build_dataset() -> Dataset:
    """Build 30 dilemmas x 6 conditions x N samples = 540 examples by default."""
    dilemmas = _read_jsonl(DILEMMAS_PATH)
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
            instruction = str(condition["instruction"])
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
                            "is_neutral_negative": role.startswith("neutral_negative"),
                            "is_anti_diagnostic": role == "anti_theory_diagnostic",
                            "is_alt_diagnostic": role == "alternative_theory_diagnostic",
                        },
                        metadata={
                            "instruction": instruction,
                            "dilemma_text": dilemma_text,
                            "answer_constraint": str(conditions_payload.get("answer_constraint") or ""),
                        },
                        cases={"dilemma_id": dilemma_id, "condition_id": condition_id},
                        case_key=dilemma_id,
                    )
                )

    return Dataset.from_examples(
        examples,
        name="morebench_theory_persona_vectors_phase01_deontology_pole_pilot_dataset",
    )


def build_capture_dataset(*, generation: Any) -> dict[str, Any]:
    """Filter generation results into a capture dataset. Drop empty/length-finished rows."""
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
        response_lengths[key] = len(generated_text)
        generated_texts[key] = generated_text
        finish_reason_labels[key] = finish_reason
        condition_role_labels[key] = str(prompt_labels.get("condition_role") or "")

    dataset = Dataset.from_examples(
        examples,
        name="morebench_theory_persona_vectors_phase01_deontology_pole_pilot_capture_dataset",
    )

    return {
        "payload": {
            "kind": "deontology_pole_pilot_capture_dataset",
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
            "generation_finish_reason": finish_reason_labels,
            "condition_role": condition_role_labels,
        },
        "metadata": {
            "source": "GenerationRunSpec result rows",
            "unit": "dilemma_id x condition_id x sample_index",
        },
        "example_keys": sorted(generated_texts),
    }


def summarize_pilot(*, capture_result: Any, capture_dataset: Any) -> TransformResult:
    """Lightweight post-capture summary."""
    capture_payload = capture_dataset.result() if hasattr(capture_dataset, "result") else {}
    summary = _mapping(_mapping(capture_payload.get("payload")).get("summary"))
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
        name="morebench_theory_persona_vectors_phase01_deontology_pole_pilot_capture_dataset",
    )


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


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()

    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="generate_terse_responses",
                runner="capture_gpu",
                description=(
                    "Generate terse recommendation-only responses for the 6-condition deontology pilot. "
                    "Three samples per (dilemma, condition) at temperature 0.7."
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
                description=(
                    "Filter the terse-generation batch into a capture dataset, dropping empty and length-finished rows. "
                    "Records combined-prompt token-section metadata for prompt-end and generated capture sites."
                ),
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_capture_dataset,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_terse_responses")},
                ),
            ),
            WorkflowStep(
                name="capture_residuals",
                runner="capture_gpu",
                description=(
                    "Capture prompt-end and generated-sequence residuals on the filtered terse-response dataset. "
                    "Layers: 0, 4, 16, 24, 32, 40 (pilot subset; primary L32)."
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
                description="Compact post-capture summary of generation+capture status for the pole pilot.",
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
                description="Package pilot generation+capture artifacts for local browsing.",
                spec=ReportSpec(
                    inputs=(
                        StepRef("generate_terse_responses"),
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
