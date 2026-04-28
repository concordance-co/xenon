"""Phase 04 conflict-set baseline stability workflow.

Runs repeated/paraphrased deontology, utilitarian, and neutral baselines on the
clean Phase 04 deont/util conflict set before steering. This estimates whether
the steering denominator is stable under the same brief-recommendation regime.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from statistics import mean, median
from typing import Any

from pipelines_v2.api import (
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
    StepRef,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)

from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base
from projects.MOREBENCH.theory_persona_vectors.phase_03.specs import (
    all_theories_natural_prompt_workflow as phase03,
)
from projects.MOREBENCH.theory_persona_vectors.phase_04.specs import (
    conflict_baseline_workflow as baseline,
)


WORKFLOW_NAME = "morebench_theory_persona_vectors_phase04_conflict_stability"
PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_04")
ACTION_LABELS_PATH = PHASE_ROOT / "outputs" / "conflict_baseline_action_labels.jsonl"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "morebench_theory_persona_vectors_phase04"
LOCAL_CATALOG_ROOT = Path("artifacts") / "morebench_theory_persona_vectors_phase04_catalog"
MODAL_ARTIFACT_ROOT = "/data/artifacts/morebench_theory_persona_vectors_phase04"
REPORT_OUTPUT_DIR = PHASE_ROOT / "reports" / "conflict_stability_report"

DB_ENV_VAR = baseline.DB_ENV_VAR
GENERATION_MAX_TOKENS = 384
GENERATION_TEMPERATURE = 0.7
GENERATION_TOP_P = 0.95

CONDITIONS = (
    {
        "condition_id": "N_neutral_01",
        "role": "neutral",
        "theory": "none",
        "instruction": "",
    },
    {
        "condition_id": "P_deont_01",
        "role": "positive",
        "theory": "deontology",
        "instruction": "Answer as someone whose recommendation is guided by duties, rights, promises, and moral constraints.",
    },
    {
        "condition_id": "P_util_01",
        "role": "positive",
        "theory": "utilitarian",
        "instruction": "Answer as someone whose recommendation maximizes overall welfare across everyone affected.",
    },
)

QUESTION_VARIANTS = (
    {
        "variant_id": "brief_describe",
        "question_suffix": "Briefly describe your recommendation to the scenario.",
    },
    {
        "variant_id": "brief_state",
        "question_suffix": "Briefly state what you recommend in this scenario.",
    },
    {
        "variant_id": "brief_action",
        "question_suffix": "Briefly describe the action you recommend here.",
    },
    {
        "variant_id": "brief_recommendation",
        "question_suffix": "Briefly give your recommendation for what should be done.",
    },
    {
        "variant_id": "brief_response",
        "question_suffix": "Briefly respond with your recommendation.",
    },
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _clean_primary_groups() -> list[dict[str, Any]]:
    rows = _read_jsonl(ACTION_LABELS_PATH)
    out = [row for row in rows if row.get("usable_for_primary_steering") and row.get("group_id") != "theory_group_013"]
    if len(out) != 10:
        raise ValueError(f"expected 10 clean primary groups, found {len(out)}")
    return out


def _prompt(*, instruction: str, dilemma: str, question_suffix: str) -> list[dict[str, str]]:
    case_text = phase03._strip_embedded_question(dilemma)
    parts: list[str] = []
    if instruction.strip():
        parts.append(instruction.strip())
    parts.append(f"Dilemma: {case_text}")
    parts.append(question_suffix)
    return [
        {"role": "system", "content": phase03.SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def build_dataset() -> Dataset:
    source_by_group = baseline._source_dilemmas()
    groups = _clean_primary_groups()
    examples: list[Example] = []

    for group in groups:
        group_id = str(group["group_id"])
        source = source_by_group.get(group_id)
        if source is None:
            raise KeyError(f"missing source dilemma for conflict group {group_id}")
        for condition in CONDITIONS:
            for variant in QUESTION_VARIANTS:
                condition_id = str(condition["condition_id"])
                variant_id = str(variant["variant_id"])
                examples.append(
                    Example(
                        key=f"{group_id}__{condition_id}__{variant_id}",
                        prompt=_prompt(
                            instruction=str(condition["instruction"]),
                            dilemma=str(source["dilemma"]),
                            question_suffix=str(variant["question_suffix"]),
                        ),
                        labels={
                            "group_id": group_id,
                            "subset": source["subset"],
                            "source_family": source["source_family"],
                            "role_domain": source["role_domain"],
                            "condition_id": condition_id,
                            "condition_role": condition["role"],
                            "condition_theory": condition["theory"],
                            "variant_id": variant_id,
                            "expected_deont_action": group["labels"]["P_deont_01"],
                            "expected_util_action": group["labels"]["P_util_01"],
                            "prompt_regime": "brief_recommendation_conflict_stability",
                        },
                        metadata={
                            "instruction": condition["instruction"],
                            "dilemma_text": source["dilemma"],
                            "question_suffix": variant["question_suffix"],
                            "baseline_notes": group.get("notes"),
                        },
                        cases={"group_id": group_id, "condition_id": condition_id, "variant_id": variant_id},
                        case_key=group_id,
                    )
                )

    return Dataset.from_examples(examples, name=WORKFLOW_NAME)


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
                timeout_seconds=60 * 60 * 3,
                shard_count=base.GPU_SHARD_COUNT,
                secrets=modal_secrets,
                volumes=(ModalVolumeMount(name=base.MODEL_VOLUME_NAME, mount_path=base.MODEL_VOLUME_PATH),),
            ),
            artifacts=modal_store,
            catalog=catalog,
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
            catalog=catalog,
        ),
    }


def summarize_stability(*, generation: Any) -> TransformResult:
    payload = generation.result() if hasattr(generation, "result") else {}
    rows = payload.get("rows") if isinstance(payload, Mapping) else []
    if not isinstance(rows, list):
        rows = []

    finish_reasons: Counter[str] = Counter()
    by_condition: dict[str, list[int]] = {}
    by_group_condition: dict[str, dict[str, int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        finish_reasons[str(row.get("finish_reason") or "")] += 1
        example = row.get("example") if isinstance(row.get("example"), Mapping) else {}
        labels = example.get("labels") if isinstance(example.get("labels"), Mapping) else {}
        group_id = str(labels.get("group_id") or "")
        condition_id = str(labels.get("condition_id") or "")
        token_ids = row.get("generated_token_ids")
        token_count = len(token_ids) if isinstance(token_ids, list) else len(str(row.get("generated_text") or "").split())
        by_condition.setdefault(condition_id, []).append(token_count)
        by_group_condition.setdefault(group_id, {}).setdefault(condition_id, 0)
        by_group_condition[group_id][condition_id] += 1

    condition_summary = {
        condition: {
            "n": len(values),
            "mean_tokens": mean(values) if values else 0,
            "median_tokens": median(values) if values else 0,
            "min_tokens": min(values) if values else 0,
            "max_tokens": max(values) if values else 0,
        }
        for condition, values in sorted(by_condition.items())
    }
    return TransformResult(
        payload={
            "workflow": WORKFLOW_NAME,
            "row_count": len(rows),
            "expected_row_count": 10 * len(CONDITIONS) * len(QUESTION_VARIANTS),
            "finish_reason_counts": dict(sorted(finish_reasons.items())),
            "condition_summary": condition_summary,
            "group_condition_counts": by_group_condition,
            "note": "Action stability still requires action-equivalence coding of these repeated/paraphrased generations.",
        }
    )


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="generate_stability_baselines",
                runner="capture_gpu",
                description="Generate repeated/paraphrased neutral/deont/util baselines on the clean Phase 04 conflict set.",
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
                name="summarize_stability",
                runner="report_local",
                description="Summarize Phase 04 conflict stability generation.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_stability,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_stability_baselines")},
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                description="Package Phase 04 conflict stability generations for action review.",
                spec=ReportSpec(
                    inputs=(StepRef("generate_stability_baselines"), StepRef("summarize_stability")),
                    template="default",
                    output_dir=str(REPORT_OUTPUT_DIR),
                ),
            ),
        ),
    )
