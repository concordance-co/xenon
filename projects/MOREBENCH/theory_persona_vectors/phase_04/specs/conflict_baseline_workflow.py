"""Phase 04 conflict-set baseline generation workflow.

Generates exact-template neutral/deontology/utilitarian/generic-moral baselines
on the prior manually judged action-conflict groups. This is the behavioral
substrate check before any activation steering.
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

from projects.MOREBENCH.phase_03.specs import experiment_02_behavior_broad_workflow as broad
from projects.MOREBENCH.phase_03.specs import experiment_02_workflow as base
from projects.MOREBENCH.theory_persona_vectors.phase_03.specs import (
    all_theories_natural_prompt_workflow as phase03,
)


WORKFLOW_NAME = "morebench_theory_persona_vectors_phase04_conflict_baseline"
PHASE_ROOT = Path("projects/MOREBENCH/theory_persona_vectors/phase_04")
CONFLICT_MANIFEST_PATH = PHASE_ROOT / "outputs" / "conflict_set_manifest.json"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "morebench_theory_persona_vectors_phase04"
LOCAL_CATALOG_ROOT = Path("artifacts") / "morebench_theory_persona_vectors_phase04_catalog"
MODAL_ARTIFACT_ROOT = "/data/artifacts/morebench_theory_persona_vectors_phase04"
REPORT_OUTPUT_DIR = PHASE_ROOT / "reports" / "conflict_baseline_report"

DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
GENERATION_MAX_TOKENS = 256
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
        "condition_id": "N_generic_moral_01",
        "role": "generic_moral_anchor",
        "theory": "generic_moral",
        "instruction": "Answer as someone trying to make a morally careful recommendation.",
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _source_dilemmas() -> dict[str, dict[str, Any]]:
    """Return source dilemma text/metadata keyed by public_conflict_### or theory_group_###."""
    out: dict[str, dict[str, Any]] = {}

    for row in _read_jsonl(broad.PUBLIC_CONFLICT_EXTENSION_PATH):
        group_id = f"public_conflict_{int(row['selection_rank']):03d}"
        out[group_id] = {
            "group_id": group_id,
            "dilemma": str(row["DILEMMA"]),
            "subset": "public_extension_60",
            "context": str(row.get("CONTEXT") or ""),
            "role_domain": str(row.get("ROLE_DOMAIN") or ""),
            "source_family": str(row.get("DILEMMA_SOURCE") or ""),
        }

    seen: set[str] = set()
    for row in _read_jsonl(broad.DIRECT_THEORY_PATH):
        group_id = str(row["group_id"])
        if group_id in seen:
            continue
        seen.add(group_id)
        out[group_id] = {
            "group_id": group_id,
            "dilemma": str(row["base_dilemma"]),
            "subset": "benchmark_theory_30",
            "context": str(row.get("context") or ""),
            "role_domain": str(row.get("role_domain") or ""),
            "source_family": str(row.get("source_family") or ""),
        }

    return out


def _prompt(*, instruction: str, dilemma: str) -> list[dict[str, str]]:
    user = phase03._user_message(instruction=instruction, dilemma=dilemma)
    return [
        {"role": "system", "content": phase03.SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_dataset() -> Dataset:
    manifest = _read_json(CONFLICT_MANIFEST_PATH)
    source_by_group = _source_dilemmas()
    examples: list[Example] = []

    for group in manifest["groups"]:
        group_id = str(group["group_id"])
        source = source_by_group.get(group_id)
        if source is None:
            raise KeyError(f"missing source dilemma for conflict group {group_id}")
        for condition in CONDITIONS:
            condition_id = condition["condition_id"]
            examples.append(
                Example(
                    key=f"{group_id}__{condition_id}",
                    prompt=_prompt(instruction=str(condition["instruction"]), dilemma=str(source["dilemma"])),
                    labels={
                        "group_id": group_id,
                        "subset": source["subset"],
                        "context": source["context"],
                        "role_domain": source["role_domain"],
                        "source_family": source["source_family"],
                        "split_shape": group.get("split_shape"),
                        "action_clusters": group.get("action_clusters") or [],
                        "minority_primes": group.get("minority_primes") or [],
                        "is_tie_3_3": bool(group.get("is_tie_3_3")),
                        "is_primary_steering_candidate": bool(group.get("is_primary_steering_candidate")),
                        "recommended_use": group.get("recommended_use"),
                        "condition_id": condition_id,
                        "condition_role": condition["role"],
                        "condition_theory": condition["theory"],
                        "prompt_regime": "brief_recommendation_conflict_baseline",
                    },
                    metadata={
                        "instruction": condition["instruction"],
                        "dilemma_text": source["dilemma"],
                        "question_suffix": phase03.QUESTION_SUFFIX,
                    },
                    cases={"group_id": group_id, "condition_id": condition_id},
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
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=4,
                memory_mb=12 * 1024,
                timeout_seconds=60 * 30,
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


def summarize_baseline(*, generation: Any) -> TransformResult:
    payload = generation.result() if hasattr(generation, "result") else {}
    rows = payload.get("rows") if isinstance(payload, Mapping) else []
    if not isinstance(rows, list):
        rows = []

    finish_reasons: Counter[str] = Counter()
    by_condition: dict[str, list[int]] = {}
    sample_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        finish_reasons[str(row.get("finish_reason") or "")] += 1
        example = row.get("example") if isinstance(row.get("example"), Mapping) else {}
        labels = example.get("labels") if isinstance(example.get("labels"), Mapping) else {}
        condition = str(labels.get("condition_id") or "")
        token_ids = row.get("generated_token_ids")
        token_count = len(token_ids) if isinstance(token_ids, list) else len(str(row.get("generated_text") or "").split())
        by_condition.setdefault(condition, []).append(token_count)
        if len(sample_rows) < 16:
            sample_rows.append(
                {
                    "group_id": labels.get("group_id"),
                    "condition_id": condition,
                    "tokens": token_count,
                    "finish_reason": row.get("finish_reason"),
                    "preview": str(row.get("generated_text") or "")[:600],
                }
            )

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
            "finish_reason_counts": dict(sorted(finish_reasons.items())),
            "condition_summary": condition_summary,
            "sample_rows": sample_rows,
            "note": "Action disagreement must be judged before steering; this is generation/length/report plumbing only.",
        }
    )


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="generate_conflict_baselines",
                runner="capture_gpu",
                description="Generate neutral/generic/deont/util baselines on all manually judged conflict groups.",
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
                name="summarize_baseline",
                runner="report_local",
                description="Summarize Phase 04 conflict baseline generation.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_baseline,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_conflict_baselines")},
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                description="Package Phase 04 conflict baseline generations for review.",
                spec=ReportSpec(
                    inputs=(StepRef("generate_conflict_baselines"), StepRef("summarize_baseline")),
                    template="default",
                    output_dir=str(REPORT_OUTPUT_DIR),
                ),
            ),
        ),
    )
