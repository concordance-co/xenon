from __future__ import annotations

"""pipelines_v2 workflow for MoReBench phase 01 rubric intake and leakage baselines."""

import os
from pathlib import Path

from pipelines_v2.api import (
    Dataset,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeStore,
    PostgresCatalog,
    PostgresSource,
    ReportSpec,
    StepRef,
    TextBaselineSpec,
    TransformBuilder,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)
from projects.MOREBENCH.shared.morebench_dataset import (
    DEFAULT_SPLIT,
    PUBLIC_CONFIG,
    build_rubric_criterion_dataset,
)
from projects.MOREBENCH.shared.rubric_validation import build_rubric_profile_labels


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
DEFAULT_REPORT_DIR = "projects/MOREBENCH/procedural_probe/phase_01/reports/pipelines_v2"


def _dataset_limit_from_env() -> int | None:
    raw = os.environ.get("MOREBENCH_DATASET_LIMIT")
    if raw is None or not raw.strip():
        return None
    limit = int(raw)
    if limit <= 0:
        raise ValueError("MOREBENCH_DATASET_LIMIT must be a positive integer")
    return limit


def build_criterion_dataset(*, limit: int | None = None) -> Dataset:
    return build_rubric_criterion_dataset(
        config=os.environ.get("MOREBENCH_PUBLIC_CONFIG", PUBLIC_CONFIG),
        split=os.environ.get("MOREBENCH_SPLIT", DEFAULT_SPLIT),
        limit=limit if limit is not None else _dataset_limit_from_env(),
        name="morebench_public_rubric_criteria",
    )


def build_dataset(*, limit: int | None = None) -> Dataset:
    return build_criterion_dataset(limit=limit)


def build_runner_specs() -> dict[str, object]:
    db = PostgresSource.from_env(DB_ENV_VAR)
    artifact_store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/morebench_phase_01",
    )
    workflow_catalog = PostgresCatalog(source=db)
    db_secret = ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon")
    return {
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=4,
                memory_mb=16 * 1024,
                secrets=(db_secret,),
            ),
            artifacts=artifact_store,
            catalog=workflow_catalog,
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(Path("artifacts") / "morebench_phase_01"),
            catalog=workflow_catalog,
        ),
    }


def build_workflow(
    criterion_dataset: Dataset | None = None,
    *,
    report_output_dir: str = DEFAULT_REPORT_DIR,
) -> WorkflowSpec:
    criterion_dataset = criterion_dataset or build_criterion_dataset()

    return WorkflowSpec(
        name="morebench_phase_01_rubric_intake",
        steps=(
            WorkflowStep(
                name="build_rubric_profiles",
                runner="analysis_cpu",
                description=(
                    "Collapse all rubric annotations for each dilemma into one validation profile. The profile captures "
                    "the shape of the expert rubric set: dominant weighted dimension, harmlessness penalties, "
                    "helpful-vs-harmless tension, complexity, and high-weight focus. This is the object Phase 1 validates."
                ),
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_rubric_profile_labels,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"criteria": criterion_dataset},
                ),
            ),
            WorkflowStep(
                name="dilemma_text_to_dominant_dimension_baseline",
                runner="analysis_cpu",
                description=(
                    "Validation baseline: predict the dominant weighted rubric dimension for the whole dilemma from "
                    "dilemma text alone. High balanced accuracy/AUROC means the scenario topic strongly determines the "
                    "rubric profile; low values mean the profile is not a simple topic label and later probes should use "
                    "the full rubric context."
                ),
                spec=TextBaselineSpec(
                    text=StepRef("build_rubric_profiles").label("dilemma_text"),
                    labels=StepRef("build_rubric_profiles").label("dominant_dimension"),
                    group_by=StepRef("build_rubric_profiles").label("base_dilemma_id"),
                    model="countvectorizer_logreg",
                    metrics=("accuracy", "balanced_accuracy", "auroc"),
                ),
            ),
            WorkflowStep(
                name="dilemma_text_to_harmless_penalty_baseline",
                runner="analysis_cpu",
                description=(
                    "Validation baseline: predict whether the rubric set contains any negative harmless-outcome penalty "
                    "from dilemma text alone. High balanced accuracy/AUROC means safety penalty requirements are topic "
                    "predictable; low values mean they are encoded in the criterion set and need explicit rubric context."
                ),
                spec=TextBaselineSpec(
                    text=StepRef("build_rubric_profiles").label("dilemma_text"),
                    labels=StepRef("build_rubric_profiles").label("has_harmless_penalty"),
                    group_by=StepRef("build_rubric_profiles").label("base_dilemma_id"),
                    model="countvectorizer_logreg",
                    metrics=("accuracy", "balanced_accuracy", "auroc"),
                ),
            ),
            WorkflowStep(
                name="dilemma_text_to_helpful_harmless_tension_baseline",
                runner="analysis_cpu",
                description=(
                    "Validation baseline: predict whether the rubric profile has both positive helpful-outcome mass and "
                    "negative harmless-outcome penalties. High balanced accuracy/AUROC means helpful-vs-harmless tension "
                    "is mostly visible in dilemma text; low values mean the tension must be recovered from the full rubric."
                ),
                spec=TextBaselineSpec(
                    text=StepRef("build_rubric_profiles").label("dilemma_text"),
                    labels=StepRef("build_rubric_profiles").label("has_helpful_harmless_tension"),
                    group_by=StepRef("build_rubric_profiles").label("base_dilemma_id"),
                    model="countvectorizer_logreg",
                    metrics=("accuracy", "balanced_accuracy", "auroc"),
                ),
            ),
            WorkflowStep(
                name="dilemma_text_to_rubric_complexity_baseline",
                runner="analysis_cpu",
                description=(
                    "Validation baseline: predict a low/medium/high bin for rubric complexity from dilemma text alone. "
                    "High metrics mean the number of evaluation conditions is topic predictable; low metrics mean rubric "
                    "complexity should be treated as additional expert structure rather than query surface form."
                ),
                spec=TextBaselineSpec(
                    text=StepRef("build_rubric_profiles").label("dilemma_text"),
                    labels=StepRef("build_rubric_profiles").label("rubric_complexity_bin"),
                    group_by=StepRef("build_rubric_profiles").label("base_dilemma_id"),
                    model="countvectorizer_logreg",
                    metrics=("accuracy", "balanced_accuracy", "auroc"),
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                description=(
                    "Assemble the rubric-profile validation report. Interpret high text-baseline metrics as evidence "
                    "that a profile attribute is predictable from query topic alone; low metrics mean the attribute "
                    "depends on the multi-criterion expert rubric set and should be passed explicitly into later captures."
                ),
                spec=ReportSpec(
                    inputs=(
                        StepRef("build_rubric_profiles"),
                        StepRef("dilemma_text_to_dominant_dimension_baseline"),
                        StepRef("dilemma_text_to_harmless_penalty_baseline"),
                        StepRef("dilemma_text_to_helpful_harmless_tension_baseline"),
                        StepRef("dilemma_text_to_rubric_complexity_baseline"),
                    ),
                    template="default",
                    output_dir=report_output_dir,
                ),
            ),
        ),
    )
