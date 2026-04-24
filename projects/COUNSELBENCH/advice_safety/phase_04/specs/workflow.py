from __future__ import annotations

"""CounselBench phase-04 intervention-readiness workflow.

This workflow intentionally stops at pairing and gate materialization. Actual
activation interchange should be added only after Phase 03b and Eval gates pass.
"""

from pathlib import Path

from pipelines_v2.api import (
    ArtifactDatasetSource,
    Dataset,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeStore,
    ReportSpec,
    StepRef,
    TransformBuilder,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)
from projects.COUNSELBENCH.shared.counselbench_dataset import (
    build_eval_aggregated_dataset,
    build_phase4_pairing_candidates,
    build_raw_eval_source_dataset,
    summarize_eval_label_support,
)


ARTIFACTS_VOLUME = "xenon-data"
MODAL_ARTIFACT_ROOT = "/data/artifacts/counselbench_phase04"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "counselbench_phase04"
REPORT_OUTPUT_DIR = "projects/COUNSELBENCH/advice_safety/phase_04/reports/readiness"


def _eval_dataset_ref() -> Dataset:
    return Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef("build_eval_aggregated_dataset"),
        result_key="dataset",
        name="counselbench_eval_aggregated_question_responses",
    )


def build_runner_specs() -> dict[str, object]:
    modal_store = ModalVolumeStore(name=ARTIFACTS_VOLUME, root=MODAL_ARTIFACT_ROOT)
    return {
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(cpu=8, memory_mb=24 * 1024, timeout_seconds=60 * 60),
            artifacts=modal_store,
        ),
        "report_local": LocalRunnerSpec(artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT)),
    }


def build_dataset(*, limit: int | None = None) -> Dataset:
    return build_raw_eval_source_dataset(limit=limit)


def build_workflow(raw_eval_dataset: Dataset | None = None) -> WorkflowSpec:
    raw_eval = raw_eval_dataset or build_dataset()
    eval_dataset = _eval_dataset_ref()
    return WorkflowSpec(
        name="counselbench_phase04_intervention_readiness",
        steps=(
            WorkflowStep(
                name="build_eval_aggregated_dataset",
                runner="analysis_cpu",
                description="Aggregate CounselBench-Eval labels for Phase 4 pairing candidates.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_eval_aggregated_dataset,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    inputs={"raw_eval": raw_eval},
                ),
            ),
            WorkflowStep(
                name="summarize_eval_label_support",
                runner="analysis_cpu",
                description="Recompute response-label support before intervention pairing.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_eval_label_support,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    inputs={"dataset": eval_dataset},
                ),
            ),
            WorkflowStep(
                name="build_phase4_pairing_candidates",
                runner="analysis_cpu",
                description="Build matched boundary pairs plus same-label and random donor controls.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_phase4_pairing_candidates,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    inputs={"dataset": eval_dataset},
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                description="Package Phase 4 readiness gates and pairing candidates without running causal interventions.",
                spec=ReportSpec(
                    inputs=(
                        StepRef("build_eval_aggregated_dataset"),
                        StepRef("summarize_eval_label_support"),
                        StepRef("build_phase4_pairing_candidates"),
                    ),
                    template="default",
                    output_dir=REPORT_OUTPUT_DIR,
                ),
            ),
        ),
    )
