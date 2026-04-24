from __future__ import annotations

"""CounselBench-Eval expert-label response-context readout workflow."""

from pathlib import Path

from pipelines_v2.api import (
    ArtifactDatasetSource,
    CaptureSpec,
    Dataset,
    GeometrySpec,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    PromptMetadataBuilder,
    ReportSpec,
    ResidualSite,
    StepRef,
    TensorStorage,
    TokenPooling,
    TokenSelector,
    TransformBuilder,
    TransformSpec,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)
from projects.COUNSELBENCH.shared.counselbench_dataset import (
    build_eval_aggregated_dataset,
    build_raw_eval_source_dataset,
    eval_chat_prompt_sections,
    run_eval_responder_transfer_readouts,
    run_eval_gated_readouts,
    summarize_eval_cheap_baselines,
    summarize_eval_confound_inventory,
    summarize_eval_label_support,
    summarize_geometry_metrics,
)


MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
ARTIFACTS_VOLUME = "xenon-data"
MODAL_ARTIFACT_ROOT = "/data/artifacts/counselbench_eval_phase03"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "counselbench_eval_phase03"
REPORT_OUTPUT_DIR = "projects/COUNSELBENCH/advice_safety/phase_03_eval/reports/expert_labels"
CAPTURED_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)
GEOMETRY_LAYERS = (8, 16, 24, 32, 40, 44)
MODEL_BOUND_SHARD_COUNT = 4


def _engine(*, max_num_seqs: int = 4) -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=30000,
        enforce_eager=False,
        enable_prefix_caching=True,
        max_num_seqs=max_num_seqs,
        add_generation_prompt=False,
        enable_thinking=False,
    )


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
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 4,
                max_containers=MODEL_BOUND_SHARD_COUNT,
                shard_count=MODEL_BOUND_SHARD_COUNT,
                volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
            ),
            artifacts=modal_store,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(cpu=8, memory_mb=24 * 1024, timeout_seconds=60 * 60 * 2),
            artifacts=modal_store,
        ),
        "report_local": LocalRunnerSpec(artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT)),
    }


def build_dataset(*, limit: int | None = None) -> Dataset:
    return build_raw_eval_source_dataset(limit=limit)


def build_workflow(raw_eval_dataset: Dataset | None = None) -> WorkflowSpec:
    raw_eval = raw_eval_dataset or build_dataset()
    eval_dataset = _eval_dataset_ref()
    capture_ref = StepRef("capture_eval_response_context_residual")
    return WorkflowSpec(
        name="counselbench_eval_phase03_expert_label_readouts",
        steps=(
            WorkflowStep(
                name="build_eval_aggregated_dataset",
                runner="analysis_cpu",
                description="Aggregate repeated CounselBench-Eval expert ratings by question-response identity.",
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
                description="Check train/test class support for frozen expert-label response targets.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_eval_label_support,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    inputs={"dataset": eval_dataset},
                ),
            ),
            WorkflowStep(
                name="summarize_eval_cheap_baselines",
                runner="analysis_cpu",
                description="Compute topic, responder, length, and lexical cheap baselines for Eval labels.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_eval_cheap_baselines,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    inputs={"dataset": eval_dataset},
                ),
            ),
            WorkflowStep(
                name="summarize_eval_confound_inventory",
                runner="analysis_cpu",
                description="Quantify Eval label imbalance by responder and within-question contrast support.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_eval_confound_inventory,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    inputs={"dataset": eval_dataset},
                ),
            ),
            WorkflowStep(
                name="capture_eval_response_context_residual",
                runner="capture_gpu",
                description="Capture response-end residuals over Eval question-response chat contexts.",
                spec=CaptureSpec(
                    engine=_engine(),
                    dataset=eval_dataset,
                    generation=GenerationSpec(enabled=False),
                    prompt_metadata_builder=PromptMetadataBuilder.from_function(
                        eval_chat_prompt_sections,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    sites=[
                        ResidualSite(name="residual_response_end", site="resid_post", layers=list(CAPTURED_LAYERS), tokens=TokenSelector.section("response_end"), storage=TensorStorage(dtype="float16", format="safetensors")),
                    ],
                ),
            ),
            WorkflowStep(
                name="geometry_eval_quality_pca",
                runner="analysis_cpu",
                description="PCA geometry for Eval response-context labels and confounds.",
                spec=GeometrySpec(
                    feature=capture_ref.feature("residual_response_end"),
                    method="pca",
                    layers=GEOMETRY_LAYERS,
                    label=eval_dataset.labels("medical_boundary_violation"),
                    color_by={
                        "medical_boundary_violation": eval_dataset.labels("medical_boundary_violation"),
                        "empathy_high": eval_dataset.labels("empathy_high"),
                        "specificity_high": eval_dataset.labels("specificity_high"),
                        "toxicity_or_judgmental": eval_dataset.labels("toxicity_or_judgmental"),
                        "overall_quality_high": eval_dataset.labels("overall_quality_high"),
                        "topic": eval_dataset.labels("topic"),
                        "responder": eval_dataset.cases("responder"),
                    },
                    normalize="rms_per_row",
                    components=3,
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                ),
            ),
            WorkflowStep(
                name="summarize_geometry_eval_quality",
                runner="analysis_cpu",
                description="Quantify Eval response-context geometry for expert labels and confounds.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_geometry_metrics,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    inputs={"geometry": StepRef("geometry_eval_quality_pca")},
                ),
            ),
            WorkflowStep(
                name="run_eval_gated_readouts",
                runner="analysis_cpu",
                description="Run per-label Eval readouts only when frozen label-support gates pass.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        run_eval_gated_readouts,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    inputs={
                        "dataset": eval_dataset,
                        "capture": StepRef("capture_eval_response_context_residual"),
                        "cheap_baselines": StepRef("summarize_eval_cheap_baselines"),
                    },
                ),
            ),
            WorkflowStep(
                name="run_eval_responder_transfer_readouts",
                runner="analysis_cpu",
                description="Test whether Eval quality readouts transfer across responder families.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        run_eval_responder_transfer_readouts,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    inputs={
                        "dataset": eval_dataset,
                        "capture": StepRef("capture_eval_response_context_residual"),
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                description="Package Eval expert-label support, readouts, controls, and geometry.",
                spec=ReportSpec(
                    inputs=(
                        StepRef("build_eval_aggregated_dataset"),
                        StepRef("summarize_eval_label_support"),
                        StepRef("summarize_eval_cheap_baselines"),
                        StepRef("summarize_eval_confound_inventory"),
                        StepRef("geometry_eval_quality_pca"),
                        StepRef("summarize_geometry_eval_quality"),
                        StepRef("run_eval_gated_readouts"),
                        StepRef("run_eval_responder_transfer_readouts"),
                    ),
                    template="default",
                    output_dir=REPORT_OUTPUT_DIR,
                ),
            ),
        ),
    )
