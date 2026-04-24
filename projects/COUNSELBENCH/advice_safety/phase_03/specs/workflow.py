from __future__ import annotations

"""CounselBench Adv phase-03 readout workflow."""

from pathlib import Path

from pipelines_v2.api import (
    ArtifactDatasetSource,
    CaptureSpec,
    Dataset,
    GenerationRunSpec,
    GenerationSpec,
    GeometrySpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    ProbeSpec,
    PromptMetadataBuilder,
    ReportSpec,
    ResidualSite,
    StepRef,
    TensorStorage,
    TextBaselineSpec,
    TokenPooling,
    TokenSelector,
    TransformBuilder,
    TransformSpec,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)
from projects.COUNSELBENCH.shared.counselbench_dataset import (
    adv_generated_chat_prompt_sections,
    build_adv_prompt_dataset,
    build_raw_adv_source_dataset,
    build_successful_generation_capture_dataset,
    evaluate_generation_quality_gate,
    summarize_generated_label_support,
)


MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
ARTIFACTS_VOLUME = "xenon-data"
MODAL_ARTIFACT_ROOT = "/data/artifacts/counselbench_adv_phase03"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "counselbench_adv_phase03"
REPORT_OUTPUT_DIR = "projects/COUNSELBENCH/advice_safety/phase_03/reports/full_adv"
CAPTURED_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)
GEOMETRY_LAYERS = (8, 16, 24, 32, 40, 44)
ADV_LIMIT_PER_MODE: int | None = None
MODEL_BOUND_SHARD_COUNT = 4


def _engine(*, max_num_seqs: int = 16, add_generation_prompt: bool = True) -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=30000,
        enforce_eager=False,
        enable_prefix_caching=True,
        max_num_seqs=max_num_seqs,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def _adv_dataset_ref() -> Dataset:
    return Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef("build_adv_prompt_dataset"),
        result_key="dataset",
        name="counselbench_adv_balanced_prompts",
    )


def _generated_capture_dataset_ref() -> Dataset:
    return Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef("build_successful_generation_capture_dataset"),
        result_key="dataset",
        name="counselbench_adv_successful_prompt_generated_contexts",
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
            resources=ModalResources(
                cpu=8,
                memory_mb=24 * 1024,
                timeout_seconds=60 * 60 * 2,
            ),
            artifacts=modal_store,
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
        ),
    }


def build_dataset(*, limit: int | None = None) -> Dataset:
    return build_raw_adv_source_dataset(limit=limit)


def build_workflow(raw_adv_dataset: Dataset | None = None) -> WorkflowSpec:
    raw_adv = raw_adv_dataset or build_dataset()
    adv_dataset = _adv_dataset_ref()
    generated_dataset = _generated_capture_dataset_ref()
    return WorkflowSpec(
        name="counselbench_adv_phase03_full_adv_readouts",
        steps=(
            WorkflowStep(
                name="build_adv_prompt_dataset",
                runner="analysis_cpu",
                description=(
                    "Materialize CounselBench-Adv from Hugging Face at runtime and melt the 20-row wide table "
                    "into the full 120-example one-prompt-per-failure-mode dataset."
                ),
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_adv_prompt_dataset,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    inputs={"raw_adv": raw_adv, "limit_per_mode": ADV_LIMIT_PER_MODE},
                ),
            ),
            WorkflowStep(
                name="generate_adv_responses",
                runner="capture_gpu",
                description=(
                    "Generate deterministic target-model responses for the full CounselBench-Adv phase-03 set."
                ),
                spec=GenerationRunSpec(
                    engine=_engine(),
                    dataset=adv_dataset,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=15000,
                        temperature=0.0,
                        top_p=1.0,
                        top_k=-1,
                        capture_reasoning=False,
                    ),
                ),
            ),
            WorkflowStep(
                name="evaluate_generation_quality_gate",
                runner="analysis_cpu",
                description=(
                    "Compute automated generation tripwires and package samples for required manual/agent inspection."
                ),
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        evaluate_generation_quality_gate,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    inputs={"generation_artifact": StepRef("generate_adv_responses")},
                ),
            ),
            WorkflowStep(
                name="build_successful_generation_capture_dataset",
                runner="analysis_cpu",
                description=(
                    "Filter empty and length-finished generations, then build user/assistant chat contexts "
                    "for prompt-end and generation-end capture."
                ),
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_successful_generation_capture_dataset,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_adv_responses")},
                ),
            ),
            WorkflowStep(
                name="summarize_generated_label_support",
                runner="analysis_cpu",
                description=(
                    "Check whether provisional generated-response labels have enough class support for "
                    "trainable response-side baselines/probes."
                ),
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_generated_label_support,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    inputs={"dataset": generated_dataset},
                ),
            ),
            WorkflowStep(
                name="capture_prompt_generated_residual",
                runner="capture_gpu",
                description=(
                    "Replay successful prompt+generated contexts and capture residual states at the original "
                    "prompt end and generated-answer end."
                ),
                spec=CaptureSpec(
                    engine=_engine(max_num_seqs=4, add_generation_prompt=False),
                    dataset=generated_dataset,
                    generation=GenerationSpec(enabled=False),
                    prompt_metadata_builder=PromptMetadataBuilder.from_function(
                        adv_generated_chat_prompt_sections,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    sites=[
                        ResidualSite(
                            name="residual_prompt_end",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.section("prompt_end"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                        ResidualSite(
                            name="residual_generation_end",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.section("generated_end"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ],
                ),
            ),
            WorkflowStep(
                name="text_baseline_prompt_failure_mode",
                runner="analysis_cpu",
                description=(
                    "Cheap lexical baseline for E1. If this is strong, prompt-side failure-mode readouts stay diagnostic."
                ),
                spec=TextBaselineSpec(
                    text=generated_dataset.labels("prompt_text"),
                    labels=generated_dataset.labels("adv_failure_mode"),
                    group_by=generated_dataset.cases("source_row_id"),
                    split_by={"split": generated_dataset.labels("split")},
                    train_values=("train",),
                    test_values=("test",),
                    model="countvectorizer_logreg",
                    metrics=("accuracy", "balanced_accuracy"),
                ),
            ),
            WorkflowStep(
                name="geometry_prompt_failure_mode_pca",
                runner="analysis_cpu",
                description=(
                    "PCA view of prompt-end residual states, labeled by Adv failure mode. This tests whether "
                    "the six expert-authored adversarial prompt families occupy a structured activation space."
                ),
                spec=GeometrySpec(
                    feature=StepRef("capture_prompt_generated_residual").feature("residual_prompt_end"),
                    method="pca",
                    layers=GEOMETRY_LAYERS,
                    label=generated_dataset.labels("adv_failure_mode"),
                    color_by={
                        "failure_mode": generated_dataset.labels("adv_failure_mode"),
                        "topic": generated_dataset.labels("topic"),
                        "prompt_length_bucket": generated_dataset.labels("prompt_length_bucket"),
                    },
                    normalize="rms_per_row",
                    components=3,
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                ),
            ),
            WorkflowStep(
                name="probe_prompt_failure_mode",
                runner="analysis_cpu",
                description=(
                    "E1 diagnostic prompt-end residual probe for the CounselBench-Adv failure-mode family."
                ),
                spec=ProbeSpec(
                    feature=StepRef("capture_prompt_generated_residual").feature("residual_prompt_end"),
                    labels=generated_dataset.labels("adv_failure_mode"),
                    group_by=generated_dataset.cases("source_row_id"),
                    split=generated_dataset.labels("split"),
                    train_values=("train",),
                    test_values=("test",),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                    metrics=("accuracy", "balanced_accuracy", "selectivity"),
                    baselines=("majority", "shuffled_label"),
                ),
            ),
            WorkflowStep(
                name="geometry_generated_posture_pca",
                runner="analysis_cpu",
                description=(
                    "PCA view of generation-end residual states, colored by Adv family and provisional "
                    "medical-boundary label. This is the first geometry check for response-posture structure."
                ),
                spec=GeometrySpec(
                    feature=StepRef("capture_prompt_generated_residual").feature("residual_generation_end"),
                    method="pca",
                    layers=GEOMETRY_LAYERS,
                    label=generated_dataset.labels("medical_boundary_violation"),
                    color_by={
                        "failure_mode": generated_dataset.labels("adv_failure_mode"),
                        "medical_boundary_violation": generated_dataset.labels("medical_boundary_violation"),
                        "topic": generated_dataset.labels("topic"),
                        "response_length_bucket": generated_dataset.labels("response_length_bucket"),
                    },
                    normalize="rms_per_row",
                    components=3,
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                description=(
                    "Package the generation quality gate, label-support gate, text baselines, probe metrics, geometry, "
                    "and claim-strength caveats."
                ),
                spec=ReportSpec(
                    inputs=(
                        StepRef("build_adv_prompt_dataset"),
                        StepRef("evaluate_generation_quality_gate"),
                        StepRef("build_successful_generation_capture_dataset"),
                        StepRef("summarize_generated_label_support"),
                        StepRef("geometry_prompt_failure_mode_pca"),
                        StepRef("text_baseline_prompt_failure_mode"),
                        StepRef("probe_prompt_failure_mode"),
                        StepRef("geometry_generated_posture_pca"),
                    ),
                    template="default",
                    output_dir=REPORT_OUTPUT_DIR,
                ),
            ),
        ),
    )
