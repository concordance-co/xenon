from __future__ import annotations

"""CounselBench Adv phase-03b controls and localization workflow."""

from pathlib import Path

from pipelines_v2.api import (
    ArtifactDatasetSource,
    CaptureSpec,
    Dataset,
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
    ResidualizedProbeSpec,
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
    adv_prompt_chat_sections,
    build_adv_prompt_dataset,
    build_raw_adv_source_dataset,
    summarize_geometry_metrics,
    triage_adv_03b_controls,
)


MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
ARTIFACTS_VOLUME = "xenon-data"
MODAL_ARTIFACT_ROOT = "/data/artifacts/counselbench_adv_phase03b"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "counselbench_adv_phase03b"
REPORT_OUTPUT_DIR = "projects/COUNSELBENCH/advice_safety/phase_03b/reports/controls"
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
    return build_raw_adv_source_dataset(limit=limit)


def build_workflow(raw_adv_dataset: Dataset | None = None) -> WorkflowSpec:
    raw_adv = raw_adv_dataset or build_dataset()
    adv_dataset = _adv_dataset_ref()
    prompt_capture_ref = StepRef("capture_adv_prompt_only_residual")
    return WorkflowSpec(
        name="counselbench_adv_phase03b_controls_localization",
        steps=(
            WorkflowStep(
                name="build_adv_prompt_dataset",
                runner="analysis_cpu",
                description="Build full 120-example CounselBench-Adv prompt dataset with raw-question chat prompts.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_adv_prompt_dataset,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    inputs={"raw_adv": raw_adv, "limit_per_mode": ADV_LIMIT_PER_MODE},
                ),
            ),
            WorkflowStep(
                name="capture_adv_prompt_only_residual",
                runner="capture_gpu",
                description="Capture prompt-only Adv residuals from the same raw-question chat-template path used for generation.",
                spec=CaptureSpec(
                    engine=_engine(max_num_seqs=4, add_generation_prompt=True),
                    dataset=adv_dataset,
                    generation=GenerationSpec(enabled=False),
                    prompt_metadata_builder=PromptMetadataBuilder.from_function(
                        adv_prompt_chat_sections,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    sites=[
                        ResidualSite(name="residual_prompt_end", site="resid_post", layers=list(CAPTURED_LAYERS), tokens=TokenSelector.section("prompt_end"), storage=TensorStorage(dtype="float16", format="safetensors")),
                        ResidualSite(name="residual_risk_span", site="resid_post", layers=list(CAPTURED_LAYERS), tokens=TokenSelector.section("risk_span"), storage=TensorStorage(dtype="float16", format="safetensors")),
                    ],
                ),
            ),
            WorkflowStep(
                name="text_baseline_failure_mode_grouped",
                runner="analysis_cpu",
                description="Grouped lexical baseline for Adv failure-mode leakage.",
                spec=TextBaselineSpec(
                    text=adv_dataset.labels("prompt_text"),
                    labels=adv_dataset.labels("adv_failure_mode"),
                    group_by=adv_dataset.cases("source_row_id"),
                    model="countvectorizer_logreg",
                    metrics=("accuracy", "balanced_accuracy"),
                ),
            ),
            WorkflowStep(
                name="baseline_failure_mode_from_topic",
                runner="analysis_cpu",
                description="Cheap topic-only baseline for Adv failure-mode leakage under source-row grouping.",
                spec=TextBaselineSpec(
                    text=adv_dataset.labels("topic"),
                    labels=adv_dataset.labels("adv_failure_mode"),
                    group_by=adv_dataset.cases("source_row_id"),
                    model="countvectorizer_logreg",
                    metrics=("accuracy", "balanced_accuracy"),
                ),
            ),
            WorkflowStep(
                name="baseline_failure_mode_from_length",
                runner="analysis_cpu",
                description="Cheap length-bucket-only baseline for Adv failure-mode leakage.",
                spec=TextBaselineSpec(
                    text=adv_dataset.labels("prompt_length_bucket"),
                    labels=adv_dataset.labels("adv_failure_mode"),
                    group_by=adv_dataset.cases("source_row_id"),
                    model="countvectorizer_logreg",
                    metrics=("accuracy", "balanced_accuracy"),
                ),
            ),
            WorkflowStep(
                name="baseline_failure_mode_from_lexical_trigger",
                runner="analysis_cpu",
                description="Cheap lexical-trigger-family baseline for Adv failure-mode leakage.",
                spec=TextBaselineSpec(
                    text=adv_dataset.labels("lexical_trigger_family"),
                    labels=adv_dataset.labels("adv_failure_mode"),
                    group_by=adv_dataset.cases("source_row_id"),
                    model="countvectorizer_logreg",
                    metrics=("accuracy", "balanced_accuracy"),
                ),
            ),
            WorkflowStep(
                name="baseline_failure_mode_from_source_row",
                runner="analysis_cpu",
                description="Source-row identity baseline; should not explain held-out source-row performance.",
                spec=TextBaselineSpec(
                    text=adv_dataset.labels("source_row_id"),
                    labels=adv_dataset.labels("adv_failure_mode"),
                    group_by=adv_dataset.cases("source_row_id"),
                    model="countvectorizer_logreg",
                    metrics=("accuracy", "balanced_accuracy"),
                ),
            ),
            WorkflowStep(
                name="probe_failure_mode_grouped",
                runner="analysis_cpu",
                description="Grouped/LOSO-style prompt-end readout for Adv failure mode.",
                spec=ProbeSpec(
                    feature=prompt_capture_ref.feature("residual_prompt_end"),
                    labels=adv_dataset.labels("adv_failure_mode"),
                    group_by=adv_dataset.cases("source_row_id"),
                    folds=20,
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                    metrics=("accuracy", "balanced_accuracy", "selectivity"),
                    baselines=("majority", "shuffled_label"),
                ),
            ),
            WorkflowStep(
                name="nuisance_probe_topic",
                runner="analysis_cpu",
                description="Measure direct decodability of topic nuisance from the same prompt-end residuals.",
                spec=ProbeSpec(
                    feature=prompt_capture_ref.feature("residual_prompt_end"),
                    labels=adv_dataset.labels("topic"),
                    group_by=adv_dataset.cases("source_row_id"),
                    folds=20,
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                    metrics=("accuracy", "balanced_accuracy"),
                ),
            ),
            WorkflowStep(
                name="nuisance_probe_prompt_length_bucket",
                runner="analysis_cpu",
                description="Measure direct decodability of prompt length bucket from prompt-end residuals.",
                spec=ProbeSpec(
                    feature=prompt_capture_ref.feature("residual_prompt_end"),
                    labels=adv_dataset.labels("prompt_length_bucket"),
                    group_by=adv_dataset.cases("source_row_id"),
                    folds=20,
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                    metrics=("accuracy", "balanced_accuracy"),
                ),
            ),
            WorkflowStep(
                name="nuisance_probe_lexical_trigger_family",
                runner="analysis_cpu",
                description="Measure direct decodability of lexical trigger family from prompt-end residuals.",
                spec=ProbeSpec(
                    feature=prompt_capture_ref.feature("residual_prompt_end"),
                    labels=adv_dataset.labels("lexical_trigger_family"),
                    group_by=adv_dataset.cases("source_row_id"),
                    folds=20,
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                    metrics=("accuracy", "balanced_accuracy"),
                ),
            ),
            WorkflowStep(
                name="residualized_failure_mode_topic",
                runner="analysis_cpu",
                description="Probe Adv failure mode after projecting out a learned topic nuisance subspace.",
                spec=ResidualizedProbeSpec(
                    feature=prompt_capture_ref.feature("residual_prompt_end"),
                    labels=adv_dataset.labels("adv_failure_mode"),
                    residualize_against=adv_dataset.labels("topic"),
                    group_by=adv_dataset.cases("source_row_id"),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                    metrics=("accuracy", "balanced_accuracy"),
                ),
            ),
            WorkflowStep(
                name="residualized_failure_mode_lexical_trigger",
                runner="analysis_cpu",
                description="Probe Adv failure mode after projecting out lexical-trigger-family nuisance.",
                spec=ResidualizedProbeSpec(
                    feature=prompt_capture_ref.feature("residual_prompt_end"),
                    labels=adv_dataset.labels("adv_failure_mode"),
                    residualize_against=adv_dataset.labels("lexical_trigger_family"),
                    group_by=adv_dataset.cases("source_row_id"),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                    metrics=("accuracy", "balanced_accuracy"),
                ),
            ),
            WorkflowStep(
                name="geometry_prompt_failure_mode_pca",
                runner="analysis_cpu",
                description="PCA projection for failure-mode, topic, length, and lexical-trigger controls.",
                spec=GeometrySpec(
                    feature=prompt_capture_ref.feature("residual_prompt_end"),
                    method="pca",
                    layers=GEOMETRY_LAYERS,
                    label=adv_dataset.labels("adv_failure_mode"),
                    color_by={
                        "failure_mode": adv_dataset.labels("adv_failure_mode"),
                        "topic": adv_dataset.labels("topic"),
                        "prompt_length_bucket": adv_dataset.labels("prompt_length_bucket"),
                        "lexical_trigger_family": adv_dataset.labels("lexical_trigger_family"),
                    },
                    normalize="rms_per_row",
                    components=3,
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                ),
            ),
            WorkflowStep(
                name="summarize_geometry_prompt_failure_mode",
                runner="analysis_cpu",
                description="Quantify PCA geometry separation for labels and confounds.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_geometry_metrics,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    inputs={"geometry": StepRef("geometry_prompt_failure_mode_pca")},
                ),
            ),
            WorkflowStep(
                name="triage_adv_03b_controls",
                runner="analysis_cpu",
                description="Apply Adv 03b promotion gate against cheap and nuisance baselines.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        triage_adv_03b_controls,
                        local_python_sources=("projects/COUNSELBENCH",),
                    ),
                    inputs={
                        "probe": StepRef("probe_failure_mode_grouped"),
                        "text_baseline": StepRef("text_baseline_failure_mode_grouped"),
                        "topic_baseline": StepRef("baseline_failure_mode_from_topic"),
                        "length_baseline": StepRef("baseline_failure_mode_from_length"),
                        "lexical_baseline": StepRef("baseline_failure_mode_from_lexical_trigger"),
                        "source_row_baseline": StepRef("baseline_failure_mode_from_source_row"),
                        "residualized_topic": StepRef("residualized_failure_mode_topic"),
                        "residualized_lexical": StepRef("residualized_failure_mode_lexical_trigger"),
                        "geometry_metrics": StepRef("summarize_geometry_prompt_failure_mode"),
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                description="Package Adv 03b controls, nuisance probes, geometry metrics, and triage.",
                spec=ReportSpec(
                    inputs=(
                        StepRef("build_adv_prompt_dataset"),
                        StepRef("text_baseline_failure_mode_grouped"),
                        StepRef("baseline_failure_mode_from_topic"),
                        StepRef("baseline_failure_mode_from_length"),
                        StepRef("baseline_failure_mode_from_lexical_trigger"),
                        StepRef("baseline_failure_mode_from_source_row"),
                        StepRef("probe_failure_mode_grouped"),
                        StepRef("nuisance_probe_topic"),
                        StepRef("nuisance_probe_prompt_length_bucket"),
                        StepRef("nuisance_probe_lexical_trigger_family"),
                        StepRef("residualized_failure_mode_topic"),
                        StepRef("residualized_failure_mode_lexical_trigger"),
                        StepRef("geometry_prompt_failure_mode_pca"),
                        StepRef("summarize_geometry_prompt_failure_mode"),
                        StepRef("triage_adv_03b_controls"),
                    ),
                    template="default",
                    output_dir=REPORT_OUTPUT_DIR,
                ),
            ),
        ),
    )
