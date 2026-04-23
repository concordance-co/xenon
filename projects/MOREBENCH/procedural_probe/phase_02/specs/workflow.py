from __future__ import annotations

"""pipelines_v2 workflow for MoReBench phase 02 raw dimension probes."""

from pathlib import Path

from pipelines_v2.api import (
    ArtifactDatasetSource,
    CaptureSpec,
    Dataset,
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
    ProbeSpec,
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
from projects.MOREBENCH.shared.dimension_probes import (
    RAW_RUBRIC_DIMENSION_TARGETS,
    build_raw_rubric_dimension_labels,
    build_successful_generation_capture_dataset,
)
from projects.MOREBENCH.shared.morebench_dataset import (
    DEFAULT_SPLIT,
    PUBLIC_CONFIG,
    build_official_reasoning_by_dilemma_dataset,
    build_rubric_criterion_dataset,
)


DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
MODEL_ID = "Qwen/Qwen3-30B-A3B"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
ARTIFACTS_VOLUME = "xenon-data"
ARTIFACTS_ROOT = "/data/artifacts/morebench_phase_02_dimension_probes"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "morebench_phase_02_dimension_probes"
DEFAULT_REPORT_DIR = "projects/MOREBENCH/procedural_probe/phase_02/reports/pipelines_v2"
CAPTURED_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)
GENERATION_MAX_TOKENS = None
PHASE_02_MAX_MODEL_LEN = 40_960
MAX_NUM_SEQS = 24
MODEL_BOUND_SHARD_COUNT = 4


def build_prompt_dataset(*, limit: int | None = None) -> Dataset:
    return build_official_reasoning_by_dilemma_dataset(
        config=PUBLIC_CONFIG,
        split=DEFAULT_SPLIT,
        limit=limit,
        name="morebench_public_official_reasoning_by_dilemma",
    )


def build_criterion_dataset(*, limit: int | None = None) -> Dataset:
    return build_rubric_criterion_dataset(
        config=PUBLIC_CONFIG,
        split=DEFAULT_SPLIT,
        limit=limit,
        name="morebench_public_rubric_criteria",
    )


def build_dataset(*, limit: int | None = None) -> Dataset:
    return build_prompt_dataset(limit=limit)


def build_two_dilemma_dataset() -> Dataset:
    return build_prompt_dataset(limit=2)


def build_runner_specs() -> dict[str, object]:
    db = PostgresSource.from_env(DB_ENV_VAR)
    artifact_store = ModalVolumeStore(name=ARTIFACTS_VOLUME, root=ARTIFACTS_ROOT)
    workflow_catalog = PostgresCatalog(source=db)
    db_secret = ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon")
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H200",
                timeout_seconds=60 * 60 * 4,
                max_containers=MODEL_BOUND_SHARD_COUNT,
                shard_count=MODEL_BOUND_SHARD_COUNT,
                secrets=(db_secret,),
                volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
            ),
            artifacts=artifact_store,
            catalog=workflow_catalog,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=6,
                memory_mb=24 * 1024,
                timeout_seconds=60 * 60 * 2,
                max_containers=1,
                secrets=(db_secret,),
            ),
            artifacts=artifact_store,
            catalog=workflow_catalog,
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
            catalog=workflow_catalog,
        ),
    }


def build_phase_02_method_plan() -> dict[str, object]:
    return {
        "phase": "raw_rubric_dimension_prompt_vs_generation_probes",
        "methodology_sources": [
            "docs/mech-interp/mech-interp-methodology-roster.md",
            "benchmark-mech-interp skill",
            "constructing-llm-probes skill",
        ],
        "core_change": (
            "This phase no longer probes aggregate rubric-weight burden. It probes raw rubric-dimension "
            "labels, ignoring criterion weights, at both prompt end and generated-answer end."
        ),
        "methods": [
            "split deterministic generation followed by a separate prompt+generated residual capture pass",
            "uncapped generation bounded by the Qwen3-30B-A3B 40,960-token model-config context window",
            "length-finished generations are excluded before activation capture and downstream probes",
            "linear residual probes over prompt-final activations",
            "linear residual probes over generated-answer-final activations from replayed prompt+generated contexts",
            "prompt-text and generated-answer text baselines for lexical/style leakage checks",
            "shuffled/selectivity controls for activation probes",
        ],
        "non_goals": [
            "No reusable LLM-as-judge abstraction in this phase.",
            "No official criterion-fulfillment scoring in this phase.",
            "No aggregate rubric-weight burden probes as headline targets.",
        ],
    }


def build_phase_02_missing_pieces() -> list[str]:
    return [
        "Criterion-fulfillment judging for generated answers, kept project-local until the abstraction is clearer.",
        "Intermediate criterion clusters within each rubric dimension, especially for identifying and logical process.",
        "Fine-grained domain labels for daily dilemmas and AI-risk settings if prompt-side controls are needed.",
        "Prompt-format cleanup for Qwen chat/instruct behavior before scaling the full generation set.",
        "Rope-scaling support would be needed before running Qwen3-30B-A3B above its 40,960-token model-config context.",
        "Causal patching or steering only after a stable generated-token readout is found.",
    ]


def _engine(*, max_model_len: int = PHASE_02_MAX_MODEL_LEN) -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        model_path_root=MODEL_VOLUME_PATH,
        max_model_len=max_model_len,
        enforce_eager=False,
        enable_prefix_caching=True,
        enable_thinking=False,
        max_num_seqs=MAX_NUM_SEQS,
    )


def _generated_capture_dataset_ref() -> Dataset:
    return Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef("build_successful_generation_capture_dataset"),
        result_key="dataset",
        provides_token_sections=True,
        name="morebench_phase_02_successful_prompt_generated_contexts",
    )


def _probe_description(target: object, *, surface: str) -> str:
    if surface == "prompt":
        surface_text = (
            "Prompt-end readout: the feature row is the final token of the official MoReBench prompt, "
            "before the model generates an answer."
        )
    else:
        surface_text = (
            "Generated-answer readout: generation is produced in an upstream step, length-finished rows are "
            "dropped, and the full prompt+answer text is replayed for activation capture. The feature row is "
            "the final token of the generated answer in that replayed context."
        )
    if getattr(target, "target_kind", "binary") == "multiclass":
        metric_text = (
            "Accuracy/balanced accuracy should be interpreted against class balance, majority, and shuffled-label "
            "controls; AUROC is intentionally omitted for this multiclass target."
        )
    else:
        metric_text = (
            "Balanced accuracy/AUROC near 0.5 means no reliable linear readout; values above text and "
            "shuffled/selectivity controls mean decodability, not causal use."
        )
    return (
        f"{surface_text} Target: {target.probe_question} The label is raw rubric dimension structure with "
        f"criterion weights ignored. {metric_text}"
    )


def _text_baseline_description(target: object, *, surface: str) -> str:
    source = "dilemma prompt text" if surface == "prompt" else "generated answer text"
    return (
        f"Lexical control over {source}: {target.text_baseline_question} High balanced accuracy/AUROC means the "
        "target is mostly visible from surface text, so residual probe metrics need to beat this baseline."
    )


def _probe_metrics(target: object) -> tuple[str, ...]:
    if getattr(target, "target_kind", "binary") == "multiclass":
        return ("accuracy", "balanced_accuracy", "selectivity")
    return ("accuracy", "balanced_accuracy", "auroc", "selectivity")


def _target_steps(target: object) -> tuple[WorkflowStep, ...]:
    labels = StepRef("build_raw_rubric_dimension_labels")
    generated_rows = StepRef("build_successful_generation_capture_dataset")
    rows = generated_rows.label("base_dilemma_id")
    label = labels.label(target.label)
    prompt_feature = StepRef("capture_prompt_generated_residual").feature("residual_prompt_end")
    generation_feature = StepRef("capture_prompt_generated_residual").feature("residual_generation_end")
    return (
        WorkflowStep(
            name=f"probe_prompt_{target.step_slug}_residual",
            runner="analysis_cpu",
            description=_probe_description(target, surface="prompt"),
            spec=ProbeSpec(
                feature=prompt_feature,
                rows=rows,
                labels=label,
                group_by=rows,
                tokens=TokenSelector.full_sequence(),
                pooling=TokenPooling.last(),
                metrics=_probe_metrics(target),
                baselines=("majority", "shuffled_label"),
            ),
        ),
        WorkflowStep(
            name=f"text_baseline_prompt_{target.step_slug}",
            runner="analysis_cpu",
            description=_text_baseline_description(target, surface="prompt"),
            spec=TextBaselineSpec(
                text=labels.label("dilemma_text"),
                rows=rows,
                labels=label,
                group_by=rows,
                model="countvectorizer_logreg",
                metrics=("accuracy", "balanced_accuracy", "auroc"),
            ),
        ),
        WorkflowStep(
            name=f"probe_generation_{target.step_slug}_residual",
            runner="analysis_cpu",
            description=_probe_description(target, surface="generation"),
            spec=ProbeSpec(
                feature=generation_feature,
                rows=rows,
                labels=label,
                group_by=rows,
                tokens=TokenSelector.full_sequence(),
                pooling=TokenPooling.last(),
                metrics=_probe_metrics(target),
                baselines=("majority", "shuffled_label"),
            ),
        ),
        WorkflowStep(
            name=f"text_baseline_generation_{target.step_slug}",
            runner="analysis_cpu",
            description=_text_baseline_description(target, surface="generation"),
            spec=TextBaselineSpec(
                text=generated_rows.label("generated_text"),
                rows=rows,
                labels=label,
                group_by=rows,
                model="countvectorizer_logreg",
                metrics=("accuracy", "balanced_accuracy", "auroc"),
            ),
        ),
    )


def build_workflow(
    prompt_dataset: Dataset | None = None,
    criterion_dataset: Dataset | None = None,
    *,
    report_output_dir: str = DEFAULT_REPORT_DIR,
) -> WorkflowSpec:
    prompt_dataset = prompt_dataset or build_prompt_dataset()
    criterion_dataset = criterion_dataset or build_criterion_dataset()

    dimension_steps: list[WorkflowStep] = []
    for target in RAW_RUBRIC_DIMENSION_TARGETS:
        dimension_steps.extend(_target_steps(target))

    report_inputs: list[object] = [
        StepRef("build_raw_rubric_dimension_labels"),
        StepRef("generate_dilemma_responses"),
        StepRef("build_successful_generation_capture_dataset"),
        StepRef("capture_prompt_generated_residual"),
    ]
    for target in RAW_RUBRIC_DIMENSION_TARGETS:
        report_inputs.extend(
            [
                StepRef(f"probe_prompt_{target.step_slug}_residual"),
                StepRef(f"text_baseline_prompt_{target.step_slug}"),
                StepRef(f"probe_generation_{target.step_slug}_residual"),
                StepRef(f"text_baseline_generation_{target.step_slug}"),
            ]
        )
    report_inputs.extend(
        [
            {"method_plan": build_phase_02_method_plan()},
            {"missing_pieces": build_phase_02_missing_pieces()},
        ]
    )

    return WorkflowSpec(
        name="morebench_phase_02_raw_dimension_prompt_generation_probes",
        steps=(
            WorkflowStep(
                name="build_raw_rubric_dimension_labels",
                runner="analysis_cpu",
                description=(
                    "Collapse criterion-level MoReBench rubrics into raw dimension-count labels per dilemma. "
                    "This ignores criterion weights and asks which rubric dimension family is most represented."
                ),
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_raw_rubric_dimension_labels,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"criteria": criterion_dataset},
                ),
            ),
            WorkflowStep(
                name="generate_dilemma_responses",
                runner="capture_gpu",
                description=(
                    "Generate deterministic MoReBench answers with no explicit max-token cap. The run is bounded "
                    "by the Qwen3-30B-A3B 40,960-token model-config context window and stores response text plus finish reasons in the "
                    "generation artifact on the Modal volume. The engine requests Qwen thinking disabled."
                ),
                spec=GenerationRunSpec(
                    engine=_engine(),
                    dataset=prompt_dataset,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=GENERATION_MAX_TOKENS,
                        temperature=0.0,
                        top_p=1.0,
                    ),
                ),
            ),
            WorkflowStep(
                name="build_successful_generation_capture_dataset",
                runner="analysis_cpu",
                description=(
                    "Filter generated rows before activation capture. Rows with finish_reason=length are ignored; "
                    "the remaining rows become prompt+generated examples with explicit char-span token sections "
                    "for prompt, generated, full, prompt_end, generated_end, and full_end."
                ),
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_successful_generation_capture_dataset,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_dilemma_responses")},
                ),
            ),
            WorkflowStep(
                name="capture_prompt_generated_residual",
                runner="capture_gpu",
                description=(
                    "Replay successful prompt+generated texts and capture residual activations separately from "
                    "generation. Token sections come from metadata emitted by the project-local generation filter; "
                    "the stored feature rows are prompt_end, generated_end, and full_end."
                ),
                spec=CaptureSpec(
                    engine=_engine(),
                    dataset=_generated_capture_dataset_ref(),
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
                        ResidualSite(
                            name="residual_full_end",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.section("full_end"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ],
                ),
            ),
            *dimension_steps,
            WorkflowStep(
                name="report",
                runner="report_local",
                description=(
                    "Assemble the Phase 2 raw-dimension report. Interpret prompt-end and generation-context "
                    "metrics against their matching text baselines and shuffled/selectivity controls."
                ),
                spec=ReportSpec(
                    inputs=tuple(report_inputs),
                    template="default",
                    output_dir=report_output_dir,
                ),
            ),
        ),
    )


def build_two_dilemma_split_smoke_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    prompt_dataset = dataset or build_two_dilemma_dataset()
    return WorkflowSpec(
        name="morebench_phase_02_split_2_dilemma_capture_smoke",
        steps=(
            WorkflowStep(
                name="generate_dilemma_responses",
                runner="capture_gpu",
                description=(
                    "Two-dilemma smoke for the split Phase 2 path. This generation step writes responses and "
                    "finish reasons to a generation artifact without activation capture. The engine requests "
                    "Qwen thinking disabled."
                ),
                spec=GenerationRunSpec(
                    engine=_engine(),
                    dataset=prompt_dataset,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=GENERATION_MAX_TOKENS,
                        temperature=0.0,
                        top_p=1.0,
                    ),
                ),
            ),
            WorkflowStep(
                name="build_successful_generation_capture_dataset",
                runner="analysis_cpu",
                description=(
                    "Convert successful generated rows into prompt+generated capture examples and drop "
                    "length-finished rows."
                ),
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_successful_generation_capture_dataset,
                        local_python_sources=("projects/MOREBENCH",),
                    ),
                    inputs={"generation": StepRef("generate_dilemma_responses")},
                ),
            ),
            WorkflowStep(
                name="capture_prompt_generated_residual",
                runner="capture_gpu",
                description=(
                    "Smoke the separate replay capture using only layer 0 endpoint sections from the generated "
                    "prompt+answer dataset."
                ),
                spec=CaptureSpec(
                    engine=_engine(),
                    dataset=_generated_capture_dataset_ref(),
                    sites=[
                        ResidualSite(
                            name="residual_prompt_end",
                            site="resid_post",
                            layers=[0],
                            tokens=TokenSelector.section("prompt_end"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                        ResidualSite(
                            name="residual_generation_end",
                            site="resid_post",
                            layers=[0],
                            tokens=TokenSelector.section("generated_end"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                        ResidualSite(
                            name="residual_full_end",
                            site="resid_post",
                            layers=[0],
                            tokens=TokenSelector.section("full_end"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ],
                ),
            ),
        ),
    )
