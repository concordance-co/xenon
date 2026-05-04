"""HealthBench scope-vs-commit activation smoke workflow."""

from __future__ import annotations

from pipelines_v2.api import (
    AddDirectionPatch,
    CaptureSpec,
    Dataset,
    DirectionSpec,
    GenerationRunSpec,
    GenerationSpec,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalSecret,
    ModalVolumeMount,
    ModalVolumeStore,
    PatchedGenerationSpec,
    PostgresCatalog,
    PostgresSource,
    PromptMetadataBuilder,
    ReportSpec,
    ResidualInterventionSite,
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

from projects.healthbench.shared.scope_vs_commit import (
    scope_vs_commit_prompt_metadata,
    summarize_steering_outputs,
)


MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
SMOKE_TABLE = "healthbench_scope_vs_commit_smoke_v1"
REPORT_DIR = "projects/healthbench/phase_01/reports/scope_vs_commit_activation_smoke"
DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
CAPTURED_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)
PATCH_LAYERS = (28, 32, 36, 40)
PATCH_STRENGTH = 2.0


def build_dataset() -> Dataset:
    db = PostgresSource.from_env(DB_ENV_VAR)
    return Dataset.from_postgres(
        source=db,
        sql=(
            "SELECT *, triple_id || '_s' || lpad(sample_index::text, 2, '0') AS contrast_case_id "
            f"FROM {SMOKE_TABLE} "
            "WHERE sample_index %% 4 = 0 "
            "ORDER BY triple_id, context_completeness_index, sample_index"
        ),
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        prompt_hash_column="prompt_sha256",
        label_columns=[
            "version",
            "triple_id",
            "condition_id",
            "context_condition",
            "context_completeness_index",
            "axis",
            "sample_index",
            "prompt_char_len",
        ],
        case_columns=["triple_id", "condition_id", "contrast_case_id"],
        case_key_column="triple_id",
        id="healthbench_scope_vs_commit_activation_smoke_v1",
        name="HealthBench scope-vs-commit activation smoke v1",
    )


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    if dataset is None:
        dataset = build_dataset()
    prompt_metadata = PromptMetadataBuilder.from_function(
        scope_vs_commit_prompt_metadata,
        local_python_sources=("projects/healthbench",),
    )
    capture_feature_user = StepRef("capture_scope_commit_residuals").feature("residual_user_prompt")
    capture_feature_generated = StepRef("capture_scope_commit_residuals").feature(
        "residual_first_generated_context"
    )
    partial_rows = dataset.labels("context_condition").equals("partial_context")
    return WorkflowSpec(
        name="healthbench_scope_vs_commit_activation_smoke_v1",
        steps=(
            WorkflowStep(
                name="capture_scope_commit_residuals",
                runner="capture_gpu",
                description=(
                    "Capture residual activations on every fourth smoke row at "
                    "user-prompt end and first generated-token context."
                ),
                spec=CaptureSpec(
                    engine=_qwen_engine(max_num_seqs=16),
                    dataset=dataset,
                    sites=(
                        ResidualSite(
                            name="residual_user_prompt",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.section("user_prompt"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                        ResidualSite(
                            name="residual_first_generated_context",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.section("generated"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ),
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=2,
                        temperature=0.0,
                        top_p=1.0,
                        capture_reasoning=False,
                        capture_generated_tokens=True,
                    ),
                    prompt_metadata_builder=prompt_metadata,
                ),
            ),
            WorkflowStep(
                name="learn_user_eot_scope_direction",
                runner="analysis_cpu",
                depends_on=("capture_scope_commit_residuals",),
                description=(
                    "Compute under-context minus full-context direction at the "
                    "end of the rendered user prompt."
                ),
                spec=DirectionSpec(
                    feature=capture_feature_user,
                    positive=dataset.labels("context_condition").equals("under_context"),
                    negative=dataset.labels("context_condition").equals("full_context"),
                    layers=list(CAPTURED_LAYERS),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.last(),
                ),
            ),
            WorkflowStep(
                name="learn_first_generated_scope_direction",
                runner="analysis_cpu",
                depends_on=("capture_scope_commit_residuals",),
                description=(
                    "Compute under-context minus full-context direction at the "
                    "first generated-token context."
                ),
                spec=DirectionSpec(
                    feature=capture_feature_generated,
                    positive=dataset.labels("context_condition").equals("under_context"),
                    negative=dataset.labels("context_condition").equals("full_context"),
                    layers=list(CAPTURED_LAYERS),
                    tokens=TokenSelector.full_sequence(),
                    pooling=TokenPooling.first(),
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                depends_on=(
                    "learn_first_generated_scope_direction",
                    "learn_user_eot_scope_direction",
                    "summarize_steering",
                ),
                description="Build local report from activation direction and steering artifacts.",
                spec=ReportSpec(
                    inputs=[
                        StepRef("learn_user_eot_scope_direction"),
                        StepRef("learn_first_generated_scope_direction"),
                        StepRef("summarize_steering"),
                    ],
                    template="summary",
                    output_dir=REPORT_DIR,
                ),
            ),
            WorkflowStep(
                name="baseline_partial_generation",
                runner="capture_gpu",
                description="Generate unpatched baseline responses on partial-context target rows.",
                spec=GenerationRunSpec(
                    engine=_qwen_engine(max_num_seqs=16),
                    dataset=dataset,
                    select_when=partial_rows,
                    generation=_steering_generation(),
                ),
            ),
            _steering_step(
                name="patch_user_eot_pos2",
                direction=StepRef("learn_user_eot_scope_direction"),
                dataset=dataset,
                select_when=partial_rows,
                target_tokens=TokenSelector.section("user_prompt"),
                prompt_metadata=prompt_metadata,
                strength=PATCH_STRENGTH,
            ),
            _steering_step(
                name="patch_user_eot_neg2",
                direction=StepRef("learn_user_eot_scope_direction"),
                dataset=dataset,
                select_when=partial_rows,
                target_tokens=TokenSelector.section("user_prompt"),
                prompt_metadata=prompt_metadata,
                strength=-PATCH_STRENGTH,
                depends_on=("patch_user_eot_pos2",),
            ),
            _steering_step(
                name="patch_first_generated_direction_pos2",
                direction=StepRef("learn_first_generated_scope_direction"),
                dataset=dataset,
                select_when=partial_rows,
                target_tokens=TokenSelector.last(),
                prompt_metadata=None,
                strength=PATCH_STRENGTH,
                depends_on=("patch_user_eot_neg2",),
            ),
            _steering_step(
                name="patch_first_generated_direction_neg2",
                direction=StepRef("learn_first_generated_scope_direction"),
                dataset=dataset,
                select_when=partial_rows,
                target_tokens=TokenSelector.last(),
                prompt_metadata=None,
                strength=-PATCH_STRENGTH,
                depends_on=("patch_first_generated_direction_pos2",),
            ),
            WorkflowStep(
                name="summarize_steering",
                runner="analysis_cpu",
                depends_on=(
                    "baseline_partial_generation",
                    "patch_user_eot_pos2",
                    "patch_user_eot_neg2",
                    "patch_first_generated_direction_pos2",
                    "patch_first_generated_direction_neg2",
                ),
                description="Summarize baseline and patched generation outputs with response-shape proxies.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_steering_outputs,
                        local_python_sources=("projects/healthbench",),
                    ),
                    inputs={
                        "baseline": StepRef("baseline_partial_generation"),
                        "user_eot_pos2": StepRef("patch_user_eot_pos2"),
                        "user_eot_neg2": StepRef("patch_user_eot_neg2"),
                        "first_generated_direction_pos2": StepRef(
                            "patch_first_generated_direction_pos2"
                        ),
                        "first_generated_direction_neg2": StepRef(
                            "patch_first_generated_direction_neg2"
                        ),
                    },
                ),
            ),
        ),
    )


def _steering_step(
    *,
    name: str,
    direction: StepRef,
    dataset: Dataset,
    select_when: object,
    target_tokens: TokenSelector,
    prompt_metadata: PromptMetadataBuilder | None,
    strength: float,
    depends_on: tuple[str, ...] = (),
) -> WorkflowStep:
    return WorkflowStep(
        name=name,
        runner="patch_gpu",
        depends_on=tuple(dict.fromkeys((direction.step, *depends_on))),
        description=f"Run patched generation with {direction.step} at strength {strength:+.1f}.",
        spec=PatchedGenerationSpec(
            engine=_qwen_engine(max_num_seqs=16),
            dataset=dataset,
            select_when=select_when,
            patch=AddDirectionPatch(
                direction=direction,
                write_site=ResidualInterventionSite(
                    site="resid_post",
                    layers=PATCH_LAYERS,
                ),
                target_tokens=target_tokens,
                strength=strength,
            ),
            generation=_steering_generation(),
            prompt_metadata_builder=prompt_metadata,
        ),
    )


def _steering_generation() -> GenerationSpec:
    return GenerationSpec(
        enabled=True,
        max_tokens=384,
        temperature=0.8,
        top_p=0.95,
        capture_reasoning=False,
    )


def build_runner_specs() -> dict[str, object]:
    db = PostgresSource.from_env(DB_ENV_VAR)
    catalog = PostgresCatalog(source=db)
    artifact_store = ModalVolumeStore(name="xenon-data", root="/data/artifacts")
    neon_secret = ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon")
    model_volume = ModalVolumeMount(name="xenon-models", mount_path="/models")
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="A100-80GB",
                cpu=12,
                memory_mb=96 * 1024,
                timeout_seconds=60 * 60 * 4,
                max_containers=2,
                shard_count=2,
                secrets=(neon_secret,),
                volumes=(model_volume,),
            ),
            artifacts=artifact_store,
            catalog=catalog,
        ),
        "patch_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="A100-80GB",
                cpu=12,
                memory_mb=96 * 1024,
                timeout_seconds=60 * 60 * 4,
                max_containers=1,
                secrets=(neon_secret,),
                volumes=(model_volume,),
            ),
            artifacts=artifact_store,
            catalog=catalog,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=4,
                memory_mb=16 * 1024,
                timeout_seconds=60 * 30,
                secrets=(neon_secret,),
            ),
            artifacts=artifact_store,
            catalog=catalog,
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(REPORT_DIR),
            catalog=catalog,
        ),
    }


def _qwen_engine(*, max_num_seqs: int) -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        model_path_root="/models",
        max_model_len=4096,
        gpu_memory_utilization=0.88,
        enforce_eager=False,
        max_num_seqs=max_num_seqs,
        enable_prefix_caching=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
