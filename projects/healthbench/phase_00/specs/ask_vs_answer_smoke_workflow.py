"""Small HealthBench-inspired ask-vs-answer behavioral smoke workflow."""

from __future__ import annotations

from pipelines_v2.api import (
    ArtifactDatasetSource,
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
    ReportSpec,
    StepRef,
    TransformBuilder,
    TransformSpec,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)

from projects.healthbench.shared.ask_vs_answer_smoke import (
    ask_vs_answer_judge_schema,
    build_judge_dataset,
    summarize_judgments,
)


MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
SMOKE_TABLE = "healthbench_ask_vs_answer_smoke_v1"
REPORT_DIR = "projects/healthbench/phase_00/reports/ask_vs_answer_smoke"
DB_ENV_VAR = "XENON_NEON_DATABASE_URL"


def build_dataset() -> Dataset:
    db = PostgresSource.from_env(DB_ENV_VAR)
    return Dataset.from_postgres(
        source=db,
        table=SMOKE_TABLE,
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        prompt_hash_column="prompt_sha256",
        label_columns=[
            "version",
            "triple_id",
            "condition",
            "expected_behavior",
            "axis",
            "condition_index",
            "sample_index",
        ],
        case_columns=["triple_id", "condition_id"],
        case_key_column="triple_id",
        id=SMOKE_TABLE,
        name="HealthBench ask-vs-answer smoke v1",
    )


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    if dataset is None:
        dataset = build_dataset()
    judge_dataset = Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef("build_judge_dataset"),
        result_key="dataset",
        id="healthbench_ask_vs_answer_smoke_judge_v1",
        name="HealthBench ask-vs-answer smoke judge v1",
    )
    return WorkflowSpec(
        name="healthbench_ask_vs_answer_smoke_v1",
        steps=(
            WorkflowStep(
                name="generate_responses",
                runner="capture_gpu",
                description="Generate Qwen responses for the 72-row matched-triple smoke set.",
                spec=GenerationRunSpec(
                    engine=_qwen_engine(max_num_seqs=24),
                    dataset=dataset,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=384,
                        temperature=0.7,
                        top_p=0.95,
                        capture_reasoning=False,
                    ),
                ),
            ),
            WorkflowStep(
                name="build_judge_dataset",
                runner="analysis_cpu",
                depends_on=("generate_responses",),
                description="Build rubric judge prompts from generated responses.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_judge_dataset,
                        local_python_sources=("projects/healthbench",),
                    ),
                    inputs={"generations": StepRef("generate_responses")},
                ),
            ),
            WorkflowStep(
                name="judge_ask_vs_answer",
                runner="capture_gpu",
                depends_on=("build_judge_dataset",),
                description="Classify each response as ask, mixed, or answer.",
                spec=GenerationRunSpec(
                    engine=_qwen_engine(max_num_seqs=32),
                    dataset=judge_dataset,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=160,
                        temperature=0.0,
                        top_p=1.0,
                        capture_reasoning=False,
                        structured_output=ask_vs_answer_judge_schema(),
                    ),
                ),
            ),
            WorkflowStep(
                name="summarize_behavior",
                runner="analysis_cpu",
                depends_on=("judge_ask_vs_answer",),
                description="Summarize behavior rates by bucket and matched triple.",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_judgments,
                        local_python_sources=("projects/healthbench",),
                    ),
                    inputs={"judgments": StepRef("judge_ask_vs_answer")},
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                depends_on=("summarize_behavior",),
                description="Build local report from the smoke summary payload.",
                spec=ReportSpec(
                    inputs=[StepRef("summarize_behavior")],
                    template="summary",
                    output_dir=REPORT_DIR,
                ),
            ),
        ),
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
        add_generation_prompt=True,
        enable_thinking=False,
    )
