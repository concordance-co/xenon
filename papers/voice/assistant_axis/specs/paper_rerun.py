"""Paper-rerun scaffold for Assistant Axis.

This is the paper implementation surface: released prompt-source rows, paper
prompt expansion, paper generation defaults, response activation capture, axis
derivation, and released-axis comparison.
"""

from __future__ import annotations

import os

from pipelines_v2.api import (
    ArtifactDatasetSource,
    AssistantAxisPrecomputedCoordinateSpec,
    AssistantAxisVectorSpec,
    CaptureSpec,
    Dataset,
    GenerationRunSpec,
    GenerationSpec,
    ProjectionSpec,
    ReportSpec,
    ResidualSite,
    SectionSelector,
    StepRef,
    TensorStorage,
    TokenSelector,
    TransformBuilder,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)

from papers.voice.assistant_axis.judge import judge_generated_role_adherence
from papers.voice.assistant_axis.paper import PAPER_GENERATION_CONFIG, source_prompt_dataset
from papers.voice.assistant_axis.runtime import (
    build_paper_generation_prompt_dataset,
    env_int,
    model_id,
    model_key_from_env,
    runner_specs,
    target_layer,
    vllm_engine,
)


WORKFLOW_NAME = "papers_voice_assistant_axis_paper_rerun"


def _artifact_dataset(step_name: str, *, result_key: str = "dataset", name: str) -> Dataset:
    return Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef(step_name),
        result_key=result_key,
        provides_token_sections=True,
        name=name,
    )


def build_dataset() -> Dataset:
    return source_prompt_dataset()


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    model_key = model_key_from_env()
    source_rows = dataset or build_dataset()
    prompt_dataset = Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef("build_paper_generation_prompts"),
        result_key="dataset",
        name=f"{WORKFLOW_NAME}_generation_prompts",
    )
    response_dataset = _artifact_dataset(
        "judge_role_adherence",
        name=f"{WORKFLOW_NAME}_generated_responses",
    )
    layer = target_layer(model_key)
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="build_paper_generation_prompts",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_paper_generation_prompt_dataset,
                        local_python_sources=("papers",),
                    ),
                    inputs={
                        "source_dataset": source_rows,
                        "model_key": model_key,
                        "role_limit": env_int("ASSISTANT_AXIS_ROLE_LIMIT", 2),
                        "question_limit": env_int("ASSISTANT_AXIS_QUESTION_LIMIT", 2),
                        "instruction_limit": env_int("ASSISTANT_AXIS_INSTRUCTION_LIMIT", 1),
                    },
                ),
            ),
            WorkflowStep(
                name="generate_paper_responses",
                runner="generation_gpu",
                spec=GenerationRunSpec(
                    engine=vllm_engine(model_key=model_key, max_num_seqs=1),
                    dataset=prompt_dataset,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=PAPER_GENERATION_CONFIG.max_tokens,
                        temperature=PAPER_GENERATION_CONFIG.temperature,
                        top_p=PAPER_GENERATION_CONFIG.top_p,
                    ),
                ),
            ),
            WorkflowStep(
                name="judge_role_adherence",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        judge_generated_role_adherence,
                        local_python_sources=("papers",),
                    ),
                    inputs={"generation": StepRef("generate_paper_responses")},
                ),
            ),
            WorkflowStep(
                name="capture_response_activations",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=vllm_engine(model_key=model_key, add_generation_prompt=False),
                    dataset=response_dataset,
                    sites=(
                        ResidualSite(
                            name="response_residual",
                            site="resid_post",
                            layers=(layer,),
                            tokens=TokenSelector.section("assistant_response"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ),
                ),
            ),
            WorkflowStep(
                name="derive_assistant_axis",
                runner="analysis_cpu",
                spec=AssistantAxisVectorSpec(
                    feature=StepRef("capture_response_activations").feature("response_residual"),
                    role_by=response_dataset.labels("role"),
                    default_when=response_dataset.labels("axis_kind").equals("default"),
                    role_when=response_dataset.labels("axis_kind").equals("role"),
                    score_by=response_dataset.labels("adherence_score"),
                    score_values=(3,),
                    min_role_examples_per_role=env_int("ASSISTANT_AXIS_MIN_ROLE_EXAMPLES_PER_ROLE", 1),
                    min_default_examples=env_int("ASSISTANT_AXIS_MIN_DEFAULT_EXAMPLES", 1),
                    layers=(layer,),
                    tokens=TokenSelector.full_sequence(),
                    model_id=model_id(model_key),
                    metadata={
                        "paper": "assistant_axis",
                        "full_paper_role_threshold": 50,
                        "judge_model_env": "ASSISTANT_AXIS_JUDGE_MODEL",
                    },
                ),
            ),
            WorkflowStep(
                name="load_released_assistant_axis",
                runner="analysis_cpu",
                spec=AssistantAxisPrecomputedCoordinateSpec(
                    model_id=model_id(model_key),
                    token_env_var="HF_TOKEN",
                ),
            ),
            WorkflowStep(
                name="score_generated_responses",
                runner="analysis_cpu",
                spec=ProjectionSpec(
                    feature=StepRef("capture_response_activations").feature("response_residual"),
                    coordinates=(StepRef("derive_assistant_axis"), StepRef("load_released_assistant_axis")),
                    slices=SectionSelector.named("assistant_response"),
                    layers=(layer,),
                    summaries=("mean", "min", "max"),
                    emit_labels=True,
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(
                        StepRef("build_paper_generation_prompts"),
                        StepRef("judge_role_adherence"),
                        StepRef("derive_assistant_axis"),
                        StepRef("load_released_assistant_axis"),
                        StepRef("score_generated_responses"),
                    ),
                    template="voice_assistant_axis_paper_rerun",
                    output_dir=f"papers/voice/assistant_axis/reports/{WORKFLOW_NAME}",
                ),
            ),
        ),
    )


def build_runner_specs() -> dict[str, object]:
    return runner_specs(
        workflow_name=os.getenv("ASSISTANT_AXIS_WORKFLOW_NAME", WORKFLOW_NAME),
        include_judge_env=True,
    )
