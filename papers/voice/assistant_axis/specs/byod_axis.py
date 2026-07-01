"""BYOD workflow for the Assistant Axis method.

Bring rows matching ``papers/voice/schemas/assistant_axis_method.schema.json``
via ``ASSISTANT_AXIS_BYOD_JSONL``. Without that env var, this runs a tiny
fixture that exercises default-vs-role derivation and probe scoring.
"""

from __future__ import annotations

import os

from pipelines_v2.api import (
    AssistantAxisPrecomputedCoordinateSpec,
    AssistantAxisVectorSpec,
    CaptureSpec,
    Dataset,
    ProjectionSpec,
    ReportSpec,
    ResidualSite,
    SectionSelector,
    StepRef,
    TensorStorage,
    TokenSelector,
    WorkflowSpec,
    WorkflowStep,
)

from papers.voice.assistant_axis.runtime import (
    byod_dataset_from_env,
    env_int,
    model_id,
    model_key_from_env,
    runner_specs,
    target_layer,
    vllm_engine,
)


WORKFLOW_NAME = "papers_voice_assistant_axis_byod"


def build_dataset() -> Dataset:
    return byod_dataset_from_env(workflow_name=WORKFLOW_NAME)


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    model_key = model_key_from_env()
    ds = dataset or build_dataset()
    layer = target_layer(model_key)
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="capture_byod_responses",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=vllm_engine(model_key=model_key, add_generation_prompt=False),
                    dataset=ds,
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
                name="derive_byod_axis",
                runner="analysis_cpu",
                spec=AssistantAxisVectorSpec(
                    feature=StepRef("capture_byod_responses").feature("response_residual"),
                    role_by=ds.labels("role"),
                    default_when=ds.labels("axis_kind").equals("default"),
                    role_when=ds.labels("axis_kind").equals("role"),
                    score_by=ds.labels("adherence_score"),
                    score_values=(3,),
                    min_role_examples_per_role=env_int("ASSISTANT_AXIS_MIN_ROLE_EXAMPLES_PER_ROLE", 1),
                    min_default_examples=env_int("ASSISTANT_AXIS_MIN_DEFAULT_EXAMPLES", 1),
                    layers=(layer,),
                    tokens=TokenSelector.full_sequence(),
                    model_id=model_id(model_key),
                    metadata={"paper_method": "assistant_axis_default_vs_role", "surface": "BYOD"},
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
                name="score_byod_rows",
                runner="analysis_cpu",
                spec=ProjectionSpec(
                    feature=StepRef("capture_byod_responses").feature("response_residual"),
                    coordinates=(StepRef("derive_byod_axis"), StepRef("load_released_assistant_axis")),
                    slices=SectionSelector.named("assistant_response"),
                    layers=(layer,),
                    summaries=("mean", "trend"),
                    emit_labels=True,
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(StepRef("derive_byod_axis"), StepRef("load_released_assistant_axis"), StepRef("score_byod_rows")),
                    template="voice_assistant_axis_byod",
                    output_dir=f"papers/voice/assistant_axis/reports/{WORKFLOW_NAME}",
                ),
            ),
        ),
    )


def build_runner_specs() -> dict[str, object]:
    return runner_specs(workflow_name=os.getenv("ASSISTANT_AXIS_WORKFLOW_NAME", WORKFLOW_NAME))
