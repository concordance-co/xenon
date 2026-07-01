"""BYOT trace-scoring workflow for released Assistant Axis coordinates."""

from __future__ import annotations

import os

from pipelines_v2.api import (
    AssistantAxisPrecomputedCoordinateSpec,
    AssistantAxisTraitCoordinateSpec,
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
    byot_dataset_from_env,
    env_list,
    model_id,
    model_key_from_env,
    runner_specs,
    target_layer,
    vllm_engine,
)


WORKFLOW_NAME = "papers_voice_assistant_axis_byot"
DEFAULT_TRAITS = ("calm", "supportive", "technical", "analytical", "confident", "verbose", "hostile", "condescending")


def build_dataset() -> Dataset:
    return byot_dataset_from_env(workflow_name=WORKFLOW_NAME)


def _trait_step_name(trait: str) -> str:
    return "trait_" + "".join(ch.lower() if ch.isalnum() else "_" for ch in trait).strip("_")


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    model_key = model_key_from_env()
    ds = dataset or build_dataset()
    layer = target_layer(model_key)
    traits = env_list("ASSISTANT_AXIS_TRAITS", DEFAULT_TRAITS)
    trait_steps = tuple(
        WorkflowStep(
            name=_trait_step_name(trait),
            runner="analysis_cpu",
            spec=AssistantAxisTraitCoordinateSpec(
                model_id=model_id(model_key),
                trait=trait,
                token_env_var="HF_TOKEN",
            ),
        )
        for trait in traits
    )
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="capture_trace",
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
                name="assistant_axis",
                runner="analysis_cpu",
                spec=AssistantAxisPrecomputedCoordinateSpec(
                    model_id=model_id(model_key),
                    token_env_var="HF_TOKEN",
                ),
            ),
            *trait_steps,
            WorkflowStep(
                name="score_trace",
                runner="analysis_cpu",
                spec=ProjectionSpec(
                    feature=StepRef("capture_trace").feature("response_residual"),
                    coordinates=(StepRef("assistant_axis"), *(StepRef(_trait_step_name(trait)) for trait in traits)),
                    slices=SectionSelector.named("assistant_response"),
                    layers=(layer,),
                    summaries=("mean", "min", "max", "trend"),
                    emit_labels=True,
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(StepRef("score_trace"),),
                    template="voice_assistant_axis_byot",
                    output_dir=f"papers/voice/assistant_axis/reports/{WORKFLOW_NAME}",
                ),
            ),
        ),
    )


def build_runner_specs() -> dict[str, object]:
    return runner_specs(workflow_name=os.getenv("ASSISTANT_AXIS_WORKFLOW_NAME", WORKFLOW_NAME))
