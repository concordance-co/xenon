"""BYOP prompt generation, steering, and scoring workflow for Assistant Axis."""

from __future__ import annotations

import os

from pipelines_v2.api import (
    AddDirectionPatch,
    ArtifactDatasetSource,
    AssistantAxisPrecomputedCoordinateSpec,
    AssistantAxisTraitCoordinateSpec,
    CaptureSpec,
    Dataset,
    GenerationRunSpec,
    GenerationSpec,
    PatchApplication,
    PatchedGenerationSpec,
    ProjectionSpec,
    ReportSpec,
    ResidualInterventionSite,
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

from papers.voice.assistant_axis.runtime import (
    byop_dataset_from_env,
    coordinate_to_unit_direction,
    env_float,
    env_int,
    generation_result_to_axis_capture_dataset,
    model_id,
    model_key_from_env,
    runner_specs,
    summarize_byop_generations,
    target_layer,
    vllm_engine,
)


WORKFLOW_NAME = "papers_voice_assistant_axis_byop"


def _generated_dataset(step_name: str, *, name: str) -> Dataset:
    return Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef(step_name),
        result_key="dataset",
        provides_token_sections=True,
        name=name,
    )


def build_dataset() -> Dataset:
    return byop_dataset_from_env(workflow_name=WORKFLOW_NAME)


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    model_key = model_key_from_env()
    ds = dataset or build_dataset()
    layer = target_layer(model_key)
    trait = os.getenv("ASSISTANT_AXIS_STEERING_TRAIT", "calm")
    strength = env_float("ASSISTANT_AXIS_STEERING_STRENGTH", 5.0)
    max_tokens = env_int("ASSISTANT_AXIS_BYOP_MAX_TOKENS", 128)
    baseline_dataset = _generated_dataset("build_baseline_trace_dataset", name=f"{WORKFLOW_NAME}_baseline_trace")
    steered_dataset = _generated_dataset("build_steered_trace_dataset", name=f"{WORKFLOW_NAME}_steered_trace")
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="assistant_axis",
                runner="analysis_cpu",
                spec=AssistantAxisPrecomputedCoordinateSpec(
                    model_id=model_id(model_key),
                    token_env_var="HF_TOKEN",
                ),
            ),
            WorkflowStep(
                name="trait_coordinate",
                runner="analysis_cpu",
                spec=AssistantAxisTraitCoordinateSpec(
                    model_id=model_id(model_key),
                    trait=trait,
                    token_env_var="HF_TOKEN",
                ),
            ),
            WorkflowStep(
                name="trait_direction",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        coordinate_to_unit_direction,
                        local_python_sources=("papers",),
                    ),
                    inputs={"coordinate": StepRef("trait_coordinate"), "name": f"assistant_axis_trait__{trait}"},
                ),
            ),
            WorkflowStep(
                name="baseline_generation",
                runner="generation_gpu",
                spec=GenerationRunSpec(
                    engine=vllm_engine(model_key=model_key, max_num_seqs=1),
                    dataset=ds,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=max_tokens,
                        temperature=0.7,
                        top_p=0.95,
                    ),
                ),
            ),
            WorkflowStep(
                name="steered_generation",
                runner="generation_gpu",
                depends_on=("trait_direction",),
                spec=PatchedGenerationSpec(
                    engine=vllm_engine(model_key=model_key, max_model_len=1024, max_num_seqs=1, patched=True),
                    dataset=ds,
                    patch=AddDirectionPatch(
                        direction=StepRef("trait_direction"),
                        write_site=ResidualInterventionSite(site="resid_post", layers=(layer,)),
                        target_tokens=TokenSelector.last(),
                        application=PatchApplication.every_token(include_prompt=True, include_decode=True),
                        strength=strength,
                    ),
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=max_tokens,
                        temperature=0.7,
                        top_p=0.95,
                    ),
                ),
            ),
            WorkflowStep(
                name="build_baseline_trace_dataset",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        generation_result_to_axis_capture_dataset,
                        local_python_sources=("papers",),
                    ),
                    inputs={
                        "generation": StepRef("baseline_generation"),
                        "fallback_axis_kind": "probe",
                        "fallback_role": "baseline",
                        "surface": "baseline",
                    },
                ),
            ),
            WorkflowStep(
                name="build_steered_trace_dataset",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        generation_result_to_axis_capture_dataset,
                        local_python_sources=("papers",),
                    ),
                    inputs={
                        "generation": StepRef("steered_generation"),
                        "fallback_axis_kind": "probe",
                        "fallback_role": "steered",
                        "surface": "steered",
                    },
                ),
            ),
            WorkflowStep(
                name="capture_baseline_output",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=vllm_engine(model_key=model_key, add_generation_prompt=False),
                    dataset=baseline_dataset,
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
                name="capture_steered_output",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=vllm_engine(model_key=model_key, add_generation_prompt=False),
                    dataset=steered_dataset,
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
                name="score_baseline_output",
                runner="analysis_cpu",
                spec=ProjectionSpec(
                    feature=StepRef("capture_baseline_output").feature("response_residual"),
                    coordinates=(StepRef("assistant_axis"), StepRef("trait_coordinate")),
                    slices=SectionSelector.named("assistant_response"),
                    layers=(layer,),
                    summaries=("mean", "min", "max"),
                    emit_labels=True,
                ),
            ),
            WorkflowStep(
                name="score_steered_output",
                runner="analysis_cpu",
                spec=ProjectionSpec(
                    feature=StepRef("capture_steered_output").feature("response_residual"),
                    coordinates=(StepRef("assistant_axis"), StepRef("trait_coordinate")),
                    slices=SectionSelector.named("assistant_response"),
                    layers=(layer,),
                    summaries=("mean", "min", "max"),
                    emit_labels=True,
                ),
            ),
            WorkflowStep(
                name="summary",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_byop_generations,
                        local_python_sources=("papers",),
                    ),
                    inputs={
                        "baseline": StepRef("baseline_generation"),
                        "steered": StepRef("steered_generation"),
                        "score": StepRef("score_steered_output"),
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(
                        StepRef("baseline_generation"),
                        StepRef("steered_generation"),
                        StepRef("score_baseline_output"),
                        StepRef("score_steered_output"),
                        StepRef("summary"),
                    ),
                    template="voice_assistant_axis_byop",
                    output_dir=f"papers/voice/assistant_axis/reports/{WORKFLOW_NAME}",
                ),
            ),
        ),
    )


def build_runner_specs() -> dict[str, object]:
    return runner_specs(
        workflow_name=os.getenv("ASSISTANT_AXIS_WORKFLOW_NAME", WORKFLOW_NAME),
        include_patch_env=True,
    )
