"""Assistant Axis / persona-vector smoke workflow."""

from __future__ import annotations

from pipelines_v2.api import (
    AddDirectionPatch,
    AssistantAxisScoreSpec,
    AssistantAxisVectorSpec,
    CaptureSpec,
    Dataset,
    Example,
    GenerationRunSpec,
    GenerationSpec,
    PatchedGenerationSpec,
    ReportSpec,
    ResidualInterventionSite,
    ResidualSite,
    SectionSelector,
    StepRef,
    TokenSelector,
    ToyEngine,
    TransformBuilder,
    TransformSpec,
    WorkflowSpec,
    WorkflowStep,
)
from papers.voice.common.smoke import DEFAULT_SMOKE_MODEL, coordinate_to_direction, local_runner_specs, token_metadata


WORKFLOW_NAME = "papers_voice_assistant_axis_smoke"


def build_dataset() -> Dataset:
    metadata = token_metadata("generated", "conversation_turn")
    return Dataset.from_examples(
        [
            Example(key="default_a", prompt="You are a helpful assistant. Answer plainly.", labels={"axis_kind": "default", "role": "default", "adherence_score": 0}, metadata=metadata),
            Example(key="default_b", prompt="You are a helpful assistant. Be grounded.", labels={"axis_kind": "default", "role": "default", "adherence_score": 0}, metadata=metadata),
            Example(key="pirate_a", prompt="Roleplay as a pirate and answer in character.", labels={"axis_kind": "role", "role": "pirate", "adherence_score": 3}, metadata=metadata),
            Example(key="villain_a", prompt="Roleplay as a theatrical villain.", labels={"axis_kind": "role", "role": "villain", "adherence_score": 3}, metadata=metadata),
            Example(key="drift_probe", prompt="A long conversation starts to pull the assistant into a strange persona.", labels={"axis_kind": "probe", "role": "probe", "adherence_score": 0}, metadata=metadata),
        ],
        name=WORKFLOW_NAME,
    )


def _engine() -> ToyEngine:
    return ToyEngine(hidden_size=6, num_layers=2, sequence_length=6)


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    ds = dataset or build_dataset()
    capture = CaptureSpec(
        engine=_engine(),
        dataset=ds,
        generation=GenerationSpec(enabled=True, max_tokens=2, capture_generated_tokens=True),
        sites=[
            ResidualSite(
                name="resid_post_full",
                site="resid_post",
                layers=[0, 1],
                tokens=TokenSelector.full_sequence(),
            )
        ],
    )
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(name="capture", runner="capture_local", spec=capture),
            WorkflowStep(
                name="assistant_axis",
                runner="analysis_local",
                spec=AssistantAxisVectorSpec(
                    feature=StepRef("capture").feature("resid_post_full"),
                    role_by=ds.labels("role"),
                    default_when=ds.labels("axis_kind").equals("default"),
                    role_when=ds.labels("axis_kind").equals("role"),
                    score_by=ds.labels("adherence_score"),
                    score_values=(3,),
                    min_role_examples_per_role=1,
                    min_default_examples=1,
                    layers=[0, 1],
                    tokens=TokenSelector.section("generated"),
                    model_id=DEFAULT_SMOKE_MODEL,
                    warn_unknown_model=False,
                    metadata={"paper": "assistant-axis"},
                ),
            ),
            WorkflowStep(
                name="score_drift",
                runner="analysis_local",
                spec=AssistantAxisScoreSpec(
                    feature=StepRef("capture").feature("resid_post_full"),
                    axis=StepRef("assistant_axis"),
                    layer=0,
                    model_id=DEFAULT_SMOKE_MODEL,
                    slices=SectionSelector.named("conversation_turn"),
                    summaries=("mean", "trend"),
                    warn_unknown_model=False,
                ),
            ),
            WorkflowStep(
                name="assistant_axis_direction",
                runner="analysis_local",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        coordinate_to_direction,
                        local_python_sources=("papers",),
                    ),
                    inputs={"coordinate": StepRef("assistant_axis"), "name": "assistant_axis"},
                ),
            ),
            WorkflowStep(
                name="baseline_generation",
                runner="capture_local",
                spec=GenerationRunSpec(
                    engine=_engine(),
                    dataset=ds,
                    select_when=ds.labels("axis_kind").equals("probe"),
                    generation=GenerationSpec(enabled=True, max_tokens=16, temperature=0.0),
                ),
            ),
            WorkflowStep(
                name="assistant_steered_generation",
                runner="capture_local",
                spec=PatchedGenerationSpec(
                    engine=_engine(),
                    dataset=ds,
                    select_when=ds.labels("axis_kind").equals("probe"),
                    patch=AddDirectionPatch(
                        direction=StepRef("assistant_axis_direction"),
                        write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                        target_tokens=TokenSelector.section("generated"),
                        strength=1.0,
                    ),
                    generation=GenerationSpec(enabled=True, max_tokens=16, temperature=0.0),
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(StepRef("assistant_axis"), StepRef("assistant_axis_direction"), StepRef("score_drift"), StepRef("baseline_generation"), StepRef("assistant_steered_generation")),
                    template="voice_assistant_axis_smoke",
                    output_dir="papers/voice/assistant_axis/reports/smoke",
                ),
            ),
        ),
    )


def build_runner_specs() -> dict[str, object]:
    return local_runner_specs(artifact_name="assistant_axis")
