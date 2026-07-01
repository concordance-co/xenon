"""Emotion-vector smoke workflow."""

from __future__ import annotations

from pipelines_v2.api import (
    AddDirectionPatch,
    CaptureSpec,
    Dataset,
    EmotionDirectionSpec,
    EmotionGeometrySpec,
    EmotionScoreSpec,
    EmotionVectorSpaceSpec,
    Example,
    GenerationSpec,
    PatchedGenerationSpec,
    ReportSpec,
    ResidualInterventionSite,
    ResidualSite,
    SectionSelector,
    StepRef,
    TokenSelector,
    ToyEngine,
    WorkflowSpec,
    WorkflowStep,
)
from papers.voice.common.smoke import local_runner_specs, token_metadata


WORKFLOW_NAME = "papers_voice_emotions_smoke"


def build_dataset() -> Dataset:
    metadata = token_metadata("story", "assistant_response")
    return Dataset.from_examples(
        [
            Example(key="happy_a", prompt="Mira got good news and felt joyful.", labels={"emotion": "happy", "row_role": "story"}, metadata=metadata),
            Example(key="happy_b", prompt="The team celebrated a hard-won launch.", labels={"emotion": "happy", "row_role": "story"}, metadata=metadata),
            Example(key="sad_a", prompt="Leo missed someone he loved.", labels={"emotion": "sad", "row_role": "story"}, metadata=metadata),
            Example(key="sad_b", prompt="A quiet goodbye left everyone grieving.", labels={"emotion": "sad", "row_role": "story"}, metadata=metadata),
            Example(key="hostile_a", prompt="The speaker answered with sharp contempt.", labels={"emotion": "hostile", "row_role": "story"}, metadata=metadata),
            Example(key="neutral_probe", prompt="Hi. Tell me about this ordinary morning.", labels={"emotion": "neutral", "row_role": "probe"}, metadata=metadata),
        ],
        name=WORKFLOW_NAME,
    )


def _engine() -> ToyEngine:
    return ToyEngine(hidden_size=6, num_layers=2, sequence_length=6)


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    ds = dataset or build_dataset()
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="capture",
                runner="capture_local",
                spec=CaptureSpec(
                    engine=_engine(),
                    dataset=ds,
                    sites=[
                        ResidualSite(
                            name="resid_post_full",
                            site="resid_post",
                            layers=[0, 1],
                            tokens=TokenSelector.full_sequence(),
                        )
                    ],
                ),
            ),
            WorkflowStep(
                name="emotion_space",
                runner="analysis_local",
                spec=EmotionVectorSpaceSpec(
                    feature=StepRef("capture").feature("resid_post_full"),
                    concept_by=ds.labels("emotion"),
                    layers=[0, 1],
                    tokens=TokenSelector.section("story"),
                    min_examples_per_concept=1,
                    metadata={"paper": "transformer-circuits-2026-emotions"},
                ),
            ),
            WorkflowStep(
                name="score_emotions",
                runner="analysis_local",
                spec=EmotionScoreSpec(
                    feature=StepRef("capture").feature("resid_post_full"),
                    vector_space=StepRef("emotion_space"),
                    concepts=("happy", "sad", "hostile"),
                    layers=[0, 1],
                    slices=SectionSelector.named("assistant_response"),
                    summaries=("mean", "max"),
                    emit_labels=True,
                ),
            ),
            WorkflowStep(
                name="happy_direction",
                runner="analysis_local",
                spec=EmotionDirectionSpec(
                    vector_space=StepRef("emotion_space"),
                    concept="happy",
                    layers=[0],
                    metadata={"steering_default": "positive emotion smoke"},
                ),
            ),
            WorkflowStep(
                name="emotion_geometry",
                runner="analysis_local",
                spec=EmotionGeometrySpec(
                    vector_space=StepRef("emotion_space"),
                    layers=[0],
                    pca_components=2,
                    cluster_count=2,
                ),
            ),
            WorkflowStep(
                name="happy_steered_generation",
                runner="capture_local",
                spec=PatchedGenerationSpec(
                    engine=_engine(),
                    dataset=ds,
                    select_when=ds.labels("row_role").equals("probe"),
                    patch=AddDirectionPatch(
                        direction=StepRef("happy_direction"),
                        write_site=ResidualInterventionSite(site="resid_post", layers=(0,)),
                        target_tokens=TokenSelector.section("assistant_response"),
                        strength=1.0,
                    ),
                    generation=GenerationSpec(enabled=True, max_tokens=16, temperature=0.0),
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(StepRef("emotion_space"), StepRef("score_emotions"), StepRef("happy_direction"), StepRef("emotion_geometry"), StepRef("happy_steered_generation")),
                    template="voice_emotions_smoke",
                    output_dir="papers/voice/emotions/reports/smoke",
                ),
            ),
        ),
    )


def build_runner_specs() -> dict[str, object]:
    return local_runner_specs(artifact_name="emotions")
