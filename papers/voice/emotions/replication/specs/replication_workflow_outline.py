"""Paper-scale emotion-vector replication workflow outline.

This file is intentionally a TODO scaffold, not the default runnable smoke
workflow. Fill the placeholders from the paper and then decide whether to turn
this into an executable workflow or copy the completed pieces into
`papers/voice/emotions/specs/workflow.py`.
"""

from __future__ import annotations

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    EmotionDirectionSpec,
    EmotionGeometrySpec,
    EmotionScoreSpec,
    EmotionVectorSpaceSpec,
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
    ResidualSite,
    SectionSelector,
    StepRef,
    TokenSelector,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)
from pathlib import Path


WORKFLOW_NAME = "papers_voice_emotions_replication_todo"
DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
ARTIFACT_VOLUME_NAME = "xenon-data"
MODAL_ARTIFACT_ROOT = "/data/artifacts/papers_voice_emotions_replication"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "papers_voice" / "emotions_replication"

# Defaults copied from configs/replication.todo.toml. Keep that file as the
# planning surface and update these constants when the replication is ready to
# run.
STORY_TABLE = "papers_voice_emotions_story_generation_v1"
NEUTRAL_TABLE = "papers_voice_emotions_neutral_dialogues_v1"
TARGET_MODEL_ID = "Qwen/Qwen3-8B"
RESIDUAL_SITE = "resid_post"
CAPTURE_LAYERS: tuple[int, ...] = (8, 16, 24, 32)
GEOMETRY_LAYERS: tuple[int, ...] = (24,)
SCORE_EMOTIONS: tuple[str, ...] = ("happy", "sad", "afraid", "angry", "calm")
EXPORT_EMOTION = "happy"


def build_story_dataset() -> Dataset:
    """TODO: Replace with the real story dataset loader."""

    return Dataset.from_postgres(
        source=PostgresSource.from_env(DB_ENV_VAR),
        table=STORY_TABLE,
        prompt_column="text",
        example_key_column="example_id",
        label_columns=("emotion",),
        case_columns=("topic",),
        case_key_column="topic",
        metadata_columns=("topic", "story_index"),
        name=f"{WORKFLOW_NAME}_stories",
    )


def build_dataset() -> Dataset:
    """Default CLI dataset: the emotional-story corpus."""

    return build_story_dataset()


def build_neutral_dataset() -> Dataset:
    """TODO: Replace with the real neutral-transcript dataset loader."""

    return Dataset.from_postgres(
        source=PostgresSource.from_env(DB_ENV_VAR),
        table=NEUTRAL_TABLE,
        prompt_column="text",
        example_key_column="example_id",
        case_columns=("topic",),
        case_key_column="topic",
        metadata_columns=("topic",),
        name=f"{WORKFLOW_NAME}_neutral",
    )


def _engine() -> VLLMEngine:
    """Return the cheap first-pass capture engine.

    TODO: prewarm the model into /models/Qwen/Qwen3-8B on xenon-models, or
    switch TARGET_MODEL_ID to an already-mounted model.
    """

    return VLLMEngine(
        model_id=TARGET_MODEL_ID,
        enable_prefix_caching=False,
        model_path_root=MODEL_VOLUME_PATH,
        max_model_len=8192,
        enforce_eager=False,
        max_num_seqs=8,
        enable_thinking=False,
    )


def build_workflow(
    story_dataset: Dataset | None = None,
    neutral_dataset: Dataset | None = None,
) -> WorkflowSpec:
    stories = story_dataset or build_story_dataset()
    neutral = neutral_dataset or build_neutral_dataset()

    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="capture_stories",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=_engine(),
                    dataset=stories,
                    sites=(
                        ResidualSite(
                            name="story_residual",
                            site=RESIDUAL_SITE,
                            layers=CAPTURE_LAYERS,
                            tokens=TokenSelector.full_sequence(),
                        ),
                    ),
                ),
            ),
            WorkflowStep(
                name="capture_neutral",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=_engine(),
                    dataset=neutral,
                    sites=(
                        ResidualSite(
                            name="neutral_residual",
                            site=RESIDUAL_SITE,
                            layers=CAPTURE_LAYERS,
                            tokens=TokenSelector.full_sequence(),
                        ),
                    ),
                ),
            ),
            WorkflowStep(
                name="emotion_vector_space",
                runner="analysis_cpu",
                spec=EmotionVectorSpaceSpec(
                    feature=StepRef("capture_stories").feature("story_residual"),
                    concept_by=stories.labels("emotion"),
                    layers=CAPTURE_LAYERS,
                    # Paper-style story recipe: average from token 50 onward.
                    # TODO: confirm exact token window.
                    tokens=TokenSelector.slice(50, None),
                    neutral_feature=StepRef("capture_neutral").feature("neutral_residual"),
                    # TODO: confirm exact neutral projection threshold.
                    neutral_variance_threshold=0.5,
                    normalize="l2",
                    min_examples_per_concept=1,
                    vector_space_kind="story",
                    metadata={
                        "paper": "TODO",
                        "target_model_id": TARGET_MODEL_ID,
                    },
                ),
            ),
            WorkflowStep(
                name="emotion_geometry",
                runner="analysis_cpu",
                spec=EmotionGeometrySpec(
                    vector_space=StepRef("emotion_vector_space"),
                    layers=GEOMETRY_LAYERS,
                    pca_components=3,
                    # TODO: set cluster_count or leave None.
                    cluster_count=None,
                ),
            ),
            WorkflowStep(
                name="heldout_emotion_scores",
                runner="analysis_cpu",
                spec=EmotionScoreSpec(
                    # TODO: replace with held-out capture if scoring a separate
                    # dataset instead of the training stories.
                    feature=StepRef("capture_stories").feature("story_residual"),
                    vector_space=StepRef("emotion_vector_space"),
                    concepts=SCORE_EMOTIONS,
                    layers=GEOMETRY_LAYERS,
                    slices=SectionSelector.all(),
                    summaries=("mean", "max", "std"),
                    emit_labels=True,
                ),
            ),
            WorkflowStep(
                name="export_emotion_direction",
                runner="analysis_cpu",
                spec=EmotionDirectionSpec(
                    vector_space=StepRef("emotion_vector_space"),
                    concept=EXPORT_EMOTION,
                    layers=GEOMETRY_LAYERS,
                    source="vector",
                    scale=1.0,
                    metadata={
                        "paper": "TODO",
                        "usage": "TODO: steering/readout/export",
                    },
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(
                        StepRef("emotion_vector_space"),
                        StepRef("emotion_geometry"),
                        StepRef("heldout_emotion_scores"),
                        StepRef("export_emotion_direction"),
                    ),
                    template="voice_emotions_replication_todo",
                    output_dir="papers/voice/emotions/replication/reports/vector_space",
                ),
            ),
        ),
    )


def build_runner_specs() -> dict[str, object]:
    """Return Modal runners wired to Neon/catalog and persistent volumes."""

    db = PostgresSource.from_env(DB_ENV_VAR)
    catalog = PostgresCatalog(source=db)
    db_secret = ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon")
    artifact_store = ModalVolumeStore(name=ARTIFACT_VOLUME_NAME, root=MODAL_ARTIFACT_ROOT)
    model_volume = ModalVolumeMount(
        name=MODEL_VOLUME_NAME,
        mount_path=MODEL_VOLUME_PATH,
        create_if_missing=True,
        commit_on_success=True,
    )
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="A10G",
                cpu=8,
                memory_mb=48 * 1024,
                timeout_seconds=60 * 60 * 4,
                max_containers=1,
                secrets=(db_secret,),
                volumes=(model_volume,),
            ),
            artifacts=artifact_store,
            catalog=catalog,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=8,
                memory_mb=32 * 1024,
                timeout_seconds=60 * 60 * 2,
                max_containers=1,
                secrets=(db_secret,),
            ),
            artifacts=artifact_store,
            catalog=catalog,
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
            catalog=catalog,
        ),
    }
