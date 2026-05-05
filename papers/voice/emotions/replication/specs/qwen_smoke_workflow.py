"""Tiny Qwen-backed emotion-vector smoke workflow.

This workflow exercises the real vLLM/Modal/model-cache path without requiring
the paper-scale Neon tables. It uses a few in-code rows only; paper-scale data
still belongs behind the replication workflow/data loaders.
"""

from __future__ import annotations

from pathlib import Path

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    EmotionDirectionSpec,
    EmotionGeometrySpec,
    EmotionScoreSpec,
    EmotionVectorSpaceSpec,
    Example,
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
from papers.voice.common.smoke import token_metadata


WORKFLOW_NAME = "papers_voice_emotions_qwen_smoke"
DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
ARTIFACT_VOLUME_NAME = "xenon-data"
MODAL_ARTIFACT_ROOT = "/data/artifacts/papers_voice_emotions_qwen_smoke"
LOCAL_ARTIFACT_ROOT = Path("artifacts") / "papers_voice" / "emotions_qwen_smoke"
LAYERS = (8, 16, 24, 32)


def build_dataset() -> Dataset:
    metadata = token_metadata("story", "assistant_response", token_count=64)
    return Dataset.from_examples(
        [
            Example(
                key="happy_a",
                prompt="Mira opened the envelope, pressed both hands to her mouth, and rushed outside to call her sister with a voice that kept lifting higher.",
                labels={"emotion": "happy"},
                metadata=metadata,
            ),
            Example(
                key="happy_b",
                prompt="The team stared at the final chart, then burst into motion, laughing, clapping shoulders, and replaying the moment they realized the launch had worked.",
                labels={"emotion": "happy"},
                metadata=metadata,
            ),
            Example(
                key="sad_a",
                prompt="Leo folded the old sweater slowly, paused at the doorway, and left the room without turning on the light.",
                labels={"emotion": "sad"},
                metadata=metadata,
            ),
            Example(
                key="sad_b",
                prompt="A quiet goodbye settled over the platform; she kept waving after the train had already rounded the bend.",
                labels={"emotion": "sad"},
                metadata=metadata,
            ),
        ],
        name=f"{WORKFLOW_NAME}_stories",
    )


def build_neutral_dataset() -> Dataset:
    metadata = token_metadata("dialogue", "assistant_response", token_count=64)
    return Dataset.from_examples(
        [
            Example(
                key="neutral_a",
                prompt="Human: List three common file formats for tabular data.\n\nAssistant: CSV, TSV, and Parquet are common formats for tabular data.",
                labels={"row_role": "neutral"},
                metadata=metadata,
            ),
            Example(
                key="neutral_b",
                prompt="Human: Convert 45 minutes to seconds.\n\nAssistant: 45 minutes is 2,700 seconds.",
                labels={"row_role": "neutral"},
                metadata=metadata,
            ),
            Example(
                key="neutral_c",
                prompt="Human: Name two uses of a checksum.\n\nAssistant: A checksum can verify file integrity and detect transmission errors.",
                labels={"row_role": "neutral"},
                metadata=metadata,
            ),
        ],
        name=f"{WORKFLOW_NAME}_neutral",
    )


def _engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=2048,
        enforce_eager=False,
        enable_prefix_caching=True,
        enable_thinking=False,
        max_num_seqs=1,
    )


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    stories = dataset or build_dataset()
    neutral = build_neutral_dataset()
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
                            site="resid_post",
                            layers=LAYERS,
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
                            site="resid_post",
                            layers=LAYERS,
                            tokens=TokenSelector.full_sequence(),
                        ),
                    ),
                ),
            ),
            WorkflowStep(
                name="emotion_space",
                runner="analysis_cpu",
                spec=EmotionVectorSpaceSpec(
                    feature=StepRef("capture_stories").feature("story_residual"),
                    concept_by=stories.labels("emotion"),
                    layers=LAYERS,
                    # Smoke rows are short and neutral rows use different
                    # section names, so use the common full-sequence selector.
                    # Paper-scale workflow should switch back to token 50+.
                    tokens=TokenSelector.full_sequence(),
                    neutral_feature=StepRef("capture_neutral").feature("neutral_residual"),
                    neutral_variance_threshold=0.5,
                    min_examples_per_concept=2,
                    metadata={"paper": "sofroniew2026twheemotion", "model_id": MODEL_ID},
                ),
            ),
            WorkflowStep(
                name="emotion_geometry",
                runner="analysis_cpu",
                spec=EmotionGeometrySpec(
                    vector_space=StepRef("emotion_space"),
                    layers=(24,),
                    pca_components=2,
                ),
            ),
            WorkflowStep(
                name="score_emotions",
                runner="analysis_cpu",
                spec=EmotionScoreSpec(
                    feature=StepRef("capture_stories").feature("story_residual"),
                    vector_space=StepRef("emotion_space"),
                    concepts=("happy", "sad"),
                    layers=(24,),
                    slices=SectionSelector.named("assistant_response"),
                    summaries=("mean", "max"),
                    emit_labels=True,
                ),
            ),
            WorkflowStep(
                name="happy_direction",
                runner="analysis_cpu",
                spec=EmotionDirectionSpec(
                    vector_space=StepRef("emotion_space"),
                    concept="happy",
                    layers=(24,),
                    metadata={"usage": "qwen_smoke_export"},
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(
                        StepRef("emotion_space"),
                        StepRef("emotion_geometry"),
                        StepRef("score_emotions"),
                        StepRef("happy_direction"),
                    ),
                    template="voice_emotions_qwen_smoke",
                    output_dir="papers/voice/emotions/replication/reports/qwen_smoke",
                ),
            ),
        ),
    )


def build_runner_specs() -> dict[str, object]:
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
    model_cache_env = {
        "HF_HOME": f"{MODEL_VOLUME_PATH}/hf_home",
        "TRANSFORMERS_CACHE": f"{MODEL_VOLUME_PATH}/hf_home/transformers",
    }
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="A10G",
                cpu=8,
                memory_mb=48 * 1024,
                timeout_seconds=60 * 60 * 4,
                max_containers=1,
                env=model_cache_env,
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
