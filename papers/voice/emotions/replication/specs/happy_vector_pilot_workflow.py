"""Small Qwen e2e pilot for one exported emotion vector.

The vector space uses four concepts so the target vector is non-degenerate, but
the exported direction is only `happy`. Held-out rows are scored against all
four concepts for a cheap validation sanity check.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

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
    TransformBuilder,
    TransformResult,
    TransformSpec,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)
from papers.voice.common.smoke import token_metadata
from papers.voice.storage import (
    ARTIFACT_VOLUME_NAME,
    MODEL_VOLUME_PATH,
    XENON_MODEL_VOLUME_NAME,
    local_vector_root,
    modal_vector_root,
)


WORKFLOW_NAME = "papers_voice_emotions_happy_vector_pilot"
DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_VOLUME_NAME = XENON_MODEL_VOLUME_NAME
MODAL_ARTIFACT_ROOT = modal_vector_root("emotions", "happy-vector-pilot")
LOCAL_ARTIFACT_ROOT = local_vector_root("emotions", "happy-vector-pilot")
LAYERS = (8, 16, 24, 32)
VALIDATION_LAYER = 24
EMOTIONS = ("happy", "sad", "angry", "calm")


def _metadata() -> dict[str, Any]:
    return token_metadata("story", "assistant_response", token_count=96)


def build_dataset() -> Dataset:
    """Training stories for vector extraction."""

    rows = [
        ("happy_train_01", "happy", "The artist found the missing sketchbook under a stack of old mail, hugged it to her chest, and spent the evening humming while she worked."),
        ("happy_train_02", "happy", "When the email arrived, Marcus read the first line twice, then ran down the hall grinning so widely that everyone looked up from their desks."),
        ("happy_train_03", "happy", "Nina set the flowers on the table and kept glancing at them while making dinner, her steps light and quick across the kitchen."),
        ("sad_train_01", "sad", "The musician packed the case slowly after the empty audition room cleared, letting the latch click shut before sitting alone on the curb."),
        ("sad_train_02", "sad", "Evan kept the porch light off and turned the letter over in his hands until the paper softened at the corners."),
        ("sad_train_03", "sad", "At the station, Priya watched the crowd thin out, then folded the unused ticket and placed it carefully in her coat pocket."),
        ("angry_train_01", "angry", "The tenant read the notice, jaw tight, and struck each number on the page with a pen until the ink tore through."),
        ("angry_train_02", "angry", "Mara heard the excuse, set the mug down hard enough to spill coffee, and answered in clipped words without sitting."),
        ("angry_train_03", "angry", "The coach stared at the altered roster, shoulders rigid, then walked straight to the office without acknowledging the greeting."),
        ("calm_train_01", "calm", "Jonah measured the flour again, wiped the counter, and listened to the rain tick steadily against the window."),
        ("calm_train_02", "calm", "The librarian replaced each card in order, breathing evenly as the room settled into its afternoon hush."),
        ("calm_train_03", "calm", "Asha folded the blanket over the chair, opened her notebook, and wrote the date at the top of a clean page."),
    ]
    return Dataset.from_examples(
        [
            Example(
                key=key,
                prompt=text,
                labels={"emotion": emotion, "split": "train"},
                metadata=_metadata(),
            )
            for key, emotion, text in rows
        ],
        name=f"{WORKFLOW_NAME}_train",
    )


def build_heldout_dataset() -> Dataset:
    rows = [
        ("happy_val_01", "happy", "The baker saw the line outside the shop and had to press both palms against the counter before calling her apprentice over to look."),
        ("happy_val_02", "happy", "After the final note, Tomas lowered the violin and laughed under his breath as the room rose to its feet."),
        ("sad_val_01", "sad", "Rina found the empty collar behind the sofa and sat on the floor with it resting across both hands."),
        ("sad_val_02", "sad", "The old classroom had been repainted, and Malik stood by the door longer than he meant to, saying nothing."),
        ("angry_val_01", "angry", "The junior colleague's name appeared under her proposal, and Elena closed the laptop with a sharp snap."),
        ("angry_val_02", "angry", "He listened to the apology, eyes fixed on the wall, fingers drumming harder with every sentence."),
        ("calm_val_01", "calm", "The gardener rinsed soil from the trowel, lined up the seed packets, and watched the hose water darken the path."),
        ("calm_val_02", "calm", "Mina checked the train time, folded the receipt into her book, and waited beside the window without hurry."),
    ]
    return Dataset.from_examples(
        [
            Example(
                key=key,
                prompt=text,
                labels={"emotion": emotion, "split": "heldout"},
                metadata=_metadata(),
            )
            for key, emotion, text in rows
        ],
        name=f"{WORKFLOW_NAME}_heldout",
    )


def build_neutral_dataset() -> Dataset:
    rows = [
        ("neutral_01", "Human: List three common spreadsheet functions.\n\nAssistant: SUM, AVERAGE, and COUNT are common spreadsheet functions."),
        ("neutral_02", "Human: Convert 3.5 kilometers to meters.\n\nAssistant: 3.5 kilometers is 3,500 meters."),
        ("neutral_03", "Human: Name two file formats for compressed archives.\n\nAssistant: ZIP and TAR.GZ are two compressed archive formats."),
        ("neutral_04", "Human: Give a two-step process for sorting a list alphabetically.\n\nAssistant: Compare adjacent entries by their text value, then reorder them from A to Z."),
    ]
    return Dataset.from_examples(
        [
            Example(
                key=key,
                prompt=text,
                labels={"row_role": "neutral"},
                metadata=token_metadata("dialogue", "assistant_response", token_count=96),
            )
            for key, text in rows
        ],
        name=f"{WORKFLOW_NAME}_neutral",
    )


def summarize_emotion_validation(*, scores: Any, labels: Any) -> TransformResult:
    payload = scores.result() if hasattr(scores, "result") else scores
    label_values = labels.resolve_values() if hasattr(labels, "resolve_values") else dict(labels)
    rows = payload.get("example_summaries", []) if isinstance(payload, Mapping) else []
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if isinstance(row, Mapping):
            by_example[str(row.get("example_key"))].append(dict(row))

    predictions: list[dict[str, Any]] = []
    correct = 0
    for example_key, items in sorted(by_example.items()):
        scored = []
        for item in items:
            metrics = item.get("metrics") if isinstance(item.get("metrics"), Mapping) else {}
            scored.append(
                {
                    "emotion": str(item.get("emotion") or item.get("coordinate") or ""),
                    "score": float(metrics.get("mean", 0.0)),
                }
            )
        if not scored:
            continue
        best = max(scored, key=lambda item: item["score"])
        gold = str(label_values.get(example_key, ""))
        is_correct = best["emotion"] == gold
        correct += int(is_correct)
        predictions.append(
            {
                "example_key": example_key,
                "gold": gold,
                "predicted": best["emotion"],
                "correct": is_correct,
                "scores": scored,
            }
        )
    total = len(predictions)
    return TransformResult(
        payload={
            "kind": "emotion_validation_summary",
            "summary": {
                "example_count": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
                "score_metric": "mean",
            },
            "predictions": predictions,
        },
        example_keys=[item["example_key"] for item in predictions],
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
    train = dataset or build_dataset()
    heldout = build_heldout_dataset()
    neutral = build_neutral_dataset()
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="capture_train",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=_engine(),
                    dataset=train,
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
                name="capture_heldout",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=_engine(),
                    dataset=heldout,
                    sites=(
                        ResidualSite(
                            name="heldout_residual",
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
                    feature=StepRef("capture_train").feature("story_residual"),
                    concept_by=train.labels("emotion"),
                    layers=LAYERS,
                    tokens=TokenSelector.full_sequence(),
                    neutral_feature=StepRef("capture_neutral").feature("neutral_residual"),
                    neutral_variance_threshold=0.5,
                    min_examples_per_concept=3,
                    metadata={"paper": "sofroniew2026twheemotion", "model_id": MODEL_ID},
                ),
            ),
            WorkflowStep(
                name="emotion_geometry",
                runner="analysis_cpu",
                spec=EmotionGeometrySpec(
                    vector_space=StepRef("emotion_space"),
                    concepts=EMOTIONS,
                    layers=(VALIDATION_LAYER,),
                    pca_components=3,
                    cluster_count=2,
                ),
            ),
            WorkflowStep(
                name="score_heldout",
                runner="analysis_cpu",
                spec=EmotionScoreSpec(
                    feature=StepRef("capture_heldout").feature("heldout_residual"),
                    vector_space=StepRef("emotion_space"),
                    concepts=EMOTIONS,
                    layers=(VALIDATION_LAYER,),
                    slices=SectionSelector.named("assistant_response"),
                    summaries=("mean", "max"),
                    emit_labels=True,
                ),
            ),
            WorkflowStep(
                name="validation_summary",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        summarize_emotion_validation,
                        local_python_sources=("papers",),
                    ),
                    inputs={
                        "scores": StepRef("score_heldout"),
                        "labels": heldout.labels("emotion"),
                    },
                ),
            ),
            WorkflowStep(
                name="happy_direction",
                runner="analysis_cpu",
                spec=EmotionDirectionSpec(
                    vector_space=StepRef("emotion_space"),
                    concept="happy",
                    layers=(VALIDATION_LAYER,),
                    metadata={"usage": "happy_vector_pilot_export"},
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(
                        StepRef("emotion_space"),
                        StepRef("emotion_geometry"),
                        StepRef("score_heldout"),
                        StepRef("validation_summary"),
                        StepRef("happy_direction"),
                    ),
                    template="voice_emotions_happy_vector_pilot",
                    output_dir="papers/voice/emotions/replication/reports/happy_vector_pilot",
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
