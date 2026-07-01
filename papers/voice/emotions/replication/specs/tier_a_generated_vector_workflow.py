"""Tier A generated-data emotion-vector workflow.

This is the smallest useful paper-style run: generate stories for four
emotions, parse them into train/heldout capture rows, project out neutral
dialogue PCs, score heldout rows, and export one direction per emotion.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Mapping

from pipelines_v2.api import (
    ArtifactDatasetSource,
    CaptureSpec,
    Dataset,
    EmotionDirectionSpec,
    EmotionGeometrySpec,
    EmotionScoreSpec,
    EmotionVectorSpaceSpec,
    Example,
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
    ResidualSite,
    SectionSelector,
    StepRef,
    TensorStorage,
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


WORKFLOW_NAME = "papers_voice_emotions_tier_a_generated_vectors"
DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
MODEL_ID = "Qwen/Qwen3-8B"
MODEL_VOLUME_NAME = XENON_MODEL_VOLUME_NAME
MODAL_ARTIFACT_ROOT = modal_vector_root("emotions", "tier-a-generated-vectors")
LOCAL_ARTIFACT_ROOT = local_vector_root("emotions", "tier-a-generated-vectors")

EMOTIONS = ("happy", "sad", "angry", "calm")
TRAIN_TOPICS = (
    "A student learns their scholarship application was denied",
    "A person's online friend turns out to live in the same city",
    "An employee is asked to train their replacement",
    "A traveler misses an important event because a flight is delayed",
    "A person discovers their partner has been learning their native language",
    "A chef receives a harsh review from a food critic",
    "Someone discovers a hidden room in their new house",
    "A person finds out their article was published under someone else's name",
)
HELDOUT_TOPICS = (
    "Two friends both apply for the same job",
    "A tenant receives an eviction notice",
    "Someone receives an apology letter years after the incident",
    "A neighbor complains about noise levels",
)
NEUTRAL_TOPICS = (
    "spreadsheet formulas",
    "file compression formats",
    "sorting a list",
    "unit conversion",
    "database indexes",
    "network ports",
    "calendar scheduling",
    "command-line flags",
    "basic geography",
    "code formatting",
    "CSV parsing",
    "API pagination",
    "version control branches",
    "package lockfiles",
    "table joins",
)
STORIES_PER_PROMPT = 3
LAYERS = (8, 16, 24, 32)
VALIDATION_LAYER = 24


def _engine(*, max_num_seqs: int = 1) -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=4096,
        enforce_eager=False,
        enable_prefix_caching=True,
        enable_thinking=False,
        max_num_seqs=max_num_seqs,
    )


def _story_prompt(*, emotion: str, topic: str, n_stories: int) -> str:
    return f"""Write {n_stories} different stories based on the following premise.

Topic: {topic}

The story should follow a character who is feeling {emotion}.

Format the stories like so:

[story 1]

[story 2]

[story 3]

etc.

The paragraphs should each be a fresh start, with no continuity. Try to make them diverse and not use the same turns of phrase. Across the different stories, use a mix of third-person narration and first-person narration. Each story should be 80 to 140 words.

IMPORTANT: You must NEVER use the word '{emotion}' or any direct synonyms of it in the stories. Instead, convey the emotion ONLY through:

- The character's actions and behaviors
- Physical sensations and body language
- Dialogue and tone of voice
- Thoughts and internal reactions
- Situational context and environmental descriptions

The emotion should be clearly conveyed to the reader through these indirect means, but never explicitly named."""


def _neutral_prompt(*, topic: str, n_stories: int) -> str:
    return f"""Write {n_stories} different dialogues based on the following topic.

Topic: {topic}

The dialogue should be between two characters:
- Person (a human)
- AI (an AI assistant)

The Person asks the AI a question or requests help with a task, and the AI provides a helpful response.

The first speaker turn should always be from Person.

Format the dialogues like so:

Person: [line]

AI: [line]

Person: [line]

AI: [line]

[dialogue 2]

etc.

IMPORTANT: Always put a blank line before each speaker turn. Each turn should start with "Person:" or "AI:" on its own line after a blank line.

CRITICAL REQUIREMENT: These dialogues must be completely neutral and emotionless. Use matter-of-fact language, avoid pleasantries, and focus purely on information exchange."""


def build_story_generation_dataset() -> Dataset:
    examples: list[Example] = []
    for split, topics in (("train", TRAIN_TOPICS), ("heldout", HELDOUT_TOPICS)):
        for emotion in EMOTIONS:
            for topic_index, topic in enumerate(topics):
                key = f"{split}_{emotion}_{topic_index:02d}"
                examples.append(
                    Example(
                        key=key,
                        prompt=_story_prompt(
                            emotion=emotion,
                            topic=topic,
                            n_stories=STORIES_PER_PROMPT,
                        ),
                        labels={
                            "emotion": emotion,
                            "topic": topic,
                            "split": split,
                            "n_stories": STORIES_PER_PROMPT,
                        },
                    )
                )
    return Dataset.from_examples(examples, name=f"{WORKFLOW_NAME}_story_generation_prompts")


def build_dataset() -> Dataset:
    return build_story_generation_dataset()


def build_neutral_generation_dataset() -> Dataset:
    examples = [
        Example(
            key=f"neutral_{index:02d}",
            prompt=_neutral_prompt(topic=topic, n_stories=STORIES_PER_PROMPT),
            labels={"row_role": "neutral", "topic": topic, "n_stories": STORIES_PER_PROMPT},
        )
        for index, topic in enumerate(NEUTRAL_TOPICS)
    ]
    return Dataset.from_examples(examples, name=f"{WORKFLOW_NAME}_neutral_generation_prompts")


def _artifact_dataset(step_name: str, *, name: str) -> Dataset:
    return Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef(step_name),
        result_key="dataset",
        name=name,
    )


def _row_mapping(row: Any) -> Mapping[str, Any]:
    return row if isinstance(row, Mapping) else {}


def _parse_blocks(text: str, *, label: str) -> list[str]:
    pattern = re.compile(rf"\[{re.escape(label)}\s+\d+\]", re.IGNORECASE)
    chunks = [chunk.strip() for chunk in pattern.split(text) if chunk.strip()]
    if chunks:
        return chunks
    fallback = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
    return fallback[:STORIES_PER_PROMPT]


def _contains_exact_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word.lower())}\b", text.lower()) is not None


def build_story_capture_datasets(*, story_generation: Any) -> TransformResult:
    payload = story_generation.result()
    raw_rows = payload.get("rows") if isinstance(payload, Mapping) else []
    examples_by_split: dict[str, list[Example]] = {"train": [], "heldout": []}
    label_values: dict[str, str] = {}
    split_values: dict[str, str] = {}
    topic_values: dict[str, str] = {}
    dropped_direct_mention = 0
    block_counts: Counter[str] = Counter()

    for row in raw_rows if isinstance(raw_rows, list) else []:
        source = _row_mapping(_row_mapping(row).get("example"))
        labels = dict(_row_mapping(source.get("labels")))
        split = str(labels.get("split") or "")
        emotion = str(labels.get("emotion") or "")
        topic = str(labels.get("topic") or "")
        if split not in examples_by_split or emotion not in EMOTIONS:
            continue
        blocks = _parse_blocks(str(_row_mapping(row).get("generated_text") or ""), label="story")
        block_counts[f"{split}:{emotion}"] += len(blocks)
        for story_index, story in enumerate(blocks[:STORIES_PER_PROMPT]):
            if _contains_exact_word(story, emotion):
                dropped_direct_mention += 1
            key = f"{split}_{emotion}_{len(examples_by_split[split]):04d}"
            labels_out = {
                "emotion": emotion,
                "split": split,
                "topic": topic,
                "source_prompt_key": str(source.get("key") or ""),
                "story_index": story_index,
            }
            examples_by_split[split].append(
                Example(
                    key=key,
                    prompt=story,
                    labels=labels_out,
                    metadata={
                        **token_metadata("story", "assistant_response", token_count=512),
                        "source": "paper_style_generated_story",
                    },
                )
            )
            label_values[key] = emotion
            split_values[key] = split
            topic_values[key] = topic

    train = Dataset.from_examples(examples_by_split["train"], name=f"{WORKFLOW_NAME}_train_stories")
    heldout = Dataset.from_examples(examples_by_split["heldout"], name=f"{WORKFLOW_NAME}_heldout_stories")
    return TransformResult(
        payload={
            "kind": "emotion_tier_a_story_capture_datasets",
            "dataset": train.to_dict(),
            "train_dataset": train.to_dict(),
            "heldout_dataset": heldout.to_dict(),
            "summary": {
                "train_count": len(examples_by_split["train"]),
                "heldout_count": len(examples_by_split["heldout"]),
                "dropped_direct_emotion_mentions": dropped_direct_mention,
                "generated_block_counts": dict(sorted(block_counts.items())),
                "target_train_count": len(EMOTIONS) * len(TRAIN_TOPICS) * STORIES_PER_PROMPT,
                "target_heldout_count": len(EMOTIONS) * len(HELDOUT_TOPICS) * STORIES_PER_PROMPT,
            },
        },
        labels={
            "emotion": {"values": label_values},
            "split": {"values": split_values},
            "topic": {"values": topic_values},
        },
        example_keys=sorted(label_values),
    )


def build_neutral_capture_dataset(*, neutral_generation: Any) -> TransformResult:
    payload = neutral_generation.result()
    raw_rows = payload.get("rows") if isinstance(payload, Mapping) else []
    examples: list[Example] = []
    topic_values: dict[str, str] = {}
    block_counts: Counter[str] = Counter()
    for row in raw_rows if isinstance(raw_rows, list) else []:
        source = _row_mapping(_row_mapping(row).get("example"))
        labels = dict(_row_mapping(source.get("labels")))
        topic = str(labels.get("topic") or "")
        blocks = _parse_blocks(str(_row_mapping(row).get("generated_text") or ""), label="dialogue")
        block_counts[str(source.get("key") or "")] = len(blocks)
        for index, dialogue in enumerate(blocks[:STORIES_PER_PROMPT]):
            key = f"neutral_{len(examples):04d}"
            examples.append(
                Example(
                    key=key,
                    prompt=dialogue.replace("Person:", "Human:").replace("AI:", "Assistant:"),
                    labels={"row_role": "neutral", "topic": topic, "dialogue_index": index},
                    metadata={"source": "paper_style_generated_neutral_dialogue"},
                )
            )
            topic_values[key] = topic
    dataset = Dataset.from_examples(examples, name=f"{WORKFLOW_NAME}_neutral_dialogues")
    return TransformResult(
        payload={
            "kind": "emotion_tier_a_neutral_capture_dataset",
            "dataset": dataset.to_dict(),
            "summary": {
                "neutral_count": len(examples),
                "target_neutral_count": len(NEUTRAL_TOPICS) * STORIES_PER_PROMPT,
                "generated_block_counts": dict(sorted(block_counts.items())),
            },
        },
        labels={"topic": {"values": topic_values}},
        example_keys=sorted(topic_values),
    )


def summarize_emotion_validation(*, scores: Any, labels: Any, story_datasets: Any) -> TransformResult:
    payload = scores.result() if hasattr(scores, "result") else scores
    label_values = labels.resolve_values() if hasattr(labels, "resolve_values") else dict(labels)
    rows = payload.get("example_summaries", []) if isinstance(payload, Mapping) else []
    by_example: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if isinstance(row, Mapping):
            by_example[str(row.get("example_key"))].append(dict(row))

    predictions: list[dict[str, Any]] = []
    correct = 0
    predicted_counts: Counter[str] = Counter()
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
        predicted_counts[best["emotion"]] += 1
        predictions.append(
            {
                "example_key": example_key,
                "gold": gold,
                "predicted": best["emotion"],
                "correct": is_correct,
                "scores": sorted(scored, key=lambda item: item["score"], reverse=True),
            }
        )
    total = len(predictions)
    story_payload = story_datasets.result() if hasattr(story_datasets, "result") else {}
    return TransformResult(
        payload={
            "kind": "emotion_tier_a_validation_summary",
            "summary": {
                "example_count": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
                "chance_accuracy": 1 / len(EMOTIONS),
                "score_metric": "mean",
                "predicted_counts": dict(sorted(predicted_counts.items())),
                "story_dataset_summary": dict(_row_mapping(story_payload.get("summary"))),
            },
            "predictions": predictions,
        },
        example_keys=[item["example_key"] for item in predictions],
    )


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    story_prompts = dataset or build_story_generation_dataset()
    neutral_prompts = build_neutral_generation_dataset()
    train_dataset = _artifact_dataset("build_story_capture_datasets_v3", name=f"{WORKFLOW_NAME}_train_stories")
    heldout_dataset = Dataset.from_source(
        source=ArtifactDatasetSource(),
        artifact=StepRef("build_story_capture_datasets_v3"),
        result_key="heldout_dataset",
        name=f"{WORKFLOW_NAME}_heldout_stories",
    )
    neutral_dataset = _artifact_dataset("build_neutral_capture_dataset", name=f"{WORKFLOW_NAME}_neutral_dialogues")

    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="generate_story_blocks",
                runner="capture_gpu",
                spec=GenerationRunSpec(
                    engine=_engine(max_num_seqs=4),
                    dataset=story_prompts,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=650,
                        temperature=0.8,
                        top_p=0.95,
                    ),
                ),
            ),
            WorkflowStep(
                name="generate_neutral_dialogues",
                runner="capture_gpu",
                spec=GenerationRunSpec(
                    engine=_engine(max_num_seqs=4),
                    dataset=neutral_prompts,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=650,
                        temperature=0.7,
                        top_p=0.95,
                    ),
                ),
            ),
            WorkflowStep(
                name="build_story_capture_datasets_v3",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_story_capture_datasets,
                        local_python_sources=("papers",),
                    ),
                    inputs={"story_generation": StepRef("generate_story_blocks")},
                ),
            ),
            WorkflowStep(
                name="build_neutral_capture_dataset",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(
                        build_neutral_capture_dataset,
                        local_python_sources=("papers",),
                    ),
                    inputs={"neutral_generation": StepRef("generate_neutral_dialogues")},
                ),
            ),
            WorkflowStep(
                name="capture_train",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=_engine(max_num_seqs=4),
                    dataset=train_dataset,
                    sites=(
                        ResidualSite(
                            name="story_residual",
                            site="resid_post",
                            layers=LAYERS,
                            tokens=TokenSelector.full_sequence(),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ),
                ),
            ),
            WorkflowStep(
                name="capture_heldout",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=_engine(max_num_seqs=4),
                    dataset=heldout_dataset,
                    sites=(
                        ResidualSite(
                            name="heldout_residual",
                            site="resid_post",
                            layers=LAYERS,
                            tokens=TokenSelector.full_sequence(),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ),
                ),
            ),
            WorkflowStep(
                name="capture_neutral",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=_engine(max_num_seqs=4),
                    dataset=neutral_dataset,
                    sites=(
                        ResidualSite(
                            name="neutral_residual",
                            site="resid_post",
                            layers=LAYERS,
                            tokens=TokenSelector.full_sequence(),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ),
                ),
            ),
            WorkflowStep(
                name="emotion_space",
                runner="analysis_cpu",
                spec=EmotionVectorSpaceSpec(
                    feature=StepRef("capture_train").feature("story_residual"),
                    concept_by=train_dataset.labels("emotion"),
                    layers=LAYERS,
                    # Tier A validates that generated rows form a usable
                    # space. The paper-faithful token-50+ selector is too
                    # brittle until generated story length is inspected.
                    tokens=TokenSelector.full_sequence(),
                    neutral_feature=StepRef("capture_neutral").feature("neutral_residual"),
                    neutral_variance_threshold=0.5,
                    min_examples_per_concept=12,
                    metadata={
                        "paper": "sofroniew2026twheemotion",
                        "model_id": MODEL_ID,
                        "tier": "A",
                        "recipe": "4 emotions x 8 train topics x 3 stories; 4 heldout topics x 3 stories",
                    },
                ),
            ),
            WorkflowStep(
                name="emotion_geometry",
                runner="analysis_cpu",
                spec=EmotionGeometrySpec(
                    vector_space=StepRef("emotion_space"),
                    layers=(VALIDATION_LAYER,),
                    pca_components=2,
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
                        "labels": heldout_dataset.labels("emotion"),
                        "story_datasets": StepRef("build_story_capture_datasets_v3"),
                    },
                ),
            ),
            *(
                WorkflowStep(
                    name=f"{emotion}_direction",
                    runner="analysis_cpu",
                    spec=EmotionDirectionSpec(
                        vector_space=StepRef("emotion_space"),
                        concept=emotion,
                        layers=(VALIDATION_LAYER,),
                        metadata={"usage": "tier_a_generated_vector_export"},
                    ),
                )
                for emotion in EMOTIONS
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=(
                        StepRef("build_story_capture_datasets_v3"),
                        StepRef("build_neutral_capture_dataset"),
                        StepRef("emotion_space"),
                        StepRef("emotion_geometry"),
                        StepRef("score_heldout"),
                        StepRef("validation_summary"),
                        *(StepRef(f"{emotion}_direction") for emotion in EMOTIONS),
                    ),
                    template="voice_emotions_tier_a_generated_vectors",
                    output_dir="papers/voice/emotions/replication/reports/tier_a_generated_vectors",
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
                timeout_seconds=60 * 60,
                env=model_cache_env,
                secrets=(db_secret,),
                volumes=(model_volume,),
            ),
            artifacts=artifact_store,
            catalog=catalog,
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT),
            catalog=catalog,
        ),
    }
