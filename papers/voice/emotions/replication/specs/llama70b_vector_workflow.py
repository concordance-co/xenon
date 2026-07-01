"""Llama 3.3 70B emotion-vector asset workflow.

Pilot mode validates plumbing with four emotions. Full mode uses all 171
paper emotions and is the intended demo-grade vector-space derivation.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
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
    TensorStorage,
    TokenPooling,
    TokenSelector,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    TransferPolicy,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)

from papers.voice.common.smoke import token_metadata
from papers.voice.emotions.assets import concept_key, direction_step_name, emotion_concepts, emotion_topics
from papers.voice.storage import (
    ARTIFACT_VOLUME_NAME,
    MODEL_VOLUME_PATH,
    YORA_MODEL_VOLUME_NAME,
    local_vector_root,
    modal_vector_root,
)


WORKFLOW_NAME = "papers_voice_emotions_llama33_70b_vectors"
DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_KEY = "llama_3_3_70b"
MODEL_VOLUME_NAME = YORA_MODEL_VOLUME_NAME
MODAL_ARTIFACT_ROOT = modal_vector_root("emotions", "llama-3.3-70b", "sofroniew-2026", "v1")
LOCAL_ARTIFACT_ROOT = local_vector_root("emotions", "llama-3.3-70b", "sofroniew-2026", "v1")
PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"
PILOT_CONCEPTS = ("happy", "sad", "angry", "calm")
DEFAULT_PILOT_LAYERS = (32, 40, 48)
DEFAULT_FULL_LAYERS = (16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68)
DEFAULT_PILOT_VALIDATION_LAYER = 40
DEFAULT_FULL_VALIDATION_LAYER = 52
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_GENERATED_ROWS_TABLE = "papers_voice_emotions_llama70b_generated_rows_v1"


def mode() -> str:
    value = os.getenv("EMOTION_ASSET_MODE", "pilot").strip().lower()
    if value not in {"pilot", "full"}:
        raise ValueError("EMOTION_ASSET_MODE must be 'pilot' or 'full'")
    return value


def selected_emotions() -> tuple[str, ...]:
    override = _env_list("EMOTION_ASSET_EMOTIONS")
    if override:
        return override
    return emotion_concepts(mode=mode())


def _emotion_subset_requested() -> bool:
    return bool(_env_list("EMOTION_ASSET_EMOTIONS"))


def selected_train_topics() -> tuple[str, ...]:
    topics = emotion_topics()
    count = _env_int("EMOTION_ASSET_TRAIN_TOPIC_COUNT", 8 if mode() == "pilot" else len(topics))
    return tuple(topics[:count])


def selected_heldout_topics() -> tuple[str, ...]:
    if mode() == "full":
        return ()
    topics = emotion_topics()
    train_count = len(selected_train_topics())
    count = _env_int("EMOTION_ASSET_HELDOUT_TOPIC_COUNT", 4 if mode() == "pilot" else 0)
    return tuple(topics[train_count : train_count + count])


def selected_neutral_topics() -> tuple[str, ...]:
    topics = emotion_topics()
    count = _env_int("EMOTION_ASSET_NEUTRAL_TOPIC_COUNT", 12 if mode() == "pilot" else len(topics))
    return tuple(topics[:count])


def stories_per_prompt() -> int:
    return _env_int("EMOTION_ASSET_STORIES_PER_PROMPT", 3 if mode() == "pilot" else 12)


def capture_layers() -> tuple[int, ...]:
    default = DEFAULT_FULL_LAYERS if mode() == "full" else DEFAULT_PILOT_LAYERS
    return _env_int_tuple("EMOTION_ASSET_LAYERS", default)


def validation_layer() -> int:
    default = DEFAULT_FULL_VALIDATION_LAYER if mode() == "full" else DEFAULT_PILOT_VALIDATION_LAYER
    return _env_int("EMOTION_ASSET_VALIDATION_LAYER", default)


def token_selector() -> TokenSelector:
    selector = os.getenv("EMOTION_ASSET_TOKEN_SELECTOR", "").strip().lower()
    if selector == "full_sequence" or (not selector and mode() == "pilot"):
        return TokenSelector.full_sequence()
    if selector in {"token_50", "token_50_plus", "slice_50"} or not selector:
        return TokenSelector.slice(50, None)
    raise ValueError("EMOTION_ASSET_TOKEN_SELECTOR must be full_sequence or token_50")


def capture_pooling() -> TokenPooling | None:
    value = os.getenv("EMOTION_ASSET_CAPTURE_POOLING", "").strip().lower()
    if not value:
        value = "mean" if mode() == "full" else "none"
    if value in {"", "none", "off", "false", "0"}:
        return None
    if value == "mean":
        return TokenPooling.mean()
    if value == "last":
        return TokenPooling.last()
    if value == "first":
        return TokenPooling.first()
    raise ValueError("EMOTION_ASSET_CAPTURE_POOLING must be mean, first, last, or none")


def vector_space_token_selector() -> TokenSelector:
    return TokenSelector.full_sequence() if capture_pooling() is not None else token_selector()


def min_capture_chars() -> int:
    explicit = os.getenv("EMOTION_ASSET_MIN_CAPTURE_CHARS")
    if explicit is not None:
        return int(explicit)
    selector = token_selector()
    if selector.kind == "slice" and int(selector.value["start"]) >= 50:
        return 300
    return 0


def capture_row_limit() -> int | None:
    limit = _env_optional_int("EMOTION_ASSET_CAPTURE_ROW_LIMIT")
    if limit is None or limit <= 0:
        return None
    if mode() == "full" and not _emotion_subset_requested():
        raise ValueError(
            "EMOTION_ASSET_CAPTURE_ROW_LIMIT is not allowed for a full production vector run. "
            "Unset it to process the complete generated dataset, or set EMOTION_ASSET_EMOTIONS "
            "for an explicit subset smoke."
        )
    minimum = len(selected_emotions()) * min_examples_per_concept()
    if mode() == "full" and limit < minimum:
        raise ValueError(
            "EMOTION_ASSET_CAPTURE_ROW_LIMIT is too small for a full balanced vector run: "
            f"got {limit}, need at least {minimum} rows "
            f"({len(selected_emotions())} emotions * {min_examples_per_concept()} min examples). "
            "For a smaller smoke, set EMOTION_ASSET_EMOTIONS to a concept subset."
        )
    return limit


def min_examples_per_concept() -> int:
    default = stories_per_prompt() if mode() == "pilot" else 12
    return _env_int("EMOTION_ASSET_MIN_EXAMPLES_PER_CONCEPT", default)


def use_heldout_validation() -> bool:
    return bool(selected_heldout_topics())


def capture_max_num_seqs() -> int:
    return _env_int("EMOTION_ASSET_CAPTURE_MAX_NUM_SEQS", 512 if mode() == "full" else 1)


def _engine(*, max_num_seqs: int = 1) -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        model_path_root=MODEL_VOLUME_PATH,
        tensor_parallel_size=_env_int("EMOTION_ASSET_TENSOR_PARALLEL_SIZE", 2),
        max_model_len=_env_int("EMOTION_ASSET_MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN),
        gpu_memory_utilization=float(os.getenv("EMOTION_ASSET_GPU_MEMORY_UTILIZATION", "0.95")),
        enforce_eager=False,
        enable_prefix_caching=True,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=_env_optional_int("EMOTION_ASSET_MAX_NUM_BATCHED_TOKENS") or (8192 if mode() == "full" else None),
        enable_chunked_prefill=_env_bool("EMOTION_ASSET_ENABLE_CHUNKED_PREFILL", True),
        enable_thinking=False,
    )


def generated_rows_table() -> str:
    return os.getenv("EMOTION_ASSET_GENERATED_ROWS_TABLE", DEFAULT_GENERATED_ROWS_TABLE).strip() or DEFAULT_GENERATED_ROWS_TABLE


@lru_cache(maxsize=None)
def _prompt_template(filename: str) -> str:
    return (PROMPT_ROOT / filename).read_text(encoding="utf-8").strip()


def _story_prompt(*, emotion: str, topic: str, n_stories: int) -> str:
    return _prompt_template("emotional_stories.md").format(
        emotion=emotion,
        topic=topic,
        n_stories=n_stories,
    )


def _neutral_prompt(*, topic: str, n_dialogues: int) -> str:
    return _prompt_template("neutral_transcripts.md").format(
        topic=topic,
        n_stories=n_dialogues,
    )


def build_story_generation_dataset() -> Dataset:
    examples: list[Example] = []
    n_stories = stories_per_prompt()
    for split, topics in (("train", selected_train_topics()), ("heldout", selected_heldout_topics())):
        for emotion in selected_emotions():
            for topic_index, topic in enumerate(topics):
                examples.append(
                    Example(
                        key=f"{split}_{concept_key(emotion)}_{topic_index:03d}",
                        prompt=_story_prompt(emotion=emotion, topic=topic, n_stories=n_stories),
                        labels={"emotion": emotion, "topic": topic, "split": split, "n_stories": n_stories},
                    )
                )
    return Dataset.from_examples(examples, name=f"{WORKFLOW_NAME}_{mode()}_story_generation_prompts")


def build_dataset() -> Dataset:
    return build_curated_story_capture_dataset(split="train")


def build_neutral_generation_dataset() -> Dataset:
    n_stories = stories_per_prompt()
    return Dataset.from_examples(
        [
            Example(
                key=f"neutral_{index:03d}",
                prompt=_neutral_prompt(topic=topic, n_dialogues=n_stories),
                labels={"row_role": "neutral", "topic": topic, "n_stories": n_stories},
            )
            for index, topic in enumerate(selected_neutral_topics())
        ],
        name=f"{WORKFLOW_NAME}_{mode()}_neutral_generation_prompts",
    )


def build_curated_story_capture_dataset(*, split: str) -> Dataset:
    if split not in {"train", "heldout"}:
        raise ValueError("split must be 'train' or 'heldout'")
    return Dataset.from_postgres(
        source=PostgresSource.from_env(DB_ENV_VAR),
        sql=_curated_story_sql(split=split),
        prompt_column="text",
        example_key_column="example_id",
        prompt_hash_column="text_sha256",
        label_columns=("emotion", "split", "topic", "source_prompt_key", "block_index"),
        case_columns=("topic", "source_prompt_key"),
        case_key_column="source_prompt_key",
        metadata_columns=("row_role", "topic", "source_prompt_key", "block_index", "section_records"),
        name=f"{WORKFLOW_NAME}_{mode()}_{split}_stories",
    )


def build_curated_neutral_capture_dataset() -> Dataset:
    return Dataset.from_postgres(
        source=PostgresSource.from_env(DB_ENV_VAR),
        sql=_curated_neutral_sql(),
        prompt_column="text",
        example_key_column="example_id",
        prompt_hash_column="text_sha256",
        label_columns=("row_role", "split", "topic", "source_prompt_key", "block_index"),
        case_columns=("topic", "source_prompt_key"),
        case_key_column="source_prompt_key",
        metadata_columns=("row_role", "topic", "source_prompt_key", "block_index", "section_records"),
        name=f"{WORKFLOW_NAME}_{mode()}_neutral_dialogues",
    )


def _curated_story_sql(*, split: str) -> str:
    emotion_filter = ", ".join(_sql_literal(emotion) for emotion in selected_emotions())
    min_chars_filter = _min_capture_chars_sql()
    split_filter = _story_split_sql(split)
    limit = capture_row_limit()
    if limit is not None:
        per_emotion_limit = math.ceil(limit / len(selected_emotions()))
        return f"""
            WITH candidate_rows AS (
                SELECT
                    example_id,
                    text,
                    text_sha256,
                    row_role,
                    split,
                    emotion,
                    topic,
                    source_prompt_key,
                    block_index,
                    {_assistant_response_section_sql()} AS section_records,
                    row_number() OVER (
                        PARTITION BY emotion
                        ORDER BY source_prompt_key, block_index, example_id
                    ) AS emotion_row_number
                FROM {_generated_rows_relation()}
                WHERE mode = {_sql_literal(mode())}
                  AND row_role = 'story'
                  {split_filter}
                  AND emotion IN ({emotion_filter})
                  {min_chars_filter}
            )
            SELECT
                example_id,
                text,
                text_sha256,
                row_role,
                split,
                emotion,
                topic,
                source_prompt_key,
                block_index,
                section_records
            FROM candidate_rows
            WHERE emotion_row_number <= {int(per_emotion_limit)}
            ORDER BY emotion_row_number, emotion, source_prompt_key, block_index
            LIMIT {int(limit)}
        """
    return f"""
        SELECT
            example_id,
            text,
            text_sha256,
            row_role,
            split,
            emotion,
            topic,
            source_prompt_key,
            block_index,
            {_assistant_response_section_sql()} AS section_records
        FROM {_generated_rows_relation()}
        WHERE mode = {_sql_literal(mode())}
          AND row_role = 'story'
          {split_filter}
          AND emotion IN ({emotion_filter})
          {min_chars_filter}
        ORDER BY emotion, source_prompt_key, block_index
    """


def _curated_neutral_sql() -> str:
    min_chars_filter = _min_capture_chars_sql()
    return f"""
        SELECT
            example_id,
            text,
            text_sha256,
            row_role,
            split,
            emotion,
            topic,
            source_prompt_key,
            block_index,
            {_assistant_response_section_sql()} AS section_records
        FROM {_generated_rows_relation()}
        WHERE mode = {_sql_literal(mode())}
          AND row_role = 'neutral'
          {min_chars_filter}
        ORDER BY source_prompt_key, block_index
        {_capture_row_limit_sql()}
    """


def _min_capture_chars_sql() -> str:
    minimum = min_capture_chars()
    if minimum <= 0:
        return ""
    return f"AND char_length(text) >= {int(minimum)}"


def _capture_row_limit_sql() -> str:
    limit = capture_row_limit()
    if limit is None:
        return ""
    return f"LIMIT {int(limit)}"


def _story_split_sql(split: str) -> str:
    if split == "train" and not use_heldout_validation():
        return ""
    return f"AND split = {_sql_literal(split)}"


def _assistant_response_section_sql() -> str:
    return (
        "jsonb_build_array(jsonb_build_object("
        "'name', 'assistant_response', "
        "'role', 'assistant', "
        "'unit', 'row', "
        "'index', 0, "
        "'char_start', 0, "
        "'char_end', char_length(text)"
        "))"
    )


def _generated_rows_relation() -> str:
    table = generated_rows_table()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?", table):
        raise ValueError("EMOTION_ASSET_GENERATED_ROWS_TABLE must be an identifier or schema-qualified identifier")
    return ".".join(f'"{part}"' for part in table.split("."))


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _row_mapping(row: Any) -> Mapping[str, Any]:
    return row if isinstance(row, Mapping) else {}


def _parse_blocks(text: str, *, label: str) -> list[str]:
    label_pattern = re.compile(rf"(?im)^\s*\[?\s*{re.escape(label)}\s+\d+\s*\]?\s*$")
    if label_pattern.search(text):
        parts = label_pattern.split(text)
        if parts and parts[0].strip():
            parts = parts[1:]
        chunks = [chunk.strip() for chunk in parts if chunk.strip()]
        return _dialogue_chunks(chunks) if label == "dialogue" else _story_chunks(chunks)
    if label == "dialogue":
        heading_chunks = _dialogue_heading_chunks(text)
        if heading_chunks:
            return heading_chunks
    return [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()][: stories_per_prompt()]


def _story_chunks(chunks: list[str]) -> list[str]:
    return [cleaned for chunk in chunks for cleaned in [_trim_story_meta(chunk)] if cleaned]


def _trim_story_meta(chunk: str) -> str:
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", chunk.strip()) if paragraph.strip()]
    kept: list[str] = []
    for paragraph in paragraphs:
        if _looks_like_story_meta_paragraph(paragraph):
            break
        trimmed = _trim_inline_story_meta(paragraph)
        if trimmed:
            kept.append(trimmed)
        if trimmed != paragraph:
            break
    return "\n\n".join(kept).strip()


def _trim_inline_story_meta(paragraph: str) -> str:
    lower = paragraph.lower()
    starts: list[int] = []

    let_me_know = lower.find("let me know if")
    if let_me_know >= 0 and any(
        marker in lower[let_me_know:]
        for marker in ("requirement", "stories", "instruction", "response", "revise", "adjustment", "improve", "feedback")
    ):
        starts.append(let_me_know)

    i_hope = lower.find("i hope")
    if i_hope >= 0:
        tail = lower[i_hope:]
        if (
            "i hope you like them" in tail
            or "i hope this is accurate" in tail
            or ("i hope these" in tail and any(marker in tail for marker in ("requirement", "prompt", "accurate", "adjustment")))
        ):
            starts.append(i_hope)

    completed = lower.find("i have now completed")
    if completed >= 0 and "as requested" in lower[completed:]:
        starts.append(completed)

    for marker in (
        "also, are there any other instructions",
        "also, please let me know",
        "please provide feedback",
        "thanks! (i have made",
        "please disregard",
        "the stories aim",
        "the stories explore",
        "to further enhance the stories",
    ):
        position = lower.find(marker)
        if position >= 0:
            starts.append(position)

    if not starts:
        return paragraph
    return paragraph[: min(starts)].strip()


def _looks_like_story_meta_paragraph(paragraph: str) -> bool:
    lines = [line.strip() for line in paragraph.strip().splitlines() if line.strip()]
    if not lines:
        return False
    first = lines[0]
    normalized = " ".join(lines).lower()
    if re.match(r"(?i)^(?:please\s+)?note(?:\s+that)?\b", first):
        return True
    if re.match(r"(?i)^here are\b", first) and "stor" in normalized:
        return True
    if re.match(r"(?i)^as requested\b", first) and "stor" in normalized:
        return True
    if re.match(r"(?i)^i hope\b", first) and any(
        marker in normalized for marker in ("requirements", "accurate", "adjustments", "prompt")
    ):
        return True
    if re.match(r"(?i)^please find\b", first) and "response" in normalized:
        return True
    if re.match(r"(?i)^let me know\b", first) and any(marker in normalized for marker in ("prompt", "proceed")):
        return True
    if re.match(r"(?i)^i can do this prompt\b", first):
        return True
    if "without using the word" in normalized and any(
        marker in normalized for marker in ("direct synonym", "convey the emotion", "requirements")
    ):
        return True
    if "direct synonym" in normalized and "convey" in normalized and "emotion" in normalized:
        return True
    return False


def _dialogue_chunks(chunks: list[str]) -> list[str]:
    return [
        cleaned
        for chunk in chunks
        for cleaned in [_strip_dialogue_heading(_trim_trailing_empty_speaker(chunk))]
        if re.search(r"(?im)^\s*Person\s*:", cleaned)
        and re.search(r"(?im)^\s*AI\s*:", cleaned)
        and re.search(r"(?ims)^\s*Person\s*:\s*\S.*?^\s*AI\s*:\s*\S", cleaned)
    ]


def _strip_dialogue_heading(chunk: str) -> str:
    lines = chunk.strip().splitlines()
    first = _next_nonempty_line_index(lines, 0)
    if first is None:
        return ""
    heading = lines[first].strip()
    if re.match(r"(?i)^(?:dialogue\s+)?\d+$", heading):
        del lines[first]
    return "\n".join(lines).strip()


def _trim_trailing_empty_speaker(chunk: str) -> str:
    lines = chunk.strip().splitlines()
    last = _previous_nonempty_line_index(lines, len(lines) - 1)
    if last is None:
        return ""
    if re.match(r"(?i)^\s*AI\s*:\s*$", lines[last]):
        previous_person = None
        for index in range(last - 1, -1, -1):
            if re.match(r"(?i)^\s*Person\s*:", lines[index]):
                previous_person = index
                break
        if previous_person is not None:
            lines = lines[:previous_person]
    elif re.match(r"(?i)^\s*Person\s*:\s*$", lines[last]):
        lines = lines[:last]
    return "\n".join(lines).strip()


def _dialogue_heading_chunks(text: str) -> list[str]:
    lines = text.splitlines()
    starts: list[int] = []
    for index, line in enumerate(lines):
        heading = line.strip()
        if not _looks_like_dialogue_heading(heading):
            continue
        next_line_index = _next_nonempty_line_index(lines, index + 1)
        if next_line_index is not None and re.match(r"(?i)^\s*System\s*:", lines[next_line_index]):
            next_line_index = _next_nonempty_line_index(lines, next_line_index + 1)
        if next_line_index is None or not re.match(r"(?i)^\s*Person\s*:", lines[next_line_index]):
            continue
        window = "\n".join(lines[next_line_index + 1 : next_line_index + 20])
        if re.search(r"(?im)^\s*AI\s*:", window):
            starts.append(index)
    chunks: list[str] = []
    for start_index, start in enumerate(starts):
        stop = starts[start_index + 1] if start_index + 1 < len(starts) else len(lines)
        chunks.append("\n".join(lines[start:stop]).strip())
    return _dialogue_chunks(chunks)


def _looks_like_dialogue_heading(text: str) -> bool:
    if not text:
        return False
    if re.match(r"(?i)^(person|ai|human|assistant|system)\s*:", text):
        return False
    if re.match(r"(?i)^here are\b", text):
        return False
    if text.startswith(("-", "*")):
        return False
    if re.match(r"^\d+[\.\)]\s+\S", text):
        return False
    return len(text) <= 220


def _next_nonempty_line_index(lines: list[str], start: int) -> int | None:
    for index in range(start, len(lines)):
        if lines[index].strip():
            return index
    return None


def _previous_nonempty_line_index(lines: list[str], start: int) -> int | None:
    for index in range(start, -1, -1):
        if lines[index].strip():
            return index
    return None


def _contains_exact_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word.lower())}\b", text.lower()) is not None


def build_story_capture_datasets(*, story_generation: Any) -> TransformResult:
    payload = story_generation.result()
    raw_rows = payload.get("rows") if isinstance(payload, Mapping) else []
    allowed_emotions = set(selected_emotions())
    examples_by_split: dict[str, list[Example]] = {"train": [], "heldout": []}
    label_values: dict[str, str] = {}
    split_values: dict[str, str] = {}
    topic_values: dict[str, str] = {}
    direct_mentions = 0
    block_counts: Counter[str] = Counter()

    for row in raw_rows if isinstance(raw_rows, list) else []:
        source = _row_mapping(_row_mapping(row).get("example"))
        labels = dict(_row_mapping(source.get("labels")))
        split = str(labels.get("split") or "")
        emotion = str(labels.get("emotion") or "")
        topic = str(labels.get("topic") or "")
        if split not in examples_by_split or emotion not in allowed_emotions:
            continue
        blocks = _parse_blocks(str(_row_mapping(row).get("generated_text") or ""), label="story")
        block_counts[f"{split}:{emotion}"] += len(blocks)
        for story_index, story in enumerate(blocks[: stories_per_prompt()]):
            if _contains_exact_word(story, emotion):
                direct_mentions += 1
            key = f"{split}_{concept_key(emotion)}_{len(examples_by_split[split]):06d}"
            examples_by_split[split].append(
                Example(
                    key=key,
                    prompt=story,
                    labels={
                        "emotion": emotion,
                        "split": split,
                        "topic": topic,
                        "source_prompt_key": str(source.get("key") or ""),
                        "story_index": story_index,
                    },
                    metadata={
                        **token_metadata("story", "assistant_response", token_count=512),
                        "source": "llama70b_generated_story",
                    },
                )
            )
            label_values[key] = emotion
            split_values[key] = split
            topic_values[key] = topic

    train = Dataset.from_examples(examples_by_split["train"], name=f"{WORKFLOW_NAME}_{mode()}_train_stories")
    heldout = Dataset.from_examples(examples_by_split["heldout"], name=f"{WORKFLOW_NAME}_{mode()}_heldout_stories")
    return TransformResult(
        payload={
            "kind": "emotion_llama70b_story_capture_datasets",
            "dataset": train.to_dict(),
            "train_dataset": train.to_dict(),
            "heldout_dataset": heldout.to_dict(),
            "summary": {
                "mode": mode(),
                "emotion_count": len(allowed_emotions),
                "train_count": len(examples_by_split["train"]),
                "heldout_count": len(examples_by_split["heldout"]),
                "direct_emotion_mentions": direct_mentions,
                "generated_block_counts": dict(sorted(block_counts.items())),
                "target_train_count": len(allowed_emotions) * len(selected_train_topics()) * stories_per_prompt(),
                "target_heldout_count": len(allowed_emotions) * len(selected_heldout_topics()) * stories_per_prompt(),
            },
        },
        labels={"emotion": {"values": label_values}, "split": {"values": split_values}, "topic": {"values": topic_values}},
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
        for index, dialogue in enumerate(blocks[: stories_per_prompt()]):
            key = f"neutral_{len(examples):06d}"
            examples.append(
                Example(
                    key=key,
                    prompt=dialogue.replace("Person:", "Human:").replace("AI:", "Assistant:"),
                    labels={"row_role": "neutral", "topic": topic, "dialogue_index": index},
                    metadata={"source": "llama70b_generated_neutral_dialogue"},
                )
            )
            topic_values[key] = topic
    dataset = Dataset.from_examples(examples, name=f"{WORKFLOW_NAME}_{mode()}_neutral_dialogues")
    return TransformResult(
        payload={
            "kind": "emotion_llama70b_neutral_capture_dataset",
            "dataset": dataset.to_dict(),
            "summary": {
                "mode": mode(),
                "neutral_count": len(examples),
                "target_neutral_count": len(selected_neutral_topics()) * stories_per_prompt(),
                "generated_block_counts": dict(sorted(block_counts.items())),
            },
        },
        labels={"topic": {"values": topic_values}},
        example_keys=sorted(topic_values),
    )


def summarize_emotion_validation(*, scores: Any, labels: Any, story_datasets: Any | None = None) -> TransformResult:
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
            scored.append({"emotion": str(item.get("emotion") or item.get("coordinate") or ""), "score": float(metrics.get("mean", 0.0))})
        if not scored:
            continue
        best = max(scored, key=lambda item: item["score"])
        gold = str(label_values.get(example_key, ""))
        is_correct = best["emotion"] == gold
        correct += int(is_correct)
        predicted_counts[best["emotion"]] += 1
        predictions.append({"example_key": example_key, "gold": gold, "predicted": best["emotion"], "correct": is_correct, "scores": sorted(scored, key=lambda item: item["score"], reverse=True)})
    total = len(predictions)
    story_payload = story_datasets.result() if story_datasets is not None and hasattr(story_datasets, "result") else {}
    return TransformResult(
        payload={
            "kind": "emotion_llama70b_validation_summary",
            "summary": {
                "mode": mode(),
                "example_count": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
                "chance_accuracy": 1 / len(selected_emotions()),
                "score_metric": "mean",
                "predicted_counts": dict(sorted(predicted_counts.items())),
                "story_dataset_summary": dict(_row_mapping(story_payload.get("summary"))),
            },
            "predictions": predictions,
        },
        example_keys=[item["example_key"] for item in predictions],
    )


def build_workflow() -> WorkflowSpec:
    emotions = selected_emotions()
    layers = capture_layers()
    read_layer = validation_layer()
    site_pooling = capture_pooling()
    site_tokens = token_selector() if site_pooling is not None else TokenSelector.full_sequence()
    train_dataset = build_curated_story_capture_dataset(split="train")
    heldout_dataset = build_curated_story_capture_dataset(split="heldout") if use_heldout_validation() else None
    neutral_dataset = build_curated_neutral_capture_dataset()
    report_inputs: tuple[Any, ...] = (
        StepRef("emotion_space"),
        StepRef("emotion_geometry"),
    )
    steps: list[WorkflowStep] = [
        WorkflowStep(
            name="capture_train",
            runner="capture_gpu",
            spec=CaptureSpec(
                engine=_engine(max_num_seqs=capture_max_num_seqs()),
                dataset=train_dataset,
                sites=(
                    ResidualSite(
                        name="story_residual",
                        site="resid_post",
                        layers=layers,
                        tokens=site_tokens,
                        pooling=site_pooling,
                        storage=TensorStorage(dtype="float16", format="safetensors"),
                    ),
                ),
            ),
        ),
    ]
    if heldout_dataset is not None:
        steps.append(
            WorkflowStep(
                name="capture_heldout",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=_engine(max_num_seqs=capture_max_num_seqs()),
                    dataset=heldout_dataset,
                    sites=(
                        ResidualSite(
                            name="heldout_residual",
                            site="resid_post",
                            layers=layers,
                            tokens=site_tokens,
                            pooling=site_pooling,
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ),
                ),
            )
        )
    steps.extend(
        [
            WorkflowStep(
                name="capture_neutral",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=_engine(max_num_seqs=capture_max_num_seqs()),
                    dataset=neutral_dataset,
                    sites=(
                        ResidualSite(
                            name="neutral_residual",
                            site="resid_post",
                            layers=layers,
                            tokens=site_tokens,
                            pooling=site_pooling,
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
                    layers=layers,
                    tokens=vector_space_token_selector(),
                    neutral_feature=StepRef("capture_neutral").feature("neutral_residual"),
                    neutral_variance_threshold=float(os.getenv("EMOTION_ASSET_NEUTRAL_VARIANCE_THRESHOLD", "0.5")),
                    min_examples_per_concept=min_examples_per_concept(),
                    metadata={
                        "asset_id": "emotions-llama-3.3-70b-sofroniew-2026-v1",
                        "paper": "sofroniew2026twheemotion",
                        "model_id": MODEL_ID,
                        "model_key": MODEL_KEY,
                        "mode": mode(),
                        "emotion_count": len(emotions),
                        "generated_rows_table": generated_rows_table(),
                        "generated_rows_source": "neon",
                        "min_capture_chars": min_capture_chars(),
                    },
                ),
            ),
            WorkflowStep(name="emotion_geometry", runner="analysis_cpu", spec=EmotionGeometrySpec(vector_space=StepRef("emotion_space"), concepts=emotions, layers=(read_layer,), pca_components=3, cluster_count=None)),
        ]
    )
    if heldout_dataset is not None:
        validation_inputs = {"scores": StepRef("score_heldout"), "labels": heldout_dataset.labels("emotion")}
        steps.extend(
            [
                WorkflowStep(name="score_heldout", runner="analysis_cpu", spec=EmotionScoreSpec(feature=StepRef("capture_heldout").feature("heldout_residual"), vector_space=StepRef("emotion_space"), concepts=emotions, layers=(read_layer,), slices=SectionSelector.named("assistant_response"), summaries=("mean", "max"), emit_labels=True)),
                WorkflowStep(
                    name="validation_summary",
                    runner="analysis_cpu",
                    spec=TransformSpec(
                        builder=TransformBuilder.from_function(summarize_emotion_validation, local_python_sources=("papers",)),
                        inputs=validation_inputs,
                    ),
                ),
            ]
        )
        report_inputs = (
            *report_inputs,
            StepRef("score_heldout"),
            StepRef("validation_summary"),
        )
    report_inputs = (
        *report_inputs,
        *(StepRef(direction_step_name(emotion)) for emotion in emotions),
    )
    steps.extend(
        [
            *(
                WorkflowStep(
                    name=direction_step_name(emotion),
                    runner="analysis_cpu",
                    spec=EmotionDirectionSpec(vector_space=StepRef("emotion_space"), concept=emotion, layers=(read_layer,), metadata={"usage": "llama70b_emotion_vector_export", "mode": mode()}),
                )
                for emotion in emotions
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=report_inputs,
                    template="voice_emotions_llama70b_vectors",
                    output_dir=f"papers/voice/emotions/replication/reports/{WORKFLOW_NAME}",
                ),
            ),
        ]
    )

    return WorkflowSpec(name=WORKFLOW_NAME, steps=tuple(steps))


def build_runner_specs() -> dict[str, object]:
    db = PostgresSource.from_env(DB_ENV_VAR)
    catalog = PostgresCatalog(source=db)
    db_secret = ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon")
    artifact_store = ModalVolumeStore(
        name=ARTIFACT_VOLUME_NAME,
        root=MODAL_ARTIFACT_ROOT,
        transfer_policy=TransferPolicy(allow_large_transfer=True),
    )
    model_volume = ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH, create_if_missing=True, commit_on_success=True)
    shared_env = {
        "HF_HOME": f"{MODEL_VOLUME_PATH}/hf_home",
        "TRANSFORMERS_CACHE": f"{MODEL_VOLUME_PATH}/hf_home/transformers",
        "VLLM_CACHE_ROOT": os.getenv("EMOTION_ASSET_VLLM_CACHE_ROOT", MODEL_VOLUME_PATH),
        "TORCHINDUCTOR_CACHE_DIR": f"{MODEL_VOLUME_PATH}/torch_compile_cache",
        **_workflow_env(),
    }
    capture_shard_count = _env_optional_int("EMOTION_ASSET_CAPTURE_SHARD_COUNT")
    if capture_shard_count is None and mode() == "full":
        capture_shard_count = 4
    capture_max_containers = _env_optional_int("EMOTION_ASSET_CAPTURE_MAX_CONTAINERS") or capture_shard_count or 1
    capture_timeout = _env_int("EMOTION_ASSET_CAPTURE_TIMEOUT_SECONDS", 60 * 60 * 24 if mode() == "full" else 60 * 60 * 8)
    capture_gpu = os.getenv("EMOTION_ASSET_CAPTURE_GPU", "H200:2" if mode() == "full" else "H100:2")
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu=capture_gpu,
                cpu=_env_int("EMOTION_ASSET_CAPTURE_CPU", 16 if mode() == "full" else 8),
                memory_mb=_env_int("EMOTION_ASSET_CAPTURE_MEMORY_MB", (128 if mode() == "full" else 96) * 1024),
                timeout_seconds=capture_timeout,
                max_containers=capture_max_containers,
                shard_count=capture_shard_count,
                enable_workflow_batching=_env_bool("EMOTION_ASSET_CAPTURE_WORKFLOW_BATCHING", True),
                env=shared_env,
                secrets=(db_secret,),
                volumes=(model_volume,),
            ),
            artifacts=artifact_store,
            catalog=catalog,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=_env_int("EMOTION_ASSET_ANALYSIS_CPU", 16 if mode() == "full" else 8),
                memory_mb=_env_int("EMOTION_ASSET_ANALYSIS_MEMORY_MB", (192 if mode() == "full" else 32) * 1024),
                timeout_seconds=_env_int("EMOTION_ASSET_ANALYSIS_TIMEOUT_SECONDS", 60 * 60 * 12 if mode() == "full" else 60 * 60 * 2),
                enable_workflow_batching=_env_bool("EMOTION_ASSET_ANALYSIS_WORKFLOW_BATCHING", True),
                env=shared_env,
                secrets=(db_secret,),
                volumes=(model_volume,),
            ),
            artifacts=artifact_store,
            catalog=catalog,
        ),
        "report_local": LocalRunnerSpec(artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT), catalog=catalog),
    }


def _env_list(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = os.getenv(name)
    if not raw:
        return default
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def _workflow_env() -> dict[str, str]:
    prefixes = ("EMOTION_ASSET_",)
    return {key: value for key, value in os.environ.items() if key.startswith(prefixes)}
