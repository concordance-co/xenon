"""Generate and persist Llama 3.3 70B emotion-story rows.

This workflow is intentionally generation-only. It writes parsed story and
neutral rows to Neon so activation capture/vector derivation can be run as a
separate, resumable stage after text quality has been inspected.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from typing import Any, Iterator, Mapping

from pipelines_v2.api import (
    Dataset,
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
    StepRef,
    TransformBuilder,
    TransformResult,
    TransformSpec,
    TransferPolicy,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)

from papers.voice.emotions.replication.specs import llama70b_vector_workflow as vector_workflow
from papers.voice.storage import (
    ARTIFACT_VOLUME_NAME,
    MODEL_VOLUME_PATH,
    YORA_MODEL_VOLUME_NAME,
    local_vector_root,
    modal_vector_root,
)


WORKFLOW_NAME = "papers_voice_emotions_llama33_70b_generation"
DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
MODEL_ID = vector_workflow.MODEL_ID
MODEL_KEY = vector_workflow.MODEL_KEY
MODEL_VOLUME_NAME = YORA_MODEL_VOLUME_NAME
ASSET_ID = "emotions-llama-3.3-70b-sofroniew-2026-v1"
ASSET_VERSION = "v1"
DEFAULT_TABLE = "papers_voice_emotions_llama70b_generated_rows_v1"
DEFAULT_GENERATION_MAX_MODEL_LEN = 16384
DEFAULT_GENERATION_MAX_TOKENS = 8192
DEFAULT_GENERATION_TIMEOUT_SECONDS = 60 * 60 * 24
MODAL_ARTIFACT_ROOT = modal_vector_root("emotions", "llama-3.3-70b", "sofroniew-2026", "v1", "generation")
LOCAL_ARTIFACT_ROOT = local_vector_root("emotions", "llama-3.3-70b", "sofroniew-2026", "v1", "generation")


def generated_rows_table() -> str:
    return os.getenv("EMOTION_ASSET_GENERATED_ROWS_TABLE", DEFAULT_TABLE).strip() or DEFAULT_TABLE


def build_story_generation_dataset() -> Dataset:
    return vector_workflow.build_story_generation_dataset()


def build_dataset() -> Dataset:
    return build_generation_dataset()


def build_neutral_generation_dataset() -> Dataset:
    return vector_workflow.build_neutral_generation_dataset()


def build_generation_dataset() -> Dataset:
    story = build_story_generation_dataset()
    neutral = build_neutral_generation_dataset()
    examples = [*story.examples, *neutral.examples]
    target_keys = _source_prompt_keys()
    if target_keys:
        examples_by_key = {str(example.key): example for example in examples}
        missing = sorted(target_keys - set(examples_by_key))
        if missing:
            raise ValueError(f"Unknown EMOTION_ASSET_SOURCE_PROMPT_KEYS: {missing}")
        examples = [examples_by_key[key] for key in sorted(target_keys)]
    return Dataset.from_examples(
        examples,
        name=f"{WORKFLOW_NAME}_{vector_workflow.mode()}_generation_prompts",
    )


def _generation_engine() -> VLLMEngine:
    return VLLMEngine(
        model_id=MODEL_ID,
        model_path_root=MODEL_VOLUME_PATH,
        tensor_parallel_size=_env_int("EMOTION_ASSET_GENERATION_TENSOR_PARALLEL_SIZE", 4),
        max_model_len=_env_int("EMOTION_ASSET_GENERATION_MAX_MODEL_LEN", DEFAULT_GENERATION_MAX_MODEL_LEN),
        gpu_memory_utilization=float(os.getenv("EMOTION_ASSET_GENERATION_GPU_MEMORY_UTILIZATION", "0.88")),
        enforce_eager=False,
        enable_prefix_caching=True,
        max_num_seqs=_env_int("EMOTION_ASSET_GENERATION_MAX_NUM_SEQS", 8),
        max_num_batched_tokens=_env_optional_int("EMOTION_ASSET_GENERATION_MAX_NUM_BATCHED_TOKENS"),
        enable_thinking=False,
    )


def persist_generated_rows(*, generation: Any) -> TransformResult:
    payload = generation.result() if hasattr(generation, "result") else generation
    story_rows, neutral_rows = _parsed_rows(payload)
    rows = [*story_rows, *neutral_rows]
    table = generated_rows_table()
    quality = _parse_quality(payload)
    if _strict_persistence_qa() and not rows:
        raise RuntimeError(f"Generated row QA failed: no parsed rows; {json.dumps(quality, sort_keys=True)}")
    _write_rows_to_neon(rows, table=table)

    counts = Counter(str(row["row_role"]) for row in rows)
    emotion_counts = Counter(str(row["emotion"]) for row in rows if row.get("emotion"))
    split_counts = Counter(str(row["split"]) for row in rows if row.get("split"))
    direct_mentions = sum(1 for row in story_rows if row.get("direct_emotion_mention"))
    return TransformResult(
        payload={
            "kind": "emotion_llama70b_generated_rows_persisted",
            "table": table,
            "summary": {
                "mode": vector_workflow.mode(),
                "row_count": len(rows),
                "story_row_count": len(story_rows),
                "neutral_row_count": len(neutral_rows),
                "direct_emotion_mentions": direct_mentions,
                "row_role_counts": dict(sorted(counts.items())),
                "split_counts": dict(sorted(split_counts.items())),
                "emotion_counts": dict(sorted(emotion_counts.items())),
                "target_story_count": len(vector_workflow.selected_emotions())
                * (len(vector_workflow.selected_train_topics()) + len(vector_workflow.selected_heldout_topics()))
                * vector_workflow.stories_per_prompt(),
                "target_neutral_count": len(vector_workflow.selected_neutral_topics())
                * vector_workflow.stories_per_prompt(),
                **quality,
            },
        },
        example_keys=[str(row["example_id"]) for row in rows],
    )


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    prompts = dataset or build_generation_dataset()
    generation_max_tokens = _env_int("EMOTION_ASSET_GENERATION_MAX_TOKENS", DEFAULT_GENERATION_MAX_TOKENS)
    return WorkflowSpec(
        name=WORKFLOW_NAME,
        steps=(
            WorkflowStep(
                name="generate_rows",
                runner="generation_gpu",
                spec=GenerationRunSpec(
                    engine=_generation_engine(),
                    dataset=prompts,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=generation_max_tokens,
                        temperature=float(os.getenv("EMOTION_ASSET_STORY_TEMPERATURE", "0.8")),
                        top_p=float(os.getenv("EMOTION_ASSET_STORY_TOP_P", "0.95")),
                    ),
                ),
            ),
            WorkflowStep(
                name="persist_generated_rows",
                runner="analysis_cpu",
                spec=TransformSpec(
                    builder=TransformBuilder.from_function(persist_generated_rows, local_python_sources=("papers",)),
                    inputs={"generation": StepRef("generate_rows")},
                    inline=True,
                ),
            ),
        ),
    )


def build_runner_specs() -> dict[str, object]:
    db = PostgresSource.from_env(DB_ENV_VAR)
    catalog = PostgresCatalog(source=db)
    db_secret = ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon")
    artifact_store = ModalVolumeStore(
        name=ARTIFACT_VOLUME_NAME,
        root=MODAL_ARTIFACT_ROOT,
        transfer_policy=TransferPolicy(allow_large_transfer=True),
    )
    model_volume = ModalVolumeMount(
        name=MODEL_VOLUME_NAME,
        mount_path=MODEL_VOLUME_PATH,
        create_if_missing=True,
        commit_on_success=True,
    )
    shared_env = {
        "HF_HOME": f"{MODEL_VOLUME_PATH}/hf_home",
        "TRANSFORMERS_CACHE": f"{MODEL_VOLUME_PATH}/hf_home/transformers",
        "VLLM_CACHE_ROOT": os.getenv("EMOTION_ASSET_VLLM_CACHE_ROOT", MODEL_VOLUME_PATH),
        **_workflow_env(),
    }
    return {
        "generation_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu=os.getenv("EMOTION_ASSET_GENERATION_GPU", "H100:4"),
                cpu=8,
                memory_mb=128 * 1024,
                timeout_seconds=_env_int("EMOTION_ASSET_GENERATION_TIMEOUT_SECONDS", DEFAULT_GENERATION_TIMEOUT_SECONDS),
                max_containers=_env_optional_int("EMOTION_ASSET_GENERATION_MAX_CONTAINERS") or 1,
                shard_count=_env_optional_int("EMOTION_ASSET_GENERATION_SHARD_COUNT"),
                env=shared_env,
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
                env=shared_env,
                secrets=(db_secret,),
            ),
            artifacts=artifact_store,
            catalog=catalog,
        ),
        "report_local": LocalRunnerSpec(artifacts=LocalArtifactStore(LOCAL_ARTIFACT_ROOT), catalog=catalog),
    }


def _parsed_rows(payload: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = _payload_rows(payload)
    story_rows: list[dict[str, Any]] = []
    neutral_rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        source = _row_mapping(_row_mapping(raw).get("example"))
        labels = dict(_row_mapping(source.get("labels")))
        if str(labels.get("row_role") or "") == "neutral":
            neutral_rows.extend(_parsed_neutral_example(raw=raw, source=source, labels=labels))
        else:
            story_rows.extend(_parsed_story_example(raw=raw, source=source, labels=labels))
    return story_rows, neutral_rows


def _parsed_story_rows(payload: Any) -> list[dict[str, Any]]:
    story_rows, _ = _parsed_rows(payload)
    return story_rows


def _parsed_story_example(*, raw: Any, source: Mapping[str, Any], labels: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split = str(labels.get("split") or "")
    emotion = str(labels.get("emotion") or "")
    topic = str(labels.get("topic") or "")
    prompt_key = str(source.get("key") or "")
    blocks = vector_workflow._parse_blocks(str(_row_mapping(raw).get("generated_text") or ""), label="story")
    for block_index, text in enumerate(blocks[: vector_workflow.stories_per_prompt()]):
        if _has_exact_target_emotion(text, emotion):
            continue
        example_id = _example_id(
            row_role="story",
            split=split,
            emotion=emotion,
            topic=topic,
            source_prompt_key=prompt_key,
            block_index=block_index,
        )
        rows.append(
            {
                "example_id": example_id,
                "row_role": "story",
                "split": split,
                "emotion": emotion,
                "topic": topic,
                "source_prompt_key": prompt_key,
                "block_index": block_index,
                "text": text,
                "direct_emotion_mention": False,
                "metadata": {
                    "source": "llama70b_generated_story",
                    "n_stories_requested": int(labels.get("n_stories") or vector_workflow.stories_per_prompt()),
                    "source_labels": dict(labels),
                },
            }
        )
    return rows


def _parsed_neutral_rows(payload: Any) -> list[dict[str, Any]]:
    raw_rows = _payload_rows(payload)
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        source = _row_mapping(_row_mapping(raw).get("example"))
        labels = dict(_row_mapping(source.get("labels")))
        rows.extend(_parsed_neutral_example(raw=raw, source=source, labels=labels))
    return rows


def _parsed_neutral_example(*, raw: Any, source: Mapping[str, Any], labels: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    topic = str(labels.get("topic") or "")
    prompt_key = str(source.get("key") or "")
    blocks = vector_workflow._parse_blocks(str(_row_mapping(raw).get("generated_text") or ""), label="dialogue")
    for block_index, text in enumerate(blocks[: vector_workflow.stories_per_prompt()]):
        normalized = _normalize_neutral_dialogue(text)
        example_id = _example_id(
            row_role="neutral",
            split="neutral",
            emotion="neutral",
            topic=topic,
            source_prompt_key=prompt_key,
            block_index=block_index,
        )
        rows.append(
            {
                "example_id": example_id,
                "row_role": "neutral",
                "split": "neutral",
                "emotion": None,
                "topic": topic,
                "source_prompt_key": prompt_key,
                "block_index": block_index,
                "text": normalized,
                "direct_emotion_mention": False,
                "metadata": {
                    "source": "llama70b_generated_neutral_dialogue",
                    "n_dialogues_requested": int(labels.get("n_stories") or vector_workflow.stories_per_prompt()),
                    "source_labels": dict(labels),
                },
            }
        )
    return rows


def _has_exact_target_emotion(text: str, emotion: str) -> bool:
    return bool(emotion) and vector_workflow._contains_exact_word(text, emotion)


def _normalize_neutral_dialogue(text: str) -> str:
    lines = text.strip().splitlines()
    first_speaker = None
    for index, line in enumerate(lines):
        if re.match(r"(?i)^\s*(?:Person|Human)\s*:", line):
            first_speaker = index
            break
    if first_speaker is not None:
        lines = lines[first_speaker:]
    trimmed = "\n".join(lines).strip()
    return trimmed.replace("Person:", "Human:").replace("AI:", "Assistant:")


def _strict_persistence_qa() -> bool:
    value = os.getenv("EMOTION_ASSET_STRICT_PERSISTENCE_QA", "").strip().lower()
    if value:
        return value in {"1", "true", "yes", "on"}
    return vector_workflow.mode() == "full"


def _parse_quality(payload: Any) -> dict[str, Any]:
    expected = vector_workflow.stories_per_prompt()
    prompt_count = 0
    length_finish_count = 0
    filtered_direct_emotion_mentions = 0
    filtered_examples: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for raw in _payload_rows(payload):
        row = _row_mapping(raw)
        source = _row_mapping(row.get("example"))
        labels = dict(_row_mapping(source.get("labels")))
        role = "neutral" if str(labels.get("row_role") or "") == "neutral" else "story"
        label = "dialogue" if role == "neutral" else "story"
        blocks = vector_workflow._parse_blocks(str(row.get("generated_text") or ""), label=label)
        parsed = min(len(blocks), expected)
        finish_reason = str(row.get("finish_reason") or "")
        if role == "story":
            emotion = str(labels.get("emotion") or "")
            for block_index, text in enumerate(blocks[:expected]):
                if not _has_exact_target_emotion(text, emotion):
                    continue
                filtered_direct_emotion_mentions += 1
                if len(filtered_examples) < 20:
                    filtered_examples.append(
                        {
                            "source_prompt_key": str(source.get("key") or ""),
                            "emotion": emotion,
                            "block_index": block_index,
                            "finish_reason": finish_reason,
                            "generated_token_count": len(row.get("generated_token_ids") or ()),
                        }
                    )
        if finish_reason == "length":
            length_finish_count += 1
        prompt_count += 1
        if parsed < expected:
            incomplete.append(
                {
                    "source_prompt_key": str(source.get("key") or ""),
                    "row_role": role,
                    "parsed_count": parsed,
                    "expected_count": expected,
                    "finish_reason": finish_reason,
                    "generated_token_count": len(row.get("generated_token_ids") or ()),
                }
            )
    return {
        "qa_passed": not incomplete,
        "prompt_count": prompt_count,
        "incomplete_prompt_count": len(incomplete),
        "incomplete_prompts": incomplete[:20],
        "length_finish_count": length_finish_count,
        "filtered_direct_emotion_mentions": filtered_direct_emotion_mentions,
        "filtered_direct_emotion_examples": filtered_examples,
    }


def _write_rows_to_neon(rows: list[dict[str, Any]], *, table: str) -> None:
    if not rows:
        return
    import psycopg
    from psycopg import sql

    db_url = os.getenv(DB_ENV_VAR, "").strip()
    if not db_url:
        raise RuntimeError(f"{DB_ENV_VAR} must be set to persist generated rows")
    table_ident = sql.Identifier(table)
    stage_ident = sql.Identifier("emotion_llama70b_generated_rows_stage")
    generation_config = _generation_config_payload()
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        example_id TEXT PRIMARY KEY,
                        asset_id TEXT NOT NULL,
                        asset_version TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        row_role TEXT NOT NULL,
                        split TEXT NOT NULL,
                        emotion TEXT,
                        topic TEXT NOT NULL,
                        source_prompt_key TEXT NOT NULL,
                        block_index INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        text_sha256 TEXT NOT NULL,
                        direct_emotion_mention BOOLEAN NOT NULL DEFAULT FALSE,
                        generator_model_id TEXT NOT NULL,
                        target_model_id TEXT NOT NULL,
                        workflow_name TEXT NOT NULL,
                        generation_config JSONB NOT NULL,
                        metadata JSONB NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                ).format(table_ident)
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (mode, row_role, split, emotion)").format(
                    sql.Identifier(f"{table}_mode_role_split_emotion_idx"),
                    table_ident,
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (topic)").format(
                    sql.Identifier(f"{table}_topic_idx"),
                    table_ident,
                )
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE TEMP TABLE {} (
                        example_id TEXT NOT NULL,
                        asset_id TEXT NOT NULL,
                        asset_version TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        row_role TEXT NOT NULL,
                        split TEXT NOT NULL,
                        emotion TEXT,
                        topic TEXT NOT NULL,
                        source_prompt_key TEXT NOT NULL,
                        block_index INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        text_sha256 TEXT NOT NULL,
                        direct_emotion_mention BOOLEAN NOT NULL,
                        generator_model_id TEXT NOT NULL,
                        target_model_id TEXT NOT NULL,
                        workflow_name TEXT NOT NULL,
                        generation_config TEXT NOT NULL,
                        metadata TEXT NOT NULL
                    ) ON COMMIT DROP
                    """
                ).format(stage_ident)
            )
            with cur.copy(
                sql.SQL(
                    """
                    COPY {} (
                        example_id, asset_id, asset_version, mode, row_role, split, emotion,
                        topic, source_prompt_key, block_index, text, text_sha256,
                        direct_emotion_mention, generator_model_id, target_model_id,
                        workflow_name, generation_config, metadata
                    ) FROM STDIN
                    """
                ).format(stage_ident)
            ) as copy:
                for payload in _db_row_payloads(rows, generation_config=generation_config):
                    copy.write_row(payload)
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (
                        example_id, asset_id, asset_version, mode, row_role, split, emotion,
                        topic, source_prompt_key, block_index, text, text_sha256,
                        direct_emotion_mention, generator_model_id, target_model_id,
                        workflow_name, generation_config, metadata, updated_at
                    )
                    SELECT
                        example_id, asset_id, asset_version, mode, row_role, split, emotion,
                        topic, source_prompt_key, block_index, text, text_sha256,
                        direct_emotion_mention, generator_model_id, target_model_id,
                        workflow_name, generation_config::jsonb, metadata::jsonb, now()
                    FROM {}
                    ON CONFLICT (example_id) DO UPDATE SET
                        text = EXCLUDED.text,
                        text_sha256 = EXCLUDED.text_sha256,
                        direct_emotion_mention = EXCLUDED.direct_emotion_mention,
                        generation_config = EXCLUDED.generation_config,
                        metadata = EXCLUDED.metadata,
                        updated_at = now()
                    """
                ).format(table_ident, stage_ident)
            )
        conn.commit()


def _generation_config_payload() -> dict[str, Any]:
    return {
        "stories_per_prompt": vector_workflow.stories_per_prompt(),
        "max_model_len": _env_int("EMOTION_ASSET_GENERATION_MAX_MODEL_LEN", DEFAULT_GENERATION_MAX_MODEL_LEN),
        "max_tokens": _env_int("EMOTION_ASSET_GENERATION_MAX_TOKENS", DEFAULT_GENERATION_MAX_TOKENS),
        "max_num_seqs": _env_int("EMOTION_ASSET_GENERATION_MAX_NUM_SEQS", 8),
        "max_num_batched_tokens": _env_optional_int("EMOTION_ASSET_GENERATION_MAX_NUM_BATCHED_TOKENS"),
        "tensor_parallel_size": _env_int("EMOTION_ASSET_GENERATION_TENSOR_PARALLEL_SIZE", 4),
        "gpu": os.getenv("EMOTION_ASSET_GENERATION_GPU", "H100:4"),
    }


def _db_row_payloads(
    rows: list[dict[str, Any]],
    *,
    generation_config: Mapping[str, Any],
) -> Iterator[tuple[Any, ...]]:
    generation_config_json = json.dumps(dict(generation_config), sort_keys=True)
    mode = vector_workflow.mode()
    for row in rows:
        text = str(row["text"])
        yield (
            row["example_id"],
            ASSET_ID,
            ASSET_VERSION,
            mode,
            row["row_role"],
            row["split"],
            row.get("emotion"),
            row["topic"],
            row["source_prompt_key"],
            row["block_index"],
            text,
            hashlib.sha256(text.encode("utf-8")).hexdigest(),
            bool(row.get("direct_emotion_mention")),
            MODEL_ID,
            MODEL_ID,
            WORKFLOW_NAME,
            generation_config_json,
            json.dumps(dict(row.get("metadata") or {}), sort_keys=True),
        )


def _payload_rows(payload: Any) -> list[Any]:
    mapped = _row_mapping(payload)
    rows = mapped.get("rows")
    return list(rows) if isinstance(rows, list) else []


def _row_mapping(row: Any) -> Mapping[str, Any]:
    return row if isinstance(row, Mapping) else {}


def _example_id(
    *,
    row_role: str,
    split: str,
    emotion: str,
    topic: str,
    source_prompt_key: str,
    block_index: int,
) -> str:
    payload = json.dumps(
        {
            "mode": vector_workflow.mode(),
            "row_role": row_role,
            "split": split,
            "emotion": emotion,
            "topic": topic,
            "source_prompt_key": source_prompt_key,
            "block_index": block_index,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"emotion_llama70b_{vector_workflow.mode()}_{row_role}_{digest}"


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    return int(raw) if raw else None


def _workflow_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key.startswith("EMOTION_ASSET_")}


def _source_prompt_keys() -> set[str]:
    raw = os.getenv("EMOTION_ASSET_SOURCE_PROMPT_KEYS", "").strip()
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}
