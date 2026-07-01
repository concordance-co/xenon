"""Import a Modal generation artifact into the Neon generated-row table.

This intentionally runs the heavy read inside Modal, next to the `xenon-data`
volume. The local process only receives a small summary, not the full
`result.json` payload.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

import modal


APP_NAME = "emotion-llama70b-generation-neon-import"
VOLUME_NAME = "xenon-data"
VOLUME_MOUNT = "/xenon-data"
DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
DEFAULT_TABLE = "papers_voice_emotions_llama70b_generated_rows_v1"
DEFAULT_ARTIFACT_PATH = (
    "/artifacts/model-assets/vectors/emotions/llama-3.3-70b/"
    "sofroniew-2026/v1/generation/generation_run_1_fbeb75cf4479"
)

ASSET_ID = "emotions-llama-3.3-70b-sofroniew-2026-v1"
ASSET_VERSION = "v1"
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
WORKFLOW_NAME = "papers_voice_emotions_llama33_70b_generation"


image = modal.Image.debian_slim(python_version="3.13").pip_install(
    "ijson>=3.3.0",
    "psycopg[binary]>=3.2.0",
)
volume = modal.Volume.from_name(VOLUME_NAME)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name("xenon-neon")],
    cpu=4,
    memory=32 * 1024,
    timeout=60 * 60 * 6,
)
def import_generation_artifact(
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
    table: str = DEFAULT_TABLE,
    mode: str = "full",
    stories_per_prompt: int = 12,
    replace_mode_rows: bool = False,
    log_every_prompts: int = 100,
) -> dict[str, Any]:
    """Stream the generated rows artifact into Neon with COPY."""
    import psycopg
    from psycopg import sql

    result_path = _volume_path(artifact_path) / "result.json"
    manifest_path = _volume_path(artifact_path) / "manifest.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result payload: {result_path}")

    db_url = os.getenv(DB_ENV_VAR, "").strip()
    if not db_url:
        raise RuntimeError(f"{DB_ENV_VAR} must be set via Modal secret xenon-neon")

    generation_config = _generation_config_from_manifest(manifest_path, stories_per_prompt=stories_per_prompt)
    generation_config_json = json.dumps(generation_config, sort_keys=True)

    summary = _ImportSummary(mode=mode, stories_per_prompt=stories_per_prompt)
    table_ident = sql.Identifier(table)
    stage_ident = sql.Identifier("emotion_llama70b_generated_rows_stage")

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            _ensure_table(cur, table_ident=table_ident, table=table)
            if replace_mode_rows:
                cur.execute(
                    sql.SQL("DELETE FROM {} WHERE asset_id = %s AND asset_version = %s AND mode = %s").format(
                        table_ident
                    ),
                    (ASSET_ID, ASSET_VERSION, mode),
                )
            _create_stage(cur, stage_ident=stage_ident)
            with cur.copy(_copy_sql(stage_ident=stage_ident)) as copy:
                for prompt_index, raw in enumerate(_iter_generation_rows(result_path), start=1):
                    parsed_rows = _parsed_rows_for_raw(
                        raw=raw,
                        mode=mode,
                        stories_per_prompt=stories_per_prompt,
                        summary=summary,
                    )
                    for row in parsed_rows:
                        copy.write_row(_db_row_payload(row, mode=mode, generation_config_json=generation_config_json))
                    if log_every_prompts and prompt_index % log_every_prompts == 0:
                        print(
                            "import progress "
                            f"prompts={prompt_index} rows={summary.persisted_row_count} "
                            f"incomplete={len(summary.incomplete_prompts)}"
                        )
            _upsert_stage(cur, table_ident=table_ident, stage_ident=stage_ident)
            _ensure_indexes(cur, table_ident=table_ident, table=table)
            cur.execute(sql.SQL("ANALYZE {}").format(table_ident))
            conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT row_role, split, count(*)
                    FROM {}
                    WHERE asset_id = %s AND asset_version = %s AND mode = %s
                    GROUP BY row_role, split
                    ORDER BY row_role, split
                    """
                ).format(table_ident),
                (ASSET_ID, ASSET_VERSION, mode),
            )
            neon_counts = [{"row_role": row[0], "split": row[1], "count": int(row[2])} for row in cur.fetchall()]

    payload = summary.to_dict()
    payload["table"] = table
    payload["artifact_path"] = artifact_path
    payload["result_bytes"] = result_path.stat().st_size
    payload["generation_config"] = generation_config
    payload["neon_counts"] = neon_counts
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


@app.local_entrypoint()
def main(
    artifact_path: str = DEFAULT_ARTIFACT_PATH,
    table: str = DEFAULT_TABLE,
    mode: str = "full",
    stories_per_prompt: int = 12,
    replace_mode_rows: bool = False,
) -> None:
    summary = import_generation_artifact.remote(
        artifact_path=artifact_path,
        table=table,
        mode=mode,
        stories_per_prompt=stories_per_prompt,
        replace_mode_rows=replace_mode_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


class _ImportSummary:
    def __init__(self, *, mode: str, stories_per_prompt: int) -> None:
        self.mode = mode
        self.stories_per_prompt = stories_per_prompt
        self.prompt_count = 0
        self.story_prompt_count = 0
        self.neutral_prompt_count = 0
        self.persisted_row_count = 0
        self.story_row_count = 0
        self.neutral_row_count = 0
        self.length_finish_count = 0
        self.filtered_direct_emotion_mentions = 0
        self.row_role_counts: Counter[str] = Counter()
        self.split_counts: Counter[str] = Counter()
        self.emotion_counts: Counter[str] = Counter()
        self.incomplete_prompts: list[dict[str, Any]] = []
        self.filtered_direct_emotion_examples: list[dict[str, Any]] = []

    def record_prompt(
        self,
        *,
        source_prompt_key: str,
        row_role: str,
        parsed_count: int,
        finish_reason: str,
        generated_token_count: int,
    ) -> None:
        self.prompt_count += 1
        if row_role == "neutral":
            self.neutral_prompt_count += 1
        else:
            self.story_prompt_count += 1
        if finish_reason == "length":
            self.length_finish_count += 1
        if parsed_count < self.stories_per_prompt and len(self.incomplete_prompts) < 50:
            self.incomplete_prompts.append(
                {
                    "source_prompt_key": source_prompt_key,
                    "row_role": row_role,
                    "parsed_count": parsed_count,
                    "expected_count": self.stories_per_prompt,
                    "finish_reason": finish_reason,
                    "generated_token_count": generated_token_count,
                }
            )

    def record_row(self, row: Mapping[str, Any]) -> None:
        self.persisted_row_count += 1
        row_role = str(row["row_role"])
        self.row_role_counts[row_role] += 1
        if row_role == "neutral":
            self.neutral_row_count += 1
        else:
            self.story_row_count += 1
        self.split_counts[str(row["split"])] += 1
        if row.get("emotion"):
            self.emotion_counts[str(row["emotion"])] += 1

    def record_filtered_direct_mention(
        self,
        *,
        source_prompt_key: str,
        emotion: str,
        block_index: int,
        finish_reason: str,
        generated_token_count: int,
    ) -> None:
        self.filtered_direct_emotion_mentions += 1
        if len(self.filtered_direct_emotion_examples) < 20:
            self.filtered_direct_emotion_examples.append(
                {
                    "source_prompt_key": source_prompt_key,
                    "emotion": emotion,
                    "block_index": block_index,
                    "finish_reason": finish_reason,
                    "generated_token_count": generated_token_count,
                }
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "stories_per_prompt": self.stories_per_prompt,
            "prompt_count": self.prompt_count,
            "story_prompt_count": self.story_prompt_count,
            "neutral_prompt_count": self.neutral_prompt_count,
            "persisted_row_count": self.persisted_row_count,
            "story_row_count": self.story_row_count,
            "neutral_row_count": self.neutral_row_count,
            "length_finish_count": self.length_finish_count,
            "incomplete_prompt_count": len(self.incomplete_prompts),
            "incomplete_prompts": self.incomplete_prompts,
            "filtered_direct_emotion_mentions": self.filtered_direct_emotion_mentions,
            "filtered_direct_emotion_examples": self.filtered_direct_emotion_examples,
            "row_role_counts": dict(sorted(self.row_role_counts.items())),
            "split_counts": dict(sorted(self.split_counts.items())),
            "emotion_count": len(self.emotion_counts),
        }


def _volume_path(artifact_path: str) -> Path:
    path = artifact_path.strip()
    if path.startswith(f"{VOLUME_NAME}:"):
        path = path.split(":", 1)[1]
    return Path(VOLUME_MOUNT) / path.lstrip("/")


def _iter_generation_rows(result_path: Path) -> Iterator[Mapping[str, Any]]:
    import ijson

    with result_path.open("rb") as handle:
        current: dict[str, Any] | None = None
        labels: dict[str, Any] | None = None
        for prefix, event, value in ijson.parse(handle):
            if prefix == "rows.item" and event == "start_map":
                labels = {}
                current = {"example": {"labels": labels}, "_generated_token_count": 0}
                continue
            if current is None:
                continue
            if prefix == "rows.item" and event == "end_map":
                yield current
                current = None
                labels = None
                continue
            if prefix == "rows.item.generated_text" and event == "string":
                current["generated_text"] = value
            elif prefix == "rows.item.finish_reason" and event == "string":
                current["finish_reason"] = value
            elif prefix == "rows.item.generated_token_ids.item" and event == "number":
                current["_generated_token_count"] += 1
            elif prefix == "rows.item.example.key" and event == "string":
                current["example"]["key"] = value
            elif prefix.startswith("rows.item.example.labels.") and event in {"string", "number", "boolean", "null"}:
                if labels is not None:
                    labels[prefix.rsplit(".", 1)[-1]] = value


def _parsed_rows_for_raw(
    *,
    raw: Mapping[str, Any],
    mode: str,
    stories_per_prompt: int,
    summary: _ImportSummary,
) -> list[dict[str, Any]]:
    source = _row_mapping(_row_mapping(raw).get("example"))
    labels = dict(_row_mapping(source.get("labels")))
    role = "neutral" if str(labels.get("row_role") or "") == "neutral" else "story"
    label = "dialogue" if role == "neutral" else "story"
    finish_reason = str(raw.get("finish_reason") or "")
    generated_token_count = int(raw.get("_generated_token_count") or 0)
    prompt_key = str(source.get("key") or "")
    generated_text = str(raw.get("generated_text") or "")
    blocks = (
        _parse_neutral_blocks_fast(generated_text, stories_per_prompt=stories_per_prompt)
        if role == "neutral"
        else _parse_blocks(generated_text, label=label, stories_per_prompt=stories_per_prompt)
    )
    summary.record_prompt(
        source_prompt_key=prompt_key,
        row_role=role,
        parsed_count=min(len(blocks), stories_per_prompt),
        finish_reason=finish_reason,
        generated_token_count=generated_token_count,
    )
    if role == "neutral":
        rows = _parsed_neutral_example(
            raw=raw,
            source=source,
            labels=labels,
            blocks=blocks,
            mode=mode,
            stories_per_prompt=stories_per_prompt,
        )
    else:
        rows = _parsed_story_example(
            raw=raw,
            source=source,
            labels=labels,
            blocks=blocks,
            mode=mode,
            stories_per_prompt=stories_per_prompt,
            summary=summary,
            finish_reason=finish_reason,
            generated_token_count=generated_token_count,
        )
    for row in rows:
        summary.record_row(row)
    return rows


def _parsed_story_example(
    *,
    raw: Mapping[str, Any],
    source: Mapping[str, Any],
    labels: Mapping[str, Any],
    blocks: list[str],
    mode: str,
    stories_per_prompt: int,
    summary: _ImportSummary,
    finish_reason: str,
    generated_token_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split = str(labels.get("split") or "")
    emotion = str(labels.get("emotion") or "")
    topic = str(labels.get("topic") or "")
    prompt_key = str(source.get("key") or "")
    for block_index, text in enumerate(blocks[:stories_per_prompt]):
        if _contains_exact_word(text, emotion):
            summary.record_filtered_direct_mention(
                source_prompt_key=prompt_key,
                emotion=emotion,
                block_index=block_index,
                finish_reason=finish_reason,
                generated_token_count=generated_token_count,
            )
            continue
        rows.append(
            {
                "example_id": _example_id(
                    mode=mode,
                    row_role="story",
                    split=split,
                    emotion=emotion,
                    topic=topic,
                    source_prompt_key=prompt_key,
                    block_index=block_index,
                ),
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
                    "n_stories_requested": int(labels.get("n_stories") or stories_per_prompt),
                    "source_labels": dict(labels),
                },
            }
        )
    return rows


def _parsed_neutral_example(
    *,
    raw: Mapping[str, Any],
    source: Mapping[str, Any],
    labels: Mapping[str, Any],
    blocks: list[str],
    mode: str,
    stories_per_prompt: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    topic = str(labels.get("topic") or "")
    prompt_key = str(source.get("key") or "")
    for block_index, text in enumerate(blocks[:stories_per_prompt]):
        normalized = _normalize_neutral_dialogue(text)
        rows.append(
            {
                "example_id": _example_id(
                    mode=mode,
                    row_role="neutral",
                    split="neutral",
                    emotion="neutral",
                    topic=topic,
                    source_prompt_key=prompt_key,
                    block_index=block_index,
                ),
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
                    "n_dialogues_requested": int(labels.get("n_stories") or stories_per_prompt),
                    "source_labels": dict(labels),
                },
            }
        )
    return rows


def _parse_blocks(text: str, *, label: str, stories_per_prompt: int) -> list[str]:
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
    return [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()][:stories_per_prompt]


def _parse_neutral_blocks_fast(text: str, *, stories_per_prompt: int) -> list[str]:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    starts = [index for index, line in enumerate(lines) if _is_dialogue_marker(line.strip())]
    chunks: list[str] = []
    if starts:
        for start_index, start in enumerate(starts):
            stop = starts[start_index + 1] if start_index + 1 < len(starts) else len(lines)
            chunk = "\n".join(lines[start:stop]).strip()
            if chunk:
                chunks.append(chunk)
    else:
        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

    cleaned: list[str] = []
    for chunk in chunks:
        trimmed = _strip_dialogue_heading(_trim_trailing_empty_speaker(chunk))
        if _has_person_then_ai_turn(trimmed):
            cleaned.append(trimmed)
        if len(cleaned) >= stories_per_prompt:
            break
    return cleaned


def _is_dialogue_marker(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower().strip("[] ")
    if lowered.startswith("dialogue "):
        suffix = lowered.removeprefix("dialogue ").strip().rstrip(":")
        return suffix.isdigit()
    return lowered.rstrip(":").isdigit()


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
    return paragraph if not starts else paragraph[: min(starts)].strip()


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
    return "direct synonym" in normalized and "convey" in normalized and "emotion" in normalized


def _dialogue_chunks(chunks: list[str]) -> list[str]:
    return [
        cleaned
        for chunk in chunks
        for cleaned in [_strip_dialogue_heading(_trim_trailing_empty_speaker(chunk))]
        if _has_person_then_ai_turn(cleaned)
    ]


def _has_person_then_ai_turn(text: str) -> bool:
    saw_nonempty_person = False
    for line in text.splitlines():
        if re.match(r"(?i)^\s*Person\s*:\s*\S", line):
            saw_nonempty_person = True
            continue
        if saw_nonempty_person and re.match(r"(?i)^\s*AI\s*:\s*\S", line):
            return True
    return False


def _strip_dialogue_heading(chunk: str) -> str:
    lines = chunk.strip().splitlines()
    first = _next_nonempty_line_index(lines, 0)
    if first is None:
        return ""
    heading = lines[first].strip().strip("[] ")
    if re.match(r"(?i)^(?:dialogue\s+)?\d+$", heading):
        del lines[first]
    return "\n".join(lines).strip()


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
    chunks = []
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
    return bool(word) and re.search(rf"\b{re.escape(word.lower())}\b", text.lower()) is not None


def _row_mapping(row: Any) -> Mapping[str, Any]:
    return row if isinstance(row, Mapping) else {}


def _example_id(
    *,
    mode: str,
    row_role: str,
    split: str,
    emotion: str,
    topic: str,
    source_prompt_key: str,
    block_index: int,
) -> str:
    payload = json.dumps(
        {
            "mode": mode,
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
    return f"emotion_llama70b_{mode}_{row_role}_{digest}"


def _generation_config_from_manifest(manifest_path: Path, *, stories_per_prompt: int) -> dict[str, Any]:
    config: dict[str, Any] = {"stories_per_prompt": stories_per_prompt}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        metadata = manifest.get("metadata") if isinstance(manifest, Mapping) else {}
        if isinstance(metadata, Mapping):
            config.update(dict(metadata.get("generation_config") or {}))
    return config


def _ensure_table(cur: Any, *, table_ident: Any, table: str) -> None:
    from psycopg import sql

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


def _create_stage(cur: Any, *, stage_ident: Any) -> None:
    from psycopg import sql

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


def _copy_sql(*, stage_ident: Any) -> Any:
    from psycopg import sql

    return sql.SQL(
        """
        COPY {} (
            example_id, asset_id, asset_version, mode, row_role, split, emotion,
            topic, source_prompt_key, block_index, text, text_sha256,
            direct_emotion_mention, generator_model_id, target_model_id,
            workflow_name, generation_config, metadata
        ) FROM STDIN
        """
    ).format(stage_ident)


def _upsert_stage(cur: Any, *, table_ident: Any, stage_ident: Any) -> None:
    from psycopg import sql

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


def _ensure_indexes(cur: Any, *, table_ident: Any, table: str) -> None:
    from psycopg import sql

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
        sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (source_prompt_key)").format(
            sql.Identifier(f"{table}_source_prompt_key_idx"),
            table_ident,
        )
    )


def _db_row_payload(row: Mapping[str, Any], *, mode: str, generation_config_json: str) -> tuple[Any, ...]:
    text = str(row["text"])
    return (
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
