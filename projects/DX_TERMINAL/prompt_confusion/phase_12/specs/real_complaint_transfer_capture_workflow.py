from __future__ import annotations

"""Capture reusable real-data residual anchors for complaint-transfer experiments."""

import os
import re
from pathlib import Path
from typing import Any

from pipelines_v2.api import (
    CaptureSpec,
    Dataset,
    LocalArtifactStore,
    LocalRunnerSpec,
    ModalSecret,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    PostgresSource,
    PromptMetadataBuilder,
    ResidualSite,
    TensorStorage,
    TokenSelector,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)
from projects.DX_TERMINAL.prompt_confusion.catalogs import build_prompt_confusion_catalog


MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
DB_ENV_VAR = "XENON_NEON_DATABASE_URL"
DEFAULT_TRANSFER_TABLE = os.environ.get(
    "REAL_COMPLAINT_TRANSFER_TABLE",
    "dx_terminal_real_complaint_transfer_ticks_v1",
)
CAPTURED_LAYERS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44)
HIGH_SIGNAL_COMPLAINT_TYPES = frozenset(
    {
        "WRONG_SIZE",
        "UNWANTED_BUY",
        "UNWANTED_SELL",
        "STRATEGY_IGNORED",
        "HOLDING_VIOLATION",
        "OVERTRADING",
        "NOT_TRADING",
        "UNWANTED_HOLD",
    }
)
HIGH_SIGNAL_EXCLUDED_ROOT_CAUSES = frozenset(
    {
        "USER_EXPECTATION_MISMATCH",
        "MARKET_LEGITIMATE",
        "CORRECT_BEHAVIOR",
    }
)
BASELINE_CONTROL_ROOT_CAUSES = frozenset(
    {
        "USER_EXPECTATION_MISMATCH",
        "CORRECT_BEHAVIOR",
        "MARKET_LEGITIMATE",
    }
)

_ROLE_MARKERS = {
    "system": "<|im_start|>system\n",
    "user": "<|im_start|>user\n",
    "assistant": "<|im_start|>assistant\n",
}
_IM_END = "<|im_end|>"


def _dataset_limit_from_env() -> int | None:
    raw = os.environ.get("REAL_COMPLAINT_TRANSFER_DATASET_LIMIT")
    if raw is None or not raw.strip():
        return None
    limit = int(raw)
    if limit <= 0:
        raise ValueError("REAL_COMPLAINT_TRANSFER_DATASET_LIMIT must be a positive integer")
    return limit


def _dataset_slice_from_env() -> str:
    raw = os.environ.get("REAL_COMPLAINT_TRANSFER_SLICE")
    value = "high_signal" if raw is None or not raw.strip() else str(raw).strip().lower()
    if value not in {"all", "high_signal", "baseline_control"}:
        raise ValueError("REAL_COMPLAINT_TRANSFER_SLICE must be one of {'all', 'high_signal', 'baseline_control'}")
    return value


def _dataset_max_prompt_chars_from_env() -> int | None:
    raw = os.environ.get("REAL_COMPLAINT_TRANSFER_MAX_PROMPT_CHARS")
    if raw is None or not raw.strip():
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError("REAL_COMPLAINT_TRANSFER_MAX_PROMPT_CHARS must be a positive integer")
    return value


def _dataset_shard_count_from_env() -> int | None:
    raw = os.environ.get("REAL_COMPLAINT_TRANSFER_SHARD_COUNT")
    if raw is None or not raw.strip():
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError("REAL_COMPLAINT_TRANSFER_SHARD_COUNT must be a positive integer")
    return value


def _dataset_shard_index_from_env(shard_count: int | None) -> int | None:
    if shard_count is None:
        return None
    raw = os.environ.get("REAL_COMPLAINT_TRANSFER_SHARD_INDEX")
    value = 0 if raw is None or not raw.strip() else int(raw)
    if value < 0 or value >= shard_count:
        raise ValueError("REAL_COMPLAINT_TRANSFER_SHARD_INDEX must be in [0, REAL_COMPLAINT_TRANSFER_SHARD_COUNT)")
    return value


def _validate_table_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe table name: {value}")
    return value


def _sql_string_list(values: frozenset[str]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in sorted(values))


def _slice_filter_sql(slice_name: str) -> str:
    if slice_name == "all":
        return "TRUE"
    if slice_name == "baseline_control":
        return (
            f"(root_cause IN ({_sql_string_list(BASELINE_CONTROL_ROOT_CAUSES)}) "
            "OR label = 'market')"
        )
    if slice_name != "high_signal":
        raise ValueError(f"Unsupported slice name: {slice_name}")
    return (
        "label <> 'market' "
        f"AND complaint_type IN ({_sql_string_list(HIGH_SIGNAL_COMPLAINT_TYPES)}) "
        f"AND (root_cause IS NULL OR root_cause NOT IN ({_sql_string_list(HIGH_SIGNAL_EXCLUDED_ROOT_CAUSES)}))"
    )


def build_dataset_sql(
    *,
    table_name: str = DEFAULT_TRANSFER_TABLE,
    slice_name: str,
    max_prompt_chars: int | None = None,
    shard_count: int | None = None,
    shard_index: int | None = None,
) -> str:
    table_name = _validate_table_name(table_name)
    filters = [
        "prompt_messages_json IS NOT NULL",
        "jsonb_typeof(prompt_messages_json) = 'array'",
        _slice_filter_sql(slice_name),
    ]
    if max_prompt_chars is not None:
        filters.append(f"length(prompt_messages_json::text) <= {int(max_prompt_chars)}")
    if shard_count is not None and shard_index is not None:
        filters.append(
            "mod((('x' || substr(md5(example_id), 1, 8))::bit(32)::bigint), "
            f"{int(shard_count)}) = {int(shard_index)}"
        )
    where_sql = "\n          AND ".join(filters)
    return f"""
        SELECT *
        FROM {table_name}
        WHERE {where_sql}
        ORDER BY example_id
    """


def build_dataset(*, limit: int | None = None, slice_name: str | None = None) -> Dataset:
    selected_slice = _dataset_slice_from_env() if slice_name is None else str(slice_name).strip().lower()
    max_prompt_chars = _dataset_max_prompt_chars_from_env()
    shard_count = _dataset_shard_count_from_env()
    shard_index = _dataset_shard_index_from_env(shard_count)
    final_limit = limit if limit is not None else _dataset_limit_from_env()
    dataset = Dataset.from_postgres(
        source=PostgresSource.from_env(DB_ENV_VAR),
        sql=build_dataset_sql(
            table_name=DEFAULT_TRANSFER_TABLE,
            slice_name=selected_slice,
            max_prompt_chars=max_prompt_chars,
            shard_count=shard_count,
            shard_index=shard_index,
        ),
        prompt_column="prompt_messages_json",
        example_key_column="example_id",
        label_columns=[
            "trace_id",
            "vault_address",
            "person_id",
            "label",
            "fault",
            "root_cause",
            "agent_was_correct",
            "severity",
            "confidence",
            "urgency",
            "complaint_type",
            "complaint_text",
            "referenced_tokens",
            "has_strategy",
            "slider_ta",
            "slider_arp",
            "slider_ts",
            "slider_hs",
            "slider_div",
            "n_relevant_ticks",
            "n_ticks_attached",
            "tick_index",
            "created_at",
            "minute_key",
            "tool",
            "llm_model",
            "prompt_message_count",
            "size_relevant_complaint",
            "activity_relevant_complaint",
            "config_conflict_like",
            "system_fault",
        ],
        case_columns=["trace_id", "vault_address"],
        case_key_column="trace_id",
        limit=final_limit,
        name=f"prompt_confusion_real_complaint_transfer_{selected_slice}",
    )
    return dataset


def _default_residual_engine() -> VLLMEngine:
    raw_max_num_seqs = os.environ.get("REAL_COMPLAINT_TRANSFER_MAX_NUM_SEQS")
    max_num_seqs = 8 if raw_max_num_seqs is None or not raw_max_num_seqs.strip() else int(raw_max_num_seqs)
    return VLLMEngine(
        model_id=MODEL_ID,
        max_model_len=40960,
        enforce_eager=False,
        max_num_seqs=max_num_seqs,
    )


def _trim_section_end(rendered_prompt: str, *, start: int, end: int) -> int:
    if end <= start:
        return int(end)
    section = rendered_prompt[int(start) : int(end)]
    section = re.sub(r"\s+\Z", "", section)
    trimmed_end = int(start) + len(section)
    return trimmed_end if trimmed_end > int(start) else int(end)


def _last_non_whitespace_char_span(rendered_prompt: str, *, start: int, end: int) -> tuple[int, int]:
    if end <= start:
        return int(start), int(end)
    index = int(end) - 1
    while index >= int(start) and rendered_prompt[index].isspace():
        index -= 1
    if index < int(start):
        return int(end) - 1, int(end)
    return index, index + 1


def _find_role_content_span(rendered_prompt: str, role: str) -> tuple[int, int] | None:
    marker = _ROLE_MARKERS.get(str(role))
    if not marker:
        return None
    marker_start = rendered_prompt.find(marker)
    if marker_start < 0:
        return None
    content_start = marker_start + len(marker)
    marker_end = rendered_prompt.find(_IM_END, content_start)
    if marker_end < 0:
        return None
    content_end = _trim_section_end(rendered_prompt, start=content_start, end=marker_end)
    return int(content_start), int(content_end)


def build_real_complaint_prompt_metadata(rendered_prompt: str) -> dict[str, Any]:
    token_sections: dict[str, dict[str, int]] = {}
    for role in ("system", "user"):
        content_span = _find_role_content_span(rendered_prompt, role)
        if content_span is None:
            continue
        start, end = content_span
        token_sections[role] = {
            "char_start": int(start),
            "char_end": int(end),
        }
        tail_start, tail_end = _last_non_whitespace_char_span(
            rendered_prompt,
            start=int(start),
            end=int(end),
        )
        token_sections[f"{role}_last_token"] = {
            "char_start": int(tail_start),
            "char_end": int(tail_end),
        }
    return {"token_sections": token_sections}


def build_runner_specs() -> dict[str, object]:
    db_secret = ModalSecret.from_env_var(DB_ENV_VAR, secret_name="xenon-neon")
    artifact_store = ModalVolumeStore(
        name="xenon-data",
        root="/data/artifacts/prompt_confusion_real_complaint_transfer",
    )
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu="H100",
                secrets=(db_secret,),
                volumes=(ModalVolumeMount(name=MODEL_VOLUME_NAME, mount_path=MODEL_VOLUME_PATH),),
            ),
            artifacts=artifact_store,
            catalog=build_prompt_confusion_catalog(__file__),
        ),
        "report_local": LocalRunnerSpec(
            artifacts=LocalArtifactStore(Path("artifacts") / "prompt_confusion_real_complaint_transfer"),
            catalog=build_prompt_confusion_catalog(__file__),
        ),
    }


def build_workflow(
    dataset: Dataset | None = None,
    *,
    residual_engine: VLLMEngine | None = None,
) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    residual_engine = residual_engine or _default_residual_engine()
    prompt_metadata = PromptMetadataBuilder.from_function(
        build_real_complaint_prompt_metadata,
        local_python_sources=(".",),
    )

    return WorkflowSpec(
        name="prompt_confusion_real_complaint_transfer_capture",
        steps=(
            WorkflowStep(
                name="capture_real_complaint_residual_anchors",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=residual_engine,
                    dataset=dataset,
                    sites=[
                        ResidualSite(
                            name="residual_prompt_last",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.last(),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                        ResidualSite(
                            name="residual_user_last",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.section("user_last_token"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                        ResidualSite(
                            name="residual_system_last",
                            site="resid_post",
                            layers=list(CAPTURED_LAYERS),
                            tokens=TokenSelector.section("system_last_token"),
                            storage=TensorStorage(dtype="float16", format="safetensors"),
                        ),
                    ],
                    prompt_metadata_builder=prompt_metadata,
                ),
            ),
        ),
    )
