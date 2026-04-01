from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

MARKET_HEADER = "## MARKET SNAPSHOT"
DOWNSTREAM_SECTIONS = [
    ("active_strategies", "## ACTIVE STRATEGIES"),
    ("active_settings", "## ACTIVE SETTINGS"),
    ("portfolio", "## PORTFOLIO CONTEXT"),
    ("constraints", "## CONSTRAINTS"),
    ("prev_decisions", "## PREVIOUS DECISIONS"),
]


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def _parse_messages(raw: Any) -> list[dict[str, str]]:
    if not raw:
        return []
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, list):
        return []
    out: list[dict[str, str]] = []
    for item in parsed:
        if isinstance(item, dict) and isinstance(item.get("role"), str) and isinstance(item.get("content"), str):
            out.append({"role": item["role"], "content": item["content"]})
    return out


def _extract_system_user(messages: list[dict[str, str]]) -> tuple[str, str] | None:
    system_text = ""
    user_text = ""
    for msg in messages:
        if msg["role"] == "system" and not system_text:
            system_text = msg["content"]
        elif msg["role"] == "user":
            user_text = msg["content"]
    if not user_text:
        return None
    return system_text, user_text


def _chat_messages(system_text: str, user_text: str) -> list[dict[str, str]]:
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    return messages


def _render_chat_text(
    tokenizer: Any,
    system_text: str,
    user_text: str,
    *,
    tools: list[dict[str, Any]] | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    kwargs: dict[str, Any] = {
        "add_generation_prompt": False,
        "tokenize": False,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if chat_template_kwargs:
        kwargs.update(chat_template_kwargs)
    rendered = tokenizer.apply_chat_template(
        _chat_messages(system_text, user_text),
        **kwargs,
    )
    if not isinstance(rendered, str):
        raise TypeError("Tokenizer did not return rendered chat text")
    return rendered


def _token_offsets_for_rendered(tokenizer: Any, rendered_text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        rendered_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None and isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
    offset_mapping = getattr(encoded, "offset_mapping", None)
    if offset_mapping is None and isinstance(encoded, dict):
        offset_mapping = encoded.get("offset_mapping")
    if input_ids is None or offset_mapping is None:
        raise ValueError("Tokenizer did not return input_ids and offset_mapping")

    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if hasattr(offset_mapping, "tolist"):
        offset_mapping = offset_mapping.tolist()
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if offset_mapping and isinstance(offset_mapping[0], list) and offset_mapping[0] and isinstance(offset_mapping[0][0], list):
        offset_mapping = offset_mapping[0]

    normalized_ids = [int(tok) for tok in input_ids]
    offsets = [(int(start), int(end)) for start, end in offset_mapping]
    return normalized_ids, offsets


def _char_to_token_span(
    offsets: list[tuple[int, int]],
    *,
    start_char: int,
    end_char: int,
) -> tuple[int, int] | None:
    token_start: int | None = None
    token_end: int | None = None
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if token_start is None and tok_end > start_char:
            token_start = idx
        if tok_start < end_char:
            token_end = idx + 1
        elif token_start is not None:
            break
    if token_start is None or token_end is None or token_start >= token_end:
        return None
    return token_start, token_end


def _trim_section_end_char(
    rendered_text: str,
    *,
    section_start_char: int,
    section_end_char: int,
) -> int:
    if section_end_char <= section_start_char:
        return section_end_char
    section_text = rendered_text[section_start_char:section_end_char]
    section_text = re.sub(r"\s+\Z", "", section_text)
    section_text = re.sub(r"(?:\n-+[ \t]*)+\Z", "", section_text)
    section_text = re.sub(r"\s+\Z", "", section_text)
    trimmed_end = section_start_char + len(section_text)
    return trimmed_end if trimmed_end > section_start_char else section_end_char


def pool_decision_residual(
    residual: Any,
    row_boundaries: list[dict[str, Any]],
    section_boundaries: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    import numpy as np

    result: dict[str, Any] = {}
    if residual is None:
        return result

    residual = np.asarray(residual)
    if residual.ndim != 3:
        raise ValueError(f"Expected residual with shape (layers, seq_len, dim), got {residual.shape}")

    _num_layers, seq_len, _hidden_dim = residual.shape

    for rb in row_boundaries:
        i = int(rb["row_index"])
        c_start = int(rb["content_start"])
        c_end = int(rb["content_end"])
        f_start = int(rb["full_start"])
        f_end = int(rb["full_end"])

        if c_start < c_end and c_end <= seq_len:
            row_slice = residual[:, c_start:c_end, :]
            result[f"row_mean_{i}"] = row_slice.mean(axis=1).astype(np.float32)
            result[f"row_eos_{i}"] = residual[:, c_end - 1, :].astype(np.float32)
        elif f_start < f_end and f_end <= seq_len:
            row_slice = residual[:, f_start:f_end, :]
            result[f"row_mean_{i}"] = row_slice.mean(axis=1).astype(np.float32)
            result[f"row_eos_{i}"] = residual[:, f_end - 1, :].astype(np.float32)

    for section_name in (
        "preamble",
        "market",
        "active_strategies",
        "active_settings",
        "portfolio",
        "constraints",
        "price_impact_limits",
        "prev_decisions",
    ):
        if section_name not in section_boundaries:
            continue
        start, end = section_boundaries[section_name]
        if not (start < end <= seq_len):
            continue
        section_slice = residual[:, start:end, :]
        result[f"{section_name}_mean"] = section_slice.mean(axis=1).astype(np.float32)
        result[f"{section_name}_eos"] = residual[:, end - 1, :].astype(np.float32)

    result["last_token"] = residual[:, -1, :].astype(np.float32)
    return result


def _save_pooled(tensors: dict[str, Any], output_path: Path) -> int:
    from safetensors.numpy import save_file

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v for k, v in tensors.items()}, str(output_path))
    return sum(int(v.nbytes) for v in tensors.values())


def _flush_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="snappy")


__all__ = [
    "DOWNSTREAM_SECTIONS",
    "MARKET_HEADER",
    "_char_to_token_span",
    "_extract_system_user",
    "_flush_table",
    "_load_existing_rows",
    "_parse_messages",
    "_render_chat_text",
    "_save_pooled",
    "_token_offsets_for_rendered",
    "_trim_section_end_char",
    "pool_decision_residual",
]
