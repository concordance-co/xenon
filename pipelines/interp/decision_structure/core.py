"""Pool full-sequence decision captures into row/section structure representations.

This operates on standard activation captures in ``data/activations`` where
full-sequence residuals were saved (``pool_on_capture=None``). It reconstructs
prompt structure from ``interp_examples_v0.prompt_messages_json`` and
``market_snapshot_json``, then exports:

- section-pooled residual activations per log_id
- tick-level labels (decision/action metadata)
- asset-level labels aligned to market rows

The output format mirrors the counterfactual pooled captures closely so later
analysis can ask when asset-target and action-valence signals become decodable.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pipelines.interp.cohort_selection import build_interp_example_query


@dataclass(slots=True)
class DecisionStructureConfig:
    activations_dir: Path = field(default_factory=lambda: Path("data/activations"))
    output_dir: Path = field(default_factory=lambda: Path("data/activations/decision_structure"))
    model_id: str = "Qwen/Qwen3-30B-A3B"
    limit: int | None = None
    skip_existing: bool = True
    metadata_flush_interval: int = 25
    cohort_view: str | None = None
    order_mode: str = "log_id"
    num_workers: int = 8
    shard_index: int = 0
    num_shards: int = 1


def _load_examples_from_neon(
    limit: int | None = None,
    *,
    cohort_view: str | None = None,
    order_mode: str = "log_id",
) -> list[dict[str, Any]]:
    import psycopg
    from psycopg.rows import dict_row

    from pipelines.db import require_neon_dsn

    query, params = build_interp_example_query(
        select_columns=[
            "ie.log_id",
            "ie.prompt_messages_json",
            "ie.market_snapshot_json",
            "ie.decision_type",
            "ie.trade_side",
            "ie.asset",
            "ie.label_quality",
        ],
        require_market_snapshot=True,
        cohort_view=cohort_view,
        order_mode=order_mode,
        limit=limit,
    )

    with psycopg.connect(require_neon_dsn(), row_factory=dict_row) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return pq.read_table(path).to_pylist()


def select_examples_for_shard(
    examples: list[dict[str, Any]],
    *,
    shard_index: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    if num_shards <= 1:
        return list(examples)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"Invalid shard_index={shard_index} for num_shards={num_shards}")
    return [row for idx, row in enumerate(examples) if idx % num_shards == shard_index]


def shard_output_paths(output_dir: Path, shard_index: int) -> tuple[Path, Path, Path]:
    shard_dir = output_dir / "shards"
    return (
        shard_dir / f"metadata_shard_{shard_index:02d}.parquet",
        shard_dir / f"tick_labels_shard_{shard_index:02d}.parquet",
        shard_dir / f"asset_labels_shard_{shard_index:02d}.parquet",
    )


def clear_decision_structure_shards(
    output_dir: Path,
    *,
    num_shards: int,
    clear_canonical: bool = True,
) -> dict[str, int]:
    removed = 0
    missing = 0

    for shard_index in range(max(0, int(num_shards))):
        for path in shard_output_paths(output_dir, shard_index):
            if path.exists():
                path.unlink()
                removed += 1
            else:
                missing += 1

    if clear_canonical:
        for path in (
            output_dir / "metadata.parquet",
            output_dir / "tick_labels.parquet",
            output_dir / "asset_labels.parquet",
        ):
            if path.exists():
                path.unlink()
                removed += 1
            else:
                missing += 1

    return {"removed": removed, "missing": missing}


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


def _safe_market_json(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return raw if isinstance(raw, dict) else None


def _executed_valence(decision_type: str | None, trade_side: str | None) -> str:
    if decision_type != "trade":
        return "neutral"
    if trade_side == "buy":
        return "bullish"
    if trade_side == "sell":
        return "bearish"
    return "neutral"


def build_asset_label_rows(
    *,
    log_id: int,
    market_rows: list[Any],
    computed_labels: dict[str, list[int]],
    decision_type: str | None,
    trade_side: str | None,
    target_asset: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_norm = target_asset.upper() if isinstance(target_asset, str) else None
    for i, row in enumerate(market_rows):
        symbol = row.symbol
        is_target = bool(target_norm and symbol.upper() == target_norm)
        if is_target and trade_side == "buy":
            asset_valence = "bullish"
        elif is_target and trade_side == "sell":
            asset_valence = "bearish"
        else:
            asset_valence = "neutral"

        out = {
            "log_id": int(log_id),
            "row_index": i,
            "symbol": symbol,
            "name": row.name,
            "decision_type": decision_type,
            "trade_side": trade_side,
            "target_asset": target_asset,
            "is_target_asset": is_target,
            "is_buy_target": bool(is_target and trade_side == "buy"),
            "is_sell_target": bool(is_target and trade_side == "sell"),
            "asset_executed_valence": asset_valence,
            "pct_5m": row.pct_5m,
            "pct_1h": row.pct_1h,
            "net_flow_5m": row.net_flow_5m,
            "vol_1h": row.vol_1h,
            "vol_5m": row.vol_5m,
            "unique_traders_5m": row.unique_traders_5m,
        }
        for label_name, label_values in computed_labels.items():
            if i < len(label_values):
                out[label_name] = int(label_values[i])
        rows.append(out)
    return rows


def build_tick_label_row(
    *,
    log_id: int,
    decision_type: str | None,
    trade_side: str | None,
    target_asset: str | None,
    n_rows: int,
    user_text: str,
) -> dict[str, Any]:
    return {
        "log_id": int(log_id),
        "decision_type": decision_type,
        "trade_side": trade_side,
        "target_asset": target_asset,
        "executed_valence": _executed_valence(decision_type, trade_side),
        "n_rows": n_rows,
        "user_chars": len(user_text),
    }


def _tokenize_text(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False, return_tensors=None)
    if isinstance(encoded, dict):
        return list(encoded.get("input_ids", []))
    return list(encoded)


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

    input_ids = [int(tok) for tok in input_ids]
    offsets = [(int(start), int(end)) for start, end in offset_mapping]
    return input_ids, offsets


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


def find_real_section_boundaries(
    tokenizer: Any,
    system_text: str,
    user_text: str,
) -> dict[str, tuple[int, int]]:
    from pipelines.interp.counterfactual import DOWNSTREAM_SECTIONS, MARKET_HEADER

    rendered = _render_chat_text(tokenizer, system_text, user_text)
    full_ids, offsets = _token_offsets_for_rendered(tokenizer, rendered)
    ordered_headers = [("market", MARKET_HEADER), *DOWNSTREAM_SECTIONS]

    starts: list[tuple[str, int, int]] = []
    search_char = 0
    for name, header in ordered_headers:
        idx = rendered.find(header, search_char)
        if idx < 0:
            continue
        token_span = _char_to_token_span(offsets, start_char=idx, end_char=idx + len(header))
        if token_span is None:
            continue
        starts.append((name, token_span[0], idx))
        search_char = idx + len(header)

    if not starts:
        return {}

    boundaries: dict[str, tuple[int, int]] = {}
    for idx, (name, _start_tok, start_char) in enumerate(starts):
        raw_end_char = starts[idx + 1][2] if idx + 1 < len(starts) else len(rendered)
        trimmed_end_char = _trim_section_end_char(
            rendered,
            section_start_char=start_char,
            section_end_char=raw_end_char,
        )
        span = _char_to_token_span(offsets, start_char=start_char, end_char=trimmed_end_char)
        if span is None:
            continue
        boundaries[name] = span

    market_start = next((start_tok for name, start_tok, _ in starts if name == "market"), None)
    if market_start is not None and market_start > 0:
        boundaries["preamble"] = (0, market_start)
    return boundaries


def find_real_row_boundaries(
    tokenizer: Any,
    system_text: str,
    user_text: str,
    market_rows: list[Any],
) -> list[dict[str, Any]]:
    rendered = _render_chat_text(tokenizer, system_text, user_text)
    _, offsets = _token_offsets_for_rendered(tokenizer, rendered)
    market_char = rendered.find("## MARKET SNAPSHOT")
    search_char = market_char if market_char >= 0 else 0

    row_bounds: list[dict[str, Any]] = []
    for i, market_row in enumerate(market_rows):
        row_text = market_row.text_block
        row_char = rendered.find(row_text, search_char)
        if row_char < 0:
            print(f"WARNING: could not locate token span for row {market_row.symbol}")
            continue
        row_span = _char_to_token_span(offsets, start_char=row_char, end_char=row_char + len(row_text))
        if row_span is None:
            print(f"WARNING: could not map token span for row {market_row.symbol}")
            continue
        row_start, row_end = row_span
        pipe_pos = row_text.find("|")
        content_start = row_start
        if pipe_pos >= 0:
            content_text = row_text[pipe_pos:]
            content_char = rendered.find(content_text, row_char, row_char + len(row_text))
            if content_char >= 0:
                content_span = _char_to_token_span(
                    offsets,
                    start_char=content_char,
                    end_char=content_char + len(content_text),
                )
                if content_span is not None:
                    content_start = content_span[0]

        row_bounds.append({
            "row_index": i,
            "symbol": market_row.symbol,
            "full_start": row_start,
            "full_end": row_end,
            "symbol_start": row_start,
            "symbol_end": content_start,
            "content_start": content_start,
            "content_end": row_end,
        })
        search_char = row_char + len(row_text)

    return row_bounds


def pool_decision_residual(
    residual: Any,
    row_boundaries: list[dict[str, Any]],
    section_boundaries: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    """Pool full-sequence residuals into row/section states for real prompts."""
    import numpy as np

    result: dict[str, Any] = {}
    if residual is None:
        return result

    residual = np.asarray(residual)
    if residual.ndim != 3:
        raise ValueError(f"Expected residual with shape (layers, seq_len, dim), got {residual.shape}")

    num_layers, seq_len, _ = residual.shape

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


def merge_decision_structure_shards(
    output_dir: Path,
    *,
    num_shards: int,
) -> dict[str, Any]:
    if num_shards <= 1:
        raise ValueError("Shard merge requires num_shards > 1")

    merged_metadata: dict[int, dict[str, Any]] = {}
    merged_ticks: dict[int, dict[str, Any]] = {}
    merged_assets: dict[tuple[int, int], dict[str, Any]] = {}
    seen_shards = 0

    for shard_index in range(num_shards):
        meta_path, tick_path, asset_path = shard_output_paths(output_dir, shard_index)
        if not (meta_path.exists() and tick_path.exists() and asset_path.exists()):
            continue
        seen_shards += 1
        for row in _load_existing_rows(meta_path):
            merged_metadata[int(row["log_id"])] = row
        for row in _load_existing_rows(tick_path):
            merged_ticks[int(row["log_id"])] = row
        for row in _load_existing_rows(asset_path):
            key = (int(row["log_id"]), int(row["row_index"]))
            merged_assets[key] = row

    metadata_rows = [merged_metadata[k] for k in sorted(merged_metadata)]
    tick_rows = [merged_ticks[k] for k in sorted(merged_ticks)]
    asset_rows = [merged_assets[k] for k in sorted(merged_assets)]

    _flush_table(output_dir / "metadata.parquet", metadata_rows)
    _flush_table(output_dir / "tick_labels.parquet", tick_rows)
    _flush_table(output_dir / "asset_labels.parquet", asset_rows)

    return {
        "seen_shards": seen_shards,
        "metadata_rows": len(metadata_rows),
        "tick_rows": len(tick_rows),
        "asset_rows": len(asset_rows),
        "output_dir": str(output_dir),
    }


def _process_pooling_example(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    residual_in_dir: Path,
    residual_out_dir: Path,
) -> dict[str, Any]:
    from safetensors.numpy import load_file

    from pipelines.interp.counterfactual import (
        build_market_rows,
        compute_labels,
        parse_market_section,
    )

    log_id = int(row["log_id"])
    residual_path = residual_in_dir / f"{log_id}.safetensors"
    if not residual_path.exists():
        return {"status": "skipped", "log_id": log_id, "reason": "missing_residual"}

    messages = _parse_messages(row["prompt_messages_json"])
    system_user = _extract_system_user(messages)
    market_json = _safe_market_json(row["market_snapshot_json"])
    if system_user is None or market_json is None:
        return {"status": "skipped", "log_id": log_id, "reason": "missing_prompt_or_market"}
    system_text, user_text = system_user

    _, row_texts = parse_market_section(user_text)
    market_rows = build_market_rows(market_json, row_texts)
    labels = compute_labels(market_rows)

    section_boundaries = find_real_section_boundaries(tokenizer, system_text, user_text)
    row_boundaries = find_real_row_boundaries(tokenizer, system_text, user_text, market_rows)

    tensors = load_file(str(residual_path))
    residual = tensors.get("residual_stream")
    if residual is None:
        return {"status": "skipped", "log_id": log_id, "reason": "missing_residual_tensor"}
    if residual.ndim != 3:
        return {"status": "skipped", "log_id": log_id, "reason": "already_pooled"}

    pooled = pool_decision_residual(residual, row_boundaries, section_boundaries)
    file_size = _save_pooled(pooled, residual_out_dir / f"{log_id}.safetensors")

    metadata_row = {
        "log_id": log_id,
        "capture_timestamp": datetime.now(UTC).isoformat(),
        "seq_len": int(residual.shape[1]),
        "num_layers_captured": int(residual.shape[0]),
        "hidden_dim": int(residual.shape[2]),
        "n_rows": len(market_rows),
        "n_residual_keys": len(pooled),
        "file_size_bytes": file_size,
    }
    tick_row = build_tick_label_row(
        log_id=log_id,
        decision_type=row.get("decision_type"),
        trade_side=row.get("trade_side"),
        target_asset=row.get("asset"),
        n_rows=len(market_rows),
        user_text=user_text,
    )
    asset_label_rows = build_asset_label_rows(
        log_id=log_id,
        market_rows=market_rows,
        computed_labels=labels,
        decision_type=row.get("decision_type"),
        trade_side=row.get("trade_side"),
        target_asset=row.get("asset"),
    )
    return {
        "status": "processed",
        "log_id": log_id,
        "metadata_row": metadata_row,
        "tick_row": tick_row,
        "asset_rows": asset_label_rows,
        "n_rows": len(market_rows),
        "n_pooled": len(pooled),
    }


def run_decision_structure_pooling(config: DecisionStructureConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    examples = _load_examples_from_neon(
        config.limit,
        cohort_view=config.cohort_view,
        order_mode=config.order_mode,
    )
    if config.num_shards > 1:
        examples = select_examples_for_shard(
            examples,
            shard_index=config.shard_index,
            num_shards=config.num_shards,
        )
        print(
            f"Loaded shard {config.shard_index + 1}/{config.num_shards} "
            f"with {len(examples)} examples from Neon",
        )
    else:
        print(f"Loaded {len(examples)} examples from Neon")

    residual_in_dir = config.activations_dir / "residual_stream"
    residual_out_dir = config.output_dir / "residual"
    if config.num_shards > 1:
        meta_path, tick_path, asset_path = shard_output_paths(config.output_dir, config.shard_index)
    else:
        meta_path = config.output_dir / "metadata.parquet"
        tick_path = config.output_dir / "tick_labels.parquet"
        asset_path = config.output_dir / "asset_labels.parquet"

    metadata_rows = _load_existing_rows(meta_path)
    tick_rows = _load_existing_rows(tick_path)
    asset_rows = _load_existing_rows(asset_path)
    existing_ids = {int(r["log_id"]) for r in metadata_rows}

    processed = 0
    skipped = 0
    errors = 0

    pending_examples: list[dict[str, Any]] = []
    for row in examples:
        log_id = int(row["log_id"])
        if config.skip_existing and log_id in existing_ids:
            skipped += 1
            continue
        pending_examples.append(row)

    max_workers = max(1, int(config.num_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _process_pooling_example,
                row,
                tokenizer=tokenizer,
                residual_in_dir=residual_in_dir,
                residual_out_dir=residual_out_dir,
            ): int(row["log_id"])
            for row in pending_examples
        }
        completed_count = 0
        for fut in as_completed(futures):
            log_id = futures[fut]
            completed_count += 1
            try:
                result = fut.result()
            except Exception as exc:
                print(f"  [{completed_count}/{len(pending_examples)}] ERROR {log_id}: {exc}")
                errors += 1
                continue

            status = result.get("status")
            if status == "processed":
                metadata_rows.append(result["metadata_row"])
                tick_rows.append(result["tick_row"])
                asset_rows.extend(result["asset_rows"])
                existing_ids.add(log_id)
                processed += 1

                if processed % config.metadata_flush_interval == 0:
                    _flush_table(meta_path, metadata_rows)
                    _flush_table(tick_path, tick_rows)
                    _flush_table(asset_path, asset_rows)

                print(
                    f"  [{completed_count}/{len(pending_examples)}] {log_id}: "
                    f"{result['n_rows']} rows, {result['n_pooled']} keys"
                )
            elif status == "skipped":
                skipped += 1
            else:
                print(f"  [{completed_count}/{len(pending_examples)}] ERROR {log_id}: unknown_status")
                errors += 1

    _flush_table(meta_path, metadata_rows)
    _flush_table(tick_path, tick_rows)
    _flush_table(asset_path, asset_rows)

    return {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "output_dir": str(config.output_dir),
        "shard_index": config.shard_index,
        "num_shards": config.num_shards,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pool full-sequence decision captures into row/section structure states")
    p.add_argument("--activations-dir", type=Path, default=Path("data/activations"))
    p.add_argument("--output-dir", type=Path, default=Path("data/activations/decision_structure"))
    p.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--metadata-flush-interval", type=int, default=25)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--cohort-view", default=None)
    p.add_argument(
        "--order-mode",
        default="log_id",
        choices=["log_id", "created_at_desc", "capture_priority_desc", "selection_rank_asc", "hash"],
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = DecisionStructureConfig(
        activations_dir=args.activations_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        limit=args.limit if args.limit > 0 else None,
        skip_existing=args.skip_existing,
        metadata_flush_interval=args.metadata_flush_interval,
        num_workers=args.num_workers,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        cohort_view=args.cohort_view,
        order_mode=args.order_mode,
    )
    result = run_decision_structure_pooling(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
