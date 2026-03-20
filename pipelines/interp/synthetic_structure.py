"""Pool synthetic full-sequence captures into row/section structure representations."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pipelines.db import connect_neon
from pipelines.interp.decision_structure.core import (
    _char_to_token_span,
    _extract_system_user,
    _flush_table,
    _load_existing_rows,
    _parse_messages,
    _render_chat_text,
    _save_pooled,
    _token_offsets_for_rendered,
    pool_decision_residual,
)
from pipelines.interp.synthetic_market_db import build_synthetic_example_query


@dataclass(slots=True)
class SyntheticStructureConfig:
    activations_dir: Path = field(default_factory=lambda: Path("data/activations/phase1"))
    output_dir: Path = field(default_factory=lambda: Path("data/activations/synthetic_structure/phase1"))
    model_id: str = "Qwen/Qwen3-30B-A3B"
    limit: int | None = None
    skip_existing: bool = True
    metadata_flush_interval: int = 25
    cohort_view: str | None = "synthetic_market_phase1_capture_v0"
    order_mode: str = "selection_rank_asc"
    num_workers: int = 8
    shard_index: int = 0
    num_shards: int = 1


def _load_examples_from_neon(
    *,
    limit: int | None,
    cohort_view: str | None,
    order_mode: str,
) -> list[dict[str, Any]]:
    query, params = build_synthetic_example_query(
        select_columns=[
            "log_id",
            "prompt_messages_json",
            "phase_name",
            "example_id",
            "family",
            "family_variant",
            "context_variant",
        ],
        cohort_view=cohort_view,
        order_mode=order_mode,
        limit=limit,
    )
    conn = connect_neon()
    try:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


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


def clear_synthetic_structure_shards(
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


def merge_synthetic_structure_shards(
    output_dir: Path,
    *,
    num_shards: int,
) -> dict[str, Any]:
    if num_shards <= 1:
        raise ValueError("Shard merge requires num_shards > 1")

    all_meta: list[dict[str, Any]] = []
    all_tick: list[dict[str, Any]] = []
    all_asset: list[dict[str, Any]] = []
    seen_shards = 0

    for shard_index in range(num_shards):
        meta_path, tick_path, asset_path = shard_output_paths(output_dir, shard_index)
        if not meta_path.exists() or not tick_path.exists() or not asset_path.exists():
            continue
        seen_shards += 1
        all_meta.extend(_load_existing_rows(meta_path))
        all_tick.extend(_load_existing_rows(tick_path))
        all_asset.extend(_load_existing_rows(asset_path))

    dedup_meta = {(int(row["log_id"]), str(row["phase_name"])): row for row in all_meta}
    dedup_tick = {(int(row["log_id"]), str(row["phase_name"])): row for row in all_tick}
    dedup_asset = {
        (int(row["log_id"]), int(row["row_index"]), str(row["symbol"])): row
        for row in all_asset
    }

    meta_rows = sorted(dedup_meta.values(), key=lambda row: (int(row["log_id"]), str(row["phase_name"])))
    tick_rows = sorted(dedup_tick.values(), key=lambda row: (int(row["log_id"]), str(row["phase_name"])))
    asset_rows = sorted(
        dedup_asset.values(),
        key=lambda row: (int(row["log_id"]), int(row["row_index"]), str(row["symbol"])),
    )

    _flush_table(output_dir / "metadata.parquet", meta_rows)
    _flush_table(output_dir / "tick_labels.parquet", tick_rows)
    _flush_table(output_dir / "asset_labels.parquet", asset_rows)

    return {
        "seen_shards": seen_shards,
        "metadata_rows": len(meta_rows),
        "tick_rows": len(tick_rows),
        "asset_rows": len(asset_rows),
    }


def _load_asset_rows(log_ids: list[int], *, phase_name: str) -> dict[int, list[dict[str, Any]]]:
    if not log_ids:
        return {}
    conn = connect_neon()
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM synthetic_market_assets_v0
            WHERE phase_name = %s AND log_id = ANY(%s)
            ORDER BY log_id, row_index
            """,
            (phase_name, log_ids),
        ).fetchall()
    finally:
        conn.close()

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["log_id"]), []).append(dict(row))
    return grouped


def _format_asset_block(row: dict[str, Any]) -> str:
    return "\n".join([
        f"- Asset {row['symbol']}",
        f"  - Archetype: {row['archetype']}",
        f"  - 5m change: {float(row['pct_5m']):+.1f}%",
        f"  - 1h change: {float(row['pct_1h']):+.1f}%",
        f"  - Net flow 5m: {float(row['net_flow_5m']):+.2f}",
        f"  - Volume 5m: {float(row['vol_5m']):.2f}",
        f"  - Volume 1h: {float(row['vol_1h']):.2f}",
        f"  - Unique traders 5m: {int(row['unique_traders_5m'])}",
        f"  - Top 20 holder pct: {float(row['top20_holder_pct']):.1f}%",
        f"  - Age bucket: {row['age_bucket']}",
    ])


def find_synthetic_section_boundaries(
    tokenizer: Any,
    system_text: str,
    user_text: str,
) -> dict[str, tuple[int, int]]:
    rendered = _render_chat_text(tokenizer, system_text, user_text)
    full_ids, offsets = _token_offsets_for_rendered(tokenizer, rendered)

    header_map = [
        ("active_settings", "## ACTIVE SETTINGS"),
        ("market", "## MARKET SNAPSHOT"),
        ("instruction", "Respond with the single best action for this tick:"),
    ]

    starts: list[tuple[str, int, int]] = []
    search_char = 0
    for name, header in header_map:
        idx = rendered.find(header, search_char)
        if idx < 0:
            continue
        span = _char_to_token_span(offsets, start_char=idx, end_char=idx + len(header))
        if span is None:
            continue
        starts.append((name, span[0], idx))
        search_char = idx + len(header)

    boundaries: dict[str, tuple[int, int]] = {}
    for i, (name, start_tok, _) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else len(full_ids)
        boundaries[name] = (start_tok, end)

    first_start = starts[0][1] if starts else None
    if first_start is not None and first_start > 0:
        boundaries["preamble"] = (0, first_start)
    return boundaries


def find_synthetic_row_boundaries(
    tokenizer: Any,
    system_text: str,
    user_text: str,
    asset_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rendered = _render_chat_text(tokenizer, system_text, user_text)
    _, offsets = _token_offsets_for_rendered(tokenizer, rendered)
    market_char = rendered.find("## MARKET SNAPSHOT")
    search_char = market_char if market_char >= 0 else 0

    row_bounds: list[dict[str, Any]] = []
    for row in asset_rows:
        row_text = _format_asset_block(row)
        row_char = rendered.find(row_text, search_char)
        if row_char < 0:
            print(f"WARNING: could not locate synthetic row span for {row['symbol']}")
            continue
        row_span = _char_to_token_span(offsets, start_char=row_char, end_char=row_char + len(row_text))
        if row_span is None:
            print(f"WARNING: could not map synthetic row span for {row['symbol']}")
            continue
        row_start, row_end = row_span
        row_bounds.append({
            "row_index": int(row["row_index"]),
            "symbol": row["symbol"],
            "full_start": row_start,
            "full_end": row_end,
            "content_start": row_start,
            "content_end": row_end,
        })
        search_char = row_char + len(row_text)
    return row_bounds


def build_tick_label_row(row: dict[str, Any], *, user_text: str, n_rows: int) -> dict[str, Any]:
    return {
        "log_id": int(row["log_id"]),
        "phase_name": row["phase_name"],
        "example_id": row["example_id"],
        "family": row["family"],
        "family_variant": row["family_variant"],
        "context_variant": row["context_variant"],
        "n_rows": int(n_rows),
        "user_chars": len(user_text),
    }


def build_asset_label_rows(log_id: int, asset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in asset_rows:
        out = dict(row)
        out["log_id"] = int(log_id)
        out["row_index"] = int(row["row_index"])
        rows.append(out)
    return rows


def _process_pooling_example(
    row: dict[str, Any],
    *,
    tokenizer: Any,
    residual_in_dir: Path,
    residual_out_dir: Path,
    asset_rows_by_log: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    from safetensors.numpy import load_file

    log_id = int(row["log_id"])
    residual_path = residual_in_dir / f"{log_id}.safetensors"
    if not residual_path.exists():
        return {"status": "skipped", "log_id": log_id, "reason": "missing_residual"}

    messages = _parse_messages(row["prompt_messages_json"])
    system_user = _extract_system_user(messages)
    if system_user is None:
        return {"status": "skipped", "log_id": log_id, "reason": "missing_prompt"}
    asset_rows = asset_rows_by_log.get(log_id)
    if not asset_rows:
        return {"status": "skipped", "log_id": log_id, "reason": "missing_asset_rows"}

    system_text, user_text = system_user
    section_boundaries = find_synthetic_section_boundaries(tokenizer, system_text, user_text)
    row_boundaries = find_synthetic_row_boundaries(tokenizer, system_text, user_text, asset_rows)

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
        "phase_name": row["phase_name"],
        "capture_timestamp": datetime.now(UTC).isoformat(),
        "seq_len": int(residual.shape[1]),
        "num_layers_captured": int(residual.shape[0]),
        "hidden_dim": int(residual.shape[2]),
        "n_rows": len(asset_rows),
        "n_residual_keys": len(pooled),
        "file_size_bytes": file_size,
    }
    tick_row = build_tick_label_row(row, user_text=user_text, n_rows=len(asset_rows))
    return {
        "status": "processed",
        "log_id": log_id,
        "metadata_row": metadata_row,
        "tick_row": tick_row,
        "asset_rows": build_asset_label_rows(log_id, asset_rows),
        "n_rows": len(asset_rows),
        "n_pooled": len(pooled),
    }


def run_synthetic_structure_pooling(config: SyntheticStructureConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    examples = _load_examples_from_neon(
        limit=config.limit,
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
            f"with {len(examples)} synthetic examples from Neon",
        )
    else:
        print(f"Loaded {len(examples)} synthetic examples from Neon")

    if not examples:
        return {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "output_dir": str(config.output_dir),
            "shard_index": config.shard_index,
            "num_shards": config.num_shards,
        }

    phase_name = str(examples[0]["phase_name"])
    asset_rows_by_log = _load_asset_rows([int(row["log_id"]) for row in examples], phase_name=phase_name)

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
    existing_ids = {int(row["log_id"]) for row in metadata_rows}

    pending_examples: list[dict[str, Any]] = []
    skipped = 0
    for row in examples:
        log_id = int(row["log_id"])
        if config.skip_existing and log_id in existing_ids:
            skipped += 1
            continue
        pending_examples.append(row)

    processed = 0
    errors = 0
    max_workers = max(1, int(config.num_workers))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _process_pooling_example,
                row,
                tokenizer=tokenizer,
                residual_in_dir=residual_in_dir,
                residual_out_dir=residual_out_dir,
                asset_rows_by_log=asset_rows_by_log,
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

            if result.get("status") == "processed":
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
            else:
                skipped += 1

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
    parser = argparse.ArgumentParser(description="Pool synthetic full-sequence captures into row/section structure states")
    parser.add_argument("--activations-dir", type=Path, default=Path("data/activations/phase1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/activations/synthetic_structure/phase1"))
    parser.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metadata-flush-interval", type=int, default=25)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cohort-view", default="synthetic_market_phase1_capture_v0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--order-mode",
        default="selection_rank_asc",
        choices=["log_id", "created_at_desc", "capture_priority_desc", "selection_rank_asc", "hash"],
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = SyntheticStructureConfig(
        activations_dir=args.activations_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        limit=args.limit if args.limit > 0 else None,
        skip_existing=args.skip_existing,
        metadata_flush_interval=args.metadata_flush_interval,
        cohort_view=args.cohort_view,
        order_mode=args.order_mode,
        num_workers=args.num_workers,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    print(json.dumps(run_synthetic_structure_pooling(config), indent=2, default=str))


if __name__ == "__main__":
    main()
