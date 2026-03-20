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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(slots=True)
class DecisionStructureConfig:
    activations_dir: Path = field(default_factory=lambda: Path("data/activations"))
    output_dir: Path = field(default_factory=lambda: Path("data/activations/decision_structure"))
    model_id: str = "Qwen/Qwen3-30B-A3B"
    limit: int | None = None
    skip_existing: bool = True
    metadata_flush_interval: int = 25


def _load_examples_from_neon(limit: int | None = None) -> list[dict[str, Any]]:
    import psycopg
    from psycopg.rows import dict_row

    from pipelines.db import require_neon_dsn

    query = """
        SELECT log_id,
               prompt_messages_json,
               market_snapshot_json,
               decision_type,
               trade_side,
               asset,
               label_quality
        FROM interp_examples_v0
        WHERE label_quality IN ('high', 'medium')
          AND prompt_messages_json IS NOT NULL
          AND market_snapshot_json IS NOT NULL
        ORDER BY log_id
    """
    params: list[Any] = []
    if limit is not None:
        query += " LIMIT %s"
        params.append(limit)

    with psycopg.connect(require_neon_dsn(), row_factory=dict_row) as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


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


def run_decision_structure_pooling(config: DecisionStructureConfig) -> dict[str, Any]:
    from safetensors.numpy import load_file
    from transformers import AutoTokenizer

    from pipelines.interp.counterfactual import (
        CanonicalPrompt,
        Snapshot,
        build_market_rows,
        compute_labels,
        find_downstream_section_boundaries,
        find_row_boundaries,
        parse_market_section,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    examples = _load_examples_from_neon(limit=config.limit)
    print(f"Loaded {len(examples)} examples from Neon")

    residual_in_dir = config.activations_dir / "residual_stream"
    residual_out_dir = config.output_dir / "residual"
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

    for idx, row in enumerate(examples):
        log_id = int(row["log_id"])
        if config.skip_existing and log_id in existing_ids:
            skipped += 1
            continue

        residual_path = residual_in_dir / f"{log_id}.safetensors"
        if not residual_path.exists():
            skipped += 1
            continue

        try:
            messages = _parse_messages(row["prompt_messages_json"])
            system_user = _extract_system_user(messages)
            market_json = _safe_market_json(row["market_snapshot_json"])
            if system_user is None or market_json is None:
                skipped += 1
                continue
            system_text, user_text = system_user

            header, row_texts = parse_market_section(user_text)
            market_rows = build_market_rows(market_json, row_texts)
            labels = compute_labels(market_rows)

            prompt = CanonicalPrompt(
                snapshot_id=str(log_id),
                variant="original",
                system_text=system_text,
                user_text=user_text,
                row_order=[r.symbol for r in market_rows],
            )
            snap = Snapshot(
                snapshot_id=str(log_id),
                vault_address="",
                snap_date="",
                week_num=0,
                market_json=market_json,
                market_header=header,
                rows=market_rows,
            )

            section_boundaries = find_downstream_section_boundaries(tokenizer, system_text, user_text)
            row_boundaries = find_row_boundaries(tokenizer, prompt, snap)

            tensors = load_file(str(residual_path))
            residual = tensors.get("residual_stream")
            if residual is None:
                skipped += 1
                continue
            if residual.ndim != 3:
                # Already pooled captures are not usable for structure pooling.
                skipped += 1
                continue

            pooled = pool_decision_residual(residual, row_boundaries, section_boundaries)
            file_size = _save_pooled(pooled, residual_out_dir / f"{log_id}.safetensors")

            metadata_rows.append({
                "log_id": log_id,
                "capture_timestamp": datetime.now(UTC).isoformat(),
                "seq_len": int(residual.shape[1]),
                "num_layers_captured": int(residual.shape[0]),
                "hidden_dim": int(residual.shape[2]),
                "n_rows": len(market_rows),
                "n_residual_keys": len(pooled),
                "file_size_bytes": file_size,
            })
            tick_rows.append(build_tick_label_row(
                log_id=log_id,
                decision_type=row.get("decision_type"),
                trade_side=row.get("trade_side"),
                target_asset=row.get("asset"),
                n_rows=len(market_rows),
                user_text=user_text,
            ))
            asset_rows.extend(build_asset_label_rows(
                log_id=log_id,
                market_rows=market_rows,
                computed_labels=labels,
                decision_type=row.get("decision_type"),
                trade_side=row.get("trade_side"),
                target_asset=row.get("asset"),
            ))
            existing_ids.add(log_id)
            processed += 1

            if processed % config.metadata_flush_interval == 0:
                _flush_table(meta_path, metadata_rows)
                _flush_table(tick_path, tick_rows)
                _flush_table(asset_path, asset_rows)

            print(f"  [{idx + 1}/{len(examples)}] {log_id}: {len(market_rows)} rows, {len(pooled)} keys")
        except Exception as exc:
            print(f"  [{idx + 1}/{len(examples)}] ERROR {log_id}: {exc}")
            errors += 1

    _flush_table(meta_path, metadata_rows)
    _flush_table(tick_path, tick_rows)
    _flush_table(asset_path, asset_rows)

    return {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "output_dir": str(config.output_dir),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Pool full-sequence decision captures into row/section structure states")
    p.add_argument("--activations-dir", type=Path, default=Path("data/activations"))
    p.add_argument("--output-dir", type=Path, default=Path("data/activations/decision_structure"))
    p.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--metadata-flush-interval", type=int, default=25)
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
    )
    result = run_decision_structure_pooling(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
