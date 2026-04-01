from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from pipelines.interp.cohort_selection import build_interp_example_query, validate_relation_name


@dataclass(slots=True)
class CaptureConfig:
    output_dir: Path = field(default_factory=lambda: Path("data/activations"))
    parquet_path: Path | None = None
    model_id: str = "Qwen/Qwen3-8B"
    device: str = "mps"
    limit: int | None = None
    layers: list[int] | None = None
    skip_existing: bool = False
    validate_tokens: bool = False
    add_generation_prompt: bool = False
    capture_router: bool = True
    capture_residual: bool = True
    pool_on_capture: str | None = None  # None = full sequence, "last_token", "mean_pool"
    router_dtype: str = "float16"  # "float16" or "float32" for router logits storage
    metadata_flush_interval: int = 10  # write metadata.parquet every N examples
    source_relation: str | None = None
    cohort_view: str | None = None
    order_mode: str = "log_id"


def _flush_metadata(metadata_rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Atomically write metadata.parquet via a .tmp rename."""
    meta_path = output_dir / "metadata.parquet"
    tmp_path = output_dir / "metadata.parquet.tmp"
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(metadata_rows)
    pq.write_table(table, tmp_path, compression="snappy")
    tmp_path.rename(meta_path)


def _load_existing_metadata(output_dir: Path) -> list[dict[str, Any]]:
    """Load previously written metadata.parquet if it exists, for resume support."""
    meta_path = output_dir / "metadata.parquet"
    if not meta_path.exists():
        return []
    try:
        table = pq.read_table(meta_path)
        rows = table.to_pylist()
        print(f"Loaded {len(rows)} existing metadata rows from {meta_path}")
        return rows
    except Exception as exc:
        print(f"WARNING: Could not load existing metadata ({exc}), starting fresh")
        return []


def _load_examples_from_neon(
    *,
    limit: int | None = None,
    source_relation: str | None = None,
    cohort_view: str | None = None,
    order_mode: str = "log_id",
) -> list[dict[str, Any]]:
    """Load high-quality interp examples directly from Neon Postgres."""
    import psycopg
    from psycopg.rows import dict_row
    from pipelines.db import require_neon_dsn

    validated_source = validate_relation_name(source_relation)
    if validated_source and validated_source != "interp_examples_v0":
        query = f"SELECT log_id, prompt_messages_json FROM {validated_source} ORDER BY log_id"
        params: list[Any] = []
        if limit is not None:
            query += " LIMIT %s"
            params.append(limit)
        source = validated_source
    else:
        query, params = build_interp_example_query(
            select_columns=["ie.log_id", "ie.prompt_messages_json"],
            require_market_snapshot=False,
            cohort_view=cohort_view,
            order_mode=order_mode,
            limit=limit,
        )
        source = f"interp_examples_v0 joined to {cohort_view}" if cohort_view else "interp_examples_v0"

    with psycopg.connect(require_neon_dsn(), row_factory=dict_row) as conn:
        rows = conn.execute(query, params).fetchall()

    print(f"Loaded {len(rows)} examples from Neon ({source})")
    return [dict(r) for r in rows]


def _load_examples(config: CaptureConfig) -> list[dict[str, Any]]:
    if config.parquet_path is not None:
        if not config.parquet_path.exists():
            raise FileNotFoundError(config.parquet_path)
        table = pq.read_table(config.parquet_path)
        rows = table.to_pylist()
        return rows[: config.limit] if config.limit is not None else rows

    return _load_examples_from_neon(
        limit=config.limit,
        source_relation=config.source_relation,
        cohort_view=config.cohort_view,
        order_mode=config.order_mode,
    )


def _parse_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    raw = row.get("prompt_messages_json")
    if not raw:
        return []
    if isinstance(raw, str):
        parsed = json.loads(raw)
    else:
        parsed = raw
    if not isinstance(parsed, list):
        return []
    return [
        {"role": m["role"], "content": m["content"]}
        for m in parsed
        if isinstance(m, dict) and "role" in m and "content" in m
    ]


def _load_model(config: CaptureConfig) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {config.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    print(f"Loading model: {config.model_id} (float16 -> {config.device})")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            dtype=torch.float16,
        ).to(config.device).eval()
    except RuntimeError as exc:
        if "mps" in config.device.lower():
            print(f"MPS load failed ({exc}), retrying with float32...")
            model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                dtype=torch.float32,
            ).to(config.device).eval()
        else:
            raise

    num_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    print(f"  {num_layers} layers, hidden_dim={hidden_dim}")
    return model, tokenizer


def _to_id_list(result: Any) -> list[int]:
    """Extract a flat list of token IDs from apply_chat_template output.

    Handles: list[int], torch.Tensor, tokenizers.Encoding,
    list[Encoding], or BatchEncoding (dict-like with input_ids).
    """
    # BatchEncoding or dict-like with input_ids
    if hasattr(result, "input_ids"):
        ids = result.input_ids
        if hasattr(ids, "tolist"):
            t = ids.squeeze(0) if ids.dim() > 1 else ids
            return t.tolist()
        if isinstance(ids, list):
            return ids[0] if ids and isinstance(ids[0], list) else ids
    if isinstance(result, list):
        if result and hasattr(result[0], "ids"):
            return result[0].ids
        return result
    if hasattr(result, "tolist"):
        t = result.squeeze(0) if result.dim() > 1 else result
        return t.tolist()
    if hasattr(result, "ids"):
        return result.ids
    raise TypeError(f"Unexpected tokenizer output type: {type(result)}")


def _validate_tokens(
    config: CaptureConfig,
    examples: list[dict[str, Any]],
    tokenizer: Any,
) -> None:
    samples = examples[:3]
    print(f"\n{'='*60}")
    print(f"Token validation mode — {len(samples)} samples")
    print(f"{'='*60}")

    for idx, row in enumerate(samples):
        messages = _parse_messages(row)
        log_id = row.get("log_id", f"sample_{idx}")
        print(f"\n--- Sample {idx + 1} (log_id={log_id}) ---")
        print(f"  Messages: {len(messages)}")
        for m in messages:
            print(f"    [{m['role']}] {len(m['content'])} chars")

        raw_no_gen = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        raw_with_gen = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )

        ids_no_gen = _to_id_list(raw_no_gen)
        ids_with_gen = _to_id_list(raw_with_gen)

        print(f"  Tokens (no gen prompt):   {len(ids_no_gen)}")
        print(f"  Tokens (with gen prompt): {len(ids_with_gen)}")
        print(f"  First 10 IDs: {ids_no_gen[:10]}")
        print(f"  Last 10 IDs:  {ids_no_gen[-10:]}")

        decoded = tokenizer.decode(ids_no_gen)
        print(f"  First 200 chars decoded: {decoded[:200]!r}")
        print(f"  Last 200 chars decoded:  {decoded[-200:]!r}")


def _is_moe_model(model: Any) -> bool:
    """Check if the model has MoE layers by looking for a gate on the first layer's MLP."""
    try:
        return hasattr(model.model.layers[0].mlp, "gate")
    except (AttributeError, IndexError):
        return False


def _make_hook(layer_idx: int, storage: dict[int, Any]) -> Any:
    import torch

    def hook_fn(module: Any, input: Any, output: Any) -> None:
        hidden_states = output[0]
        storage[layer_idx] = hidden_states.detach().cpu().to(torch.float16)

    return hook_fn


def _make_router_hook(
    layer_idx: int,
    logits_storage: dict[int, Any],
    indices_storage: dict[int, Any],
) -> Any:
    import torch

    def hook_fn(module: Any, input: Any, output: Any) -> None:
        # Gate forward returns (router_logits, router_scores, router_indices)
        router_logits = output[0]
        router_indices = output[2]
        logits_storage[layer_idx] = router_logits.detach().cpu().to(torch.float32)
        indices_storage[layer_idx] = router_indices.detach().cpu().to(torch.int16)

    return hook_fn


def _capture_one(
    *,
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    config: CaptureConfig,
) -> tuple[Any, Any, Any, Any]:
    """Run a forward pass and capture activations.

    Returns (residual, router_logits, router_indices, input_ids).
    residual/router tensors are None when their capture flag is off or the model
    doesn't support MoE.
    """
    import torch

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=config.add_generation_prompt,
    )
    # apply_chat_template may return a BatchEncoding or a bare tensor
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids.to(config.device)
    else:
        input_ids = encoded.to(config.device)

    seq_len = input_ids.shape[1]
    max_pos = getattr(model.config, "max_position_embeddings", None)
    if max_pos is not None and seq_len > max_pos:
        print(f"  WARNING: seq_len {seq_len} exceeds max_position_embeddings {max_pos}")

    all_layers = list(range(len(model.model.layers)))
    target_layers = config.layers if config.layers is not None else all_layers

    residual_storage: dict[int, Any] = {}
    router_logits_storage: dict[int, Any] = {}
    router_indices_storage: dict[int, Any] = {}
    handles = []

    is_moe = _is_moe_model(model)

    # Residual stream hooks
    if config.capture_residual:
        for layer_idx in target_layers:
            handle = model.model.layers[layer_idx].register_forward_hook(
                _make_hook(layer_idx, residual_storage)
            )
            handles.append(handle)

    # Router hooks (MoE only)
    if config.capture_router and is_moe:
        for layer_idx in target_layers:
            handle = model.model.layers[layer_idx].mlp.gate.register_forward_hook(
                _make_router_hook(layer_idx, router_logits_storage, router_indices_storage)
            )
            handles.append(handle)

    try:
        with torch.no_grad():
            model(input_ids, use_cache=False, output_attentions=False)
    finally:
        for handle in handles:
            handle.remove()

    # Stack residual activations
    residual = None
    if config.capture_residual and residual_storage:
        ordered = [residual_storage[i].squeeze(0) for i in sorted(residual_storage.keys())]
        residual = torch.stack(ordered, dim=0)

    # Stack router tensors
    router_logits = None
    router_indices = None
    if config.capture_router and is_moe and router_logits_storage:
        ordered_logits = [router_logits_storage[i].squeeze(0) for i in sorted(router_logits_storage.keys())]
        ordered_indices = [router_indices_storage[i].squeeze(0) for i in sorted(router_indices_storage.keys())]
        router_logits = torch.stack(ordered_logits, dim=0)
        router_indices = torch.stack(ordered_indices, dim=0)

    return residual, router_logits, router_indices, input_ids


def _apply_pooling(
    residual: Any,
    router_logits: Any,
    router_indices: Any,
    pooling: str,
) -> tuple[Any, Any, Any]:
    """Pool the sequence dimension: (layers, seq_len, dim) → (layers, dim).

    For router_indices with top-k, last_token gives (layers, top_k),
    mean_pool gives (layers, num_experts) as a frequency histogram.
    """
    import torch

    if pooling == "last_token":
        if residual is not None:
            residual = residual[:, -1, :]       # (layers, dim)
        if router_logits is not None:
            router_logits = router_logits[:, -1, :]  # (layers, num_experts)
        if router_indices is not None:
            router_indices = router_indices[:, -1, :]  # (layers, top_k)
    elif pooling == "mean_pool":
        if residual is not None:
            residual = residual.mean(dim=1)     # (layers, dim)
        if router_logits is not None:
            router_logits = router_logits.mean(dim=1)  # (layers, num_experts)
        if router_indices is not None:
            # For indices, mean doesn't make sense — keep last token
            router_indices = router_indices[:, -1, :]
    else:
        raise ValueError(f"Unknown pooling: {pooling}")

    return residual, router_logits, router_indices


def _save_activations(
    tensor: Any,
    output_path: Path,
) -> int:
    from safetensors.torch import save_file

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t = tensor.contiguous()
    save_file({"residual_stream": t}, str(output_path))
    return t.nelement() * t.element_size()


def _save_router(
    router_logits: Any,
    router_indices: Any,
    output_path: Path,
    router_dtype: str = "float16",
) -> int:
    import torch
    from safetensors.torch import save_file

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Cast router logits to the configured storage dtype
    target_dtype = torch.float16 if router_dtype == "float16" else torch.float32
    rl = router_logits.to(target_dtype).contiguous()
    ri = router_indices.contiguous()
    save_file({"router_logits": rl, "router_indices": ri}, str(output_path))
    return rl.nelement() * rl.element_size() + ri.nelement() * ri.element_size()


def run_capture(config: CaptureConfig) -> dict[str, Any]:
    import torch

    examples = _load_examples(config)
    if not examples:
        print("No examples to process.")
        return {"processed": 0, "skipped": 0, "errors": 0}

    model, tokenizer = _load_model(config)

    if config.validate_tokens:
        _validate_tokens(config, examples, tokenizer)
        return {"validated": len(examples[:3])}

    is_moe = _is_moe_model(model)
    if config.capture_router and not is_moe:
        print("WARNING: capture_router=True but model has no MoE layers; skipping router hooks")

    residual_dir = config.output_dir / "residual_stream"
    router_dir = config.output_dir / "router_logits"
    if config.capture_residual:
        residual_dir.mkdir(parents=True, exist_ok=True)
    if config.capture_router and is_moe:
        router_dir.mkdir(parents=True, exist_ok=True)

    # Load existing metadata for resume support
    metadata_rows: list[dict[str, Any]] = _load_existing_metadata(config.output_dir)
    existing_log_ids: set[int] = {int(r["log_id"]) for r in metadata_rows}

    processed = 0
    skipped = 0
    errors = 0
    flush_counter = 0  # counts new items since last flush

    for idx, row in enumerate(examples):
        log_id = row.get("log_id")
        if log_id is None:
            print(f"  [{idx + 1}/{len(examples)}] Skipping row with no log_id")
            skipped += 1
            continue

        # Skip if all output files already exist
        if config.skip_existing:
            # Check metadata-based skip (from previous partial run)
            in_metadata = int(log_id) in existing_log_ids
            residual_exists = not config.capture_residual or (residual_dir / f"{log_id}.safetensors").exists()
            router_exists = not (config.capture_router and is_moe) or (router_dir / f"{log_id}.safetensors").exists()
            if (in_metadata and residual_exists and router_exists) or (residual_exists and router_exists):
                print(f"  [{idx + 1}/{len(examples)}] Skipping existing: {log_id}")
                skipped += 1
                continue

        messages = _parse_messages(row)
        if not messages:
            print(f"  [{idx + 1}/{len(examples)}] Skipping {log_id}: no messages")
            skipped += 1
            continue

        try:
            t0 = time.monotonic()
            residual, router_logits, router_indices, input_ids = _capture_one(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                config=config,
            )
            elapsed = time.monotonic() - t0

            # Pool before saving if requested
            seq_len = int(input_ids.shape[1])
            if config.pool_on_capture:
                residual, router_logits, router_indices = _apply_pooling(
                    residual, router_logits, router_indices, config.pool_on_capture
                )

            file_size = 0
            if residual is not None:
                file_size += _save_activations(
                    residual, residual_dir / f"{log_id}.safetensors"
                )
            if router_logits is not None and router_indices is not None:
                file_size += _save_router(
                    router_logits, router_indices,
                    router_dir / f"{log_id}.safetensors",
                    router_dtype=config.router_dtype,
                )

            prompt_hash = hashlib.sha256(
                input_ids.cpu().numpy().tobytes()
            ).hexdigest()

            num_model_layers = len(model.model.layers)
            captured_layers = sorted(config.layers) if config.layers is not None else list(range(num_model_layers))

            meta_row: dict[str, Any] = {
                "log_id": int(log_id),
                "seq_len": seq_len,
                "prompt_hash": prompt_hash,
                "capture_timestamp": datetime.now(UTC).isoformat(),
                "file_size_bytes": file_size,
                "elapsed_s": round(elapsed, 2),
                "has_router": router_logits is not None,
                "captured_layers": json.dumps(captured_layers),
                "pooling": config.pool_on_capture or "none",
            }
            if residual is not None:
                meta_row["num_layers_captured"] = int(residual.shape[0])
                meta_row["hidden_dim"] = int(residual.shape[-1])
            else:
                # Still record layer count from router tensors
                meta_row["num_layers_captured"] = int(router_logits.shape[0]) if router_logits is not None else 0
                meta_row["hidden_dim"] = 0
            if router_logits is not None:
                meta_row["num_experts"] = int(router_logits.shape[-1])

            metadata_rows.append(meta_row)
            flush_counter += 1

            # Incremental metadata flush
            if flush_counter >= config.metadata_flush_interval:
                _flush_metadata(metadata_rows, config.output_dir)
                flush_counter = 0

            processed += 1
            shape_parts = []
            if residual is not None:
                shape_parts.append(f"residual={tuple(residual.shape)}")
            if router_logits is not None:
                shape_parts.append(f"router={tuple(router_logits.shape)}")
            print(
                f"  [{idx + 1}/{len(examples)}] {log_id}: "
                f"{', '.join(shape_parts)}, "
                f"{file_size / 1024 / 1024:.1f}MB, "
                f"{elapsed:.1f}s"
            )

        except Exception as exc:
            import traceback

            errors += 1
            print(f"  [{idx + 1}/{len(examples)}] ERROR {log_id}: {exc}")
            traceback.print_exc()

    if metadata_rows:
        _flush_metadata(metadata_rows, config.output_dir)
        meta_path = config.output_dir / "metadata.parquet"
        print(f"\nWrote metadata: {meta_path} ({len(metadata_rows)} rows)")

    print(f"\nDone: {processed} captured, {skipped} skipped, {errors} errors")
    return {"processed": processed, "skipped": skipped, "errors": errors}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture activations (residual stream and/or MoE router logits) for interp examples"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/activations"),
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=None,
        help="Optional local parquet source for capture rows. Defaults to loading examples from Neon.",
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-8B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source-relation", default=None)
    parser.add_argument("--cohort-view", default=None)
    parser.add_argument(
        "--order-mode",
        default="log_id",
        choices=["log_id", "created_at_desc", "capture_priority_desc", "selection_rank_asc", "hash"],
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (e.g. 0,12,24,35). Default: all layers.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--validate-tokens",
        action="store_true",
        help="Print tokenization details for 3 samples and exit",
    )
    parser.add_argument(
        "--add-generation-prompt",
        action="store_true",
        help="Append assistant turn start tokens to the chat template",
    )
    parser.add_argument(
        "--capture-router",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture MoE router logits (default: True, --no-capture-router to disable)",
    )
    parser.add_argument(
        "--capture-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture residual stream (default: True, --no-capture-residual to disable)",
    )
    parser.add_argument(
        "--pool-on-capture",
        choices=["last_token", "mean_pool"],
        default=None,
        help="Pool the sequence dimension during capture to reduce file size ~9000x. "
             "Stores (layers, dim) instead of (layers, seq_len, dim).",
    )
    parser.add_argument(
        "--router-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Storage dtype for router logits (default: float16). "
             "Use float32 for backwards compatibility with older captures.",
    )
    parser.add_argument(
        "--metadata-flush-interval",
        type=int,
        default=10,
        help="Write metadata.parquet every N examples for crash resilience (default: 10).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    layers: list[int] | None = None
    if args.layers is not None:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    cfg = CaptureConfig(
        output_dir=args.output_dir,
        parquet_path=args.parquet_path,
        model_id=args.model_id,
        device=args.device,
        limit=args.limit,
        layers=layers,
        skip_existing=args.skip_existing,
        validate_tokens=args.validate_tokens,
        add_generation_prompt=args.add_generation_prompt,
        capture_router=args.capture_router,
        capture_residual=args.capture_residual,
        pool_on_capture=args.pool_on_capture,
        router_dtype=args.router_dtype,
        metadata_flush_interval=args.metadata_flush_interval,
        source_relation=args.source_relation,
        cohort_view=args.cohort_view,
        order_mode=args.order_mode,
    )
    run_capture(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
