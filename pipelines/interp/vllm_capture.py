"""Activation capture pipeline using vLLM for Qwen3 MoE models.

Uses vLLM's ``extract_hidden_states`` (speculative decode auxiliary output)
for residual stream capture, and patched MoE block forwards for router
logit capture.  Produces the same output format as :mod:`pipelines.interp.capture`.

Key vLLM engine constraints:
- ``enforce_eager=True`` — no CUDA graphs so patched forwards work.
- ``max_num_seqs=1`` — one request at a time so router buffers are unambiguous.
- ``enable_chunked_prefill=False`` — full prefill per request.
- ``max_tokens=1`` in SamplingParams — prefill only, no generation.

Usage::

    from pipelines.interp.vllm_capture import VLLMCaptureConfig, run_vllm_capture

    cfg = VLLMCaptureConfig(
        parquet_path=Path("data/interp_exports/interp_examples_v0_high_quality.parquet"),
        output_dir=Path("data/activations"),
        model_id="Qwen/Qwen3-30B-A3B",
    )
    run_vllm_capture(cfg)
"""

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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class VLLMCaptureConfig:
    """Configuration for the vLLM-based capture pipeline."""

    parquet_path: Path = field(
        default_factory=lambda: Path(
            "data/interp_exports/interp_examples_v0_high_quality.parquet"
        )
    )
    output_dir: Path = field(default_factory=lambda: Path("data/activations"))
    model_id: str = "Qwen/Qwen3-30B-A3B"
    limit: int | None = None
    layers: list[int] | None = None
    skip_existing: bool = False
    add_generation_prompt: bool = False
    capture_router: bool = True
    capture_residual: bool = True
    pool_on_capture: str | None = None  # None = full sequence, "last_token", "mean_pool"

    # vLLM engine knobs
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int | None = None
    max_tokens_buffer: int = 8192  # pre-allocated router buffer size

    # Router capture knobs
    router_top_k: int = 8  # top-k indices to save alongside full logits
    router_dtype: str = "float16"  # storage dtype for router logits


# ---------------------------------------------------------------------------
# Helpers reused from capture.py
# ---------------------------------------------------------------------------

def _load_examples(config: VLLMCaptureConfig) -> list[dict[str, Any]]:
    table = pq.read_table(config.parquet_path)
    rows = table.to_pylist()
    print(f"Loaded {len(rows)} examples from {config.parquet_path}")
    if config.limit is not None:
        rows = rows[: config.limit]
        print(f"  Limited to {len(rows)} examples")
    return rows


# Re-export helpers from the HF capture module so everything needed is
# importable from this module too.
from pipelines.interp.capture import (  # noqa: E402
    _apply_pooling,
    _parse_messages,
    _save_activations,
    _save_router,
)


# ---------------------------------------------------------------------------
# vLLM model access
# ---------------------------------------------------------------------------

def _apply_to_model(llm: Any, func: Any) -> Any:
    """Run a function on the model inside each vLLM worker.

    Uses ``LLM.apply_model()`` which is the official API for accessing
    the underlying ``nn.Module`` across all vLLM versions (v0.17+).

    Args:
        llm: The vLLM ``LLM`` instance.
        func: A callable that takes an ``nn.Module`` and returns a result.

    Returns:
        The result from the first (driver) worker.
    """
    results = llm.apply_model(func)
    return results[0]


def _setup_router_capture(model: Any, *, max_tokens: int = 8192) -> bool:
    """Top-level function for apply_model — must be picklable (no closures)."""
    from pipelines.interp.vllm_qwen3_moe import (
        enable_router_capture,
        find_moe_blocks,
        init_router_capture,
    )

    blocks = find_moe_blocks(model)
    if not blocks:
        return False
    init_router_capture(model, max_tokens=max_tokens)
    enable_router_capture(model)
    return True


def _init_router_capture_on_model(llm: Any, max_tokens: int = 8192) -> bool:
    """Initialise router capture on the model inside vLLM workers.

    Returns True if MoE blocks were found and patched.
    """
    from functools import partial

    func = partial(_setup_router_capture, max_tokens=max_tokens)
    return _apply_to_model(llm, func)


def _collect_router_logits_from_model(llm: Any) -> dict[int, Any]:
    """Collect captured router logits from the model inside vLLM workers."""
    from pipelines.interp.vllm_qwen3_moe import collect_router_logits

    return _apply_to_model(llm, collect_router_logits)


def _reset_router_buffers_on_model(llm: Any) -> None:
    """Reset router capture buffers on the model inside vLLM workers."""
    from pipelines.interp.vllm_qwen3_moe import reset_router_buffers

    _apply_to_model(llm, reset_router_buffers)


def _resolve_num_layers(model_id: str) -> int:
    """Determine the number of hidden layers from the model config.

    Loads only the HF config (not the full model) to read
    ``num_hidden_layers``.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    return cfg.num_hidden_layers


def _create_llm(config: VLLMCaptureConfig) -> Any:
    """Create a vLLM ``LLM`` instance configured for activation capture.

    The engine is configured with:
    - ``extract_hidden_states`` speculative decode method for residual stream
      capture via the stock ``ExampleHiddenStatesConnector``.
    - ``enforce_eager=True`` to avoid CUDA graph issues with patched forwards.
    - ``max_num_seqs=1`` so router buffers contain exactly one request's data.
    """
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": config.model_id,
        "enforce_eager": True,
        "max_num_seqs": 1,
        "enable_chunked_prefill": False,
        "tensor_parallel_size": config.tensor_parallel_size,
        "gpu_memory_utilization": config.gpu_memory_utilization,
    }

    if config.max_model_len is not None:
        kwargs["max_model_len"] = config.max_model_len

    # Set up extract_hidden_states for residual capture
    if config.capture_residual:
        storage_path = str(config.output_dir / "residual_stream")
        Path(storage_path).mkdir(parents=True, exist_ok=True)

        # Determine which layers to capture
        if config.layers is not None:
            layer_ids = config.layers
        else:
            num_layers = _resolve_num_layers(config.model_id)
            layer_ids = list(range(num_layers))

        kwargs["speculative_config"] = {
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": layer_ids,
                }
            },
        }
        kwargs["kv_transfer_config"] = {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": storage_path,
            },
        }
        print(f"Creating vLLM engine: {config.model_id}")
        print(f"  enforce_eager=True, max_num_seqs=1, enable_chunked_prefill=False")
        print(f"  Residual capture: {len(layer_ids)} layers -> {storage_path}")
    else:
        print(f"Creating vLLM engine: {config.model_id}")
        print(f"  enforce_eager=True, max_num_seqs=1, enable_chunked_prefill=False")
        print(f"  Residual capture: disabled")

    llm = LLM(**kwargs)
    return llm


# ---------------------------------------------------------------------------
# Single-prompt capture
# ---------------------------------------------------------------------------

def _capture_one_vllm(
    *,
    llm: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    config: VLLMCaptureConfig,
    log_id: str | int,
) -> tuple[Any, Any, Any, Any]:
    """Run a single prompt through vLLM and capture activations.

    Returns ``(residual, router_logits, router_indices, input_ids_list)``
    where:
    - ``residual``: ``(num_layers, seq_len, hidden_dim)`` float16 tensor or None
    - ``router_logits``: ``(num_moe_layers, seq_len, num_experts)`` float16 tensor or None
    - ``router_indices``: ``(num_moe_layers, seq_len, top_k)`` int16 tensor or None
    - ``input_ids_list``: list[int] of token IDs
    """
    import torch
    from vllm import SamplingParams

    # Tokenize via the HF tokenizer (same as HF pipeline for consistency)
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=config.add_generation_prompt,
    )
    if not isinstance(input_ids, list):
        # Handle tensor/encoding returns
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.squeeze().tolist()
        elif hasattr(input_ids, "input_ids"):
            input_ids = input_ids.input_ids
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.squeeze().tolist()

    # Reset router buffers before each request
    if config.capture_router:
        _reset_router_buffers_on_model(llm)

    # Run through vLLM -- single prompt, max_tokens=1 (prefill only)
    sampling_params = SamplingParams(max_tokens=1)
    outputs = llm.generate(
        prompts=[{"prompt_token_ids": input_ids}],
        sampling_params=sampling_params,
    )

    seq_len = len(input_ids)

    # --- Collect residual stream ---
    residual = None
    if config.capture_residual:
        residual_dir = config.output_dir / "residual_stream"
        output = outputs[0]

        # The ExampleHiddenStatesConnector returns the path via
        # kv_transfer_params["hidden_states_path"] on request completion.
        hidden_states_path = None
        if hasattr(output, "kv_transfer_params") and output.kv_transfer_params:
            hidden_states_path = output.kv_transfer_params.get("hidden_states_path")

        if hidden_states_path and Path(hidden_states_path).exists():
            connector_file = Path(hidden_states_path)
            target_file = residual_dir / f"{log_id}.safetensors"

            from safetensors.torch import load_file

            tensors = load_file(str(connector_file))
            # The connector saves with key "hidden_states" and "token_ids".
            # Shape: (num_layers, seq_len, hidden_dim).
            hs = tensors.get("hidden_states") or next(iter(tensors.values()))
            residual = hs.to(torch.float16)

            # Re-save in our canonical format with the "residual_stream" key
            # and rename from vLLM's req_id to our log_id.
            _save_activations(residual, target_file)
            if connector_file != target_file:
                connector_file.unlink(missing_ok=True)
        else:
            print(
                f"  WARNING: Residual file not found for request. "
                f"kv_transfer_params={getattr(output, 'kv_transfer_params', None)}"
            )

    # --- Collect router logits ---
    router_logits_tensor = None
    router_indices_tensor = None
    if config.capture_router:
        router_data = _collect_router_logits_from_model(llm)
        if router_data:
            sorted_layers = sorted(router_data.keys())
            # Stack: {layer_idx: (seq_len, num_experts)} -> (num_layers, seq_len, num_experts)
            stacked = torch.stack(
                [router_data[i] for i in sorted_layers], dim=0
            )

            # Compute top-k indices
            _, topk_indices = torch.topk(stacked, k=config.router_top_k, dim=-1)

            # Cast to storage dtypes
            target_dtype = (
                torch.float16
                if config.router_dtype == "float16"
                else torch.float32
            )
            router_logits_tensor = stacked.to(target_dtype)
            router_indices_tensor = topk_indices.to(torch.int16)

    return residual, router_logits_tensor, router_indices_tensor, input_ids


# ---------------------------------------------------------------------------
# Metadata I/O
# ---------------------------------------------------------------------------

def _load_existing_metadata(meta_path: Path) -> list[dict[str, Any]]:
    """Load existing metadata rows from parquet, or return empty list."""
    if meta_path.exists():
        table = pq.read_table(meta_path)
        rows = table.to_pylist()
        print(f"Loaded {len(rows)} existing metadata rows from {meta_path}")
        return rows
    return []


def _flush_metadata(
    meta_path: Path, metadata_rows: list[dict[str, Any]]
) -> None:
    """Write metadata rows to parquet (overwrites)."""
    if not metadata_rows:
        return
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(metadata_rows)
    pq.write_table(table, meta_path, compression="snappy")


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------

def run_vllm_capture(config: VLLMCaptureConfig) -> dict[str, Any]:
    """Run the vLLM capture pipeline.

    Processes each example one at a time (required by ``max_num_seqs=1``
    constraint for router capture correctness).  Produces the same output
    directory structure and metadata format as :func:`capture.run_capture`.

    Returns:
        Dict with ``processed``, ``skipped``, ``errors`` counts.
    """
    import torch
    from transformers import AutoTokenizer

    if not config.parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {config.parquet_path}")

    examples = _load_examples(config)
    if not examples:
        print("No examples to process.")
        return {"processed": 0, "skipped": 0, "errors": 0}

    # --- Create vLLM engine ---
    llm = _create_llm(config)

    # --- Load tokenizer separately for chat template application ---
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    # --- Set up router capture via apply_model ---
    is_moe = False
    if config.capture_router:
        is_moe = _init_router_capture_on_model(
            llm, max_tokens=config.max_tokens_buffer
        )
        if is_moe:
            print("Router capture enabled on MoE blocks")
        else:
            print(
                "WARNING: capture_router=True but model has no MoE layers; "
                "skipping router capture"
            )

    # Get layer count for metadata
    num_model_layers = _resolve_num_layers(config.model_id)

    # --- Prepare output directories ---
    residual_dir = config.output_dir / "residual_stream"
    router_dir = config.output_dir / "router_logits"
    if config.capture_residual:
        residual_dir.mkdir(parents=True, exist_ok=True)
    if config.capture_router and is_moe:
        router_dir.mkdir(parents=True, exist_ok=True)

    # --- Load or initialise metadata ---
    meta_path = config.output_dir / "metadata.parquet"
    metadata_rows = _load_existing_metadata(meta_path)
    existing_log_ids = {r["log_id"] for r in metadata_rows}

    processed = 0
    skipped = 0
    errors = 0
    flush_interval = 10  # flush metadata every N examples

    for idx, row in enumerate(examples):
        log_id = row.get("log_id")
        if log_id is None:
            print(f"  [{idx + 1}/{len(examples)}] Skipping row with no log_id")
            skipped += 1
            continue

        # Skip if already captured
        if config.skip_existing:
            if int(log_id) in existing_log_ids:
                skipped += 1
                continue
            residual_exists = (
                not config.capture_residual
                or (residual_dir / f"{log_id}.safetensors").exists()
            )
            router_exists = (
                not (config.capture_router and is_moe)
                or (router_dir / f"{log_id}.safetensors").exists()
            )
            if residual_exists and router_exists:
                print(
                    f"  [{idx + 1}/{len(examples)}] Skipping existing: {log_id}"
                )
                skipped += 1
                continue

        messages = _parse_messages(row)
        if not messages:
            print(f"  [{idx + 1}/{len(examples)}] Skipping {log_id}: no messages")
            skipped += 1
            continue

        try:
            t0 = time.monotonic()
            residual, router_logits, router_indices, input_ids = (
                _capture_one_vllm(
                    llm=llm,
                    tokenizer=tokenizer,
                    messages=messages,
                    config=config,
                    log_id=log_id,
                )
            )
            elapsed = time.monotonic() - t0

            seq_len = len(input_ids)

            # Pool before saving if requested
            if config.pool_on_capture:
                residual, router_logits, router_indices = _apply_pooling(
                    residual, router_logits, router_indices, config.pool_on_capture
                )

            file_size = 0

            # Residual is already saved by the connector + rename in
            # _capture_one_vllm, but we may need to re-save after pooling.
            if residual is not None and config.pool_on_capture:
                file_size += _save_activations(
                    residual, residual_dir / f"{log_id}.safetensors"
                )
            elif residual is not None:
                # Already saved; compute file size
                p = residual_dir / f"{log_id}.safetensors"
                if p.exists():
                    file_size += p.stat().st_size

            if router_logits is not None and router_indices is not None:
                file_size += _save_router(
                    router_logits,
                    router_indices,
                    router_dir / f"{log_id}.safetensors",
                )

            # Build metadata row (same schema as HF capture)
            prompt_hash = hashlib.sha256(
                bytes(json.dumps(input_ids), "utf-8")
            ).hexdigest()

            captured_layers = (
                sorted(config.layers)
                if config.layers is not None
                else list(range(num_model_layers))
            )

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
                meta_row["num_layers_captured"] = (
                    int(router_logits.shape[0]) if router_logits is not None else 0
                )
                meta_row["hidden_dim"] = 0
            if router_logits is not None:
                meta_row["num_experts"] = int(router_logits.shape[-1])

            metadata_rows.append(meta_row)
            existing_log_ids.add(int(log_id))
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

            # Periodic metadata flush
            if processed % flush_interval == 0:
                _flush_metadata(meta_path, metadata_rows)

        except Exception as exc:
            import traceback

            errors += 1
            print(f"  [{idx + 1}/{len(examples)}] ERROR {log_id}: {exc}")
            traceback.print_exc()

    # Final metadata flush
    if metadata_rows:
        _flush_metadata(meta_path, metadata_rows)
        print(f"\nWrote metadata: {meta_path} ({len(metadata_rows)} rows)")

    print(f"\nDone: {processed} captured, {skipped} skipped, {errors} errors")
    return {"processed": processed, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture activations (residual stream and/or MoE router logits) "
            "using vLLM for efficient GPU inference"
        )
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=Path(
            "data/interp_exports/interp_examples_v0_high_quality.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/activations"),
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices (e.g. 0,12,24,35). Default: all layers.",
    )
    parser.add_argument("--skip-existing", action="store_true")
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
        help=(
            "Pool the sequence dimension during capture to reduce file size. "
            "Stores (layers, dim) instead of (layers, seq_len, dim)."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory for vLLM (default: 0.90)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length. Default: model config value.",
    )
    parser.add_argument(
        "--max-tokens-buffer",
        type=int,
        default=8192,
        help="Pre-allocated router buffer size (default: 8192)",
    )
    parser.add_argument(
        "--router-top-k",
        type=int,
        default=8,
        help="Number of top-k router indices to save (default: 8)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    layers: list[int] | None = None
    if args.layers is not None:
        layers = [int(x.strip()) for x in args.layers.split(",")]

    cfg = VLLMCaptureConfig(
        parquet_path=args.parquet_path,
        output_dir=args.output_dir,
        model_id=args.model_id,
        limit=args.limit,
        layers=layers,
        skip_existing=args.skip_existing,
        add_generation_prompt=args.add_generation_prompt,
        capture_router=args.capture_router,
        capture_residual=args.capture_residual,
        pool_on_capture=args.pool_on_capture,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_tokens_buffer=args.max_tokens_buffer,
        router_top_k=args.router_top_k,
    )
    run_vllm_capture(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
