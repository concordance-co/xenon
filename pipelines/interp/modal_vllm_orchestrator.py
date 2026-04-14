"""Canonical Modal capture orchestrator for generic workflow capture runs."""

import modal
import re
from typing import Any

app = modal.App("xenon-vllm-capture")

volume = modal.Volume.from_name("xenon-data", create_if_missing=True)
model_volume = modal.Volume.from_name("xenon-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
neon_secret = modal.Secret.from_name("xenon-neon")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "vllm",
        "torch",
        "transformers",
        "safetensors",
        "pyarrow",
        "huggingface_hub",
        "psycopg[binary]",
    )
    .env({"VLLM_ALLOW_INSECURE_SERIALIZATION": "1"})
    .add_local_python_source("pipelines")
)


@app.function(
    volumes={"/models": model_volume},
    image=image,
    secrets=[hf_secret],
    timeout=1800,
)
def download_model(model_id: str = "Qwen/Qwen3-30B-A3B"):
    """Download model weights to the persistent volume."""
    from huggingface_hub import snapshot_download

    local_dir = f"/models/{model_id}"
    print(f"Downloading {model_id} to {local_dir}...")
    snapshot_download(model_id, local_dir=local_dir)
    model_volume.commit()
    print(f"Done. Model cached at {local_dir}")


@app.cls(
    gpu="A100-80GB",
    volumes={"/data": volume, "/models": model_volume},
    image=image,
    timeout=7200,
    scaledown_window=300,
)
class VLLMCaptureWorker:
    """Modal worker that holds a vLLM engine in memory across requests."""

    model_id: str = modal.parameter(default="Qwen/Qwen3-30B-A3B")
    tensor_parallel_size: int = modal.parameter(default=1)
    gpu_memory_utilization: str = modal.parameter(default="0.90")
    enforce_eager: bool = modal.parameter(default=True)
    max_num_seqs: int = modal.parameter(default=1)
    enable_chunked_prefill: bool = modal.parameter(default=False)
    max_model_len: int = modal.parameter(default=0)
    capture_residual: bool = modal.parameter(default=True)
    capture_layers_csv: str = modal.parameter(default="")
    init_router_capture: bool = modal.parameter(default=True)
    add_generation_prompt: bool = modal.parameter(default=False)
    capture_reasoning: bool = modal.parameter(default=True)
    enable_thinking: bool = modal.parameter(default=True)
    reasoning_parser: str = modal.parameter(default="")

    @modal.enter()
    def setup(self):
        """Create vLLM engine and set up router capture."""
        from pathlib import Path

        from transformers import AutoConfig, AutoTokenizer
        from vllm import LLM

        from pipelines.interp.modal_vllm_engine import _init_router_capture_on_model

        local_path = f"/models/{self.model_id}"
        print(f"Loading tokenizer from {local_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)

        hf_config = AutoConfig.from_pretrained(local_path, trust_remote_code=True)
        self.num_layers = hf_config.num_hidden_layers
        self.capture_layer_ids = [
            int(token.strip()) for token in self.capture_layers_csv.split(",") if token.strip()
        ]

        kwargs: dict = {
            "model": local_path,
            "enforce_eager": bool(self.enforce_eager),
            "max_num_seqs": int(self.max_num_seqs),
            "enable_chunked_prefill": bool(self.enable_chunked_prefill),
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": float(self.gpu_memory_utilization),
        }
        reasoning_parser = (self.reasoning_parser or "").strip()
        if self.capture_reasoning:
            if not reasoning_parser and "qwen3" in self.model_id.lower():
                reasoning_parser = "qwen3"
            if reasoning_parser:
                kwargs["structured_outputs_config"] = {
                    "reasoning_parser": reasoning_parser,
                }
        if self.capture_residual:
            layer_ids = self.capture_layer_ids or list(range(self.num_layers))
            storage_path = "/data/activations/residual_stream"
            Path(storage_path).mkdir(parents=True, exist_ok=True)
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
            print(f"  Residual capture: {len(layer_ids)} layers")
        if self.max_model_len > 0:
            kwargs["max_model_len"] = self.max_model_len

        print(f"Creating vLLM engine: {local_path}")
        self.llm = LLM(**kwargs)

        buffer_size = self.max_model_len if self.max_model_len > 0 else 32768
        if self.init_router_capture:
            self.is_moe = _init_router_capture_on_model(self.llm, max_tokens=buffer_size)
        else:
            self.is_moe = False
        if self.is_moe:
            print("Router capture enabled on MoE blocks")
        elif self.init_router_capture:
            print("Model has no MoE blocks; router capture disabled")
        else:
            print("Router capture initialization skipped for generation-only worker")

    @modal.exit()
    def teardown(self):
        from pipelines.interp.modal_vllm_engine import _cleanup_cuda_memory, _destroy_llm

        _destroy_llm(getattr(self, "llm", None))
        _cleanup_cuda_memory()

    @modal.method()
    def capture_batch(
        self,
        rows: list[dict],
        layers: list[int] | None = None,
        capture_router: bool = True,
        capture_residual: bool = True,
        capture_generation: bool = False,
        generation_max_tokens: int = 256,
        generation_temperature: float = 0.0,
        generation_top_p: float = 1.0,
        pool_on_capture: str | None = None,
        router_top_k: int = 8,
        router_dtype: str = "float16",
        output_subdir: str = "",
    ) -> list[dict]:
        """Capture activations for a batch of examples."""
        import hashlib
        import json
        import time
        from datetime import UTC, datetime
        from pathlib import Path

        from pipelines.interp.local_capture import (
            _apply_pooling,
            _artifact_basename_for_row,
            _parse_messages,
            _save_activations,
            _save_router,
        )
        from pipelines.interp.modal_vllm_engine import (
            VLLMCaptureConfig,
            _capture_one_vllm,
        )

        output_dir = Path("/data/activations")
        if output_subdir:
            output_dir = output_dir / output_subdir
        residual_dir = output_dir / "residual_stream"
        router_dir = output_dir / "router_logits"
        if capture_residual:
            residual_dir.mkdir(parents=True, exist_ok=True)
        if capture_router and self.is_moe:
            router_dir.mkdir(parents=True, exist_ok=True)

        config = VLLMCaptureConfig(
            output_dir=output_dir,
            model_id=self.model_id,
            add_generation_prompt=self.add_generation_prompt,
            layers=layers or self.capture_layer_ids or None,
            capture_router=capture_router and self.is_moe,
            capture_residual=capture_residual,
            capture_generation=capture_generation,
            capture_reasoning=self.capture_reasoning and capture_generation,
            reasoning_parser=self.reasoning_parser,
            generation_max_tokens=generation_max_tokens,
            generation_temperature=generation_temperature,
            generation_top_p=generation_top_p,
            pool_on_capture=pool_on_capture,
            router_top_k=router_top_k,
            router_dtype=router_dtype,
        )

        metadata_rows: list[dict] = []
        for row in rows:
            log_id = row.get("log_id")
            if log_id is None:
                continue
            artifact_id = _artifact_basename_for_row(row)

            messages = _parse_messages(row)
            if not messages:
                continue
            try:
                t0 = time.monotonic()
                residual, router_logits, router_indices, input_ids, generation_result = _capture_one_vllm(
                    llm=self.llm,
                    tokenizer=self.tokenizer,
                    messages=messages,
                    config=config,
                    log_id=artifact_id,
                )
                elapsed = time.monotonic() - t0
                seq_len = len(input_ids)
                if pool_on_capture:
                    residual, router_logits, router_indices = _apply_pooling(
                        residual,
                        router_logits,
                        router_indices,
                        pool_on_capture,
                        input_ids=input_ids,
                        eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                    )

                file_size = 0
                if residual is not None and pool_on_capture:
                    file_size += _save_activations(residual, residual_dir / f"{artifact_id}.safetensors")
                elif residual is not None:
                    residual_path = residual_dir / f"{artifact_id}.safetensors"
                    if residual_path.exists():
                        file_size += residual_path.stat().st_size

                if router_logits is not None and router_indices is not None:
                    file_size += _save_router(
                        router_logits,
                        router_indices,
                        router_dir / f"{artifact_id}.safetensors",
                        router_dtype=router_dtype,
                    )

                prompt_hash = hashlib.sha256(bytes(json.dumps(input_ids), "utf-8")).hexdigest()
                effective_layers = layers or self.capture_layer_ids or list(range(self.num_layers))
                captured_layers = sorted(effective_layers)
                meta_row: dict = {
                    "log_id": int(log_id),
                    "row_key": row.get("row_key"),
                    "artifact_id": artifact_id,
                    "source_prompt_hash": row.get("source_prompt_hash"),
                    "source_relation": row.get("source_relation"),
                    "workflow_spec_id": row.get("workflow_spec_id"),
                    "workflow_spec_version": row.get("workflow_spec_version"),
                    "seq_len": seq_len,
                    "prompt_hash": prompt_hash,
                    "capture_timestamp": datetime.now(UTC).isoformat(),
                    "file_size_bytes": file_size,
                    "elapsed_s": round(elapsed, 2),
                    "has_router": router_logits is not None,
                    "captured_layers": json.dumps(captured_layers),
                    "pooling": pool_on_capture or "none",
                    "example_id": row.get("example_id"),
                    "generated_text": generation_result.get("generated_text", ""),
                    "generated_token_ids": json.dumps(generation_result.get("generated_token_ids", [])),
                    "finish_reason": generation_result.get("finish_reason", ""),
                    "reasoning_text": generation_result.get("reasoning_text", ""),
                }
                if residual is not None:
                    meta_row["num_layers_captured"] = int(residual.shape[0])
                    meta_row["hidden_dim"] = int(residual.shape[-1])
                else:
                    meta_row["num_layers_captured"] = int(router_logits.shape[0]) if router_logits is not None else 0
                    meta_row["hidden_dim"] = 0
                if router_logits is not None:
                    meta_row["num_experts"] = int(router_logits.shape[-1])
                metadata_rows.append(meta_row)
                if capture_generation:
                    print(
                        f"  generated {log_id}: finish_reason={generation_result.get('finish_reason', '')!r}, "
                        f"text_chars={len(generation_result.get('generated_text', '') or '')}, "
                        f"reasoning_chars={len(generation_result.get('reasoning_text', '') or '')}, "
                        f"elapsed={elapsed:.1f}s"
                    )
                else:
                    print(f"  {log_id}: seq_len={seq_len}, {file_size / 1024 / 1024:.1f}MB, {elapsed:.1f}s")
            except Exception as exc:
                import traceback

                # Once the vLLM engine core dies, every subsequent row tends to fail with
                # the same secondary EngineDeadError. Re-raise immediately so Modal surfaces
                # the first fatal cause instead of looping through the whole batch.
                if exc.__class__.__name__ == "EngineDeadError":
                    print(f"  FATAL {log_id}: vLLM engine died; aborting batch")
                    raise
                print(f"  ERROR {log_id}: {exc}")
                traceback.print_exc()
                if "Engine core initialization failed" in str(exc):
                    raise

        volume.commit()
        return metadata_rows

    @modal.method()
    def generate_batch(
        self,
        rows: list[dict],
        generation_max_tokens: int = 256,
        generation_temperature: float = 0.0,
        generation_top_p: float = 1.0,
    ) -> list[dict]:
        """Generate model responses for a batch without activation capture."""
        import json
        from datetime import UTC, datetime

        from pipelines.interp.local_capture import _parse_messages
        from pipelines.interp.modal_vllm_engine import (
            VLLMCaptureConfig,
            _generate_one_vllm,
        )

        config = VLLMCaptureConfig(
            model_id=self.model_id,
            add_generation_prompt=self.add_generation_prompt,
            capture_reasoning=self.capture_reasoning,
            reasoning_parser=self.reasoning_parser,
            capture_residual=False,
            capture_router=False,
        )

        metadata_rows: list[dict] = []
        for row in rows:
            log_id = row.get("log_id")
            if log_id is None:
                continue
            messages = _parse_messages(row)
            if not messages:
                continue
            try:
                generation_result = _generate_one_vllm(
                    llm=self.llm,
                    tokenizer=self.tokenizer,
                    messages=messages,
                    config=config,
                    max_tokens=generation_max_tokens,
                    temperature=generation_temperature,
                    top_p=generation_top_p,
                    top_k=-1,
                    chat_template_kwargs={"enable_thinking": self.enable_thinking},
                )
                metadata_rows.append(
                    {
                        "log_id": int(log_id),
                        "capture_timestamp": datetime.now(UTC).isoformat(),
                        "generated_text": generation_result.get("generated_text", ""),
                        "generated_token_ids": json.dumps(generation_result.get("generated_token_ids", [])),
                        "finish_reason": generation_result.get("finish_reason", ""),
                        "reasoning_text": (
                            generation_result.get("reasoning_text", "")
                            if self.capture_reasoning
                            else ""
                        ),
                    }
                )
            except Exception as exc:
                import traceback

                error_text = f"{exc.__class__.__name__}: {exc}"
                print(f"  ERROR generating {log_id}: {error_text}")
                traceback.print_exc()
                metadata_rows.append(
                    {
                        "log_id": int(log_id),
                        "capture_timestamp": datetime.now(UTC).isoformat(),
                        "generated_text": "",
                        "generated_token_ids": "[]",
                        "finish_reason": "engine_dead" if exc.__class__.__name__ == "EngineDeadError" else "error",
                        "reasoning_text": error_text,
                    }
                )
                if exc.__class__.__name__ == "EngineDeadError":
                    print("  Generation engine died; returning partial batch results")
                    break
        return metadata_rows



@app.function(volumes={"/data": volume}, image=image, timeout=300)
def inspect_volume(log_id: str = ""):
    """List and inspect captured activations on the volume."""
    from pathlib import Path

    from safetensors import safe_open

    base = Path("/data/activations")
    residual_dir = base / "residual_stream"
    router_dir = base / "router_logits"
    residual_files = sorted(residual_dir.glob("*.safetensors")) if residual_dir.exists() else []
    router_files = sorted(router_dir.glob("*.safetensors")) if router_dir.exists() else []

    print(f"Volume contents ({base}):")
    print(f"  residual_stream/: {len(residual_files)} files")
    print(f"  router_logits/:   {len(router_files)} files")
    if not log_id:
        for file_path in residual_files[:5]:
            print(f"    {file_path.name}  ({file_path.stat().st_size / 1024 / 1024:.1f}MB)")
        for file_path in router_files[:5]:
            print(f"    {file_path.name}  ({file_path.stat().st_size / 1024 / 1024:.1f}MB)")
        return

    print(f"\n--- log_id={log_id} ---")
    residual_path = residual_dir / f"{log_id}.safetensors"
    router_path = router_dir / f"{log_id}.safetensors"
    if residual_path.exists():
        with safe_open(str(residual_path), framework="pt") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                print(f"  residual_stream/{key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    else:
        print("  residual_stream: not found")
    if router_path.exists():
        with safe_open(str(router_path), framework="pt") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                print(f"  router_logits/{key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    else:
        print("  router_logits: not found")


def _residual_path_has_full_sequence_shape(path) -> bool:
    """Return True when a safetensors residual file stores [layer, seq, dim]."""
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as handle:
        if "residual_stream" not in handle.keys():
            return False
        shape = tuple(handle.get_slice("residual_stream").get_shape())
    return len(shape) == 3


def _limit_uncaptured_rows(
    rows: list[dict],
    completed: dict[str, dict[str, Any]],
    *,
    limit: int,
) -> list[dict]:
    from pipelines.interp.local_capture import _can_reuse_capture, _row_identity_token

    filtered = []
    for row in rows:
        token = _row_identity_token(row)
        existing = completed.get(token) if token is not None else None
        if not _can_reuse_capture(row, existing):
            filtered.append(row)
    if limit > 0:
        return filtered[:limit]
    return filtered


def _run_neon_transaction(transaction_fn, *, max_attempts: int = 2):
    """Run a Neon transaction on a fresh connection, retrying once on stale SSL."""
    from psycopg import OperationalError

    from pipelines.db import connect_neon

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        conn = connect_neon()
        try:
            result = transaction_fn(conn)
            conn.commit()
            return result
        except OperationalError as exc:
            last_exc = exc
            if attempt >= max_attempts:
                raise
            print(
                f"Neon transaction failed with OperationalError on attempt {attempt}/{max_attempts}; "
                "reconnecting and retrying once"
            )
        finally:
            conn.close()

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Neon transaction failed without raising an exception")


@app.function(volumes={"/data": volume}, image=image, timeout=300)
def write_metadata_to_volume(metadata_rows: list[dict], output_subdir: str = "") -> int:
    """Merge new metadata rows into metadata.parquet on the volume."""
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq_

    meta_path = Path("/data/activations")
    if output_subdir:
        meta_path = meta_path / output_subdir
    meta_path = meta_path / "metadata.parquet"
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict] = []
    if meta_path.exists():
        existing_rows = pq_.read_table(meta_path).to_pylist()

    from pipelines.interp.local_capture import _row_identity_token

    # Build lookup by stable row identity when present.
    by_id: dict[str, dict] = {}
    for row in existing_rows:
        token = _row_identity_token(row)
        if token is not None:
            by_id[token] = row
    for row in metadata_rows:
        token = _row_identity_token(row)
        if token is not None:
            by_id[token] = row

    merged = sorted(
        by_id.values(),
        key=lambda row: (str(row.get("row_key") or ""), int(row.get("log_id") or 0)),
    )
    pq_.write_table(pa.Table.from_pylist(merged), meta_path, compression="snappy")
    volume.commit()
    print(f"Wrote metadata to volume: {meta_path} ({len(merged)} rows, {len(metadata_rows)} new)")
    return len(merged)


@app.function(volumes={"/data": volume}, image=image, timeout=300)
def get_completed_log_ids(
    capture_router: bool = True,
    capture_residual: bool = True,
    pool_on_capture: str | None = None,
    output_subdir: str = "",
) -> dict[str, dict]:
    """Return reusable captured rows keyed by stable row identity."""
    from pathlib import Path

    import pyarrow.parquet as pq_

    from pipelines.interp.local_capture import (
        _artifact_basename_for_row,
        _row_identity_token,
    )

    base = Path("/data/activations")
    if output_subdir:
        base = base / output_subdir
    meta_path = base / "metadata.parquet"
    if not meta_path.exists():
        return {}

    rows = pq_.read_table(meta_path).to_pylist()
    completed: dict[str, dict] = {}
    expected_pooling = pool_on_capture or "none"
    residual_dir = base / "residual_stream"
    router_dir = base / "router_logits"

    for row in rows:
        artifact_id = _artifact_basename_for_row(row)
        token = _row_identity_token(row)
        if token is None:
            continue
        if row.get("pooling", "none") != expected_pooling:
            continue
        if capture_router:
            if not row.get("has_router", False):
                continue
            if not (router_dir / f"{artifact_id}.safetensors").exists():
                continue
        if capture_residual:
            residual_path = residual_dir / f"{artifact_id}.safetensors"
            if not residual_path.exists():
                continue
            if expected_pooling == "none" and not _residual_path_has_full_sequence_shape(residual_path):
                continue
        completed[token] = row

    return completed


@app.function(
    image=image,
    secrets=[neon_secret],
    volumes={"/data": volume},
    timeout=7200,
)
def run_vllm_capture(
    limit: int = 0,
    layers_str: str = "",
    capture_router: bool = True,
    capture_residual: bool = True,
    capture_generation: bool = False,
    capture_reasoning: bool = True,
    enable_thinking: bool = True,
    batch_size: int = 10,
    model_id: str = "Qwen/Qwen3-30B-A3B",
    add_generation_prompt: bool = False,
    pool: str = "",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: str = "0.90",
    max_model_len: int = 0,
    reasoning_parser: str = "",
    generation_max_tokens: int = 256,
    generation_temperature: float = 0.0,
    generation_top_p: float = 1.0,
    router_top_k: int = 8,
    router_dtype: str = "float16",
    gpu: str = "A100-80GB",
    max_containers: int = 0,
    cohort_view: str = "",
    order_mode: str = "log_id",
    source_relation: str = "",
    workflow_run_id: str = "",
    output_subdir: str = "",
) -> str:
    """Load examples from Neon, fan out to GPU workers, and write metadata."""
    from psycopg import sql
    from pipelines.interp.local_capture import (
        _can_reuse_capture,
        _load_examples_from_neon,
        _row_identity_token,
    )

    rows = _load_examples_from_neon(
        limit=None,
        source_relation=source_relation or None,
        cohort_view=cohort_view or None,
        order_mode=order_mode,
    )
    if not rows:
        return "No examples to capture"

    spec_id = str(rows[0].get("workflow_spec_id") or "workflow_capture").strip()
    table_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", spec_id).strip("_").lower() or "workflow_capture"
    response_table_name = f"capture_outputs_{table_slug}"
    metadata_run_id = str(workflow_run_id or output_subdir or source_relation or "adhoc").strip()

    parsed_layers: list[int] | None = None
    if layers_str:
        parsed_layers = [int(token.strip()) for token in layers_str.split(",") if token.strip()]
    pool_val = pool if pool else None

    # Check which rows are already captured with matching config
    print("Checking for existing captures on volume...")
    completed = get_completed_log_ids.remote(
        capture_router=capture_router,
        capture_residual=capture_residual,
        pool_on_capture=pool_val,
        output_subdir=output_subdir,
    )
    total_rows = len(rows)
    uncaptured = []
    for row in rows:
        token = _row_identity_token(row)
        existing = completed.get(token) if token is not None else None
        if not _can_reuse_capture(row, existing):
            uncaptured.append(row)
    already_captured = total_rows - len(uncaptured)
    rows = _limit_uncaptured_rows(rows, completed, limit=limit)
    if already_captured > 0:
        print(f"  Skipping {already_captured} already-captured examples ({len(uncaptured)} uncaptured available)")
    else:
        print(f"  No existing captures found, {len(uncaptured)} uncaptured examples available")
    if limit > 0 and len(uncaptured) > limit:
        print(f"  Limiting run to first {len(rows)} uncaptured examples")
    if not rows:
        return f"All {total_rows} examples already captured. Nothing to do."

    batches = [rows[index:index + batch_size] for index in range(0, len(rows), batch_size)]
    print(f"  {len(batches)} batches of up to {batch_size}")

    worker_kwargs: dict = {"gpu": gpu}
    if max_containers > 0:
        worker_kwargs["max_containers"] = max_containers
    WorkerCls = VLLMCaptureWorker.with_options(**worker_kwargs)
    worker = WorkerCls(
        model_id=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=str(gpu_memory_utilization),
        enforce_eager=True,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        max_model_len=max_model_len,
        capture_residual=capture_residual,
        capture_layers_csv=",".join(str(layer) for layer in parsed_layers) if parsed_layers else "",
        init_router_capture=True,
        # Keep activation capture anchored to the raw prompt rather than the
        # assistant-start token(s) added for decoding.
        add_generation_prompt=False,
        capture_reasoning=capture_reasoning,
        enable_thinking=enable_thinking,
        reasoning_parser=reasoning_parser,
    )
    generation_worker = None
    if capture_generation:
        generation_worker = WorkerCls(
            model_id=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=str(gpu_memory_utilization),
            enforce_eager=False,
            max_num_seqs=4,
            enable_chunked_prefill=True,
            max_model_len=max_model_len,
            capture_residual=False,
            capture_layers_csv="",
            init_router_capture=False,
            add_generation_prompt=add_generation_prompt,
            capture_reasoning=capture_reasoning,
            enable_thinking=enable_thinking,
            reasoning_parser=reasoning_parser,
        )

    all_metadata: list[dict] = []
    def _ensure_tables(conn) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS capture_metadata (
                    run_id                TEXT NOT NULL DEFAULT '',
                    log_id                INT PRIMARY KEY,
                    row_key               TEXT,
                    artifact_id           TEXT,
                    source_prompt_hash    TEXT,
                    source_relation       TEXT,
                    workflow_spec_id      TEXT,
                    workflow_spec_version INT,
                    seq_len               INT NOT NULL,
                    prompt_hash           TEXT,
                    capture_timestamp     TIMESTAMPTZ NOT NULL DEFAULT now(),
                    file_size_bytes       BIGINT NOT NULL DEFAULT 0,
                    elapsed_s             REAL NOT NULL DEFAULT 0,
                    has_router            BOOLEAN NOT NULL DEFAULT false,
                    captured_layers       TEXT,
                    pooling               TEXT NOT NULL DEFAULT 'none',
                    num_layers_captured INT NOT NULL DEFAULT 0,
                    hidden_dim          INT NOT NULL DEFAULT 0,
                    num_experts         INT
                )
                """
            )
            # Older deployments may already have capture_metadata with a narrower schema.
            # Keep the table forward-compatible rather than requiring a manual drop.
            for ddl in (
                "ALTER TABLE capture_metadata ADD COLUMN IF NOT EXISTS run_id TEXT NOT NULL DEFAULT ''",
                "ALTER TABLE capture_metadata ADD COLUMN IF NOT EXISTS row_key TEXT",
                "ALTER TABLE capture_metadata ADD COLUMN IF NOT EXISTS artifact_id TEXT",
                "ALTER TABLE capture_metadata ADD COLUMN IF NOT EXISTS source_prompt_hash TEXT",
                "ALTER TABLE capture_metadata ADD COLUMN IF NOT EXISTS source_relation TEXT",
                "ALTER TABLE capture_metadata ADD COLUMN IF NOT EXISTS workflow_spec_id TEXT",
                "ALTER TABLE capture_metadata ADD COLUMN IF NOT EXISTS workflow_spec_version INT",
                "ALTER TABLE capture_metadata ADD COLUMN IF NOT EXISTS prompt_hash TEXT",
                "ALTER TABLE capture_metadata ADD COLUMN IF NOT EXISTS captured_layers TEXT",
                "ALTER TABLE capture_metadata ADD COLUMN IF NOT EXISTS pooling TEXT NOT NULL DEFAULT 'none'",
                "ALTER TABLE capture_metadata ADD COLUMN IF NOT EXISTS num_layers_captured INT NOT NULL DEFAULT 0",
                "ALTER TABLE capture_metadata ADD COLUMN IF NOT EXISTS hidden_dim INT NOT NULL DEFAULT 0",
                "ALTER TABLE capture_metadata ADD COLUMN IF NOT EXISTS num_experts INT",
                "ALTER TABLE capture_metadata DROP CONSTRAINT IF EXISTS capture_metadata_pkey",
                "CREATE UNIQUE INDEX IF NOT EXISTS capture_metadata_run_log_id_idx ON capture_metadata (run_id, log_id)",
                "CREATE INDEX IF NOT EXISTS capture_metadata_log_id_idx ON capture_metadata (log_id)",
                "CREATE INDEX IF NOT EXISTS capture_metadata_source_row_idx ON capture_metadata (source_relation, row_key)",
            ):
                cur.execute(ddl)
            if capture_generation:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {} (
                            run_id                TEXT NOT NULL,
                            log_id                INT NOT NULL,
                            example_id            TEXT,
                            row_key               TEXT,
                            source_relation       TEXT,
                            workflow_spec_id      TEXT NOT NULL,
                            workflow_spec_version INT,
                            model_id              TEXT NOT NULL,
                            reasoning_parser      TEXT,
                            capture_timestamp     TIMESTAMPTZ NOT NULL DEFAULT now(),
                            generated_text        TEXT NOT NULL DEFAULT '',
                            generated_token_ids   JSONB NOT NULL DEFAULT '[]'::jsonb,
                            finish_reason         TEXT NOT NULL DEFAULT '',
                            reasoning_text        TEXT NOT NULL DEFAULT '',
                            PRIMARY KEY (run_id, log_id)
                        )
                        """
                    ).format(sql.Identifier(response_table_name))
                )

    _run_neon_transaction(_ensure_tables)

    def _flush_capture_batch(conn, batch_rows: list[dict]) -> None:
        with conn.cursor() as cur:
            for row in batch_rows:
                cur.execute(
                    """
                    INSERT INTO capture_metadata
                        (run_id, log_id, row_key, artifact_id, source_prompt_hash, source_relation,
                         workflow_spec_id, workflow_spec_version,
                         seq_len, prompt_hash, capture_timestamp,
                         file_size_bytes, elapsed_s, has_router, captured_layers,
                         pooling, num_layers_captured, hidden_dim, num_experts)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, log_id) DO UPDATE SET
                        row_key = EXCLUDED.row_key,
                        artifact_id = EXCLUDED.artifact_id,
                        source_prompt_hash = EXCLUDED.source_prompt_hash,
                        source_relation = EXCLUDED.source_relation,
                        workflow_spec_id = EXCLUDED.workflow_spec_id,
                        workflow_spec_version = EXCLUDED.workflow_spec_version,
                        seq_len = EXCLUDED.seq_len,
                        file_size_bytes = EXCLUDED.file_size_bytes,
                        elapsed_s = EXCLUDED.elapsed_s,
                        has_router = EXCLUDED.has_router,
                        captured_layers = EXCLUDED.captured_layers,
                        pooling = EXCLUDED.pooling,
                        num_layers_captured = EXCLUDED.num_layers_captured,
                        hidden_dim = EXCLUDED.hidden_dim,
                        num_experts = EXCLUDED.num_experts,
                        capture_timestamp = EXCLUDED.capture_timestamp
                    """,
                    (
                        metadata_run_id,
                        row["log_id"],
                        row.get("row_key"),
                        row.get("artifact_id"),
                        row.get("source_prompt_hash"),
                        row.get("source_relation"),
                        row.get("workflow_spec_id"),
                        row.get("workflow_spec_version"),
                        row["seq_len"],
                        row.get("prompt_hash"),
                        row["capture_timestamp"],
                        row["file_size_bytes"],
                        row["elapsed_s"],
                        row.get("has_router", False),
                        row.get("captured_layers"),
                        row.get("pooling", "none"),
                        row.get("num_layers_captured", 0),
                        row.get("hidden_dim", 0),
                        row.get("num_experts"),
                    ),
                )

    def _flush_generation_batch(conn, batch_rows: list[dict]) -> None:
        with conn.cursor() as cur:
            for row in batch_rows:
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {} (
                            run_id, log_id, example_id, row_key, source_relation,
                            workflow_spec_id, workflow_spec_version, model_id, reasoning_parser,
                            capture_timestamp, generated_text, generated_token_ids,
                            finish_reason, reasoning_text
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                        ON CONFLICT (run_id, log_id) DO UPDATE SET
                            example_id = EXCLUDED.example_id,
                            row_key = EXCLUDED.row_key,
                            source_relation = EXCLUDED.source_relation,
                            workflow_spec_id = EXCLUDED.workflow_spec_id,
                            workflow_spec_version = EXCLUDED.workflow_spec_version,
                            model_id = EXCLUDED.model_id,
                            reasoning_parser = EXCLUDED.reasoning_parser,
                            capture_timestamp = EXCLUDED.capture_timestamp,
                            generated_text = EXCLUDED.generated_text,
                            generated_token_ids = EXCLUDED.generated_token_ids,
                            finish_reason = EXCLUDED.finish_reason,
                            reasoning_text = EXCLUDED.reasoning_text
                        """
                    ).format(sql.Identifier(response_table_name)),
                    (
                        workflow_run_id,
                        row["log_id"],
                        row.get("example_id"),
                        row.get("row_key"),
                        row.get("source_relation"),
                        row.get("workflow_spec_id"),
                        row.get("workflow_spec_version"),
                        model_id,
                        reasoning_parser if capture_reasoning else "",
                        row["capture_timestamp"],
                        row.get("generated_text", ""),
                        row.get("generated_token_ids", "[]"),
                        row.get("finish_reason", ""),
                        row.get("reasoning_text", ""),
                    ),
                )

    capture_iter = worker.capture_batch.map(
        batches,
        kwargs=dict(
            layers=parsed_layers,
            capture_router=capture_router,
            capture_residual=capture_residual,
            pool_on_capture=pool_val,
            router_top_k=router_top_k,
            router_dtype=router_dtype,
            output_subdir=output_subdir,
        ),
    )
    for batch_meta in capture_iter:
        all_metadata.extend(batch_meta)
        if batch_meta:
            _run_neon_transaction(lambda conn, batch_rows=batch_meta: _flush_capture_batch(conn, batch_rows))
            print(f"  Flushed {len(batch_meta)} capture rows to Neon ({len(all_metadata)} total)")
        if len(all_metadata) % 50 == 0 or len(all_metadata) == len(rows):
            write_metadata_to_volume.remote(all_metadata, output_subdir=output_subdir)

    generated_rows = 0
    if capture_generation and generation_worker is not None:
        generation_iter = generation_worker.generate_batch.map(
            batches,
            kwargs=dict(
                generation_max_tokens=generation_max_tokens,
                generation_temperature=generation_temperature,
                generation_top_p=generation_top_p,
            ),
        )
        for source_batch, generation_meta in zip(batches, generation_iter, strict=True):
            source_by_log_id = {
                int(row["log_id"]): row for row in source_batch if row.get("log_id") is not None
            }
            enriched_generation_rows: list[dict] = []
            for row in generation_meta:
                log_id = row.get("log_id")
                if log_id is None:
                    continue
                source_row = source_by_log_id.get(int(log_id))
                if source_row is None:
                    continue
                enriched_generation_rows.append(
                    {
                        "log_id": int(log_id),
                        "example_id": source_row.get("example_id"),
                        "row_key": source_row.get("row_key"),
                        "source_relation": source_row.get("source_relation"),
                        "workflow_spec_id": source_row.get("workflow_spec_id"),
                        "workflow_spec_version": source_row.get("workflow_spec_version"),
                        "capture_timestamp": row.get("capture_timestamp") or source_row.get("capture_timestamp") or "",
                        "generated_text": row.get("generated_text", ""),
                        "generated_token_ids": row.get("generated_token_ids", "[]"),
                        "finish_reason": row.get("finish_reason", ""),
                        "reasoning_text": row.get("reasoning_text", ""),
                    }
                )
            if enriched_generation_rows:
                _run_neon_transaction(
                    lambda conn, batch_rows=enriched_generation_rows: _flush_generation_batch(conn, batch_rows)
                )
                generated_rows += len(enriched_generation_rows)
                sample = enriched_generation_rows[0]
                print(
                    f"  Flushed {len(enriched_generation_rows)} generation rows to {response_table_name} "
                    f"({generated_rows} total); sample text_chars={len(sample.get('generated_text', '') or '')}, "
                    f"reasoning_chars={len(sample.get('reasoning_text', '') or '')}"
                )

    if all_metadata:
        write_metadata_to_volume.remote(all_metadata, output_subdir=output_subdir)

    summary = f"Done: {len(all_metadata)} examples captured, {already_captured} skipped (already done)"
    print(f"\n{summary}")
    return summary


@app.local_entrypoint()
def main(
    mode: str = "capture",
    limit: int = 0,
    layers: str = "",
    capture_router: bool = True,
    capture_residual: bool = True,
    capture_generation: bool = False,
    capture_reasoning: bool = True,
    enable_thinking: bool = True,
    batch_size: int = 10,
    model_id: str = "Qwen/Qwen3-30B-A3B",
    add_generation_prompt: bool = False,
    pool: str = "",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: str = "0.90",
    max_model_len: int = 0,
    reasoning_parser: str = "",
    generation_max_tokens: int = 256,
    generation_temperature: float = 0.0,
    generation_top_p: float = 1.0,
    router_top_k: int = 8,
    router_dtype: str = "float16",
    gpu: str = "A100-80GB",
    max_containers: int = 0,
    cohort_view: str = "",
    order_mode: str = "log_id",
    source_relation: str = "",
    workflow_run_id: str = "",
    output_subdir: str = "",
):
    if mode != "capture":
        print(f"Unknown mode: {mode}. Use 'capture'.")
        return
    result = run_vllm_capture.remote(
        limit=limit,
        layers_str=layers,
        capture_router=capture_router,
        capture_residual=capture_residual,
        capture_generation=capture_generation,
        capture_reasoning=capture_reasoning,
        enable_thinking=enable_thinking,
        batch_size=batch_size,
        model_id=model_id,
        add_generation_prompt=add_generation_prompt,
        pool=pool,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        reasoning_parser=reasoning_parser,
        generation_max_tokens=generation_max_tokens,
        generation_temperature=generation_temperature,
        generation_top_p=generation_top_p,
        router_top_k=router_top_k,
        router_dtype=router_dtype,
        gpu=gpu,
        max_containers=max_containers,
        cohort_view=cohort_view,
        order_mode=order_mode,
        source_relation=source_relation,
        workflow_run_id=workflow_run_id,
        output_subdir=output_subdir,
    )
    print(result)
