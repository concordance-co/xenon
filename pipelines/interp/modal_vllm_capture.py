"""Modal deployment for vLLM-based activation capture on A100-80GB.

Usage (via Modal CLI)::

    # Download model weights to persistent volume (one-time)
    modal run pipelines/interp/modal_vllm_capture.py::download_model

    # Smoke test: 1 example
    modal run pipelines/interp/modal_vllm_capture.py --limit 1

    # Full capture: residual + router
    modal run pipelines/interp/modal_vllm_capture.py

    # Router logits only
    modal run pipelines/interp/modal_vllm_capture.py --no-capture-residual

    # Inspect volume contents
    modal run pipelines/interp/modal_vllm_capture.py::inspect_volume
"""

import modal

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
    """Download model weights to the persistent volume (one-time)."""
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
    """Modal worker that holds a vLLM engine in memory across requests.

    The engine is created once in ``@modal.enter()`` with all the
    constraints needed for correct activation capture (eager mode,
    max_num_seqs=1, etc.).
    """

    model_id: str = modal.parameter(default="Qwen/Qwen3-30B-A3B")
    tensor_parallel_size: int = modal.parameter(default=1)
    # Modal doesn't support float parameters — pass as str, convert in setup
    gpu_memory_utilization: str = modal.parameter(default="0.90")
    max_model_len: int = modal.parameter(default=0)  # 0 = use model default
    capture_residual: bool = modal.parameter(default=True)

    @modal.enter()
    def setup(self):
        """Create vLLM engine and set up router capture."""
        from pathlib import Path

        from transformers import AutoConfig, AutoTokenizer
        from vllm import LLM

        from pipelines.interp.vllm_capture import _init_router_capture_on_model

        local_path = f"/models/{self.model_id}"

        # Load tokenizer
        print(f"Loading tokenizer from {local_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)

        # Always resolve layer count for metadata
        hf_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=True
        )
        self.num_layers = hf_config.num_hidden_layers

        # Build vLLM engine config
        kwargs: dict = {
            "model": local_path,
            "enforce_eager": True,
            "max_num_seqs": 1,
            "enable_chunked_prefill": False,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": float(self.gpu_memory_utilization),
        }

        # Only set up extract_hidden_states when residual capture is enabled.
        # The speculative decode + KV cache for all layers uses significant
        # GPU memory, so skip it when only router capture is needed.
        if self.capture_residual:
            layer_ids = list(range(self.num_layers))

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
            print(f"  Residual capture: {self.num_layers} layers")

        if self.max_model_len > 0:
            kwargs["max_model_len"] = self.max_model_len

        print(f"Creating vLLM engine: {local_path}")
        self.llm = LLM(**kwargs)

        # Set up router capture via apply_model (no direct model access)
        # Size buffer to max_model_len so no sequence can overflow
        buffer_size = self.max_model_len if self.max_model_len > 0 else 32768
        self.is_moe = _init_router_capture_on_model(self.llm, max_tokens=buffer_size)
        if self.is_moe:
            print("Router capture enabled on MoE blocks")
        else:
            print("Model has no MoE blocks; router capture disabled")

    @modal.method()
    def capture_batch(
        self,
        rows: list[dict],
        layers: list[int] | None = None,
        capture_router: bool = True,
        capture_residual: bool = True,
        pool_on_capture: str | None = None,
        router_top_k: int = 8,
        router_dtype: str = "float16",
    ) -> list[dict]:
        """Capture activations for a batch of examples.

        Each example is processed one at a time (required by max_num_seqs=1).
        Returns a list of metadata dicts for successfully captured examples.
        """
        import hashlib
        import json
        import time
        from datetime import UTC, datetime
        from pathlib import Path

        from pipelines.interp.capture import (
            _apply_pooling,
            _parse_messages,
            _save_activations,
            _save_router,
        )
        from pipelines.interp.vllm_capture import _capture_one_vllm, VLLMCaptureConfig

        output_dir = Path("/data/activations")
        residual_dir = output_dir / "residual_stream"
        router_dir = output_dir / "router_logits"

        if capture_residual:
            residual_dir.mkdir(parents=True, exist_ok=True)
        if capture_router and self.is_moe:
            router_dir.mkdir(parents=True, exist_ok=True)

        config = VLLMCaptureConfig(
            output_dir=output_dir,
            model_id=self.model_id,
            layers=layers,
            capture_router=capture_router and self.is_moe,
            capture_residual=capture_residual,
            pool_on_capture=pool_on_capture,
            router_top_k=router_top_k,
            router_dtype=router_dtype,
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
                t0 = time.monotonic()
                residual, router_logits, router_indices, input_ids = (
                    _capture_one_vllm(
                        llm=self.llm,
                        tokenizer=self.tokenizer,
                        messages=messages,
                        config=config,
                        log_id=log_id,
                    )
                )
                elapsed = time.monotonic() - t0

                seq_len = len(input_ids)
                if pool_on_capture:
                    residual, router_logits, router_indices = _apply_pooling(
                        residual, router_logits, router_indices, pool_on_capture
                    )

                file_size = 0

                # Residual may already be saved; re-save if pooled
                if residual is not None and pool_on_capture:
                    file_size += _save_activations(
                        residual, residual_dir / f"{log_id}.safetensors"
                    )
                elif residual is not None:
                    p = residual_dir / f"{log_id}.safetensors"
                    if p.exists():
                        file_size += p.stat().st_size

                if router_logits is not None and router_indices is not None:
                    file_size += _save_router(
                        router_logits,
                        router_indices,
                        router_dir / f"{log_id}.safetensors",
                        router_dtype=router_dtype,
                    )

                prompt_hash = hashlib.sha256(
                    bytes(json.dumps(input_ids), "utf-8")
                ).hexdigest()

                captured_layers = (
                    sorted(layers) if layers is not None
                    else list(range(self.num_layers))
                )

                meta_row: dict = {
                    "log_id": int(log_id),
                    "seq_len": seq_len,
                    "prompt_hash": prompt_hash,
                    "capture_timestamp": datetime.now(UTC).isoformat(),
                    "file_size_bytes": file_size,
                    "elapsed_s": round(elapsed, 2),
                    "has_router": router_logits is not None,
                    "captured_layers": json.dumps(captured_layers),
                    "pooling": pool_on_capture or "none",
                }
                if residual is not None:
                    meta_row["num_layers_captured"] = int(residual.shape[0])
                    meta_row["hidden_dim"] = int(residual.shape[-1])
                else:
                    meta_row["num_layers_captured"] = (
                        int(router_logits.shape[0])
                        if router_logits is not None
                        else 0
                    )
                    meta_row["hidden_dim"] = 0
                if router_logits is not None:
                    meta_row["num_experts"] = int(router_logits.shape[-1])

                metadata_rows.append(meta_row)
                print(
                    f"  {log_id}: seq_len={seq_len}, "
                    f"{file_size / 1024 / 1024:.1f}MB, {elapsed:.1f}s"
                )

            except Exception as exc:
                import traceback

                print(f"  ERROR {log_id}: {exc}")
                traceback.print_exc()

        volume.commit()
        return metadata_rows


@app.function(volumes={"/data": volume}, image=image, timeout=300)
def inspect_volume(log_id: str = ""):
    """List and inspect captured activations on the volume."""
    from pathlib import Path

    from safetensors import safe_open

    base = Path("/data/activations")
    residual_dir = base / "residual_stream"
    router_dir = base / "router_logits"

    residual_files = (
        sorted(residual_dir.glob("*.safetensors"))
        if residual_dir.exists()
        else []
    )
    router_files = (
        sorted(router_dir.glob("*.safetensors"))
        if router_dir.exists()
        else []
    )

    print(f"Volume contents ({base}):")
    print(f"  residual_stream/: {len(residual_files)} files")
    print(f"  router_logits/:   {len(router_files)} files")

    if not log_id:
        for f in residual_files[:5]:
            print(f"    {f.name}  ({f.stat().st_size / 1024 / 1024:.1f}MB)")
        for f in router_files[:5]:
            print(f"    {f.name}  ({f.stat().st_size / 1024 / 1024:.1f}MB)")
        return

    print(f"\n--- log_id={log_id} ---")
    residual_path = residual_dir / f"{log_id}.safetensors"
    router_path = router_dir / f"{log_id}.safetensors"

    if residual_path.exists():
        with safe_open(str(residual_path), framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                print(
                    f"  residual_stream/{key}: "
                    f"shape={tuple(t.shape)}, dtype={t.dtype}"
                )
    else:
        print("  residual_stream: not found")

    if router_path.exists():
        with safe_open(str(router_path), framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                print(
                    f"  router_logits/{key}: "
                    f"shape={tuple(t.shape)}, dtype={t.dtype}"
                )
    else:
        print("  router_logits: not found")


@app.function(volumes={"/data": volume}, image=image, timeout=300)
def write_metadata_to_volume(metadata_rows: list[dict]) -> int:
    """Merge new metadata rows into metadata.parquet on the volume."""
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq_

    meta_path = Path("/data/activations/metadata.parquet")
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing metadata
    existing_rows: list[dict] = []
    if meta_path.exists():
        existing_rows = pq_.read_table(meta_path).to_pylist()

    # Build lookup by log_id — new rows overwrite old ones
    by_id = {r["log_id"]: r for r in existing_rows}
    for r in metadata_rows:
        by_id[r["log_id"]] = r

    merged = sorted(by_id.values(), key=lambda r: r["log_id"])
    table = pa.Table.from_pylist(merged)
    pq_.write_table(table, meta_path, compression="snappy")
    volume.commit()
    print(f"Wrote metadata to volume: {meta_path} ({len(merged)} rows, {len(metadata_rows)} new)")
    return len(merged)


@app.function(volumes={"/data": volume}, image=image, timeout=300)
def get_completed_log_ids(
    capture_router: bool = True,
    capture_residual: bool = True,
    pool_on_capture: str | None = None,
) -> set[int]:
    """Return log_ids that already have captures on the volume."""
    from pathlib import Path

    import pyarrow.parquet as pq_

    base = Path("/data/activations")
    meta_path = base / "metadata.parquet"
    if not meta_path.exists():
        return set()

    table = pq_.read_table(meta_path)
    rows = table.to_pylist()
    completed = set()
    expected_pooling = pool_on_capture or "none"

    residual_dir = base / "residual_stream"
    router_dir = base / "router_logits"

    for r in rows:
        log_id = r["log_id"]
        if r.get("pooling", "none") != expected_pooling:
            continue
        if capture_router:
            if not r.get("has_router", False):
                continue
            if not (router_dir / f"{log_id}.safetensors").exists():
                continue
        if capture_residual:
            if not (residual_dir / f"{log_id}.safetensors").exists():
                continue
        completed.add(log_id)

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
    batch_size: int = 10,
    model_id: str = "Qwen/Qwen3-30B-A3B",
    pool: str = "",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: str = "0.90",
    max_model_len: int = 0,
    router_top_k: int = 8,
    router_dtype: str = "float16",
) -> str:
    """Orchestrator: load examples from Neon, fan out to GPU workers, write metadata."""
    from pipelines.interp.capture import _load_examples_from_neon

    rows = _load_examples_from_neon(limit=limit if limit > 0 else None)
    if not rows:
        return "No examples to capture"

    parsed_layers: list[int] | None = None
    if layers_str:
        parsed_layers = [int(x.strip()) for x in layers_str.split(",")]

    pool_val = pool if pool else None

    # Check which log_ids are already captured with matching config
    print("Checking for existing captures on volume...")
    completed = get_completed_log_ids.remote(
        capture_router=capture_router,
        capture_residual=capture_residual,
        pool_on_capture=pool_val,
    )
    before = len(rows)
    rows = [r for r in rows if r.get("log_id") not in completed]
    skipped = before - len(rows)
    if skipped > 0:
        print(f"  Skipping {skipped} already-captured examples ({len(rows)} remaining)")
    else:
        print(f"  No existing captures found, processing all {len(rows)} examples")

    if not rows:
        return f"All {before} examples already captured. Nothing to do."

    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    print(f"  {len(batches)} batches of up to {batch_size}")

    worker = VLLMCaptureWorker(
        model_id=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=str(gpu_memory_utilization),
        max_model_len=max_model_len,
        capture_residual=capture_residual,
    )

    all_metadata: list[dict] = []
    for batch_meta in worker.capture_batch.map(
        batches,
        kwargs=dict(
            layers=parsed_layers,
            capture_router=capture_router,
            capture_residual=capture_residual,
            pool_on_capture=pool_val,
            router_top_k=router_top_k,
            router_dtype=router_dtype,
        ),
    ):
        all_metadata.extend(batch_meta)

        # Incremental metadata flush to volume every batch
        if len(all_metadata) % 50 == 0 or len(all_metadata) == len(rows):
            write_metadata_to_volume.remote(all_metadata)

    if all_metadata:
        write_metadata_to_volume.remote(all_metadata)

    summary = f"Done: {len(all_metadata)} examples captured, {skipped} skipped (already done)"
    print(f"\n{summary}")
    return summary


@app.local_entrypoint()
def main(
    limit: int = 0,
    layers: str = "",
    capture_router: bool = True,
    capture_residual: bool = True,
    batch_size: int = 10,
    model_id: str = "Qwen/Qwen3-30B-A3B",
    pool: str = "",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: str = "0.90",
    max_model_len: int = 0,
    router_top_k: int = 8,
    router_dtype: str = "float16",
):
    result = run_vllm_capture.remote(
        limit=limit,
        layers_str=layers,
        capture_router=capture_router,
        capture_residual=capture_residual,
        batch_size=batch_size,
        model_id=model_id,
        pool=pool,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        router_top_k=router_top_k,
        router_dtype=router_dtype,
    )
    print(result)
