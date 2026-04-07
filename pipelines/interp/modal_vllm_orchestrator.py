"""Canonical Modal capture orchestrator for generic workflow capture runs."""

from __future__ import annotations

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
    max_model_len: int = modal.parameter(default=0)
    capture_residual: bool = modal.parameter(default=True)

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

        kwargs: dict = {
            "model": local_path,
            "enforce_eager": True,
            "max_num_seqs": 1,
            "enable_chunked_prefill": False,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": float(self.gpu_memory_utilization),
        }
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
            _parse_messages,
            _save_activations,
            _save_router,
        )
        from pipelines.interp.modal_vllm_engine import VLLMCaptureConfig, _capture_one_vllm

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
                residual, router_logits, router_indices, input_ids = _capture_one_vllm(
                    llm=self.llm,
                    tokenizer=self.tokenizer,
                    messages=messages,
                    config=config,
                    log_id=log_id,
                )
                elapsed = time.monotonic() - t0
                seq_len = len(input_ids)
                if pool_on_capture:
                    residual, router_logits, router_indices = _apply_pooling(
                        residual, router_logits, router_indices, pool_on_capture
                    )

                file_size = 0
                if residual is not None and pool_on_capture:
                    file_size += _save_activations(residual, residual_dir / f"{log_id}.safetensors")
                elif residual is not None:
                    residual_path = residual_dir / f"{log_id}.safetensors"
                    if residual_path.exists():
                        file_size += residual_path.stat().st_size
                if router_logits is not None and router_indices is not None:
                    file_size += _save_router(
                        router_logits,
                        router_indices,
                        router_dir / f"{log_id}.safetensors",
                        router_dtype=router_dtype,
                    )

                prompt_hash = hashlib.sha256(bytes(json.dumps(input_ids), "utf-8")).hexdigest()
                captured_layers = sorted(layers) if layers is not None else list(range(self.num_layers))
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
                    meta_row["num_layers_captured"] = int(router_logits.shape[0]) if router_logits is not None else 0
                    meta_row["hidden_dim"] = 0
                if router_logits is not None:
                    meta_row["num_experts"] = int(router_logits.shape[-1])
                metadata_rows.append(meta_row)
                print(f"  {log_id}: seq_len={seq_len}, {file_size / 1024 / 1024:.1f}MB, {elapsed:.1f}s")
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


def _limit_uncaptured_rows(rows: list[dict], completed: set[int], *, limit: int) -> list[dict]:
    filtered = [row for row in rows if row.get("log_id") not in completed]
    if limit > 0:
        return filtered[:limit]
    return filtered


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

    by_id = {row["log_id"]: row for row in existing_rows}
    for row in metadata_rows:
        by_id[row["log_id"]] = row

    merged = sorted(by_id.values(), key=lambda row: row["log_id"])
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
) -> set[int]:
    """Return log_ids that already have captures on the volume."""
    from pathlib import Path

    import pyarrow.parquet as pq_

    base = Path("/data/activations")
    if output_subdir:
        base = base / output_subdir
    meta_path = base / "metadata.parquet"
    if not meta_path.exists():
        return set()

    rows = pq_.read_table(meta_path).to_pylist()
    completed: set[int] = set()
    expected_pooling = pool_on_capture or "none"
    residual_dir = base / "residual_stream"
    router_dir = base / "router_logits"

    for row in rows:
        log_id = row["log_id"]
        if row.get("pooling", "none") != expected_pooling:
            continue
        if capture_router:
            if not row.get("has_router", False):
                continue
            if not (router_dir / f"{log_id}.safetensors").exists():
                continue
        if capture_residual:
            residual_path = residual_dir / f"{log_id}.safetensors"
            if not residual_path.exists():
                continue
            if expected_pooling == "none" and not _residual_path_has_full_sequence_shape(residual_path):
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
    gpu: str = "A100-80GB",
    max_containers: int = 0,
    cohort_view: str = "",
    order_mode: str = "log_id",
    source_relation: str = "",
    output_subdir: str = "",
) -> str:
    """Load examples from Neon, fan out to GPU workers, and write metadata."""
    from pipelines.interp.local_capture import _load_examples_from_neon
    from pipelines.db import connect_neon

    rows = _load_examples_from_neon(
        limit=None,
        source_relation=source_relation or None,
        cohort_view=cohort_view or None,
        order_mode=order_mode,
    )
    if not rows:
        return "No examples to capture"

    parsed_layers: list[int] | None = None
    if layers_str:
        parsed_layers = [int(token.strip()) for token in layers_str.split(",") if token.strip()]
    pool_val = pool if pool else None

    print("Checking for existing captures on volume...")
    completed = get_completed_log_ids.remote(
        capture_router=capture_router,
        capture_residual=capture_residual,
        pool_on_capture=pool_val,
        output_subdir=output_subdir,
    )
    total_rows = len(rows)
    uncaptured = [row for row in rows if row.get("log_id") not in completed]
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
        max_model_len=max_model_len,
        capture_residual=capture_residual,
    )

    all_metadata: list[dict] = []
    db_conn = connect_neon()
    with db_conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS capture_metadata (
                log_id              INT PRIMARY KEY,
                seq_len             INT NOT NULL,
                prompt_hash         TEXT,
                capture_timestamp   TIMESTAMPTZ NOT NULL DEFAULT now(),
                file_size_bytes     BIGINT NOT NULL DEFAULT 0,
                elapsed_s           REAL NOT NULL DEFAULT 0,
                has_router          BOOLEAN NOT NULL DEFAULT false,
                captured_layers     TEXT,
                pooling             TEXT NOT NULL DEFAULT 'none',
                num_layers_captured INT NOT NULL DEFAULT 0,
                hidden_dim          INT NOT NULL DEFAULT 0,
                num_experts         INT
            )
            """
        )
    db_conn.commit()

    for batch_meta in worker.capture_batch.map(
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
    ):
        all_metadata.extend(batch_meta)
        if batch_meta:
            with db_conn.cursor() as cur:
                for row in batch_meta:
                    cur.execute(
                        """
                        INSERT INTO capture_metadata
                            (log_id, seq_len, prompt_hash, capture_timestamp,
                             file_size_bytes, elapsed_s, has_router, captured_layers,
                             pooling, num_layers_captured, hidden_dim, num_experts)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (log_id) DO UPDATE SET
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
                            row["log_id"],
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
            db_conn.commit()
            print(f"  Flushed {len(batch_meta)} rows to Neon ({len(all_metadata)} total)")
        if len(all_metadata) % 50 == 0 or len(all_metadata) == len(rows):
            write_metadata_to_volume.remote(all_metadata, output_subdir=output_subdir)

    db_conn.close()
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
    batch_size: int = 10,
    model_id: str = "Qwen/Qwen3-30B-A3B",
    pool: str = "",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: str = "0.90",
    max_model_len: int = 0,
    router_top_k: int = 8,
    router_dtype: str = "float16",
    gpu: str = "A100-80GB",
    max_containers: int = 0,
    cohort_view: str = "",
    order_mode: str = "log_id",
    source_relation: str = "",
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
        batch_size=batch_size,
        model_id=model_id,
        pool=pool,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        router_top_k=router_top_k,
        router_dtype=router_dtype,
        gpu=gpu,
        max_containers=max_containers,
        cohort_view=cohort_view,
        order_mode=order_mode,
        source_relation=source_relation,
        output_subdir=output_subdir,
    )
    print(result)
