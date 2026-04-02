"""Modal deployment for synthetic market activation capture on a dedicated volume."""

import modal

app = modal.App("xenon-synthetic-vllm-capture")

synthetic_volume = modal.Volume.from_name("xenon-synthetic-data", create_if_missing=True)
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
    from huggingface_hub import snapshot_download

    local_dir = f"/models/{model_id}"
    print(f"Downloading {model_id} to {local_dir}...")
    snapshot_download(model_id, local_dir=local_dir)
    model_volume.commit()
    print(f"Done. Model cached at {local_dir}")


@app.cls(
    gpu="H200",
    volumes={"/data": synthetic_volume, "/models": model_volume},
    image=image,
    timeout=7200,
    scaledown_window=300,
)
class SyntheticVLLMCaptureWorker:
    model_id: str = modal.parameter(default="Qwen/Qwen3-30B-A3B")
    tensor_parallel_size: int = modal.parameter(default=1)
    gpu_memory_utilization: str = modal.parameter(default="0.90")
    max_model_len: int = modal.parameter(default=0)
    capture_residual: bool = modal.parameter(default=True)

    @modal.enter()
    def setup(self):
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
        print("Router capture enabled on MoE blocks" if self.is_moe else "Router capture disabled")

    @modal.method()
    def capture_batch(
        self,
        rows: list[dict],
        phase_name: str = "phase1",
        layers: list[int] | None = None,
        capture_router: bool = True,
        capture_residual: bool = True,
        pool_on_capture: str | None = None,
        router_top_k: int = 8,
        router_dtype: str = "float16",
    ) -> list[dict]:
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

        output_dir = Path(f"/data/activations/{phase_name}")
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
                        residual, router_logits, router_indices, pool_on_capture,
                    )

                file_size = 0
                if residual is not None and pool_on_capture:
                    file_size += _save_activations(
                        residual,
                        residual_dir / f"{log_id}.safetensors",
                    )
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

                prompt_hash = hashlib.sha256(bytes(json.dumps(input_ids), "utf-8")).hexdigest()[:16]
                metadata_rows.append({
                    "phase_name": phase_name,
                    "log_id": log_id,
                    "seq_len": seq_len,
                    "prompt_hash": prompt_hash,
                    "capture_timestamp": datetime.now(UTC).isoformat(),
                    "file_size_bytes": file_size,
                    "elapsed_s": round(elapsed, 3),
                    "has_router": bool(router_logits is not None and router_indices is not None),
                    "captured_layers": ",".join(str(layer) for layer in (layers or range(self.num_layers))),
                    "pooling": pool_on_capture or "none",
                    "num_layers_captured": len(layers or list(range(self.num_layers))),
                    "hidden_dim": residual.shape[-1] if residual is not None else 0,
                    "num_experts": int(router_logits.shape[-1]) if router_logits is not None else None,
                })
                print(
                    f"  captured log_id={log_id} seq={seq_len} "
                    f"router={router_logits is not None} size={file_size / 1024:.0f}KB "
                    f"elapsed={elapsed:.1f}s"
                )
            except Exception as exc:
                import traceback

                print(f"  ERROR log_id={log_id}: {exc}")
                traceback.print_exc()

        synthetic_volume.commit()
        return metadata_rows


def _residual_path_has_full_sequence_shape(path) -> bool:
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        if "residual_stream" not in f.keys():
            return False
        shape = tuple(f.get_slice("residual_stream").get_shape())
    return len(shape) == 3


def _limit_uncaptured_rows(rows: list[dict], completed: set[int], *, limit: int) -> list[dict]:
    filtered = [row for row in rows if row.get("log_id") not in completed]
    if limit > 0:
        return filtered[:limit]
    return filtered


@app.function(volumes={"/data": synthetic_volume}, image=image, timeout=300)
def write_metadata_to_volume(metadata_rows: list[dict], phase_name: str = "phase1") -> int:
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    meta_path = Path(f"/data/activations/{phase_name}/metadata.parquet")
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict] = []
    if meta_path.exists():
        existing_rows = pq.read_table(meta_path).to_pylist()

    by_id = {(row["phase_name"], row["log_id"]): row for row in existing_rows}
    for row in metadata_rows:
        by_id[(row["phase_name"], row["log_id"])] = row

    merged = sorted(by_id.values(), key=lambda row: (row["phase_name"], row["log_id"]))
    pq.write_table(pa.Table.from_pylist(merged), meta_path, compression="snappy")
    synthetic_volume.commit()
    print(f"Wrote metadata to volume: {meta_path} ({len(merged)} rows, {len(metadata_rows)} new)")
    return len(merged)


@app.function(volumes={"/data": synthetic_volume}, image=image, timeout=300)
def get_completed_log_ids(
    phase_name: str = "phase1",
    capture_router: bool = True,
    capture_residual: bool = True,
    pool_on_capture: str | None = None,
) -> set[int]:
    from pathlib import Path

    import pyarrow.parquet as pq

    base = Path(f"/data/activations/{phase_name}")
    meta_path = base / "metadata.parquet"
    if not meta_path.exists():
        return set()

    rows = pq.read_table(meta_path).to_pylist()
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


@app.function(volumes={"/data": synthetic_volume}, image=image, timeout=300)
def inspect_volume(log_id: int = 0, phase_name: str = "phase1"):
    from pathlib import Path

    from safetensors import safe_open

    base = Path(f"/data/activations/{phase_name}")
    residual_dir = base / "residual_stream"
    router_dir = base / "router_logits"

    if not log_id:
        print(f"Inspecting synthetic capture volume under {base}")
        print(f"  residual files: {len(list(residual_dir.glob('*.safetensors')))}")
        print(f"  router files: {len(list(router_dir.glob('*.safetensors')))}")
        meta_path = base / "metadata.parquet"
        print(f"  metadata: {'present' if meta_path.exists() else 'missing'}")
        return

    print(f"\n--- synthetic log_id={log_id} phase={phase_name} ---")
    residual_path = residual_dir / f"{log_id}.safetensors"
    router_path = router_dir / f"{log_id}.safetensors"

    if residual_path.exists():
        with safe_open(str(residual_path), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                print(f"  residual_stream/{key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    else:
        print("  residual_stream: not found")

    if router_path.exists():
        with safe_open(str(router_path), framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                print(f"  router_logits/{key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    else:
        print("  router_logits: not found")


@app.function(
    image=image,
    secrets=[neon_secret],
    volumes={"/data": synthetic_volume},
    timeout=7200,
)
def run_synthetic_vllm_capture(
    phase_name: str = "phase1",
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
    gpu: str = "H200",
    max_containers: int = 0,
    cohort_view: str = "synthetic_market_phase1_capture_v0",
    order_mode: str = "selection_rank_asc",
) -> str:
    from pipelines.interp.synthetic.db import load_examples_from_neon

    rows = load_examples_from_neon(
        limit=None,
        cohort_view=cohort_view or None,
        order_mode=order_mode,
    )
    if not rows:
        return "No synthetic examples to capture"

    parsed_layers: list[int] | None = None
    if layers_str:
        parsed_layers = [int(part.strip()) for part in layers_str.split(",")]
    pool_val = pool if pool else None

    print("Checking for existing synthetic captures on volume...")
    completed = get_completed_log_ids.remote(
        phase_name=phase_name,
        capture_router=capture_router,
        capture_residual=capture_residual,
        pool_on_capture=pool_val,
    )
    total_rows = len(rows)
    uncaptured = [row for row in rows if row.get("log_id") not in completed]
    already_captured = total_rows - len(uncaptured)
    rows = _limit_uncaptured_rows(rows, completed, limit=limit)
    if already_captured > 0:
        print(f"  Skipping {already_captured} already-captured synthetic examples ({len(uncaptured)} uncaptured available)")
    else:
        print(f"  No existing captures found, {len(uncaptured)} uncaptured synthetic examples available")
    if limit > 0 and len(uncaptured) > limit:
        print(f"  Limiting run to first {len(rows)} uncaptured examples")
    if not rows:
        return f"All {total_rows} synthetic examples already captured. Nothing to do."

    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    print(f"  {len(batches)} batches of up to {batch_size}")

    worker_kwargs: dict = {"gpu": gpu}
    if max_containers > 0:
        worker_kwargs["max_containers"] = max_containers

    WorkerCls = SyntheticVLLMCaptureWorker.with_options(**worker_kwargs)
    worker = WorkerCls(
        model_id=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=str(gpu_memory_utilization),
        max_model_len=max_model_len,
        capture_residual=capture_residual,
    )

    from pipelines.db import connect_neon

    db_conn = connect_neon()
    with db_conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS synthetic_capture_metadata (
                phase_name            TEXT NOT NULL,
                log_id                INT NOT NULL,
                seq_len               INT NOT NULL,
                prompt_hash           TEXT,
                capture_timestamp     TIMESTAMPTZ NOT NULL DEFAULT now(),
                file_size_bytes       BIGINT NOT NULL DEFAULT 0,
                elapsed_s             REAL NOT NULL DEFAULT 0,
                has_router            BOOLEAN NOT NULL DEFAULT false,
                captured_layers       TEXT,
                pooling               TEXT NOT NULL DEFAULT 'none',
                num_layers_captured   INT NOT NULL DEFAULT 0,
                hidden_dim            INT NOT NULL DEFAULT 0,
                num_experts           INT,
                PRIMARY KEY (phase_name, log_id)
            )
            """
        )
    db_conn.commit()

    all_metadata: list[dict] = []
    for batch_meta in worker.capture_batch.map(
        batches,
        kwargs=dict(
            phase_name=phase_name,
            layers=parsed_layers,
            capture_router=capture_router,
            capture_residual=capture_residual,
            pool_on_capture=pool_val,
            router_top_k=router_top_k,
            router_dtype=router_dtype,
        ),
    ):
        all_metadata.extend(batch_meta)
        if batch_meta:
            with db_conn.cursor() as cur:
                for row in batch_meta:
                    cur.execute(
                        """
                        INSERT INTO synthetic_capture_metadata (
                            phase_name, log_id, seq_len, prompt_hash, capture_timestamp,
                            file_size_bytes, elapsed_s, has_router, captured_layers,
                            pooling, num_layers_captured, hidden_dim, num_experts
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                        ON CONFLICT (phase_name, log_id) DO UPDATE SET
                            seq_len = EXCLUDED.seq_len,
                            prompt_hash = EXCLUDED.prompt_hash,
                            capture_timestamp = EXCLUDED.capture_timestamp,
                            file_size_bytes = EXCLUDED.file_size_bytes,
                            elapsed_s = EXCLUDED.elapsed_s,
                            has_router = EXCLUDED.has_router,
                            captured_layers = EXCLUDED.captured_layers,
                            pooling = EXCLUDED.pooling,
                            num_layers_captured = EXCLUDED.num_layers_captured,
                            hidden_dim = EXCLUDED.hidden_dim,
                            num_experts = EXCLUDED.num_experts
                        """,
                        (
                            row["phase_name"],
                            row["log_id"],
                            row["seq_len"],
                            row["prompt_hash"],
                            row["capture_timestamp"],
                            row["file_size_bytes"],
                            row["elapsed_s"],
                            row["has_router"],
                            row["captured_layers"],
                            row["pooling"],
                            row["num_layers_captured"],
                            row["hidden_dim"],
                            row["num_experts"],
                        ),
                    )
            db_conn.commit()
            print(f"  Flushed {len(batch_meta)} synthetic rows to Neon ({len(all_metadata)} total)")

        if len(all_metadata) % 50 == 0 or len(all_metadata) == len(rows):
            write_metadata_to_volume.remote(all_metadata, phase_name=phase_name)

    db_conn.close()

    if all_metadata:
        write_metadata_to_volume.remote(all_metadata, phase_name=phase_name)

    summary = f"Done: {len(all_metadata)} synthetic examples captured, {already_captured} skipped (already done)"
    print(f"\n{summary}")
    return summary


@app.local_entrypoint()
def main(
    mode: str = "capture",
    phase_name: str = "phase1",
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
    gpu: str = "H200",
    max_containers: int = 0,
    cohort_view: str = "synthetic_market_phase1_capture_v0",
    order_mode: str = "selection_rank_asc",
):
    if mode == "capture":
        result = run_synthetic_vllm_capture.remote(
            phase_name=phase_name,
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
        )
        print(result)
    else:
        print(f"Unknown mode: {mode}. Use 'capture'.")
