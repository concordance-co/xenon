"""Modal deployment for activation capture on A100-80GB.

Usage (via wrapper script):
    ./scripts/modal_capture.sh download     # cache model weights
    ./scripts/modal_capture.sh smoke        # 1 example, layer 24
    ./scripts/modal_capture.sh router       # router logits only
    ./scripts/modal_capture.sh full         # residual + router
    ./scripts/modal_capture.sh inspect      # list volume contents

Or directly:
    uv run --extra interp --extra modal modal run pipelines/interp/modal_capture.py --limit 1
"""

import modal

app = modal.App("xenon-activation-capture")

volume = modal.Volume.from_name("xenon-data", create_if_missing=True)
model_volume = modal.Volume.from_name("xenon-models", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch", "transformers", "safetensors", "pyarrow", "huggingface_hub")
    .add_local_python_source("pipelines")
)


@app.function(
    volumes={"/models": model_volume}, image=image, secrets=[hf_secret], timeout=1800,
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
    timeout=3600,
)
class CaptureWorker:
    model_id: str = modal.parameter(default="Qwen/Qwen3-30B-A3B")

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        local_path = f"/models/{self.model_id}"
        print(f"Loading tokenizer from {local_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(local_path)

        print(f"Loading model from {local_path} (float16 -> cuda)...")
        self.model = AutoModelForCausalLM.from_pretrained(
            local_path,
            dtype=torch.float16,
        ).to("cuda").eval()

        num_layers = len(self.model.model.layers)
        hidden_dim = self.model.config.hidden_size
        print(f"  {num_layers} layers, hidden_dim={hidden_dim}")

    @modal.method()
    def capture_batch(
        self,
        rows: list[dict],
        layers: list[int] | None = None,
        capture_router: bool = True,
        capture_residual: bool = True,
        pool_on_capture: str | None = None,
    ) -> list[dict]:
        import hashlib
        import json
        import time
        from datetime import UTC, datetime
        from pathlib import Path

        from pipelines.interp.capture import (
            CaptureConfig,
            _apply_pooling,
            _capture_one,
            _parse_messages,
            _save_activations,
            _save_router,
        )

        output_dir = Path("/data/activations")
        config = CaptureConfig(
            output_dir=output_dir,
            model_id=self.model_id,
            device="cuda",
            layers=layers,
            capture_router=capture_router,
            capture_residual=capture_residual,
            pool_on_capture=pool_on_capture,
        )

        residual_dir = output_dir / "residual_stream"
        router_dir = output_dir / "router_logits"
        if capture_residual:
            residual_dir.mkdir(parents=True, exist_ok=True)
        if capture_router:
            router_dir.mkdir(parents=True, exist_ok=True)

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
                residual, router_logits, router_indices, input_ids = _capture_one(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    messages=messages,
                    config=config,
                )
                elapsed = time.monotonic() - t0

                seq_len = int(input_ids.shape[1])
                if pool_on_capture:
                    residual, router_logits, router_indices = _apply_pooling(
                        residual, router_logits, router_indices, pool_on_capture
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
                    )

                prompt_hash = hashlib.sha256(
                    input_ids.cpu().numpy().tobytes()
                ).hexdigest()

                num_model_layers = len(self.model.model.layers)
                captured_layers_list = sorted(layers) if layers is not None else list(range(num_model_layers))

                meta_row = {
                    "log_id": int(log_id),
                    "seq_len": seq_len,
                    "prompt_hash": prompt_hash,
                    "capture_timestamp": datetime.now(UTC).isoformat(),
                    "file_size_bytes": file_size,
                    "elapsed_s": round(elapsed, 2),
                    "has_router": router_logits is not None,
                    "captured_layers": json.dumps(captured_layers_list),
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
                print(f"  {log_id}: {file_size / 1024 / 1024:.1f}MB, {elapsed:.1f}s")

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

    # List files
    residual_files = sorted(residual_dir.glob("*.safetensors")) if residual_dir.exists() else []
    router_files = sorted(router_dir.glob("*.safetensors")) if router_dir.exists() else []

    print(f"Volume contents ({base}):")
    print(f"  residual_stream/: {len(residual_files)} files")
    print(f"  router_logits/:   {len(router_files)} files")

    if not log_id:
        # Show first few filenames
        for f in residual_files[:5]:
            print(f"    {f.name}  ({f.stat().st_size / 1024 / 1024:.1f}MB)")
        for f in router_files[:5]:
            print(f"    {f.name}  ({f.stat().st_size / 1024 / 1024:.1f}MB)")
        return

    # Inspect a specific log_id
    print(f"\n--- log_id={log_id} ---")
    residual_path = residual_dir / f"{log_id}.safetensors"
    router_path = router_dir / f"{log_id}.safetensors"

    if residual_path.exists():
        with safe_open(str(residual_path), framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                print(f"  residual_stream/{key}: shape={tuple(t.shape)}, dtype={t.dtype}")
    else:
        print("  residual_stream: not found")

    if router_path.exists():
        with safe_open(str(router_path), framework="pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                print(f"  router_logits/{key}: shape={tuple(t.shape)}, dtype={t.dtype}")
    else:
        print("  router_logits: not found")


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=300,
)
def write_metadata_to_volume(metadata_rows: list[dict]) -> int:
    """Write metadata.parquet to the volume so analysis can find it."""
    import pyarrow as pa
    import pyarrow.parquet as pq_
    from pathlib import Path

    meta_path = Path("/data/activations/metadata.parquet")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(metadata_rows)
    pq_.write_table(table, meta_path, compression="snappy")
    volume.commit()
    print(f"Wrote metadata to volume: {meta_path} ({len(metadata_rows)} rows)")
    return len(metadata_rows)


@app.local_entrypoint()
def main(
    parquet_path: str = "data/interp_exports/interp_examples_v0_high_quality.parquet",
    limit: int = 0,
    layers: str = "",
    capture_router: bool = True,
    capture_residual: bool = True,
    batch_size: int = 10,
    model_id: str = "Qwen/Qwen3-30B-A3B",
    pool: str = "",
):
    import json
    from pathlib import Path

    import pyarrow.parquet as pq

    parquet = Path(parquet_path)
    if not parquet.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet}")

    table = pq.read_table(parquet)
    rows = table.to_pylist()
    print(f"Loaded {len(rows)} examples from {parquet}")

    if limit > 0:
        rows = rows[:limit]
        print(f"  Limited to {len(rows)} examples")

    parsed_layers: list[int] | None = None
    if layers:
        parsed_layers = [int(x.strip()) for x in layers.split(",")]

    # Partition into batches
    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    print(f"  {len(batches)} batches of up to {batch_size}")

    worker = CaptureWorker(model_id=model_id)

    all_metadata: list[dict] = []
    for batch_meta in worker.capture_batch.map(
        batches,
        kwargs=dict(
            layers=parsed_layers,
            capture_router=capture_router,
            capture_residual=capture_residual,
            pool_on_capture=pool if pool else None,
        ),
    ):
        all_metadata.extend(batch_meta)

    if all_metadata:
        import pyarrow as pa

        meta_table = pa.Table.from_pylist(all_metadata)
        # Write metadata locally
        meta_path = Path("data/activations/metadata.parquet")
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(meta_table, meta_path, compression="snappy")
        print(f"\nWrote metadata locally: {meta_path} ({len(all_metadata)} rows)")

        # Write metadata to volume so analysis can find it
        write_metadata_to_volume.remote(all_metadata)

    print(f"\nDone: {len(all_metadata)} examples captured")
