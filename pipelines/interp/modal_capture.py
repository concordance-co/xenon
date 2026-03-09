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
    timeout=7200,
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


@app.function(volumes={"/data": volume}, image=image, timeout=300)
def get_completed_log_ids(
    capture_router: bool = True,
    capture_residual: bool = True,
    pool_on_capture: str | None = None,
) -> set[int]:
    """Return log_ids that already have matching captures on the volume.

    Checks compact files first (post-compaction), then falls back to
    per-log safetensors files + metadata.parquet.
    """
    from pathlib import Path

    base = Path("/data/activations")
    compact_dir = base / "compact"
    pooling = pool_on_capture or "last_token"

    # --- Check compact files first ---
    # Compact files are named: {source}_{pooling}_layer{N}.safetensors
    # and contain a "log_ids" array with all captured log_ids.
    compact_ids = _check_compact(compact_dir, pooling, capture_router, capture_residual)
    if compact_ids is not None:
        return compact_ids

    # --- Fall back to per-log files + metadata ---
    import pyarrow.parquet as pq_

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


def _check_compact(
    compact_dir, pooling: str, capture_router: bool, capture_residual: bool
) -> set[int] | None:
    """Read log_ids from compact files. Returns None if no compact files match."""
    from pathlib import Path

    if not compact_dir.exists():
        return None

    # Find any compact file matching this source+pooling to read log_ids
    sources_needed = []
    if capture_router:
        sources_needed.append("router")
    if capture_residual:
        sources_needed.append("residual")
    if not sources_needed:
        sources_needed.append("router")

    result_ids: set[int] | None = None

    for source in sources_needed:
        # Find any layer file for this source+pooling
        pattern = f"{source}_{pooling}_layer*.safetensors"
        matches = sorted(compact_dir.glob(pattern))
        if not matches:
            return None  # No compact files for this source — can't use compact path

        # Read log_ids from the first matching file
        from safetensors import safe_open

        with safe_open(str(matches[0]), framework="numpy") as f:
            if "log_ids" not in f.keys():
                return None
            ids = set(int(x) for x in f.get_tensor("log_ids"))

        if result_ids is None:
            result_ids = ids
        else:
            # If we need both router and residual, only count log_ids present in both
            result_ids = result_ids & ids

    return result_ids


@app.function(
    volumes={"/data": volume},
    image=image,
    timeout=300,
)
def write_metadata_to_volume(metadata_rows: list[dict]) -> int:
    """Merge new metadata rows into metadata.parquet on the volume."""
    import pyarrow as pa
    import pyarrow.parquet as pq_
    from pathlib import Path

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
        print("\nAll examples already captured. Nothing to do.")
        return

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
            pool_on_capture=pool_val,
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

        # Merge metadata into volume (appends to existing)
        write_metadata_to_volume.remote(all_metadata)

    print(f"\nDone: {len(all_metadata)} examples captured, {skipped} skipped (already done)")
    if all_metadata and skipped > 0:
        print("Note: New captures added alongside compacted data. Re-run compact to consolidate.")
