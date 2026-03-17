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
neon_secret = modal.Secret.from_name("xenon-neon")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("torch", "transformers", "safetensors", "pyarrow", "huggingface_hub", "psycopg[binary]")
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
        batch_idx: int = 0,
        layers: list[int] | None = None,
        capture_router: bool = True,
        capture_residual: bool = True,
        pool_on_capture: str | None = None,
        router_dtype: str = "float16",
    ) -> list[dict]:
        import hashlib
        import json
        import time
        from datetime import UTC, datetime
        from pathlib import Path

        import numpy as np
        import torch
        from safetensors.numpy import save_file

        from pipelines.interp.capture import (
            CaptureConfig,
            _apply_pooling,
            _capture_one,
            _parse_messages,
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
            router_dtype=router_dtype,
        )

        num_model_layers = len(self.model.model.layers)
        captured_layers_list = sorted(layers) if layers is not None else list(range(num_model_layers))
        layer_to_idx = {layer: idx for idx, layer in enumerate(captured_layers_list)}
        pooling = pool_on_capture or "last_token"

        # Accumulators: per-layer lists of vectors
        residual_acc: dict[int, list[np.ndarray]] = {l: [] for l in captured_layers_list}
        router_acc: dict[int, list[np.ndarray]] = {l: [] for l in captured_layers_list}
        router_idx_acc: dict[int, list[np.ndarray]] = {l: [] for l in captured_layers_list}
        log_ids: list[int] = []
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

                # Pool to single vector per layer and accumulate
                log_ids.append(int(log_id))
                for layer in captured_layers_list:
                    tidx = layer_to_idx[layer]
                    if residual is not None:
                        layer_data = residual[tidx].cpu().float().numpy()
                        if layer_data.ndim > 1:
                            vec = layer_data[-1] if pooling == "last_token" else layer_data.mean(axis=0)
                        else:
                            vec = layer_data
                        residual_acc[layer].append(vec)

                    if router_logits is not None:
                        rl_data = router_logits[tidx].cpu().float().numpy()
                        if rl_data.ndim > 1:
                            vec = rl_data[-1] if pooling == "last_token" else rl_data.mean(axis=0)
                        else:
                            vec = rl_data
                        router_acc[layer].append(vec)

                    if router_indices is not None:
                        ri_data = router_indices[tidx].cpu().numpy()
                        if ri_data.ndim > 1:
                            ri_vec = ri_data[-1] if pooling == "last_token" else ri_data.reshape(-1)
                        else:
                            ri_vec = ri_data
                        router_idx_acc[layer].append(ri_vec)

                file_size = 0
                if residual is not None:
                    file_size += residual.nelement() * residual.element_size()
                if router_logits is not None:
                    file_size += router_logits.nelement() * router_logits.element_size()

                prompt_hash = hashlib.sha256(
                    input_ids.cpu().numpy().tobytes()
                ).hexdigest()

                meta_row = {
                    "log_id": int(log_id),
                    "seq_len": seq_len,
                    "prompt_hash": prompt_hash,
                    "capture_timestamp": datetime.now(UTC).isoformat(),
                    "file_size_bytes": file_size,
                    "elapsed_s": round(elapsed, 2),
                    "has_router": router_logits is not None,
                    "captured_layers": json.dumps(captured_layers_list),
                    "pooling": pooling,
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

        # Write consolidated batch files in compact format
        if log_ids:
            compact_dir = output_dir / "compact"
            compact_dir.mkdir(parents=True, exist_ok=True)
            log_id_arr = np.array(log_ids, dtype=np.int64)

            for layer in captured_layers_list:
                if residual_acc[layer]:
                    out_path = compact_dir / f"residual_{pooling}_layer{layer}_batch{batch_idx}.safetensors"
                    tensors: dict[str, np.ndarray] = {
                        "features": np.stack(residual_acc[layer]).astype(np.float32),
                        "log_ids": log_id_arr,
                    }
                    save_file(tensors, str(out_path))

                if router_acc[layer]:
                    out_path = compact_dir / f"router_{pooling}_layer{layer}_batch{batch_idx}.safetensors"
                    tensors = {
                        "features": np.stack(router_acc[layer]).astype(np.float32),
                        "log_ids": log_id_arr,
                    }
                    if router_idx_acc[layer]:
                        tensors["router_indices"] = np.stack(router_idx_acc[layer]).astype(np.int64)
                    save_file(tensors, str(out_path))

            print(f"  Wrote batch {batch_idx} compact files ({len(log_ids)} examples, {len(captured_layers_list)} layers)")

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


@app.function(volumes={"/data": volume}, image=image, timeout=600)
def merge_compact_batches(num_batches: int) -> int:
    """Merge per-batch compact files into final per-layer compact files."""
    import re
    from pathlib import Path

    import numpy as np
    from safetensors import safe_open
    from safetensors.numpy import save_file

    compact_dir = Path("/data/activations/compact")
    if not compact_dir.exists():
        print("No compact dir found")
        return 0

    # Discover batch files and group by (source, pooling, layer)
    batch_pattern = re.compile(r"^(.+)_batch(\d+)\.safetensors$")
    groups: dict[str, list[Path]] = {}

    for f in sorted(compact_dir.iterdir()):
        m = batch_pattern.match(f.name)
        if m:
            key = m.group(1)  # e.g. "residual_last_token_layer16"
            groups.setdefault(key, []).append(f)

    merged = 0
    for key, batch_files in sorted(groups.items()):
        all_features = []
        all_log_ids = []
        all_ri = []

        # Load existing final file first so we don't lose previous data
        out_path = compact_dir / f"{key}.safetensors"
        if out_path.exists():
            with safe_open(str(out_path), framework="numpy") as f:
                all_features.append(f.get_tensor("features"))
                all_log_ids.append(f.get_tensor("log_ids"))
                if "router_indices" in f.keys():
                    all_ri.append(f.get_tensor("router_indices"))

        for bf in sorted(batch_files):
            with safe_open(str(bf), framework="numpy") as f:
                all_features.append(f.get_tensor("features"))
                all_log_ids.append(f.get_tensor("log_ids"))
                if "router_indices" in f.keys():
                    all_ri.append(f.get_tensor("router_indices"))

        features = np.concatenate(all_features, axis=0)
        log_ids = np.concatenate(all_log_ids, axis=0)
        ri = np.concatenate(all_ri, axis=0) if all_ri else None

        # Deduplicate by log_id — keep last occurrence (newest data wins)
        _, unique_idx = np.unique(log_ids, return_index=True)
        unique_idx = np.sort(unique_idx)  # preserve order
        features = features[unique_idx]
        log_ids = log_ids[unique_idx]

        tensors: dict[str, np.ndarray] = {
            "features": features,
            "log_ids": log_ids,
        }
        if ri is not None:
            tensors["router_indices"] = ri[unique_idx]

        # Write to tmp then rename for atomicity
        tmp_path = compact_dir / f"{key}.tmp.safetensors"
        save_file(tensors, str(tmp_path))
        tmp_path.rename(out_path)

        size_mb = out_path.stat().st_size / 1024 / 1024
        n = log_ids.shape[0]
        print(f"  {key}: {n} examples, {size_mb:.1f} MB")

        # Remove batch files
        for bf in batch_files:
            bf.unlink()

        merged += 1

    volume.commit()
    print(f"Merged {merged} layer files from {num_batches} batches")
    return merged


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


@app.function(
    image=image,
    secrets=[neon_secret],
    volumes={"/data": volume},
    timeout=7200,
)
def run_capture(
    limit: int = 0,
    layers_str: str = "",
    capture_router: bool = True,
    capture_residual: bool = True,
    batch_size: int = 10,
    model_id: str = "Qwen/Qwen3-30B-A3B",
    pool: str = "",
    router_dtype: str = "float16",
) -> str:
    """Orchestrator: load examples from Neon, fan out to GPU workers, merge results."""
    from pipelines.interp.capture import _load_examples_from_neon

    rows = _load_examples_from_neon(limit=limit if limit > 0 else None)
    if not rows:
        return "No examples to capture"

    parsed_layers: list[int] | None = None
    if layers_str:
        parsed_layers = [int(x.strip()) for x in layers_str.split(",")]

    batches = [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
    batch_indices = list(range(len(batches)))
    print(f"  {len(batches)} batches of up to {batch_size}")

    worker = CaptureWorker(model_id=model_id)

    all_metadata: list[dict] = []
    for batch_meta in worker.capture_batch.map(
        batches,
        batch_indices,
        kwargs=dict(
            layers=parsed_layers,
            capture_router=capture_router,
            capture_residual=capture_residual,
            pool_on_capture=pool if pool else None,
            router_dtype=router_dtype,
        ),
    ):
        all_metadata.extend(batch_meta)

    if all_metadata:
        write_metadata_to_volume.remote(all_metadata)
        merge_compact_batches.remote(len(batches))

    summary = f"Done: {len(all_metadata)} examples captured in {len(batches)} batches"
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
    router_dtype: str = "float16",
):
    result = run_capture.remote(
        limit=limit,
        layers_str=layers,
        capture_router=capture_router,
        capture_residual=capture_residual,
        batch_size=batch_size,
        model_id=model_id,
        pool=pool,
        router_dtype=router_dtype,
    )
    print(result)
