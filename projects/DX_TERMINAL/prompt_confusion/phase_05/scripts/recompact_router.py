"""Rebuild router compact files covering all 288 rows.

The existing router_prompt_eos_layer*.safetensors compact files were
produced during Phase 04's conflict-readout analysis and only contain
the 123-row conflict-only subset. The per-example router_logits/ data
exists for all 288 rows, so recompacting is cheap.

Overwrites: /data/activations/workflows/conflict_probe_v3/16474bceae4e/compact/router_prompt_eos_layer{N}.safetensors
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "xenon-prompt-confusion-phase5-recompact-router"
DEFAULT_CAPTURE_RUN_ID = "16474bceae4e"
DEFAULT_ACTIVATIONS_SUBDIR = f"workflows/conflict_probe_v3/{DEFAULT_CAPTURE_RUN_ID}"
CAPTURED_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44]

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("xenon-data", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("numpy", "pyarrow", "safetensors")
)


@app.function(volumes={"/data": data_volume}, image=base_image, timeout=1800, cpu=4)
def recompact_router(activations_subdir: str = DEFAULT_ACTIVATIONS_SUBDIR) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq
    from safetensors import safe_open
    from safetensors.numpy import save_file

    activations_dir = Path("/data/activations") / activations_subdir
    compact_dir = activations_dir / "compact"
    compact_dir.mkdir(parents=True, exist_ok=True)
    router_dir = activations_dir / "router_logits"

    metadata_path = activations_dir / "metadata.parquet"
    meta_table = pq.read_table(str(metadata_path))
    meta_rows = meta_table.to_pylist()

    captured_layers_raw = meta_rows[0].get("captured_layers")
    if isinstance(captured_layers_raw, str):
        captured_layers = json.loads(captured_layers_raw)
    elif isinstance(captured_layers_raw, list):
        captured_layers = list(captured_layers_raw)
    else:
        captured_layers = CAPTURED_LAYERS
    layer_to_idx = {int(layer): idx for idx, layer in enumerate(captured_layers)}

    # Per-layer collectors
    layer_features: dict[int, list[Any]] = {layer: [] for layer in captured_layers}
    layer_router_indices: dict[int, list[Any]] = {layer: [] for layer in captured_layers}
    log_ids: list[int] = []
    skipped = 0
    ri_present_rows = 0

    for row in meta_rows:
        log_id = int(row["log_id"])
        artifact_id = row.get("artifact_id")
        if not artifact_id:
            skipped += 1
            continue
        path = router_dir / f"{artifact_id}.safetensors"
        if not path.exists():
            skipped += 1
            continue

        with safe_open(str(path), framework="numpy") as f:
            tensor = f.get_tensor("router_logits")
            has_ri = "router_indices" in f.keys()
            ri_tensor = f.get_tensor("router_indices") if has_ri else None

        log_ids.append(log_id)
        if has_ri:
            ri_present_rows += 1

        for layer in captured_layers:
            tidx = layer_to_idx[int(layer)]
            layer_data = tensor[tidx]
            if layer_data.ndim != 1:
                raise ValueError(
                    f"Expected prompt_eos-pooled (1D per layer) router logits, "
                    f"got shape {layer_data.shape} at layer {layer} for log_id {log_id}"
                )
            layer_features[int(layer)].append(layer_data)
            if ri_tensor is not None:
                ri_layer = ri_tensor[tidx]
                if ri_layer.ndim != 1:
                    raise ValueError(
                        f"Expected 1D router_indices per layer, got shape {ri_layer.shape}"
                    )
                layer_router_indices[int(layer)].append(ri_layer)

    log_id_arr = np.array(log_ids, dtype=np.int64)

    written: list[dict[str, Any]] = []
    for layer in captured_layers:
        out_path = compact_dir / f"router_prompt_eos_layer{layer}.safetensors"
        features = np.stack(layer_features[int(layer)]).astype(np.float32)
        tensors: dict[str, Any] = {
            "features": features,
            "log_ids": log_id_arr,
        }
        if layer_router_indices[int(layer)]:
            tensors["router_indices"] = np.stack(
                layer_router_indices[int(layer)]
            ).astype(np.int64)
        save_file(tensors, str(out_path))
        written.append({
            "layer": int(layer),
            "path": str(out_path),
            "features_shape": list(features.shape),
            "has_router_indices": "router_indices" in tensors,
            "size_bytes": out_path.stat().st_size,
        })

    return {
        "activations_dir": str(activations_dir),
        "n_rows_compacted": len(log_ids),
        "n_rows_skipped": skipped,
        "n_rows_with_router_indices": ri_present_rows,
        "captured_layers": [int(l) for l in captured_layers],
        "files_written": written,
    }


@app.local_entrypoint()
def main() -> None:
    result = recompact_router.remote()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
