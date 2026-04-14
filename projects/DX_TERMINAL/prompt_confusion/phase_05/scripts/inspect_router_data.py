"""One-off inspection: figure out what router activation data actually exists
on the Modal volume for capture run 16474bceae4e.

Checks:
- files present in compact/ dir (with row counts per safetensors file)
- files present in router_logits/ per-example dir (if any)
- captured_layers from metadata.parquet
- a sample log_id check between residual and router compact files
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

APP_NAME = "xenon-prompt-confusion-phase5-inspect-router"
DEFAULT_CAPTURE_RUN_ID = "16474bceae4e"
DEFAULT_ACTIVATIONS_SUBDIR = f"workflows/conflict_probe_v3/{DEFAULT_CAPTURE_RUN_ID}"

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name("xenon-data", create_if_missing=True)

base_image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install("numpy", "pyarrow", "safetensors")
)


@app.function(volumes={"/data": data_volume}, image=base_image, timeout=600, cpu=2)
def inspect(activations_subdir: str = DEFAULT_ACTIVATIONS_SUBDIR) -> dict[str, Any]:
    from safetensors import safe_open
    import pyarrow.parquet as pq

    activations_dir = Path("/data/activations") / activations_subdir
    report: dict[str, Any] = {"activations_dir": str(activations_dir)}

    if not activations_dir.exists():
        report["error"] = f"activations_dir does not exist"
        return report

    # Top-level listing
    report["top_level"] = sorted([p.name for p in activations_dir.iterdir()])

    # compact dir
    compact_dir = activations_dir / "compact"
    compact_info: list[dict[str, Any]] = []
    if compact_dir.exists():
        for path in sorted(compact_dir.iterdir()):
            entry: dict[str, Any] = {"name": path.name, "size_bytes": path.stat().st_size}
            if path.suffix == ".safetensors":
                try:
                    with safe_open(str(path), framework="numpy") as f:
                        keys = list(f.keys())
                        entry["keys"] = keys
                        if "log_ids" in keys:
                            log_ids = f.get_tensor("log_ids")
                            entry["n_log_ids"] = int(len(log_ids))
                            entry["log_id_min"] = int(log_ids.min())
                            entry["log_id_max"] = int(log_ids.max())
                        if "features" in keys:
                            feat = f.get_tensor("features")
                            entry["features_shape"] = list(feat.shape)
                except Exception as e:
                    entry["error"] = str(e)
            compact_info.append(entry)
    report["compact_dir"] = compact_info

    # per-example router_logits dir
    router_dir = activations_dir / "router_logits"
    if router_dir.exists():
        router_files = sorted(router_dir.iterdir())
        report["router_logits_per_example"] = {
            "n_files": len(router_files),
            "sample_names": [p.name for p in router_files[:5]],
        }
    else:
        report["router_logits_per_example"] = None

    # per-example residual_stream dir
    residual_dir = activations_dir / "residual_stream"
    if residual_dir.exists():
        residual_files = sorted(residual_dir.iterdir())
        report["residual_stream_per_example"] = {
            "n_files": len(residual_files),
            "sample_names": [p.name for p in residual_files[:5]],
        }
    else:
        report["residual_stream_per_example"] = None

    # metadata.parquet
    metadata_path = activations_dir / "metadata.parquet"
    if metadata_path.exists():
        table = pq.read_table(str(metadata_path))
        rows = table.to_pylist()
        report["metadata_parquet"] = {
            "n_rows": len(rows),
            "columns": table.column_names,
            "sample_row": rows[0] if rows else None,
        }
    else:
        report["metadata_parquet"] = None

    # Compare log_ids between a residual and a router compact file
    res_path = compact_dir / "residual_prompt_eos_layer24.safetensors"
    rtr_path = compact_dir / "router_prompt_eos_layer24.safetensors"
    if res_path.exists() and rtr_path.exists():
        with safe_open(str(res_path), framework="numpy") as f:
            res_ids = set(int(x) for x in f.get_tensor("log_ids"))
        with safe_open(str(rtr_path), framework="numpy") as f:
            rtr_ids = set(int(x) for x in f.get_tensor("log_ids"))
        report["residual_vs_router_layer24"] = {
            "n_residual": len(res_ids),
            "n_router": len(rtr_ids),
            "router_subset_of_residual": rtr_ids.issubset(res_ids),
            "n_in_residual_only": len(res_ids - rtr_ids),
            "n_in_router_only": len(rtr_ids - res_ids),
        }

    return report


@app.local_entrypoint()
def main() -> None:
    result = inspect.remote()
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
