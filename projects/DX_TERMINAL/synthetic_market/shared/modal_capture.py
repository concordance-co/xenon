"""Project wrapper for synthetic-market activation capture.

The capture runtime lives in ``pipelines.interp.modal_vllm_orchestrator``.
This module only resolves the synthetic-market relation and output naming,
then delegates to the generic interp capture runner.
"""

from __future__ import annotations

import re

import modal

app = modal.App("xenon-synthetic-vllm-capture")


def _normalize_phase_tag(phase_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", phase_name.strip()).strip("_").lower()
    if not normalized:
        raise ValueError("phase_name must contain at least one alphanumeric character")
    return normalized


def resolve_synthetic_capture_relation(
    phase_name: str,
    source_relation: str = "",
) -> str:
    text = source_relation.strip()
    if text:
        return text
    return f"synthetic_market_{_normalize_phase_tag(phase_name)}_capture_v0"


def resolve_synthetic_capture_output_subdir(
    phase_name: str,
    output_subdir: str = "",
) -> str:
    text = output_subdir.strip()
    if text:
        return text
    return f"projects/DX_TERMINAL/synthetic_market/captures/{phase_name}"


@app.local_entrypoint()
def main(
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
    source_relation: str = "",
    output_subdir: str = "",
    order_mode: str = "selection_rank_asc",
) -> None:
    from pipelines.interp.modal_vllm_orchestrator import run_vllm_capture

    resolved_source_relation = resolve_synthetic_capture_relation(
        phase_name=phase_name,
        source_relation=source_relation,
    )
    resolved_output_subdir = resolve_synthetic_capture_output_subdir(
        phase_name=phase_name,
        output_subdir=output_subdir,
    )

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
        order_mode=order_mode,
        source_relation=resolved_source_relation,
        output_subdir=resolved_output_subdir,
    )
    print(result)
