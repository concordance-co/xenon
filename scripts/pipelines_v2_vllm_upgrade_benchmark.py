"""Paired Modal benchmark for Xenon's pinned vLLM runtime.

Run this workflow from both the v0.19 ground-truth checkout and the upgrade
branch with the same label-independent model/GPU settings. Generated payloads
stay on the Modal volume; the final transform emits only aggregate timing,
token-count, finish-reason, and deterministic output-digest metadata.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pipelines_v2.api import (
    Dataset,
    Example,
    GenerationRunSpec,
    GenerationSpec,
    ModalResources,
    ModalRunnerSpec,
    ModalVolumeMount,
    ModalVolumeStore,
    StepRef,
    TransformBuilder,
    TransformSpec,
    VLLMEngine,
    WorkflowSpec,
    WorkflowStep,
)


MODEL_VOLUME_NAME = "xenon-models"
MODEL_VOLUME_PATH = "/models"
DEFAULT_MODEL_ID = "/models/Qwen/Qwen3-30B-A3B"
ARTIFACT_VOLUME_NAME = "xenon-data"


def _benchmark_label() -> str:
    raw = str(os.getenv("XENON_VLLM_BENCHMARK_LABEL", "unlabeled") or "unlabeled")
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw).strip("-") or "unlabeled"


def build_dataset() -> Dataset:
    examples = []
    for index in range(16):
        examples.append(
            Example(
                key=f"throughput_{index:02d}",
                prompt=[
                    {
                        "role": "user",
                        "content": (
                            "Generate a numbered list of concise, standalone observations "
                            f"about the integer {1000 + index}. Continue until the response "
                            "reaches the generation limit. Do not include a preamble or conclusion."
                        ),
                    }
                ],
                labels={"benchmark": "throughput"},
            )
        )
    return Dataset.from_examples(examples, name="vllm_upgrade_throughput_fixture")


def summarize_generation(*, generation: Any) -> dict[str, Any]:
    payload = generation.result() if hasattr(generation, "result") else dict(generation)
    rows = [dict(row) for row in payload.get("rows", ())]
    token_counts = [len(row.get("generated_token_ids") or ()) for row in rows]
    finish_reasons = Counter(str(row.get("finish_reason") or "") for row in rows)
    output_digests = {
        str(row.get("example_key") or ""): hashlib.sha256(
            str(row.get("generated_text") or "").encode("utf-8")
        ).hexdigest()
        for row in rows
    }
    metadata = dict(payload.get("metadata") or {})
    performance = dict(metadata.get("performance") or {})
    summary = {
        "request_count": len(rows),
        "nonempty_output_count": sum(bool(str(row.get("generated_text") or "").strip()) for row in rows),
        "generated_token_count": sum(token_counts),
        "generated_tokens_min": min(token_counts, default=0),
        "generated_tokens_max": max(token_counts, default=0),
        "generated_tokens_mean": (
            float(sum(token_counts)) / float(len(token_counts)) if token_counts else 0.0
        ),
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "performance": performance,
        "output_digests": output_digests,
    }
    return {
        "payload": {
            "kind": "vllm_upgrade_benchmark_result",
            "summary": summary,
        },
        "metadata": {
            "benchmark_label": _benchmark_label(),
            "model_id": str(os.getenv("XENON_VLLM_BENCHMARK_MODEL_ID", DEFAULT_MODEL_ID)),
        },
        "example_keys": sorted(output_digests),
    }


def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec:
    dataset = dataset or build_dataset()
    label = _benchmark_label()
    model_id = str(os.getenv("XENON_VLLM_BENCHMARK_MODEL_ID", DEFAULT_MODEL_ID))
    engine = VLLMEngine(
        model_id=model_id,
        max_model_len=1024,
        enforce_eager=False,
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        async_scheduling=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    summarizer = TransformBuilder.from_function(
        summarize_generation,
        local_python_sources=("scripts",),
    )
    return WorkflowSpec(
        name=f"pipelines_v2_vllm_upgrade_benchmark_{label}",
        steps=(
            WorkflowStep(
                name="generation_throughput",
                runner="capture_gpu",
                description="Run a fixed 16-request, 128-token compiled generation workload.",
                spec=GenerationRunSpec(
                    engine=engine,
                    dataset=dataset,
                    generation=GenerationSpec(
                        enabled=True,
                        max_tokens=128,
                        temperature=0.0,
                        capture_reasoning=False,
                    ),
                ),
            ),
            WorkflowStep(
                name="summarize_benchmark",
                runner="analysis_cpu",
                description="Persist only aggregate token/performance data and output digests.",
                spec=TransformSpec(
                    builder=summarizer,
                    inputs={"generation": StepRef("generation_throughput")},
                ),
            ),
        ),
    )


def build_runner_specs() -> dict[str, object]:
    label = _benchmark_label()
    artifact_root = Path("/data/artifacts/pipelines_v2_vllm_upgrade_benchmark") / label
    artifact_store = ModalVolumeStore(
        name=ARTIFACT_VOLUME_NAME,
        root=str(artifact_root),
    )
    return {
        "capture_gpu": ModalRunnerSpec(
            resources=ModalResources(
                gpu=str(os.getenv("XENON_VLLM_BENCHMARK_GPU", "A100-80GB")),
                timeout_seconds=3600,
                max_containers=1,
                volumes=(
                    ModalVolumeMount(
                        name=MODEL_VOLUME_NAME,
                        mount_path=MODEL_VOLUME_PATH,
                    ),
                ),
            ),
            artifacts=artifact_store,
        ),
        "analysis_cpu": ModalRunnerSpec(
            resources=ModalResources(
                cpu=4,
                memory_mb=8 * 1024,
                timeout_seconds=900,
                max_containers=1,
                volumes=(
                    ModalVolumeMount(
                        name=ARTIFACT_VOLUME_NAME,
                        mount_path="/data",
                    ),
                ),
            ),
            artifacts=artifact_store,
        ),
    }


__all__ = [
    "build_dataset",
    "build_runner_specs",
    "build_workflow",
    "summarize_generation",
]
