"""Modal probes for Assistant Axis Llama 70B startup behavior.

The probe intentionally tests only vLLM construction. It keeps
``enforce_eager=False`` fixed and varies the patched worker / compile path and
small vLLM startup toggles.
"""

from __future__ import annotations

import time
from typing import Any

import modal


MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_PATH = f"/models/{MODEL_ID}"
MODEL_VOLUME_NAME = "yora-models"
MODEL_VOLUME_PATH = "/models"


image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "vllm",
        "fastapi[standard]",
        "torch",
        "transformers>=4.51.0",
        "safetensors",
        "numpy",
        "huggingface_hub",
        "pydantic>=2.7.0",
        "pyarrow>=23.0.1",
        "psycopg[binary]>=3.2.0",
        "openai>=2.35.0",
    )
    .env(
        {
            "PYTHONPATH": "/root/pipelines_v2_workspace",
            "HF_HOME": f"{MODEL_VOLUME_PATH}/hf_home",
            "TRANSFORMERS_CACHE": f"{MODEL_VOLUME_PATH}/hf_home/transformers",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "VLLM_USE_DEEP_GEMM": "0",
            "XENON_ACTIVATION_PATCH_MAX_TOKENS": "1",
        }
    )
    .add_local_dir("pipelines_v2", remote_path="/root/pipelines_v2_workspace/pipelines_v2")
    .add_local_dir("papers/voice/assistant_axis", remote_path="/root/pipelines_v2_workspace/papers/voice/assistant_axis")
)

app = modal.App("assistant-axis-startup-probe")


MODAL_KWARGS = {
    "image": image,
    "gpu": "H200:1",
    "cpu": 8,
    "memory": 96 * 1024,
    "timeout": 60 * 45,
    "startup_timeout": 60 * 30,
    "volumes": {MODEL_VOLUME_PATH: modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)},
}


def _probe_vllm_startup_impl(
    *,
    mode: str,
    enable_chunked_prefill: bool = False,
    enable_prefix_caching: bool = False,
    async_scheduling: bool = True,
    gpu_memory_utilization: float = 0.88,
    max_num_batched_tokens: int | None = None,
    preload_tokenizer: bool = False,
) -> dict[str, Any]:
    from vllm import LLM

    mode = str(mode)
    kwargs: dict[str, Any] = {
        "model": MODEL_PATH,
        "served_model_name": MODEL_ID,
        "enforce_eager": False,
        "max_model_len": 1024,
        "max_num_seqs": 1,
        "enable_chunked_prefill": bool(enable_chunked_prefill),
        "enable_prefix_caching": bool(enable_prefix_caching),
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "disable_log_stats": True,
    }
    if max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
    if bool(async_scheduling):
        kwargs["async_scheduling"] = True
    if mode == "patched":
        kwargs["worker_cls"] = "pipelines_v2.engine.vllm.activation_patch_request_worker.ActivationPatchGPUWorker"
        kwargs["compilation_config"] = {"custom_ops": ["none", "+activation_patch_hidden_states"]}
        kwargs["additional_config"] = {
            "xenon_activation_patch_worker_cls": "pipelines_v2.engine.vllm.activation_patch_request_worker.ActivationPatchGPUWorker",
            "xenon_activation_patch_compiled_operator": "subspace",
        }
    elif mode != "stock":
        raise ValueError("mode must be one of {'stock', 'patched'}")

    print(
        "[startup-probe] constructing LLM",
        {
            "mode": mode,
            "enable_chunked_prefill": enable_chunked_prefill,
            "enable_prefix_caching": enable_prefix_caching,
            "async_scheduling": async_scheduling,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_batched_tokens": max_num_batched_tokens,
            "preload_tokenizer": preload_tokenizer,
        },
        flush=True,
    )
    if preload_tokenizer:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        print("[startup-probe] preloaded tokenizer", flush=True)
    started = time.time()
    llm = LLM(**kwargs)
    elapsed = time.time() - started
    print("[startup-probe] constructed LLM", {"elapsed_s": round(elapsed, 3)}, flush=True)

    output = llm.generate("Hello", sampling_params=None)
    text = output[0].outputs[0].text if output and output[0].outputs else ""
    return {
        "mode": mode,
        "elapsed_s": elapsed,
        "generated_prefix": text[:80],
    }


@app.function(**MODAL_KWARGS)
def probe_vllm_startup(
    *,
    mode: str,
    enable_chunked_prefill: bool = False,
    enable_prefix_caching: bool = False,
    async_scheduling: bool = True,
    gpu_memory_utilization: float = 0.88,
    max_num_batched_tokens: int | None = None,
    preload_tokenizer: bool = False,
) -> dict[str, Any]:
    return _probe_vllm_startup_impl(
        mode=mode,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_prefix_caching=enable_prefix_caching,
        async_scheduling=async_scheduling,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_batched_tokens=max_num_batched_tokens,
        preload_tokenizer=preload_tokenizer,
    )


@app.function(**MODAL_KWARGS)
@modal.fastapi_endpoint(method="POST", label="fastapi-probe")
def fastapi_probe(item: dict[str, Any]) -> dict[str, Any]:
    return _probe_vllm_startup_impl(
        mode=str(item.get("mode") or "patched"),
        enable_chunked_prefill=bool(item.get("enable_chunked_prefill", False)),
        enable_prefix_caching=bool(item.get("enable_prefix_caching", False)),
        async_scheduling=bool(item.get("async_scheduling", True)),
        gpu_memory_utilization=float(item.get("gpu_memory_utilization", 0.99)),
        max_num_batched_tokens=item.get("max_num_batched_tokens"),
        preload_tokenizer=bool(item.get("preload_tokenizer", False)),
    )


@app.cls(**MODAL_KWARGS)
class StartupProbeClass:
    @modal.method()
    def run(
        self,
        *,
        mode: str,
        enable_chunked_prefill: bool = False,
        enable_prefix_caching: bool = False,
        async_scheduling: bool = True,
        gpu_memory_utilization: float = 0.88,
        max_num_batched_tokens: int | None = None,
        preload_tokenizer: bool = False,
    ) -> dict[str, Any]:
        return _probe_vllm_startup_impl(
            mode=mode,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=enable_prefix_caching,
            async_scheduling=async_scheduling,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_num_batched_tokens,
            preload_tokenizer=preload_tokenizer,
        )


@app.cls(**{**MODAL_KWARGS, "serialized": True})
class SerializedStartupProbeClass:
    @modal.method()
    def run(
        self,
        *,
        mode: str,
        enable_chunked_prefill: bool = False,
        enable_prefix_caching: bool = False,
        async_scheduling: bool = True,
        gpu_memory_utilization: float = 0.88,
        max_num_batched_tokens: int | None = None,
        preload_tokenizer: bool = False,
    ) -> dict[str, Any]:
        return _probe_vllm_startup_impl(
            mode=mode,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=enable_prefix_caching,
            async_scheduling=async_scheduling,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_num_batched_tokens,
            preload_tokenizer=preload_tokenizer,
        )


@app.local_entrypoint()
def main(
    mode: str = "stock",
    runner: str = "function",
    enable_chunked_prefill: bool = False,
    enable_prefix_caching: bool = False,
    async_scheduling: bool = True,
    gpu_memory_utilization: float = 0.88,
    max_num_batched_tokens: int = 0,
    preload_tokenizer: bool = False,
) -> None:
    kwargs = {
        "mode": mode,
        "enable_chunked_prefill": enable_chunked_prefill,
        "enable_prefix_caching": enable_prefix_caching,
        "async_scheduling": async_scheduling,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_num_batched_tokens": max_num_batched_tokens or None,
        "preload_tokenizer": preload_tokenizer,
    }
    if runner == "function":
        result = probe_vllm_startup.remote(**kwargs)
    elif runner == "class":
        result = StartupProbeClass().run.remote(**kwargs)
    elif runner == "serialized-class":
        result = SerializedStartupProbeClass().run.remote(**kwargs)
    else:
        raise ValueError("runner must be one of {'function', 'class', 'serialized-class'}")
    print(result)
