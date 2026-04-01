from __future__ import annotations

import json
import os
import signal
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


DEFAULT_BENCHMARK_MESSAGES: list[dict[str, str]] = [
    {
        "role": "system",
        "content": "You are a concise quantitative trading assistant. Answer directly and avoid unnecessary filler.",
    },
    {
        "role": "user",
        "content": (
            "A trader has 100 ETH available and wants to deploy a single position. "
            "Review this market snapshot and recommend one asset plus a position size.\n\n"
            "Assets:\n"
            "- ALP: 1h return +6.2%, 24h return +18.1%, volume 42.1m, realized vol 0.88, drawdown 0.19\n"
            "- BRV: 1h return +1.3%, 24h return +4.6%, volume 15.7m, realized vol 0.31, drawdown 0.07\n"
            "- CRN: 1h return -2.4%, 24h return +9.8%, volume 51.4m, realized vol 1.12, drawdown 0.28\n"
            "- DLT: 1h return +3.8%, 24h return +12.4%, volume 27.9m, realized vol 0.54, drawdown 0.11\n"
            "- EOS: 1h return +0.5%, 24h return +2.1%, volume 8.2m, realized vol 0.22, drawdown 0.04\n\n"
            "Respond with the ticker, a one-paragraph rationale, and an allocation percentage."
        ),
    },
]


def _benchmark_section_char_span(rendered_text: str) -> tuple[int, int]:
    start_char = rendered_text.find("Assets:")
    if start_char < 0:
        raise ValueError("Could not find benchmark 'Assets:' section in rendered prompt")
    end_candidates = [
        rendered_text.find(marker, start_char + len("Assets:"))
        for marker in ("Respond with", "Return ", "Output ")
    ]
    end_char = min(candidate for candidate in end_candidates if candidate > start_char) if any(
        candidate > start_char for candidate in end_candidates
    ) else len(rendered_text)
    if end_char <= start_char:
        raise ValueError("Benchmark section end marker falls before the section start")
    return int(start_char), int(end_char)


def _comparison_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def build_vllm_serve_command(
    *,
    model_id: str,
    api_key: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    served_model_name: str = "benchmark-model",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 32,
    max_model_len: int | None = None,
    enable_chunked_prefill: bool = True,
) -> list[str]:
    command = [
        "vllm",
        "serve",
        model_id,
        "--host",
        str(host),
        "--port",
        str(port),
        "--api-key",
        api_key,
        "--served-model-name",
        served_model_name,
        "--generation-config",
        "vllm",
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        str(max_num_seqs),
    ]
    if max_model_len is not None:
        command.extend(["--max-model-len", str(max_model_len)])
    if enable_chunked_prefill:
        command.append("--enable-chunked-prefill")
    return command


def build_vllm_bench_serve_command(
    *,
    base_url: str,
    auth_token: str,
    model_id: str,
    served_model_name: str,
    input_len: int,
    output_len: int,
    num_prompts: int,
    max_concurrency: int,
    result_dir: Path,
    result_filename: str = "bench_serve_random.json",
) -> list[str]:
    return [
        "vllm",
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        base_url,
        "--endpoint",
        "/v1/completions",
        "--header",
        f"Authorization=Bearer {auth_token}",
        "--model",
        model_id,
        "--served-model-name",
        served_model_name,
        "--dataset-name",
        "random",
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(max_concurrency),
        "--disable-tqdm",
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
    ]


def wait_for_vllm_server(*, base_url: str, api_key: str, timeout_s: float = 600.0) -> None:
    deadline = time.time() + float(timeout_s)
    headers = {"Authorization": f"Bearer {api_key}"}
    request = urllib.request.Request(f"{base_url}/v1/models", headers=headers)
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(request, timeout=5.0) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - exercised in Modal.
            last_error = exc
            time.sleep(2.0)
            continue
    raise RuntimeError(f"Timed out waiting for vLLM server at {base_url}") from last_error


def _json_post(
    *,
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def benchmark_chat_request(
    *,
    base_url: str,
    api_key: str,
    served_model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    warmup_requests: int = 1,
    measured_requests: int = 3,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    payload = {
        "model": served_model_name,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    for index in range(max(0, int(warmup_requests)) + max(1, int(measured_requests))):
        start = time.perf_counter()
        response = _json_post(
            url=f"{base_url}/v1/chat/completions",
            payload=payload,
            api_key=api_key,
        )
        latency_s = time.perf_counter() - start
        if index < int(warmup_requests):
            continue
        usage = dict(response.get("usage", {}))
        completion_tokens = int(usage.get("completion_tokens", 0))
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        runs.append(
            {
                "latency_s": float(latency_s),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "output_toks_per_s": (
                    float(completion_tokens / latency_s)
                    if completion_tokens > 0 and latency_s > 0
                    else 0.0
                ),
                "response_id": response.get("id"),
                "finish_reason": (
                    response.get("choices", [{}])[0].get("finish_reason")
                    if isinstance(response.get("choices"), list)
                    else None
                ),
            }
        )

    latencies = [run["latency_s"] for run in runs]
    output_tps = [run["output_toks_per_s"] for run in runs]
    prompt_tokens = [run["prompt_tokens"] for run in runs]
    completion_tokens = [run["completion_tokens"] for run in runs]
    return {
        "measured_requests": len(runs),
        "latency_s_mean": float(statistics.fmean(latencies)),
        "latency_s_median": float(statistics.median(latencies)),
        "latency_s_max": float(max(latencies)),
        "output_toks_per_s_mean": float(statistics.fmean(output_tps)),
        "output_toks_per_s_median": float(statistics.median(output_tps)),
        "prompt_tokens_median": int(statistics.median(prompt_tokens)),
        "completion_tokens_median": int(statistics.median(completion_tokens)),
        "runs": runs,
    }


def benchmark_chat_request_parallel(
    *,
    base_url: str,
    api_key: str,
    served_model_name: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    warmup_requests: int = 1,
    num_prompts: int = 32,
    max_concurrency: int = 32,
) -> dict[str, Any]:
    payload = {
        "model": served_model_name,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }

    for _ in range(max(0, int(warmup_requests))):
        _json_post(
            url=f"{base_url}/v1/chat/completions",
            payload=payload,
            api_key=api_key,
        )

    def _run_one() -> dict[str, Any]:
        start = time.perf_counter()
        response = _json_post(
            url=f"{base_url}/v1/chat/completions",
            payload=payload,
            api_key=api_key,
        )
        latency_s = time.perf_counter() - start
        usage = dict(response.get("usage", {}))
        completion_tokens = int(usage.get("completion_tokens", 0))
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        return {
            "latency_s": float(latency_s),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "output_toks_per_s": (
                float(completion_tokens / latency_s)
                if completion_tokens > 0 and latency_s > 0
                else 0.0
            ),
            "response_id": response.get("id"),
            "finish_reason": (
                response.get("choices", [{}])[0].get("finish_reason")
                if isinstance(response.get("choices"), list)
                else None
            ),
        }

    runs: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, int(max_concurrency))) as executor:
        futures = [executor.submit(_run_one) for _ in range(max(1, int(num_prompts)))]
        for future in as_completed(futures):
            runs.append(future.result())
    wall_time_s = time.perf_counter() - started

    prompt_tokens_total = int(sum(run["prompt_tokens"] for run in runs))
    completion_tokens_total = int(sum(run["completion_tokens"] for run in runs))
    latencies = [run["latency_s"] for run in runs]

    return {
        "num_prompts": len(runs),
        "max_concurrency": int(max_concurrency),
        "wall_time_s": float(wall_time_s),
        "request_throughput": float(len(runs) / wall_time_s) if wall_time_s > 0 else 0.0,
        "output_toks_per_s": float(completion_tokens_total / wall_time_s) if wall_time_s > 0 else 0.0,
        "prompt_toks_per_s": float(prompt_tokens_total / wall_time_s) if wall_time_s > 0 else 0.0,
        "completion_tokens_total": completion_tokens_total,
        "prompt_tokens_total": prompt_tokens_total,
        "latency_s_mean": float(statistics.fmean(latencies)),
        "latency_s_median": float(statistics.median(latencies)),
        "runs": runs,
    }


def _extract_bench_summary(payload: dict[str, Any]) -> dict[str, Any]:
    preferred_keys = (
        "request_throughput",
        "output_throughput",
        "input_throughput",
        "mean_ttft_ms",
        "mean_tpot_ms",
        "mean_e2el_ms",
        "median_ttft_ms",
        "median_tpot_ms",
        "median_e2el_ms",
        "completed",
        "total_input_tokens",
        "total_output_tokens",
    )
    summary: dict[str, Any] = {}
    for key in preferred_keys:
        if key in payload:
            summary[key] = payload[key]
    if not summary and isinstance(payload.get("summary"), dict):
        nested = dict(payload["summary"])
        for key in preferred_keys:
            if key in nested:
                summary[key] = nested[key]
    return summary


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - exercised in Modal.
        process.kill()
        process.wait(timeout=30)


def run_standard_vllm_serve_benchmark(
    *,
    output_dir: Path,
    model_id: str,
    served_model_name: str = "benchmark-model",
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: str = "benchmark-token",
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = 1,
    max_num_seqs: int = 32,
    max_model_len: int | None = None,
    enable_chunked_prefill: bool = True,
    benchmark_messages: list[dict[str, str]] | None = None,
    benchmark_max_tokens: int = 256,
    warmup_requests: int = 1,
    measured_requests: int = 3,
    fixed_prompt_num_prompts: int = 32,
    fixed_prompt_max_concurrency: int = 32,
    bench_input_len: int = 2048,
    bench_output_len: int = 256,
    bench_num_prompts: int = 64,
    bench_max_concurrency: int = 32,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://{host}:{port}"
    serve_log_path = output_dir / "server.log"
    bench_log_path = output_dir / "bench_serve.log"
    bench_result_path = output_dir / "bench_serve_random.json"
    env = dict(os.environ)
    env.setdefault("OPENAI_API_KEY", api_key)

    serve_command = build_vllm_serve_command(
        model_id=model_id,
        api_key=api_key,
        host=host,
        port=port,
        served_model_name=served_model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        enable_chunked_prefill=enable_chunked_prefill,
    )
    with serve_log_path.open("wb") as serve_log:
        server_proc = subprocess.Popen(
            serve_command,
            stdout=serve_log,
            stderr=subprocess.STDOUT,
            env=env,
        )
    try:
        wait_for_vllm_server(base_url=base_url, api_key=api_key, timeout_s=900.0)

        request_summary = benchmark_chat_request(
            base_url=base_url,
            api_key=api_key,
            served_model_name=served_model_name,
            messages=list(benchmark_messages or DEFAULT_BENCHMARK_MESSAGES),
            max_tokens=benchmark_max_tokens,
            warmup_requests=warmup_requests,
            measured_requests=measured_requests,
        )
        fixed_prompt_parallel_summary = benchmark_chat_request_parallel(
            base_url=base_url,
            api_key=api_key,
            served_model_name=served_model_name,
            messages=list(benchmark_messages or DEFAULT_BENCHMARK_MESSAGES),
            max_tokens=benchmark_max_tokens,
            warmup_requests=1,
            num_prompts=fixed_prompt_num_prompts,
            max_concurrency=fixed_prompt_max_concurrency,
        )

        bench_command = build_vllm_bench_serve_command(
            base_url=base_url,
            auth_token=api_key,
            model_id=model_id,
            served_model_name=served_model_name,
            input_len=bench_input_len,
            output_len=bench_output_len,
            num_prompts=bench_num_prompts,
            max_concurrency=bench_max_concurrency,
            result_dir=output_dir,
            result_filename=bench_result_path.name,
        )
        with bench_log_path.open("wb") as bench_log:
            completed = subprocess.run(
                bench_command,
                stdout=bench_log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"vllm bench serve failed with exit code {completed.returncode}. See {bench_log_path}"
            )

        bench_payload = json.loads(bench_result_path.read_text())
        result = {
            "model_id": model_id,
            "served_model_name": served_model_name,
            "base_url": base_url,
            "serve_command": serve_command,
            "bench_command": bench_command,
            "serve_log_path": str(serve_log_path),
            "bench_log_path": str(bench_log_path),
            "bench_result_path": str(bench_result_path),
            "chat_request_benchmark": request_summary,
            "fixed_prompt_parallel_benchmark": fixed_prompt_parallel_summary,
            "bench_serve_summary": _extract_bench_summary(bench_payload),
            "bench_serve_result": bench_payload,
            "config": {
                "gpu_memory_utilization": float(gpu_memory_utilization),
                "tensor_parallel_size": int(tensor_parallel_size),
                "max_num_seqs": int(max_num_seqs),
                "max_model_len": None if max_model_len is None else int(max_model_len),
                "enable_chunked_prefill": bool(enable_chunked_prefill),
                "benchmark_max_tokens": int(benchmark_max_tokens),
                "fixed_prompt_num_prompts": int(fixed_prompt_num_prompts),
                "fixed_prompt_max_concurrency": int(fixed_prompt_max_concurrency),
                "bench_input_len": int(bench_input_len),
                "bench_output_len": int(bench_output_len),
                "bench_num_prompts": int(bench_num_prompts),
                "bench_max_concurrency": int(bench_max_concurrency),
            },
        }
        (output_dir / "benchmark_summary.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        _terminate_process(server_proc)


def _build_benchmark_patch_artifacts(
    *,
    tokenizer: Any,
    benchmark_messages: list[dict[str, str]],
    add_generation_prompt: bool,
    basis_npz_path: Path,
    results_json_path: Path,
    patch_mode: str,
    target_layers: tuple[int, ...],
    components_per_layer: int,
    strength: float,
) -> dict[str, Any]:
    from pipelines.interp.pooling import (
        _char_to_token_span,
        _token_offsets_for_rendered,
    )
    from pipelines.interp.patching.basis import default_phase17_market_patch_basis
    from pipelines.interp.patching.market_patch import MarketPatchSpec

    rendered = tokenizer.apply_chat_template(
        list(benchmark_messages),
        tokenize=False,
        add_generation_prompt=bool(add_generation_prompt),
    )
    if not isinstance(rendered, str):
        raise TypeError("Tokenizer did not return rendered benchmark chat text")
    _, offsets = _token_offsets_for_rendered(tokenizer, rendered)
    start_char, end_char = _benchmark_section_char_span(rendered)
    token_span = _char_to_token_span(offsets, start_char=start_char, end_char=end_char)
    if token_span is None:
        raise RuntimeError("Failed to map benchmark section chars onto token span")

    basis = default_phase17_market_patch_basis(
        basis_npz_path=basis_npz_path,
        results_json_path=results_json_path,
        layers=tuple(int(layer) for layer in target_layers),
        components_per_layer=int(components_per_layer),
    )
    basis_payload = basis.to_payload()
    component_indices_by_layer: dict[int, tuple[int, ...]] = {}
    for layer in target_layers:
        payload = basis_payload.get(int(layer))
        if payload is None:
            continue
        component_indices_by_layer[int(layer)] = tuple(
            range(min(int(components_per_layer), int(payload["components"].shape[0])))
        )

    patch_spec = MarketPatchSpec(
        mode=str(patch_mode),
        target_layers=tuple(int(layer) for layer in target_layers),
        token_span=(int(token_span[0]), int(token_span[1])),
        strength=float(strength),
        component_indices_by_layer=component_indices_by_layer,
    )
    return {
        "rendered_prompt_chars": len(rendered),
        "token_span": [int(token_span[0]), int(token_span[1])],
        "basis_payload": basis_payload,
        "patch_spec": patch_spec.to_payload(),
    }


def run_customop_vllm_benchmark(
    *,
    output_dir: Path,
    model_id: str,
    benchmark_messages: list[dict[str, str]] | None = None,
    benchmark_max_tokens: int = 256,
    warmup_requests: int = 1,
    measured_requests: int = 3,
    fixed_prompt_num_prompts: int = 32,
    measured_batch_runs: int = 3,
    max_num_seqs: int = 32,
    gpu_memory_utilization: float = 0.85,
    enable_chunked_prefill: bool = True,
    enable_prefix_caching: bool = False,
    use_custom_worker: bool = True,
    patch_mode: str = "",
    target_layers: tuple[int, ...] = (4, 35),
    components_per_layer: int = 4,
    strength: float = 1.0,
    basis_npz_path: Path | None = None,
    results_json_path: Path | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    from research.synthetic_market.synthetic_market_behavior_runner import _destroy_llm
    from pipelines.interp.vllm_capture import (
        VLLMCaptureConfig,
        _create_llm,
        _generate_batch_vllm,
        _generate_one_vllm,
        _register_market_patch_basis_on_model,
    )

    benchmark_messages = list(benchmark_messages or DEFAULT_BENCHMARK_MESSAGES)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if patch_mode.strip() and not use_custom_worker:
        raise ValueError("Active patch benchmark requires use_custom_worker=True")
    request_scoped_cfg = VLLMCaptureConfig(
        output_dir=output_dir / "_tmp_capture",
        model_id=model_id,
        capture_router=False,
        capture_residual=False,
        add_generation_prompt=True,
        gpu_memory_utilization=float(gpu_memory_utilization),
        enforce_eager=False,
        max_num_batched_tokens=max(40960, int(max_num_seqs) * 4096),
        max_num_seqs=max(1, int(max_num_seqs)),
        enable_prefix_caching=bool(enable_prefix_caching),
        enable_chunked_prefill=bool(enable_chunked_prefill),
        async_scheduling=False if int(max_num_seqs) > 1 else None,
        worker_cls=(
            "pipelines.interp.patching.request_worker.MarketPatchGPUWorker"
            if use_custom_worker
            else ""
        ),
        request_scoped_patching=bool(use_custom_worker),
    )

    patch_spec: dict[str, Any] | None = None
    patch_setup: dict[str, Any] | None = None
    if patch_mode.strip() and basis_npz_path is not None and results_json_path is not None:
        patch_setup = _build_benchmark_patch_artifacts(
            tokenizer=tokenizer,
            benchmark_messages=benchmark_messages,
            add_generation_prompt=request_scoped_cfg.add_generation_prompt,
            basis_npz_path=basis_npz_path,
            results_json_path=results_json_path,
            patch_mode=patch_mode,
            target_layers=tuple(int(layer) for layer in target_layers),
            components_per_layer=int(components_per_layer),
            strength=float(strength),
        )
        patch_spec = dict(patch_setup["patch_spec"])

    llm = _create_llm(request_scoped_cfg)
    try:
        if patch_setup is not None:
            _register_market_patch_basis_on_model(llm, patch_setup["basis_payload"])

        for _ in range(max(0, int(warmup_requests))):
            _generate_one_vllm(
                llm=llm,
                tokenizer=tokenizer,
                messages=benchmark_messages,
                config=request_scoped_cfg,
                max_tokens=int(benchmark_max_tokens),
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                patch_spec=patch_spec,
            )

        single_runs: list[dict[str, Any]] = []
        first_patch_stats: dict[str, Any] = {}
        for _ in range(max(1, int(measured_requests))):
            started = time.perf_counter()
            output = _generate_one_vllm(
                llm=llm,
                tokenizer=tokenizer,
                messages=benchmark_messages,
                config=request_scoped_cfg,
                max_tokens=int(benchmark_max_tokens),
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
                patch_spec=patch_spec,
            )
            latency_s = time.perf_counter() - started
            completion_tokens = len(output.get("generated_token_ids", []))
            prompt_tokens = len(output.get("input_ids", []))
            single_runs.append(
                {
                    "latency_s": float(latency_s),
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "output_toks_per_s": (
                        float(completion_tokens / latency_s)
                        if completion_tokens > 0 and latency_s > 0
                        else 0.0
                    ),
                    "finish_reason": output.get("finish_reason"),
                    "request_id": output.get("request_id"),
                    "patch_stats": output.get("patch_stats", {}),
                }
            )
            if not first_patch_stats and output.get("patch_stats"):
                first_patch_stats = dict(output["patch_stats"])

        batch_requests = [
            {"messages": benchmark_messages, "patch_spec": patch_spec}
            for _ in range(max(1, int(fixed_prompt_num_prompts)))
        ]
        _generate_batch_vllm(
            llm=llm,
            tokenizer=tokenizer,
            batch_requests=batch_requests,
            config=request_scoped_cfg,
            max_tokens=int(benchmark_max_tokens),
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
        )

        batch_runs: list[dict[str, Any]] = []
        for _ in range(max(1, int(measured_batch_runs))):
            started = time.perf_counter()
            outputs = _generate_batch_vllm(
                llm=llm,
                tokenizer=tokenizer,
                batch_requests=batch_requests,
                config=request_scoped_cfg,
                max_tokens=int(benchmark_max_tokens),
                temperature=0.0,
                top_p=1.0,
                top_k=-1,
            )
            wall_time_s = time.perf_counter() - started
            completion_tokens_total = int(
                sum(len(output.get("generated_token_ids", [])) for output in outputs)
            )
            prompt_tokens_total = int(
                sum(len(output.get("input_ids", [])) for output in outputs)
            )
            if not first_patch_stats:
                for output in outputs:
                    if output.get("patch_stats"):
                        first_patch_stats = dict(output["patch_stats"])
                        break
            batch_runs.append(
                {
                    "wall_time_s": float(wall_time_s),
                    "request_count": len(outputs),
                    "completion_tokens_total": completion_tokens_total,
                    "prompt_tokens_total": prompt_tokens_total,
                    "request_throughput": (
                        float(len(outputs) / wall_time_s)
                        if wall_time_s > 0
                        else 0.0
                    ),
                    "output_toks_per_s": (
                        float(completion_tokens_total / wall_time_s)
                        if wall_time_s > 0
                        else 0.0
                    ),
                    "prompt_toks_per_s": (
                        float(prompt_tokens_total / wall_time_s)
                        if wall_time_s > 0
                        else 0.0
                    ),
                }
            )

        latency_values = [run["latency_s"] for run in single_runs]
        single_output_tps = [run["output_toks_per_s"] for run in single_runs]
        batch_request_tps = [run["request_throughput"] for run in batch_runs]
        batch_output_tps = [run["output_toks_per_s"] for run in batch_runs]

        result = {
            "model_id": model_id,
            "patch_mode": str(patch_mode or "none"),
            "patch_enabled": bool(patch_spec is not None),
            "single_request_benchmark": {
                "measured_requests": len(single_runs),
                "latency_s_mean": float(statistics.fmean(latency_values)),
                "latency_s_median": float(statistics.median(latency_values)),
                "output_toks_per_s_mean": float(statistics.fmean(single_output_tps)),
                "output_toks_per_s_median": float(statistics.median(single_output_tps)),
                "prompt_tokens_median": int(statistics.median(run["prompt_tokens"] for run in single_runs)),
                "completion_tokens_median": int(
                    statistics.median(run["completion_tokens"] for run in single_runs)
                ),
                "runs": single_runs,
            },
            "fixed_prompt_parallel_benchmark": {
                "num_prompts": int(fixed_prompt_num_prompts),
                "max_concurrency": int(max_num_seqs),
                "wall_time_s_mean": float(statistics.fmean(run["wall_time_s"] for run in batch_runs)),
                "wall_time_s_median": float(statistics.median(run["wall_time_s"] for run in batch_runs)),
                "request_throughput_mean": float(statistics.fmean(batch_request_tps)),
                "request_throughput_median": float(statistics.median(batch_request_tps)),
                "output_toks_per_s_mean": float(statistics.fmean(batch_output_tps)),
                "output_toks_per_s_median": float(statistics.median(batch_output_tps)),
                "runs": batch_runs,
            },
            "patch_validation": first_patch_stats,
            "config": {
                "benchmark_max_tokens": int(benchmark_max_tokens),
                "max_num_seqs": int(max_num_seqs),
                "enable_chunked_prefill": bool(enable_chunked_prefill),
                "enable_prefix_caching": bool(enable_prefix_caching),
                "gpu_memory_utilization": float(gpu_memory_utilization),
                "use_custom_worker": bool(use_custom_worker),
                "target_layers": [int(layer) for layer in target_layers],
                "components_per_layer": int(components_per_layer),
                "strength": float(strength),
            },
        }
        if patch_setup is not None:
            result["patch_setup"] = {
                "token_span": list(patch_setup["token_span"]),
                "rendered_prompt_chars": int(patch_setup["rendered_prompt_chars"]),
            }
        (output_dir / "benchmark_summary.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        _destroy_llm(llm)


def run_customop_vs_stock_vllm_benchmark(
    *,
    output_dir: Path,
    model_id: str,
    served_model_name: str = "benchmark-model",
    gpu_memory_utilization: float = 0.85,
    max_num_seqs: int = 32,
    max_model_len: int | None = None,
    enable_chunked_prefill: bool = True,
    benchmark_messages: list[dict[str, str]] | None = None,
    benchmark_max_tokens: int = 256,
    warmup_requests: int = 1,
    measured_requests: int = 3,
    fixed_prompt_num_prompts: int = 32,
    fixed_prompt_max_concurrency: int = 32,
    measured_batch_runs: int = 3,
    bench_input_len: int = 2048,
    bench_output_len: int = 256,
    bench_num_prompts: int = 64,
    bench_max_concurrency: int = 32,
    patch_mode: str = "project_out",
    target_layers: tuple[int, ...] = (4, 35),
    components_per_layer: int = 4,
    strength: float = 1.0,
    basis_npz_path: Path | None = None,
    results_json_path: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_messages = list(benchmark_messages or DEFAULT_BENCHMARK_MESSAGES)

    stock_result = run_standard_vllm_serve_benchmark(
        output_dir=output_dir / "stock_serve",
        model_id=model_id,
        served_model_name=served_model_name,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        enable_chunked_prefill=enable_chunked_prefill,
        benchmark_messages=benchmark_messages,
        benchmark_max_tokens=benchmark_max_tokens,
        warmup_requests=warmup_requests,
        measured_requests=measured_requests,
        fixed_prompt_num_prompts=fixed_prompt_num_prompts,
        fixed_prompt_max_concurrency=fixed_prompt_max_concurrency,
        bench_input_len=bench_input_len,
        bench_output_len=bench_output_len,
        bench_num_prompts=bench_num_prompts,
        bench_max_concurrency=bench_max_concurrency,
    )
    custom_no_patch = run_customop_vllm_benchmark(
        output_dir=output_dir / "customop_no_patch",
        model_id=model_id,
        benchmark_messages=benchmark_messages,
        benchmark_max_tokens=benchmark_max_tokens,
        warmup_requests=warmup_requests,
        measured_requests=measured_requests,
        fixed_prompt_num_prompts=fixed_prompt_num_prompts,
        measured_batch_runs=measured_batch_runs,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_chunked_prefill=enable_chunked_prefill,
        patch_mode="",
    )
    custom_patch = run_customop_vllm_benchmark(
        output_dir=output_dir / "customop_patch",
        model_id=model_id,
        benchmark_messages=benchmark_messages,
        benchmark_max_tokens=benchmark_max_tokens,
        warmup_requests=warmup_requests,
        measured_requests=measured_requests,
        fixed_prompt_num_prompts=fixed_prompt_num_prompts,
        measured_batch_runs=measured_batch_runs,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_chunked_prefill=enable_chunked_prefill,
        patch_mode=patch_mode,
        target_layers=target_layers,
        components_per_layer=components_per_layer,
        strength=strength,
        basis_npz_path=basis_npz_path,
        results_json_path=results_json_path,
    )

    stock_single = float(stock_result["chat_request_benchmark"]["output_toks_per_s_mean"])
    stock_parallel = float(stock_result["fixed_prompt_parallel_benchmark"]["output_toks_per_s"])
    no_patch_single = float(custom_no_patch["single_request_benchmark"]["output_toks_per_s_mean"])
    no_patch_parallel = float(custom_no_patch["fixed_prompt_parallel_benchmark"]["output_toks_per_s_mean"])
    patch_single = float(custom_patch["single_request_benchmark"]["output_toks_per_s_mean"])
    patch_parallel = float(custom_patch["fixed_prompt_parallel_benchmark"]["output_toks_per_s_mean"])

    result = {
        "model_id": model_id,
        "served_model_name": served_model_name,
        "stock_serve": stock_result,
        "customop_no_patch": custom_no_patch,
        "customop_patch": custom_patch,
        "comparison": {
            "single_request_output_tps": {
                "stock_serve": stock_single,
                "customop_no_patch": no_patch_single,
                "customop_patch": patch_single,
                "customop_no_patch_vs_stock_ratio": _comparison_ratio(no_patch_single, stock_single),
                "customop_patch_vs_stock_ratio": _comparison_ratio(patch_single, stock_single),
            },
            "fixed_prompt_parallel_output_tps": {
                "stock_serve": stock_parallel,
                "customop_no_patch": no_patch_parallel,
                "customop_patch": patch_parallel,
                "customop_no_patch_vs_stock_ratio": _comparison_ratio(no_patch_parallel, stock_parallel),
                "customop_patch_vs_stock_ratio": _comparison_ratio(patch_parallel, stock_parallel),
            },
        },
    }
    (output_dir / "comparison_summary.json").write_text(json.dumps(result, indent=2))
    return result
