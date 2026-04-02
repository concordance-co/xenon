from __future__ import annotations

from pathlib import Path

from pipelines.interp.patching.serve_benchmark import (
    _benchmark_section_char_span,
    _comparison_ratio,
    build_vllm_bench_serve_command,
    build_vllm_serve_command,
)


def test_build_vllm_serve_command_includes_chunked_prefill_when_enabled():
    cmd = build_vllm_serve_command(
        model_id="/models/Qwen/Qwen3-30B-A3B",
        api_key="token-123",
        max_num_seqs=32,
        enable_chunked_prefill=True,
    )

    assert cmd[:3] == ["vllm", "serve", "/models/Qwen/Qwen3-30B-A3B"]
    assert "--api-key" in cmd
    assert "--generation-config" in cmd
    assert "--max-num-seqs" in cmd
    assert "--enable-chunked-prefill" in cmd


def test_build_vllm_serve_command_can_disable_chunked_prefill():
    cmd = build_vllm_serve_command(
        model_id="/models/Qwen/Qwen3-30B-A3B",
        api_key="token-123",
        enable_chunked_prefill=False,
    )

    assert "--enable-chunked-prefill" not in cmd


def test_build_vllm_bench_serve_command_targets_stock_server():
    cmd = build_vllm_bench_serve_command(
        base_url="http://127.0.0.1:8000",
        auth_token="token-123",
        model_id="/models/Qwen/Qwen3-30B-A3B",
        served_model_name="benchmark-model",
        input_len=2048,
        output_len=256,
        num_prompts=64,
        max_concurrency=32,
        result_dir=Path("/tmp/bench"),
    )

    assert cmd[:3] == ["vllm", "bench", "serve"]
    assert "--backend" in cmd
    assert "openai" in cmd
    assert "--endpoint" in cmd
    assert "/v1/completions" in cmd
    assert "--dataset-name" in cmd
    assert "random" in cmd
    assert "--header" in cmd
    assert "--served-model-name" in cmd


def test_benchmark_section_char_span_finds_assets_block():
    rendered = (
        "<sys>system</sys>\n"
        "<user>Intro text\n"
        "Assets:\n"
        "- ALP: momentum\n"
        "- BRV: slower\n\n"
        "Respond with the ticker.</user>"
    )

    start_char, end_char = _benchmark_section_char_span(rendered)

    assert rendered[start_char:].startswith("Assets:")
    assert rendered[end_char:].startswith("Respond with")


def test_comparison_ratio_handles_zero_denominator():
    assert _comparison_ratio(10.0, 5.0) == 2.0
    assert _comparison_ratio(10.0, 0.0) is None
