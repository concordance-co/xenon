from __future__ import annotations

from pipelines.interp.synthetic_market_behavior_runner import (
    SyntheticMarketBehaviorConfig,
    _build_generation_config,
    _run_generation_batch,
)


def test_build_generation_config_uses_request_scoped_worker_for_patched_runs():
    cfg = _build_generation_config(
        SyntheticMarketBehaviorConfig(
            patch_mode="project_out",
            batch_size=4,
        )
    )

    assert cfg.max_num_seqs == 4
    assert cfg.request_scoped_patching is True
    assert cfg.worker_cls == "pipelines.interp.vllm_request_patch_worker.MarketPatchGPUWorker"
    assert cfg.enable_chunked_prefill is False
    assert cfg.enable_prefix_caching is False
    assert cfg.async_scheduling is False
    assert cfg.max_num_batched_tokens == 40960


def test_build_generation_config_can_enable_chunked_prefill():
    cfg = _build_generation_config(
        SyntheticMarketBehaviorConfig(
            patch_mode="project_out",
            batch_size=8,
            enable_chunked_prefill=True,
        )
    )

    assert cfg.max_num_seqs == 8
    assert cfg.enable_chunked_prefill is True
    assert cfg.request_scoped_patching is True


def test_run_generation_batch_uses_single_prompt_path_for_single_request(monkeypatch):
    calls: list[str] = []

    def fake_generate_one_vllm(**kwargs):
        calls.append("single")
        return {"generated_token_ids": [], "generated_text": "", "finish_reason": "", "input_ids": [], "patch_stats": {}}

    monkeypatch.setattr(
        "pipelines.interp.synthetic_market_behavior_runner._generate_one_vllm",
        fake_generate_one_vllm,
    )

    out = _run_generation_batch(
        llm=object(),
        tokenizer=object(),
        requests=[{"messages": [{"role": "user", "content": "x"}]}],
        config=_build_generation_config(SyntheticMarketBehaviorConfig()),
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        tools=None,
        tool_choice=None,
    )

    assert calls == ["single"]
    assert len(out) == 1


def test_run_generation_batch_uses_batched_path_for_multiple_requests(monkeypatch):
    calls: list[str] = []

    def fake_generate_batch_vllm(**kwargs):
        calls.append("batch")
        return [
            {"generated_token_ids": [], "generated_text": "", "finish_reason": "", "input_ids": [], "patch_stats": {}},
            {"generated_token_ids": [], "generated_text": "", "finish_reason": "", "input_ids": [], "patch_stats": {}},
        ]

    monkeypatch.setattr(
        "pipelines.interp.synthetic_market_behavior_runner._generate_batch_vllm",
        fake_generate_batch_vllm,
    )

    out = _run_generation_batch(
        llm=object(),
        tokenizer=object(),
        requests=[
            {"messages": [{"role": "user", "content": "x"}]},
            {"messages": [{"role": "user", "content": "y"}]},
        ],
        config=_build_generation_config(SyntheticMarketBehaviorConfig(batch_size=2)),
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        tools=None,
        tool_choice=None,
    )

    assert calls == ["batch"]
    assert len(out) == 2
