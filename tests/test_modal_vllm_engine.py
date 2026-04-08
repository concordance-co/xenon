from __future__ import annotations

import importlib
import sys
from types import ModuleType
from types import SimpleNamespace


def _load_modal_vllm_engine():
    vllm_module = ModuleType("vllm")
    sampling_params_module = ModuleType("vllm.sampling_params")

    class _SamplingParams:
        def __init__(self, **kwargs):
            self.max_tokens = kwargs.get("max_tokens")
            self.temperature = kwargs.get("temperature")
            self.top_p = kwargs.get("top_p")
            self.top_k = kwargs.get("top_k")
            self.extra_args = None
            self.structured_outputs = None

    class _StructuredOutputsParams:
        def __init__(self, **kwargs):
            self.json = kwargs.get("json")

    vllm_module.SamplingParams = _SamplingParams
    sampling_params_module.StructuredOutputsParams = _StructuredOutputsParams
    sys.modules["vllm"] = vllm_module
    sys.modules["vllm.sampling_params"] = sampling_params_module
    return importlib.import_module("pipelines.interp.modal_vllm_engine")


class _FakeTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return [len(messages), 7, 11]


class _FakeOutput:
    def __init__(self, request_id: str, text: str):
        self.request_id = request_id
        self.outputs = [
            SimpleNamespace(
                token_ids=[101, 202],
                text=text,
                finish_reason="stop",
            )
        ]


class _FakeLLM:
    def __init__(self):
        self.chat_calls = []
        self.generate_calls = []

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        return [
            _FakeOutput("req-0", '{"name":"buy_token"}'),
            _FakeOutput("req-1", '{"name":"sell_token"}'),
        ]

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [
            _FakeOutput("req-0", '{"name":"buy_token"}'),
            _FakeOutput("req-1", '{"name":"sell_token"}'),
        ]


def test_generate_batch_uses_generate_when_tools_are_present(monkeypatch):
    engine = _load_modal_vllm_engine()
    llm = _FakeLLM()
    tokenizer = _FakeTokenizer()

    monkeypatch.setattr(engine, "_collect_activation_patch_stats_from_model", lambda *args, **kwargs: {})

    config = engine.VLLMCaptureConfig(
        add_generation_prompt=True,
        request_scoped_patching=False,
    )
    requests = [
        {"messages": [{"role": "user", "content": "prompt-a"}]},
        {"messages": [{"role": "user", "content": "prompt-b"}]},
    ]

    out = engine._generate_batch_vllm(
        llm=llm,
        tokenizer=tokenizer,
        batch_requests=requests,
        config=config,
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        tools=[{"type": "function", "function": {"name": "buy_token"}}],
        tool_choice="required",
    )

    assert llm.chat_calls == []
    assert len(llm.generate_calls) == 1
    assert llm.generate_calls[0]["prompts"] == [
        {"prompt_token_ids": [1, 7, 11]},
        {"prompt_token_ids": [1, 7, 11]},
    ]
    sampling_params = llm.generate_calls[0]["sampling_params"]
    assert len(sampling_params) == 2
    assert sampling_params[0].structured_outputs is not None
    assert sampling_params[0].structured_outputs.json == {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": ["buy_token"]},
            "arguments": {"type": "object", "properties": {}},
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }
    assert [row["generated_text"] for row in out] == ['{"name":"buy_token"}', '{"name":"sell_token"}']
    assert [row["input_ids"] for row in out] == [[1, 7, 11], [1, 7, 11]]


def test_capture_one_vllm_keeps_capture_pass_prefill_only_when_generation_enabled(monkeypatch):
    engine = _load_modal_vllm_engine()
    llm = _FakeLLM()
    tokenizer = _FakeTokenizer()

    generation_calls: list[dict[str, object]] = []

    monkeypatch.setattr(engine, "_reset_router_buffers_on_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(engine, "_collect_router_logits_from_model", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        engine,
        "_generate_one_vllm",
        lambda **kwargs: generation_calls.append(dict(kwargs)) or {
            "input_ids": [1, 7, 11],
            "generated_token_ids": [303],
            "generated_text": "final answer",
            "finish_reason": "stop",
            "reasoning_text": "chain",
            "request_id": "req-final",
        },
    )

    config = engine.VLLMCaptureConfig(
        capture_router=True,
        capture_residual=False,
        capture_generation=True,
        capture_reasoning=True,
        generation_max_tokens=12,
        generation_temperature=0.3,
        generation_top_p=0.8,
    )

    residual, router_logits, router_indices, input_ids, generation_result = engine._capture_one_vllm(
        llm=llm,
        tokenizer=tokenizer,
        messages=[{"role": "user", "content": "prompt-a"}],
        config=config,
        log_id="row-1",
    )

    assert residual is None
    assert router_logits is None
    assert router_indices is None
    assert input_ids == [1, 7, 11]
    assert len(llm.generate_calls) == 1
    assert llm.generate_calls[0]["sampling_params"].max_tokens == 1
    assert len(generation_calls) == 1
    assert generation_calls[0]["max_tokens"] == 12
    assert generation_result["generated_text"] == "final answer"
    assert generation_result["reasoning_text"] == "chain"
