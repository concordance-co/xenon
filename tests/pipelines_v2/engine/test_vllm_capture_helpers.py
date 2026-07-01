from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from pipelines_v2.api import Example, MoERoutingSite, RoutingRecord, TokenSelector
from pipelines_v2.engine.vllm.capture import _capture_prompt_batch, _fill_router_features, _prompt_token_ids
from pipelines_v2.engine.vllm.generate import _generation_rows_from_outputs


class _ForwardingTokenizer:
    def __init__(self) -> None:
        self.chat_template_calls: list[dict[str, Any]] = []

    def apply_chat_template(self, prompt: Any, **kwargs: Any) -> Any:
        self.chat_template_calls.append(dict(kwargs))
        assert isinstance(prompt, list)
        if kwargs.get("tokenize") is True:
            return [101, 201, 202, 102]
        return "AB"

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool) -> Any:
        assert text == "AB"
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return types.SimpleNamespace(
            input_ids=[201, 202],
            offset_mapping=[(0, 1), (1, 2)],
        )


@pytest.mark.unit
@pytest.mark.vllm
def test_prompt_token_ids_forwards_chat_template_controls_and_rebases_metadata() -> None:
    tokenizer = _ForwardingTokenizer()
    tool = {"type": "function", "function": {"name": "lookup"}}

    result = _prompt_token_ids(
        tokenizer=tokenizer,
        example=Example(
            key="chat",
            prompt=[{"role": "user", "content": "ignored"}],
            metadata={
                "token_sections": {"BODY": {"char_start": 0, "char_end": 2}},
                "section_records": [{"name": "BODY", "char_start": 0, "char_end": 2, "unit": "section"}],
            },
        ),
        add_generation_prompt=True,
        require_sections=True,
        prompt_metadata_builder=None,
        tools=[tool],
        tool_choice={"type": "function", "function": {"name": "lookup"}},
        enable_thinking=False,
        chat_template_kwargs={"tokenize_extra_flag": "kept"},
    )

    assert result["token_ids"] == [101, 201, 202, 102]
    assert result["token_sections"] == {"BODY": [1, 2]}
    assert result["section_records"][0]["token_positions"] == [1, 2]
    assert len(tokenizer.chat_template_calls) == 2
    for call in tokenizer.chat_template_calls:
        assert call["add_generation_prompt"] is True
        assert call["tools"] == [tool]
        assert call["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
        assert call["enable_thinking"] is False
        assert call["tokenize_extra_flag"] == "kept"
    assert [call["tokenize"] for call in tokenizer.chat_template_calls] == [True, False]


@pytest.mark.unit
@pytest.mark.vllm
def test_fill_router_features_materializes_observed_records_and_rebases_sections() -> None:
    feature_payloads = {
        "router_body": {
            "kind": "moe_routing",
            "routing_policy": {"source": "vllm_gate_logits", "observed_routing_decisions": True},
            "layers": {"3": {}},
        }
    }
    site = MoERoutingSite(
        name="router_body",
        layers=[3],
        tokens=TokenSelector.section("BODY"),
        record=[
            RoutingRecord.gate_logits(dtype="float32"),
            RoutingRecord.gate_probs(dtype="float32"),
            RoutingRecord.routing_decisions(required=True),
            RoutingRecord.topk_from_gate(k=2),
            RoutingRecord.expert_load(source="routing_decisions"),
        ],
    )
    logits = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 1.0, 0.0, -1.0],
            [-2.0, 5.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    topk_ids = np.asarray([[3, 2], [0, 1], [1, 3]], dtype=np.int64)
    topk_weights = np.asarray([[0.8, 0.2], [0.9, 0.1], [0.7, 0.3]], dtype=np.float32)

    _fill_router_features(
        feature_payloads=feature_payloads,
        routing_sites=[site],
        router_data={3: {"logits": logits, "topk_ids": topk_ids, "topk_weights": topk_weights}},
        example=Example(key="ex_router", prompt="unused"),
        token_count=3,
        token_sections={"BODY": [1, 2]},
        section_records=[
            {
                "name": "BODY",
                "unit": "section",
                "index": 0,
                "token_positions": [1, 2],
                "tags": {"source": "test"},
            }
        ],
        discovered_router_layers=[3],
    )

    row = feature_payloads["router_body"]["layers"]["3"]["ex_router"]
    assert row["tokens"] == [1, 2]
    assert row["token_sections"] == {"BODY": [0, 1]}
    assert row["section_records"][0]["token_positions"] == [0, 1]
    assert set(row["records"]) == {"1", "2"}

    first = row["records"]["1"]
    np.testing.assert_allclose(first["gate_logits"], logits[1])
    np.testing.assert_allclose(first["gate_probs"], np.exp(logits[1] - logits[1].max()) / np.exp(logits[1] - logits[1].max()).sum())
    assert first["routing_decisions"]["source"] == "observed"
    np.testing.assert_array_equal(first["routing_decisions"]["expert_ids"], topk_ids[1])
    np.testing.assert_allclose(first["routing_decisions"]["weights"], topk_weights[1])
    assert first["topk_from_gate"]["expert_ids"].tolist() == [0, 1]
    assert first["expert_load"]["source"] == "routing_decisions"
    assert first["expert_load"]["counts"] == {"0": 1, "1": 1}


@pytest.mark.unit
@pytest.mark.vllm
def test_capture_prompt_batch_rejects_vllm_output_count_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    vllm_module = types.ModuleType("vllm")

    class _SamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = dict(kwargs)

    vllm_module.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)

    class _Tokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool) -> Any:
            del text, add_special_tokens, return_offsets_mapping
            return types.SimpleNamespace(input_ids=[1], offset_mapping=[(0, 1)])

    class _LLM:
        def generate(self, *, prompts: list[dict[str, Any]], sampling_params: Any) -> list[Any]:
            del prompts, sampling_params
            return []

    with pytest.raises(RuntimeError, match="different number of request outputs"):
        _capture_prompt_batch(
            llm=_LLM(),
            tokenizer=_Tokenizer(),
            examples=[Example(key="a", prompt="a")],
            add_generation_prompt=False,
            require_sections=False,
            prompt_metadata_builder=None,
            wants_residual=False,
            wants_routing=False,
            wants_generation=True,
            generation_max_tokens=1,
            generation_temperature=0.0,
        )


@pytest.mark.unit
@pytest.mark.vllm
def test_generation_rows_preserve_reasoning_and_structured_payloads_and_reject_mismatch() -> None:
    examples = [Example(key="a", prompt="alpha"), Example(key="b", prompt="beta")]
    rows = _generation_rows_from_outputs(
        examples,
        [
            {
                "text": "answer-a",
                "generated_token_ids": [10, 11],
                "finish_reason": "stop",
                "request_id": "req-a",
                "reasoning_text": "thought-a",
                "structured_output": {"action": "buy"},
            },
            {
                "text": "answer-b",
                "generated_token_ids": [20],
                "finish_reason": "length",
                "request_id": "req-b",
            },
        ],
    )

    assert rows[0]["example_key"] == "a"
    assert rows[0]["example"]["key"] == "a"
    assert rows[0]["generated_text"] == "answer-a"
    assert rows[0]["generated_token_ids"] == [10, 11]
    assert rows[0]["reasoning_text"] == "thought-a"
    assert rows[0]["structured_output"] == {"action": "buy"}
    assert "reasoning_text" not in rows[1]

    with pytest.raises(RuntimeError, match="output count does not match"):
        _generation_rows_from_outputs(examples, rows[:1])


class _CountingTokenizer:
    """Tokenizer stub: one token per character of the prompt string."""

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool) -> Any:
        return types.SimpleNamespace(
            input_ids=list(range(len(text))),
            offset_mapping=[(i, i + 1) for i in range(len(text))],
        )


@pytest.mark.unit
@pytest.mark.vllm
def test_preflight_prompt_lengths_rejects_overlong_examples() -> None:
    from pipelines_v2.core.types import SpecValidationError
    from pipelines_v2.engine.vllm.capture import _preflight_prompt_lengths

    tokenizer = _CountingTokenizer()
    examples = [
        Example(key="fits", prompt="ab"),
        Example(key="too_long", prompt="abcdefgh"),
    ]

    # limit = max_model_len - 1 reserved token = 7; "too_long" is 8 tokens.
    with pytest.raises(SpecValidationError, match=r"too_long=8 tokens"):
        _preflight_prompt_lengths(
            tokenizer=tokenizer,
            examples=examples,
            max_model_len=8,
            add_generation_prompt=False,
            generation_max_tokens=None,
        )

    # Same prompts pass once the window covers them.
    _preflight_prompt_lengths(
        tokenizer=tokenizer,
        examples=examples,
        max_model_len=9,
        add_generation_prompt=False,
        generation_max_tokens=None,
    )

    # No max_model_len means no constraint to enforce.
    _preflight_prompt_lengths(
        tokenizer=tokenizer,
        examples=examples,
        max_model_len=None,
        add_generation_prompt=False,
        generation_max_tokens=None,
    )

    # Generation budget tightens the usable window.
    with pytest.raises(SpecValidationError, match=r"2 example\(s\) exceed"):
        _preflight_prompt_lengths(
            tokenizer=tokenizer,
            examples=examples,
            max_model_len=8,
            add_generation_prompt=False,
            generation_max_tokens=7,
        )
