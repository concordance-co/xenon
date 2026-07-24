from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from pipelines_v2.api import (
    ActivationPatchSpec,
    Dataset,
    Example,
    GenerationSpec,
    PatchedGenerationSpec,
    ResidualInterventionSite,
    TokenSelector,
    VLLMEngine,
)
from pipelines_v2.engine.vllm.intervene import _run_unpaired, vllm_intervention_session_key
from pipelines_v2.engine.vllm.intervention_output import stats_for_request


class _SamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)
        self.structured_outputs = None


class _ChatTokenizer:
    def __init__(self) -> None:
        self.chat_template_calls: list[dict[str, Any]] = []

    def apply_chat_template(self, prompt: Any, **kwargs: Any) -> Any:
        self.chat_template_calls.append(dict(kwargs))
        assert isinstance(prompt, list)
        return [11, 22, 33]


class _FakeLLM:
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[dict[str, Any]] = []
        self.sampling_params: list[Any] = []
        self.model = types.SimpleNamespace(_xenon_vllm_model_runner="v2")

    def generate(self, *, prompts: list[dict[str, Any]], sampling_params: list[Any]) -> list[Any]:
        self.prompts.extend(prompts)
        self.sampling_params.extend(sampling_params)
        return list(self.outputs)

    def apply_model(self, fn: Any) -> list[Any]:
        return [fn(self.model)]


def _install_fake_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    vllm_module = types.ModuleType("vllm")
    vllm_module.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)


def _unpaired_spec(engine: VLLMEngine, *, examples: list[Example] | None = None) -> PatchedGenerationSpec:
    return PatchedGenerationSpec(
        engine=engine,
        dataset=Dataset.from_examples(
            examples
            or [
                Example(
                    key="target",
                    prompt=[{"role": "user", "content": "ABC"}],
                )
            ],
            name="vllm_unpaired_patch_targets",
        ),
        patch=ActivationPatchSpec(
            write_site=ResidualInterventionSite(site="resid_post", layers=(4,)),
            target_tokens=TokenSelector.last(),
            strength=0.75,
        ),
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )


def _request_output(*, request_id: str = "req-1-extra") -> Any:
    return types.SimpleNamespace(
        request_id=request_id,
        outputs=[
            types.SimpleNamespace(
                text="patched answer",
                token_ids=[91, 92],
                finish_reason="stop",
            )
        ],
    )


@pytest.mark.unit
@pytest.mark.vllm
def test_stats_for_request_matches_request_id_suffixes_without_prefix_collision() -> None:
    batch_stats = {
        "req-2": {"4": {"marker": "wrong"}},
        "req": {"4": {"marker": "too_broad"}},
        "req-1": {"4": {"marker": "expected"}},
    }

    assert stats_for_request(batch_stats, "req-1-extra") == {"4": {"marker": "expected"}}
    assert stats_for_request({"req-1-extra": {"4": {"marker": "expected"}}}, "req-1") == {
        "4": {"marker": "expected"}
    }
    assert stats_for_request(batch_stats, "missing") == {}


@pytest.mark.unit
@pytest.mark.vllm
def test_vllm_intervention_session_key_tracks_runtime_relevant_engine_config() -> None:
    engine = VLLMEngine(model_id="fake-model", enable_prefix_caching=False, enforce_eager=False)
    spec = _unpaired_spec(engine)
    same_spec = _unpaired_spec(engine)
    wider_engine = VLLMEngine(
        model_id="fake-model",
        enable_prefix_caching=False,
        enforce_eager=False,
        max_num_seqs=2,
    )

    assert vllm_intervention_session_key(engine=engine, spec=spec) == vllm_intervention_session_key(
        engine=engine,
        spec=same_spec,
    )
    assert vllm_intervention_session_key(engine=engine, spec=spec) != vllm_intervention_session_key(
        engine=wider_engine,
        spec=_unpaired_spec(wider_engine),
    )


@pytest.mark.unit
@pytest.mark.vllm
def test_unpaired_patched_generation_forwards_chat_template_kwargs_and_builds_request_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vllm(monkeypatch)
    engine = VLLMEngine(
        model_id="fake-model",
        enable_prefix_caching=False,
        add_generation_prompt=True,
        extra={"chat_template_kwargs": {"tokenize_extra_flag": "kept"}},
    )
    tokenizer = _ChatTokenizer()
    llm = _FakeLLM([_request_output()])

    result = _run_unpaired(
        engine=engine,
        spec=_unpaired_spec(engine),
        llm=llm,
        tokenizer=tokenizer,
        reasoning_parser_instance=None,
        batch_size=8,
    )

    assert result.summary == {"example_count": 1, "patched_count": 1, "skipped_count": 0, "target_count": 1}
    assert result.rows[0]["generated_text"] == "patched answer"
    assert result.rows[0]["target_tokens"] == [2]
    assert result.metadata["model_runner"] == "v2"
    assert llm.prompts == [{"prompt_token_ids": [11, 22, 33]}]
    assert tokenizer.chat_template_calls[0]["add_generation_prompt"] is True
    assert tokenizer.chat_template_calls[0]["tokenize_extra_flag"] == "kept"

    activation_patch_spec = llm.sampling_params[0].kwargs["extra_args"]["activation_patch_spec"]
    assert activation_patch_spec["operator"] == "activation_patch"
    assert activation_patch_spec["target_layers"] == [4]
    assert activation_patch_spec["target_positions"] == [2]
    assert activation_patch_spec["source_layer_map"] == {"4": 4}
    assert activation_patch_spec["strength"] == 0.75


@pytest.mark.unit
@pytest.mark.vllm
def test_unpaired_patched_generation_rejects_output_count_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_vllm(monkeypatch)
    engine = VLLMEngine(model_id="fake-model", enable_prefix_caching=False)

    with pytest.raises(RuntimeError, match="patched request outputs"):
        _run_unpaired(
            engine=engine,
            spec=_unpaired_spec(engine),
            llm=_FakeLLM([]),
            tokenizer=_ChatTokenizer(),
            reasoning_parser_instance=None,
            batch_size=8,
        )
