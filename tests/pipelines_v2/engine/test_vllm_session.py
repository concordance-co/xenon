from __future__ import annotations

from pipelines_v2.api import (
    Dataset,
    Example,
    GenerationRunSpec,
    GenerationSpec,
    InterchangePatch,
    PatchedGenerationSpec,
    ProjectOutPatch,
    ResidualInterventionSite,
    ResidualSite,
    TokenSelector,
    VLLMEngine,
)
from pipelines_v2.engine.vllm.session import build_vllm_session_llm_kwargs, vllm_session_key
from pipelines_v2.operations.capture import CaptureSpec


def test_vllm_session_builds_superset_runtime_for_capture_generation_and_patching() -> None:
    engine = VLLMEngine(
        model_id="Qwen/Qwen3-test",
        enforce_eager=False,
        enable_prefix_caching=False,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
    )
    dataset = Dataset.from_examples(
        [
            Example(key="a", prompt="alpha", labels={"class": "positive"}, case_key="case"),
            Example(key="b", prompt="bravo", labels={"class": "negative"}, case_key="case"),
        ]
    )
    capture = CaptureSpec(
        engine=engine,
        dataset=dataset,
        sites=[
            ResidualSite(name="resid_a", site="resid_post", layers=(2,)),
            ResidualSite(name="resid_b", site="resid_post", layers=(0,)),
        ],
        generation=GenerationSpec(
            enabled=True,
            max_tokens=4,
            capture_reasoning=True,
            capture_generated_tokens=True,
        ),
    )
    generation = GenerationRunSpec(
        engine=engine,
        dataset=dataset,
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )
    patch = PatchedGenerationSpec(
        engine=engine,
        dataset=dataset,
        patch=ProjectOutPatch(
            write_site=ResidualInterventionSite(site="resid_post", layers=(4,)),
            target_tokens=TokenSelector.last(),
            subspace={"layers": {"4": {"components": []}}},
        ),
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )
    paired_patch = PatchedGenerationSpec(
        engine=engine,
        dataset=dataset,
        patch=InterchangePatch(
            write_site=ResidualInterventionSite(site="resid_post", layers=(4,)),
            target_tokens=TokenSelector.last(),
            activation_bank={"layers": {}},
            donor_tokens=TokenSelector.last(),
        ),
        pair_by=dataset.cases("case_key"),
        target_when=dataset.labels("class").equals("positive"),
        donor_when=dataset.labels("class").equals("negative"),
        generation=GenerationSpec(enabled=True, max_tokens=2),
    )

    kwargs, reasoning_parser, tempdir = build_vllm_session_llm_kwargs(
        engine=engine,
        specs=(capture, generation, patch, paired_patch),
    )
    try:
        assert kwargs["worker_cls"] == "pipelines_v2.engine.vllm.activation_patch_request_worker.ActivationPatchGPUWorker"
        assert kwargs["max_num_seqs"] == 4
        assert kwargs["max_num_batched_tokens"] == 4096
        assert kwargs["compilation_config"]["custom_ops"] == ["none", "+activation_patch_hidden_states"]
        assert kwargs["additional_config"]["xenon_activation_patch_compiled_operator"] == "subspace"
        assert kwargs["speculative_config"]["draft_model_config"]["hf_config"][
            "eagle_aux_hidden_state_layer_ids"
        ] == [0, 2]
        assert kwargs["kv_transfer_config"]["kv_connector"] == "PipelinesV2HiddenStatesConnector"
        assert kwargs["kv_transfer_config"]["kv_connector_module_path"] == (
            "pipelines_v2.engine.vllm.hidden_states_connector"
        )
        assert reasoning_parser == "qwen3"
        assert vllm_session_key(engine=engine, specs=(capture, generation, patch, paired_patch)) == vllm_session_key(
            engine=engine,
            specs=(generation, capture, paired_patch, patch),
        )
    finally:
        if tempdir is not None:
            tempdir.cleanup()


def test_vllm_session_forwards_allowed_extra_llm_kwargs() -> None:
    engine = VLLMEngine(
        model_id="fake-model",
        enforce_eager=True,
        extra={"attention_backend": "FLASH_ATTN"},
    )

    kwargs, _, tempdir = build_vllm_session_llm_kwargs(engine=engine, specs=())
    try:
        assert kwargs["attention_backend"] == "FLASH_ATTN"
    finally:
        if tempdir is not None:
            tempdir.cleanup()
