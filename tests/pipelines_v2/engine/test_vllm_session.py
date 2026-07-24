from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

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
from pipelines_v2.engine.vllm.activation_patch_request_worker import (
    ActivationPatchGPUModelRunner,
    ActivationPatchGPUModelRunnerV2,
    ActivationPatchGPUWorker,
    ActivationPatchRequestHelper,
    compiled_operator_hint_from_config,
    force_v1_model_runner_for_activation_patching,
    force_v2_model_runner_for_activation_patching,
)
from pipelines_v2.engine.vllm import activation_patch_request_worker
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
        assert kwargs["compilation_config"]["cudagraph_mode"] == "PIECEWISE"
        assert kwargs["additional_config"]["xenon_activation_patch_compiled_operator"] == "subspace"
        assert kwargs["speculative_config"]["draft_model_config"]["hf_config"][
            "eagle_aux_hidden_state_layer_ids"
        ] == [0, 2]
        assert kwargs["kv_transfer_config"]["kv_connector"] == "ExampleHiddenStatesConnector"
        assert "kv_connector_module_path" not in kwargs["kv_transfer_config"]
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


def test_activation_patch_worker_reads_compiled_operator_hint_from_additional_config() -> None:
    class _Config:
        additional_config = {"xenon_activation_patch_compiled_operator": "subspace"}

    assert compiled_operator_hint_from_config(_Config()) == "subspace"


def test_activation_patch_worker_forces_v1_model_runner_before_engine_construction(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")

    force_v1_model_runner_for_activation_patching()

    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_activation_patch_worker_selects_v2_model_runner_before_engine_construction(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    force_v2_model_runner_for_activation_patching()

    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"


def test_activation_patch_runner_selection_invalidates_vllm_environment_cache(
    monkeypatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setitem(
        sys.modules,
        "vllm.envs",
        SimpleNamespace(disable_envs_cache=lambda: calls.append("disabled")),
    )

    force_v2_model_runner_for_activation_patching()

    assert calls == ["disabled"]


def test_activation_patch_runner_preserves_vllm_deferred_state_correction(
    monkeypatch,
) -> None:
    def correction() -> None:
        return None

    monkeypatch.setattr(
        activation_patch_request_worker.GPUModelRunner,
        "_update_states",
        lambda _runner, _scheduler_output: correction,
        raising=False,
    )
    runner = object.__new__(ActivationPatchGPUModelRunner)
    runner.activation_patch_request_helper = ActivationPatchRequestHelper()
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        finished_req_ids=set(),
    )

    assert runner._update_states(scheduler_output) is correction


def test_activation_patch_request_helper_supports_model_runner_v2_input_batch() -> None:
    helper = ActivationPatchRequestHelper()
    helper.req_id_to_patch_spec["req-a"] = {
        "operator": "project_out",
        "target_positions": [2],
        "target_policy": {"kind": "static"},
    }
    input_batch = SimpleNamespace(
        req_ids=["req-a"],
        num_computed_tokens_np=[1],
        prefill_len_np=[4],
    )

    helper.build_step_specs(
        input_batch=input_batch,
        num_scheduled_tokens=[3],
        num_prompt_tokens=[4],
    )

    assert len(helper.current_step_specs) == 1
    assert helper.current_step_specs[0]["req_id"] == "req-a"
    assert helper.current_step_specs[0]["patch_spec"]["query_positions"] == [1]
    assert helper.current_step_specs[0]["chunk_abs_span"] == [1, 4]


def test_activation_patch_model_runner_v2_prepares_request_scoped_batch_specs(
    monkeypatch,
) -> None:
    input_batch = SimpleNamespace(
        req_ids=["req-a"],
        num_computed_tokens_np=[1],
        prefill_len_np=[4],
        num_scheduled_tokens=[3],
        idx_mapping_np=np.asarray([0]),
    )
    monkeypatch.setattr(
        activation_patch_request_worker.GPUModelRunnerV2,
        "prepare_inputs",
        lambda _runner, _scheduler_output, _batch_desc: input_batch,
        raising=False,
    )
    captured: list[tuple[object, list[dict[str, object]]]] = []
    monkeypatch.setattr(
        "pipelines_v2.engine.vllm.activation_patch_core.set_batch_patch_specs",
        lambda model, specs: captured.append((model, specs)),
    )
    runner = object.__new__(ActivationPatchGPUModelRunnerV2)
    runner.model = object()
    runner.req_states = SimpleNamespace(prompt_len=SimpleNamespace(np=np.asarray([4])))
    runner.activation_patch_request_helper = ActivationPatchRequestHelper()
    runner.activation_patch_request_helper.req_id_to_patch_spec["req-a"] = {
        "operator": "project_out",
        "target_positions": [2],
        "target_policy": {"kind": "static"},
    }

    prepared = runner.prepare_inputs(SimpleNamespace(), SimpleNamespace())

    assert prepared is input_batch
    assert captured[0][0] is runner.model
    assert captured[0][1][0]["req_id"] == "req-a"


def test_activation_patch_model_runner_v2_uses_original_prompt_length_after_resume(
    monkeypatch,
) -> None:
    input_batch = SimpleNamespace(
        req_ids=["req-a"],
        num_computed_tokens_np=[4],
        prefill_len_np=[6],
        num_scheduled_tokens=[2],
        idx_mapping_np=np.asarray([0]),
    )
    monkeypatch.setattr(
        activation_patch_request_worker.GPUModelRunnerV2,
        "prepare_inputs",
        lambda _runner, _scheduler_output, _batch_desc: input_batch,
        raising=False,
    )
    captured: list[list[dict[str, object]]] = []
    monkeypatch.setattr(
        "pipelines_v2.engine.vllm.activation_patch_core.set_batch_patch_specs",
        lambda _model, specs: captured.append(specs),
    )
    runner = object.__new__(ActivationPatchGPUModelRunnerV2)
    runner.model = object()
    runner.req_states = SimpleNamespace(prompt_len=SimpleNamespace(np=np.asarray([3])))
    runner.activation_patch_request_helper = ActivationPatchRequestHelper()
    runner.activation_patch_request_helper.req_id_to_patch_spec["req-a"] = {
        "operator": "project_out",
        "target_policy": {
            "kind": "every_token",
            "include_prompt": False,
            "include_decode": True,
        },
    }

    runner.prepare_inputs(SimpleNamespace(), SimpleNamespace())

    patch_spec = captured[0][0]["patch_spec"]
    assert patch_spec["query_span"] == [0, 2]
    assert patch_spec["phase_counts"] == {"prompt": 0, "decode": 2}


def test_activation_patch_request_replacement_drops_stale_payload() -> None:
    helper = ActivationPatchRequestHelper()
    helper.req_id_to_patch_spec["req-a"] = {"operator": "project_out"}

    helper.process_new_reqs(
        [
            SimpleNamespace(
                req_id="req-a",
                sampling_params=SimpleNamespace(extra_args=None),
            )
        ]
    )

    assert "req-a" not in helper.req_id_to_patch_spec


def test_activation_patch_worker_installs_and_restores_model_runner_v2_class(
    monkeypatch,
) -> None:
    module = activation_patch_request_worker.gpu_model_runner_v2
    original_runner_cls = module.GPUModelRunner

    def init_device(worker: object) -> None:
        worker.model_runner = module.GPUModelRunner()

    monkeypatch.setattr(
        activation_patch_request_worker.gpu_worker.Worker,
        "init_device",
        init_device,
        raising=False,
    )
    worker = object.__new__(ActivationPatchGPUWorker)
    worker.use_v2_model_runner = True

    worker.init_device()

    assert isinstance(worker.model_runner, ActivationPatchGPUModelRunnerV2)
    assert module.GPUModelRunner is original_runner_cls
