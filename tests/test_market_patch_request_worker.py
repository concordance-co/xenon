from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from types import ModuleType


def _load_request_helper_class():
    vllm_module = ModuleType("vllm")
    v1_module = ModuleType("vllm.v1")
    worker_module = ModuleType("vllm.v1.worker")
    gpu_model_runner_module = ModuleType("vllm.v1.worker.gpu_model_runner")
    gpu_worker_module = ModuleType("vllm.v1.worker.gpu_worker")

    class _GPUModelRunner:
        pass

    class _Worker:
        pass

    gpu_model_runner_module.GPUModelRunner = _GPUModelRunner
    gpu_worker_module.Worker = _Worker
    worker_module.gpu_model_runner = gpu_model_runner_module
    worker_module.gpu_worker = gpu_worker_module
    v1_module.worker = worker_module
    vllm_module.v1 = v1_module

    sys.modules.setdefault("vllm", vllm_module)
    sys.modules.setdefault("vllm.v1", v1_module)
    sys.modules.setdefault("vllm.v1.worker", worker_module)
    sys.modules.setdefault("vllm.v1.worker.gpu_model_runner", gpu_model_runner_module)
    sys.modules.setdefault("vllm.v1.worker.gpu_worker", gpu_worker_module)

    module = importlib.import_module("pipelines.interp.patching.request_worker")
    return module.ActivationPatchRequestHelper


def test_request_helper_builds_step_specs_for_multiple_patch_specs() -> None:
    ActivationPatchRequestHelper = _load_request_helper_class()
    helper = ActivationPatchRequestHelper()
    helper.process_new_reqs(
        [
            SimpleNamespace(
                req_id="req-1",
                sampling_params=SimpleNamespace(
                    extra_args={
                        "patch_specs": [
                            {"mode": "project_out", "token_span": [2, 6], "target_layers": [4]},
                            {"mode": "swap_components", "token_span": [2, 6], "target_layers": [40]},
                        ]
                    }
                ),
            )
        ]
    )

    helper.build_step_specs(
        input_batch=SimpleNamespace(
            req_ids=["req-1"],
            num_computed_tokens_cpu=[2],
            num_prompt_tokens=[8],
        ),
        num_scheduled_tokens=[4],
    )

    assert len(helper.current_step_specs) == 2
    assert [spec["patch_spec"]["mode"] for spec in helper.current_step_specs] == [
        "project_out",
        "swap_components",
    ]
    assert all(spec["patch_spec"]["token_span"] == [0, 4] for spec in helper.current_step_specs)
