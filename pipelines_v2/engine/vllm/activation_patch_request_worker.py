"""Request-scoped eager activation patch worker for pipelines_v2."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

try:
    from vllm.v1.worker import gpu_model_runner, gpu_worker
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
except ModuleNotFoundError:  # pragma: no cover - exercised indirectly in local tests
    class GPUModelRunner:  # type: ignore[no-redef]
        pass

    class _GPUModelRunnerModule:
        GPUModelRunner = GPUModelRunner

    class _GPUWorkerBase:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()

    class _GPUWorkerModule:
        Worker = _GPUWorkerBase

    gpu_model_runner = _GPUModelRunnerModule()
    gpu_worker = _GPUWorkerModule()

if TYPE_CHECKING:
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
else:
    IntermediateTensors = Any
    NewRequestData = Any
    SchedulerOutput = Any


class ActivationPatchRequestHelper:
    def __init__(self) -> None:
        self.req_id_to_patch_spec: dict[str, dict[str, Any]] = {}
        self.current_step_specs: list[dict[str, Any]] = []

    def process_new_reqs(self, new_reqs: list[NewRequestData]) -> None:
        for new_req in new_reqs:
            extra_args = getattr(new_req.sampling_params, "extra_args", None)
            if not isinstance(extra_args, dict):
                continue
            payload = extra_args.get("activation_patch_spec")
            if isinstance(payload, dict):
                self.req_id_to_patch_spec[str(new_req.req_id)] = dict(payload)

    def build_step_specs(
        self,
        *,
        input_batch: Any,
        num_scheduled_tokens: Any,
    ) -> None:
        req_ids = [str(req_id) for req_id in list(getattr(input_batch, "req_ids", ()))]
        if not req_ids:
            self.current_step_specs = []
            return

        num_computed_tokens_cpu = getattr(input_batch, "num_computed_tokens_cpu", None)
        num_prompt_tokens = getattr(input_batch, "num_prompt_tokens", None)
        if (
            num_scheduled_tokens is None
            or num_computed_tokens_cpu is None
            or num_prompt_tokens is None
        ):
            self.current_step_specs = []
            return

        step_specs: list[dict[str, Any]] = []
        query_start = 0
        for i, req_id in enumerate(req_ids):
            payload = self.req_id_to_patch_spec.get(req_id)
            scheduled_tokens = int(num_scheduled_tokens[i])
            query_end = query_start + scheduled_tokens
            if payload is None or scheduled_tokens <= 0:
                query_start = query_end
                continue

            computed_prefill_before = int(num_computed_tokens_cpu[i])
            prefill_len = int(num_prompt_tokens[i])
            prefill_chunk_len = max(
                0,
                min(scheduled_tokens, prefill_len - computed_prefill_before),
            )
            if prefill_chunk_len <= 0:
                query_start = query_end
                continue

            chunk_abs_start = computed_prefill_before
            chunk_abs_end = computed_prefill_before + prefill_chunk_len
            target_positions = [int(pos) for pos in payload.get("target_positions", ())]
            donor_positions = [int(pos) for pos in payload.get("donor_positions", ())]
            kept_query_positions: list[int] = []
            kept_donor_positions: list[int] = []
            for target_pos, donor_pos in zip(target_positions, donor_positions, strict=False):
                if chunk_abs_start <= int(target_pos) < chunk_abs_end:
                    kept_query_positions.append(int(query_start + (int(target_pos) - chunk_abs_start)))
                    kept_donor_positions.append(int(donor_pos))
            if not kept_query_positions:
                query_start = query_end
                continue

            local_payload = copy.deepcopy(payload)
            local_payload["query_positions"] = kept_query_positions
            local_payload["donor_positions"] = kept_donor_positions
            local_payload.pop("target_positions", None)
            step_specs.append(
                {
                    "req_id": req_id,
                    "patch_spec": local_payload,
                    "chunk_abs_span": [int(chunk_abs_start), int(chunk_abs_end)],
                    "query_span": [int(query_start), int(query_end)],
                }
            )
            query_start = query_end

        self.current_step_specs = step_specs

    def cleanup_finished(self, finished_req_ids: list[str] | None) -> None:
        if not finished_req_ids:
            return
        for req_id in finished_req_ids:
            self.req_id_to_patch_spec.pop(str(req_id), None)


class ActivationPatchGPUModelRunner(GPUModelRunner):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.activation_patch_request_helper = ActivationPatchRequestHelper()

    def load_model(self, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        from pipelines_v2.engine.vllm.activation_patch_core import init_activation_patching

        init_activation_patching(self.model)

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_states(scheduler_output)
        self.activation_patch_request_helper.process_new_reqs(
            scheduler_output.scheduled_new_reqs
        )
        finished_req_ids = getattr(scheduler_output, "finished_req_ids", None)
        self.activation_patch_request_helper.cleanup_finished(finished_req_ids)

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: Any,
    ) -> Any:
        from pipelines_v2.engine.vllm.activation_patch_core import set_batch_patch_specs

        prepared = super()._prepare_inputs(scheduler_output, num_scheduled_tokens)
        self.activation_patch_request_helper.build_step_specs(
            input_batch=self.input_batch,
            num_scheduled_tokens=num_scheduled_tokens,
        )
        set_batch_patch_specs(
            self.model,
            list(self.activation_patch_request_helper.current_step_specs),
        )
        return prepared

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> Any:
        from pipelines_v2.engine.vllm.activation_patch_core import clear_batch_patch_specs

        try:
            return super().execute_model(scheduler_output, intermediate_tensors)
        finally:
            clear_batch_patch_specs(self.model)


class ActivationPatchGPUWorker(gpu_worker.Worker):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        gpu_model_runner.GPUModelRunner = ActivationPatchGPUModelRunner
        super().__init__(*args, **kwargs)


__all__ = [
    "ActivationPatchGPUModelRunner",
    "ActivationPatchGPUWorker",
    "ActivationPatchRequestHelper",
]
