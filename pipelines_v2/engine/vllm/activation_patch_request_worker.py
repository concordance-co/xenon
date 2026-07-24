"""Request-scoped activation patch worker for pipelines_v2."""

from __future__ import annotations

import copy
import os
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

from pipelines_v2.engine.vllm.patching.base import debug_log


_VLLM_USE_V2_MODEL_RUNNER = "VLLM_USE_V2_MODEL_RUNNER"


def force_v1_model_runner_for_activation_patching() -> None:
    """Keep activation patching on the model-runner API it instruments.

    vLLM 0.25 enables Model Runner V2 for dense generation models by
    default. Xenon's request-scoped patch state is integrated with the
    V1 ``GPUModelRunner`` lifecycle, so intervention runtimes must opt out
    before vLLM constructs its engine, scheduler, and workers.
    """

    os.environ[_VLLM_USE_V2_MODEL_RUNNER] = "0"


def compiled_operator_hint_from_config(vllm_config: Any) -> str:
    additional_config = getattr(vllm_config, "additional_config", None)
    config_hint = (
        dict(additional_config).get("xenon_activation_patch_compiled_operator")
        if isinstance(additional_config, dict)
        else None
    )
    return str(
        config_hint or os.getenv("XENON_ACTIVATION_PATCH_COMPILED_OPERATOR", "") or ""
    ).strip()


class ActivationPatchRequestHelper:
    def __init__(self) -> None:
        self.req_id_to_patch_spec: dict[str, dict[str, Any]] = {}
        self.current_step_specs: list[dict[str, Any]] = []

    def _lookup_patch_spec(self, req_id: str) -> dict[str, Any] | None:
        direct = self.req_id_to_patch_spec.get(req_id)
        if direct is not None:
            return dict(direct)
        req_aliases = {req_id}
        req_aliases.add(req_id.split("-", 1)[0])
        req_aliases.add(req_id.rsplit("-", 1)[0])
        for candidate_id, payload in self.req_id_to_patch_spec.items():
            candidate_text = str(candidate_id)
            candidate_aliases = {
                candidate_text,
                candidate_text.split("-", 1)[0],
                candidate_text.rsplit("-", 1)[0],
            }
            if req_aliases & candidate_aliases:
                return dict(payload)
        return None

    def process_new_reqs(self, new_reqs: list[NewRequestData]) -> None:
        for new_req in new_reqs:
            extra_args = getattr(new_req.sampling_params, "extra_args", None)
            if not isinstance(extra_args, dict):
                continue
            payload = extra_args.get("activation_patch_spec")
            if isinstance(payload, dict):
                self.req_id_to_patch_spec[str(new_req.req_id)] = dict(payload)
                debug_log(
                    "request_patch_registered",
                    req_id=str(new_req.req_id),
                    operator=payload.get("operator"),
                    layers=payload.get("target_layers") or payload.get("write_layers"),
                    target_positions=(
                        len(payload.get("target_positions", ()))
                        or len(payload.get("query_positions", ()))
                    ),
                )

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
            payload = self._lookup_patch_spec(req_id)
            scheduled_tokens = int(num_scheduled_tokens[i])
            query_end = query_start + scheduled_tokens
            if payload is None or scheduled_tokens <= 0:
                query_start = query_end
                continue

            computed_before = int(num_computed_tokens_cpu[i])
            prefill_len = int(num_prompt_tokens[i])
            operator = str(payload.get("operator") or "interchange")
            target_policy = dict(payload.get("target_policy") or {"kind": "static"})
            target_policy_kind = str(target_policy.get("kind") or "static")

            if target_policy_kind == "every_token":
                scheduled_abs_start = computed_before
                scheduled_abs_end = computed_before + scheduled_tokens
                selected_abs_start = scheduled_abs_start
                selected_abs_end = scheduled_abs_end
                if not bool(target_policy.get("include_prompt", True)):
                    selected_abs_start = max(selected_abs_start, prefill_len)
                if not bool(target_policy.get("include_decode", False)):
                    selected_abs_end = min(selected_abs_end, prefill_len)
                selected_abs_start = max(selected_abs_start, scheduled_abs_start)
                selected_abs_end = min(selected_abs_end, scheduled_abs_end)
                if selected_abs_end <= selected_abs_start:
                    query_start = query_end
                    continue

                query_span_start = query_start + (selected_abs_start - scheduled_abs_start)
                query_span_end = query_span_start + (selected_abs_end - selected_abs_start)
                prompt_count = max(
                    0,
                    min(selected_abs_end, prefill_len) - min(max(selected_abs_start, 0), prefill_len),
                )
                token_count = int(selected_abs_end - selected_abs_start)
                local_payload = copy.deepcopy(payload)
                local_payload["query_span"] = [int(query_span_start), int(query_span_end)]
                local_payload["covered_abs_spans"] = [[int(selected_abs_start), int(selected_abs_end)]]
                local_payload["phase_counts"] = {
                    "prompt": int(prompt_count),
                    "decode": int(token_count - prompt_count),
                }
                local_payload["rowwise"] = True
                local_payload["target_policy"] = target_policy
                local_payload.pop("target_positions", None)
                local_payload.pop("donor_positions", None)
                local_payload.pop("target_read_positions", None)
                step_specs.append(
                    {
                        "req_id": req_id,
                        "patch_spec": local_payload,
                        "chunk_abs_span": [int(selected_abs_start), int(selected_abs_end)],
                        "query_span": [int(query_span_start), int(query_span_end)],
                    }
                )
                query_start = query_end
                continue

            prefill_chunk_len = max(
                0,
                min(scheduled_tokens, prefill_len - computed_before),
            )
            if prefill_chunk_len <= 0:
                query_start = query_end
                continue

            chunk_abs_start = computed_before
            chunk_abs_end = computed_before + prefill_chunk_len
            target_positions = [int(pos) for pos in payload.get("target_positions", ())]
            donor_positions = [int(pos) for pos in payload.get("donor_positions", ())]
            target_read_positions = [int(pos) for pos in payload.get("target_read_positions", ())]
            kept_query_positions: list[int] = []
            kept_donor_positions: list[int] = []
            kept_target_read_positions: list[int] = []
            covered_abs_positions: list[int] = []
            if operator == "interchange":
                for target_pos, donor_pos in zip(target_positions, donor_positions, strict=False):
                    if chunk_abs_start <= int(target_pos) < chunk_abs_end:
                        covered_abs_positions.append(int(target_pos))
                        kept_query_positions.append(int(query_start + (int(target_pos) - chunk_abs_start)))
                        kept_donor_positions.append(int(donor_pos))
            elif operator == "residual_path":
                if target_read_positions:
                    for target_pos, donor_pos, target_read_pos in zip(
                        target_positions,
                        donor_positions,
                        target_read_positions,
                        strict=False,
                    ):
                        if chunk_abs_start <= int(target_pos) < chunk_abs_end:
                            covered_abs_positions.append(int(target_pos))
                            kept_query_positions.append(int(query_start + (int(target_pos) - chunk_abs_start)))
                            kept_donor_positions.append(int(donor_pos))
                            kept_target_read_positions.append(int(target_read_pos))
                else:
                    for target_pos, donor_pos in zip(target_positions, donor_positions, strict=False):
                        if chunk_abs_start <= int(target_pos) < chunk_abs_end:
                            covered_abs_positions.append(int(target_pos))
                            kept_query_positions.append(int(query_start + (int(target_pos) - chunk_abs_start)))
                            kept_donor_positions.append(int(donor_pos))
            else:
                for target_pos in target_positions:
                    if chunk_abs_start <= int(target_pos) < chunk_abs_end:
                        covered_abs_positions.append(int(target_pos))
                        kept_query_positions.append(int(query_start + (int(target_pos) - chunk_abs_start)))
            if not kept_query_positions:
                query_start = query_end
                continue

            local_payload = copy.deepcopy(payload)
            local_payload["query_positions"] = kept_query_positions
            local_payload["target_abs_positions"] = [int(pos) for pos in target_positions]
            local_payload["covered_abs_positions"] = [int(pos) for pos in covered_abs_positions]
            local_payload["target_policy"] = target_policy
            if operator in {"interchange", "residual_path"}:
                local_payload["donor_positions"] = kept_donor_positions
                local_payload["target_read_positions"] = kept_target_read_positions
            else:
                local_payload.pop("donor_positions", None)
                local_payload.pop("target_read_positions", None)
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
        if step_specs:
            debug_log(
                "prepared_batch_specs",
                count=len(step_specs),
                req_ids=[str(item.get("req_id")) for item in step_specs],
            )

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
        from vllm.model_executor.model_loader import get_model_architecture

        from pipelines_v2.engine.vllm.activation_patch_core import (
            init_activation_patching,
            install_activation_patch_model_init_hook,
            restore_activation_patch_model_init_hook,
        )

        debug_env = str(os.getenv("XENON_ACTIVATION_PATCH_DEBUG", "") or "").strip()
        compiled_operator_hint = compiled_operator_hint_from_config(self.vllm_config)
        if debug_env:
            print(f"[activation-patch] debug_env={debug_env}")
        model_cls, model_arch = get_model_architecture(self.model_config)
        install_activation_patch_model_init_hook(model_cls)
        print(
            "[activation-patch] installed model init hook "
            f"architecture={model_arch} class={model_cls.__module__}.{model_cls.__name__}"
        )
        try:
            # Delegate the complete loading lifecycle to the installed vLLM
            # version. The init hook makes the patch op visible to compilation
            # without copying vLLM's fast-changing ``load_model`` implementation.
            super().load_model(load_dummy_weights=load_dummy_weights)
        finally:
            restore_activation_patch_model_init_hook(model_cls)

        model = self.get_model()
        init_activation_patching(model)
        model._v2_activation_patch_force_custom_op_presence = not bool(
            self.vllm_config.model_config.enforce_eager
        )
        model._v2_activation_patch_compiled_operator_hint = compiled_operator_hint
        print(
            "[activation-patch] worker initialized "
            f"model={self.model_config.model} "
            f"enforce_eager={self.vllm_config.model_config.enforce_eager} "
            f"compiled_operator_hint={compiled_operator_hint or '<none>'}"
        )

    def _update_states(self, scheduler_output: SchedulerOutput) -> Any:
        deferred_state_corrections = super()._update_states(scheduler_output)
        self.activation_patch_request_helper.process_new_reqs(
            scheduler_output.scheduled_new_reqs
        )
        finished_req_ids = getattr(scheduler_output, "finished_req_ids", None)
        self.activation_patch_request_helper.cleanup_finished(finished_req_ids)
        return deferred_state_corrections

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
        from pipelines_v2.engine.vllm.activation_patch_core import (
            clear_batch_patch_specs,
            harvest_batch_patch_stats,
        )

        try:
            return super().execute_model(scheduler_output, intermediate_tensors)
        finally:
            harvest_batch_patch_stats(
                self.model,
                list(self.activation_patch_request_helper.current_step_specs),
            )
            clear_batch_patch_specs(self.model)


class ActivationPatchGPUWorker(gpu_worker.Worker):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        force_v1_model_runner_for_activation_patching()
        gpu_model_runner.GPUModelRunner = ActivationPatchGPUModelRunner
        super().__init__(*args, **kwargs)
        if bool(getattr(self, "use_v2_model_runner", False)):
            raise RuntimeError(
                "Activation patching requires vLLM Model Runner V1; "
                "set VLLM_USE_V2_MODEL_RUNNER=0 before constructing LLM."
            )


__all__ = [
    "ActivationPatchGPUModelRunner",
    "ActivationPatchGPUWorker",
    "ActivationPatchRequestHelper",
    "force_v1_model_runner_for_activation_patching",
]
