from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from vllm.v1.worker import gpu_model_runner, gpu_worker
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


if TYPE_CHECKING:
    from vllm.sequence import IntermediateTensors
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
else:
    IntermediateTensors = Any
    NewRequestData = Any
    SchedulerOutput = Any


class MarketPatchRequestHelper:
    def __init__(self) -> None:
        self.req_id_to_patch_spec: dict[str, dict[str, Any]] = {}
        self.current_step_specs: list[dict[str, Any]] = []

    def _lookup_patch_spec(self, req_id: str) -> dict[str, Any] | None:
        direct = self.req_id_to_patch_spec.get(req_id)
        if direct is not None:
            return direct
        req_base = req_id.rsplit("-", 1)[0]
        for candidate_id, payload in self.req_id_to_patch_spec.items():
            if candidate_id == req_id or candidate_id.rsplit("-", 1)[0] == req_base:
                return payload
        return None

    def process_new_reqs(self, new_reqs: list[NewRequestData]) -> None:
        for new_req in new_reqs:
            extra_args = getattr(new_req.sampling_params, "extra_args", None)
            if not extra_args:
                continue
            payload = extra_args.get("market_patch_spec")
            if not isinstance(payload, dict):
                continue
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
            patch_spec = self._lookup_patch_spec(req_id)
            scheduled_tokens = int(num_scheduled_tokens[i])
            query_end = query_start + scheduled_tokens
            if patch_spec is None or scheduled_tokens <= 0:
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

            target_start, target_end = patch_spec["token_span"]
            chunk_abs_start = computed_prefill_before
            chunk_abs_end = computed_prefill_before + prefill_chunk_len
            overlap_start = max(int(target_start), int(chunk_abs_start))
            overlap_end = min(int(target_end), int(chunk_abs_end))
            if overlap_end <= overlap_start:
                continue

            local_payload = copy.deepcopy(patch_spec)
            local_payload["token_span"] = [
                int(query_start + (overlap_start - chunk_abs_start)),
                int(query_start + (overlap_end - chunk_abs_start)),
            ]
            step_specs.append(
                {
                        "req_id": req_id,
                        "patch_spec": local_payload,
                        "target_span": [int(target_start), int(target_end)],
                        "chunk_abs_span": [int(chunk_abs_start), int(chunk_abs_end)],
                        "overlap_abs_span": [int(overlap_start), int(overlap_end)],
                        "query_span": [int(query_start), int(query_end)],
                        "prefill_chunk_len": int(prefill_chunk_len),
                    }
                )
            query_start = query_end

        self.current_step_specs = step_specs

    def cleanup_finished(self, finished_req_ids: list[str] | None) -> None:
        if not finished_req_ids:
            return
        for req_id in finished_req_ids:
            self.req_id_to_patch_spec.pop(str(req_id), None)


class MarketPatchGPUModelRunner(GPUModelRunner):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.market_patch_request_helper = MarketPatchRequestHelper()

    def load_model(self, load_dummy_weights: bool = False) -> None:
        import torch
        import vllm.v1.worker.gpu_model_runner as gm

        from pipelines.interp.vllm_market_patch import (
            init_market_patching,
            install_qwen3_moe_market_patch_hooks,
        )

        gm.logger.info_once(
            "Starting to load model %s...",
            self.model_config.model,
            scope="global",
        )
        if self.parallel_config.enable_eplb:
            self.eplb_state = gm.EplbState(self.parallel_config, self.device)

        try:
            with gm.DeviceMemoryProfiler() as m:
                time_before_load = gm.time.perf_counter()
                if load_dummy_weights:
                    self.load_config.load_format = "dummy"
                if not self.vllm_config.model_config.enforce_eager:
                    install_qwen3_moe_market_patch_hooks()
                model_loader = gm.get_model_loader(self.load_config)
                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config, model_config=self.model_config
                )
                if self.lora_config:
                    self.model = self.load_lora_model(
                        self.model, self.vllm_config, self.device
                    )
                if hasattr(self, "drafter"):
                    gm.logger.info_once("Loading drafter model...")
                    self.drafter.load_model(self.model)
                    if (
                        hasattr(self.drafter, "model")
                        and gm.is_mixture_of_experts(self.drafter.model)
                        and self.parallel_config.enable_eplb
                    ):
                        assert not self.parallel_config.enable_elastic_ep, (
                            "Elastic EP is not supported with drafter model."
                        )
                        spec_config = self.vllm_config.speculative_config
                        assert spec_config is not None
                        assert spec_config.draft_model_config is not None
                        gm.logger.info_once(
                            "EPLB is enabled for drafter model %s.",
                            spec_config.draft_model_config.model,
                        )
                        if self.eplb_state is None:
                            self.eplb_state = gm.EplbState(
                                self.parallel_config, self.device
                            )
                        self.eplb_state.add_model(
                            self.drafter.model,
                            spec_config.draft_model_config,
                        )
                if self.use_aux_hidden_state_outputs:
                    if not gm.supports_eagle3(self.get_model()):
                        raise RuntimeError(
                            "Model does not support EAGLE3 interface but "
                            "aux_hidden_state_outputs was requested"
                        )
                    aux_layers = self._get_eagle3_aux_layers_from_config()
                    if aux_layers:
                        gm.logger.info(
                            "Using auxiliary layers from speculative config: %s",
                            aux_layers,
                        )
                    else:
                        aux_layers = (
                            self.model.get_eagle3_default_aux_hidden_state_layers()
                        )
                    self.model.set_aux_hidden_state_layers(aux_layers)

                # Install patching while still inside vLLM's model-init context so
                # compile/custom-op discovery can see the modified model graph.
                init_market_patching(self.model)
                self.model._market_patch_force_custom_op_presence = not bool(
                    self.vllm_config.model_config.enforce_eager
                )
                time_after_load = gm.time.perf_counter()

            self.model_memory_usage = m.consumed_memory
        except torch.cuda.OutOfMemoryError as e:
            msg = (
                "Failed to load model - not enough GPU memory. "
                "Try lowering --gpu-memory-utilization to free memory for weights, "
                "increasing --tensor-parallel-size, or using --quantization. "
                "See https://docs.vllm.ai/en/latest/configuration/conserving_memory/ "
                "for more tips."
            )
            gm.logger.error(f"{msg} (original error: {e})")
            raise e

        gm.logger.info_once(
            "Model loading took %s GiB memory and %.6f seconds",
            gm.format_gib(self.model_memory_usage),
            time_after_load - time_before_load,
            scope="local",
        )
        if not load_dummy_weights:
            gm.prepare_communication_buffer_for_model(self.model)
            if (drafter := getattr(self, "drafter", None)) and (
                drafter_model := getattr(drafter, "model", None)
            ):
                gm.prepare_communication_buffer_for_model(drafter_model)
        mm_config = self.model_config.multimodal_config
        self.is_multimodal_pruning_enabled = (
            gm.supports_multimodal_pruning(self.get_model())
            and mm_config is not None
            and mm_config.is_multimodal_pruning_enabled()
        )
        self.requires_sequential_video_encoding = hasattr(
            self.get_model(), "requires_sequential_video_encoding"
        )
        if (
            gm.is_mixture_of_experts(self.model)
            and self.parallel_config.enable_eplb
            and not load_dummy_weights
        ):
            gm.logger.info_once(
                "EPLB is enabled for model %s.", self.model_config.model
            )
            assert self.eplb_state is not None
            self.eplb_state.add_model(
                self.model,
                self.model_config,
            )
            if self.eplb_state.is_async:
                self.eplb_state.start_async_loop()
        if (
            self.vllm_config.compilation_config.mode
            == gm.CompilationMode.STOCK_TORCH_COMPILE
        ):
            backend = self.vllm_config.compilation_config.init_backend(
                self.vllm_config
            )
            from vllm.compilation.counter import compilation_counter

            compilation_counter.stock_torch_compile_count += 1
            self.model.compile(fullgraph=True, backend=backend)
            return
        cudagraph_mode = self.compilation_config.cudagraph_mode
        assert cudagraph_mode is not None
        if (
            cudagraph_mode.has_full_cudagraphs()
            and not self.parallel_config.use_ubatching
        ):
            self.model = gm.CUDAGraphWrapper(
                self.model, self.vllm_config, runtime_mode=gm.CUDAGraphMode.FULL
            )
        elif self.parallel_config.use_ubatching:
            if cudagraph_mode.has_full_cudagraphs():
                self.model = gm.UBatchWrapper(
                    self.model,
                    self.vllm_config,
                    gm.CUDAGraphMode.FULL,
                    self.device,
                )
            else:
                self.model = gm.UBatchWrapper(
                    self.model,
                    self.vllm_config,
                    gm.CUDAGraphMode.NONE,
                    self.device,
                )
        gm.get_offloader().post_init()

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_states(scheduler_output)
        self.market_patch_request_helper.process_new_reqs(
            scheduler_output.scheduled_new_reqs
        )
        finished_req_ids = getattr(scheduler_output, "finished_req_ids", None)
        self.market_patch_request_helper.cleanup_finished(finished_req_ids)

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: Any,
    ) -> Any:
        from pipelines.interp.vllm_market_patch import (
            set_batch_patch_specs,
        )

        prepared = super()._prepare_inputs(scheduler_output, num_scheduled_tokens)
        self.market_patch_request_helper.build_step_specs(
            input_batch=self.input_batch,
            num_scheduled_tokens=num_scheduled_tokens,
        )
        set_batch_patch_specs(
            self.model,
            list(self.market_patch_request_helper.current_step_specs),
        )
        return prepared

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> Any:
        from pipelines.interp.vllm_market_patch import (
            clear_batch_patch_specs,
            harvest_batch_tensor_stats,
        )

        try:
            result = super().execute_model(scheduler_output, intermediate_tensors)
            harvest_batch_tensor_stats(
                self.model,
                list(self.market_patch_request_helper.current_step_specs),
            )
            return result
        finally:
            clear_batch_patch_specs(self.model)


class MarketPatchGPUWorker(gpu_worker.Worker):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        gpu_model_runner.GPUModelRunner = MarketPatchGPUModelRunner
        super().__init__(*args, **kwargs)
