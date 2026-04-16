"""vLLM activation patch execution."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Mapping

from pipelines_v2.data.datasets import Example
from pipelines_v2.operations.interventions import ActivationPatchSpec
from pipelines_v2.operations.interventions.runtime import (
    aggregate_patch_rows,
    control_for_name,
    evaluate_patch_row,
    load_residual_source_feature,
    resolve_patch_cases,
)

from .activation_patch_core import collect_patch_stats, register_activation_patch_bank
from .capture import _apply_to_model, _build_reasoning_parser, _generation_result_from_output, _prompt_token_ids

if TYPE_CHECKING:
    from pipelines_v2.engine.base import EngineInterventionResult
    from pipelines_v2.engine.vllm.engine import VLLMEngine


def run_vllm_intervention(*, engine: "VLLMEngine", spec: ActivationPatchSpec) -> "EngineInterventionResult":
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from pipelines_v2.engine.base import EngineInterventionResult

    if not bool(engine.enforce_eager):
        raise RuntimeError("ActivationPatchSpec currently requires VLLMEngine(enforce_eager=True)")

    tokenizer = AutoTokenizer.from_pretrained(engine.model_id, trust_remote_code=True)
    source_payload = load_residual_source_feature(spec)
    resolved_cases, skipped_cases = resolve_patch_cases(spec)
    batch_size = max(1, int(engine.max_num_seqs or 1))

    llm_kwargs: dict[str, Any] = {
        "model": engine.model_id,
        "enforce_eager": bool(engine.enforce_eager),
        "max_num_seqs": int(engine.max_num_seqs or 1),
        "enable_chunked_prefill": bool(engine.enable_chunked_prefill),
        "enable_prefix_caching": bool(engine.enable_prefix_caching),
        "tensor_parallel_size": int(engine.tensor_parallel_size or 1),
        "gpu_memory_utilization": float(engine.gpu_memory_utilization or 0.90),
        "worker_cls": "pipelines_v2.engine.vllm.activation_patch_request_worker.ActivationPatchGPUWorker",
    }
    if engine.max_model_len:
        llm_kwargs["max_model_len"] = int(engine.max_model_len)
    reasoning_parser = (engine.reasoning_parser or "").strip()
    if not reasoning_parser and "qwen3" in str(engine.model_id).lower():
        reasoning_parser = "qwen3"
    if reasoning_parser:
        llm_kwargs["structured_outputs_config"] = {"reasoning_parser": reasoning_parser}

    donor_bank_payload = _build_donor_bank_payload(spec=spec, source_payload=source_payload, resolved_cases=resolved_cases)
    llm = LLM(**llm_kwargs)
    reasoning_parser_instance = _build_reasoning_parser(
        tokenizer=tokenizer,
        parser_name=reasoning_parser,
        enable_thinking=engine.enable_thinking,
    )
    _apply_to_model(llm, partial(register_activation_patch_bank, bank_payload=donor_bank_payload))

    rows: list[dict[str, Any]] = []
    for skipped in skipped_cases:
        rows.append(
            {
                "case_key": skipped.get("case_key"),
                "status": "skipped",
                "skip_reason": skipped.get("skip_reason"),
                "controls": {},
                "patch_stats": {},
            }
        )

    for batch in _iter_batches(resolved_cases, batch_size):
        planned_rows: list[dict[str, Any]] = []
        baseline_prompts: list[dict[str, Any]] = []
        patched_prompts: list[dict[str, Any]] = []
        patched_params: list[Any] = []
        control_batches: dict[str, tuple[list[dict[str, Any]], list[Any]]] = {}

        for item in batch:
            case_key = str(item["case_key"])
            target: Example = item["target"]
            donor: Example = item["donor"]
            controls: dict[str, Example] = dict(item["controls"])
            tokenized = _prompt_token_ids(
                tokenizer=tokenizer,
                example=target,
                add_generation_prompt=bool(engine.add_generation_prompt),
                require_sections=spec.target_tokens.kind == "section",
                prompt_metadata_builder=spec.prompt_metadata_builder,
                enable_thinking=engine.enable_thinking,
            )
            target_positions = spec.target_tokens.resolve(
                len(tokenized["token_ids"]),
                token_sections=tokenized["token_sections"],
            )
            donor_selector = spec.donor_tokens or spec.target_tokens
            donor_positions, patch_skip = _donor_positions(
                source_payload=source_payload,
                donor_key=donor.key,
                donor_selector=donor_selector,
                layers=tuple(int(layer) for layer in spec.write_site.layers),
                expected_count=len(target_positions),
            )
            if patch_skip is not None:
                rows.append(
                    {
                        "case_key": case_key,
                        "example_key": target.key,
                        "donor_example_key": donor.key,
                        "status": "skipped",
                        "skip_reason": patch_skip,
                        "controls": {},
                        "patch_stats": {},
                    }
                )
                continue

            row_plan = {
                "case_key": case_key,
                "target": target,
                "donor": donor,
                "controls": controls,
                "prompt_token_ids": tokenized["token_ids"],
                "target_positions": target_positions,
                "donor_positions": donor_positions,
            }
            planned_rows.append(row_plan)
            baseline_prompts.append({"prompt_token_ids": tokenized["token_ids"]})
            patched_prompts.append({"prompt_token_ids": tokenized["token_ids"]})
            patched_params.append(
                SamplingParams(
                    max_tokens=int(spec.generation.max_tokens),
                    temperature=float(spec.generation.temperature or 0.0),
                    extra_args={
                        "activation_patch_spec": {
                            "target_layers": [int(layer) for layer in spec.write_site.layers],
                            "target_positions": [int(pos) for pos in target_positions],
                            "donor_example_key": donor.key,
                            "donor_positions": [int(pos) for pos in donor_positions],
                            "case_key": case_key,
                            "control_name": "",
                        }
                    },
                )
            )
            for name, control_example in controls.items():
                control_spec = control_for_name(tuple(spec.controls), name)
                control_selector = (
                    control_spec.donor_tokens
                    if control_spec is not None and control_spec.donor_tokens is not None
                    else donor_selector
                )
                control_donor_positions, control_skip = _donor_positions(
                    source_payload=source_payload,
                    donor_key=control_example.key,
                    donor_selector=control_selector,
                    layers=tuple(int(layer) for layer in spec.write_site.layers),
                    expected_count=len(target_positions),
                )
                prompts, params = control_batches.setdefault(name, ([], []))
                prompts.append({"prompt_token_ids": tokenized["token_ids"]})
                params.append(
                    SamplingParams(
                        max_tokens=int(spec.generation.max_tokens),
                        temperature=float(spec.generation.temperature or 0.0),
                        extra_args=(
                            {
                                "activation_patch_spec": {
                                    "target_layers": [int(layer) for layer in spec.write_site.layers],
                                    "target_positions": [int(pos) for pos in target_positions],
                                    "donor_example_key": control_example.key,
                                    "donor_positions": [int(pos) for pos in (control_donor_positions or ())],
                                    "case_key": case_key,
                                    "control_name": name,
                                }
                            }
                            if control_skip is None
                            else None
                        ),
                    )
                )

        if not planned_rows:
            continue

        baseline_outputs = llm.generate(
            prompts=baseline_prompts,
            sampling_params=SamplingParams(
                max_tokens=int(spec.generation.max_tokens),
                temperature=float(spec.generation.temperature or 0.0),
            ),
        )
        patched_outputs = llm.generate(prompts=patched_prompts, sampling_params=patched_params)
        control_outputs: dict[str, list[Any]] = {}
        for name, (prompts, params) in control_batches.items():
            control_outputs[name] = llm.generate(prompts=prompts, sampling_params=params)

        for index, row_plan in enumerate(planned_rows):
            target = row_plan["target"]
            donor = row_plan["donor"]
            baseline = _normalize_generation_output(
                baseline_outputs[index],
                capture_reasoning=bool(spec.generation.capture_reasoning),
                reasoning_parser=reasoning_parser_instance,
            )
            patched = _normalize_generation_output(
                patched_outputs[index],
                capture_reasoning=bool(spec.generation.capture_reasoning),
                reasoning_parser=reasoning_parser_instance,
            )
            request_id = str(patched.get("request_id") or "")
            patch_stats = _apply_to_model(llm, partial(collect_patch_stats, req_id=request_id)) or {}
            controls_payload: dict[str, dict[str, Any]] = {}
            for name in sorted(control_outputs):
                outputs = control_outputs[name]
                if index < len(outputs):
                    controls_payload[name] = _normalize_generation_output(
                        outputs[index],
                        capture_reasoning=bool(spec.generation.capture_reasoning),
                        reasoning_parser=reasoning_parser_instance,
                    )
            evaluation = evaluate_patch_row(
                spec=spec,
                example=target,
                baseline=baseline,
                patched=patched,
                controls=controls_payload,
            )
            rows.append(
                {
                    "case_key": row_plan["case_key"],
                    "example_key": target.key,
                    "donor_example_key": donor.key,
                    "status": "ok",
                    "skip_reason": "",
                    "baseline": baseline,
                    "patched": patched,
                    "controls": controls_payload,
                    "evaluation": evaluation,
                    "patch_stats": patch_stats,
                    "target_tokens": list(row_plan["target_positions"]),
                    "donor_tokens": list(row_plan["donor_positions"]),
                }
            )

    summary = aggregate_patch_rows(rows)
    summary["case_count"] = len(resolved_cases) + len(skipped_cases)
    return EngineInterventionResult(
        summary=summary,
        rows=rows,
        metadata={
            "backend": "vllm",
            "batch_size": batch_size,
            "write_site": spec.write_site.site,
        },
    )


def _build_donor_bank_payload(
    *,
    spec: ActivationPatchSpec,
    source_payload: dict[str, Any],
    resolved_cases: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    donor_keys = {item["donor"].key for item in resolved_cases}
    for item in resolved_cases:
        donor_keys.update(example.key for example in dict(item["controls"]).values())
    layers_payload = source_payload["layers"]
    payload: dict[int, dict[str, Any]] = {}
    for layer in spec.write_site.layers:
        layer_items: dict[str, Any] = {}
        layer_payload = dict(layers_payload[str(int(layer))])
        for donor_key in donor_keys:
            example_payload = dict(layer_payload[donor_key])
            layer_items[str(donor_key)] = {
                "values": example_payload["values"],
                "token_count": len(example_payload.get("values", ())),
            }
        payload[int(layer)] = layer_items
    return payload


def _donor_positions(
    *,
    source_payload: dict[str, Any],
    donor_key: str,
    donor_selector: Any,
    layers: tuple[int, ...],
    expected_count: int,
) -> tuple[list[int] | None, str | None]:
    first_layer_payload = dict(source_payload["layers"][str(int(layers[0]))])
    donor_record = first_layer_payload.get(donor_key)
    if not isinstance(donor_record, Mapping):
        return None, "source_feature is missing donor activation rows"
    donor_values = donor_record.get("values")
    donor_sections = donor_record.get("token_sections")
    donor_positions = donor_selector.resolve(
        len(donor_values),
        token_sections=donor_sections,
    )
    if len(donor_positions) != int(expected_count):
        return donor_positions, "target and donor token selections must have equal length"
    return donor_positions, None


def _normalize_generation_output(
    request_output: Any,
    *,
    capture_reasoning: bool,
    reasoning_parser: Any | None,
) -> dict[str, Any]:
    result = _generation_result_from_output(
        request_output,
        capture_reasoning=capture_reasoning,
        reasoning_parser=reasoning_parser,
    )
    payload = {
        "generated_text": str(result.get("text") or ""),
        "generated_token_ids": list(result.get("generated_token_ids") or ()),
        "finish_reason": str(result.get("finish_reason") or ""),
        "request_id": str(result.get("request_id") or ""),
    }
    if capture_reasoning:
        payload["reasoning_text"] = str(result.get("reasoning_text") or "")
    return payload


def _iter_batches(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


__all__ = ["run_vllm_intervention"]
