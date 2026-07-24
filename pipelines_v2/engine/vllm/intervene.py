"""vLLM patched-generation execution."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from pipelines_v2.data.datasets import Example
from pipelines_v2.operations.interventions import (
    AddDirectionPatch,
    InterchangePatch,
    PatchedGenerationSpec,
    ResidualPathPatch,
    SwapComponentsPatch,
    SwapMeanPatch,
)
from pipelines_v2.operations.interventions.runtime import (
    load_activation_bank_source,
    load_centroid_source,
    load_direction_source,
    load_path_mask_source,
    load_subspace_source,
    partition_cases_by_activation_bank,
    resolve_patched_generation_cases,
    resolve_patched_generation_targets,
)

from .activation_patch_core import (
    collect_patch_stats,
    register_activation_patch_bank,
    register_activation_patch_centroids,
    register_activation_patch_directions,
    register_activation_patch_subspace,
)
from .capture import _apply_to_model, _build_reasoning_parser, _prompt_token_ids
from .intervention_build import (
    build_activation_bank_runtime_payload,
    build_centroid_runtime_payload,
    build_direction_runtime_payload,
    build_llm_kwargs,
    build_subspace_runtime_payload,
    paired_request_payload,
    patched_sampling_params,
    unpaired_request_payload,
)
from .intervention_output import (
    iter_batches,
    missing_patch_stats,
    normalize_generation_output,
    stats_for_request,
)
from .patching.base import debug_log, debug_mode_enabled

if TYPE_CHECKING:
    from pipelines_v2.engine.base import EngineInterventionResult
    from pipelines_v2.engine.vllm.engine import VLLMEngine


_SUBSPACE_PATCH_OPERATORS = frozenset(
    {"project_out", "add_direction", "swap_mean", "swap_components", "random_control"}
)


@dataclass(slots=True)
class VLLMInterventionRuntime:
    """Loaded vLLM intervention runtime reused within one remote execution session."""

    engine: "VLLMEngine"
    llm: Any
    tokenizer: Any
    reasoning_parser_instance: Any | None
    batch_size: int
    session_key: str


def run_vllm_intervention(*, engine: "VLLMEngine", spec: PatchedGenerationSpec) -> "EngineInterventionResult":
    runtime = build_vllm_intervention_runtime(engine=engine, spec=spec)
    return run_vllm_intervention_with_runtime(runtime=runtime, spec=spec)


def build_vllm_intervention_runtime(
    *,
    engine: "VLLMEngine",
    spec: PatchedGenerationSpec,
) -> VLLMInterventionRuntime:
    """Construct one vLLM intervention runtime for a compatible patch family."""

    from .activation_patch_request_worker import (
        force_v1_model_runner_for_activation_patching,
    )

    # vLLM 0.25 defaults dense generation models to Model Runner V2.
    # Xenon's intervention worker instruments the V1 request lifecycle, so
    # select it before importing/constructing the vLLM engine and scheduler.
    force_v1_model_runner_for_activation_patching()

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(engine.resolved_model_path(), trust_remote_code=True)
    llm_kwargs, reasoning_parser = build_llm_kwargs(
        engine,
        compiled_operator_hint=_compiled_operator_hint(spec),
    )
    llm = LLM(**llm_kwargs)
    reasoning_parser_instance = _build_reasoning_parser(
        tokenizer=tokenizer,
        parser_name=reasoning_parser,
        enable_thinking=engine.enable_thinking,
    )
    batch_size = max(1, int(engine.max_num_seqs or 1))
    return VLLMInterventionRuntime(
        engine=engine,
        llm=llm,
        tokenizer=tokenizer,
        reasoning_parser_instance=reasoning_parser_instance,
        batch_size=batch_size,
        session_key=vllm_intervention_session_key(engine=engine, spec=spec),
    )


def run_vllm_intervention_with_runtime(
    *,
    runtime: VLLMInterventionRuntime,
    spec: PatchedGenerationSpec,
) -> "EngineInterventionResult":
    """Run one patch spec against an already-loaded vLLM intervention runtime."""

    if spec.patch.requires_pairing():
        return _run_paired(
            engine=runtime.engine,
            spec=spec,
            llm=runtime.llm,
            tokenizer=runtime.tokenizer,
            reasoning_parser_instance=runtime.reasoning_parser_instance,
            batch_size=runtime.batch_size,
        )
    return _run_unpaired(
        engine=runtime.engine,
        spec=spec,
        llm=runtime.llm,
        tokenizer=runtime.tokenizer,
        reasoning_parser_instance=runtime.reasoning_parser_instance,
        batch_size=runtime.batch_size,
    )


def vllm_intervention_session_key(
    *,
    engine: "VLLMEngine",
    spec: PatchedGenerationSpec,
) -> str:
    """Return a stable key for vLLM runtimes that can safely share one loaded LLM."""

    from pipelines_v2.core.types import stable_hash

    llm_kwargs, reasoning_parser = build_llm_kwargs(
        engine,
        compiled_operator_hint=_compiled_operator_hint(spec),
    )
    return stable_hash(
        {
            "kind": "vllm_intervention_runtime",
            "family": _patch_family(spec),
            "llm_kwargs": llm_kwargs,
            "reasoning_parser": reasoning_parser,
            "engine": engine.identity(),
        }
    )


def _compiled_operator_hint(spec: PatchedGenerationSpec) -> str | None:
    if spec.patch.operator in _SUBSPACE_PATCH_OPERATORS:
        return "subspace"
    return None


def _patch_family(spec: PatchedGenerationSpec) -> str:
    if spec.patch.operator in _SUBSPACE_PATCH_OPERATORS:
        return "subspace"
    if spec.patch.requires_pairing():
        return "paired"
    return str(spec.patch.operator)


def _run_paired(
    *,
    engine: "VLLMEngine",
    spec: PatchedGenerationSpec,
    llm: Any,
    tokenizer: Any,
    reasoning_parser_instance: Any | None,
    batch_size: int,
) -> "EngineInterventionResult":
    from pipelines_v2.engine.base import EngineInterventionResult

    activation_bank = load_activation_bank_source(spec.patch)
    resolved_cases, skipped_cases = resolve_patched_generation_cases(spec)
    resolved_cases, source_skips = partition_cases_by_activation_bank(
        spec=spec,
        activation_bank=activation_bank,
        resolved_cases=resolved_cases,
    )
    skipped_cases.extend(source_skips)

    path_mask_payload = load_path_mask_source(spec.patch) if isinstance(spec.patch, ResidualPathPatch) else None
    _apply_to_model(
        llm,
        partial(
            register_activation_patch_bank,
            bank_payload=build_activation_bank_runtime_payload(
                spec=spec,
                activation_bank=activation_bank,
                resolved_cases=resolved_cases,
            ),
        ),
    )

    rows: list[dict[str, Any]] = [
        {
            "case_key": skipped.get("case_key"),
            "status": "skipped",
            "skip_reason": skipped.get("skip_reason", ""),
            "patch_stats": {},
        }
        for skipped in skipped_cases
    ]

    for batch in iter_batches(resolved_cases, batch_size):
        planned_rows: list[dict[str, Any]] = []
        patched_prompts: list[dict[str, Any]] = []
        patched_params: list[Any] = []

        for item in batch:
            case_key = str(item["case_key"])
            target: Example = item["target"]
            donor: Example = item["donor"]
            tokenized = _prompt_token_ids(
                tokenizer=tokenizer,
                example=target,
                add_generation_prompt=bool(engine.add_generation_prompt),
                require_sections=spec.patch.target_tokens.kind == "section",
                prompt_metadata_builder=spec.prompt_metadata_builder,
                tools=spec.generation.chat_tools,
                tool_choice=spec.generation.tool_choice,
                enable_thinking=engine.enable_thinking,
                chat_template_kwargs=dict(engine.extra.get("chat_template_kwargs") or {}),
            )
            target_positions = spec.patch.target_tokens.resolve(
                len(tokenized["token_ids"]),
                token_sections=tokenized["token_sections"],
            )
            every_token_application = str(getattr(getattr(spec.patch, "application", None), "kind", "static")) == "every_token"
            if not target_positions and not every_token_application:
                rows.append(
                    {
                        "case_key": case_key,
                        "example_key": target.key,
                        "donor_example_key": donor.key,
                        "example": target.to_dict(),
                        "status": "skipped",
                        "skip_reason": "target token selection resolved to no positions",
                        "patch_stats": {},
                    }
                )
                continue

            request_payload = paired_request_payload(
                spec=spec,
                activation_bank=activation_bank,
                path_mask_payload=path_mask_payload,
                target=target,
                donor=donor,
                case_key=case_key,
                tokenized=tokenized,
                target_positions=target_positions,
            )
            if isinstance(request_payload, str):
                rows.append(
                    {
                        "case_key": case_key,
                        "example_key": target.key,
                        "donor_example_key": donor.key,
                        "example": target.to_dict(),
                        "status": "skipped",
                        "skip_reason": request_payload,
                        "patch_stats": {},
                    }
                )
                continue

            planned_rows.append(
                {
                    "case_key": case_key,
                    "target": target,
                    "donor": donor,
                    "target_positions": list(target_positions),
                    **{
                        key: value
                        for key, value in request_payload.items()
                        if key in {"donor_positions", "target_read_positions", "path_edges", "transport"}
                    },
                }
            )
            patched_prompts.append({"prompt_token_ids": tokenized["token_ids"]})
            patched_params.append(
                patched_sampling_params(
                    max_tokens=spec.generation.max_tokens,
                    temperature=float(spec.generation.temperature or 0.0),
                    top_p=float(spec.generation.top_p),
                    top_k=int(spec.generation.top_k),
                    structured_output=spec.generation.structured_output,
                    extra_args={"activation_patch_spec": request_payload},
                )
            )

        if not planned_rows:
            continue

        patched_outputs = llm.generate(prompts=patched_prompts, sampling_params=patched_params)
        if len(patched_outputs) != len(planned_rows):
            raise RuntimeError(
                "vLLM returned a different number of patched request outputs than prompts: "
                f"got {len(patched_outputs)}, expected {len(planned_rows)}"
            )
        batch_stats = _apply_to_model(llm, collect_patch_stats) or {}

        for index, row_plan in enumerate(planned_rows):
            target = row_plan["target"]
            donor = row_plan["donor"]
            patched = normalize_generation_output(
                patched_outputs[index],
                capture_reasoning=bool(spec.generation.capture_reasoning),
                reasoning_parser=reasoning_parser_instance,
            )
            request_id = str(patched.get("request_id") or "")
            patch_stats = stats_for_request(batch_stats, request_id)
            if not patch_stats:
                patch_stats = missing_patch_stats(
                    spec=spec,
                    target_positions=row_plan["target_positions"],
                    extra={
                        "case_key": row_plan["case_key"],
                        "donor_example_key": donor.key,
                        "path_edges": row_plan.get("path_edges"),
                        "transport": row_plan.get("transport"),
                    },
                )
            row = {
                "case_key": row_plan["case_key"],
                "example_key": target.key,
                "donor_example_key": donor.key,
                "example": target.to_dict(),
                "status": "ok",
                "skip_reason": "",
                **patched,
                "patch_stats": patch_stats,
                "target_tokens": list(row_plan["target_positions"]),
            }
            if isinstance(spec.patch, InterchangePatch):
                row["donor_tokens"] = list(row_plan["donor_positions"])
            else:
                row["read_tokens"] = list(row_plan["donor_positions"])
                row["target_read_tokens"] = list(row_plan.get("target_read_positions") or ())
            rows.append(row)

    usable_rows = [row for row in rows if str(row.get("status") or "ok") == "ok"]
    return EngineInterventionResult(
        summary={
            "example_count": len(rows),
            "patched_count": len(usable_rows),
            "skipped_count": len(rows) - len(usable_rows),
            "case_count": len(resolved_cases) + len(skipped_cases),
        },
        rows=rows,
        metadata={
            "backend": "vllm",
            "batch_size": batch_size,
            "write_site": spec.patch.write_site.site,
            "write_layers": [int(layer) for layer in spec.patch.write_site.layers],
            "operator": spec.patch.operator,
        },
    )


def _run_unpaired(
    *,
    engine: "VLLMEngine",
    spec: PatchedGenerationSpec,
    llm: Any,
    tokenizer: Any,
    reasoning_parser_instance: Any | None,
    batch_size: int,
) -> "EngineInterventionResult":
    from pipelines_v2.engine.base import EngineInterventionResult

    if spec.patch.uses_subspace() and getattr(spec.patch, "subspace", None) is not None:
        _apply_to_model(
            llm,
            partial(
                register_activation_patch_subspace,
                subspace_payload=build_subspace_runtime_payload(spec.patch, load_subspace_source(spec.patch)),
            ),
        )
    if isinstance(spec.patch, AddDirectionPatch):
        direction_payload = build_direction_runtime_payload(spec.patch, load_direction_source(spec.patch))
        debug_summary = {
            int(layer): {
                "has_raw_vector": "raw_vector" in dict(payload),
                "raw_vector_len": len(dict(payload).get("raw_vector", ())),
                "has_subspace_weights": "subspace_weights" in dict(payload),
            }
            for layer, payload in direction_payload.items()
        }
        if debug_mode_enabled("log", "direction_registration"):
            debug_log(
                "direction_registration",
                layers=sorted(int(layer) for layer in direction_payload),
                summary=debug_summary,
                strength=float(spec.patch.strength),
            )
        _apply_to_model(
            llm,
            partial(
                register_activation_patch_directions,
                direction_payload=direction_payload,
            ),
        )
    if isinstance(spec.patch, (SwapMeanPatch, SwapComponentsPatch)):
        _apply_to_model(
            llm,
            partial(
                register_activation_patch_centroids,
                centroid_payload=build_centroid_runtime_payload(spec.patch, load_centroid_source(spec.patch)),
            ),
        )

    targets = resolve_patched_generation_targets(spec)
    rows: list[dict[str, Any]] = []

    for batch in iter_batches([{"target": target} for target in targets], batch_size):
        planned_rows: list[dict[str, Any]] = []
        patched_prompts: list[dict[str, Any]] = []
        patched_params: list[Any] = []

        for item in batch:
            target: Example = item["target"]
            tokenized = _prompt_token_ids(
                tokenizer=tokenizer,
                example=target,
                add_generation_prompt=bool(engine.add_generation_prompt),
                require_sections=spec.patch.target_tokens.kind == "section",
                prompt_metadata_builder=spec.prompt_metadata_builder,
                tools=spec.generation.chat_tools,
                tool_choice=spec.generation.tool_choice,
                enable_thinking=engine.enable_thinking,
                chat_template_kwargs=dict(engine.extra.get("chat_template_kwargs") or {}),
            )
            target_positions = spec.patch.target_tokens.resolve(
                len(tokenized["token_ids"]),
                token_sections=tokenized["token_sections"],
            )
            if not target_positions:
                rows.append(
                    {
                        "example_key": target.key,
                        "example": target.to_dict(),
                        "status": "skipped",
                        "skip_reason": "target token selection resolved to no positions",
                        "patch_stats": {},
                    }
                )
                continue

            request_payload = unpaired_request_payload(spec=spec, target=target, target_positions=target_positions)
            if isinstance(spec.patch, AddDirectionPatch):
                debug_log(
                    "add_direction_request_payload",
                    example_key=target.key,
                    strength=request_payload.get("strength"),
                    target_layers=request_payload.get("target_layers"),
                    target_policy=request_payload.get("target_policy"),
                )
            planned_rows.append(
                {
                    "target": target,
                    "target_positions": list(target_positions),
                    "request_payload": request_payload,
                }
            )
            patched_prompts.append({"prompt_token_ids": tokenized["token_ids"]})
            patched_params.append(
                patched_sampling_params(
                    max_tokens=spec.generation.max_tokens,
                    temperature=float(spec.generation.temperature or 0.0),
                    top_p=float(spec.generation.top_p),
                    top_k=int(spec.generation.top_k),
                    structured_output=spec.generation.structured_output,
                    extra_args={"activation_patch_spec": request_payload},
                )
            )

        if not planned_rows:
            continue

        patched_outputs = llm.generate(prompts=patched_prompts, sampling_params=patched_params)
        if len(patched_outputs) != len(planned_rows):
            raise RuntimeError(
                "vLLM returned a different number of patched request outputs than prompts: "
                f"got {len(patched_outputs)}, expected {len(planned_rows)}"
            )
        batch_stats = _apply_to_model(llm, collect_patch_stats) or {}

        for index, row_plan in enumerate(planned_rows):
            target = row_plan["target"]
            patched = normalize_generation_output(
                patched_outputs[index],
                capture_reasoning=bool(spec.generation.capture_reasoning),
                reasoning_parser=reasoning_parser_instance,
            )
            request_id = str(patched.get("request_id") or "")
            patch_stats = stats_for_request(batch_stats, request_id)
            if not patch_stats:
                patch_stats = missing_patch_stats(
                    spec=spec,
                    target_positions=row_plan["target_positions"],
                    extra={"example_key": target.key},
                )
            rows.append(
                {
                    "example_key": target.key,
                    "example": target.to_dict(),
                    "status": "ok",
                    "skip_reason": "",
                    **patched,
                    "patch_stats": patch_stats,
                    "target_tokens": list(row_plan["target_positions"]),
                }
            )

    usable_rows = [row for row in rows if str(row.get("status") or "ok") == "ok"]
    return EngineInterventionResult(
        summary={
            "example_count": len(rows),
            "patched_count": len(usable_rows),
            "skipped_count": len(rows) - len(usable_rows),
            "target_count": len(targets),
        },
        rows=rows,
        metadata={
            "backend": "vllm",
            "batch_size": batch_size,
            "write_site": spec.patch.write_site.site,
            "write_layers": [int(layer) for layer in spec.patch.write_site.layers],
            "operator": spec.patch.operator,
        },
    )


__all__ = [
    "VLLMInterventionRuntime",
    "build_vllm_intervention_runtime",
    "run_vllm_intervention",
    "run_vllm_intervention_with_runtime",
    "vllm_intervention_session_key",
]
