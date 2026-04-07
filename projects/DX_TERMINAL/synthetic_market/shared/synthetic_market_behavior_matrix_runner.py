from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gc
import pyarrow as pa
import pyarrow.parquet as pq

from pipelines.interp.modal_vllm_engine import _build_chat_template_kwargs
from pipelines.interp.tool_schemas import resolve_tool_schema_mode
from projects.DX_TERMINAL.synthetic_market.shared.patch_basis import load_phase17_activation_patch_basis
from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_battery import SyntheticMarketBehaviorPlanItem
from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_behavior_runner import (
    SyntheticMarketBehaviorConfig,
    _build_generation_config,
    _cleanup_cuda_memory,
    _chunk_list,
    _compute_donor_mean_by_layer,
    _decode_first_token,
    _destroy_llm,
    _extract_first_tool_call_fields,
    _flush_table,
    _prepare_behavior_rows,
    _run_generation_batch,
)
from projects.DX_TERMINAL.synthetic_market.shared.synthetic_market_patching_runner import _build_patch_spec
from pipelines.interp.modal_vllm_engine import (
    VLLMCaptureConfig,
    _create_llm,
    _init_activation_patching_on_model,
    _register_activation_patch_basis_on_model,
)


@dataclass(slots=True)
class SyntheticMarketBehaviorMatrixConfig:
    base_config: SyntheticMarketBehaviorConfig
    cells: tuple[SyntheticMarketBehaviorPlanItem, ...]
    output_dir: Path


def _requires_eager_runtime(
    base_config: SyntheticMarketBehaviorConfig,
    cell: SyntheticMarketBehaviorPlanItem,
) -> bool:
    if bool(base_config.enforce_eager):
        return True
    if not bool(cell.config.patch_enabled):
        return False
    return str(cell.config.patch_mode).strip() not in {"", "none", "project_out"}


def _prep_group_key(cell: SyntheticMarketBehaviorPlanItem) -> tuple[Any, ...]:
    cfg = cell.config
    return (
        str(cfg.phase_name),
        str(cfg.context_variant),
        str(cfg.order_mode),
        str(cfg.selection_strategy),
        cfg.limit,
        tuple(str(item) for item in cfg.family_allowlist),
        str(cfg.pair_metric),
        str(cfg.pair_mode),
        float(cfg.min_pair_gap),
    )


def _cell_component_count(cell: SyntheticMarketBehaviorPlanItem) -> int:
    if cell.config.component_indices_by_layer:
        highest = 0
        for indices in cell.config.component_indices_by_layer.values():
            if indices:
                highest = max(highest, max(int(index) for index in indices) + 1)
        if highest > 0:
            return highest
    return max(1, int(cell.config.components_per_layer))


def _build_union_basis_payload(cells: list[SyntheticMarketBehaviorPlanItem], base_config: SyntheticMarketBehaviorConfig) -> dict[int, dict[str, Any]]:
    patch_cells = [cell for cell in cells if cell.config.patch_enabled]
    if not patch_cells:
        return {}
    layers = sorted({int(layer) for cell in patch_cells for layer in cell.config.target_layers})
    components_per_layer = max(_cell_component_count(cell) for cell in patch_cells)
    basis = load_phase17_activation_patch_basis(
        basis_npz_path=base_config.basis_npz_path,
        results_json_path=base_config.basis_results_path,
        state_key=base_config.basis_state_key,
        layers=tuple(layers),
        components_per_layer=int(components_per_layer),
    )
    return basis.to_payload()


def _build_donor_mean_cache(
    *,
    llm_capture: Any,
    tokenizer: Any,
    base_config: SyntheticMarketBehaviorConfig,
    prepared_rows_by_group: dict[tuple[Any, ...], list[dict[str, Any]]],
    cells_by_group: dict[tuple[Any, ...], list[SyntheticMarketBehaviorPlanItem]],
) -> dict[int, dict[int, Any]]:
    donor_target_layers = sorted(
        {
            int(layer)
            for group_key, cells in cells_by_group.items()
            for cell in cells
            if cell.config.patch_mode in {"swap_mean", "swap_components"}
            for layer in cell.config.target_layers
        }
    )
    if not donor_target_layers:
        return {}

    donor_mean_cache: dict[int, dict[int, Any]] = {}
    for group_key, prepared_rows in prepared_rows_by_group.items():
        group_cells = cells_by_group[group_key]
        if not any(cell.config.patch_mode in {"swap_mean", "swap_components"} for cell in group_cells):
            continue
        for prepared in prepared_rows:
            source_log_id = prepared["source_log_id"]
            source_row_messages = prepared["source_row_messages"]
            source_market_span = prepared["source_market_span"]
            if source_log_id is None or source_row_messages is None or source_market_span is None:
                continue
            if int(source_log_id) in donor_mean_cache:
                continue
            donor_mean_cache[int(source_log_id)] = _compute_donor_mean_by_layer(
                llm_capture=llm_capture,
                tokenizer=tokenizer,
                messages=source_row_messages,
                capture_cfg=VLLMCaptureConfig(
                    output_dir=base_config.output_dir / "_tmp_capture",
                    model_id=base_config.model_id,
                    capture_router=False,
                    capture_residual=True,
                    add_generation_prompt=bool(base_config.add_generation_prompt),
                    tensor_parallel_size=int(base_config.tensor_parallel_size),
                    gpu_memory_utilization=float(base_config.gpu_memory_utilization),
                    enable_prefix_caching=False,
                ),
                source_log_id=int(source_log_id),
                market_span=source_market_span,
                target_layers=tuple(donor_target_layers),
            )
    return donor_mean_cache


def run_synthetic_market_behavior_matrix(config: SyntheticMarketBehaviorMatrixConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer

    if not config.cells:
        raise ValueError("Matrix config must include at least one cell")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = config.output_dir / "metadata.parquet"
    plan_path = config.output_dir / "plan.json"

    tokenizer = AutoTokenizer.from_pretrained(config.base_config.model_id)
    tools = resolve_tool_schema_mode(config.base_config.tool_schema_mode)
    tool_choice = config.base_config.tool_choice.strip() or None
    chat_template_kwargs = _build_chat_template_kwargs(tools=tools, tool_choice=tool_choice)

    cells_by_group: dict[tuple[Any, ...], list[SyntheticMarketBehaviorPlanItem]] = defaultdict(list)
    for cell in config.cells:
        cells_by_group[_prep_group_key(cell)].append(cell)

    prepared_rows_by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    skipped_by_group: dict[tuple[Any, ...], int] = {}
    for group_key, group_cells in cells_by_group.items():
        prep_config = replace(config.base_config, **{
            "phase_name": group_cells[0].config.phase_name,
            "context_variant": group_cells[0].config.context_variant,
            "order_mode": group_cells[0].config.order_mode,
            "selection_strategy": group_cells[0].config.selection_strategy,
            "limit": group_cells[0].config.limit,
            "family_allowlist": group_cells[0].config.family_allowlist,
            "pair_metric": group_cells[0].config.pair_metric,
            "pair_mode": group_cells[0].config.pair_mode,
            "min_pair_gap": group_cells[0].config.min_pair_gap,
            "output_dir": config.output_dir,
        })
        prepared_rows, skipped = _prepare_behavior_rows(
            config=prep_config,
            tokenizer=tokenizer,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        prepared_rows_by_group[group_key] = prepared_rows
        skipped_by_group[group_key] = skipped

    if not any(prepared_rows_by_group.values()):
        result = {"error": "no_valid_examples"}
        (config.output_dir / "results.json").write_text(json.dumps(result, indent=2))
        return result

    donor_mean_cache: dict[int, dict[int, Any]] = {}
    if any(cell.config.patch_mode in {"swap_mean", "swap_components"} for cell in config.cells):
        capture_cfg = VLLMCaptureConfig(
            output_dir=config.output_dir / "_tmp_capture",
            model_id=config.base_config.model_id,
            capture_router=False,
            capture_residual=True,
            add_generation_prompt=bool(config.base_config.add_generation_prompt),
            tensor_parallel_size=int(config.base_config.tensor_parallel_size),
            gpu_memory_utilization=float(config.base_config.gpu_memory_utilization),
            enable_prefix_caching=False,
        )
        llm_capture = _create_llm(capture_cfg)
        try:
            donor_mean_cache = _build_donor_mean_cache(
                llm_capture=llm_capture,
                tokenizer=tokenizer,
                base_config=config.base_config,
                prepared_rows_by_group=prepared_rows_by_group,
                cells_by_group=cells_by_group,
            )
        finally:
            _destroy_llm(llm_capture)
            _cleanup_cuda_memory()

    engine_cells_by_group: dict[bool, dict[tuple[Any, ...], list[SyntheticMarketBehaviorPlanItem]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for group_key, group_cells in cells_by_group.items():
        for cell in group_cells:
            runtime_enforce_eager = _requires_eager_runtime(config.base_config, cell)
            engine_cells_by_group[bool(runtime_enforce_eager)][group_key].append(cell)

    source_behavior_cache: dict[int, dict[str, Any]] = {}
    metadata_rows: list[dict[str, Any]] = []
    processed = 0

    def _fill_source_behavior_cache(*, llm_generate: Any, generate_cfg: Any, grouped_cells: dict[tuple[Any, ...], list[SyntheticMarketBehaviorPlanItem]]) -> None:
        if not any(cell.config.generate_source_behavior for cells in grouped_cells.values() for cell in cells):
            return
        unique_source_requests: list[dict[str, Any]] = []
        seen_source_log_ids: set[int] = set(source_behavior_cache)
        for group_key, prepared_rows in prepared_rows_by_group.items():
            if not any(cell.config.generate_source_behavior for cell in grouped_cells.get(group_key, [])):
                continue
            for prepared in prepared_rows:
                source_log_id = prepared["source_log_id"]
                source_row_messages = prepared["source_row_messages"]
                if source_log_id is None or source_row_messages is None or int(source_log_id) in seen_source_log_ids:
                    continue
                seen_source_log_ids.add(int(source_log_id))
                unique_source_requests.append({"source_log_id": int(source_log_id), "messages": source_row_messages})

        for source_chunk in _chunk_list(unique_source_requests, int(config.base_config.batch_size)):
            requests = [{"messages": item["messages"]} for item in source_chunk]
            outputs = _run_generation_batch(
                llm=llm_generate,
                tokenizer=tokenizer,
                requests=requests,
                config=generate_cfg,
                max_tokens=int(config.base_config.max_tokens),
                temperature=float(config.base_config.temperature),
                top_p=float(config.base_config.top_p),
                top_k=int(config.base_config.top_k),
                tools=tools,
                tool_choice=tool_choice,
                chat_template_kwargs=chat_template_kwargs,
            )
            for item, source_output in zip(source_chunk, outputs, strict=False):
                source_generated_token_ids = [int(token) for token in source_output["generated_token_ids"]]
                source_tool_fields = _extract_first_tool_call_fields(str(source_output["generated_text"]))
                source_behavior_cache[int(item["source_log_id"])] = {
                    "source_generated_text": source_output["generated_text"],
                    "source_generated_token_count": len(source_generated_token_ids),
                    "source_first_generated_token_id": (
                        source_generated_token_ids[0] if source_generated_token_ids else None
                    ),
                    "source_first_generated_token_text": _decode_first_token(tokenizer, source_generated_token_ids),
                    **{f"source_{key}": value for key, value in source_tool_fields.items()},
                }

    for runtime_enforce_eager in sorted(engine_cells_by_group):
        grouped_cells = engine_cells_by_group[runtime_enforce_eager]
        if not grouped_cells:
            continue
        runtime_base_config = replace(
            config.base_config,
            output_dir=config.output_dir,
            enforce_eager=bool(runtime_enforce_eager),
        )
        generate_cfg = _build_generation_config(runtime_base_config)
        llm_generate = _create_llm(generate_cfg)
        basis_payload = _build_union_basis_payload(
            [cell for cells in grouped_cells.values() for cell in cells],
            config.base_config,
        )
        if basis_payload:
            _init_activation_patching_on_model(llm_generate)
            _register_activation_patch_basis_on_model(llm_generate, basis_payload)

        try:
            _fill_source_behavior_cache(
                llm_generate=llm_generate,
                generate_cfg=generate_cfg,
                grouped_cells=grouped_cells,
            )

            for group_key, prepared_rows in prepared_rows_by_group.items():
                group_cells = grouped_cells.get(group_key, [])
                if not group_cells:
                    continue
                request_items: list[dict[str, Any]] = []
                for cell in group_cells:
                    for prepared in prepared_rows:
                        request_items.append({"cell": cell, "prepared": prepared})

                for request_chunk in _chunk_list(request_items, int(config.base_config.batch_size)):
                    batch_requests: list[dict[str, Any]] = []
                    chunk_context: list[dict[str, Any]] = []
                    for item in request_chunk:
                        cell = item["cell"]
                        prepared = item["prepared"]
                        row = prepared["row"]
                        market_span = prepared["market_span"]
                        source_log_id = prepared["source_log_id"]
                        source_row_messages = prepared["source_row_messages"]
                        source_market_span = prepared["source_market_span"]
                        cell_cfg = cell.config
                        source_behavior_fields: dict[str, Any] = {}
                        if cell_cfg.generate_source_behavior and source_log_id is not None:
                            source_behavior_fields = dict(source_behavior_cache.get(int(source_log_id), {}))

                        patch_spec = None
                        if cell_cfg.patch_enabled:
                            if cell_cfg.patch_mode in {"swap_mean", "swap_components"}:
                                if (
                                    not cell_cfg.paired_mode_enabled
                                    or source_row_messages is None
                                    or source_market_span is None
                                    or source_log_id is None
                                ):
                                    raise ValueError(
                                        f"{cell_cfg.patch_mode} matrix cells require paired mode with a valid source row"
                                    )
                                donor_mean_by_layer = donor_mean_cache.get(int(source_log_id))
                                if donor_mean_by_layer is None:
                                    raise RuntimeError(f"Missing donor mean cache for source log_id={source_log_id}")
                                patch_spec = _build_patch_spec(
                                    config=cell_cfg,  # type: ignore[arg-type]
                                    market_span=market_span,
                                    basis_payload=basis_payload,
                                    donor_mean_by_layer=donor_mean_by_layer,
                                ).to_payload()
                            else:
                                patch_spec = _build_patch_spec(
                                    config=cell_cfg,  # type: ignore[arg-type]
                                    market_span=market_span,
                                    basis_payload=basis_payload,
                                ).to_payload()

                        batch_requests.append({"messages": prepared["messages"], "patch_spec": patch_spec})
                        chunk_context.append(
                            {
                                "cell": cell,
                                "row": row,
                                "source_log_id": source_log_id,
                                "source_behavior_fields": source_behavior_fields,
                            }
                        )

                    outputs = _run_generation_batch(
                        llm=llm_generate,
                        tokenizer=tokenizer,
                        requests=batch_requests,
                        config=generate_cfg,
                        max_tokens=int(config.base_config.max_tokens),
                        temperature=float(config.base_config.temperature),
                        top_p=float(config.base_config.top_p),
                        top_k=int(config.base_config.top_k),
                        tools=tools,
                        tool_choice=tool_choice,
                        chat_template_kwargs=chat_template_kwargs,
                    )

                    for context, output in zip(chunk_context, outputs, strict=False):
                        cell = context["cell"]
                        row = context["row"]
                        cell_cfg = cell.config
                        generated_token_ids = [int(token) for token in output["generated_token_ids"]]
                        tool_fields = _extract_first_tool_call_fields(str(output["generated_text"]))
                        metadata_rows.append(
                            {
                                "matrix_cell_id": cell.run_name,
                                "matrix_sweep_kind": cell.sweep_kind,
                                "matrix_sweep_value": cell.sweep_value,
                                "matrix_cell_description": cell.description,
                                "runtime_enforce_eager": bool(runtime_enforce_eager),
                                "log_id": int(row["log_id"]),
                                "phase_name": row["phase_name"],
                                "example_id": row["example_id"],
                                "family": row["family"],
                                "family_variant": row["family_variant"],
                                "context_variant": row["context_variant"],
                                "roster_key": row.get("roster_key") or "",
                                "pair_metric_name": row.get("pair_metric_name") or None,
                                "pair_mode": row.get("pair_mode") or None,
                                "pair_id": row.get("pair_id"),
                                "pair_metric_value": (
                                    float(row.get("base_pair_metric_value", row.get("pair_metric_value")))
                                    if row.get("base_pair_metric_value", row.get("pair_metric_value")) is not None
                                    else None
                                ),
                                "source_pair_metric_value": (
                                    float(row.get("source_pair_metric_value"))
                                    if row.get("source_pair_metric_value") is not None
                                    else None
                                ),
                                "pair_metric_gap": (
                                    float(row.get("pair_metric_gap")) if row.get("pair_metric_gap") is not None else None
                                ),
                                "source_log_id": context["source_log_id"],
                                "source_example_id": row.get("source_example_id"),
                                "source_family_variant": row.get("source_family_variant"),
                                "source_roster_key": row.get("source_roster_key"),
                                "capture_timestamp": datetime.now(UTC).isoformat(),
                                "seq_len": int(len(output["input_ids"])),
                                "max_tokens": int(config.base_config.max_tokens),
                                "temperature": float(config.base_config.temperature),
                                "top_p": float(config.base_config.top_p),
                                "top_k": int(config.base_config.top_k),
                                "tool_schema_mode": config.base_config.tool_schema_mode,
                                "tool_choice": config.base_config.tool_choice,
                                "tool_count": len(tools) if tools is not None else 0,
                                "patch_mode": cell_cfg.patch_mode,
                                "target_layers": ",".join(str(int(layer)) for layer in cell_cfg.target_layers),
                                "components_per_layer": int(cell_cfg.components_per_layer),
                                "component_indices_by_layer_json": json.dumps(
                                    {
                                        str(layer): [int(index) for index in indices]
                                        for layer, indices in cell_cfg.component_indices_by_layer.items()
                                    }
                                ),
                                "direction_name": cell_cfg.direction_name,
                                "selection_strategy": cell_cfg.selection_strategy,
                                "basis_state_key": cell_cfg.basis_state_key,
                                "strength": float(cell_cfg.strength),
                                "random_seed": int(cell_cfg.random_seed),
                                "generated_token_ids_json": json.dumps(generated_token_ids),
                                "generated_token_count": len(generated_token_ids),
                                "first_generated_token_id": generated_token_ids[0] if generated_token_ids else None,
                                "first_generated_token_text": _decode_first_token(tokenizer, generated_token_ids),
                                "generated_text": output["generated_text"],
                                "finish_reason": output["finish_reason"],
                                "request_id": output.get("request_id") or None,
                                **tool_fields,
                                **context["source_behavior_fields"],
                                "patch_stats_json": json.dumps(output["patch_stats"]),
                                "all_patch_stats_json": json.dumps(output.get("all_patch_stats", {})),
                            }
                        )
                        processed += 1
        finally:
            _destroy_llm(llm_generate)
            _cleanup_cuda_memory()

    _flush_table(metadata_path, metadata_rows)
    plan_payload = {
        "count": len(config.cells),
        "runs": [cell.to_dict() for cell in config.cells],
    }
    plan_path.write_text(json.dumps(plan_payload, indent=2))

    rows_by_cell: dict[str, int] = defaultdict(int)
    for row in metadata_rows:
        rows_by_cell[str(row["matrix_cell_id"])] += 1

    result = {
        "phase_name": config.base_config.phase_name,
        "processed": processed,
        "output_dir": str(config.output_dir),
        "cell_count": len(config.cells),
        "rows_per_cell": dict(sorted(rows_by_cell.items())),
        "counts_by_sweep_kind": {
            sweep_kind: sum(1 for cell in config.cells if cell.sweep_kind == sweep_kind)
            for sweep_kind in sorted({cell.sweep_kind for cell in config.cells})
        },
        "pair_modes": sorted({str(cell.config.pair_mode) for cell in config.cells if cell.config.pair_mode}),
        "runtime_groups": {
            ("eager" if runtime_enforce_eager else "compiled"): sum(
                len(cells) for cells in grouped_cells.values()
            )
            for runtime_enforce_eager, grouped_cells in engine_cells_by_group.items()
        },
        "skipped_by_group": {json.dumps(list(key)): int(value) for key, value in skipped_by_group.items()},
        "batch_size": int(config.base_config.batch_size),
        "max_tokens": int(config.base_config.max_tokens),
    }
    (config.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    return result
