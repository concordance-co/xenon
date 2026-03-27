from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipelines.interp.decision_structure.core import _extract_system_user, _parse_messages
from pipelines.interp.decision_tools import resolve_tool_schema_mode
from pipelines.interp.market_patch_basis import default_phase17_market_patch_basis
from pipelines.interp.synthetic_market_pairing import build_matched_metric_examples
from pipelines.interp.synthetic_market_patching_runner import (
    _build_patch_spec,
    _load_examples,
    _parse_component_indices_spec,
)
from pipelines.interp.synthetic_structure import find_synthetic_section_boundaries
from pipelines.interp.vllm_capture import (
    VLLMCaptureConfig,
    _capture_one_vllm,
    _create_llm,
    _generate_one_vllm,
    _init_market_patching_on_model,
    _register_market_patch_basis_on_model,
)


@dataclass(slots=True)
class SyntheticMarketBehaviorConfig:
    phase_name: str = "phase15_market_basis_discovery_v1"
    output_dir: Path = field(
        default_factory=lambda: Path("data/analysis_results/synthetic_market_behavior/phase18_market_behavior_v1")
    )
    model_id: str = "Qwen/Qwen3-30B-A3B"
    context_variant: str = "market_only"
    order_mode: str = "selection_rank_asc"
    selection_strategy: str = "ordered"
    limit: int | None = None
    family_allowlist: tuple[str, ...] = ()
    pair_metric: str = ""
    pair_mode: str = ""
    min_pair_gap: float = 0.0
    generate_source_behavior: bool = False
    donor_means_path: Path | None = None
    patch_mode: str = ""
    target_layers: tuple[int, ...] = (4,)
    components_per_layer: int = 4
    component_indices_by_layer: dict[int, tuple[int, ...]] = field(default_factory=dict)
    direction_name: str = ""
    strength: float = 1.0
    random_seed: int = 42
    max_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    tool_schema_mode: str = ""
    tool_choice: str = ""
    add_generation_prompt: bool = True
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    basis_npz_path: Path = Path(
        "data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/pca_basis.npz"
    )
    basis_results_path: Path = Path(
        "data/analysis_results/synthetic_market_axis_decomposition/phase17_market_axis_decomposition_v1/results.json"
    )

    @property
    def patch_enabled(self) -> bool:
        return bool(self.patch_mode and self.patch_mode.lower() != "none")

    @property
    def paired_mode_enabled(self) -> bool:
        return bool(self.pair_metric.strip() and self.pair_mode.strip())


def _flush_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _decode_first_token(tokenizer: Any, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    try:
        return str(tokenizer.decode([int(token_ids[0])]))
    except Exception:
        return ""


_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _normalize_tool_call_payload(payload: Any, *, raw_json: str) -> dict[str, Any]:
    if isinstance(payload, list):
        payload = payload[0] if payload else None
    arguments = payload.get("arguments", {}) if isinstance(payload, dict) else {}
    return {
        "has_tool_call": payload is not None,
        "tool_call_parse_ok": isinstance(payload, dict),
        "first_tool_name": payload.get("name") if isinstance(payload, dict) else None,
        "first_tool_token": arguments.get("token") if isinstance(arguments, dict) else None,
        "first_tool_spend_pct": arguments.get("spend_pct") if isinstance(arguments, dict) else None,
        "first_tool_strategy": arguments.get("strategy") if isinstance(arguments, dict) else None,
        "first_tool_content": arguments.get("content") if isinstance(arguments, dict) else None,
        "first_tool_raw_json": raw_json,
    }


def _extract_first_tool_call_fields(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            payload = json.loads(stripped)
            return _normalize_tool_call_payload(payload, raw_json=stripped)
        except json.JSONDecodeError:
            pass

    match = _TOOL_CALL_PATTERN.search(text or "")
    if match is None:
        return {
            "has_tool_call": False,
            "tool_call_parse_ok": False,
            "first_tool_name": None,
            "first_tool_token": None,
            "first_tool_spend_pct": None,
            "first_tool_strategy": None,
            "first_tool_content": None,
            "first_tool_raw_json": None,
        }
    raw_json = match.group(1)
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        return {
            "has_tool_call": True,
            "tool_call_parse_ok": False,
            "first_tool_name": None,
            "first_tool_token": None,
            "first_tool_spend_pct": None,
            "first_tool_strategy": None,
            "first_tool_content": None,
            "first_tool_raw_json": raw_json,
        }
    return _normalize_tool_call_payload(payload, raw_json=raw_json)


def _compute_donor_mean_by_layer(
    *,
    llm_capture: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    capture_cfg: VLLMCaptureConfig,
    source_log_id: int,
    market_span: tuple[int, int],
    target_layers: tuple[int, ...],
) -> dict[int, Any]:
    residual, _, _, _ = _capture_one_vllm(
        llm=llm_capture,
        tokenizer=tokenizer,
        messages=messages,
        config=capture_cfg,
        log_id=f"swap_mean_src_{source_log_id}",
        patch_spec=None,
        skip_residual_save=True,
    )
    if residual is None:
        raise RuntimeError(f"Residual capture returned None for source log_id={source_log_id}")

    residual_np = residual.detach().cpu().numpy() if hasattr(residual, "detach") else residual
    start, end = market_span
    donor_mean_by_layer: dict[int, Any] = {}
    for layer in target_layers:
        donor_mean_by_layer[int(layer)] = residual_np[int(layer), int(start):int(end)].mean(axis=0).astype("float32")
    return donor_mean_by_layer


def _destroy_llm(llm: Any | None) -> None:
    if llm is None:
        return
    try:
        llm_engine = getattr(llm, "llm_engine", None)
        shutdown = getattr(llm_engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception:
        pass


def _prepare_behavior_rows(
    *,
    config: SyntheticMarketBehaviorConfig,
    tokenizer: Any,
    tools: list[dict[str, Any]] | None,
    chat_template_kwargs: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], int]:
    load_limit = None if config.paired_mode_enabled else config.limit
    examples = _load_examples(
        phase_name=config.phase_name,
        context_variant=config.context_variant,
        order_mode=config.order_mode,
        selection_strategy=config.selection_strategy,
        limit=load_limit,
        family_allowlist=config.family_allowlist,
    )
    if config.paired_mode_enabled:
        examples = build_matched_metric_examples(
            examples,
            phase_name=config.phase_name,
            pair_metric=config.pair_metric,
            pair_mode=config.pair_mode,
            min_metric_gap=float(config.min_pair_gap),
            limit=config.limit,
        )

    prepared_rows: list[dict[str, Any]] = []
    skipped = 0
    for row in examples:
        messages = _parse_messages(row["prompt_messages_json"])
        system_user = _extract_system_user(messages)
        if system_user is None:
            skipped += 1
            continue

        system_text, user_text = system_user
        market_span = find_synthetic_section_boundaries(
            tokenizer,
            system_text,
            user_text,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        ).get("market")
        if market_span is None:
            skipped += 1
            continue

        source_log_id = None
        source_row_messages = None
        source_market_span = None
        if row.get("source_prompt_messages_json"):
            source_log_id = int(row["source_log_id"])
            source_row_messages = _parse_messages(row["source_prompt_messages_json"])
            source_system_user = _extract_system_user(source_row_messages)
            if source_system_user is None:
                skipped += 1
                continue
            source_system_text, source_user_text = source_system_user
            source_market_span = find_synthetic_section_boundaries(
                tokenizer,
                source_system_text,
                source_user_text,
                tools=tools,
                chat_template_kwargs=chat_template_kwargs,
            ).get("market")
            if source_market_span is None:
                skipped += 1
                continue

        prepared_rows.append(
            {
                "row": row,
                "messages": messages,
                "market_span": market_span,
                "source_log_id": source_log_id,
                "source_row_messages": source_row_messages,
                "source_market_span": source_market_span,
            }
        )
    return prepared_rows, skipped


def _save_donor_mean_cache(path: Path, donor_mean_cache: dict[int, dict[int, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {}
    for source_log_id, layer_map in donor_mean_cache.items():
        for layer, donor_mean in layer_map.items():
            arrays[f"src_{int(source_log_id)}_layer_{int(layer)}"] = np.asarray(donor_mean, dtype=np.float32)
    np.savez(path, **arrays)


def _load_donor_mean_cache(path: Path) -> dict[int, dict[int, Any]]:
    donor_mean_cache: dict[int, dict[int, Any]] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            match = re.fullmatch(r"src_(\d+)_layer_(\d+)", key)
            if match is None:
                continue
            source_log_id = int(match.group(1))
            layer = int(match.group(2))
            donor_mean_cache.setdefault(source_log_id, {})[layer] = np.asarray(data[key], dtype=np.float32)
    return donor_mean_cache


def prepare_synthetic_market_behavior_donors(config: SyntheticMarketBehaviorConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config.output_dir.mkdir(parents=True, exist_ok=True)
    donor_path = config.donor_means_path or (config.output_dir / "donor_means.npz")

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tools = resolve_tool_schema_mode(config.tool_schema_mode)
    tool_choice = config.tool_choice.strip() or None
    chat_template_kwargs = {"tool_choice": tool_choice} if tool_choice is not None else None

    prepared_rows, skipped = _prepare_behavior_rows(
        config=config,
        tokenizer=tokenizer,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )
    if not prepared_rows:
        result = {"error": "no_valid_examples"}
        (config.output_dir / "donor_results.json").write_text(json.dumps(result, indent=2))
        return result

    capture_cfg = VLLMCaptureConfig(
        output_dir=config.output_dir / "_tmp_capture",
        model_id=config.model_id,
        capture_router=False,
        capture_residual=True,
        add_generation_prompt=bool(config.add_generation_prompt),
        tensor_parallel_size=int(config.tensor_parallel_size),
        gpu_memory_utilization=float(config.gpu_memory_utilization),
        enable_prefix_caching=False,
    )
    donor_mean_cache: dict[int, dict[int, Any]] = {}
    llm_capture = _create_llm(capture_cfg)
    try:
        for prepared in prepared_rows:
            source_log_id = prepared["source_log_id"]
            source_row_messages = prepared["source_row_messages"]
            source_market_span = prepared["source_market_span"]
            if source_log_id is None or source_row_messages is None or source_market_span is None:
                continue
            if source_log_id in donor_mean_cache:
                continue
            donor_mean_cache[source_log_id] = _compute_donor_mean_by_layer(
                llm_capture=llm_capture,
                tokenizer=tokenizer,
                messages=source_row_messages,
                capture_cfg=capture_cfg,
                source_log_id=source_log_id,
                market_span=source_market_span,
                target_layers=config.target_layers,
            )
    finally:
        _destroy_llm(llm_capture)

    _save_donor_mean_cache(donor_path, donor_mean_cache)
    result = {
        "phase_name": config.phase_name,
        "prepared_examples": len(prepared_rows),
        "skipped": skipped,
        "donor_source_count": len(donor_mean_cache),
        "target_layers": [int(layer) for layer in config.target_layers],
        "pair_metric": config.pair_metric,
        "pair_mode": config.pair_mode,
        "donor_means_path": str(donor_path),
    }
    (config.output_dir / "donor_results.json").write_text(json.dumps(result, indent=2))
    return result
    del llm
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def run_synthetic_market_behavior(config: SyntheticMarketBehaviorConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = config.output_dir / "metadata.parquet"

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tools = resolve_tool_schema_mode(config.tool_schema_mode)
    tool_choice = config.tool_choice.strip() or None
    chat_template_kwargs = {"tool_choice": tool_choice} if tool_choice is not None else None

    prepared_rows, skipped = _prepare_behavior_rows(
        config=config,
        tokenizer=tokenizer,
        tools=tools,
        chat_template_kwargs=chat_template_kwargs,
    )

    if not prepared_rows:
        result = {"error": "no_valid_examples"}
        (config.output_dir / "results.json").write_text(json.dumps(result, indent=2))
        return result

    needs_donor_capture = bool(
        config.patch_enabled and config.patch_mode in {"swap_mean", "swap_components"} and config.paired_mode_enabled
    )
    donor_mean_cache: dict[int, dict[int, Any]] = {}
    if config.donor_means_path is not None:
        donor_mean_cache = _load_donor_mean_cache(config.donor_means_path)
    elif needs_donor_capture:
        capture_cfg = VLLMCaptureConfig(
            output_dir=config.output_dir / "_tmp_capture",
            model_id=config.model_id,
            capture_router=False,
            capture_residual=True,
            add_generation_prompt=bool(config.add_generation_prompt),
            tensor_parallel_size=int(config.tensor_parallel_size),
            gpu_memory_utilization=float(config.gpu_memory_utilization),
            enable_prefix_caching=False,
        )
        llm_capture = _create_llm(capture_cfg)
        try:
            for prepared in prepared_rows:
                source_log_id = prepared["source_log_id"]
                source_row_messages = prepared["source_row_messages"]
                source_market_span = prepared["source_market_span"]
                if source_log_id is None or source_row_messages is None or source_market_span is None:
                    continue
                if source_log_id in donor_mean_cache:
                    continue
                donor_mean_cache[source_log_id] = _compute_donor_mean_by_layer(
                    llm_capture=llm_capture,
                    tokenizer=tokenizer,
                    messages=source_row_messages,
                    capture_cfg=capture_cfg,
                    source_log_id=source_log_id,
                    market_span=source_market_span,
                    target_layers=config.target_layers,
                )
        finally:
            _destroy_llm(llm_capture)

    llm_generate = _create_llm(
        VLLMCaptureConfig(
            output_dir=config.output_dir / "_tmp_capture",
            model_id=config.model_id,
            capture_router=False,
            capture_residual=False,
            add_generation_prompt=bool(config.add_generation_prompt),
            tensor_parallel_size=int(config.tensor_parallel_size),
            gpu_memory_utilization=float(config.gpu_memory_utilization),
            enable_prefix_caching=False,
        )
    )

    basis_payload: dict[int, dict[str, Any]] = {}
    if config.patch_enabled:
        basis = default_phase17_market_patch_basis(
            basis_npz_path=config.basis_npz_path,
            results_json_path=config.basis_results_path,
            layers=tuple(int(layer) for layer in config.target_layers),
            components_per_layer=int(config.components_per_layer),
        )
        basis_payload = basis.to_payload()
        _init_market_patching_on_model(llm_generate)
        _register_market_patch_basis_on_model(llm_generate, basis_payload)

    generate_cfg = VLLMCaptureConfig(
        output_dir=config.output_dir / "_tmp_capture",
        model_id=config.model_id,
        capture_router=False,
        capture_residual=False,
        add_generation_prompt=bool(config.add_generation_prompt),
        tensor_parallel_size=int(config.tensor_parallel_size),
        gpu_memory_utilization=float(config.gpu_memory_utilization),
        enable_prefix_caching=False,
    )
    source_behavior_cache: dict[int, dict[str, Any]] = {}

    metadata_rows: list[dict[str, Any]] = []
    processed = 0
    for prepared in prepared_rows:
        row = prepared["row"]
        messages = prepared["messages"]
        market_span = prepared["market_span"]
        source_log_id = prepared["source_log_id"]
        source_row_messages = prepared["source_row_messages"]
        source_market_span = prepared["source_market_span"]

        pair_metric_name = row.get("pair_metric_name")
        pair_metric_value = row.get("base_pair_metric_value", row.get("pair_metric_value"))
        source_pair_metric_value = row.get("source_pair_metric_value")
        pair_metric_gap = row.get("pair_metric_gap")
        source_behavior_fields: dict[str, Any] = {}

        patch_spec = None
        if config.patch_enabled:
            if config.patch_mode in {"swap_mean", "swap_components"}:
                if not config.paired_mode_enabled or source_row_messages is None or source_market_span is None or source_log_id is None:
                    raise ValueError(f"{config.patch_mode} behavior runs require paired mode with a valid source row")
                donor_mean_by_layer = donor_mean_cache.get(source_log_id)
                if donor_mean_by_layer is None:
                    raise RuntimeError(f"Missing donor mean cache for source log_id={source_log_id}")
                patch_spec = _build_patch_spec(
                    config=config,  # type: ignore[arg-type]
                    market_span=market_span,
                    basis_payload=basis_payload,
                    donor_mean_by_layer=donor_mean_by_layer,
                ).to_payload()
            else:
                patch_spec = _build_patch_spec(
                    config=config,  # type: ignore[arg-type]
                    market_span=market_span,
                    basis_payload=basis_payload,
                ).to_payload()

        if config.generate_source_behavior and source_row_messages is not None and source_log_id is not None:
            cached_source = source_behavior_cache.get(source_log_id)
            if cached_source is None:
                source_output = _generate_one_vllm(
                    llm=llm_generate,
                    tokenizer=tokenizer,
                    messages=source_row_messages,
                    config=generate_cfg,
                    max_tokens=int(config.max_tokens),
                    temperature=float(config.temperature),
                    top_p=float(config.top_p),
                    top_k=int(config.top_k),
                    patch_spec=None,
                    tools=tools,
                    tool_choice=tool_choice,
                )
                source_generated_token_ids = [int(token) for token in source_output["generated_token_ids"]]
                source_tool_fields = _extract_first_tool_call_fields(str(source_output["generated_text"]))
                cached_source = {
                    "source_generated_text": source_output["generated_text"],
                    "source_generated_token_count": len(source_generated_token_ids),
                    "source_first_generated_token_id": source_generated_token_ids[0] if source_generated_token_ids else None,
                    "source_first_generated_token_text": _decode_first_token(tokenizer, source_generated_token_ids),
                    **{f"source_{key}": value for key, value in source_tool_fields.items()},
                }
                source_behavior_cache[source_log_id] = cached_source
            source_behavior_fields = dict(cached_source)

        output = _generate_one_vllm(
            llm=llm_generate,
            tokenizer=tokenizer,
            messages=messages,
            config=generate_cfg,
            max_tokens=int(config.max_tokens),
            temperature=float(config.temperature),
            top_p=float(config.top_p),
            top_k=int(config.top_k),
            patch_spec=patch_spec,
            tools=tools,
            tool_choice=tool_choice,
        )
        generated_token_ids = [int(token) for token in output["generated_token_ids"]]
        tool_fields = _extract_first_tool_call_fields(str(output["generated_text"]))
        metadata_rows.append(
            {
                "log_id": int(row["log_id"]),
                "phase_name": row["phase_name"],
                "example_id": row["example_id"],
                "family": row["family"],
                "family_variant": row["family_variant"],
                "context_variant": row["context_variant"],
                "roster_key": row.get("roster_key") or "",
                "pair_metric_name": pair_metric_name or None,
                "pair_mode": row.get("pair_mode") or None,
                "pair_id": row.get("pair_id"),
                "pair_metric_value": float(pair_metric_value) if pair_metric_value is not None else None,
                "source_pair_metric_value": float(source_pair_metric_value) if source_pair_metric_value is not None else None,
                "pair_metric_gap": float(pair_metric_gap) if pair_metric_gap is not None else None,
                "source_log_id": source_log_id,
                "source_example_id": row.get("source_example_id"),
                "source_family_variant": row.get("source_family_variant"),
                "source_roster_key": row.get("source_roster_key"),
                "capture_timestamp": datetime.now(UTC).isoformat(),
                "seq_len": int(len(output["input_ids"])),
                "max_tokens": int(config.max_tokens),
                "temperature": float(config.temperature),
                "top_p": float(config.top_p),
                "top_k": int(config.top_k),
                "tool_schema_mode": config.tool_schema_mode,
                "tool_choice": config.tool_choice,
                "tool_count": len(tools) if tools is not None else 0,
                "patch_mode": config.patch_mode,
                "target_layers": ",".join(str(int(layer)) for layer in config.target_layers),
                "components_per_layer": int(config.components_per_layer),
                "component_indices_by_layer_json": json.dumps(
                    {
                        str(layer): [int(index) for index in indices]
                        for layer, indices in config.component_indices_by_layer.items()
                    }
                ),
                "direction_name": config.direction_name,
                "selection_strategy": config.selection_strategy,
                "strength": float(config.strength),
                "generated_token_ids_json": json.dumps(generated_token_ids),
                "generated_token_count": len(generated_token_ids),
                "first_generated_token_id": generated_token_ids[0] if generated_token_ids else None,
                "first_generated_token_text": _decode_first_token(tokenizer, generated_token_ids),
                "generated_text": output["generated_text"],
                "finish_reason": output["finish_reason"],
                **tool_fields,
                **source_behavior_fields,
                "patch_stats_json": json.dumps(output["patch_stats"]),
            }
        )
        processed += 1

    _flush_table(metadata_path, metadata_rows)

    result = {
        "phase_name": config.phase_name,
        "processed": processed,
        "skipped": skipped,
        "output_dir": str(config.output_dir),
        "patch_mode": config.patch_mode,
        "patch_enabled": bool(config.patch_enabled),
        "pair_metric": config.pair_metric,
        "pair_mode": config.pair_mode,
        "generate_source_behavior": bool(config.generate_source_behavior),
        "target_layers": [int(layer) for layer in config.target_layers],
        "components_per_layer": int(config.components_per_layer),
        "component_indices_by_layer": {
            str(layer): [int(index) for index in indices]
            for layer, indices in config.component_indices_by_layer.items()
        },
        "max_tokens": int(config.max_tokens),
        "selection_strategy": config.selection_strategy,
        "donor_means_path": str(config.donor_means_path) if config.donor_means_path is not None else None,
    }
    (config.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic market behavior patching on the current vLLM stack.")
    parser.add_argument("--phase-name", default="phase15_market_basis_discovery_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis_results/synthetic_market_behavior/phase18_market_behavior_v1"),
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--context-variant", default="market_only")
    parser.add_argument("--order-mode", default="selection_rank_asc")
    parser.add_argument("--selection-strategy", default="ordered")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--family-allowlist", default="")
    parser.add_argument("--pair-metric", default="")
    parser.add_argument("--pair-mode", default="")
    parser.add_argument("--min-pair-gap", type=float, default=0.0)
    parser.add_argument("--generate-source-behavior", action="store_true")
    parser.add_argument("--patch-mode", default="")
    parser.add_argument("--target-layers", default="4")
    parser.add_argument("--components-per-layer", type=int, default=4)
    parser.add_argument("--component-indices", default="")
    parser.add_argument("--direction-name", default="")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--tool-schema-mode", default="")
    parser.add_argument("--tool-choice", default="")
    parser.add_argument("--add-generation-prompt", action="store_true")
    parser.add_argument("--donor-means-path", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    target_layers = tuple(int(token) for token in args.target_layers.split(",") if token.strip())
    component_indices_by_layer = _parse_component_indices_spec(
        args.component_indices,
        target_layers=target_layers or (4,),
    )
    family_allowlist = tuple(token.strip() for token in args.family_allowlist.split(",") if token.strip())
    result = run_synthetic_market_behavior(
        SyntheticMarketBehaviorConfig(
            phase_name=args.phase_name,
            output_dir=args.output_dir,
            model_id=args.model_id,
            context_variant=args.context_variant,
            order_mode=args.order_mode,
            selection_strategy=args.selection_strategy,
            limit=args.limit if args.limit > 0 else None,
            family_allowlist=family_allowlist,
            pair_metric=args.pair_metric,
            pair_mode=args.pair_mode,
            min_pair_gap=args.min_pair_gap,
            generate_source_behavior=bool(args.generate_source_behavior),
            patch_mode=args.patch_mode,
            target_layers=target_layers or (4,),
            components_per_layer=args.components_per_layer,
            component_indices_by_layer=component_indices_by_layer,
            direction_name=args.direction_name,
            strength=args.strength,
            random_seed=args.random_seed,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            tool_schema_mode=args.tool_schema_mode,
            tool_choice=args.tool_choice,
            add_generation_prompt=bool(args.add_generation_prompt),
            donor_means_path=args.donor_means_path,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
