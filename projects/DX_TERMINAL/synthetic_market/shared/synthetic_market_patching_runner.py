from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipelines.db import connect_neon
from pipelines.interp.pooling import (
    _extract_system_user,
    _flush_table,
    _parse_messages,
    _save_pooled,
    pool_decision_residual,
)
from pipelines.interp.patching.basis import default_phase17_market_patch_basis
from pipelines.datasets.synthetic.db import validate_order_mode
from pipelines.datasets.synthetic.structure import (
    build_asset_label_rows,
    build_tick_label_row,
    find_synthetic_row_boundaries,
    find_synthetic_section_boundaries,
)
from pipelines.interp.modal_vllm_engine import (
    VLLMCaptureConfig,
    _capture_one_vllm,
    _create_llm,
    _init_market_patching_on_model,
    _register_market_patch_basis_on_model,
    _collect_market_patch_stats_from_model,
)
from pipelines.interp.patching.market_patch import (
    PATCH_MODE_ADD_DIRECTION,
    PATCH_MODE_SWAP_COMPONENTS,
    PATCH_MODE_SWAP_MEAN,
    MarketPatchSpec,
)


@dataclass(slots=True)
class SyntheticMarketPatchingConfig:
    phase_name: str = "phase15_market_basis_discovery_v1"
    output_dir: Path = field(
        default_factory=lambda: Path("data/activations/synthetic_market_patching/phase18_market_patching_v1")
    )
    model_id: str = "Qwen/Qwen3-30B-A3B"
    context_variant: str = "market_only"
    order_mode: str = "selection_rank_asc"
    selection_strategy: str = "ordered"
    limit: int | None = None
    family_allowlist: tuple[str, ...] = ()
    patch_mode: str = "project_out"
    target_layers: tuple[int, ...] = (4,)
    components_per_layer: int = 4
    component_indices_by_layer: dict[int, tuple[int, ...]] = field(default_factory=dict)
    direction_name: str = ""
    strength: float = 1.0
    random_seed: int = 42
    capture_router: bool = False
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.85
    basis_npz_path: Path = Path(
        "data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/pca_basis.npz"
    )
    basis_results_path: Path = Path(
        "data/analysis_results/synthetic_market_axis_decomposition/phase17_market_axis_decomposition_v1/results.json"
    )


_ROSTER_KEY_RE = re.compile(r"_r(\d+)_")


def _extract_roster_key(example_id: str) -> str:
    match = _ROSTER_KEY_RE.search(str(example_id))
    if match is None:
        return "unknown"
    return f"r{match.group(1)}"


def _selection_bucket_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("family", "")),
        str(row.get("family_variant", "")),
        _extract_roster_key(str(row.get("example_id", ""))),
    )


def _parse_component_indices_spec(
    spec_text: str,
    *,
    target_layers: tuple[int, ...],
) -> dict[int, tuple[int, ...]]:
    text = str(spec_text).strip()
    if not text:
        return {}

    def _parse_index_list(raw: str) -> tuple[int, ...]:
        values = [int(token.strip()) for token in raw.split(",") if token.strip()]
        deduped: list[int] = []
        seen: set[int] = set()
        for value in values:
            if value < 0 or value in seen:
                continue
            deduped.append(value)
            seen.add(value)
        return tuple(deduped)

    if "=" not in text and ":" not in text:
        indices = _parse_index_list(text)
        return {int(layer): indices for layer in target_layers} if indices else {}

    assignments = re.split(r"[;|]", text)
    parsed: dict[int, tuple[int, ...]] = {}
    for assignment in assignments:
        chunk = assignment.strip()
        if not chunk:
            continue
        if "=" in chunk:
            layer_text, index_text = chunk.split("=", 1)
        elif ":" in chunk:
            layer_text, index_text = chunk.split(":", 1)
        else:
            raise ValueError(
                f"Invalid component index assignment {assignment!r}. "
                "Expected '0,1,2' or '4=0,1;35=0,1'."
            )
        layer = int(layer_text.strip())
        indices = _parse_index_list(index_text)
        if indices:
            parsed[layer] = indices
    return parsed


def _apply_selection_strategy(
    rows: list[dict[str, Any]],
    *,
    strategy: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    if limit is None or limit <= 0:
        return rows
    text = str(strategy).strip().lower() or "ordered"
    if text == "ordered":
        return rows[:limit]
    if text != "stratified_family_variant_roster":
        raise ValueError(
            f"Unknown selection strategy: {strategy!r}. "
            "Expected 'ordered' or 'stratified_family_variant_roster'."
        )

    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    bucket_order: list[tuple[str, str, str]] = []
    for row in rows:
        key = _selection_bucket_key(row)
        if key not in buckets:
            buckets[key] = []
            bucket_order.append(key)
        buckets[key].append(row)

    selected: list[dict[str, Any]] = []
    bucket_indices = {key: 0 for key in bucket_order}
    while len(selected) < limit:
        progressed = False
        for key in bucket_order:
            idx = bucket_indices[key]
            bucket = buckets[key]
            if idx >= len(bucket):
                continue
            selected.append(bucket[idx])
            bucket_indices[key] = idx + 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def _load_examples(
    *,
    phase_name: str,
    context_variant: str,
    order_mode: str,
    selection_strategy: str,
    limit: int | None,
    family_allowlist: tuple[str, ...],
) -> list[dict[str, Any]]:
    order_mode = validate_order_mode(order_mode)
    query = f"""
        SELECT
            log_id,
            phase_name,
            example_id,
            family,
            family_variant,
            context_variant,
            prompt_messages_json
        FROM synthetic_market_examples_v0
        WHERE phase_name = %s
          AND context_variant = %s
          {"AND family = ANY(%s)" if family_allowlist else ""}
        ORDER BY
          {"selection_rank ASC NULLS LAST, log_id" if order_mode == "selection_rank_asc" else "log_id"}
    """
    params: list[Any] = [phase_name, context_variant]
    if family_allowlist:
        params.append(list(family_allowlist))

    conn = connect_neon()
    try:
        rows = conn.execute(query, params).fetchall()
        loaded = [dict(row) for row in rows]
        for row in loaded:
            row["roster_key"] = _extract_roster_key(str(row.get("example_id", "")))
        return _apply_selection_strategy(loaded, strategy=selection_strategy, limit=limit)
    finally:
        conn.close()


def _load_asset_rows(log_ids: list[int], *, phase_name: str) -> dict[int, list[dict[str, Any]]]:
    if not log_ids:
        return {}
    conn = connect_neon()
    try:
        rows = conn.execute(
            """
            SELECT *
            FROM synthetic_market_assets_v0
            WHERE phase_name = %s AND log_id = ANY(%s)
            ORDER BY log_id, row_index
            """,
            (phase_name, log_ids),
        ).fetchall()
    finally:
        conn.close()

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["log_id"]), []).append(dict(row))
    return grouped


def _as_numpy_residual(residual: Any) -> Any:
    if hasattr(residual, "detach"):
        residual = residual.detach()
    if hasattr(residual, "cpu"):
        residual = residual.cpu()
    if hasattr(residual, "numpy"):
        return residual.numpy()
    return residual


def _build_patch_spec(
    *,
    config: SyntheticMarketPatchingConfig,
    market_span: tuple[int, int],
    basis_payload: dict[int, dict[str, Any]],
    donor_mean_by_layer: dict[int, Any] | None = None,
) -> MarketPatchSpec:
    component_indices: dict[int, tuple[int, ...]] = {}
    for layer in config.target_layers:
        payload = basis_payload.get(int(layer))
        if payload is None:
            continue
        explicit_indices = config.component_indices_by_layer.get(int(layer))
        if explicit_indices:
            component_indices[int(layer)] = tuple(int(index) for index in explicit_indices)
        elif config.direction_name:
            named_components = dict(payload.get("named_components", {}))
            if config.direction_name not in named_components:
                raise KeyError(
                    f"Direction {config.direction_name!r} not available at layer {layer}. "
                    f"Known directions: {sorted(named_components)}"
                )
            component_indices[int(layer)] = (int(named_components[config.direction_name]),)
        else:
            component_indices[int(layer)] = tuple(
                range(min(config.components_per_layer, payload["components"].shape[0]))
            )

    if config.patch_mode == PATCH_MODE_ADD_DIRECTION:
        if not config.direction_name:
            raise ValueError("direction_name is required for add_direction")
        direction_weights_by_layer: dict[int, Any] = {}
        for layer in config.target_layers:
            payload = basis_payload.get(int(layer))
            if payload is None:
                continue
            named_components = dict(payload.get("named_components", {}))
            if config.direction_name not in named_components:
                raise KeyError(
                    f"Direction {config.direction_name!r} not available at layer {layer}. "
                    f"Known directions: {sorted(named_components)}"
                )
            weights = np.zeros((payload["components"].shape[0],), dtype=np.float32)
            weights[int(named_components[config.direction_name])] = 1.0
            direction_weights_by_layer[int(layer)] = weights
        return MarketPatchSpec(
            mode=config.patch_mode,
            target_layers=tuple(int(layer) for layer in config.target_layers),
            token_span=market_span,
            strength=float(config.strength),
            direction_weights_by_layer=direction_weights_by_layer,
            random_seed=int(config.random_seed),
        )

    if config.patch_mode in {PATCH_MODE_SWAP_MEAN, PATCH_MODE_SWAP_COMPONENTS}:
        if not donor_mean_by_layer:
            raise ValueError(f"{config.patch_mode} requires donor_mean_by_layer")
        return MarketPatchSpec(
            mode=config.patch_mode,
            target_layers=tuple(int(layer) for layer in config.target_layers),
            token_span=market_span,
            strength=float(config.strength),
            donor_mean_by_layer=donor_mean_by_layer,
            random_seed=int(config.random_seed),
        )

    return MarketPatchSpec(
        mode=config.patch_mode,
        target_layers=tuple(int(layer) for layer in config.target_layers),
        token_span=market_span,
        strength=float(config.strength),
        component_indices_by_layer=component_indices,
        random_seed=int(config.random_seed),
    )


def run_synthetic_market_patching(config: SyntheticMarketPatchingConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config.output_dir.mkdir(parents=True, exist_ok=True)
    residual_out_dir = config.output_dir / "residual"
    tmp_capture_dir = config.output_dir / "_tmp_capture"
    metadata_path = config.output_dir / "metadata.parquet"
    tick_path = config.output_dir / "tick_labels.parquet"
    asset_path = config.output_dir / "asset_labels.parquet"

    examples = _load_examples(
        phase_name=config.phase_name,
        context_variant=config.context_variant,
        order_mode=config.order_mode,
        selection_strategy=config.selection_strategy,
        limit=config.limit,
        family_allowlist=config.family_allowlist,
    )
    if not examples:
        result = {"error": "no_examples"}
        (config.output_dir / "results.json").write_text(json.dumps(result, indent=2))
        return result

    asset_rows_by_log = _load_asset_rows([int(row["log_id"]) for row in examples], phase_name=config.phase_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    basis = default_phase17_market_patch_basis(
        basis_npz_path=config.basis_npz_path,
        results_json_path=config.basis_results_path,
        layers=tuple(int(layer) for layer in config.target_layers),
        components_per_layer=int(config.components_per_layer),
    )
    basis_payload = basis.to_payload()

    llm = _create_llm(
        VLLMCaptureConfig(
            output_dir=tmp_capture_dir,
            model_id=config.model_id,
            capture_router=bool(config.capture_router),
            capture_residual=True,
            tensor_parallel_size=int(config.tensor_parallel_size),
            gpu_memory_utilization=float(config.gpu_memory_utilization),
            enable_prefix_caching=False,
        )
    )
    _init_market_patching_on_model(llm)
    _register_market_patch_basis_on_model(llm, basis_payload)

    capture_cfg = VLLMCaptureConfig(
        output_dir=tmp_capture_dir,
        model_id=config.model_id,
        capture_router=bool(config.capture_router),
        capture_residual=True,
        tensor_parallel_size=int(config.tensor_parallel_size),
        gpu_memory_utilization=float(config.gpu_memory_utilization),
        enable_prefix_caching=False,
    )

    metadata_rows: list[dict[str, Any]] = []
    tick_rows: list[dict[str, Any]] = []
    asset_rows: list[dict[str, Any]] = []
    processed = 0
    skipped = 0

    for row in examples:
        log_id = int(row["log_id"])
        messages = _parse_messages(row["prompt_messages_json"])
        system_user = _extract_system_user(messages)
        prompt_asset_rows = asset_rows_by_log.get(log_id)
        if system_user is None or not prompt_asset_rows:
            skipped += 1
            continue
        system_text, user_text = system_user
        section_boundaries = find_synthetic_section_boundaries(tokenizer, system_text, user_text)
        row_boundaries = find_synthetic_row_boundaries(tokenizer, system_text, user_text, prompt_asset_rows)
        market_span = section_boundaries.get("market")
        if market_span is None:
            skipped += 1
            continue

        patch_spec = _build_patch_spec(
            config=config,
            market_span=market_span,
            basis_payload=basis_payload,
        )
        residual, _, _, input_ids = _capture_one_vllm(
            llm=llm,
            tokenizer=tokenizer,
            messages=messages,
            config=capture_cfg,
            log_id=f"patch_{log_id}",
            patch_spec=patch_spec.to_payload(),
            skip_residual_save=True,
        )
        if residual is None:
            skipped += 1
            continue

        pooled = pool_decision_residual(
            _as_numpy_residual(residual),
            row_boundaries=row_boundaries,
            section_boundaries=section_boundaries,
        )
        file_size = _save_pooled(pooled, residual_out_dir / f"{log_id}.safetensors")
        patch_stats = _collect_market_patch_stats_from_model(llm)

        metadata_rows.append({
            "log_id": log_id,
            "phase_name": row["phase_name"],
            "example_id": row["example_id"],
            "family": row["family"],
            "family_variant": row["family_variant"],
            "context_variant": row["context_variant"],
            "roster_key": row.get("roster_key") or "",
            "capture_timestamp": datetime.now(UTC).isoformat(),
            "seq_len": int(len(input_ids)),
            "n_rows": len(row_boundaries),
            "n_pooled": len(pooled),
            "file_size_bytes": int(file_size),
            "patch_mode": config.patch_mode,
            "target_layers": ",".join(str(int(layer)) for layer in config.target_layers),
            "components_per_layer": int(config.components_per_layer),
            "component_indices_by_layer": {
                str(layer): [int(index) for index in indices]
                for layer, indices in config.component_indices_by_layer.items()
            },
            "direction_name": config.direction_name,
            "selection_strategy": config.selection_strategy,
            "strength": float(config.strength),
            "patch_stats_json": json.dumps(patch_stats),
        })
        tick_rows.append(build_tick_label_row(row, user_text=user_text, n_rows=len(row_boundaries)))
        asset_rows.extend(build_asset_label_rows(log_id, prompt_asset_rows))
        processed += 1

    _flush_table(metadata_path, metadata_rows)
    _flush_table(tick_path, tick_rows)
    _flush_table(asset_path, asset_rows)

    result = {
        "phase_name": config.phase_name,
        "processed": processed,
        "skipped": skipped,
        "output_dir": str(config.output_dir),
        "patch_mode": config.patch_mode,
        "target_layers": [int(layer) for layer in config.target_layers],
        "components_per_layer": int(config.components_per_layer),
        "component_indices_by_layer": {
            str(layer): [int(index) for index in indices]
            for layer, indices in config.component_indices_by_layer.items()
        },
        "direction_name": config.direction_name,
        "selection_strategy": config.selection_strategy,
    }
    (config.output_dir / "results.json").write_text(json.dumps(result, indent=2))
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run synthetic market patching captures on the current vLLM stack.")
    parser.add_argument("--phase-name", default="phase15_market_basis_discovery_v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/activations/synthetic_market_patching/phase18_market_patching_v1"),
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--context-variant", default="market_only")
    parser.add_argument("--order-mode", default="selection_rank_asc")
    parser.add_argument("--selection-strategy", default="ordered")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--family-allowlist", default="")
    parser.add_argument("--patch-mode", default="project_out")
    parser.add_argument("--target-layers", default="4")
    parser.add_argument("--components-per-layer", type=int, default=4)
    parser.add_argument("--component-indices", default="")
    parser.add_argument("--direction-name", default="")
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    target_layers = tuple(int(token) for token in args.target_layers.split(",") if token.strip())
    component_indices_by_layer = _parse_component_indices_spec(
        args.component_indices,
        target_layers=target_layers or (4,),
    )
    family_allowlist = tuple(token.strip() for token in args.family_allowlist.split(",") if token.strip())
    result = run_synthetic_market_patching(
        SyntheticMarketPatchingConfig(
            phase_name=args.phase_name,
            output_dir=args.output_dir,
            model_id=args.model_id,
            context_variant=args.context_variant,
            order_mode=args.order_mode,
            selection_strategy=args.selection_strategy,
            limit=args.limit if args.limit > 0 else None,
            family_allowlist=family_allowlist,
            patch_mode=args.patch_mode,
            target_layers=target_layers or (4,),
            components_per_layer=args.components_per_layer,
            component_indices_by_layer=component_indices_by_layer,
            direction_name=args.direction_name,
            strength=args.strength,
            random_seed=args.random_seed,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
