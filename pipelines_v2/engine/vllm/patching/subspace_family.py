"""Shared helpers for subspace-style activation patch operators."""

from __future__ import annotations

from typing import Any

import torch

from ..activation_patch_math import (
    ADD_DIRECTION_MODE_ID,
    ADD_DIRECTION_OPERATOR,
    PROJECT_OUT_MODE_ID,
    PROJECT_OUT_OPERATOR,
    RANDOM_CONTROL_MODE_ID,
    RANDOM_CONTROL_OPERATOR,
    SWAP_COMPONENTS_MODE_ID,
    SWAP_COMPONENTS_OPERATOR,
    SWAP_MEAN_MODE_ID,
    SWAP_MEAN_OPERATOR,
)
from .base import (
    random_orthogonal_rows,
    selected_component_rows,
)


SUBSPACE_OPERATORS = frozenset(
    {
        PROJECT_OUT_OPERATOR,
        ADD_DIRECTION_OPERATOR,
        SWAP_MEAN_OPERATOR,
        SWAP_COMPONENTS_OPERATOR,
        RANDOM_CONTROL_OPERATOR,
    }
)
SUBSPACE_OPERATOR_MODE_IDS = frozenset(
    {
        PROJECT_OUT_MODE_ID,
        ADD_DIRECTION_MODE_ID,
        SWAP_MEAN_MODE_ID,
        SWAP_COMPONENTS_MODE_ID,
        RANDOM_CONTROL_MODE_ID,
    }
)
SUBSPACE_STATS_SCALAR_DIM = 8


def is_subspace_operator(operator: str) -> bool:
    return str(operator) in SUBSPACE_OPERATORS


def is_subspace_mode_id(mode_id: int) -> bool:
    return int(mode_id) in SUBSPACE_OPERATOR_MODE_IDS


def zero_subspace_moments(*, hidden_dim: int, device: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = torch.zeros((int(hidden_dim),), device=device, dtype=torch.float32)
    scale = torch.ones((int(hidden_dim),), device=device, dtype=torch.float32)
    safe_scale = torch.ones((int(hidden_dim),), device=device, dtype=torch.float32)
    return mean, scale, safe_scale


def resolve_subspace_inputs(
    *,
    owner_model: Any,
    spec: Any,
    layer_idx: int,
    hidden_dim: int,
    device: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    source_layer = spec.source_layer_for(int(layer_idx))
    subspace = getattr(owner_model, "_v2_activation_patch_subspace", {})
    layer_payload = subspace.get(int(source_layer)) if isinstance(subspace, dict) else None

    if isinstance(layer_payload, dict):
        mean = layer_payload["mean"]
        scale = layer_payload["scale"]
        safe_scale = layer_payload["safe_scale"]
        selected_rows = selected_component_rows(
            layer_payload=layer_payload,
            spec=spec,
            layer_idx=int(layer_idx),
        )
    else:
        mean, scale, safe_scale = zero_subspace_moments(hidden_dim=int(hidden_dim), device=device)
        selected_rows = torch.empty((0, int(hidden_dim)), device=device, dtype=torch.float32)

    direction_raw = None
    direction_std = None
    donor_mean = None
    random_rows = None

    if spec.operator in {PROJECT_OUT_OPERATOR, RANDOM_CONTROL_OPERATOR, SWAP_COMPONENTS_OPERATOR} or (
        spec.operator == ADD_DIRECTION_OPERATOR and spec.selected_components_for(int(layer_idx))
    ):
        if not isinstance(layer_payload, dict):
            return None, {
                "layer": int(layer_idx),
                "status": "skipped",
                "reason": f"missing_subspace_source_layer:{int(source_layer)}",
                "operator": spec.operator,
            }

    if spec.operator == ADD_DIRECTION_OPERATOR:
        directions = getattr(owner_model, "_v2_activation_patch_directions", {})
        direction_payload = directions.get(int(source_layer)) if isinstance(directions, dict) else None
        if not isinstance(direction_payload, dict):
            return None, {
                "layer": int(layer_idx),
                "status": "skipped",
                "reason": f"missing_direction_source_layer:{int(source_layer)}",
                "operator": spec.operator,
            }
        direction_raw = direction_payload.get("raw_vector")
        direction_weights = direction_payload.get("subspace_weights")
        if direction_weights is not None and isinstance(layer_payload, dict):
            weights = direction_weights.to(torch.float32)
            selected = list(spec.selected_components_for(int(layer_idx)))
            components = layer_payload["components"]
            if selected:
                valid = [int(index) for index in selected if 0 <= int(index) < int(components.shape[0])]
                components = components[valid]
                if weights.numel():
                    weights = weights[valid]
            if components.ndim == 1:
                components = components.unsqueeze(0)
            if weights.numel() and components.numel():
                direction_std = (weights @ components).to(torch.float32)
        if direction_std is None and direction_raw is None:
            return None, {
                "layer": int(layer_idx),
                "status": "skipped",
                "reason": "missing_direction_vector",
                "operator": spec.operator,
            }
    elif spec.operator in {SWAP_MEAN_OPERATOR, SWAP_COMPONENTS_OPERATOR}:
        centroids = getattr(owner_model, "_v2_activation_patch_centroids", {})
        centroid_layer = centroids.get(int(source_layer)) if isinstance(centroids, dict) else None
        if not isinstance(centroid_layer, dict):
            return None, {
                "layer": int(layer_idx),
                "status": "skipped",
                "reason": f"missing_centroid_source_layer:{int(source_layer)}",
                "operator": spec.operator,
            }
        donor_mean = dict(centroid_layer.get("centroids", {})).get(spec.centroid_name)
        if donor_mean is None:
            return None, {
                "layer": int(layer_idx),
                "status": "skipped",
                "reason": f"missing_centroid:{spec.centroid_name}",
                "operator": spec.operator,
            }
    elif spec.operator == RANDOM_CONTROL_OPERATOR:
        if int(selected_rows.shape[0]) <= 0:
            return None, {
                "layer": int(layer_idx),
                "status": "skipped",
                "reason": "no_selected_components",
                "operator": spec.operator,
                "source_layer": int(source_layer),
            }
        random_rows = random_orthogonal_rows(
            target_rows=selected_rows,
            num_rows=max(1, int(selected_rows.shape[0])),
            dim=int(mean.shape[0]),
            seed=int(spec.random_seed) + int(layer_idx),
            device=device,
            dtype=torch.float32,
        )

    if spec.operator in {PROJECT_OUT_OPERATOR, RANDOM_CONTROL_OPERATOR, SWAP_COMPONENTS_OPERATOR} and int(selected_rows.shape[0]) <= 0:
        return None, {
            "layer": int(layer_idx),
            "status": "skipped",
            "reason": "no_selected_components",
            "operator": spec.operator,
            "source_layer": int(source_layer),
        }

    return (
        {
            "source_layer": int(source_layer),
            "mean": mean.to(torch.float32),
            "scale": scale.to(torch.float32),
            "safe_scale": safe_scale.to(torch.float32),
            "selected_rows": selected_rows.to(torch.float32),
            "direction_raw": direction_raw.to(torch.float32) if direction_raw is not None else None,
            "direction_std": direction_std.to(torch.float32) if direction_std is not None else None,
            "donor_mean": donor_mean.to(torch.float32) if donor_mean is not None else None,
            "random_rows": random_rows.to(torch.float32) if random_rows is not None else None,
        },
        None,
    )


def summarize_harvested_subspace_stats(
    *,
    spec: Any,
    layer_idx: int,
    scalars: list[float],
    coeff_before: list[float],
    coeff_after: list[float],
    covered_abs_positions: list[int],
) -> dict[str, Any]:
    unique_positions = sorted({int(pos) for pos in covered_abs_positions})
    target_abs_tokens = len({int(pos) for pos in getattr(spec, "target_abs_positions", ())})
    covered_abs_spans: list[list[int]] = [
        [int(start), int(end)]
        for start, end in getattr(spec, "covered_abs_spans", ())
    ]
    if unique_positions:
        covered_abs_spans = []
        start = prev = unique_positions[0]
        for pos in unique_positions[1:]:
            if pos == prev + 1:
                prev = pos
                continue
            covered_abs_spans.append([int(start), int(prev + 1)])
            start = prev = pos
        covered_abs_spans.append([int(start), int(prev + 1)])

    token_count = int(spec.token_count()) if hasattr(spec, "token_count") else int(len(spec.query_positions))
    stats = {
        "layer": int(layer_idx),
        "source_layer": int(spec.source_layer_for(int(layer_idx))),
        "status": "ok",
        "operator": spec.operator,
        "dispatch": "compiled_custom_op",
        "token_count": int(token_count),
        "case_key": spec.case_key,
        "control_name": spec.control_name,
        "covered_abs_spans": covered_abs_spans,
        "covered_abs_tokens": int(len(unique_positions) or sum(max(0, int(end) - int(start)) for start, end in covered_abs_spans)),
        "target_abs_tokens": int(target_abs_tokens),
        "coverage_fraction": (
            float(len(unique_positions)) / float(target_abs_tokens)
            if int(target_abs_tokens) > 0
            else 0.0
        ),
    }
    if spec.query_span:
        stats["query_span"] = [int(spec.query_span[0]), int(spec.query_span[1])]
    else:
        stats["query_positions"] = [int(pos) for pos in spec.query_positions]
    if spec.phase_counts:
        stats["phase_counts"] = {str(name): int(count) for name, count in spec.phase_counts}
    if spec.target_policy:
        stats["target_policy"] = dict(spec.target_policy)
    if spec.rowwise:
        stats["rowwise"] = True

    if spec.operator == PROJECT_OUT_OPERATOR:
        stats.update(
            {
                "delta_norm_raw": float(scalars[0]),
                "delta_norm_std": float(scalars[1]),
                "mean_norm_before": float(scalars[2]),
                "mean_norm_after": float(scalars[3]),
                "mean_std_norm_before": float(scalars[4]),
                "mean_std_norm_after": float(scalars[5]),
                "selected_proj_norm_before": float(scalars[6]),
                "strength": float(scalars[7]),
                "selected_component_count": int(len(coeff_before)),
                "selected_coeff_before": [float(v) for v in coeff_before],
                "selected_coeff_after": [float(v) for v in coeff_after],
            }
        )
        return stats

    stats["strength"] = float(scalars[7]) if len(scalars) > 7 else float(spec.strength)
    stats["delta_norm_raw"] = float(scalars[0]) if scalars else 0.0
    stats["delta_norm_std"] = float(scalars[1]) if len(scalars) > 1 else 0.0

    if spec.operator == ADD_DIRECTION_OPERATOR:
        stats["direction_norm_raw"] = float(scalars[6]) if len(scalars) > 6 else 0.0
    elif spec.operator == SWAP_MEAN_OPERATOR:
        stats["mean_norm_before"] = float(scalars[2]) if len(scalars) > 2 else 0.0
        stats["mean_norm_after"] = float(scalars[3]) if len(scalars) > 3 else 0.0
    elif spec.operator in {SWAP_COMPONENTS_OPERATOR, RANDOM_CONTROL_OPERATOR}:
        stats["selected_component_count"] = int(len(coeff_before))
        stats["selected_coeff_before"] = [float(v) for v in coeff_before]
        stats["selected_coeff_after"] = [float(v) for v in coeff_after]
        stats["selected_proj_norm_before"] = float(scalars[6]) if len(scalars) > 6 else 0.0

    return stats


__all__ = [
    "SUBSPACE_OPERATORS",
    "SUBSPACE_OPERATOR_MODE_IDS",
    "SUBSPACE_STATS_SCALAR_DIM",
    "is_subspace_mode_id",
    "is_subspace_operator",
    "resolve_subspace_inputs",
    "summarize_harvested_subspace_stats",
]
