"""Shared activation patch operator math.

This module is the canonical patching layer below the vLLM runtime glue.
Operators here define the actual hidden-state edits and diagnostic summaries.
"""

from __future__ import annotations

from typing import Any

import torch

INTERCHANGE_OPERATOR = "interchange"
PROJECT_OUT_OPERATOR = "project_out"
ADD_DIRECTION_OPERATOR = "add_direction"
SWAP_MEAN_OPERATOR = "swap_mean"
SWAP_COMPONENTS_OPERATOR = "swap_components"
RANDOM_CONTROL_OPERATOR = "random_control"
RESIDUAL_PATH_OPERATOR = "residual_path"

INTERCHANGE_MODE_ID = 0
PROJECT_OUT_MODE_ID = 1
ADD_DIRECTION_MODE_ID = 2
SWAP_MEAN_MODE_ID = 3
SWAP_COMPONENTS_MODE_ID = 4
RANDOM_CONTROL_MODE_ID = 5
RESIDUAL_PATH_MODE_ID = 6

OPERATOR_TO_MODE_ID = {
    INTERCHANGE_OPERATOR: INTERCHANGE_MODE_ID,
    PROJECT_OUT_OPERATOR: PROJECT_OUT_MODE_ID,
    ADD_DIRECTION_OPERATOR: ADD_DIRECTION_MODE_ID,
    SWAP_MEAN_OPERATOR: SWAP_MEAN_MODE_ID,
    SWAP_COMPONENTS_OPERATOR: SWAP_COMPONENTS_MODE_ID,
    RANDOM_CONTROL_OPERATOR: RANDOM_CONTROL_MODE_ID,
    RESIDUAL_PATH_OPERATOR: RESIDUAL_PATH_MODE_ID,
}
MODE_ID_TO_OPERATOR = {value: key for key, value in OPERATOR_TO_MODE_ID.items()}


def operator_mode_id(operator: str) -> int:
    try:
        return int(OPERATOR_TO_MODE_ID[str(operator)])
    except KeyError as exc:
        raise ValueError(f"Unsupported activation patch operator: {operator!r}") from exc


def operator_from_mode_id(mode_id: int) -> str:
    try:
        return str(MODE_ID_TO_OPERATOR[int(mode_id)])
    except KeyError as exc:
        raise ValueError(f"Unsupported activation patch mode id: {mode_id!r}") from exc


def compute_section_state(
    section: torch.Tensor,
    *,
    mean: torch.Tensor,
    safe_scale: torch.Tensor,
    selected_rows: torch.Tensor,
) -> dict[str, Any]:
    mu = section.mean(dim=0)
    centered_std = (mu - mean) / safe_scale
    if selected_rows.numel() > 0:
        selected_coeff = centered_std @ selected_rows.T
        selected_projected_std = selected_coeff @ selected_rows
        selected_proj_norm = float(torch.linalg.norm(selected_projected_std).item())
    else:
        selected_coeff = torch.empty((0,), device=centered_std.device, dtype=torch.float32)
        selected_projected_std = torch.zeros_like(centered_std)
        selected_proj_norm = 0.0
    return {
        "mu": mu,
        "centered_std": centered_std,
        "selected_coeff": selected_coeff,
        "selected_projected_std": selected_projected_std,
        "selected_proj_norm": selected_proj_norm,
    }


def summarize_subspace_patch(
    *,
    original_section: torch.Tensor,
    patched_section: torch.Tensor,
    mean: torch.Tensor,
    safe_scale: torch.Tensor,
    selected_rows: torch.Tensor,
    strength: float,
) -> dict[str, Any]:
    before = compute_section_state(
        original_section,
        mean=mean,
        safe_scale=safe_scale,
        selected_rows=selected_rows,
    )
    after = compute_section_state(
        patched_section,
        mean=mean,
        safe_scale=safe_scale,
        selected_rows=selected_rows,
    )
    delta_raw = patched_section.mean(dim=0) - original_section.mean(dim=0)
    return {
        "delta_norm_raw": float(torch.linalg.norm(delta_raw).item()),
        "delta_norm_std": float(torch.linalg.norm(delta_raw / safe_scale).item()),
        "mean_norm_before": float(torch.linalg.norm(before["mu"]).item()),
        "mean_norm_after": float(torch.linalg.norm(after["mu"]).item()),
        "mean_std_norm_before": float(torch.linalg.norm(before["centered_std"]).item()),
        "mean_std_norm_after": float(torch.linalg.norm(after["centered_std"]).item()),
        "selected_proj_norm_before": float(before["selected_proj_norm"]),
        "selected_coeff_before": before["selected_coeff"].detach().cpu().tolist(),
        "selected_coeff_after": after["selected_coeff"].detach().cpu().tolist(),
        "selected_component_count": int(selected_rows.shape[0]),
        "strength": float(strength),
    }


def apply_interchange(
    hidden_states: torch.Tensor,
    *,
    query_positions: list[int] | tuple[int, ...],
    donor_rows: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not query_positions:
        return hidden_states, {"status": "skipped", "reason": "no_query_positions"}
    patched = hidden_states.clone()
    before = hidden_states[list(query_positions)].to(torch.float32)
    donor_rows_f32 = donor_rows.to(torch.float32)
    patched[list(query_positions)] = donor_rows.to(dtype=hidden_states.dtype)
    stats = {
        "status": "ok",
        "token_count": int(len(query_positions)),
        "delta_norm_raw": float(torch.linalg.norm(donor_rows_f32 - before).item()),
    }
    return patched, stats


def apply_subspace_operator(
    hidden_states: torch.Tensor,
    *,
    query_positions: list[int] | tuple[int, ...],
    operator: str,
    mean: torch.Tensor,
    scale: torch.Tensor,
    safe_scale: torch.Tensor,
    selected_rows: torch.Tensor,
    strength: float,
    direction_raw: torch.Tensor | None = None,
    direction_std: torch.Tensor | None = None,
    donor_mean: torch.Tensor | None = None,
    random_rows: torch.Tensor | None = None,
    match_projected_norm: bool = True,
    rowwise: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    operator_name = operator_from_mode_id(int(operator)) if isinstance(operator, int) else str(operator)
    if not query_positions:
        return hidden_states, {"status": "skipped", "reason": "no_query_positions", "operator": operator_name}

    section = hidden_states[list(query_positions)].to(torch.float32)
    before = compute_section_state(section, mean=mean, safe_scale=safe_scale, selected_rows=selected_rows)

    if bool(rowwise):
        centered_rows = (section - mean) / safe_scale
        if selected_rows.numel() > 0:
            selected_coeff = centered_rows @ selected_rows.T
            selected_projected_std = selected_coeff @ selected_rows
            selected_proj_norm = torch.linalg.vector_norm(selected_projected_std, dim=1)
        else:
            selected_projected_std = torch.zeros_like(centered_rows)
            selected_proj_norm = torch.zeros((section.shape[0],), device=section.device, dtype=torch.float32)

        if operator_name == PROJECT_OUT_OPERATOR:
            delta_std = -float(strength) * selected_projected_std
            patched_section = section + (delta_std * scale)
        elif operator_name == ADD_DIRECTION_OPERATOR:
            if direction_std is not None:
                patched_section = section + (float(strength) * direction_std.to(torch.float32) * scale)
            elif direction_raw is not None:
                patched_section = section + (float(strength) * direction_raw.to(torch.float32))
            else:
                raise ValueError("add_direction requires direction_std or direction_raw")
        elif operator_name == SWAP_MEAN_OPERATOR:
            if donor_mean is None:
                raise ValueError("swap_mean requires donor_mean")
            patched_section = section + (float(strength) * (donor_mean.to(torch.float32) - section))
        elif operator_name == SWAP_COMPONENTS_OPERATOR:
            if donor_mean is None:
                raise ValueError("swap_components requires donor_mean")
            donor_centered_std = (donor_mean.to(torch.float32) - mean) / safe_scale
            if selected_rows.numel() > 0:
                donor_selected_coeff = donor_centered_std @ selected_rows.T
                donor_selected_projected_std = donor_selected_coeff @ selected_rows
            else:
                donor_selected_projected_std = torch.zeros_like(before["centered_std"])
            delta_std = float(strength) * (donor_selected_projected_std.unsqueeze(0) - selected_projected_std)
            patched_section = section + (delta_std * scale)
        elif operator_name == RANDOM_CONTROL_OPERATOR:
            if random_rows is None:
                raise ValueError("random_control requires random_rows")
            random_coeff = centered_rows @ random_rows.T
            random_projected_std = random_coeff @ random_rows
            if bool(match_projected_norm):
                random_norm = torch.linalg.vector_norm(random_projected_std, dim=1).clamp_min(1e-8)
                scale_factor = torch.where(
                    selected_proj_norm > 0,
                    selected_proj_norm / random_norm,
                    torch.ones_like(random_norm),
                )
                random_projected_std = random_projected_std * scale_factor.unsqueeze(1)
            delta_std = -float(strength) * random_projected_std
            patched_section = section + (delta_std * scale)
        else:
            raise ValueError(f"Unsupported subspace operator: {operator_name!r}")

        patched = hidden_states.clone()
        patched[list(query_positions)] = patched_section.to(dtype=hidden_states.dtype)
        stats = summarize_subspace_patch(
            original_section=section,
            patched_section=patched_section,
            mean=mean,
            safe_scale=safe_scale,
            selected_rows=selected_rows,
            strength=float(strength),
        )
        if operator_name == ADD_DIRECTION_OPERATOR:
            if direction_raw is not None:
                stats["direction_norm_raw"] = float(torch.linalg.norm(direction_raw.to(torch.float32)).item())
            elif direction_std is not None:
                stats["direction_norm_raw"] = float(torch.linalg.norm((direction_std.to(torch.float32) * scale)).item())
        stats["status"] = "ok"
        stats["operator"] = operator_name
        stats["token_count"] = int(len(query_positions))
        stats["rowwise"] = True
        return patched, stats

    if operator_name == PROJECT_OUT_OPERATOR:
        delta_std = -float(strength) * before["selected_projected_std"]
        patched_section = section + (delta_std * scale)
    elif operator_name == ADD_DIRECTION_OPERATOR:
        if direction_std is not None:
            delta_std = float(strength) * direction_std.to(torch.float32)
            patched_section = section + (delta_std * scale)
        elif direction_raw is not None:
            patched_section = section + (float(strength) * direction_raw.to(torch.float32))
        else:
            raise ValueError("add_direction requires direction_std or direction_raw")
    elif operator_name == SWAP_MEAN_OPERATOR:
        if donor_mean is None:
            raise ValueError("swap_mean requires donor_mean")
        delta_raw = float(strength) * (donor_mean.to(torch.float32) - before["mu"])
        patched_section = section + delta_raw
    elif operator_name == SWAP_COMPONENTS_OPERATOR:
        if donor_mean is None:
            raise ValueError("swap_components requires donor_mean")
        donor_centered_std = (donor_mean.to(torch.float32) - mean) / safe_scale
        if selected_rows.numel() > 0:
            donor_selected_coeff = donor_centered_std @ selected_rows.T
            donor_selected_projected_std = donor_selected_coeff @ selected_rows
        else:
            donor_selected_projected_std = torch.zeros_like(before["centered_std"])
        delta_std = float(strength) * (donor_selected_projected_std - before["selected_projected_std"])
        patched_section = section + (delta_std * scale)
    elif operator_name == RANDOM_CONTROL_OPERATOR:
        if random_rows is None:
            raise ValueError("random_control requires random_rows")
        random_coeff = before["centered_std"] @ random_rows.T
        random_projected_std = random_coeff @ random_rows
        if bool(match_projected_norm) and before["selected_proj_norm"] > 0.0:
            random_norm = float(torch.linalg.norm(random_projected_std).item())
            if random_norm > 1e-8:
                random_projected_std = random_projected_std * (
                    float(before["selected_proj_norm"]) / random_norm
                )
        delta_std = -float(strength) * random_projected_std
        patched_section = section + (delta_std * scale)
    else:
        raise ValueError(f"Unsupported subspace operator: {operator_name!r}")

    patched = hidden_states.clone()
    patched[list(query_positions)] = patched_section.to(dtype=hidden_states.dtype)
    stats = summarize_subspace_patch(
        original_section=section,
        patched_section=patched_section,
        mean=mean,
        safe_scale=safe_scale,
        selected_rows=selected_rows,
        strength=float(strength),
    )
    if operator_name == ADD_DIRECTION_OPERATOR:
        if direction_raw is not None:
            stats["direction_norm_raw"] = float(torch.linalg.norm(direction_raw.to(torch.float32)).item())
        elif direction_std is not None:
            stats["direction_norm_raw"] = float(torch.linalg.norm((direction_std.to(torch.float32) * scale)).item())
    stats["status"] = "ok"
    stats["operator"] = operator_name
    stats["token_count"] = int(len(query_positions))
    return patched, stats


def apply_residual_path_transport(
    hidden_states: torch.Tensor,
    *,
    query_positions: list[int] | tuple[int, ...],
    donor_rows: torch.Tensor,
    target_source_rows: torch.Tensor | None,
    weight: float,
    strength: float,
    transport: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not query_positions:
        return hidden_states, {"status": "skipped", "reason": "no_query_positions", "operator": RESIDUAL_PATH_OPERATOR}
    section = hidden_states[list(query_positions)].to(torch.float32)
    donor_rows_f32 = donor_rows.to(torch.float32)
    if transport == "replace":
        delta = donor_rows_f32 - section
    elif transport == "delta":
        if target_source_rows is None:
            raise ValueError("residual path delta transport requires target_source_rows")
        delta = donor_rows_f32 - target_source_rows.to(torch.float32)
    else:
        raise ValueError(f"Unsupported residual path transport: {transport!r}")
    scaled_delta = float(weight) * float(strength) * delta
    patched_section = section + scaled_delta
    patched = hidden_states.clone()
    patched[list(query_positions)] = patched_section.to(dtype=hidden_states.dtype)
    stats = {
        "status": "ok",
        "operator": RESIDUAL_PATH_OPERATOR,
        "transport": str(transport),
        "weight": float(weight),
        "token_count": int(len(query_positions)),
        "delta_norm_raw": float(torch.linalg.norm(scaled_delta).item()),
    }
    return patched, stats


__all__ = [
    "ADD_DIRECTION_MODE_ID",
    "ADD_DIRECTION_OPERATOR",
    "INTERCHANGE_MODE_ID",
    "INTERCHANGE_OPERATOR",
    "PROJECT_OUT_MODE_ID",
    "PROJECT_OUT_OPERATOR",
    "RANDOM_CONTROL_MODE_ID",
    "RANDOM_CONTROL_OPERATOR",
    "RESIDUAL_PATH_MODE_ID",
    "RESIDUAL_PATH_OPERATOR",
    "SWAP_COMPONENTS_MODE_ID",
    "SWAP_COMPONENTS_OPERATOR",
    "SWAP_MEAN_MODE_ID",
    "SWAP_MEAN_OPERATOR",
    "apply_interchange",
    "apply_residual_path_transport",
    "apply_subspace_operator",
    "compute_section_state",
    "operator_from_mode_id",
    "operator_mode_id",
    "summarize_subspace_patch",
]
