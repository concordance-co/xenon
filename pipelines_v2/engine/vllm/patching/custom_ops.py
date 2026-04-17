"""Custom-op registration and dispatch for activation patching."""

from __future__ import annotations

from typing import Any

import torch

from ..activation_patch_math import (
    ADD_DIRECTION_MODE_ID,
    PROJECT_OUT_MODE_ID,
    RANDOM_CONTROL_MODE_ID,
    RESIDUAL_PATH_MODE_ID,
    SWAP_COMPONENTS_MODE_ID,
    SWAP_MEAN_MODE_ID,
)
from .subspace_family import (
    SUBSPACE_OPERATOR_MODE_IDS,
    is_subspace_mode_id,
)


_TORCH_LIBRARY_INTERCHANGE_BATCH_REGISTERED = False
_TORCH_LIBRARY_RESIDUAL_PATH_BATCH_REGISTERED = False
_TORCH_LIBRARY_SUBSPACE_REGISTERED = False
_TORCH_LIBRARY_SUBSPACE_BATCH_REGISTERED = False
_TORCH_LIBRARY_PROJECT_OUT_ALIAS_REGISTERED = False
_TORCH_LIBRARY_PROJECT_OUT_BATCH_ALIAS_REGISTERED = False


def _valid_subspace_mode_mask(batch_mode_ids: torch.Tensor) -> torch.Tensor:
    valid = torch.zeros_like(batch_mode_ids, dtype=torch.bool)
    for mode_id in SUBSPACE_OPERATOR_MODE_IDS:
        valid = valid | (batch_mode_ids == int(mode_id))
    return valid


def _apply_subspace_batch_tensorized(
    *,
    hidden_states: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    safe_scale: torch.Tensor,
    batch_mode_ids: torch.Tensor,
    batch_selected_rows: torch.Tensor,
    batch_row_counts: torch.Tensor,
    batch_token_spans: torch.Tensor,
    batch_strengths: torch.Tensor,
    batch_active: torch.Tensor,
    batch_direction_raw: torch.Tensor,
    batch_direction_std: torch.Tensor,
    batch_donor_means: torch.Tensor,
    batch_random_rows: torch.Tensor,
    batch_match_projected_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_f32 = hidden_states.to(torch.float32)
    mean_f32 = mean.to(torch.float32)
    scale_f32 = scale.to(torch.float32)
    safe_scale_f32 = safe_scale.to(torch.float32)
    slot_count = int(batch_active.shape[0])
    hidden_dim = int(hidden_f32.shape[-1])
    total_tokens = int(hidden_f32.shape[0])

    starts = batch_token_spans[:, 0].to(torch.int64).clamp_(0, total_tokens)
    ends = batch_token_spans[:, 1].to(torch.int64).clamp_(0, total_tokens)
    span_counts = (ends - starts).clamp_min_(0)
    active_mask = batch_active.to(torch.bool)
    valid_mode_mask = _valid_subspace_mode_mask(batch_mode_ids)
    slot_valid = active_mask & valid_mode_mask & (span_counts > 0)

    zero_prefix = torch.zeros((1, hidden_dim), device=hidden_f32.device, dtype=torch.float32)
    prefix_sum = torch.cat((zero_prefix, torch.cumsum(hidden_f32, dim=0)), dim=0)
    section_sums = prefix_sum.index_select(0, ends) - prefix_sum.index_select(0, starts)
    safe_counts = span_counts.clamp_min(1).to(torch.float32).unsqueeze(1)
    mean_before = section_sums / safe_counts
    centered_std_before = (mean_before - mean_f32.unsqueeze(0)) / safe_scale_f32.unsqueeze(0)

    max_rows = int(batch_selected_rows.shape[1])
    row_positions = torch.arange(max_rows, device=batch_selected_rows.device, dtype=batch_row_counts.dtype)
    selected_row_mask = row_positions.unsqueeze(0) < batch_row_counts.unsqueeze(1)
    selected_rows = batch_selected_rows.to(torch.float32) * selected_row_mask.unsqueeze(-1).to(torch.float32)

    coeff_before = torch.einsum("sh,srh->sr", centered_std_before, selected_rows)
    projected_std_before = torch.einsum("sr,srh->sh", coeff_before, selected_rows)
    selected_proj_norm_before = torch.linalg.vector_norm(projected_std_before, dim=1)

    strengths = batch_strengths.to(torch.float32).unsqueeze(1)
    direction_raw_f32 = batch_direction_raw.to(torch.float32)
    direction_std_f32 = batch_direction_std.to(torch.float32)
    donor_means_f32 = batch_donor_means.to(torch.float32)
    random_rows_f32 = batch_random_rows.to(torch.float32) * selected_row_mask.unsqueeze(-1).to(torch.float32)

    row_nonempty = batch_row_counts > 0
    direction_std_available = torch.linalg.vector_norm(direction_std_f32, dim=1) > 0
    direction_raw_available = torch.linalg.vector_norm(direction_raw_f32, dim=1) > 0
    donor_mean_available = torch.linalg.vector_norm(donor_means_f32, dim=1) > 0
    random_rows_available = torch.linalg.vector_norm(random_rows_f32.reshape(slot_count, -1), dim=1) > 0

    mode_project_out = slot_valid & (batch_mode_ids == int(PROJECT_OUT_MODE_ID)) & row_nonempty
    mode_add_direction = slot_valid & (batch_mode_ids == int(ADD_DIRECTION_MODE_ID)) & (
        direction_std_available | direction_raw_available
    )
    mode_swap_mean = slot_valid & (batch_mode_ids == int(SWAP_MEAN_MODE_ID)) & donor_mean_available
    mode_swap_components = (
        slot_valid
        & (batch_mode_ids == int(SWAP_COMPONENTS_MODE_ID))
        & donor_mean_available
        & row_nonempty
    )
    mode_random_control = (
        slot_valid
        & (batch_mode_ids == int(RANDOM_CONTROL_MODE_ID))
        & row_nonempty
        & random_rows_available
    )
    op_valid = (
        mode_project_out
        | mode_add_direction
        | mode_swap_mean
        | mode_swap_components
        | mode_random_control
    )

    donor_centered_std = (donor_means_f32 - mean_f32.unsqueeze(0)) / safe_scale_f32.unsqueeze(0)
    donor_coeff = torch.einsum("sh,srh->sr", donor_centered_std, selected_rows)
    donor_projected_std = torch.einsum("sr,srh->sh", donor_coeff, selected_rows)

    random_coeff = torch.einsum("sh,srh->sr", centered_std_before, random_rows_f32)
    random_projected_std = torch.einsum("sr,srh->sh", random_coeff, random_rows_f32)
    random_proj_norm = torch.linalg.vector_norm(random_projected_std, dim=1)
    match_projected = batch_match_projected_norm.to(torch.bool)
    random_scale_factor = torch.where(
        mode_random_control & match_projected & (selected_proj_norm_before > 0) & (random_proj_norm > 1e-8),
        selected_proj_norm_before / random_proj_norm.clamp_min(1e-8),
        torch.ones_like(random_proj_norm),
    )
    random_projected_std = random_projected_std * random_scale_factor.unsqueeze(1)

    zero_delta = torch.zeros_like(mean_before)
    delta_raw = zero_delta

    project_out_delta = (-strengths * projected_std_before) * scale_f32.unsqueeze(0)
    delta_raw = torch.where(mode_project_out.unsqueeze(1), project_out_delta, delta_raw)

    add_direction_delta = torch.where(
        direction_std_available.unsqueeze(1),
        direction_std_f32 * scale_f32.unsqueeze(0),
        direction_raw_f32,
    ) * strengths
    delta_raw = torch.where(mode_add_direction.unsqueeze(1), add_direction_delta, delta_raw)

    swap_mean_delta = strengths * (donor_means_f32 - mean_before)
    delta_raw = torch.where(mode_swap_mean.unsqueeze(1), swap_mean_delta, delta_raw)

    swap_components_delta = strengths * (donor_projected_std - projected_std_before) * scale_f32.unsqueeze(0)
    delta_raw = torch.where(mode_swap_components.unsqueeze(1), swap_components_delta, delta_raw)

    random_control_delta = (-strengths * random_projected_std) * scale_f32.unsqueeze(0)
    delta_raw = torch.where(mode_random_control.unsqueeze(1), random_control_delta, delta_raw)

    patched_mean = mean_before + delta_raw
    patched_centered_std = centered_std_before + (delta_raw / safe_scale_f32.unsqueeze(0))
    coeff_after = torch.einsum("sh,srh->sr", patched_centered_std, selected_rows)

    delta_norm_raw = torch.linalg.vector_norm(delta_raw, dim=1)
    delta_norm_std = torch.linalg.vector_norm(delta_raw / safe_scale_f32.unsqueeze(0), dim=1)
    mean_norm_before = torch.linalg.vector_norm(mean_before, dim=1)
    mean_norm_after = torch.linalg.vector_norm(patched_mean, dim=1)
    mean_std_norm_before = torch.linalg.vector_norm(centered_std_before, dim=1)
    mean_std_norm_after = torch.linalg.vector_norm(patched_centered_std, dim=1)
    direction_norm_raw = torch.where(
        direction_std_available,
        torch.linalg.vector_norm(direction_std_f32 * scale_f32.unsqueeze(0), dim=1),
        torch.linalg.vector_norm(direction_raw_f32, dim=1),
    )
    extra_scalar = torch.where(mode_add_direction, direction_norm_raw, selected_proj_norm_before)

    scalars = torch.stack(
        (
            delta_norm_raw,
            delta_norm_std,
            mean_norm_before,
            mean_norm_after,
            mean_std_norm_before,
            mean_std_norm_after,
            extra_scalar,
            batch_strengths.to(torch.float32),
        ),
        dim=1,
    )

    delta_markers = torch.zeros((total_tokens + 1, hidden_dim), device=hidden_f32.device, dtype=torch.float32)
    active_delta = delta_raw * op_valid.unsqueeze(1).to(torch.float32)
    delta_markers.index_add_(0, starts, active_delta)
    delta_markers.index_add_(0, ends, -active_delta)
    token_delta = torch.cumsum(delta_markers[:-1], dim=0)
    patched_hidden = hidden_states + token_delta.to(hidden_states.dtype)
    return patched_hidden, op_valid, scalars, coeff_before, coeff_after


def register_torch_library_interchange_batch_op() -> None:
    global _TORCH_LIBRARY_INTERCHANGE_BATCH_REGISTERED
    if _TORCH_LIBRARY_INTERCHANGE_BATCH_REGISTERED:
        return

    try:
        custom_op = torch.library.custom_op
    except Exception:
        return

    @custom_op(
        "xenon_activation_patch_v2::interchange_batch",
        mutates_args={"stats_valid", "stats_scalars"},
    )
    def _interchange_batch_op(
        hidden_states: torch.Tensor,
        batch_query_positions: torch.Tensor,
        batch_donor_rows: torch.Tensor,
        batch_token_counts: torch.Tensor,
        batch_active: torch.Tensor,
        stats_valid: torch.Tensor,
        stats_scalars: torch.Tensor,
    ) -> torch.Tensor:
        patched = hidden_states.clone()
        num_hidden_tokens = hidden_states.shape[0]
        slot_count = batch_active.shape[0]
        max_patch_tokens = batch_query_positions.shape[1]

        row_positions = torch.arange(max_patch_tokens, device=batch_query_positions.device, dtype=batch_token_counts.dtype)
        row_mask = row_positions.unsqueeze(0) < batch_token_counts.unsqueeze(1)
        active_mask = batch_active.to(torch.bool)
        in_bounds = (batch_query_positions >= 0) & (batch_query_positions < num_hidden_tokens)
        slot_valid = active_mask & (batch_token_counts > 0) & torch.all(in_bounds | ~row_mask, dim=1)
        apply_mask = row_mask & slot_valid.unsqueeze(1)
        safe_positions = torch.where(
            apply_mask,
            batch_query_positions,
            torch.zeros_like(batch_query_positions),
        ).to(torch.int64)

        hidden_f32 = hidden_states.to(torch.float32)
        donor_f32 = batch_donor_rows.to(torch.float32)
        gathered_before = hidden_f32.index_select(0, safe_positions.reshape(-1)).reshape(slot_count, max_patch_tokens, hidden_f32.shape[1])
        masked_delta = (donor_f32 - gathered_before) * apply_mask.unsqueeze(-1).to(torch.float32)

        scatter_positions = safe_positions.reshape(-1, 1).expand(-1, hidden_states.shape[1])
        scatter_delta = masked_delta.reshape(-1, hidden_states.shape[1]).to(hidden_states.dtype)
        patched_delta = torch.zeros_like(hidden_states)
        patched_delta.scatter_add_(0, scatter_positions, scatter_delta)
        patched = hidden_states + patched_delta
        stats_valid.copy_(torch.maximum(stats_valid, slot_valid.to(stats_valid.dtype)))
        stats_scalars[:, 0].copy_(
            torch.where(
                slot_valid,
                torch.linalg.norm(masked_delta.reshape(slot_count, -1), dim=1).to(stats_scalars.dtype),
                stats_scalars[:, 0],
            )
        )
        stats_scalars[:, 1].copy_(
            torch.where(
                slot_valid,
                apply_mask.to(torch.float32).sum(dim=1).to(stats_scalars.dtype),
                stats_scalars[:, 1],
            )
        )
        return patched

    @_interchange_batch_op.register_fake
    def _interchange_batch_op_fake(
        hidden_states: torch.Tensor,
        batch_query_positions: torch.Tensor,
        batch_donor_rows: torch.Tensor,
        batch_token_counts: torch.Tensor,
        batch_active: torch.Tensor,
        stats_valid: torch.Tensor,
        stats_scalars: torch.Tensor,
    ) -> torch.Tensor:
        del batch_query_positions, batch_donor_rows, batch_token_counts, batch_active, stats_valid, stats_scalars
        return torch.empty_like(hidden_states)

    _TORCH_LIBRARY_INTERCHANGE_BATCH_REGISTERED = True


def register_torch_library_residual_path_batch_op() -> None:
    global _TORCH_LIBRARY_RESIDUAL_PATH_BATCH_REGISTERED
    if _TORCH_LIBRARY_RESIDUAL_PATH_BATCH_REGISTERED:
        return

    try:
        custom_op = torch.library.custom_op
    except Exception:
        return

    @custom_op(
        "xenon_activation_patch_v2::residual_path_batch",
        mutates_args={"stats_valid", "stats_scalars"},
    )
    def _residual_path_batch_op(
        hidden_states: torch.Tensor,
        batch_query_positions: torch.Tensor,
        batch_payload_rows: torch.Tensor,
        batch_token_counts: torch.Tensor,
        batch_transport_modes: torch.Tensor,
        batch_replace_alphas: torch.Tensor,
        batch_active: torch.Tensor,
        stats_valid: torch.Tensor,
        stats_scalars: torch.Tensor,
    ) -> torch.Tensor:
        patched = hidden_states.clone()
        num_hidden_tokens = hidden_states.shape[0]
        slot_count = batch_active.shape[0]
        max_patch_tokens = batch_query_positions.shape[1]

        row_positions = torch.arange(
            max_patch_tokens,
            device=batch_query_positions.device,
            dtype=batch_token_counts.dtype,
        )
        row_mask = row_positions.unsqueeze(0) < batch_token_counts.unsqueeze(1)
        active_mask = batch_active.to(torch.bool)
        in_bounds = (batch_query_positions >= 0) & (batch_query_positions < num_hidden_tokens)
        slot_valid = active_mask & (batch_token_counts > 0) & torch.all(in_bounds | ~row_mask, dim=1)
        apply_mask = row_mask & slot_valid.unsqueeze(1)
        safe_positions = torch.where(
            apply_mask,
            batch_query_positions,
            torch.zeros_like(batch_query_positions),
        ).to(torch.int64)

        hidden_f32 = hidden_states.to(torch.float32)
        payload_f32 = batch_payload_rows.to(torch.float32)
        gathered_before = hidden_f32.index_select(0, safe_positions.reshape(-1)).reshape(
            slot_count,
            max_patch_tokens,
            hidden_f32.shape[1],
        )
        transport_is_replace = batch_transport_modes.to(torch.bool).unsqueeze(1).unsqueeze(2)
        replace_alphas = batch_replace_alphas.to(torch.float32).unsqueeze(1).unsqueeze(2)
        delta_rows = torch.where(
            transport_is_replace,
            payload_f32 - (replace_alphas * gathered_before),
            payload_f32,
        )
        masked_delta = delta_rows * apply_mask.unsqueeze(-1).to(torch.float32)

        scatter_positions = safe_positions.reshape(-1, 1).expand(-1, hidden_states.shape[1])
        scatter_delta = masked_delta.reshape(-1, hidden_states.shape[1]).to(hidden_states.dtype)
        patched_delta = torch.zeros_like(hidden_states)
        patched_delta.scatter_add_(0, scatter_positions, scatter_delta)
        patched = hidden_states + patched_delta

        stats_valid.copy_(torch.maximum(stats_valid, slot_valid.to(stats_valid.dtype)))
        stats_scalars[:, 0].copy_(
            torch.where(
                slot_valid,
                torch.linalg.norm(masked_delta.reshape(slot_count, -1), dim=1).to(stats_scalars.dtype),
                stats_scalars[:, 0],
            )
        )
        stats_scalars[:, 1].copy_(
            torch.where(
                slot_valid,
                apply_mask.to(torch.float32).sum(dim=1).to(stats_scalars.dtype),
                stats_scalars[:, 1],
            )
        )
        stats_scalars[:, 2].copy_(
            torch.where(
                slot_valid,
                batch_replace_alphas.to(stats_scalars.dtype),
                stats_scalars[:, 2],
            )
        )
        return patched

    @_residual_path_batch_op.register_fake
    def _residual_path_batch_op_fake(
        hidden_states: torch.Tensor,
        batch_query_positions: torch.Tensor,
        batch_payload_rows: torch.Tensor,
        batch_token_counts: torch.Tensor,
        batch_transport_modes: torch.Tensor,
        batch_replace_alphas: torch.Tensor,
        batch_active: torch.Tensor,
        stats_valid: torch.Tensor,
        stats_scalars: torch.Tensor,
    ) -> torch.Tensor:
        del (
            batch_query_positions,
            batch_payload_rows,
            batch_token_counts,
            batch_transport_modes,
            batch_replace_alphas,
            batch_active,
            stats_valid,
            stats_scalars,
        )
        return torch.empty_like(hidden_states)

    _TORCH_LIBRARY_RESIDUAL_PATH_BATCH_REGISTERED = True


def register_torch_library_subspace_op() -> None:
    global _TORCH_LIBRARY_SUBSPACE_REGISTERED
    if _TORCH_LIBRARY_SUBSPACE_REGISTERED:
        return

    try:
        custom_op = torch.library.custom_op
    except Exception:
        return

    @custom_op("xenon_activation_patch_v2::subspace", mutates_args=())
    def _subspace_op(
        hidden_states: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        safe_scale: torch.Tensor,
        selected_rows: torch.Tensor,
        token_span: torch.Tensor,
        mode_id: int,
        strength: float,
        direction_raw: torch.Tensor,
        direction_std: torch.Tensor,
        donor_mean: torch.Tensor,
        random_rows: torch.Tensor,
        match_projected_norm: torch.Tensor,
    ) -> torch.Tensor:
        hidden_dim = int(hidden_states.shape[-1])
        row_count = int(selected_rows.shape[0])
        patched, _, _, _, _ = _apply_subspace_batch_tensorized(
            hidden_states=hidden_states,
            mean=mean,
            scale=scale,
            safe_scale=safe_scale,
            batch_mode_ids=torch.full((1,), int(mode_id), device=hidden_states.device, dtype=torch.int32),
            batch_selected_rows=selected_rows.unsqueeze(0),
            batch_row_counts=torch.full((1,), row_count, device=hidden_states.device, dtype=torch.int32),
            batch_token_spans=token_span.reshape(1, 2).to(device=hidden_states.device, dtype=torch.int32),
            batch_strengths=torch.full((1,), float(strength), device=hidden_states.device, dtype=torch.float32),
            batch_active=torch.ones((1,), device=hidden_states.device, dtype=torch.int32),
            batch_direction_raw=direction_raw.reshape(1, hidden_dim).to(device=hidden_states.device, dtype=torch.float32),
            batch_direction_std=direction_std.reshape(1, hidden_dim).to(device=hidden_states.device, dtype=torch.float32),
            batch_donor_means=donor_mean.reshape(1, hidden_dim).to(device=hidden_states.device, dtype=torch.float32),
            batch_random_rows=random_rows.unsqueeze(0).to(device=hidden_states.device, dtype=torch.float32),
            batch_match_projected_norm=match_projected_norm.reshape(1).to(device=hidden_states.device, dtype=torch.int32),
        )
        return patched

    @_subspace_op.register_fake
    def _subspace_op_fake(
        hidden_states: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        safe_scale: torch.Tensor,
        selected_rows: torch.Tensor,
        token_span: torch.Tensor,
        mode_id: int,
        strength: float,
        direction_raw: torch.Tensor,
        direction_std: torch.Tensor,
        donor_mean: torch.Tensor,
        random_rows: torch.Tensor,
        match_projected_norm: torch.Tensor,
    ) -> torch.Tensor:
        del (
            mean,
            scale,
            safe_scale,
            selected_rows,
            token_span,
            mode_id,
            strength,
            direction_raw,
            direction_std,
            donor_mean,
            random_rows,
            match_projected_norm,
        )
        return torch.empty_like(hidden_states)

    _TORCH_LIBRARY_SUBSPACE_REGISTERED = True


def register_torch_library_subspace_batch_op() -> None:
    global _TORCH_LIBRARY_SUBSPACE_BATCH_REGISTERED
    if _TORCH_LIBRARY_SUBSPACE_BATCH_REGISTERED:
        return

    try:
        custom_op = torch.library.custom_op
    except Exception:
        return

    @custom_op(
        "xenon_activation_patch_v2::subspace_batch",
        mutates_args={"stats_valid", "stats_scalars", "stats_coeff_before", "stats_coeff_after"},
    )
    def _subspace_batch_op(
        hidden_states: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        safe_scale: torch.Tensor,
        batch_mode_ids: torch.Tensor,
        batch_selected_rows: torch.Tensor,
        batch_row_counts: torch.Tensor,
        batch_token_spans: torch.Tensor,
        batch_strengths: torch.Tensor,
        batch_active: torch.Tensor,
        stats_valid: torch.Tensor,
        stats_scalars: torch.Tensor,
        stats_coeff_before: torch.Tensor,
        stats_coeff_after: torch.Tensor,
        batch_direction_raw: torch.Tensor,
        batch_direction_std: torch.Tensor,
        batch_donor_means: torch.Tensor,
        batch_random_rows: torch.Tensor,
        batch_match_projected_norm: torch.Tensor,
    ) -> torch.Tensor:
        patched_hidden, op_valid, scalars, coeff_before, coeff_after = _apply_subspace_batch_tensorized(
            hidden_states=hidden_states,
            mean=mean,
            scale=scale,
            safe_scale=safe_scale,
            batch_mode_ids=batch_mode_ids,
            batch_selected_rows=batch_selected_rows,
            batch_row_counts=batch_row_counts,
            batch_token_spans=batch_token_spans,
            batch_strengths=batch_strengths,
            batch_active=batch_active,
            batch_direction_raw=batch_direction_raw,
            batch_direction_std=batch_direction_std,
            batch_donor_means=batch_donor_means,
            batch_random_rows=batch_random_rows,
            batch_match_projected_norm=batch_match_projected_norm,
        )
        valid_mask = op_valid.unsqueeze(1)
        stats_valid.copy_(torch.maximum(stats_valid, op_valid.to(stats_valid.dtype)))
        stats_scalars.copy_(
            torch.where(
                valid_mask,
                scalars.to(device=stats_scalars.device, dtype=stats_scalars.dtype),
                stats_scalars,
            )
        )
        coeff_before_cast = coeff_before.to(device=stats_coeff_before.device, dtype=stats_coeff_before.dtype)
        coeff_after_cast = coeff_after.to(device=stats_coeff_after.device, dtype=stats_coeff_after.dtype)
        stats_coeff_before.copy_(torch.where(valid_mask, coeff_before_cast, stats_coeff_before))
        stats_coeff_after.copy_(torch.where(valid_mask, coeff_after_cast, stats_coeff_after))
        return patched_hidden

    @_subspace_batch_op.register_fake
    def _subspace_batch_op_fake(
        hidden_states: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        safe_scale: torch.Tensor,
        batch_mode_ids: torch.Tensor,
        batch_selected_rows: torch.Tensor,
        batch_row_counts: torch.Tensor,
        batch_token_spans: torch.Tensor,
        batch_strengths: torch.Tensor,
        batch_active: torch.Tensor,
        stats_valid: torch.Tensor,
        stats_scalars: torch.Tensor,
        stats_coeff_before: torch.Tensor,
        stats_coeff_after: torch.Tensor,
        batch_direction_raw: torch.Tensor,
        batch_direction_std: torch.Tensor,
        batch_donor_means: torch.Tensor,
        batch_random_rows: torch.Tensor,
        batch_match_projected_norm: torch.Tensor,
    ) -> torch.Tensor:
        del (
            mean,
            scale,
            safe_scale,
            batch_mode_ids,
            batch_selected_rows,
            batch_row_counts,
            batch_token_spans,
            batch_strengths,
            batch_active,
            stats_valid,
            stats_scalars,
            stats_coeff_before,
            stats_coeff_after,
            batch_direction_raw,
            batch_direction_std,
            batch_donor_means,
            batch_random_rows,
            batch_match_projected_norm,
        )
        return torch.empty_like(hidden_states)

    _TORCH_LIBRARY_SUBSPACE_BATCH_REGISTERED = True


def register_torch_library_project_out_op() -> None:
    global _TORCH_LIBRARY_PROJECT_OUT_ALIAS_REGISTERED
    register_torch_library_subspace_op()
    if _TORCH_LIBRARY_PROJECT_OUT_ALIAS_REGISTERED:
        return

    try:
        custom_op = torch.library.custom_op
    except Exception:
        return

    @custom_op("xenon_activation_patch_v2::project_out", mutates_args=())
    def _project_out_op_alias(
        hidden_states: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        safe_scale: torch.Tensor,
        selected_rows: torch.Tensor,
        token_span: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        namespace = getattr(torch.ops, "xenon_activation_patch_v2", None)
        return namespace.subspace(
            hidden_states,
            mean,
            scale,
            safe_scale,
            selected_rows,
            token_span,
            1,
            strength,
            torch.zeros((hidden_states.shape[-1],), device=hidden_states.device, dtype=torch.float32),
            torch.zeros((hidden_states.shape[-1],), device=hidden_states.device, dtype=torch.float32),
            torch.zeros((hidden_states.shape[-1],), device=hidden_states.device, dtype=torch.float32),
            torch.zeros((1, hidden_states.shape[-1]), device=hidden_states.device, dtype=torch.float32),
            torch.ones((1,), device=hidden_states.device, dtype=torch.int32),
        )

    @_project_out_op_alias.register_fake
    def _project_out_op_alias_fake(
        hidden_states: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        safe_scale: torch.Tensor,
        selected_rows: torch.Tensor,
        token_span: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        del mean, scale, safe_scale, selected_rows, token_span, strength
        return torch.empty_like(hidden_states)

    _TORCH_LIBRARY_PROJECT_OUT_ALIAS_REGISTERED = True


def register_torch_library_project_out_batch_op() -> None:
    global _TORCH_LIBRARY_PROJECT_OUT_BATCH_ALIAS_REGISTERED
    register_torch_library_subspace_batch_op()
    if _TORCH_LIBRARY_PROJECT_OUT_BATCH_ALIAS_REGISTERED:
        return

    try:
        custom_op = torch.library.custom_op
    except Exception:
        return

    @custom_op(
        "xenon_activation_patch_v2::project_out_batch",
        mutates_args={"stats_valid", "stats_scalars", "stats_coeff_before", "stats_coeff_after"},
    )
    def _project_out_batch_op_alias(
        hidden_states: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        safe_scale: torch.Tensor,
        batch_selected_rows: torch.Tensor,
        batch_row_counts: torch.Tensor,
        batch_token_spans: torch.Tensor,
        batch_strengths: torch.Tensor,
        batch_active: torch.Tensor,
        stats_valid: torch.Tensor,
        stats_scalars: torch.Tensor,
        stats_coeff_before: torch.Tensor,
        stats_coeff_after: torch.Tensor,
    ) -> torch.Tensor:
        namespace = getattr(torch.ops, "xenon_activation_patch_v2", None)
        batch_mode_ids = torch.full_like(batch_active, 1)
        hidden_dim = int(hidden_states.shape[-1])
        return namespace.subspace_batch(
            hidden_states,
            mean,
            scale,
            safe_scale,
            batch_mode_ids,
            batch_selected_rows,
            batch_row_counts,
            batch_token_spans,
            batch_strengths,
            batch_active,
            stats_valid,
            stats_scalars,
            stats_coeff_before,
            stats_coeff_after,
            torch.zeros((batch_active.shape[0], hidden_dim), device=hidden_states.device, dtype=torch.float32),
            torch.zeros((batch_active.shape[0], hidden_dim), device=hidden_states.device, dtype=torch.float32),
            torch.zeros((batch_active.shape[0], hidden_dim), device=hidden_states.device, dtype=torch.float32),
            torch.zeros((batch_active.shape[0], batch_selected_rows.shape[1], hidden_dim), device=hidden_states.device, dtype=torch.float32),
            torch.ones((batch_active.shape[0],), device=hidden_states.device, dtype=torch.int32),
        )

    @_project_out_batch_op_alias.register_fake
    def _project_out_batch_op_alias_fake(
        hidden_states: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        safe_scale: torch.Tensor,
        batch_selected_rows: torch.Tensor,
        batch_row_counts: torch.Tensor,
        batch_token_spans: torch.Tensor,
        batch_strengths: torch.Tensor,
        batch_active: torch.Tensor,
        stats_valid: torch.Tensor,
        stats_scalars: torch.Tensor,
        stats_coeff_before: torch.Tensor,
        stats_coeff_after: torch.Tensor,
    ) -> torch.Tensor:
        del (
            mean,
            scale,
            safe_scale,
            batch_selected_rows,
            batch_row_counts,
            batch_token_spans,
            batch_strengths,
            batch_active,
            stats_valid,
            stats_scalars,
            stats_coeff_before,
            stats_coeff_after,
        )
        return torch.empty_like(hidden_states)

    _TORCH_LIBRARY_PROJECT_OUT_BATCH_ALIAS_REGISTERED = True


def flatten_hidden_for_patch(hidden: Any) -> tuple[Any, Any]:
    if hidden.ndim <= 2:
        return hidden, lambda patched: patched
    original_shape = tuple(hidden.shape)
    hidden_dim = int(original_shape[-1])
    flat_hidden = hidden.reshape(-1, hidden_dim)
    return flat_hidden, lambda patched: patched.reshape(original_shape)


def noop_custom_op(hidden: Any, *, custom_op: Any, operator_id: int) -> Any:
    flat_hidden, restore_hidden = flatten_hidden_for_patch(hidden)
    hidden_dim = int(flat_hidden.shape[-1]) if flat_hidden.ndim >= 2 else 1
    device = flat_hidden.device
    patched = custom_op(
        flat_hidden,
        int(operator_id),
        torch.zeros((hidden_dim,), device=device, dtype=torch.float32),
        torch.ones((hidden_dim,), device=device, dtype=torch.float32),
        torch.ones((hidden_dim,), device=device, dtype=torch.float32),
        torch.zeros((1, 1), device=device, dtype=torch.int32),
        torch.zeros((1,), device=device, dtype=torch.int32),
        torch.zeros((1, 1, hidden_dim), device=device, dtype=torch.float32),
        torch.zeros((1, 1, hidden_dim), device=device, dtype=torch.float32),
        torch.zeros((1,), device=device, dtype=torch.int32),
        torch.zeros((1,), device=device, dtype=torch.float32),
        torch.zeros((1,), device=device, dtype=torch.int32),
        torch.zeros((1,), device=device, dtype=torch.int32),
        torch.zeros((1, 8), device=device, dtype=torch.float32),
        torch.zeros((1, 1), device=device, dtype=torch.float32),
        torch.zeros((1, 1), device=device, dtype=torch.float32),
    )
    return restore_hidden(patched)


def run_custom_op(
    *,
    hidden: Any,
    custom_op: Any,
    operator_id: int,
    mean: Any,
    scale: Any,
    safe_scale: Any,
    query_positions: Any,
    token_counts: Any,
    donor_rows: Any,
    selected_rows: Any,
    row_counts: Any,
    strengths: Any,
    active: Any,
    stats_valid: Any,
    stats_scalars: Any,
    stats_coeff_before: Any,
    stats_coeff_after: Any,
    token_spans: Any | None = None,
    mode_ids: Any | None = None,
    direction_raw: Any | None = None,
    direction_std: Any | None = None,
    donor_means: Any | None = None,
    random_rows: Any | None = None,
    match_projected_norm: Any | None = None,
    residual_path_transport_modes: Any | None = None,
    residual_path_replace_alphas: Any | None = None,
) -> Any:
    flat_hidden, restore_hidden = flatten_hidden_for_patch(hidden)
    if is_subspace_mode_id(int(operator_id)) and token_spans is not None:
        patched = custom_op(
            flat_hidden,
            mean,
            scale,
            safe_scale,
            selected_rows[0, :0],
            token_spans[0],
            int(operator_id),
            1.0,
            direction_raw[0] if direction_raw is not None else torch.zeros((flat_hidden.shape[-1],), device=flat_hidden.device, dtype=torch.float32),
            direction_std[0] if direction_std is not None else torch.zeros((flat_hidden.shape[-1],), device=flat_hidden.device, dtype=torch.float32),
            donor_means[0] if donor_means is not None else torch.zeros((flat_hidden.shape[-1],), device=flat_hidden.device, dtype=torch.float32),
            random_rows[0] if random_rows is not None else torch.zeros((1, flat_hidden.shape[-1]), device=flat_hidden.device, dtype=torch.float32),
            torch.ones((1,), device=flat_hidden.device, dtype=torch.int32),
            batch_selected_rows=selected_rows,
            batch_mode_ids=mode_ids,
            batch_row_counts=row_counts,
            batch_token_spans=token_spans,
            batch_strengths=strengths,
            batch_active=active,
            stats_valid=stats_valid,
            stats_scalars=stats_scalars,
            stats_coeff_before=stats_coeff_before,
            stats_coeff_after=stats_coeff_after,
            batch_direction_raw=direction_raw,
            batch_direction_std=direction_std,
            batch_donor_means=donor_means,
            batch_random_rows=random_rows,
            batch_match_projected_norm=match_projected_norm,
        )
        return restore_hidden(patched)
    extra_kwargs: dict[str, Any] = {}
    if residual_path_transport_modes is not None:
        extra_kwargs["batch_residual_path_transport_modes"] = residual_path_transport_modes
    if residual_path_replace_alphas is not None:
        extra_kwargs["batch_residual_path_replace_alphas"] = residual_path_replace_alphas
    patched = custom_op(
        flat_hidden,
        int(operator_id),
        mean,
        scale,
        safe_scale,
        query_positions,
        token_counts,
        donor_rows,
        selected_rows,
        row_counts,
        strengths,
        active,
        stats_valid,
        stats_scalars,
        stats_coeff_before,
        stats_coeff_after,
        **extra_kwargs,
    )
    return restore_hidden(patched)


__all__ = [
    "noop_custom_op",
    "register_torch_library_interchange_batch_op",
    "register_torch_library_residual_path_batch_op",
    "register_torch_library_subspace_batch_op",
    "register_torch_library_subspace_op",
    "register_torch_library_project_out_batch_op",
    "register_torch_library_project_out_op",
    "run_custom_op",
]
