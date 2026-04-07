from __future__ import annotations

from typing import Any

import torch

PATCH_MODE_PROJECT_OUT = "project_out"
PATCH_MODE_ADD_DIRECTION = "add_direction"
PATCH_MODE_SWAP_MEAN = "swap_mean"
PATCH_MODE_SWAP_COMPONENTS = "swap_components"
PATCH_MODE_RANDOM_CONTROL = "random_control"

PATCH_MODE_TO_ID = {
    PATCH_MODE_PROJECT_OUT: 0,
    PATCH_MODE_ADD_DIRECTION: 1,
    PATCH_MODE_SWAP_MEAN: 2,
    PATCH_MODE_SWAP_COMPONENTS: 3,
    PATCH_MODE_RANDOM_CONTROL: 4,
}
PATCH_ID_TO_MODE = {value: key for key, value in PATCH_MODE_TO_ID.items()}

_TORCH_LIBRARY_PROJECT_OUT_REGISTERED = False
_TORCH_LIBRARY_PROJECT_OUT_BATCH_REGISTERED = False
_TORCH_LIBRARY_SWAP_COMPONENTS_BATCH_REGISTERED = False


def patch_mode_to_id(mode: str) -> int:
    try:
        return PATCH_MODE_TO_ID[str(mode)]
    except KeyError as exc:
        raise ValueError(f"Unsupported patch mode: {mode}") from exc


def patch_mode_from_id(mode_id: int) -> str:
    try:
        return PATCH_ID_TO_MODE[int(mode_id)]
    except KeyError as exc:
        raise ValueError(f"Unsupported patch mode id: {mode_id}") from exc


def _empty_like_1d(reference: torch.Tensor) -> torch.Tensor:
    return torch.empty((0,), device=reference.device, dtype=torch.float32)


def _compute_section_state(
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
        selected_coeff = _empty_like_1d(centered_std)
        selected_projected_std = torch.zeros_like(centered_std)
        selected_proj_norm = 0.0
    return {
        "mu": mu,
        "centered_std": centered_std,
        "selected_coeff": selected_coeff,
        "selected_projected_std": selected_projected_std,
        "selected_proj_norm": selected_proj_norm,
    }


def summarize_activation_patch_tensor(
    original_hidden: torch.Tensor,
    patched_hidden: torch.Tensor,
    *,
    mean: torch.Tensor,
    safe_scale: torch.Tensor,
    selected_rows: torch.Tensor,
    token_span: tuple[int, int],
    strength: float,
) -> dict[str, Any]:
    start, end = int(token_span[0]), int(token_span[1])
    before = _compute_section_state(
        original_hidden[start:end].to(torch.float32),
        mean=mean,
        safe_scale=safe_scale,
        selected_rows=selected_rows,
    )
    after = _compute_section_state(
        patched_hidden[start:end].to(torch.float32),
        mean=mean,
        safe_scale=safe_scale,
        selected_rows=selected_rows,
    )
    delta_raw = (patched_hidden[start:end].to(torch.float32) - original_hidden[start:end].to(torch.float32)).mean(dim=0)
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
        "strength": float(strength),
    }


def summarize_activation_patch_tensor_tensors(
    original_hidden: torch.Tensor,
    patched_hidden: torch.Tensor,
    *,
    mean: torch.Tensor,
    safe_scale: torch.Tensor,
    selected_rows: torch.Tensor,
    token_span: tuple[int, int],
    strength: float,
) -> dict[str, torch.Tensor]:
    start, end = int(token_span[0]), int(token_span[1])
    before = _compute_section_state(
        original_hidden[start:end].to(torch.float32),
        mean=mean,
        safe_scale=safe_scale,
        selected_rows=selected_rows,
    )
    after = _compute_section_state(
        patched_hidden[start:end].to(torch.float32),
        mean=mean,
        safe_scale=safe_scale,
        selected_rows=selected_rows,
    )
    delta_raw = (
        patched_hidden[start:end].to(torch.float32)
        - original_hidden[start:end].to(torch.float32)
    ).mean(dim=0)
    return {
        "scalars": torch.stack(
            (
                torch.linalg.norm(delta_raw),
                torch.linalg.norm(delta_raw / safe_scale),
                torch.linalg.norm(before["mu"]),
                torch.linalg.norm(after["mu"]),
                torch.linalg.norm(before["centered_std"]),
                torch.linalg.norm(after["centered_std"]),
                torch.as_tensor(before["selected_proj_norm"], device=delta_raw.device, dtype=torch.float32),
                torch.as_tensor(float(strength), device=delta_raw.device, dtype=torch.float32),
            )
        ),
        "selected_coeff_before": before["selected_coeff"].to(torch.float32),
        "selected_coeff_after": after["selected_coeff"].to(torch.float32),
    }


def apply_activation_patch_tensor(
    hidden_states: torch.Tensor,
    *,
    mean: torch.Tensor,
    scale: torch.Tensor,
    safe_scale: torch.Tensor,
    selected_rows: torch.Tensor,
    token_span: tuple[int, int],
    mode: str,
    strength: float,
    direction_std: torch.Tensor | None = None,
    donor_mean: torch.Tensor | None = None,
    random_rows: torch.Tensor | None = None,
    collect_stats: bool = True,
) -> tuple[torch.Tensor, dict[str, Any]]:
    original_dtype = hidden_states.dtype
    start, end = int(token_span[0]), int(token_span[1])
    hidden_f32 = hidden_states.to(torch.float32)
    section = hidden_f32[start:end]

    before = _compute_section_state(
        section,
        mean=mean,
        safe_scale=safe_scale,
        selected_rows=selected_rows,
    )
    mu = before["mu"]
    centered_std = before["centered_std"]
    selected_coeff_before = before["selected_coeff"]
    selected_projected_std = before["selected_projected_std"]
    selected_proj_norm = before["selected_proj_norm"]

    if mode == PATCH_MODE_PROJECT_OUT:
        delta_std = -float(strength) * selected_projected_std
        patched_section = section + (delta_std * scale)
    elif mode == PATCH_MODE_ADD_DIRECTION:
        if direction_std is None:
            raise ValueError("direction_std is required for add_direction")
        delta_std = float(strength) * direction_std
        patched_section = section + (delta_std * scale)
    elif mode == PATCH_MODE_SWAP_MEAN:
        if donor_mean is None:
            raise ValueError("donor_mean is required for swap_mean")
        delta_raw = float(strength) * (donor_mean - mu)
        patched_section = section + delta_raw
        delta_std = delta_raw / safe_scale
    elif mode == PATCH_MODE_SWAP_COMPONENTS:
        if donor_mean is None:
            raise ValueError("donor_mean is required for swap_components")
        donor_centered_std = (donor_mean - mean) / safe_scale
        if selected_rows.numel() > 0:
            donor_selected_coeff = donor_centered_std @ selected_rows.T
            donor_selected_projected_std = donor_selected_coeff @ selected_rows
        else:
            donor_selected_projected_std = torch.zeros_like(centered_std)
        delta_std = float(strength) * (donor_selected_projected_std - selected_projected_std)
        patched_section = section + (delta_std * scale)
    elif mode == PATCH_MODE_RANDOM_CONTROL:
        if random_rows is None:
            raise ValueError("random_rows is required for random_control")
        random_coeff = centered_std @ random_rows.T
        random_projected_std = random_coeff @ random_rows
        if selected_proj_norm > 0.0:
            random_norm = float(torch.linalg.norm(random_projected_std).item())
            if random_norm > 1e-8:
                random_projected_std = random_projected_std * (selected_proj_norm / random_norm)
        delta_std = -float(strength) * random_projected_std
        patched_section = section + (delta_std * scale)
    else:
        raise ValueError(f"Unsupported patch mode: {mode}")

    patched_hidden = hidden_f32.clone()
    patched_hidden[start:end] = patched_section

    stats: dict[str, Any] = {}
    if collect_stats:
        stats = summarize_activation_patch_tensor(
            hidden_f32,
            patched_hidden,
            mean=mean,
            safe_scale=safe_scale,
            selected_rows=selected_rows,
            token_span=(start, end),
            strength=float(strength),
        )

    return patched_hidden.to(original_dtype), stats


def _register_torch_library_project_out_op() -> None:
    global _TORCH_LIBRARY_PROJECT_OUT_REGISTERED
    if _TORCH_LIBRARY_PROJECT_OUT_REGISTERED:
        return

    try:
        custom_op = torch.library.custom_op
    except Exception:
        return

    @custom_op("xenon_activation_patch::project_out", mutates_args=())
    def _project_out_op(
        hidden_states: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        safe_scale: torch.Tensor,
        selected_rows: torch.Tensor,
        token_span: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        patched, _ = apply_activation_patch_tensor(
            hidden_states,
            mean=mean,
            scale=scale,
            safe_scale=safe_scale,
            selected_rows=selected_rows,
            token_span=(int(token_span[0].item()), int(token_span[1].item())),
            mode=PATCH_MODE_PROJECT_OUT,
            strength=float(strength),
            collect_stats=False,
        )
        return patched

    @_project_out_op.register_fake
    def _project_out_op_fake(
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

    _TORCH_LIBRARY_PROJECT_OUT_REGISTERED = True


_register_torch_library_project_out_op()


def _register_torch_library_project_out_batch_op() -> None:
    global _TORCH_LIBRARY_PROJECT_OUT_BATCH_REGISTERED
    if _TORCH_LIBRARY_PROJECT_OUT_BATCH_REGISTERED:
        return

    try:
        custom_op = torch.library.custom_op
    except Exception:
        return

    @custom_op(
        "xenon_activation_patch::project_out_batch",
        mutates_args={"valid", "scalars", "coeff_before", "coeff_after"},
    )
    def _project_out_batch_op(
        hidden_states: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        safe_scale: torch.Tensor,
        batch_selected_rows: torch.Tensor,
        row_counts: torch.Tensor,
        token_spans: torch.Tensor,
        strengths: torch.Tensor,
        active: torch.Tensor,
        valid: torch.Tensor,
        scalars: torch.Tensor,
        coeff_before: torch.Tensor,
        coeff_after: torch.Tensor,
    ) -> torch.Tensor:
        hidden_f32 = hidden_states.to(torch.float32)
        mean_f32 = mean.to(torch.float32)
        scale_f32 = scale.to(torch.float32)
        safe_scale_f32 = safe_scale.to(torch.float32)
        rows_f32 = batch_selected_rows.to(torch.float32)
        strengths_f32 = strengths.to(torch.float32)
        active_i32 = active.to(torch.int32)
        valid.zero_()
        scalars.zero_()
        coeff_before.zero_()
        coeff_after.zero_()
        num_tokens = hidden_f32.shape[0]
        max_rows = rows_f32.shape[1]

        token_positions = torch.arange(
            num_tokens,
            device=hidden_f32.device,
            dtype=token_spans.dtype,
        )
        token_mask = (
            (token_positions.unsqueeze(0) >= token_spans[:, 0:1])
            & (token_positions.unsqueeze(0) < token_spans[:, 1:2])
            & active_i32.unsqueeze(1).to(torch.bool)
        )
        token_mask_f32 = token_mask.to(torch.float32)
        section_lengths = token_mask_f32.sum(dim=1, keepdim=True)
        safe_section_lengths = torch.clamp(section_lengths, min=1.0)

        row_positions = torch.arange(
            max_rows,
            device=hidden_f32.device,
            dtype=row_counts.dtype,
        )
        row_mask = row_positions.unsqueeze(0) < row_counts.unsqueeze(1)
        row_mask_f32 = row_mask.to(torch.float32)

        mu = (token_mask_f32 @ hidden_f32) / safe_section_lengths
        centered_std = (mu - mean_f32.unsqueeze(0)) / safe_scale_f32.unsqueeze(0)
        coeff_before_all = torch.einsum("sh,skh->sk", centered_std, rows_f32)
        coeff_before_masked = coeff_before_all * row_mask_f32
        projected_std = torch.einsum("sk,skh->sh", coeff_before_masked, rows_f32)
        delta_std = (
            -strengths_f32.unsqueeze(1)
            * projected_std
            * active_i32.unsqueeze(1).to(torch.float32)
        )
        delta_raw = delta_std * scale_f32.unsqueeze(0)
        patched_hidden = hidden_f32 + torch.einsum(
            "st,sh->th",
            token_mask_f32,
            delta_raw,
        )

        patched_mu = mu + delta_raw
        patched_centered_std = (
            patched_mu - mean_f32.unsqueeze(0)
        ) / safe_scale_f32.unsqueeze(0)
        coeff_after_all = torch.einsum("sh,skh->sk", patched_centered_std, rows_f32)
        coeff_after_masked = coeff_after_all * row_mask_f32

        scalars_out = torch.stack(
            (
                torch.linalg.norm(delta_raw, dim=1),
                torch.linalg.norm(delta_std, dim=1),
                torch.linalg.norm(mu, dim=1),
                torch.linalg.norm(patched_mu, dim=1),
                torch.linalg.norm(centered_std, dim=1),
                torch.linalg.norm(patched_centered_std, dim=1),
                torch.linalg.norm(projected_std, dim=1),
                strengths_f32,
            ),
            dim=1,
        )

        valid.copy_(
            (
                active_i32
                * (section_lengths.squeeze(1) > 0).to(torch.int32)
            ).to(valid.dtype)
        )
        scalars.copy_(scalars_out.to(scalars.dtype))
        coeff_before.copy_(coeff_before_masked.to(coeff_before.dtype))
        coeff_after.copy_(coeff_after_masked.to(coeff_after.dtype))

        return patched_hidden.to(hidden_states.dtype)

    @_project_out_batch_op.register_fake
    def _project_out_batch_op_fake(
        hidden_states: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        safe_scale: torch.Tensor,
        batch_selected_rows: torch.Tensor,
        row_counts: torch.Tensor,
        token_spans: torch.Tensor,
        strengths: torch.Tensor,
        active: torch.Tensor,
        valid: torch.Tensor,
        scalars: torch.Tensor,
        coeff_before: torch.Tensor,
        coeff_after: torch.Tensor,
    ) -> torch.Tensor:
        del (
            mean,
            scale,
            safe_scale,
            batch_selected_rows,
            row_counts,
            token_spans,
            strengths,
            active,
            valid,
            scalars,
            coeff_before,
            coeff_after,
        )
        return torch.empty_like(hidden_states)

    _TORCH_LIBRARY_PROJECT_OUT_BATCH_REGISTERED = True


_register_torch_library_project_out_batch_op()


def _register_torch_library_swap_components_batch_op() -> None:
    global _TORCH_LIBRARY_SWAP_COMPONENTS_BATCH_REGISTERED
    if _TORCH_LIBRARY_SWAP_COMPONENTS_BATCH_REGISTERED:
        return

    try:
        custom_op = torch.library.custom_op
    except Exception:
        return

    @custom_op(
        "xenon_activation_patch::swap_components_batch",
        mutates_args={"valid", "scalars", "coeff_before", "coeff_after"},
    )
    def _swap_components_batch_op(
        hidden_states: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        safe_scale: torch.Tensor,
        batch_selected_rows: torch.Tensor,
        batch_donor_means: torch.Tensor,
        row_counts: torch.Tensor,
        token_spans: torch.Tensor,
        strengths: torch.Tensor,
        active: torch.Tensor,
        valid: torch.Tensor,
        scalars: torch.Tensor,
        coeff_before: torch.Tensor,
        coeff_after: torch.Tensor,
    ) -> torch.Tensor:
        hidden_f32 = hidden_states.to(torch.float32)
        mean_f32 = mean.to(torch.float32)
        scale_f32 = scale.to(torch.float32)
        safe_scale_f32 = safe_scale.to(torch.float32)
        rows_f32 = batch_selected_rows.to(torch.float32)
        donor_means_f32 = batch_donor_means.to(torch.float32)
        strengths_f32 = strengths.to(torch.float32)
        active_i32 = active.to(torch.int32)
        valid.zero_()
        scalars.zero_()
        coeff_before.zero_()
        coeff_after.zero_()
        num_tokens = hidden_f32.shape[0]
        max_rows = rows_f32.shape[1]

        token_positions = torch.arange(
            num_tokens,
            device=hidden_f32.device,
            dtype=token_spans.dtype,
        )
        token_mask = (
            (token_positions.unsqueeze(0) >= token_spans[:, 0:1])
            & (token_positions.unsqueeze(0) < token_spans[:, 1:2])
            & active_i32.unsqueeze(1).to(torch.bool)
        )
        token_mask_f32 = token_mask.to(torch.float32)
        section_lengths = token_mask_f32.sum(dim=1, keepdim=True)
        safe_section_lengths = torch.clamp(section_lengths, min=1.0)

        row_positions = torch.arange(
            max_rows,
            device=hidden_f32.device,
            dtype=row_counts.dtype,
        )
        row_mask = row_positions.unsqueeze(0) < row_counts.unsqueeze(1)
        row_mask_f32 = row_mask.to(torch.float32)

        mu = (token_mask_f32 @ hidden_f32) / safe_section_lengths
        centered_std = (mu - mean_f32.unsqueeze(0)) / safe_scale_f32.unsqueeze(0)
        coeff_before_all = torch.einsum("sh,skh->sk", centered_std, rows_f32)
        coeff_before_masked = coeff_before_all * row_mask_f32
        projected_std = torch.einsum("sk,skh->sh", coeff_before_masked, rows_f32)

        donor_centered_std = (donor_means_f32 - mean_f32.unsqueeze(0)) / safe_scale_f32.unsqueeze(0)
        donor_coeff_all = torch.einsum("sh,skh->sk", donor_centered_std, rows_f32)
        donor_coeff_masked = donor_coeff_all * row_mask_f32
        donor_projected_std = torch.einsum("sk,skh->sh", donor_coeff_masked, rows_f32)

        delta_std = (
            strengths_f32.unsqueeze(1)
            * (donor_projected_std - projected_std)
            * active_i32.unsqueeze(1).to(torch.float32)
        )
        delta_raw = delta_std * scale_f32.unsqueeze(0)
        patched_hidden = hidden_f32 + torch.einsum(
            "st,sh->th",
            token_mask_f32,
            delta_raw,
        )

        patched_mu = mu + delta_raw
        patched_centered_std = (
            patched_mu - mean_f32.unsqueeze(0)
        ) / safe_scale_f32.unsqueeze(0)
        coeff_after_all = torch.einsum("sh,skh->sk", patched_centered_std, rows_f32)
        coeff_after_masked = coeff_after_all * row_mask_f32

        scalars_out = torch.stack(
            (
                torch.linalg.norm(delta_raw, dim=1),
                torch.linalg.norm(delta_std, dim=1),
                torch.linalg.norm(mu, dim=1),
                torch.linalg.norm(patched_mu, dim=1),
                torch.linalg.norm(centered_std, dim=1),
                torch.linalg.norm(patched_centered_std, dim=1),
                torch.linalg.norm(projected_std, dim=1),
                strengths_f32,
            ),
            dim=1,
        )

        valid.copy_(
            (
                active_i32
                * (section_lengths.squeeze(1) > 0).to(torch.int32)
            ).to(valid.dtype)
        )
        scalars.copy_(scalars_out.to(scalars.dtype))
        coeff_before.copy_(coeff_before_masked.to(coeff_before.dtype))
        coeff_after.copy_(coeff_after_masked.to(coeff_after.dtype))

        return patched_hidden.to(hidden_states.dtype)

    @_swap_components_batch_op.register_fake
    def _swap_components_batch_op_fake(
        hidden_states: torch.Tensor,
        mean: torch.Tensor,
        scale: torch.Tensor,
        safe_scale: torch.Tensor,
        batch_selected_rows: torch.Tensor,
        batch_donor_means: torch.Tensor,
        row_counts: torch.Tensor,
        token_spans: torch.Tensor,
        strengths: torch.Tensor,
        active: torch.Tensor,
        valid: torch.Tensor,
        scalars: torch.Tensor,
        coeff_before: torch.Tensor,
        coeff_after: torch.Tensor,
    ) -> torch.Tensor:
        del (
            mean,
            scale,
            safe_scale,
            batch_selected_rows,
            batch_donor_means,
            row_counts,
            token_spans,
            strengths,
            active,
            valid,
            scalars,
            coeff_before,
            coeff_after,
        )
        return torch.empty_like(hidden_states)

    _TORCH_LIBRARY_SWAP_COMPONENTS_BATCH_REGISTERED = True


_register_torch_library_swap_components_batch_op()


try:
    from vllm.model_executor.custom_op import CustomOp as _VLLMCustomOp
except Exception:
    _VLLMCustomOp = None


if _VLLMCustomOp is not None:

    @_VLLMCustomOp.register("activation_patch_hidden_states")
    class ActivationPatchHiddenStatesOp(_VLLMCustomOp):
        def __init__(self) -> None:
            try:
                super().__init__(enforce_enable=True)
            except TypeError:
                super().__init__()

        def forward_cuda(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            hidden_states = args[0]
            mean = args[1]
            scale = args[2]
            safe_scale = args[3]
            selected_rows = args[4]
            token_span = args[5]
            mode_id = int(args[6])
            strength = float(args[7])
            direction_std = kwargs.get("direction_std")
            donor_mean = kwargs.get("donor_mean")
            random_rows = kwargs.get("random_rows")
            batch_selected_rows = kwargs.get("batch_selected_rows")
            batch_row_counts = kwargs.get("batch_row_counts")
            batch_token_spans = kwargs.get("batch_token_spans")
            batch_strengths = kwargs.get("batch_strengths")
            batch_active = kwargs.get("batch_active")
            batch_donor_means = kwargs.get("batch_donor_means")
            stats_valid = kwargs.get("stats_valid")
            stats_scalars = kwargs.get("stats_scalars")
            stats_coeff_before = kwargs.get("stats_coeff_before")
            stats_coeff_after = kwargs.get("stats_coeff_after")

            if (
                batch_selected_rows is not None
                and batch_row_counts is not None
                and batch_token_spans is not None
                and batch_strengths is not None
                and batch_active is not None
                and stats_valid is not None
                and stats_scalars is not None
                and stats_coeff_before is not None
                and stats_coeff_after is not None
                and mode_id == PATCH_MODE_TO_ID[PATCH_MODE_PROJECT_OUT]
                and direction_std is None
                and donor_mean is None
                and random_rows is None
                and hasattr(torch.ops, "xenon_activation_patch")
                and hasattr(torch.ops.xenon_activation_patch, "project_out_batch")
            ):
                return torch.ops.xenon_activation_patch.project_out_batch(
                    hidden_states,
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

            if (
                batch_selected_rows is not None
                and batch_donor_means is not None
                and batch_row_counts is not None
                and batch_token_spans is not None
                and batch_strengths is not None
                and batch_active is not None
                and stats_valid is not None
                and stats_scalars is not None
                and stats_coeff_before is not None
                and stats_coeff_after is not None
                and mode_id == PATCH_MODE_TO_ID[PATCH_MODE_SWAP_COMPONENTS]
                and direction_std is None
                and donor_mean is None
                and random_rows is None
                and hasattr(torch.ops, "xenon_activation_patch")
                and hasattr(torch.ops.xenon_activation_patch, "swap_components_batch")
            ):
                return torch.ops.xenon_activation_patch.swap_components_batch(
                    hidden_states,
                    mean,
                    scale,
                    safe_scale,
                    batch_selected_rows,
                    batch_donor_means,
                    batch_row_counts,
                    batch_token_spans,
                    batch_strengths,
                    batch_active,
                    stats_valid,
                    stats_scalars,
                    stats_coeff_before,
                    stats_coeff_after,
                )

            if (
                mode_id == PATCH_MODE_TO_ID[PATCH_MODE_PROJECT_OUT]
                and direction_std is None
                and donor_mean is None
                and random_rows is None
                and hasattr(torch.ops, "xenon_activation_patch")
                and hasattr(torch.ops.xenon_activation_patch, "project_out")
            ):
                return torch.ops.xenon_activation_patch.project_out(
                    hidden_states,
                    mean,
                    scale,
                    safe_scale,
                    selected_rows,
                    token_span,
                    strength,
                )

            return self.forward_native(*args, **kwargs)

        def forward_native(
            self,
            hidden_states: torch.Tensor,
            mean: torch.Tensor,
            scale: torch.Tensor,
            safe_scale: torch.Tensor,
            selected_rows: torch.Tensor,
            token_span: torch.Tensor,
            mode_id: int,
            strength: float,
            direction_std: torch.Tensor | None = None,
            donor_mean: torch.Tensor | None = None,
            random_rows: torch.Tensor | None = None,
        ) -> torch.Tensor:
            patched, _ = apply_activation_patch_tensor(
                hidden_states,
                mean=mean,
                scale=scale,
                safe_scale=safe_scale,
                selected_rows=selected_rows,
                token_span=(int(token_span[0].item()), int(token_span[1].item())),
                mode=patch_mode_from_id(int(mode_id)),
                strength=float(strength),
                direction_std=direction_std,
                donor_mean=donor_mean,
                random_rows=random_rows,
                collect_stats=False,
            )
            return patched

else:

    class ActivationPatchHiddenStatesOp:  # pragma: no cover - exercised only when vLLM is installed.
        pass


def build_activation_patch_hidden_states_op() -> Any | None:
    if _VLLMCustomOp is None:
        return None
    return ActivationPatchHiddenStatesOp()
