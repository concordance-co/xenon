"""vLLM custom-op wrapper for activation patch hidden-state edits."""

from __future__ import annotations

import os
from typing import Any

import torch

from .activation_patch_math import INTERCHANGE_MODE_ID, RESIDUAL_PATH_MODE_ID
from .patching.subspace_family import is_subspace_mode_id


try:
    from vllm.model_executor.custom_op import CustomOp as _VLLMCustomOp
except Exception:  # pragma: no cover - exercised only when vLLM is installed.
    _VLLMCustomOp = None


if _VLLMCustomOp is not None:

    def _in_compiled_region() -> bool:
        compiler = getattr(torch, "compiler", None)
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling):
            try:
                return bool(is_compiling())
            except Exception:
                return False
        dynamo = getattr(torch, "_dynamo", None)
        is_dynamo_compiling = getattr(dynamo, "is_compiling", None)
        if callable(is_dynamo_compiling):
            try:
                return bool(is_dynamo_compiling())
            except Exception:
                return False
        return False

    def _debug_mode_enabled(*labels: str) -> bool:
        if _in_compiled_region():
            return False
        raw = str(os.getenv("XENON_ACTIVATION_PATCH_DEBUG", "") or "").strip()
        if not raw:
            return False
        enabled = {item.strip() for item in raw.split(",") if item.strip()}
        return any(label in enabled for label in labels)

    def _debug_panic(label: str, **fields: Any) -> None:
        if not _debug_mode_enabled("panic", label):
            return
        payload = " ".join(f"{key}={value!r}" for key, value in sorted(fields.items()))
        raise RuntimeError(f"activation-patch panic [{label}] {payload}".rstrip())

    @_VLLMCustomOp.register("activation_patch_hidden_states")
    class ActivationPatchHiddenStatesOp(_VLLMCustomOp):
        def __init__(self) -> None:
            try:
                super().__init__(enforce_enable=True)
            except TypeError:
                super().__init__()

        def forward_cuda(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            namespace = getattr(torch.ops, "xenon_activation_patch_v2", None)
            if kwargs.get("batch_token_spans") is not None and len(args) >= 8:
                hidden_states = args[0]
                mean = args[1]
                scale = args[2]
                safe_scale = args[3]
                selected_rows = args[4]
                token_span = args[5]
                mode_id = int(args[6])
                strength = float(args[7])
                direction_raw = args[8]
                direction_std = args[9]
                donor_mean = args[10]
                random_rows = args[11]
                match_projected_norm = args[12]
                if is_subspace_mode_id(mode_id) and hasattr(namespace, "subspace_batch"):
                    active = kwargs["batch_active"]
                    _debug_panic(
                        "subspace_custom_op_cuda",
                        hidden_shape=tuple(hidden_states.shape),
                        active_sum=int(active.sum().item()),
                        token_spans_shape=tuple(kwargs["batch_token_spans"].shape),
                        row_counts=kwargs["batch_row_counts"].detach().cpu().tolist(),
                    )
                    return namespace.subspace_batch(
                        hidden_states,
                        mean,
                        scale,
                        safe_scale,
                        kwargs["batch_mode_ids"],
                        kwargs["batch_selected_rows"],
                        kwargs["batch_row_counts"],
                        kwargs["batch_token_spans"],
                        kwargs["batch_query_positions"],
                        kwargs["batch_token_counts"],
                        kwargs["batch_strengths"],
                        kwargs["batch_active"],
                        kwargs["stats_valid"],
                        kwargs["stats_scalars"],
                        kwargs["stats_coeff_before"],
                        kwargs["stats_coeff_after"],
                        kwargs["batch_direction_raw"],
                        kwargs["batch_direction_std"],
                        kwargs["batch_donor_means"],
                        kwargs["batch_random_rows"],
                        kwargs["batch_rowwise"],
                        kwargs["batch_match_projected_norm"],
                    )
                if is_subspace_mode_id(mode_id) and hasattr(namespace, "subspace"):
                    return namespace.subspace(
                        hidden_states,
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
                return hidden_states

            hidden_states = args[0]
            operator_id = int(args[1])
            if operator_id == INTERCHANGE_MODE_ID and hasattr(namespace, "interchange_batch"):
                batch_query_positions = args[5]
                batch_token_counts = args[6]
                batch_donor_rows = args[7]
                batch_active = args[11]
                stats_valid = args[12]
                stats_scalars = args[13]
                return namespace.interchange_batch(
                    hidden_states,
                    batch_query_positions,
                    batch_donor_rows,
                    batch_token_counts,
                    batch_active,
                    stats_valid,
                    stats_scalars,
                )
            if operator_id == RESIDUAL_PATH_MODE_ID and hasattr(namespace, "residual_path_batch"):
                batch_query_positions = args[5]
                batch_token_counts = args[6]
                batch_payload_rows = args[7]
                batch_transport_modes = kwargs.get("batch_residual_path_transport_modes")
                batch_replace_alphas = kwargs.get("batch_residual_path_replace_alphas")
                if batch_transport_modes is None or batch_replace_alphas is None:
                    return self.forward_native(*args, **kwargs)
                batch_active = args[11]
                stats_valid = args[12]
                stats_scalars = args[13]
                return namespace.residual_path_batch(
                    hidden_states,
                    batch_query_positions,
                    batch_payload_rows,
                    batch_token_counts,
                    batch_transport_modes,
                    batch_replace_alphas,
                    batch_active,
                    stats_valid,
                    stats_scalars,
                )
            return self.forward_native(*args, **kwargs)

        def forward_native(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            namespace = getattr(torch.ops, "xenon_activation_patch_v2", None)
            if kwargs.get("batch_token_spans") is not None and len(args) >= 8:
                hidden_states = args[0]
                mean = args[1]
                scale = args[2]
                safe_scale = args[3]
                selected_rows = args[4]
                token_span = args[5]
                mode_id = int(args[6])
                strength = float(args[7])
                direction_raw = args[8]
                direction_std = args[9]
                donor_mean = args[10]
                random_rows = args[11]
                match_projected_norm = args[12]
                if is_subspace_mode_id(mode_id) and hasattr(namespace, "subspace_batch"):
                    active = kwargs["batch_active"]
                    _debug_panic(
                        "subspace_custom_op_native",
                        hidden_shape=tuple(hidden_states.shape),
                        active_sum=int(active.sum().item()),
                        token_spans_shape=tuple(kwargs["batch_token_spans"].shape),
                        row_counts=kwargs["batch_row_counts"].detach().cpu().tolist(),
                    )
                    return namespace.subspace_batch(
                        hidden_states,
                        mean,
                        scale,
                        safe_scale,
                        kwargs["batch_mode_ids"],
                        kwargs["batch_selected_rows"],
                        kwargs["batch_row_counts"],
                        kwargs["batch_token_spans"],
                        kwargs["batch_query_positions"],
                        kwargs["batch_token_counts"],
                        kwargs["batch_strengths"],
                        kwargs["batch_active"],
                        kwargs["stats_valid"],
                        kwargs["stats_scalars"],
                        kwargs["stats_coeff_before"],
                        kwargs["stats_coeff_after"],
                        kwargs["batch_direction_raw"],
                        kwargs["batch_direction_std"],
                        kwargs["batch_donor_means"],
                        kwargs["batch_random_rows"],
                        kwargs["batch_rowwise"],
                        kwargs["batch_match_projected_norm"],
                    )
                if is_subspace_mode_id(mode_id) and hasattr(namespace, "subspace"):
                    return namespace.subspace(
                        hidden_states,
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
                return hidden_states

            hidden_states = args[0]
            operator_id = int(args[1])
            if operator_id == INTERCHANGE_MODE_ID and hasattr(namespace, "interchange_batch"):
                batch_query_positions = args[5]
                batch_token_counts = args[6]
                batch_donor_rows = args[7]
                batch_active = args[11]
                stats_valid = args[12]
                stats_scalars = args[13]
                return namespace.interchange_batch(
                    hidden_states,
                    batch_query_positions,
                    batch_donor_rows,
                    batch_token_counts,
                    batch_active,
                    stats_valid,
                    stats_scalars,
                )
            if operator_id == RESIDUAL_PATH_MODE_ID and hasattr(namespace, "residual_path_batch"):
                batch_query_positions = args[5]
                batch_token_counts = args[6]
                batch_payload_rows = args[7]
                batch_transport_modes = kwargs.get("batch_residual_path_transport_modes")
                batch_replace_alphas = kwargs.get("batch_residual_path_replace_alphas")
                if batch_transport_modes is None or batch_replace_alphas is None:
                    return hidden_states
                batch_active = args[11]
                stats_valid = args[12]
                stats_scalars = args[13]
                return namespace.residual_path_batch(
                    hidden_states,
                    batch_query_positions,
                    batch_payload_rows,
                    batch_token_counts,
                    batch_transport_modes,
                    batch_replace_alphas,
                    batch_active,
                    stats_valid,
                    stats_scalars,
                )
            return hidden_states

else:

    class ActivationPatchHiddenStatesOp:  # pragma: no cover - exercised only when vLLM is installed.
        pass


def build_activation_patch_hidden_states_op() -> Any | None:
    if _VLLMCustomOp is None:
        return None
    return ActivationPatchHiddenStatesOp()


__all__ = [
    "INTERCHANGE_MODE_ID",
    "RESIDUAL_PATH_MODE_ID",
    "build_activation_patch_hidden_states_op",
]
