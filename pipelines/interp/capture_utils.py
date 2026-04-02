"""Helpers for pooled capture outputs.

These utilities convert full-sequence residual/router captures into pooled
row-level and section-level tensors and save them in safetensors format.
They are reused by specialized capture and analysis paths and are not a
workflow entrypoint on their own.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def pool_per_row(
    residual: Any,
    row_boundaries: list[dict[str, Any]],
    section_boundaries: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {}

    if residual is None:
        return result

    _num_layers, seq_len, _hidden_dim = residual.shape

    for rb in row_boundaries:
        i = rb["row_index"]
        c_start = rb["content_start"]
        c_end = rb["content_end"]
        f_start = rb["full_start"]
        f_end = rb["full_end"]

        if c_start < c_end and c_end <= seq_len:
            row_slice = residual[:, c_start:c_end, :]
            result[f"row_mean_{i}"] = row_slice.mean(dim=1).to(torch.float16)
            result[f"row_eos_{i}"] = residual[:, c_end - 1, :].clone().to(torch.float16)
        elif f_start < f_end and f_end <= seq_len:
            row_slice = residual[:, f_start:f_end, :]
            result[f"row_mean_{i}"] = row_slice.mean(dim=1).to(torch.float16)
            result[f"row_eos_{i}"] = residual[:, f_end - 1, :].clone().to(torch.float16)

    if "market" in section_boundaries:
        m_start, m_end = section_boundaries["market"]
        if m_start < m_end and m_end <= seq_len:
            market_slice = residual[:, m_start:m_end, :]
            result["market_mean"] = market_slice.mean(dim=1).to(torch.float16)
            result["market_eos"] = residual[:, m_end - 1, :].clone().to(torch.float16)

    result["last_token"] = residual[:, -1, :].clone().to(torch.float16)

    if "preamble" in section_boundaries:
        p_start, p_end = section_boundaries["preamble"]
        if p_start < p_end and p_end <= seq_len:
            preamble_slice = residual[:, p_start:p_end, :]
            result["preamble_mean"] = preamble_slice.mean(dim=1).to(torch.float16)

    for section_name in ["active_settings", "portfolio", "constraints", "prev_decisions"]:
        if section_name in section_boundaries:
            s_start, s_end = section_boundaries[section_name]
            if s_start < s_end and s_end <= seq_len:
                section_slice = residual[:, s_start:s_end, :]
                result[f"{section_name}_mean"] = section_slice.mean(dim=1).to(torch.float16)
                result[f"{section_name}_eos"] = residual[:, s_end - 1, :].clone().to(torch.float16)

    return result


def pool_router_per_row(
    router_indices: Any,
    row_boundaries: list[dict[str, Any]],
    section_boundaries: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if router_indices is None:
        return result

    _num_layers, seq_len, _top_k = router_indices.shape

    for rb in row_boundaries:
        i = rb["row_index"]
        c_start = rb["content_start"]
        c_end = rb["content_end"]
        if c_start < c_end and c_end <= seq_len:
            result[f"router_row_{i}"] = router_indices[:, c_start:c_end, :].contiguous()

    if "market" in section_boundaries:
        m_start, m_end = section_boundaries["market"]
        if m_start < m_end and m_end <= seq_len:
            result["router_market"] = router_indices[:, m_start:m_end, :].contiguous()

    for section_name in ["active_settings", "portfolio", "constraints", "prev_decisions"]:
        if section_name in section_boundaries:
            s_start, s_end = section_boundaries[section_name]
            if s_start < s_end and s_end <= seq_len:
                result[f"router_{section_name}"] = router_indices[:, s_start:s_end, :].contiguous()

    return result


def _save_pooled(
    tensors: dict[str, Any],
    output_path: Path,
) -> int:
    from safetensors.torch import save_file

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v.contiguous() for k, v in tensors.items()}
    save_file(clean, str(output_path))
    return sum(v.nelement() * v.element_size() for v in clean.values())


__all__ = [
    "_save_pooled",
    "pool_per_row",
    "pool_router_per_row",
]
