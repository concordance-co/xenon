"""vLLM-side MoE router hooks.

These hooks are intentionally narrow: they capture gate logits for MoE blocks
that expose `.gate` and `.experts`. Observed routing decisions are not claimed
unless a backend exposes them directly.
"""

from __future__ import annotations

from typing import Any


def find_moe_blocks(model: Any) -> dict[int, Any]:
    blocks: dict[int, Any] = {}
    layers = getattr(getattr(model, "model", None), "layers", [])
    for idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "gate") and hasattr(mlp, "experts"):
            blocks[idx] = mlp
    return blocks


def _make_patched_forward(block: Any) -> Any:
    def patched_forward(hidden_states: Any) -> Any:
        import torch

        assert hidden_states.dim() <= 2, "MoE block expected 1D or 2D hidden states"
        is_input_1d = hidden_states.dim() == 1
        if is_input_1d:
            hidden_states = hidden_states.unsqueeze(0)
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if getattr(block, "is_sequence_parallel", False):
            from vllm.model_executor.models.utils import sequence_parallel_chunk

            hidden_states = sequence_parallel_chunk(hidden_states)

        router_logits, _ = block.gate(hidden_states)

        if getattr(block, "_router_capture_enabled", False):
            start = int(getattr(block, "_router_num_captured", 0))
            end = start + num_tokens
            if end > block._router_logits_buffer.shape[0]:
                raise RuntimeError(
                    "Router logits buffer overflow: "
                    f"needed {end} rows, allocated {block._router_logits_buffer.shape[0]}. "
                    "Increase the configured max token budget for router capture."
                )
            block._router_logits_buffer[start:end].copy_(router_logits.float())
            block._router_num_captured = end

        shared_out, fused_out = block.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        final_hidden_states = shared_out + fused_out if shared_out is not None else fused_out

        if getattr(block, "is_sequence_parallel", False):
            from vllm.distributed import tensor_model_parallel_all_gather

            final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
            final_hidden_states = final_hidden_states[:num_tokens]
        elif getattr(block, "tp_size", 1) > 1:
            final_hidden_states = block.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states

    return patched_forward


def init_router_capture(model: Any, max_tokens: int) -> bool:
    import torch

    blocks = find_moe_blocks(model)
    if not blocks:
        return False

    for layer_idx, block in blocks.items():
        if hasattr(block, "_original_forward"):
            continue
        gate = block.gate
        weight = gate.weight if hasattr(gate, "weight") else gate.linear.weight
        num_experts = int(weight.shape[0])
        device = weight.device

        block._original_forward = block.forward
        block._router_logits_buffer = torch.zeros(
            (max_tokens, num_experts),
            dtype=torch.float32,
            device=device,
        )
        block._router_num_captured = 0
        block._router_capture_enabled = False
        block._router_layer_idx = layer_idx
        block.forward = _make_patched_forward(block)

    return True


def enable_router_capture(model: Any) -> None:
    for block in find_moe_blocks(model).values():
        block._router_capture_enabled = True


def reset_router_buffers(model: Any) -> None:
    for block in find_moe_blocks(model).values():
        if hasattr(block, "_router_logits_buffer"):
            block._router_logits_buffer.zero_()
            block._router_num_captured = 0


def collect_router_logits(model: Any) -> dict[int, Any]:
    result: dict[int, Any] = {}
    for layer_idx, block in find_moe_blocks(model).items():
        n = int(getattr(block, "_router_num_captured", 0))
        if n > 0:
            result[layer_idx] = block._router_logits_buffer[:n].detach().cpu().clone()
    return result
