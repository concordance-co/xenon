"""Router logit capture for Qwen3 MoE models running under vLLM.

Provides module-level functions that patch MoE block forward methods to
capture router logits during prefill. Works with any Qwen3 MoE variant
(Qwen3-30B-A3B, Qwen3-235B-A22B, etc.).

Design constraints:
- ``enforce_eager=True`` in vLLM config so patched forwards work reliably
  (no CUDA graph compilation).
- ``max_num_seqs=1`` so router buffers contain exactly one request's data.
- All vLLM imports are lazy to avoid import errors when vLLM is not installed.
"""

from __future__ import annotations

import types
from typing import Any


def find_moe_blocks(model: Any) -> dict[int, Any]:
    """Find all MoE sparse blocks by checking for ``.gate`` and ``.experts``.

    Args:
        model: The vLLM model instance (the object returned by the model
            runner, typically a ``Qwen3MoeForCausalLM``).

    Returns:
        ``{layer_idx: mlp_module}`` for every transformer layer whose MLP
        has both a ``gate`` and ``experts`` attribute.
    """
    blocks: dict[int, Any] = {}
    layers = model.model.layers
    for idx, layer in enumerate(layers):
        mlp = layer.mlp
        if hasattr(mlp, "gate") and hasattr(mlp, "experts"):
            blocks[idx] = mlp
    return blocks


def _make_patched_forward(block: Any) -> Any:
    """Create a patched forward that captures router logits.

    The returned callable replaces ``block.forward`` (not ``__call__``).
    When ``nn.Module.__call__`` invokes ``self.forward(hidden_states)``,
    it finds our function on the instance (shadowing the class method).
    The function receives only ``hidden_states`` -- no ``self`` -- because
    it captures ``block`` via closure.
    """
    def patched_forward(hidden_states: Any) -> Any:
        import torch  # noqa: F811 — lazy import inside hot path kept minimal

        assert hidden_states.dim() <= 2, (
            "Qwen3MoeSparseMoeBlock only supports 1D or 2D inputs"
        )
        is_input_1d = hidden_states.dim() == 1
        if is_input_1d:
            hidden_states = hidden_states.unsqueeze(0)
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if block.is_sequence_parallel:
            from vllm.model_executor.models.utils import sequence_parallel_chunk
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_logits, _ = block.gate(hidden_states)

        # --- CAPTURE ---
        if block._router_capture_enabled:
            block._router_logits_buffer[:num_tokens].copy_(router_logits.float())
            block._router_num_captured = num_tokens

        shared_out, fused_out = block.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        final_hidden_states = (
            shared_out + fused_out if shared_out is not None else fused_out
        )

        if block.is_sequence_parallel:
            from vllm.distributed import tensor_model_parallel_all_gather
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        elif block.tp_size > 1:
            final_hidden_states = (
                block.experts.maybe_all_reduce_tensor_model_parallel(
                    final_hidden_states
                )
            )

        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states

    return patched_forward


def init_router_capture(model: Any, max_tokens: int = 8192) -> None:
    """Initialise router capture on all MoE blocks.

    For each MoE block:
    - Saves the original ``forward`` as ``_original_forward``.
    - Pre-allocates a ``(max_tokens, num_experts)`` float32 buffer on the
      same device as the gate weights.
    - Installs a patched forward via closure.

    Capture starts **disabled**; call :func:`enable_router_capture` to
    begin recording.

    Args:
        model: The vLLM model instance.
        max_tokens: Maximum sequence length to pre-allocate buffers for.
    """
    import torch

    blocks = find_moe_blocks(model)
    if not blocks:
        raise RuntimeError(
            "No MoE blocks found on model. Is this a Qwen3 MoE variant?"
        )

    for layer_idx, block in blocks.items():
        gate = block.gate
        # Determine num_experts from the gate's weight shape.
        # ReplicatedLinear wraps a standard linear; weight is (out_features, in_features).
        weight = gate.weight if hasattr(gate, "weight") else gate.linear.weight
        num_experts = weight.shape[0]
        device = weight.device

        # Save original forward for potential restoration.
        block._original_forward = block.forward

        # Pre-allocate buffer.
        block._router_logits_buffer = torch.zeros(
            (max_tokens, num_experts), dtype=torch.float32, device=device
        )
        block._router_num_captured = 0
        block._router_capture_enabled = False
        block._router_layer_idx = layer_idx

        # Install patched forward.
        block.forward = _make_patched_forward(block)

    print(
        f"[vllm_qwen3_moe] Initialised router capture on {len(blocks)} MoE blocks "
        f"(max_tokens={max_tokens})"
    )


def enable_router_capture(model: Any) -> None:
    """Enable router logit recording on all patched MoE blocks."""
    blocks = find_moe_blocks(model)
    for block in blocks.values():
        block._router_capture_enabled = True


def disable_router_capture(model: Any) -> None:
    """Disable router logit recording on all patched MoE blocks."""
    blocks = find_moe_blocks(model)
    for block in blocks.values():
        block._router_capture_enabled = False


def collect_router_logits(model: Any) -> dict[int, Any]:
    """Collect captured router logits from all MoE blocks.

    Returns:
        ``{layer_idx: tensor}`` where tensor has shape
        ``(num_tokens, num_experts)`` in float32 on CPU.
        Only includes blocks that captured at least one token.
    """
    blocks = find_moe_blocks(model)
    result: dict[int, Any] = {}
    for layer_idx, block in blocks.items():
        n = getattr(block, "_router_num_captured", 0)
        if n > 0:
            result[layer_idx] = (
                block._router_logits_buffer[:n].detach().cpu().clone()
            )
    return result


def reset_router_buffers(model: Any) -> None:
    """Zero out all router capture buffers and reset counters."""
    blocks = find_moe_blocks(model)
    for block in blocks.values():
        if hasattr(block, "_router_logits_buffer"):
            block._router_logits_buffer.zero_()
            block._router_num_captured = 0


def restore_original_forwards(model: Any) -> None:
    """Restore original forward methods on all patched MoE blocks."""
    blocks = find_moe_blocks(model)
    for block in blocks.values():
        original = getattr(block, "_original_forward", None)
        if original is not None:
            block.forward = original
            del block._original_forward
        # Clean up capture state.
        for attr in (
            "_router_logits_buffer",
            "_router_num_captured",
            "_router_capture_enabled",
            "_router_layer_idx",
        ):
            if hasattr(block, attr):
                delattr(block, attr)
