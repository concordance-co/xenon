# Native text sites

`NativeTorchEngine` adds two deliberately narrow CPU FP32 eager adapters. Existing `TorchEngine` residual behavior is unchanged.

| Architecture | Quantity | Tensor before selection | Meaning |
|---|---|---|---|
| Pinned GPT-2 | `attention_head_output` | batch, token, head, head_dim | Attention probability × value vectors, immediately before `attn.c_proj`; not probabilities, not projected residual contributions |
| Pinned dense Qwen3-0.6B | `gate_preactivation` | batch, token, neuron | `gate_proj(x)`, preserving negative values |
| Qwen3 | `gate_activation` | batch, token, neuron | `silu(gate_proj(x))`, not ReLU |
| Qwen3 | `up_projection` | batch, token, neuron | `up_proj(x)`, preserving signs |
| Qwen3 | `mlp_product` | batch, token, neuron | `silu(gate_proj(x)) * up_proj(x)`, before `down_proj` |

Qwen's MLP input is the decoder's post-attention RMS-normalized residual. GPT-2 head vectors come from attention on the pre-attention layer-normalized residual. Qwen attention-head/GQA hooks are unsupported: its 16 query / 8 KV heads are not exposed or conflated with GPT-2's ordinary heads. Model identity, adapter version, module coordinate, quantity, axes and explicit selection travel with each feature. Sites from a different checkpoint or adapter fail before model execution.

Use `engine.site(name, quantity, layer, neurons=(...), heads=(...), tokens=..., model_identity=...)` with exactly one nonempty explicit unit selector. Supported prompt token selectors are last, slice and full_sequence. All examples are processed in bounded existing checkpoint units; only selected layers/tokens/units are retained. No all-layer/all-neuron enumeration occurs implicitly.

`NativeInterventionSite.from_capture(site)` creates the corresponding write coordinate. `NativeCoordinatePatch` supports scalar scale (one is identity, zero ablation) and replacement by a scalar or exact selected tensor shape. It edits only declared coordinates. Gate edits recompute activation/product; product edits leave the recorded gate/up unchanged. During greedy generation the selected positions remain anchored to the original prompt. Decode-token edits, other operators, architectures and device/dtype combinations fail explicitly. Scoped hooks and temporary MLP forwards clean up even on exceptions; explicit NativeSession forwards preserve gradients when requested.

`ExpertLocalCoordinates` describes observed routed tokens and positive mixture weights for one expert. Unrouted tokens have no record, not measured zeros. A tiny explicit mixture fixture verifies this schema; no real MoE engine capability is advertised.

## Prepare and run one real example

Use the exact isolated `core-local` environment installation from [local pack setup](../workspaces/local_research/phase_00/README.md). The pinned core lock already includes the required Transformers implementation; no shared environment modification is required. Prepare the official checkpoint separately:

```bash
export XENON_NATIVE_MODEL_CACHE="$HOME/.cache/xenon/qwen3-0.6b-c1899de"
workspaces/local_research/.cache/runtime/bin/hf download Qwen/Qwen3-0.6B \
  config.json generation_config.json merges.txt model.safetensors tokenizer.json tokenizer_config.json vocab.json README.md LICENSE \
  --revision c1899de289a04d12100db370d81485cdf75e47ca --local-dir "$XENON_NATIVE_MODEL_CACHE"
export XENON_NATIVE_ARCHITECTURE=qwen3
export XENON_HOME="$HOME/.xenon-native-smoke"
export PYTHONPATH="$PWD"
workspaces/local_research/.cache/runtime/bin/python -m pipelines_v2.cli workflow run \
  --file platform/examples/07_native_sites.py --profile local-scratch \
  --scratch-model-cache "$XENON_NATIVE_MODEL_CACHE" --run-id wr_native_001
```

This command captures three selected neurons for each of the four Qwen quantities, performs one selected gate ablation with two generated tokens, and produces a local ReportSpec artifact. It uses one plain text prompt, no chat template, no thinking-mode prompt wrapper and no network during execution. The default scratch envelope enforces 8GiB aggregate RSS, 600s and 256MiB outputs. Use a new run ID for a rerun.

For GPT-2, use the prepared pinned GPT-2 cache and `XENON_NATIVE_ARCHITECTURE=gpt2`; the same workflow selects head zero before projection. Synthetic tensor oracles verify both adapters; real Qwen CPU evidence is recorded in the T12 handoff. Modal L4 FP32/eager capture and edits also passed. MPS, Qwen3-32B and real MoE remain outside current support. These checks establish plumbing, not research findings.

## Configurable CPU/Modal execution

`NativeTorchEngine(..., device='cuda')` supports the same pinned GPT-2 head and Qwen3 MLP coordinates on Modal FP32/eager execution. Typed selections, datasets and intervention values remain caller-owned. Native route pairs/sites/IG intervals are bounded by explicit `max_work_units` and token budgets rather than the earlier fixture caps. See [LLM_RESEARCH.md](LLM_RESEARCH.md); larger architectures, GQA head mappings and MoE adapters are separate work.
