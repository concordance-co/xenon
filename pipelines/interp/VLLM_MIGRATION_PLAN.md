# vLLM Capture Pipeline Migration Plan

## Goal

Replace HuggingFace Transformers with vLLM for activation capture (residual stream
+ MoE router logits/indices) on Modal, achieving 3-5x throughput improvement while
maintaining output format compatibility with the existing analysis pipeline.

## Current Architecture

The existing pipeline (`capture.py` + `modal_capture.py`) uses:

- `AutoModelForCausalLM.from_pretrained()` to load Qwen3-30B-A3B
- PyTorch `register_forward_hook()` on `model.model.layers[i]` for residual stream
- PyTorch `register_forward_hook()` on `model.model.layers[i].mlp.gate` for MoE router logits/indices
- Sequential per-example forward passes (no batching within a batch)
- Modal `.map()` to distribute batches of ~10 examples across A100-80GB containers

Output format per example:
- `residual_stream/{log_id}.safetensors` — key `"residual_stream"`, shape `(num_layers, [seq_len,] hidden_dim)` fp16
- `router_logits/{log_id}.safetensors` — keys `"router_logits"` fp16 + `"router_indices"` int16, shape `(num_layers, [seq_len,] num_experts|top_k)`
- `metadata.parquet` — per-example metadata (log_id, seq_len, prompt_hash, etc.)

---

## Approach: Custom vLLM Model with Pre-allocated Router Buffers

### Why This Works

vLLM already has official support for extracting **residual stream** hidden states at
arbitrary intermediate layers via the `extract_hidden_states` speculative decoding
method. This is first-party code in `vllm/v1/spec_decode/extract_hidden_states.py`.

For **MoE router logits**, we register a custom model class that overrides
`Qwen3MoeSparseMoeBlock.forward()` to `copy_()` router logits into a pre-allocated
GPU buffer. In-place `copy_()` to a fixed-address tensor is CUDA-graph-safe because
CUDA graphs replay the same kernel calls with the same memory pointers.

### How vLLM's `extract_hidden_states` Works

The mechanism piggybacks on speculative decoding infrastructure:

1. `Qwen3MoeModel.forward()` already collects auxiliary hidden states at layers
   specified by `aux_hidden_state_layers` (set via `eagle_aux_hidden_state_layer_ids`
   in the draft model config). It returns `(hidden_states, aux_hidden_states_list)`.

2. `ExtractHiddenStatesProposer.propose()` receives the aux hidden states, stacks
   them to shape `[num_tokens, num_selected_layers, hidden_size]`, and passes them
   to `ExtractHiddenStatesModel`.

3. `ExtractHiddenStatesModel` stores them in a `CacheOnlyAttentionLayer` (KV cache
   slot), and `ExampleHiddenStatesConnector.save_kv_layer()` extracts them from the
   cache and writes per-request safetensors files with shape
   `[num_selected_layers, prompt_len, hidden_size]`.

4. On request completion, `request_finished()` returns `{"hidden_states_path": path}`
   which appears in `output.kv_transfer_params`.

Configuration:
```python
LLM(
    model="Qwen/Qwen3-30B-A3B",
    speculative_config={
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {
            "hf_config": {
                "eagle_aux_hidden_state_layer_ids": list(range(48)),
            }
        },
    },
    kv_transfer_config={
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "shared_storage_path": "/data/activations/residual_stream",
        },
    },
)
```

### How Router Logit Capture Works (Our Custom Extension)

In vLLM's `Qwen3MoeSparseMoeBlock.forward()`, router logits are computed as a plain
`torch.Tensor` before entering the fused MoE CUDA kernel:

```python
# vllm/model_executor/models/qwen3_moe.py — Qwen3MoeSparseMoeBlock.forward()
router_logits, _ = self.gate(hidden_states)   # [num_tokens, num_experts]
#                    ↑ THIS is where we intercept
shared_out, fused_out = self.experts(hidden_states=hidden_states, router_logits=router_logits)
```

Our custom model class adds a pre-allocated buffer and an in-place copy at this point.
This is the same approach used by Adobe Research's SteerMoE project, but we use
in-place tensor ops instead of their `.cpu().numpy()` approach (which would sync the
GPU pipeline on every layer and destroy throughput).

#### CUDA Graph Compatibility

PyTorch forward hooks (`register_forward_hook`) do NOT work during CUDA graph replay —
graph replay bypasses the module hierarchy entirely. This is why we must modify the
model's `forward()` method directly rather than using hooks.

In-place `copy_()` to a pre-allocated tensor at a fixed GPU memory address IS
CUDA-graph-safe because:
- CUDA graphs record kernel calls with fixed memory pointers
- `copy_()` is recorded as a memcpy kernel with source/destination addresses
- On replay, the same memcpy executes with the same addresses
- The buffer content updates correctly on each replay

#### Expert Index Capture

vLLM's `FusedMoE` layer calls `select_experts()` which computes `topk_weights` and
`topk_ids` from the router logits inside the fused kernel. We cannot intercept these
post-selection. Instead, we capture the raw router logits (shape
`[num_tokens, num_experts]`) and compute top-k indices ourselves post-hoc. This is
cheap (a single `torch.topk` on CPU after transfer) and gives us both logits and
indices, matching the current pipeline's output.

---

## Implementation Plan

### New Files

#### 1. `pipelines/interp/vllm_qwen3_moe.py` — Custom Model Class

Registers a custom `Qwen3MoeForCausalLM` that captures router logits via
pre-allocated buffers. Key design:

```python
from vllm.model_executor.models.qwen3_moe import (
    Qwen3MoeForCausalLM as _BaseQwen3MoeForCausalLM,
    Qwen3MoeSparseMoeBlock,
)

class Qwen3MoeSparseMoeBlockWithCapture(Qwen3MoeSparseMoeBlock):
    """MoE block that writes router logits to a pre-allocated buffer."""

    def init_capture_buffer(self, layer_idx: int, max_tokens: int, num_experts: int):
        """Called after model load to set up the capture buffer."""
        self.layer_idx = layer_idx
        self.capture_enabled = False
        # Pre-allocate on the same device as the gate weights
        self.router_logits_buffer = torch.zeros(
            (max_tokens, num_experts),
            dtype=torch.float32,
            device=self.gate.weight.device,
        )
        self.num_captured_tokens = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Reproduce the base forward but intercept router_logits
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_logits, _ = self.gate(hidden_states)

        # --- CAPTURE: in-place copy to pre-allocated buffer ---
        if self.capture_enabled:
            self.router_logits_buffer[:num_tokens].copy_(router_logits)
            self.num_captured_tokens = num_tokens

        shared_out, fused_out = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states = (
            shared_out + fused_out if shared_out is not None else fused_out
        )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        return final_hidden_states

class Qwen3MoeForCausalLMWithCapture(_BaseQwen3MoeForCausalLM):
    """Qwen3MoE with router logit capture buffers on every MoE layer."""

    def init_router_capture(self, max_tokens: int = 8192):
        """Initialize capture buffers on all MoE blocks."""
        for layer_idx, layer in enumerate(self.model.layers):
            if hasattr(layer, 'mlp') and isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                layer.mlp.__class__ = Qwen3MoeSparseMoeBlockWithCapture
                layer.mlp.init_capture_buffer(
                    layer_idx=layer_idx,
                    max_tokens=max_tokens,
                    num_experts=self.config.num_experts,
                )

    def enable_router_capture(self):
        for layer in self.model.layers:
            if hasattr(layer.mlp, 'capture_enabled'):
                layer.mlp.capture_enabled = True

    def disable_router_capture(self):
        for layer in self.model.layers:
            if hasattr(layer.mlp, 'capture_enabled'):
                layer.mlp.capture_enabled = False

    def collect_router_logits(self) -> dict[int, torch.Tensor]:
        """Collect captured router logits from all MoE layers. Returns {layer_idx: tensor}."""
        result = {}
        for layer in self.model.layers:
            if hasattr(layer.mlp, 'capture_enabled') and layer.mlp.capture_enabled:
                n = layer.mlp.num_captured_tokens
                result[layer.mlp.layer_idx] = (
                    layer.mlp.router_logits_buffer[:n].detach().cpu()
                )
        return result
```

**Critical consideration — class swapping vs subclassing:**

The base `Qwen3MoeForCausalLM.__init__` creates `Qwen3MoeSparseMoeBlock` instances
internally. We can't easily make it create our subclass instead without duplicating
the entire init. Two approaches:

- **Class swap post-init** (shown above): After vLLM loads the model, we swap each
  MoE block's `__class__` to our subclass and call `init_capture_buffer()`. This is
  simpler but requires our subclass to be layout-compatible (same `__slots__`, etc.).

- **Monkey-patch `forward`**: Instead of swapping the class, use
  `types.MethodType` to replace just the `forward` method on each block instance.
  This is what the `HiddenStatesWorkerExtension` in `vllm-project/speculators` does.
  Safer but slightly more complex.

The monkey-patch approach is recommended for production. The class-swap is shown
above for clarity.

**Registration:**

```python
from vllm import ModelRegistry
ModelRegistry.register_model(
    "Qwen3MoeForCausalLM",
    "pipelines.interp.vllm_qwen3_moe:Qwen3MoeForCausalLMWithCapture"
)
```

#### 2. `pipelines/interp/vllm_connector.py` — Custom KV Connector

Extends `ExampleHiddenStatesConnector` to also save router logits alongside residual
hidden states. The connector intercepts `save_kv_layer()` (for residual via the
existing KV cache mechanism) and `request_finished()` (to trigger router logit
collection from the model's buffers).

The connector writes to the same directory structure as the current pipeline:
```
/data/activations/
├── residual_stream/{log_id}.safetensors   # key: "residual_stream"
├── router_logits/{log_id}.safetensors     # keys: "router_logits", "router_indices"
└── metadata.parquet
```

**Key challenge:** The connector runs on the worker side and has access to the KV
cache but not directly to the model's router logit buffers. Options:

- **Option A**: The connector gets a reference to the model instance during
  `register_kv_caches()` (the model is accessible via
  `get_layers_from_vllm_config()`). It calls `model.collect_router_logits()` during
  `save_kv_layer()`.

- **Option B**: Use `collective_rpc()` from the worker extension pattern to call
  `collect_router_logits()` on the worker process.

Option A is simpler and recommended for single-GPU deployment.

#### 3. `pipelines/interp/vllm_capture.py` — Capture Orchestration

Replaces `capture.py` for vLLM-based capture. Handles:

- Model initialization with custom model class + connector
- Tokenization and prompt formatting (reuses `_parse_messages()`)
- Batch submission to vLLM engine
- Post-processing: pooling, format conversion, metadata generation
- Output in the same safetensors format for downstream compatibility

```python
def run_vllm_capture(config: VLLMCaptureConfig) -> dict[str, Any]:
    from vllm import LLM, SamplingParams

    # Register custom model
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM",
        "pipelines.interp.vllm_qwen3_moe:Qwen3MoeForCausalLMWithCapture"
    )

    llm = LLM(
        model=config.model_id,
        speculative_config={
            "method": "extract_hidden_states",
            "num_speculative_tokens": 1,
            "draft_model_config": {
                "hf_config": {
                    "eagle_aux_hidden_state_layer_ids": config.layers or list(range(48)),
                }
            },
        },
        kv_transfer_config={
            "kv_connector": "XenonCaptureConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": str(config.output_dir),
                "capture_router": config.capture_router,
            },
        },
        gpu_memory_utilization=0.95,
        max_model_len=config.max_seq_len,
        enforce_eager=False,  # Enable CUDA graphs
    )

    # Submit all prompts at once — vLLM handles batching internally
    # via continuous batching + PagedAttention
    sampling_params = SamplingParams(max_tokens=1)  # prefill only
    outputs = llm.generate(tokenized_prompts, sampling_params)

    # Each output.kv_transfer_params["hidden_states_path"] points to
    # a safetensors file with residual hidden states.
    # Router logits are saved by the connector alongside.
```

#### 4. `pipelines/interp/modal_vllm_capture.py` — Modal Deployment

Modal wrapper for the vLLM capture pipeline. Key differences from `modal_capture.py`:

- Image includes `vllm` + `flashinfer-python` instead of `transformers`
- Uses H100 GPUs (better vLLM perf characteristics)
- Submits larger batches (vLLM handles internal batching efficiently)
- `scaledown_window=5` for burst workloads

```python
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "safetensors", "pyarrow", "flashinfer-python")
    .add_local_python_source("pipelines")
)

@app.cls(
    gpu="H100",
    volumes={"/data": volume, "/models": model_volume},
    image=image,
    timeout=7200,
    scaledown_window=5,
)
class VLLMCaptureWorker:
    model_id: str = modal.parameter(default="Qwen/Qwen3-30B-A3B")

    @modal.enter()
    def setup(self):
        # Register custom model + initialize vLLM engine once
        from vllm import LLM, ModelRegistry
        ModelRegistry.register_model(...)
        self.llm = LLM(
            model=f"/models/{self.model_id}",
            speculative_config={...},
            kv_transfer_config={...},
        )

    @modal.method()
    def capture_batch(self, rows: list[dict], **kwargs) -> list[dict]:
        # Tokenize all rows, submit to vLLM, collect results
        ...
```

### Modified Files

#### 5. `pipelines/interp/capture.py` — No changes

The existing HF Transformers capture code is preserved as-is for local/MPS usage
and as a fallback. The vLLM pipeline is a separate code path.

#### 6. Output Format Compatibility

The vLLM pipeline must produce **identical output** to the HF pipeline so that
`analysis.py`, `_try_load_compact()`, and `preload_features()` work unchanged:

| Artifact | Current Shape | vLLM Shape | Match? |
|---|---|---|---|
| `residual_stream/{id}.safetensors` key `"residual_stream"` | `(L, S, H)` fp16 | `(L, S, H)` fp16 | Yes — connector saves in same layout |
| `router_logits/{id}.safetensors` key `"router_logits"` | `(L, S, E)` fp16 | `(L, S, E)` fp16 | Yes — from buffer, cast to fp16 |
| `router_logits/{id}.safetensors` key `"router_indices"` | `(L, S, K)` int16 | `(L, S, K)` int16 | Yes — computed via `torch.topk` post-hoc |
| `metadata.parquet` | same schema | same schema | Yes |

Where L=num_layers, S=seq_len, H=hidden_dim, E=num_experts, K=top_k.

The connector handles the format conversion. For router indices, since vLLM's fused
kernel doesn't expose post-selection indices, we compute them from the raw logits:

```python
# Post-hoc top-k from raw router logits
# router_logits: (num_layers, seq_len, num_experts) float32
topk_values, topk_indices = torch.topk(router_logits, k=top_k, dim=-1)
# topk_indices: (num_layers, seq_len, top_k) → cast to int16
```

This matches the HF pipeline's behavior since the gate's forward returns indices
from the same top-k operation internally.

---

## Memory Budget (Qwen3-30B-A3B on H100-80GB)

| Component | Size |
|---|---|
| Model weights (fp16) | ~60 GB |
| vLLM KV cache (PagedAttention) | ~10 GB |
| Router logit buffers (48 layers × 8192 tokens × 128 experts × fp32) | ~1.9 GB |
| Hidden states staging (speculative decode) | ~1.5 GB |
| Working memory | ~5 GB |
| **Total** | **~78 GB** |

Tight on H100-80GB. Mitigations:
- Use `gpu_memory_utilization=0.90` to leave headroom
- Reduce `max_model_len` to 4096 if sequences are short
- Router buffers could use fp16 instead of fp32 (saves ~1 GB)
- Capture a subset of layers instead of all 48

---

## Risks and Mitigations

### 1. CUDA Graph + `copy_()` Correctness

**Risk:** In-place copy inside a CUDA-graph-captured region may behave unexpectedly
with dynamic batch sizes (different `num_tokens` per forward pass).

**Mitigation:** vLLM's CUDA graph system pads to fixed batch sizes for graph replay.
The `copy_()` uses a slice `[:num_tokens]` which records the actual token count.
Test with `enforce_eager=True` first, then enable CUDA graphs and compare outputs.

**Fallback:** If CUDA graph issues arise, run with `enforce_eager=True`. This loses
some throughput but still benefits from PagedAttention and continuous batching.

### 2. vLLM Version Coupling

**Risk:** Our custom model code tracks vLLM's internal `Qwen3MoeSparseMoeBlock`
implementation. Changes in vLLM releases could break our forward override.

**Mitigation:** Pin vLLM version in requirements. The `forward()` method we override
is stable (gate → experts → reduce pattern). Add a version check assertion.

### 3. Memory Pressure

**Risk:** Pre-allocated router buffers + model weights + KV cache may exceed 80GB.

**Mitigation:** Make buffer allocation lazy (only allocate when capture is enabled).
Support per-layer capture to reduce buffer count. Monitor with
`torch.cuda.memory_summary()`.

### 4. Multi-GPU / Tensor Parallelism

**Risk:** With `tensor_parallel_size > 1`, the router gate is `ReplicatedLinear`
(replicated across GPUs), so each GPU sees full router logits. Our buffer capture
should work but needs testing.

**Mitigation:** Start with single-GPU. For TP>1, only capture on rank 0.

### 5. Combining Two Capture Mechanisms

**Risk:** Residual capture via speculative decode and router capture via custom model
are two independent code paths that must stay synchronized per-request.

**Mitigation:** The connector handles both. Residual comes via `save_kv_layer()`,
router comes via model buffer collection in `request_finished()`. Both are keyed
by request ID.

---

## Performance Expectations

| Metric | Current (HF on A100) | vLLM (H100) | Speedup |
|---|---|---|---|
| Forward pass | Sequential, 1 example | Continuous batching | ~3-5x |
| KV cache | Disabled (`use_cache=False`) | PagedAttention | Memory efficient |
| CUDA graphs | None | Piecewise compilation | ~1.3x kernel launch |
| GPU utilization | ~30-50% | ~80-95% | ~2x |
| **Overall throughput** | ~2-10 ex/s | ~10-30 ex/s | **~3-5x** |

---

## Implementation Order

1. **`vllm_qwen3_moe.py`** — Custom model class with router capture buffers.
   Test locally with `enforce_eager=True` first.

2. **`vllm_connector.py`** — Custom KV connector that saves both residual and
   router outputs to the existing directory structure.

3. **`vllm_capture.py`** — Orchestration layer. Test end-to-end locally with a
   small model (Qwen3-8B on a single GPU) before scaling to 30B.

4. **`modal_vllm_capture.py`** — Modal deployment. Test with `--limit 5` first.

5. **Validation** — Run both HF and vLLM pipelines on the same inputs, compare
   safetensors outputs numerically (allow fp16 tolerance).

---

## Sources

### vLLM Official

- [Extract Hidden States Example](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/extract_hidden_states.py) — Official example showing `eagle_aux_hidden_state_layer_ids` config
- [ExtractHiddenStatesProposer](https://github.com/vllm-project/vllm/blob/main/vllm/v1/spec_decode/extract_hidden_states.py) — Proposer that stacks target model hidden states and passes to draft model
- [ExtractHiddenStatesModel](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/extract_hidden_states.py) — Dummy model with `CacheOnlyAttentionLayer` that stores hidden states in KV cache
- [ExampleHiddenStatesConnector](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py) — Connector that extracts hidden states from KV cache and saves as safetensors
- [Qwen3MoE Model](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_moe.py) — Base model with `Qwen3MoeSparseMoeBlock.forward()` showing router_logits interception point
- [Model Registration](https://docs.vllm.ai/en/stable/contributing/model/registration/) — How to register custom model classes
- [Plugin System](https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html) — vLLM plugin architecture
- [CUDA Graphs Design](https://docs.vllm.ai/en/stable/design/cuda_graphs/) — How piecewise CUDA graphs work in vLLM

### vLLM Issues

- [#19342: How to get router logits for MoE model](https://github.com/vllm-project/vllm/issues/19342) — Closed, references SteerMoE approach
- [#17501: Accessing Model Gate Logits in v1](https://github.com/vllm-project/vllm/issues/17501) — Closed, suggests `collective_rpc`
- [#3594: Hidden states feature request](https://github.com/vllm-project/vllm/issues/3594) — Closed as "not planned"
- [#6165: Return hidden states](https://github.com/vllm-project/vllm/issues/6165) — In progress
- [#12249: HiddenStatesProcessor RFC](https://github.com/vllm-project/vllm/issues/12249) — Merged as PR #22820 (final layer only)

### Third-Party References

- [SteerMoE (Adobe Research)](https://github.com/adobe-research/SteerMoE) — Production example of capturing router logits via custom vLLM model. Uses `.cpu().numpy()` per layer (GPU sync, slow). Our approach uses in-place `copy_()` instead.
- [vllm-hidden-states-extractor](https://github.com/fynnsu/vllm-hidden-states-extractor) — PoC plugin using speculative decode infrastructure. Now superseded by official `extract_hidden_states`.
- [HiddenStatesWorkerExtension (speculators)](https://github.com/vllm-project/speculators) — Worker extension pattern using `types.MethodType` to patch model forward. Alternative to custom model registration.
- [Implementing Hidden State Probes (vLLM Forum)](https://discuss.vllm.ai/t/implementing-hidden-state-probes/2291) — Community discussion confirming no built-in probe API

### Modal

- [Modal vLLM Inference Example](https://modal.com/docs/examples/vllm_inference) — Standard vLLM deployment pattern
- [Modal vLLM Throughput Example](https://modal.com/docs/examples/vllm_throughput) — Bulk inference with synchronous `LLM` class, ~30k input tok/s per H100
- [Modal Scaling Docs](https://modal.com/docs/guide/scale) — `.map()` limits, container lifecycle, `scaledown_window`

### PyTorch / CUDA Graphs

- [PyTorch CUDA Graph Trees](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html) — How hooks interact with CUDA graph capture/replay
- [PyTorch Custom Ops](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html) — Alternative: register capture as custom op opaque to compiler
