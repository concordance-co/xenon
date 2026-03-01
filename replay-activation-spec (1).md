# Replay & Activation Capture Pipeline — Spec

## Context

We're running a mechanistic interpretability study on AI financial decision-making. We have a dataset of inference logs from DX Terminal Pro (terminal.markets) — a live trading competition where ~hundreds of AI agents (all running Qwen3-235B-A22B on the same inference stack) trade meme tokens on Base. The ingestion pipeline is already built and produces a Parquet dataset where each row is a complete inference: full prompt, model completion, tool call, and labeled trade outcome (PnL at various time horizons).

The goal of this pipeline is to replay those prompts through a smaller Qwen3 MoE model, capture activations, and produce a dataset suitable for linear probing and MoE routing analysis to isolate features involved in financial decision-making.

---

## Hardware

**Ingestion pipeline:** Already running locally on a 64GB M4 Max. Produces the Parquet dataset.

**Replay & activation capture:** Runs on **Modal**. Target GPU: **A100 80GB**, which fits Qwen3-30B-A3B at fp16 (~60GB weights) with ~20GB headroom for activation buffers.

| Model | Params | fp16 VRAM | MoE Routing? |
|-------|--------|-----------|--------------|
| Qwen3-30B-A3B | 30B MoE (3B active) | ~60GB | **Yes** |
| Qwen3-8B | 8B dense | ~16GB | No |

**Qwen3-30B-A3B is the primary target.** Same MoE architecture as the production 235B (expert routing via top-k gating over a shared expert pool), which means router logit analysis, expert activation patterns, and per-expert feature decomposition are all available. This is the key reason to use remote compute rather than running a dense model locally — the MoE routing patterns are arguably the most interesting signal for interp.

Qwen3-8B remains a viable local fallback for quick iteration on attention/MLP features.

---

## Framework

PyTorch with HuggingFace Transformers on CUDA (Modal).

TransformerLens doesn't natively support Qwen3, so activation capture uses manual `register_forward_hook()` on the HF model. Target layers:

- **Router logits** at each MoE layer — the primary signal for routing analysis. Which experts are selected and with what weights for each token position. This is a natural sparse feature decomposition: the gating network projects the hidden state into a num_experts-dimensional space and selects top-k.
- **Residual stream** post-attention and post-MLP at each transformer block — standard targets for linear probing. Needed to train layer-sweep probes for decision type, profitability, token identity, etc.
- **Expert outputs** for the active experts per token — if memory allows. These let you attribute specific computations to specific experts, connecting routing analysis to what the experts actually compute.

---

## Modal Setup

The replay pipeline is a batch job.

### Modal Image

PyTorch + Transformers + safetensors + pyarrow. Pin Qwen3-30B-A3B weights to a **Modal Volume** (not baked into the image) so they persist across runs and don't re-download. Use `snapshot_download` from `huggingface_hub` in a setup function.

### Modal Volume

A single shared volume mounting:
- `/data/input/` — the Parquet dataset uploaded from local
- `/data/activations/` — output activation files written by the replay job
- `/models/` — cached Qwen3-30B-A3B weights

### Modal Function

```python
@modal.function(gpu="A100-80GB", timeout=3600)
def capture_activations(batch: list[dict]) -> None:
    """
    Takes a batch of rows from the Parquet dataset.
    Each row contains at minimum: log_id, prompt_text (or structured messages).
    Runs forward pass with hooks, writes activations to volume.
    """
```

The function should:
1. Load model once (use `@modal.enter()` on a class, or `modal.Cls` pattern so the model stays warm across batches)
2. For each prompt in the batch, tokenize and run a forward pass
3. Capture activations via hooks
4. Write to volume as .safetensor files keyed by `log_id`

### Batching

Process prompts individually (batch size 1) unless profiling shows headroom for more. The 20GB activation buffer needs to hold:

- Router logits: `num_layers × seq_len × num_experts` × fp32 — relatively small
- Residual stream: `num_layers × seq_len × hidden_dim` × fp16 — this is the big one

For Qwen3-30B-A3B: 48 layers, hidden_dim 4096, and assuming average prompt length ~4K tokens:
- Residual stream per inference: 48 × 4096 × 4096 × 2 bytes ≈ 1.5GB
- Router logits per inference: 48 × 4096 × 128 × 4 bytes ≈ 96MB

So ~1.6GB per inference at 4K tokens. With 20GB headroom, you could hold ~12 in memory, but writing to disk per-inference is simpler and safer.

Profile actual prompt lengths from the dataset first — they may be shorter or longer than 4K.

---

## Activation Storage Format

Write **separate files per inference** to avoid needing to hold the full dataset in memory:

```
/data/activations/
├── router_logits/
│   ├── {log_id}.safetensor    # shape: (num_layers, seq_len, num_experts)
│   └── ...
├── residual_stream/
│   ├── {log_id}.safetensor    # shape: (num_layers, seq_len, hidden_dim)
│   └── ...
└── metadata.parquet           # log_id, seq_len, prompt_hash, capture_timestamp
```

Router logits and residual streams go in separate directories — they feed different analysis pipelines. Router logits are small enough to pull locally in bulk for routing analysis. Residual streams may need to stay on-volume for probe training runs on Modal.

Use safetensors format (fast, memory-mapped, no pickle). Store router logits as fp32 (they're small and precision matters for gating analysis). Store residual streams as fp16.

---

## Prompt Fidelity

**This is critical.** The prompt fed to Qwen3-30B-A3B during replay must produce the same token sequence that Qwen3-235B-A22B saw in production. They share the same tokenizer, but:

- If the `/full-log/{id}` payload contains **structured chat messages** (system/user/assistant), apply the **exact Qwen3 chat template** (`tokenizer.apply_chat_template()`) rather than manual string concatenation. Different formatting = different token boundaries = different activations.
- Store the raw message structure from the payload alongside the formatted text so you can verify template application.
- The chat template may include special tokens (`<|im_start|>`, `<|im_end|>`, etc.) — make sure these are present and correctly placed.
- **Validate:** For a handful of samples, compare tokenized output from your replay pipeline against what you'd expect from the raw prompt. Spot-check token counts.

---

## Replay Plan

1. Upload Parquet dataset to Modal Volume (`/data/input/`)
2. On Modal: load Qwen3-30B-A3B fp16, register hooks on router layers + residual stream
3. Iterate through dataset rows, run forward pass per prompt
4. Write activations as .safetensor files to Modal Volume (`/data/activations/`)
5. Write `metadata.parquet` mapping log_id → seq_len, prompt hash, etc.
6. For router logit analysis: pull router_logits/ directory locally (small enough)
7. For probe training and MoE routing analysis on residual stream: either pull locally or run as a separate Modal job

---

## Downstream Analysis Targets

Primary focus is **linear probing** and **MoE routing analysis**, not SAE training.

### Linear Probes

Train lightweight linear classifiers on activations to test whether specific information is linearly represented at various layers:

- **Decision probe:** Given residual stream at layer L, can a linear probe predict buy/sell/hold? At which layer does this become linearly separable? Early separation suggests the decision is made quickly; late separation suggests deliberation across layers.
- **Profitability probe:** Can a linear probe on pre-decision activations predict whether the trade will be profitable (1h/4h/1d)? If yes, the model "knows" something about trade quality that it may not be acting on.
- **Token identity probe:** Can a probe predict which token is being traded from intermediate activations? At which layer does the model commit to a specific asset?
- **Risk probe:** Can a probe predict trade size or ETH amount from activations? Does this correlate with the vault's risk preference config, or does the model develop its own internal risk representation?

Probes are cheap — a logistic regression or single linear layer on frozen activations. This means you can sweep across all layers quickly to build a "layer-by-layer" picture of when different types of information crystallize in the forward pass.

### MoE Routing Analysis

This is the primary reason for using Qwen3-30B-A3B over a dense model. Analysis targets:

- **Expert specialization by decision type:** For each MoE layer, compute the average expert selection distribution conditioned on buy vs sell vs hold decisions. Are certain experts preferentially activated for certain decision types? Use mutual information or chi-squared tests between expert selection and decision label.
- **Routing entropy and trade quality:** Compute the entropy of the router's softmax distribution per token per layer. Hypothesis: lower entropy (more confident routing) at decision-critical token positions correlates with better trade outcomes. Alternatively, higher entropy may indicate uncertainty that the model correctly translates into smaller position sizes.
- **Expert co-activation patterns:** Which experts tend to fire together? Cluster trades by their expert activation vectors and see if clusters correspond to trading strategies (momentum, mean-reversion, etc.) or to profitable/unprofitable groupings.
- **Temporal routing shifts:** As the competition progresses and tokens get reaped, do routing patterns shift? Does the model route differently when fewer tokens are available?
- **Cross-vault routing comparison:** Two vaults seeing similar market conditions but with different strategies — do they show different routing patterns? This isolates the effect of the strategy prompt on expert selection.
- **Token-position routing analysis:** At which sequence positions (strategy text, price data, portfolio state, token names) do routing patterns diverge most between profitable and unprofitable trades? This tells you which parts of the prompt the MoE structure is most sensitive to.

### Combined Probe + Routing

The most interesting results will come from combining both:

- Train a decision probe at each layer. At the layer where the probe first achieves high accuracy, examine the router logits — which experts are active at that layer for buy vs sell?
- Use routing patterns as features for probes (instead of raw residual stream). If expert selection vectors are more predictive than residual stream for trade outcome, that suggests the routing mechanism itself is where the "work" happens.

---

## Data Volume Estimates

Assume ~500K inference logs over the full 21-day competition (conservative).

| Data | Per Inference | Total (500K) |
|------|-------------|--------------|
| Router logits | ~96MB @ 4K tokens | ~48TB — **too large, need selective capture** |
| Residual stream | ~1.5GB @ 4K tokens | ~750TB — **way too large** |

These numbers make it clear: **you cannot capture full activations for every inference.** Strategies:

1. **Sample:** Capture activations for a representative subset (e.g., 10K inferences stratified by vault, token, outcome). That's ~15GB router logits, ~15TB residual stream — still large for residual.
2. **Selective layers:** Only capture residual stream at a few layers of interest (e.g., first, middle, last, and any layers where router entropy is notably different between profitable/unprofitable trades).
3. **Router-first approach:** Capture only router logits for the full dataset (~960GB for 500K at 4K avg tokens — feasible on a large volume). Use router analysis to identify interesting subsets, then do targeted residual stream capture on those.
4. **Reduce sequence dimension:** Only capture activations at specific token positions (e.g., the final token before tool call, tokens corresponding to price values, the token where the model names the asset).

**Recommended approach:** Start with router logits only on a 10-50K sample. Run routing analysis and train layer-sweep probes. Use findings to identify which layers and token positions matter most, then do targeted residual stream capture on those for deeper probe analysis.
