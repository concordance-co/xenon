# GPT-2 capture and steering

`TorchEngine` runs a prepared GPT-2 checkpoint on CPU or Modal CUDA in FP32 with eager
attention. It supports prompt residual capture (`resid_pre` / `resid_post`),
greedy generation, selected target-token log probabilities, exact donor-state
interchange, raw additive directions, and residual scaling. Weights are frozen
and dropout disabled. Modal L4 capture, generation and residual-edit checks have passed. MPS and other architectures remain unsupported.

Prepare a model cache separately, then run the ordinary workflow through the
bounded offline profile:

```bash
export XENON_GPT2_CACHE=/absolute/path/to/prepared/gpt2
uv run python -m pipelines_v2.cli workflow run --profile local-scratch \
  --scratch-model-cache "$XENON_GPT2_CACHE" \
  --file platform/examples/06_gpt2_local_steering.py
```

The example uses GPT-2 revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
Preparation may use `hf download openai-community/gpt2 --revision
607a30d783dfa663caf39e06633721c8d4cfcd7e --local-dir "$XENON_GPT2_CACHE"` outside
scratch. Keep `config.json`, `model.safetensors` and tokenizer assets together.
Execution never downloads models. Install the target's tested core runtime lock;
imports of `pipelines_v2.api` alone do not load torch or transformers.

The example catalogs five outputs: capture, baseline, scale-one no-op,
scale-zero ablation, and a local report. The prompt is “The capital of France
is”; it retains only the requested ` Paris` and ` London` target logprobs. These
are full-vocabulary log-softmax probabilities at the last nonpadding prompt
position before generation, not probabilities normalized over the two targets.
The dictionary keys are token ID strings, retaining requested order. Empty
`target_token_ids` preserves old row and spec serialization. Toy/vLLM reject
nonempty target requests during planning.

Residual sites refer to the inputs/outputs of numbered GPT-2 transformer blocks;
`resid_post` is before the model's final `ln_f`. Token positions are zero-based,
relative to the unpadded prompt. Capture preserves example keys and explicit
coordinates. Select `TensorStorage(dtype="float32")` when persistence must retain
the adapter's full precision. Pooling, decode-time edits, chat tools, stochastic
generation, MoE routing and custom prompt-metadata builders are rejected. Explicit
example `token_sections` are supported.

`ScalePatch(scale=1)` is identity; zero ablates. `AddDirectionPatch` adds
`strength * raw_vector` at each selected token and does not accept a subspace
projection. `InterchangePatch` requires strength one and an ActivationBank
created from this adapter's same actual checkpoint identity and site; donor and
recipient selections must have equal length. Every donor token state remains
distinct. Missing coordinates, wrong identity or width, and nonfinite values
fail rather than become pooled replacements. Different layers require an
explicit source-layer mapping.

For operation-scoped differentiation and warm reuse:

```python
with engine.open_session() as session:
    ids, attention_mask = session.encode(["A short prompt"])
    embeddings = session.model.transformer.wte(ids).detach().requires_grad_(True)
    logits, captured = session.forward(
        inputs_embeds=embeddings, attention_mask=attention_mask,
        capture=(("resid_post", 5),), differentiable=True,
    )
    # Differentiate a bounded scalar with torch.autograd.grad.
    # An edits={(site, layer): callable} mapping can carry a trainable residual mask.
    session.reset()
```

Session hooks preserve autograd and are removed even after an exception. Normal
capture detaches returned tensors; no request graphs or KV caches are retained.
Close releases the model. Sessions are explicit; T06 owns worker scheduling and
warm lifetimes. The adapter's CPU numerical policy and actual checkpoint/tokenizer
bytes enter C01, alongside ordinary C03 payload verification. An engineering
steering effect is not a scientific claim about a localized mechanism.

## CUDA and configurable inputs

`TorchEngine(..., device='cuda')` uses the separate pinned Linux CUDA runtime. GPT-2 eager FP32 capture/generation/edits run on the requested device; exported tensors return to CPU only at artifact serialization. Modal plans require explicit GPU, host memory, timeout and mounted prepared assets. Readiness checks tokenizer/context bounds before weights. See [LLM_RESEARCH.md](LLM_RESEARCH.md) for configurable workflows. Local MPS and arbitrary model architectures are unsupported.
