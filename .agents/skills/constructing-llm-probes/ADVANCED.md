# Advanced Probing Techniques

## Concept erasure: LEACE and RLACE

Remove a concept from representations to test necessity, or to create debiased representations.

### LEACE (LEAst-squares Concept Erasure)

```python
# pip install concept-erasure
from concept_erasure import LeaceFitter

fitter = LeaceFitter.fit(
    x=torch.tensor(X, dtype=torch.float32),      # [N, hidden_dim]
    z=torch.tensor(labels, dtype=torch.long),     # [N] concept labels
)
eraser = fitter.eraser

# Erase concept from activations
X_erased = eraser(torch.tensor(X, dtype=torch.float32)).numpy()

# Verify: probe on erased representations should be at chance
probe = LogisticRegression(max_iter=1000)
scores = cross_val_score(probe, X_erased, labels, cv=5)
print(f"Post-erasure accuracy: {scores.mean():.4f}")  # should be ~chance
```

### RLACE (Relaxed Linear Adversarial Concept Erasure)

Iteratively finds and removes the best linear classifier direction:

```python
# pip install rlace
from rlace import RLACE

rlace = RLACE(input_dim=hidden_dim, num_classes=2)
P = rlace.fit(X, labels)  # projection matrix that erases the concept
X_erased = X @ P.T
```

## Distributed Alignment Search (DAS)

Find linear subspaces where a concept is represented, even when distributed across multiple dimensions:

```python
# pip install pyvene
import pyvene as pv

# DAS learns a rotation matrix R such that intervening on R @ h
# at specific dimensions has maximal causal effect
config = pv.IntervenableConfig(
    representations=[{
        "layer": layer_idx,
        "component": "block_output",
        "low_rank_dimension": k,  # subspace dimension to search
    }]
)
intervenable = pv.IntervenableModel(config, model)
# Train with counterfactual pairs...
```

## Probe-to-steering pipeline

Use a trained probe's learned direction to steer model behavior:

```python
# 1. Train a linear probe
probe = LogisticRegression(max_iter=1000)
probe.fit(X_train, y_train)

# 2. Extract the learned direction (for binary classification)
direction = torch.tensor(probe.coef_[0], dtype=torch.float32)
direction = direction / direction.norm()

# 3. Steer: add/subtract this direction during inference
def steering_hook(direction, alpha):
    def fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        h = h + alpha * direction.to(h.device)
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return fn

layer_module = model.transformer.h[layer_idx]  # GPT-2 style
handle = layer_module.register_forward_hook(steering_hook(direction, alpha=3.0))
# Generate with the hook active, then remove
# handle.remove()
```

## Probing across training checkpoints

Study when information emerges during training using Pythia checkpoints:

```python
from transformers import AutoModelForCausalLM

checkpoints = [0, 1000, 10000, 50000, 143000]  # Pythia revision steps
results_by_step = []

for step in checkpoints:
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m-deduped",
        revision=f"step{step}",
        output_hidden_states=True,
    )
    # Extract activations and probe...
    metrics = fit_probe(hiddens, labels)
    metrics["step"] = step
    results_by_step.append(metrics)

# Plot probe accuracy vs training step to see when representations form
```

## Multi-token / sequence-level probing

When the property spans multiple tokens (e.g., sentence sentiment, coreference):

```python
# Option 1: Probe on the last token's hidden state
h_last = hidden_states[:, -1, :]  # [batch, hidden_dim]

# Option 2: Mean-pool across all tokens
h_mean = hidden_states.mean(dim=1)  # [batch, hidden_dim]

# Option 3: Mean-pool across specific span
span_start, span_end = 5, 10
h_span = hidden_states[:, span_start:span_end, :].mean(dim=1)

# Option 4: Attention-weighted pooling using attention weights
```

## Causal scrubbing / path patching

Test specific computational subgraphs:

```python
def path_patch(model, clean_input, corrupt_input,
               sender_layer, sender_pos,
               receiver_layer, receiver_head):
    """
    Patch the output of sender_layer at sender_pos from clean run
    into the input of receiver_head at receiver_layer during corrupt run.
    Tests whether information flows along this specific path.
    """
    # 1. Run clean forward, cache sender output
    # 2. Run corrupt forward with hook that patches sender output
    #    into receiver's input at the specific head
    # 3. Compare output to clean/corrupt baselines
    pass  # Implementation depends on model architecture
```

## Probing with TransformerLens

For mechanistic interpretability, TransformerLens provides cleaner access to activations:

```python
# pip install transformer-lens
import transformer_lens as tl

model = tl.HookedTransformer.from_pretrained("gpt2")

# Run with caching — gets all intermediate activations
logits, cache = model.run_with_cache(tokens)

# Access specific activations by name
resid_post = cache["blocks.6.hook_resid_post"]    # [batch, seq, hidden]
attn_out = cache["blocks.6.attn.hook_result"]       # [batch, seq, hidden]
mlp_out = cache["blocks.6.hook_mlp_out"]            # [batch, seq, hidden]
attn_pattern = cache["blocks.6.attn.hook_pattern"]  # [batch, head, q, k]

# Probe on any of these
X = resid_post[:, -1, :].cpu().numpy()  # last token, layer 6
```

## Information-theoretic probing (MDL probes)

Instead of accuracy, measure the *compression* a probe achieves — avoids the issue of powerful probes memorizing:

```python
# MDL probe: measure codelength = bits needed to transmit labels given representations
# Lower codelength = more information in representations

from sklearn.linear_model import LogisticRegression
import numpy as np

def online_codelength(X, y, block_sizes=None):
    """Prequential (online) codelength for MDL probing."""
    n = len(X)
    if block_sizes is None:
        block_sizes = [2 ** i for i in range(int(np.log2(n)) + 1)]
        block_sizes.append(n - sum(block_sizes))
        block_sizes = [b for b in block_sizes if b > 0]

    total_codelength = 0.0
    seen = 0

    for block_size in block_sizes:
        end = seen + block_size
        if seen == 0:
            # Uniform code for first block
            num_classes = len(np.unique(y))
            total_codelength += block_size * np.log2(num_classes)
        else:
            probe = LogisticRegression(max_iter=1000)
            probe.fit(X[:seen], y[:seen])
            probs = probe.predict_proba(X[seen:end])
            for i, yi in enumerate(y[seen:end]):
                class_idx = list(probe.classes_).index(yi)
                total_codelength -= np.log2(max(probs[i, class_idx], 1e-10))
        seen = end

    return total_codelength  # in bits

# Compare: codelength(X, y_real) vs codelength(X, y_random)
# Large gap = representations genuinely encode the concept
```
