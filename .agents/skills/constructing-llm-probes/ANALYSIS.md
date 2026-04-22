# Representation Analysis Techniques

## Readout is not mechanism by itself

A strong probe or PCA separation supports a representational claim:

- the information is present and decodable

It does **not** by itself establish:

- where the computation first formed
- whether the site is causally important
- whether the same layer is the best place to intervene

For mechanistic workflows, treat representation analysis as one part of a broader chain:

1. behavioral sanity
2. probe readout
3. span / position localization
4. causal intervention
5. mechanism follow-up

## PCA on activations

Visualize how representations are organized by projecting to 2D/3D:

### PCA on mean activations (grouped by label)

```python
from sklearn.decomposition import PCA
import numpy as np

# Group hidden states by label value and average
unique_labels = sorted(set(all_labels))
mean_hiddens = np.stack([
    hiddens[labels == val].mean(axis=0) for val in unique_labels
])

pca = PCA(n_components=3)
projected = pca.fit_transform(mean_hiddens)

# Plot with matplotlib
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
scatter = ax.scatter(projected[:, 0], projected[:, 1], c=unique_labels, cmap="viridis")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
plt.colorbar(scatter, label="Label value")
```

### PCA on all token activations

```python
# Subsample if needed (PCA on millions of tokens is slow)
rng = np.random.default_rng(42)
idx = rng.choice(len(X), size=min(50000, len(X)), replace=False)
X_sub, y_sub = X[idx], y[idx]

pca = PCA(n_components=50)
pca.fit(X_sub)

# Explained variance curve — how many dimensions matter?
cumvar = np.cumsum(pca.explained_variance_ratio_)
# Plot cumvar to see dimensionality of the representation
```

### Variance explained per layer

```python
results = []
for layer_idx in range(num_layers):
    mean_h = compute_mean_hiddens(layer_idx)
    pca = PCA(n_components=min(10, len(mean_h)))
    pca.fit(mean_h)
    results.append({
        "layer": layer_idx,
        "var_explained_2d": pca.explained_variance_ratio_[:2].sum(),
        "var_explained_5d": pca.explained_variance_ratio_[:5].sum(),
    })
```

## SAE (Sparse Autoencoder) analysis

Decompose activations into interpretable sparse features using `sae-lens`:

```python
from sae_lens import SAE

# Load a pretrained SAE for your model
sae = SAE.from_pretrained(
    release="gpt2-small-resid-post-v5-128k",
    sae_id=f"blocks.{layer}.hook_resid_post",
    device="cpu",
)
sae = sae.to(device)

# Encode activations into sparse feature space
with torch.no_grad():
    # hidden_states: [N, hidden_dim]
    feature_acts = sae.encode(hidden_states)  # [N, num_sae_features]

# Weight by decoder vector norms for interpretable magnitudes
wdec_norm = torch.linalg.norm(sae.W_dec, dim=1)  # [num_sae_features]
weighted_acts = feature_acts * wdec_norm
```

### Finding features that correlate with your label

```python
# Group activations by label, compute mean activation per feature per group
unique_labels = sorted(set(labels))
mean_acts_per_label = []
for val in unique_labels:
    mask = (labels == val)
    mean_acts_per_label.append(weighted_acts[mask].mean(dim=0))
mean_acts = torch.stack(mean_acts_per_label)  # [num_labels, num_features]

# Features with highest variance across label groups are most informative
std_per_feature = mean_acts.std(dim=0)
top_k = 20
top_features = std_per_feature.argsort(descending=True)[:top_k]
print(f"Top {top_k} features: {top_features.tolist()}")
```

### Projecting SAE features into PCA space

To see how SAE features relate to the PCA manifold:

```python
# Get the decoder vectors for top features
top_decoder_vecs = sae.W_dec[top_features].float().cpu().numpy()  # [k, hidden_dim]

# Project into PCA space (fit PCA on mean_hiddens first)
projected_features = pca.transform(top_decoder_vecs)[:, :2]
# Plot these as arrows/vectors overlaid on the PCA scatter
```

### Available pretrained SAEs

| Model | sae-lens release | sae_id pattern |
|-------|-----------------|----------------|
| GPT-2 | `gpt2-small-resid-post-v5-128k` | `blocks.{L}.hook_resid_post` |
| Gemma-2-9B | `gemma-scope-9b-pt-res-canonical` | `layer_{L}/width_131k/canonical` |
| Llama-3.1-8B | `llama_scope_lxr_32x` | `l{L}r_32x` |

## Logit lens

Project intermediate hidden states through the model's unembedding matrix to see what the model "would predict" at each layer:

```python
def logit_lens(model, hidden_states, tokenizer, top_k=5):
    """hidden_states: [seq_len, hidden_dim] from a specific layer."""
    # Apply layer norm if the model uses one before the LM head
    if hasattr(model.transformer, "ln_f"):  # GPT-2 style
        normed = model.transformer.ln_f(hidden_states)
    elif hasattr(model.model, "norm"):  # Llama style
        normed = model.model.norm(hidden_states)
    else:
        normed = hidden_states

    logits = model.lm_head(normed)  # [seq_len, vocab_size]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_ids = probs.topk(top_k, dim=-1)

    results = []
    for pos in range(len(hidden_states)):
        tokens = [tokenizer.decode([tid]) for tid in top_ids[pos]]
        results.append(list(zip(tokens, top_probs[pos].tolist())))
    return results

# Usage: for each layer, see how predictions evolve
for layer_idx in range(num_layers):
    h = outputs.hidden_states[layer_idx + 1][0]  # [seq_len, hidden_dim]
    preds = logit_lens(model, h, tokenizer)
    # preds[pos] = [("token", prob), ...]
```

### Tuned lens

The tuned lens learns an affine transformation per layer to better align intermediate representations with the output space. Install via `pip install tuned-lens`:

```python
from tuned_lens import TunedLens

tuned_lens = TunedLens.from_model_and_pretrained(model)
# Then use tuned_lens.transform_hidden(hidden, layer_idx) instead of raw projection
```

## Causal interventions (activation patching)

Test whether information is *causally used* by the model, not just present:

```python
def activation_patch(model, clean_input, corrupt_input, layer_idx, positions):
    """
    Run model on corrupt_input, but patch in clean activations at
    specific layer and positions. If output changes toward clean output,
    those activations causally matter.
    """
    clean_cache = {}
    corrupt_cache = {}

    def save_hook(name, cache):
        def fn(module, input, output):
            cache[name] = output.detach().clone()
        return fn

    def patch_hook(name, clean_acts, positions):
        def fn(module, input, output):
            patched = output.clone()
            for pos in positions:
                patched[0, pos] = clean_acts[name][0, pos]
            return patched
        return fn

    # Run clean to get activations
    handles = []
    target_module = get_layer_module(model, layer_idx)
    handles.append(target_module.register_forward_hook(save_hook("target", clean_cache)))
    with torch.no_grad():
        clean_out = model(**clean_input)
    for h in handles:
        h.remove()

    # Run corrupt with patching
    handles = []
    handles.append(target_module.register_forward_hook(
        patch_hook("target", clean_cache, positions)))
    with torch.no_grad():
        patched_out = model(**corrupt_input)
    for h in handles:
        h.remove()

    return patched_out
```

## Difference-in-means for concept directions

Find a direction in activation space that separates two groups (e.g., true vs false statements):

```python
def concept_direction(hiddens_group_a, hiddens_group_b):
    """Returns a unit vector pointing from group B's mean to group A's mean."""
    mean_a = hiddens_group_a.mean(dim=0)
    mean_b = hiddens_group_b.mean(dim=0)
    direction = mean_a - mean_b
    return direction / direction.norm()

# This direction can be used for:
# 1. Binary probe: project activations onto this direction
# 2. Steering: add alpha * direction to activations at inference time
```
