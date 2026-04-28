# Probe Types, Training, and Evaluation

## Start with the split you actually need

Before fitting a probe, decide whether the evaluation should be:

- cross-validation on one pooled dataset
- explicit train/test split
- lexical-family holdout
- action/domain holdout

If the benchmark already encodes a meaningful split, prefer using it directly instead of defaulting to CV. Otherwise you may overstate abstraction.

### Validate lexical-family holdouts before claiming non-lexical signal

A lexical-family holdout (train on one prompt variant, test on another that shares the target latent but differs lexically) only earns a non-lexical-signal claim if the variants produce response text that a text classifier cannot distinguish.

Mandatory validation step before running the activation probe:

1. train a text classifier (e.g., char TF-IDF 3–5 + ridge) on responses from variant A vs variant B
2. record balanced accuracy and AUROC
3. the within-variant text classifier must land near chance (BA ≤ ~0.65, AUROC ≤ ~0.75) for the holdout to be valid

If the within-variant text classifier is at ceiling (AUROC ≥ ~0.95), the variants are not functioning as a holdout — they are two different prompts with the same target label, and the activation probe across them remains confounded by the lexical signal the text classifier exploits. Repair by adding format constraints (output schema, length bound, vocabulary bans on each variant's canonical lexical family) until the validation passes.

A probe-transfer test that has not passed this within-variant text-classifier validation has not isolated lexical confound. Report results from such transfers as Level 2 representational only, with the lexical confound explicitly listed as not-yet-controlled.

This validation is a stricter operationalization of the technique referenced in `methodology/PRINCIPLES.md §12` (response-side probing requires active confound reduction). The skill `latent-label-data-augmentation` covers the variant-construction side of the same loop.

## Probe selection guide

| Probe | When to use | Interpretability | Capacity |
|-------|-------------|------------------|----------|
| Logistic Regression | Binary/multi-class, interpretability claims | High — linear decodability | Low |
| Ridge Regression | Continuous targets | High | Low |
| Least-squares (torch) | Large-scale regression, GPU-friendly | High | Low |
| MLP (1-2 layers) | Nonlinear encoding hypothesis | Medium | Medium |
| k-NN | Quick sanity check, no training | Medium | Nonparametric |

## Linear probes

### Classification (sklearn)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

probe = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, C=1.0)
)

# Cross-validated accuracy
scores = cross_val_score(probe, X, y, cv=5, scoring="accuracy")
print(f"Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
```

### Regression (sklearn)

```python
from sklearn.linear_model import Ridge

probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
scores = cross_val_score(probe, X, y, cv=5, scoring="r2")
print(f"R²: {scores.mean():.4f} +/- {scores.std():.4f}")
```

### Regression (torch, no sklearn)

For large-scale probing directly on GPU tensors:

```python
def fit_linear_probe(hiddens, labels):
    """Least-squares linear probe. hiddens: list of [T, H], labels: list of [T]."""
    X = torch.cat([h.float() for h in hiddens], dim=0)  # [N, H]
    y = torch.cat([torch.as_tensor(l).float().view(-1) for l in labels])  # [N]

    # Add bias column
    Xb = torch.cat([X, torch.ones(X.shape[0], 1)], dim=1)  # [N, H+1]
    beta = torch.linalg.lstsq(Xb, y).solution  # [H+1]
    y_hat = Xb @ beta

    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot).item()
    rmse = (ss_res / len(y)).sqrt().item()
    mae = (y - y_hat).abs().mean().item()
    return {"r2": r2, "rmse": rmse, "mae": mae, "coef": beta}
```

## MLP probe (nonlinear)

Use when you suspect nonlinear encoding, or to establish an upper bound on extractable information:

```python
import torch.nn as nn
import torch.optim as optim

class MLPProbe(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

def train_mlp_probe(X_train, y_train, X_val, y_val, input_dim, num_classes,
                    epochs=50, lr=1e-3, batch_size=256):
    probe = MLPProbe(input_dim, num_classes)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(epochs):
        probe.train()
        perm = torch.randperm(len(X_train_t))
        for i in range(0, len(X_train_t), batch_size):
            idx = perm[i:i + batch_size]
            logits = probe(X_train_t[idx])
            loss = loss_fn(logits, y_train_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        val_logits = probe(torch.tensor(X_val, dtype=torch.float32))
        preds = val_logits.argmax(dim=-1).numpy()
    return probe, accuracy_score(y_val, preds)
```

The gap between MLP probe accuracy and linear probe accuracy reveals how much information is nonlinearly encoded.

## Per-layer sweep

The most informative analysis — shows where information appears, peaks, and fades across depth:

```python
import pandas as pd

results = []
for layer_idx in range(model.config.num_hidden_layers):
    hiddens = extract_layer(dataset, model, tokenizer, layer_idx)
    metrics = fit_probe(hiddens, labels)
    metrics["layer"] = layer_idx
    results.append(metrics)

df = pd.DataFrame(results)
# Plot: df.plot(x="layer", y="r2")  or  y="accuracy"
```

## Evaluation and baselines

### Binary metrics beyond accuracy

For binary probes, accuracy alone is often not enough.

- **Balanced accuracy**: useful when classes are imbalanced or when you want one thresholded operating point
- **AUROC**: useful when you care about score separability across all thresholds
- **AUPRC**: useful when the positive class is rare

Interpretation:

- high AUROC + imperfect balanced accuracy often means the representation is strong but the threshold is imperfect
- low AUROC means the classes are not cleanly separated by the probe score

If available, also report:

- `TPR@FPR=5%`
- `TPR@FPR=10%`

These are especially useful when the downstream use case is monitoring rather than raw classification.

Always report both:

- thresholded metrics:
  accuracy / balanced accuracy / F1 as appropriate
- threshold-free metrics:
  AUROC and AUPRC when applicable

### Mandatory baselines

1. **Majority class**: always predict the most common label
2. **Random baseline**: predict uniformly at random
3. **Shuffled control**: train probe on randomly permuted labels — this measures the probe's capacity to memorize
4. **Selectivity**: `accuracy_real - accuracy_control`. High selectivity = information is genuinely in the activations

### Control task methodology (Hewitt & Liang, 2019)

```python
# Train probe on real labels
real_acc = cross_val_score(probe, X, y_real, cv=5).mean()

# Train probe on random control labels (same cardinality)
rng = np.random.default_rng(42)
y_control = rng.permutation(y_real)
control_acc = cross_val_score(probe, X, y_control, cv=5).mean()

selectivity = real_acc - control_acc
print(f"Selectivity: {selectivity:.4f}")
```

### Cautions

- High-dimensional activations + small datasets = overfitting. Use regularization (C parameter in LogisticRegression, alpha in Ridge)
- More probe parameters != better. The probe should be a *diagnostic tool*, not a powerful model
- Cross-validate. Don't report single train/test split results
- If the benchmark uses an explicit split, do not replace it with CV unless you are very clear that you are changing the claim
- Consider **MDL probes** (minimum description length) as an alternative to accuracy — they measure how compressible the labels are given the representations
- Always compare against cheap baselines:
  majority, shuffled labels, and any benchmark-specific surface baseline already identified in the control plan

## Span-local probes

For many relational tasks, probing only the last token is too coarse.

Useful alternatives:

- pooled activation over a named span
- probe at a claim span
- probe at a rule / evidence span
- probe at post-comparison context

This is especially valuable when you want to distinguish:

- lexical identity
- null sites
- comparison-result sites
- late consolidated readout

## Saving and loading probes

```python
import pickle

# Save
with open("probe_layer6.pkl", "wb") as f:
    pickle.dump(probe, f)

# Load
with open("probe_layer6.pkl", "rb") as f:
    probe = pickle.load(f)
```

For torch MLP probes, use `torch.save(probe.state_dict(), path)`.
