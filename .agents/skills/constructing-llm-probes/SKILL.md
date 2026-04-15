---
name: constructing-llm-probes
description: Constructs linear and nonlinear probes for LLM interpretability in Python. Covers extracting hidden states from transformer models, training probing classifiers/regressors on activations, PCA visualization of representations, SAE feature analysis, logit lens, and causal interventions. Use when the user wants to probe, analyze, or interpret LLM internal representations, build probing classifiers, extract hidden states, or study what information is encoded in model activations.
---

# Constructing LLM Probes in Python

Build probes that test what information is linearly (or nonlinearly) encoded in LLM hidden states. This covers the full pipeline: activation extraction, probe design, training, evaluation, and visualization.

## Approach

1. **Identify the hypothesis**: what information do you believe the model encodes? (e.g., syntax, position, sentiment, factual knowledge)
2. **Design the labeling function**: map each token position to a ground-truth label
3. **Extract activations**: run inference with `output_hidden_states=True`, collect per-layer hidden states
4. **Train probes**: fit a linear (or nonlinear) model from activations to labels
5. **Evaluate**: compare probe accuracy against baselines, sweep across layers
6. **Analyze**: PCA/SAE on activations, logit lens, intervention experiments

For details on each step, see the reference files:
- [EXTRACTION.md](EXTRACTION.md) — hidden state extraction patterns (batched, memory-efficient, multi-model)
- [PROBES.md](PROBES.md) — probe types, training, evaluation, and baselines
- [ANALYSIS.md](ANALYSIS.md) — PCA, SAE, logit lens, causal interventions, steering vectors
- [ADVANCED.md](ADVANCED.md) — concept erasure, DAS, difference-in-means, probe-to-steering pipelines

## Quick start: minimal linear probe

```python
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

target_layer = 6  # which layer to probe

all_hiddens, all_labels = [], []
with torch.no_grad():
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        # hidden_states = (embeddings, layer_0, ..., layer_N-1)
        h = outputs.hidden_states[target_layer + 1].squeeze(0).cpu()
        all_hiddens.append(h)
        all_labels.append(your_labeling_fn(text, tokenizer))

X = torch.cat(all_hiddens, dim=0).numpy()
y = np.concatenate(all_labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Classification probe:
probe = LogisticRegression(max_iter=1000)
probe.fit(X_train, y_train)
print(f"Accuracy: {accuracy_score(y_test, probe.predict(X_test)):.4f}")

# Regression probe:
# probe = Ridge(alpha=1.0)
# probe.fit(X_train, y_train)
# print(f"R²: {r2_score(y_test, probe.predict(X_test)):.4f}")
```

## Common probing targets

| Hypothesis | Labels | Probe type |
|------------|--------|------------|
| Character/token position | int count | Regression |
| Part-of-speech | POS tag | Multi-class classification |
| Named entity type | BIO tag | Multi-class classification |
| Sentiment polarity | pos/neg | Binary classification |
| Syntactic depth | int | Regression |
| Factual knowledge / truthfulness | true/false | Binary classification |
| Next-token prediction (logit lens) | token ID | Read off via unembedding |

## Key principles

- **Prefer linear probes** for interpretability claims — high linear probe accuracy means the information is *linearly decodable* from the representation
- **Always compare against baselines**: random labels, majority class, shuffled controls
- **Selectivity**: real accuracy minus control accuracy; this isolates what the *model* encodes vs what the *probe* memorizes
- **Sweep layers**: plot probe performance per layer to find where information emerges, peaks, and decays
- **Report R² for regression, accuracy + F1 for classification**
- **Mind the dataset size**: linear probes in high-dimensional spaces can overfit; use cross-validation or held-out test sets

## Dependencies

Core: `torch`, `transformers`, `scikit-learn`, `numpy`
Optional: `sae-lens` (SAE analysis), `datasets` (HF datasets), `accelerate` (multi-GPU)
