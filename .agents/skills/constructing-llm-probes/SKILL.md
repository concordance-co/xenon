---
name: constructing-llm-probes
description: Constructs linear and nonlinear probes for LLM interpretability in Python. Covers extracting hidden states from transformer models, training probing classifiers/regressors on activations, PCA visualization of representations, SAE feature analysis, logit lens, and causal interventions. Use when the user wants to probe, analyze, or interpret LLM internal representations, build probing classifiers, extract hidden states, or study what information is encoded in model activations.
---

# Constructing LLM Probes in Python

Build probes that test what information is linearly (or nonlinearly) encoded in LLM hidden states. This covers the full pipeline: activation extraction, probe design, training, evaluation, and visualization.

## Before you probe

`Behavioral sanity comes first.`

Before treating probe results as meaningful, verify that the underlying task is behaviorally sane:

- inspect real prompt examples from each class
- run the base model on a small slice
- verify outputs are parseable
- verify the model is actually solving the intended task
- inspect failures manually

If the task is malformed, ambiguous, or role-format dependent, probe results may reflect prompt artifacts rather than the target variable.

## Approach

1. **Identify the hypothesis**: what information do you believe the model encodes? (e.g., syntax, position, sentiment, factual knowledge)
2. **Design the labeling function**: map each token position to a ground-truth label
3. **Choose the split scheme**: lexical holdout, domain holdout, action holdout, or explicit train/test split that matches the claim
4. **Choose the capture location**: prompt span, prompt end, generation span, or full prompt-plus-generation sequence
5. **Extract activations**: run inference with `output_hidden_states=True`, collect per-layer hidden states
6. **Train probes**: fit a linear (or nonlinear) model from activations to labels
7. **Evaluate**: compare probe metrics against baselines, sweep across a broad range of layers
8. **Localize**: compare positions / spans instead of only the last token when the variable is relational
9. **Analyze**: PCA/SAE on activations, logit lens, intervention experiments

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
- **Sweep layers broadly**: plot probe performance over early, middle, and late layers to find where information emerges, peaks, and decays
- **Use explicit train/test splits when the benchmark already defines them** — especially for lexical holdout or action holdout claims
- **Treat capture location as part of the claim** — prompt-end probes read pre-answer state or prediction of later behavior; generation-token probes read state while the model is producing the behavior
- **For binary probes, report AUROC and balanced accuracy as the headline metrics** — AUROC captures separability while BA captures thresholded performance under imbalance
- **For relational tasks, compare multiple spans or positions** — a strong last-token probe does not tell you where the signal first appears
- **Do not equate best readout layer with best intervention layer** — later layers may be easiest to decode while earlier layers are better causal sites
- **Report R² for regression, accuracy + F1 for classification, and AUROC / AUPRC for binary tasks when possible**
- **Mind the dataset size**: linear probes in high-dimensional spaces can overfit; use cross-validation or held-out test sets
- **Treat early-layer strength as a warning**: high AUROC or BA in very early layers often means lexical, template, role, duplicate-row, or prompt-side leakage

## Evidence discipline

Probe reports must include `evidence_rung` and `claim_boundary`.

Use `evidence_rung: representational` for ordinary readout results. Use
`localized_representational` only when the result compares layers, positions,
spans, sections, or tokens and supports a specific localization claim. Do not
use `causal` or `mechanistic` for probe results without downstream intervention
evidence.

## Gotchas

- Do not probe before behavioral sanity is established for the target model and
  task.
- Do not use random train/test splits when the claim depends on lexical, domain,
  action, or carrier generalization.
- Do not equate best readout layer with best intervention site.
- Do not call within-dataset lexical leakage a blocker by itself; check whether
  the shortcut transfers across the train-to-heldout split used for the claim.
- Do not treat prompt-end and generation-token activations as interchangeable;
  they support different claim ceilings unless both were tested.
- Do not ignore cheap text baselines on the exact split when response text or
  prompt wording could carry the label.
- Do not interpret a nonlinear probe as clean evidence of a simple internal
  feature without stronger controls.

## Dependencies

Core: `torch`, `transformers`, `scikit-learn`, `numpy`
Optional: `sae-lens` (SAE analysis), `datasets` (HF datasets), `accelerate` (multi-GPU)
