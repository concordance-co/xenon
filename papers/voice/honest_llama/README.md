# Honest LLaMA / ITI

Goal: reproduce a Xenon-native residual ITI smoke path for TruthfulQA-style
truthful-vs-untruthful contrasts: direction discovery, validation selection,
steering, and project-out comparison.

Smoke command:

```bash
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/honest_llama/specs/workflow.py
```

Expected artifacts: capture, candidate truthfulness direction, truthfulness
scores, selected direction, ablation subspace, baseline generation,
truthfulness-steered generation, project-out generation, and report summary.

Real run default: `Qwen/Qwen3-8B`. LLaMA-family ITI runs are optional and may
require gated weights.

Data hooks:

- `truthfulqa_generation_dataset()` loads TruthfulQA question rows for
  evaluation-style captures.
- `truthfulqa_answer_contrast_dataset()` expands correct and incorrect answers
  into truthful/untruthful contrast rows for residual direction discovery.
- `prompt_template=...` lets users reuse the same truthfulness labels with
  their own chat, QA, or agent-answer formatting.

Claim boundary: this is residual-direction ITI plumbing, not head-specific ITI.
Paper-level claims require TruthfulQA evaluation, helpfulness tradeoff checks,
and attention-head intervention support.
