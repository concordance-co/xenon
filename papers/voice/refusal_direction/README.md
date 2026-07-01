# Refusal Direction

Goal: reproduce the reusable Xenon workflow for harmful-vs-harmless refusal
directions: direction discovery, validation-layer selection, projection scoring,
add-direction steering, and project-out ablation.

Smoke command:

```bash
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/refusal_direction/specs/workflow.py
```

Expected artifacts: capture, candidate refusal direction, refusal scores,
selected direction, one-component ablation subspace, baseline generation,
activation-add generation, project-out generation, and report summary.

Real run default: `Qwen/Qwen3-8B`. Gated Llama-family models can be configured
by swapping the engine model id.

Data hooks:

- `refusal_direction_split_dataset(split="train")` loads the repo's harmful and
  harmless split JSON files as a deferred dataset.
- `refusal_direction_processed_dataset("advbench")` loads one processed source
  dataset from the paper repo.
- Both helpers accept `prompt_template=...`, so product teams can preserve the
  harmful/harmless labels while changing the instruction format.

Claim boundary: the smoke does not include jailbreak/safety scoring or CE-loss
evaluation. Paper-level claims need harmful/harmless train/validation splits,
behavioral refusal metrics, harmless capability checks, and destabilization
controls.
