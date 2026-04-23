---
benchmark: morebench
phase: 03
experiment: experiment_01_theory_identity
date: 2026-04-23
---

# Experiment 01: Theory-Identity Prompt Readout

## What Ran

- dataset:
  `experiment_01_prompt_dataset.jsonl`
- model:
  `/models/Qwen/Qwen3-30B-A3B`
- prompt families:
  `theory_direct`, `theory_wording_variant`, `anchor_only`, `neutral_control`
- completed readouts:
  - prompt-end probe on `theory_or_control`
  - prompt-family transfer on `theory_identity`
  - named-theory-clause probe on `theory_identity`
  - text baseline on `anchor_text`

## Headline Result

The readout is extremely strong, but the experiment is still not a good mechanistic target in its current form.

- prompt-end readout reaches balanced accuracy `1.0`
- transfer across `theory_direct`, `theory_wording_variant`, and `anchor_only` also reaches balanced accuracy `1.0`
- named-theory-clause localization reaches balanced accuracy `1.0`
- but the cheap `anchor_text` baseline is also balanced accuracy `1.0`

## Scientific Take

This means the current phase-02 theory prompt family makes theory identity trivially available from explicit prompt semantics.

That is still useful process evidence:

- the current anti-shortcut control is not hard enough
- perfect transfer across `anchor_only` does not show abstraction here
- the benchmark-first process caught the shortcut before we promoted the family to causal work

## Routing

- experiment verdict:
  `TRIVIAL_OR_NULL`
- immediate consequence:
  do not promote `theory_identity` in its current prompt form
- best next move:
  proceed to the response-side pilot path, and only revisit theory after stronger phase-02 anti-shortcut augmentation
