---
benchmark: healthbench
phase: 00
version: v1
frozen_date: 2026-04-28
input_artifacts:
  - https://github.com/openai/simple-evals
  - https://openai.com/index/healthbench/
  - https://github.com/openai/simple-evals/blob/main/healthbench_eval.py
  - https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl
---

# HealthBench Validation Memo

## Benchmark Identity

HealthBench is OpenAI's public health-assistant evaluation released in
`openai/simple-evals`. OpenAI describes it as 5,000 realistic health
conversations with physician-created, example-specific rubrics. The project will
start from HealthBench Consensus, the 3,671-example subset whose criteria were
filtered for physician consensus.

Initial probe-target model: `/models/Qwen/Qwen3-30B-A3B`.

## Access And Usability

Recommendation: proceed.

The benchmark has public source code, public JSONL blob paths, an MIT-licensed
reference eval repository, and a simple runnable harness through
`healthbench_eval.py`. The public implementation exposes the full, Consensus,
and Hard source blobs and a grader prompt that evaluates each response against
rubric items.

The schema is usable for Xenon ingestion: each row has a stable prompt id,
chat-message prompt, example tags, rubric items with criteria/points/tags, and
optional physician completion metadata. Durable ingestion should go to Neon via
a repeatable uploader. Raw rows should not be committed to the repository.

## Label Richness

HealthBench is label-rich enough for deeper benchmark-first work.

Useful native surfaces include:

- example-level tags for theme/category structure
- rubric-item tags for axis-style scoring
- per-example physician-written criteria
- signed point values, including negative criteria for undesirable behaviors
- optional physician completion metadata for baseline comparisons

These labels still need translation. Rubric and score surfaces were designed to
grade responses, not to directly define internal latent variables.

## Product Relevance

Product relevance is high. The task resembles realistic health-assistant use:
users and clinicians ask health questions, often under uncertainty, and the
assistant must balance helpfulness, safety, escalation, scope limits, and
audience-appropriate communication.

Potential downstream monitors or interventions would matter if they track:

- missing emergency escalation
- overconfident medical advice under uncertainty
- failure to seek context on underspecified questions
- response-depth miscalibration
- clinician-vs-layperson communication mismatch

## Plausible Mechanistic Questions

- Does the model represent a context-needed state before answering
  underspecified health prompts?
- Does a triage-urgency or emergency-escalation state appear before the model
  commits to advice?
- Is scope calibration separable from generic refusal or generic safety style?
- Does the model encode clinician-vs-layperson audience framing independently
  of topic and vocabulary?
- Can response-depth control be separated from answer correctness and
  verbosity?

## Major Confounds

- Topic and label leakage: emergency and specialty terms may make some labels
  text-recoverable.
- Rubric leakage: rubric criteria should not be included in generation prompts
  for activation work unless the target explicitly studies rubric-following.
- Response lexical leakage: generated health responses will often contain
  label-adjacent words like emergency, context, uncertainty, or referral.
- Length and turn-count effects: longer prompts may correlate with complexity,
  rubric count, and score.
- Consensus selection: Consensus improves label trust but may narrow examples
  toward easier-to-agree criteria.
- Hard selection: Hard membership is model-performance-conditioned and should
  not be treated as a clean clinical-risk label.

## Recommendation

Proceed to `benchmark-to-latent-labels`.

Start Phase 01 with HealthBench Consensus only. Freeze a small ontology before
activation capture: one or two prompt-side labels, one response-side orientation
label if Qwen generations are in scope, and a tracked nuisance set. Require
cheap text baselines and nuisance-stratified cell-size checks before promoting
any readout result.
