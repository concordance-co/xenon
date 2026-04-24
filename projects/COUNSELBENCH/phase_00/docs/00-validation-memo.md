---
benchmark: counselbench
phase: 00
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - https://llm-eval-mental-health.github.io/counselbench-2025/
  - https://github.com/llm-eval-mental-health/CounselBench
  - https://huggingface.co/datasets/izi-ano/CounselBench-Eval
  - https://huggingface.co/datasets/izi-ano/CounselBench-Adv
---

# CounselBench Validation Memo

## Benchmark Identity

CounselBench is a mental-health counseling QA benchmark with two public surfaces:

- `CounselBench-Eval`: 2,000 expert evaluation rows over model and human responses to CounselChat questions.
- `CounselBench-Adv`: 120 expert-authored adversarial prompts arranged as 20 rows by six failure-mode columns.

The HF dataset pages report license `cc-by-nc-nd-4.0`. Treat this as a research/evaluation dataset unless licensing is re-reviewed for any product use.

## Access And Usability

Recommendation: proceed.

The public HF datasets expose prompt text, response text, model/human responder identity, expert scores, span-copy fields, rationales, and adversarial failure-mode prompt families. This is enough to build deferred `pipelines_v2` loaders and run activation capture on prompt and generated-response contexts.

## Label Richness

CounselBench is label-rich enough for benchmark-first work:

- rubric dimensions: overall quality, empathy, specificity, medical advice, factual consistency, toxicity
- span-copy/rationale fields for medical advice, factual consistency, and toxicity
- adversarial prompt families for six failure modes
- responder and topic metadata

Main caveat: Eval labels are response-side expert judgments. They are not prompt-side labels and must be aggregated by question-response pair before probing.

## Product Relevance

The benchmark is directly relevant to high-stakes assistant behavior in mental-health advice. Useful downstream artifacts could include safety-boundary monitors, empathy/specificity readouts, and warning signals for supportive-but-unsafe generations.

## Plausible Mechanistic Questions

- Does the model represent adversarial counseling trigger family before answering?
- Does a medical-boundary or unsafe-clinical-advice state emerge during generation?
- Are empathy and specificity separable from safety caution?
- Can supportive-but-unsafe and safe-but-cold response states be separated?
- Does the model encode topic/crisis risk independently of surface lexical triggers?

## Major Confounds

- Adv failure-mode families are likely text-recoverable from obvious words.
- Eval labels are entangled with responder identity, topic, and question.
- Repeated expert annotation rows create leakage if row-level splits are used.
- Generated-response labels are not expert labels unless separately annotated or validated.

## Recommendation

Proceed to latent-label formation and a conservative first workflow. Start with CounselBench-Adv for balanced smoke/readout work. Defer full CounselBench-Eval probing until aggregation and label-freeze artifacts are in place.

