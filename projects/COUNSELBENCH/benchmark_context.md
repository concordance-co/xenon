# CounselBench Context

**Date:** 2026-04-23
**Version:** `v1`
**Freeze Date:** `2026-04-23`

Canonical shared rules live in:
- [mech-interp-principles.md](../../docs/mech-interp/mech-interp-principles.md)

## 1. Benchmark Snapshot

- Project page: [CounselBench 2025](https://llm-eval-mental-health.github.io/counselbench-2025/)
- Code: [llm-eval-mental-health/CounselBench](https://github.com/llm-eval-mental-health/CounselBench)
- Eval data: [izi-ano/CounselBench-Eval](https://huggingface.co/datasets/izi-ano/CounselBench-Eval)
- Adv data: [izi-ano/CounselBench-Adv](https://huggingface.co/datasets/izi-ano/CounselBench-Adv)
- Public license surfaced on HF: `cc-by-nc-nd-4.0`
- Scope convention: claims in this context file refer to the public HF datasets and project page reviewed on 2026-04-23.

## 2. Why It Matters

CounselBench is product-relevant because it tests open-ended mental-health advice rather than multiple-choice clinical facts. The useful mechanistic target is not "therapy skill" in general; it is the model's internal posture around empathy, personalization, factual caution, and professional-boundary preservation in high-stakes advice.

## 3. Native Label Surfaces

- `CounselBench-Eval`: 2,000 expert evaluation rows over 100 CounselChat questions, four responder families, repeated expert ratings, numeric scores, span copies, and rationales.
- `CounselBench-Adv`: 120 adversarial prompts represented as 20 rows by six failure-mode columns: `apathetic`, `assumptions`, `symptoms`, `judgmental`, `medication`, and `therapy`.

Important interpretation:

- Eval labels are response-side expert judgments, not direct prompt labels.
- Adv labels are prompt-side trigger families, not proof that a generated answer failed in that way.

## 4. Refined Latent Label Spec

Prompt-side first pass:

- `adv_failure_mode`
- `topic`
- `question_id`
- `prompt_length_bucket`

Response-side first pass:

- `empathy_high`
- `specificity_high`
- `medical_boundary_violation`
- `factuality_low`
- `toxicity_or_judgmental`
- `overall_quality_high`

Tracked nuisance variables:

- `responder`
- `questionID`
- `topic`
- length buckets
- lexical trigger flags for medication, diagnosis, crisis/self-harm, therapy, and boundary/ethics wording

## 5. Known Confounds

- Failure-mode labels in Adv are likely recoverable from surface words.
- Eval quality labels are entangled with responder identity and question topic.
- Multiple expert rows share the same question-response pair, so row-level splits leak annotation target identity.
- Safety labels may be sparse; any generated-response safety label is provisional until manually or expert validated.

## 6. Behavioral Sanity Notes

Current status:

- The benchmark is public and label-rich enough to start.
- A project-local loader and first `pipelines_v2` workflow now define the initial Adv smoke and readout path.
- The workflow can run a tripwire smoke, but a manual/agent inspection step is still required before treating the behavioral gate as passed.

## 7. Strong Candidate Feature Hypotheses

- `adv_failure_mode` -> prompt-side high-risk counseling trigger representation.
- `medical_boundary_violation` -> unsafe clinical-advice / professional-boundary policy state.
- `empathy_high` -> supportive attunement representation.
- `specificity_high` -> personalization/context-use representation.
- `supportive_but_unsafe` vs `safe_but_cold` -> separable warmth and safety-control axes.

## 8. Methods That Look Promising

- Text baselines before activation probes.
- Linear residual probes over prompt-end and generation-end states.
- PCA geometry views over prompt-end failure-family states and generation-end response-posture states.
- Label-support gates before training generated-response safety baselines or probes.
- Grouped splits by `questionID` or adversarial source row.
- Later hard-negative augmentation for supportive-but-unsafe and safe-but-cold contrasts.

## 9. Data Gap List

- Response-side labels for fresh target-model generations need annotation or validated heuristic/LLM labeling before strong claims.
- Adv prompt labels need anti-shortcut controls before any result is promoted beyond diagnostic readout.
- Eval aggregation must preserve expert disagreement and responder identity before probing.
