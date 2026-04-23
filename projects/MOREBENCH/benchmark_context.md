# MoReBench Context

**Date:** 2026-04-23
**Version:** `v2`
**Freeze Date:** `2026-04-23`

Canonical shared rules live in:
- [mech-interp-principles.md](/Users/trentelmore/Projects/concordance/xenon/docs/mech-interp/mech-interp-principles.md)

## 1. Benchmark Snapshot

- Dataset:
  [HF dataset](https://huggingface.co/datasets/morebench/morebench)
- Public split viewer:
  [morebench_public/test](https://huggingface.co/datasets/morebench/morebench/viewer/morebench_public/test)
- Example row discussed:
  [row 36](https://huggingface.co/datasets/morebench/morebench/viewer/morebench_public/test?row=36)
- Public split schema:
  `DILEMMA`, `DILEMMA_SOURCE`, `DILEMMA_TYPE`, `THEORY`, `RUBRIC`, `ROLE_DOMAIN`, `CONTEXT`
- Public split size reviewed:
  `500` rows in `morebench_public/test`
- Theory config exposed in dataset card:
  `morebench_theory`
- Theory split size reviewed:
  `150` rows forming `30` unique dilemmas x `5` theory labels
- Scope convention:
  unless otherwise noted, structural claims here refer to the reviewed public and theory CSVs as loaded from the HF dataset on `2026-04-22`

## 2. Why It Matters

MoReBench is currently one of the strongest benchmark-first substrates for value-conflict and structured deliberation work because it offers:

- dense rubric structure
- multiple response-quality axes
- role distinctions
- prompt-source variation
- a theory-conditioned subset

Its main appeal is not generic "moral reasoning."
It is that the benchmark appears to surface:

- dilemma representation
- objective weighting
- recommendation policy
- role-conditioned control states

## 3. Native Label Surfaces

Top-level fields:

- `DILEMMA`
- `DILEMMA_SOURCE`
- `DILEMMA_TYPE`
- `THEORY`
- `RUBRIC`
- `ROLE_DOMAIN`
- `CONTEXT`

Public rubric dimensions observed:

- `identifying`
- `logical process`
- `clear process`
- `helpful outcome`
- `harmless outcome`
- small `other` bucket

Important note:

- the interesting supervision is mostly inside the stringified `RUBRIC` field
- rubric criteria are weighted and signed
- negative weights are present and are concentrated in `harmless outcome`

## 4. Refined Latent Label View

### Prompt-side candidates

- `action_locus`
  advisor-like vs agent-like setup, derived from `ROLE_DOMAIN` plus prompt text
  currently not probeable on the public split without augmentation
- `dilemma_structure`
  long-case / short-case / expert-case framing, primarily a nuisance or auxiliary label
- `stakeholder_tradeoff_density`
  how many distinct stakeholder or consequence clusters are explicitly live in the prompt
- `domain_topic`
  broad scenario domain, primarily nuisance
- `theory_identity`
  high-priority augmentation target rather than a current clean prompt-side label
- `source_template`
  kept as nuisance variable
- `length_bucket`
  kept as nuisance variable

### Response-side candidates

- `tradeoff_engagement`
- `commitment_style`
- `refuses_or_hedges`
- `helpfulness_invoked`
- `harm_avoidance_invoked`
- `uncertainty_and_scope_calibration`

### Validation-only or caution labels

- `Helpful Outcome`
  useful downstream validation signal
- `Harmless Outcome`
  useful downstream validation signal
- `Identifying`
  not a direct target in current form
- `Logical Process`
  likely too noisy as a direct latent target
- `Clear Process`
  likely too noisy as a direct latent target

## 5. Known Confounds

Most important current confounds:

- `DILEMMA_SOURCE` is heavily aliased with role and topic
- `DILEMMA_SOURCE` is also heavily aliased with `DILEMMA_TYPE`
- `ai_agent` rows are almost entirely tied to `ai_risk_dilemmas`, with only `7` public `expert_written_collab` exceptions
- source also predicts topic mix and writing style
- rubric titles, especially `Identifying`, are often case-specific grader instructions
- prompt length and type vary by source
- theory is not automatically a prompt-side variable just because a `THEORY` field exists

Observed public-split pattern from review:

- `daily_dilemmas` were all `ai_advisor`
- `ai_risk_dilemmas` were all `ai_agent`
- `expert_written_ethic_bowl`, `expert_written_ethic_unwrapped`, and `expert_written_literature` were all `ai_advisor`
- `expert_written_collab` contributed the only public `expert_case` agent rows

This means advisor-vs-agent is interesting, but currently high-confound.
More strongly: it is effectively not probeable on the current public split without augmentation, because the number of source-controlled usable mixed-role cells is zero.

## 5.1 Benchmark-Specific Gotchas

- The public viewer path is under `morebench/morebench`, but practical loading may differ depending on tooling or mirrors; always verify the actual load path used in code.
- Dataset docs and actual field values may differ in naming conventions; verify the concrete values in the loaded split before building labelers.
- Treat viewer descriptions and README summaries as suggestive, not authoritative, until checked against the actual rows.
- The theory split is structurally paired at the evaluator level: `30` unique dilemma texts appear under all `5` theory labels.
- But those repeated theory rows share the same `DILEMMA` text, so theory should not be treated as prompt-side signal unless the actual runtime prompt explicitly injects the `THEORY` field.
- This still looks like one of the easiest and highest-value augmentation opportunities: build matched prompts that expose theory explicitly while preserving the paired dilemma structure.
- Phase 02 now keeps the first explicit-theory repair slice as known-broken for `theory_identity` and materializes a second shortcut-stress-test slice:
  legacy `150` direct theory-exposed prompt variants, `150` same-label wording variants, `30` structurally matched neutral controls, plus `2250` harder theory repair rows and `180` theory repair controls under the MoReBench phase 02 outputs.
- The new theory repair slice includes name-only, alias-only, description-only, and name-plus-description families plus prompt-side cheap-baseline preflight.
- Current prompt-side read:
  `alias_only` is now the strongest prompt-side diagnostic family, but its strongest held-out text baseline is still too high (`0.675`) for a clean prompt-side retry; `description_only` is better treated as a generation-time priming family, and its strongest held-out description baseline remains `1.0`.
- Phase 02 also now has a real `pipelines_v2` behavioral smoke run on a `20`-prompt stratified augmented slice.
  On the provisional smoke model (`Qwen/Qwen3-30B-A3B` mounted under Xenon), the current benchmark-faithful phase-02 smoke protocol passes:
  `nonempty_rate = 1.0`, `recommendation_present_rate = 1.0`, `manual_pass_rate = 1.0`, overall decision `pass`.
- The smoke interpretation is now stronger than a tripwire pass.
  The `20` existing responses were re-inspected directly against the proposed response-side labels, and the current judgment is that the augmented slice is substantively labelable for:
  `tradeoff_engagement`, `commitment_style`, `refuses_or_hedges`, `helpfulness_invoked`, `harm_avoidance_invoked`, and `uncertainty_and_scope_calibration`.
- `CONTEXT` is partially missing in the theory split.

## 6. Behavioral Sanity Notes

What has been checked so far:

- schema inspection
- direct CSV inspection for `morebench_public` and `morebench_theory`
- public row review
- rubric distribution review
- exact source/role/type cross-tab review
- theory pairing review
- high-level label-pattern analysis
- generation-time behavioral smoke on a `20`-prompt augmented slice via `pipelines_v2`
- direct response-level labelability inspection of that smoke slice against the proposed response-side labels
- phase-03 analysis planning artifacts under `projects/MOREBENCH/phase_03/docs/03-*`
- phase-03 Experiment 1 execution on `theory_identity`, with result now routed as `AUGMENTATION_NEEDED` because the cheap `anchor_text` baseline was also perfect

What has not yet been completed:

- response-side grading on fresh generations under the intended protocol
- the phase-03 response-label pilot / freeze workflow
- a prompt-side theory family that beats the stronger held-out alias text baselines cleanly enough for retry
- generation-time theory-persistence execution on matched theory-primed responses
- validated gold slice for `stakeholder_tradeoff_density`

Current gate judgment:

- phase 02 smoke exists and now includes substantive labelability inspection on the current gate model
- that is sufficient to proceed with phase-03 planning and response-label pilot design without rerunning smoke immediately
- any newly added execution model should satisfy the same labelability standard before its responses are used for probing

### Worked Example

The row 36 Rwanda-genocide case is a good reminder that hard cases often mix multiple moral structures at once.
It is useful as a caution against forcing binary philosophical axes too early when the operational question is really about how many live considerations the model keeps active before commitment.

## 7. Strong Candidate Feature Hypotheses

- `helpful vs harmless`
  likely the strongest first-pass objective-orientation contrast, but best treated as two separable labels before any combined relation label
- `advisor vs agent`
  plausible control-state distinction, but blocked on augmentation in the current public split
- `multi-consideration representation`
  strongest benchmark-native prompt/generation bridge
- `tradeoff-engaged vs single-axis collapse`
  good response-side deliberative-process target

### Label -> Feature Hypothesis Sketch

- `Helpful Outcome` -> helpfulness-oriented response policy
- `Harmless Outcome` -> harm-avoidance response policy
- `Helpful Outcome x Harmless Outcome` -> objective weighting or tradeoff handling
- `ROLE_DOMAIN` -> advisor-vs-agent control-state difference

## 8. Methods That Look Promising

For first-pass readout:

- linear probes
- difference-in-means directions
- layer/token sweeps
- generation-time token-position probes
- paired-generation theory-prime comparisons

For localization:

- span-aware or section-aware probes
- generation-time capture
- commitment-transition analysis
- held-out alias-bank transfer checks for prompt-side theory diagnostics
- generated-token-only readout for theory persistence

For follow-up:

- transfer tests across source/template slices
- intervention only after matched/control design is improved
- residualized readout for helpfulness-vs-harm-avoidance separability

## 9. Methods Or Hypotheses To Be Careful About

- theory overlay claims from `morebench_theory`
  paired at the evaluator level, but still not automatically prompt-side
- obligation-vs-prohibition taxonomy
  conceptually interesting, but likely low-yield without dedicated annotation
- direct probing of raw `Identifying`
  too case-specific in current form

## 10. Data Gap List

- response generations under the intended model/protocol
- matched theory-primed generation slice using the `description_only` family plus generic-ethics controls
- matched or rewritten advisor/agent contrasts
- source-balanced or template-balanced slices
- structure-normalized, length-matched, and person-grammar control variants
- behavioral smoke rerun on the final target model
- possible counterbalanced rewrites for prompt-side structure labels
- validated gold slice for `stakeholder_tradeoff_density`

## 11. Open Questions

- how cleanly separable is `helpful` vs `harmless` in generation-time activations?
- does advisor-vs-agent survive once source/template confounds are controlled?
- when in the generation does the model move from dilemma representation to recommendation commitment?
- can tradeoff engagement be labeled reliably from generated responses at useful scale?
- does theory conditioning persist into generated reasoning once prompt tokens are out of view?
- can an alias-only theory family ever beat stronger held-out text baselines well enough to be more than a prompt-side diagnostic?

## 12. Phase 03 Plan Snapshot

Current phase-03 priority order:

- `theory_conditioned_generation_persistence`
  main theory target
  current read on `2026-04-23`: `description_only` should be treated as a theory-priming family for generated-response experiments rather than as a clean prompt-side retry family
- `theory_identity`
  prompt-side diagnostic target only
  current read on `2026-04-23`: `alias_only` is the best current diagnostic family, but the prompt-side retry gate remains closed because the strongest held-out alias text baseline is still `0.675`
- `tradeoff_engagement`
  first response-side deliberation target
- `commitment_style`
  first response-side transition target
- `helpfulness_invoked` and `harm_avoidance_invoked`
  response-side pilot targets, with an explicit generate -> annotate -> validate -> freeze gate before probing

Phase-03 execution artifacts now also include:

- `03-execution-targets.md`
  multi-model-friendly execution policy and current gate-model config
- `03-response-label-pilot.md`
  response-side generation / annotation / validation / freeze contract
- `03-phase-04-entry-criteria.md`
  explicit promotion gate for causal follow-up
- `03-stakeholder-tradeoff-density-gold-slice-spec.md`
  concrete prerequisite for the deferred prompt-side density label
  first separability target

Methods to be careful about:

- broad `action_locus` claims from the current 10-pair starter rewrite set
- any `stakeholder_tradeoff_density` result before gold-slice validation
- any causal interpretation before split-robust localized representational evidence exists
