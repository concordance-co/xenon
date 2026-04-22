# MoReBench Context

**Date:** 2026-04-22
**Version:** `v1`
**Freeze Date:** `2026-04-22`

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
- Scope convention:
  unless otherwise noted, structural claims here refer to the `500` reviewed rows of `morebench_public/test`

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

## 4. Refined Latent Label View

### Prompt-side candidates

- `role_framing`
  advisor-like vs agent-like setup
- `subject_of_action`
  AI system, human actor, institutional actor
- `person_grammar`
  first / second / third person framing
- `information_stance`
  facts stipulated vs facts uncertain
- `stakes_framing`
  life-safety / welfare / relational / institutional / mixed
- `source_template`
  kept as nuisance variable
- `length_bucket`
  kept as nuisance variable

### Response-side candidates

- `considerations_represented`
- `uncertainty_markers_present`
- `commitment_transition`
- `refuses_or_hedges`
- `harm_avoidance_invoked`
- `helpfulness_invoked`

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
- `ai_agent` rows are largely tied to `ai_risk_dilemmas`
- source also predicts topic mix and writing style
- rubric titles, especially `Identifying`, are often case-specific grader instructions
- prompt length and type vary by source

Observed public-split pattern from review:

- `daily_dilemmas` were all `ai_advisor`
- `ai_risk_dilemmas` were all `ai_agent`

This means advisor-vs-agent is interesting, but currently high-confound.

## 5.1 Benchmark-Specific Gotchas

- The public viewer path is under `morebench/morebench`, but practical loading may differ depending on tooling or mirrors; always verify the actual load path used in code.
- Dataset docs and actual field values may differ in naming conventions; verify the concrete values in the loaded split before building labelers.
- Treat viewer descriptions and README summaries as suggestive, not authoritative, until checked against the actual rows.

## 6. Behavioral Sanity Notes

What has been checked so far:

- schema inspection
- public row review
- rubric distribution review
- high-level label-pattern analysis

What has not yet been completed:

- generation-time behavioral sanity on the target model
- parseability check for generated responses
- model-fit check on a stratified prompt slice

This should happen before any probe or intervention work.

### Worked Example

The row 36 Rwanda-genocide case is a good reminder that hard cases often mix multiple moral structures at once.
It is useful as a caution against forcing binary philosophical axes too early when the operational question is really about how many live considerations the model keeps active before commitment.

## 7. Strong Candidate Feature Hypotheses

- `helpful vs harmless`
  likely the strongest first-pass objective-orientation contrast
- `advisor vs agent`
  plausible control-state distinction, but needs confound care
- `dilemma representation vs recommendation selection`
  likely a generation-trace distinction rather than a prompt-side label
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

For localization:

- span-aware or section-aware probes
- generation-time capture
- commitment-transition analysis

For follow-up:

- transfer tests across source/template slices
- intervention only after matched/control design is improved

## 9. Methods Or Hypotheses To Be Careful About

- theory overlay claims from `morebench_theory`
  currently likely underpaired and confounded
- obligation-vs-prohibition taxonomy
  conceptually interesting, but likely low-yield without dedicated annotation
- direct probing of raw `Identifying`
  too case-specific in current form

## 10. Data Gap List

- response generations under the intended model/protocol
- matched or rewritten advisor/agent contrasts
- source-balanced or template-balanced slices
- clearer paired theory variants if theory is a priority
- possible counterbalanced rewrites for prompt-side structure labels

## 11. Open Questions

- how cleanly separable is `helpful` vs `harmless` in generation-time activations?
- does advisor-vs-agent survive once source/template confounds are controlled?
- when in the generation does the model move from dilemma representation to recommendation commitment?
- can tradeoff engagement be labeled reliably from generated responses at useful scale?
