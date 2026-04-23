---
benchmark: morebench
phase: 03
version: v2
frozen_date: 2026-04-23
input_artifacts:
  - projects/MOREBENCH/phase_01/docs/01-latent-label-spec.md
  - projects/MOREBENCH/phase_02/docs/02-augmented-data-manifest.json
  - projects/MOREBENCH/phase_02/docs/02-behavioral-smoke-report.md
  - projects/MOREBENCH/phase_03/docs/03-execution-targets.md
  - projects/MOREBENCH/phase_03/docs/03-response-label-pilot.md
---

# MoReBench 03 Experiment Specs

## Experiment 1: Theory-Alias Prompt Diagnostic

- target label family:
  `theory_identity`
- execution target:
  any declared phase-03 execution model recorded in `03-execution-targets.md`
- data:
  phase-02 `alias_only` shortcut-stress-test family only
- method:
  prompt-side linear probes plus difference-in-means directions
- capture regime:
  full prompt pass; sweep alias-bearing cue spans and final prompt token across a layer range
- primary shortcut risk:
  the probe succeeds by reading alias-surface regularities that still generalize across held-out banks
- required controls:
  held-out alias-bank transfer, strongest held-out text baseline suite from phase 02, and neutral/generic ethics controls where useful
- success criteria:
  readout survives held-out alias banks and beats the strongest held-out alias text baseline by a meaningful margin
- failure criteria:
  signal is matched by the held-out alias text baseline or collapses when alias banks are held out
- claim ceiling if successful:
  diagnostic representational evidence only, not a main benchmark claim

## Experiment 2: Theory-Conditioned Generation Persistence

- target label family:
  `theory_conditioned_generation_persistence`
- execution target:
  same declared model and generation batch as the response-label pilot
- data:
  fresh generations for matched dilemmas under `description_only` primes plus generic-ethics controls
- label-freeze prerequisite:
  generation batch must be materialized and direct theory-copying cases must be flagged before probing begins
- method:
  paired-generation comparison plus generation-time probes on generated tokens only
- capture regime:
  generated reasoning tokens and recommendation spans, with prompt tokens excluded from the main readout
- primary shortcut risk:
  the readout succeeds only because generated text explicitly repeats theory names or near-verbatim prime language
- required controls:
  same-dilemma prime swaps, generated-text lexical baselines, explicit theory-copying flags, and generic-ethics control generations
- success criteria:
  generated-token readout distinguishes theory-conditioned responses on matched dilemmas, survives generated-text lexical baselines, and is not reducible to direct theory copying
- failure criteria:
  signal collapses after controlling for generated-text lexical cues or is driven mainly by direct prime copying
- claim ceiling if successful:
  localized representational

## Experiment 3: Generation-Time Commitment Transition Pilot

- target label family:
  `commitment_style`
- execution target:
  same declared model and generation batch as the response-label pilot
- data:
  fresh generations defined and frozen through `03-response-label-pilot.md`
- label-freeze prerequisite:
  response-side labeled slice must be generated, annotated, validated, and frozen before probing begins
- method:
  tokenize generated responses, label commitment style, then train token-position probes
- capture regime:
  generated response tokens, compared across early / middle / late windows plus explicit conclusion-span baselines
- primary shortcut risk:
  the readout collapses into conclusion-token classification rather than detecting a genuine transition into commitment
- required controls:
  conclusion-span vs non-conclusion-span comparison, family holdouts, and label-shuffled sanity checks
- success criteria:
  there is a stable late-generation transition band predictive of direct recommendation vs hedge / refusal, and the signal generalizes beyond the explicit conclusion span
- failure criteria:
  only final-token memorization works, the signal vanishes outside the conclusion span, or it fails under family holdouts
- claim ceiling if successful:
  localized representational

## Experiment 4: Tradeoff-Engagement Response Readout

- target label family:
  `tradeoff_engagement`
- execution target:
  same declared model and generation batch as Experiment 2
- data:
  same frozen response-side slice as Experiment 2
- label-freeze prerequisite:
  use the validated freeze produced by `03-response-label-pilot.md`
- method:
  response-side annotation pilot plus generation-time linear probes
- capture regime:
  generated reasoning tokens before the final recommendation span
- primary shortcut risk:
  the probe reads generic verbosity or response length instead of sustained multi-consideration reasoning
- required controls:
  response-length baselines, generic verbosity baselines, family holdouts, and label-shuffled sanity checks
- success criteria:
  probe distinguishes sustained multi-consideration engagement from collapsed single-axis reasoning across prompt families and beats simple verbosity / length baselines
- failure criteria:
  signal is explained by response length, verbosity, or conclusion-only cues
- claim ceiling if successful:
  representational or localized representational

## Experiment 5: Helpfulness vs Harm-Avoidance Separability

- target label family:
  `helpfulness_invoked`, `harm_avoidance_invoked`
- execution target:
  same declared model and generation batch as Experiments 2 and 3
- data:
  same frozen response-side slice as Experiments 2 and 3
- label-freeze prerequisite:
  use the validated freeze produced by `03-response-label-pilot.md`
- method:
  separate readouts for each label, then residualized and transfer comparisons
- capture regime:
  late-generation tokens and conclusion spans, compared to non-conclusion windows where useful
- primary shortcut risk:
  both readouts reduce to generic assertive-vs-cautious tone or broad recommendation caution instead of separable policy orientation
- required controls:
  residualized comparison, assertive-vs-cautious baseline, response-length baseline, and family holdouts
- success criteria:
  each readout retains usable signal after controlling for the other, does not collapse into the assertive-vs-cautious baseline, and shows at least partial split robustness
- failure criteria:
  one label fully subsumes the other or both reduce to generic caution / tone
- claim ceiling if successful:
  representational separability, not yet mechanistic separability

## Experiment 6: Stakeholder-Tradeoff-Density Prompt Pilot

- target label family:
  `stakeholder_tradeoff_density`
- prerequisite:
  `projects/MOREBENCH/phase_03/docs/03-stakeholder-tradeoff-density-gold-slice-spec.md`
- execution target:
  same declared model family as Experiment 1 unless the gold-slice validation picks a better prompt-side target
- data:
  public prompts plus any repaired phase-02 slices where needed
- method:
  prompt-end probe and geometry check
- capture regime:
  prompt-only, with section-aware span comparisons if prompt metadata is available
- primary shortcut risk:
  signal is dominated by prompt length, source family, or dilemma structure rather than stakeholder-density content
- required controls:
  prompt-length baseline, source/topic-aware split, dilemma-structure baseline, and validated gold slice
- success criteria:
  signal tracks validated density labels and survives source/topic-aware splits
- failure criteria:
  signal is dominated by length, source, or dilemma structure
- claim ceiling if successful:
  representational
