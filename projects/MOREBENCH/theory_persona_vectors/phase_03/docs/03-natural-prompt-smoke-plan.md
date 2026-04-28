---
project: MOREBENCH
subproject: theory_persona_vectors
phase: 03
artifact: natural_prompt_smoke_plan
date: 2026-04-26
status: draft
---

# Phase 03 Natural-Prompt Smoke

## Purpose

Phase 02 showed that paired theory-prime vs neutral/default deltas are stable at
L32 response-mean, but it left two unresolved problems:

- the generated responses were extremely terse because the prompt explicitly
  said "Give only the recommendation. Do not explain."
- primed-vs-neutral directions were highly cosine-similar across theories,
  suggesting a large shared moral-prime component.

This phase removes the over-prescriptive answer constraint. The first natural
prompt asked `What would you recommend?` and proved lexically too loud. The
active brief variant asks: `Briefly describe your recommendation to the
scenario.`

## Design

Reuse the same 30 synthetic dilemmas and all theory-pole hypotheses:

- positive theory primes for deontology, utilitarianism, virtue ethics, and
  contractualism
- positive variants for each theory
- neutral recommendation prompt
- generic moral-prime anchor
- anti-theory diagnostics
- theory-vs-theory diagnostics computed from positive conditions

The generic moral anchor is not the primary negative pole. It is a third anchor
for specificity:

- `generic_moral - neutral` estimates the shared moral-prime direction
- `theory - neutral` estimates theory-prime direction
- cosine and residualization against `generic_moral - neutral` test
  theory-specificity

## Unit Of Analysis

All same-dilemma condition contrasts must be analyzed as paired deltas:

`activation(condition_a, dilemma_i) - activation(condition_b, dilemma_i)`

Nulls must recompute the same statistic under paired sign flips. Do not use the
old unpaired fake-direction-vs-real-direction null.

## Smoke Questions

Before treating the capture as interpretable:

- Do responses become longer under the natural prompt?
- Does the model spontaneously explain or still answer tersely?
- Do positive theory primes produce distinguishable response styles?
- Does `generic_moral - neutral` explain most of the theory-prime direction?
- Do theory-vs-theory paired directions become cleaner than theory-vs-neutral
  after the paired correction?

## Planned Run

- 30 dilemmas
- 15 conditions
- one generation per dilemma-condition
- max tokens: 384 for the full capture run
- temperature: 0.7
- captured layers: 0, 4, 16, 24, 32, 40
- capture sites: prompt_end and generated_sequence

This is still a smoke, not the scale run. If it works, later phases can add
independent samples and broader layer/locus sweeps.

## Behavior-First Execution

Before paying for full capture, run a behavior-only smoke on a small dilemma
slice. This is not a claim gate; it checks whether the natural prompt actually
lets the model write enough content for a persona-vector-style response-state
direction to have room to appear.

Behavior-smoke readouts:

- response token and character distribution by condition
- qualitative examples for neutral, generic moral, positive theory, anti-theory
- crude text-divergence table by same-dilemma condition pairs
- whether the model spontaneously explains, hedges, or remains terse

If the behavior smoke still produces mostly bare 5-10 token imperatives, the
next loopback is prompt design, not capture. If it produces natural
recommendation paragraphs or at least multi-clause recommendations, proceed to
full capture and paired sign-flip analysis.

Length-finished rows are kept for phase 03 capture and explicitly labeled.
The natural prompt is expected to produce longer responses, and dropping
length-finished rows would select against the persona-expressive tail. The
analysis report must include finish-reason counts so truncation remains visible.
