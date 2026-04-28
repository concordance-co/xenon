---
project: MOREBENCH
subproject: theory_persona_vectors
phase: 02
artifact: paired_tail_reanalysis
date: 2026-04-26
status: analysis_addendum
---

# Phase 02 Addendum — Paired Tail Reanalysis

## Why This Exists

The original phase 02 analysis used an unpaired difference-in-means estimator
and compared its split-half cosine to a non-comparable random-label statistic.
That made the smoke gate hard to interpret.

This addendum reruns the existing phase 02 capture with:

- paired per-dilemma deltas: `activation(pos, dilemma_i) - activation(neg, dilemma_i)`
- repeated split-half cosine over paired deltas
- sign-flip nulls that recompute the same split-half statistic
- response-length filters to test the "terse substrate" diagnosis

No new generation or capture was run.

## Artifacts

- script:
  `projects/MOREBENCH/theory_persona_vectors/phase_02/scripts/analyze_all_theories_paired_tail.py`
- report:
  `projects/MOREBENCH/theory_persona_vectors/phase_02/reports/all_theories_paired_tail_analysis/report.md`
- summary:
  `projects/MOREBENCH/theory_persona_vectors/phase_02/reports/all_theories_paired_tail_analysis/summary.json`

Input artifacts:

- generation rows:
  `generation_run_1_15bc125de56b`
- capture:
  `capture_1_c2684db0530c`
- site/layer:
  `generated_sequence_residual`, L32

## Response-Length Diagnosis

The substrate is indeed very terse:

- mean generated tokens: `10.44`
- median generated tokens: `9`
- share under 10 tokens: `0.552`
- share under 20 tokens: `0.912`

This supports the concern that the response-mean locus is averaging mostly
bare recommendation tokens rather than theory-expression tokens.

However, the tail is too small to use as a decisive existing-capture test:

- deontology positive rows with `>=20` tokens: `5`
- utilitarian positive rows with `>=20` tokens: `2`
- virtue positive rows with `>=20` tokens: `8`
- contractualism positive rows with `>=20` tokens: `5`

So the existing capture confirms the substrate is too terse, but it cannot
cleanly answer whether the long-response tail rescues the recipe.

## Corrected Paired Smoke

The bigger correction is statistical. With paired deltas and a comparable
sign-flip null, the "marginal" phase 02 result becomes much stronger.

Using all rows at L32 generated:

| theory | construction | n | real median split-half | null p95 | gap |
|---|---|---:|---:|---:|---:|
| deontology | neutral_length_matched | 30 | 0.825 | 0.375 | 0.449 |
| utilitarian | neutral_length_matched | 30 | 0.770 | 0.345 | 0.424 |
| virtue_ethics | neutral_length_matched | 30 | 0.847 | 0.424 | 0.423 |
| contractualism | neutral_length_matched | 30 | 0.834 | 0.402 | 0.432 |

This means the previous "marginal across the board" interpretation was mostly
an artifact of the unpaired / wrong-null analysis. The paired design contains
stable condition-specific signal.

## Tail Filters

Filtering to positive responses with at least 10 tokens preserves strong
neutral-length-matched gaps:

| theory | n | gap |
|---|---:|---:|
| deontology | 17 | 0.356 |
| utilitarian | 13 | 0.260 |
| virtue_ethics | 20 | 0.409 |
| contractualism | 21 | 0.359 |

Filtering both sides to at least 10 tokens also preserves strong gaps:

| theory | n | gap |
|---|---:|---:|
| deontology | 15 | 0.277 |
| utilitarian | 11 | 0.207 |
| virtue_ethics | 17 | 0.351 |
| contractualism | 16 | 0.330 |

Filtering at 20 tokens is not interpretable because the resulting N is too
small for most theories.

## Revised Interpretation

The terse-substrate diagnosis is real but not sufficient. The existing phase 02
capture does contain stable paired signal at L32 response-mean. The prior
"marginal" result was primarily a problem with the analysis gate, not proof that
the recipe was failing.

The remaining hard problem is **specificity**:

- primed-vs-neutral directions remain highly cosine-similar across theories
- neutral-length-matched cross-theory cosines remain roughly `0.64–0.87`
- this can still be generic moral-framework priming, instruction style, or
  response-length/style, not theory-specific semantics

So the corrected claim is:

> At L32 response-mean, paired theory-prime vs neutral/default deltas are stable
> above a sign-flip null even on terse responses. This reopens the
> persona-vector track, but the current evidence does not yet establish
> theory-specific directions because the four primed-vs-neutral directions
> still share a large common component.

## Consequences For Next Phase

Drop these as load-bearing:

- the original unpaired phase 02 smoke table
- the claim that theory-vs-theory is dead because of negative gaps under that
  table
- the claim that the recipe is merely marginal at N=30

Keep these as load-bearing:

- the substrate is too terse for a faithful persona-vectors replication
- paired analysis is mandatory for all same-dilemma condition contrasts
- sign-flip nulls should replace fake-direction-vs-real-direction nulls
- shared moral-prompt centroid / specificity controls are mandatory

Next phase should run an elaborated-response substrate:

- same 30 dilemmas
- same 14 theory conditions
- prompt asks for recommendation plus 2–3 sentences of reasoning
- max tokens around `256`
- independent samples must actually be independent, not deduped
- primary analysis: paired deltas, response-token pools, sign-flip nulls
- specificity analysis: generic moral-prime control and centroid-residual
  theory-specific directions
