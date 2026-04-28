---
project: MOREBENCH
subproject: theory_persona_vectors
phase: 02
artifact: all_theories_pole_pilot_plan
date: 2026-04-26
status: draft_for_review
---

# All-Theories Pole Pilot

## Premise (continuation of phase_01 PHASE.md)

Phase_01's deontology pole pilot failed at L32 prompt-end with a tight gap
between split-half and random-label nulls (gap 0.06–0.10 across all four
constructions). The qualitative reading of the 180 generations suggested that
**Qwen3-30B-A3B's default-mode response is approximately deontology-aligned
on this dilemma set**: N_neutral and P_deont give the same recommendation
on most spot-checked dilemmas, while N_anti or N_alt_util are what flip the
behavior.

The phase_01 exit jumped to "theory-vs-theory pole construction" as the
recommended pivot. That conclusion was premature. The data only tells us that
deont-vs-default is small; it does not tell us whether util-vs-default,
virtue-vs-default, or contract-vs-default are small. If the persona-vectors
recipe is sound but deontology is the bad starting case, then non-default
theories should extract cleanly against the same neutral pole.

This phase tests that.

## Hypothesis

For each of the four moral theories (deontology, utilitarian, virtue_ethics,
contractualism), repeat the phase_01 smoke at L32 prompt-end:

- H_per_theory: split-half cosine clears the random-label null_p95 by
  ≥ 0.20 for at least one neutral-pole construction
  ({theory}_neutral_short or {theory}_neutral_length_matched).

The interesting cross-theory pattern to look for: **does the gap vary across
theories**? If deontology fails but utilitarian/virtue/contractualism pass,
that confirms the default-mode-aligned interpretation. If all four fail,
the persona-vectors recipe may simply not work on this model. If all four
pass, the phase_01 result was a measurement artifact and we should re-check
phase_01.

## Conditions (14 total)

For each theory T ∈ {deontology, utilitarian, virtue_ethics,
contractualism}: P_T_01 (primary), P_T_02 (variant), N_anti_T_01.

Shared: N_neutral_01 (short), N_neutral_02 (length-matched, with
"consequences" replaced by "factors" to avoid utilitarian leakage).

Cross-theory alt diagnostics are computed at analysis time using each
theory's positive as the negative pole for the other three theories — no
extra conditions needed.

Artifact: `specs/all_theories_pole_pilot_prompt_conditions.json`.

## Data

Reused: `outputs/all_theories_pole_pilot_synth_dilemmas.jsonl` is a copy of
`phase_01/outputs/deontology_pole_pilot_synth_dilemmas.jsonl` (same 30
dilemmas, same MoReBench-shaped conflict surfaces).

## Generation and capture

- model: `Qwen/Qwen3-30B-A3B`
- 30 dilemmas × 14 conditions × 1 sample = 420 generations
  (SAMPLES_PER_CONDITION explicitly 1 because the engine deduplicates
  identical prompts; this is correction #5 from phase_01 PHASE.md)
- max_tokens 96, temperature 0.7, top_p 0.95
- captures at L0/L4/L16/L24/L32/L40, prompt_end + generated_sequence

## Smoke gate

For each (theory, neutral construction) at L32 prompt_end:

- pass: gap = split_half − null_p95 ≥ 0.20
- marginal: 0.10 ≤ gap < 0.20
- fail: gap < 0.10

Anti and alt-theory constructions are reported but not required for pass.

## Decision rules

- All four theories pass at the same neutral construction: persona-vectors
  recipe works; phase_01's deont-only failure is the anomaly. Investigate
  why deont specifically failed.
- Some theories pass, some fail: the pass/fail pattern is the result.
  Theories that fail are likely close to the model's default mode.
- All four fail: the persona-vectors primed-vs-default recipe does not
  work on this model. Pivot to contrastive theory-vs-theory directions.

## Out of scope

- Transfer to MoReBench: still held out until a synthetic construction
  passes.
- Prompt-variant scaling: P_02 is included, but we are not pre-committing
  to P_02-derived directions as primary.
- Locus sweep: the L32 prompt-end primary is locked in for direct
  comparison with phase_01.

## Artifacts

- spec: `specs/all_theories_pole_pilot_workflow.py`
- conditions: `specs/all_theories_pole_pilot_prompt_conditions.json`
- dilemmas: `outputs/all_theories_pole_pilot_synth_dilemmas.jsonl`
- analysis: `scripts/analyze_all_theories_pole_pilot.py`
- run id (this run): `wr_7fb9542ae82d_0c284562`
