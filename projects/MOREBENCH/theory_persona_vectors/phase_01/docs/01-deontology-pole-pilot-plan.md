---
project: MOREBENCH
subproject: theory_persona_vectors
phase: 01
artifact: deontology_pole_pilot_plan
date: 2026-04-26
status: draft_for_review
---

# Deontology Persona-Vector Pole Pilot

## Purpose

This pilot tests the pole-construction choice before scaling moral-theory
persona vectors. The immediate question is not whether all moral theories can be
recovered. It is whether a deontology-primed synthetic direction can be made
stable enough, and clean enough, to justify a full persona-vector run.

MoReBench is treated as held-out real data. The synthetic extraction set is a
new compact dilemma bank that is MoReBench-shaped but not copied from
MoReBench.

## Central Hypothesis

Prompting the model to answer from a deontological stance creates a residual
direction that is more than theory-name recognition or explanation style. If the
direction is real, it should be stable across synthetic dilemma splits and
should transfer back to existing MoReBench theory-primed captures.

## Why This Phase Exists

The earlier MoReBench theory track found a real prompt-side result, but it was
entangled with theory cues and prompt construction. The persona-vector strategy
changes the substrate:

- extract directions from a separate synthetic acting-as-the-theory task
- suppress explanation text by asking for recommendations only
- return to MoReBench as held-out evaluation

The load-bearing design risk is the negative pole. This pilot compares three
ways to define the contrast before committing to scale.

## Data

Synthetic dilemma bank:

- `30` compact moral-choice dilemmas
- no explicit moral-theory names
- no requests for reasoning
- MoReBench-like conflict surfaces: privacy, safety, family, institutional
  constraint, disclosure, allocation, professional role, fairness, autonomy
- intended answer format: terse recommendation only

Artifact:

- `projects/MOREBENCH/theory_persona_vectors/phase_01/outputs/deontology_pole_pilot_synth_dilemmas.jsonl`

## Prompt Conditions

The pilot uses one positive target and four negative/comparison poles. A second
positive phrasing is included to measure prompt-variant stability.

Primary positive:

- `P_deont_01`: duty/rights/promises/constraints framing

Positive variant:

- `P_deont_02`: commitments/boundaries/standing constraints framing

Negative/comparison poles:

- `N_neutral_01`: no framework mention; recommendation only
- `N_neutral_02`: no framework mention; length-matched neutral instruction
- `N_anti_01`: explicit non-deontological suppression; diagnostic only
- `N_alt_util_01`: utilitarian/consequentialist alternative; diagnostic theory-vs-theory pole

Artifact:

- `projects/MOREBENCH/theory_persona_vectors/phase_01/specs/deontology_pole_pilot_prompt_conditions.json`

## Extraction Directions

For the first smoke, compute:

- `deont_neutral`: mean(`P_deont`) - mean(`N_neutral_01`)
- `deont_neutral_length_matched`: mean(`P_deont`) - mean(`N_neutral_02`)
- `deont_anti`: mean(`P_deont`) - mean(`N_anti_01`)
- `deont_util`: mean(`P_deont`) - mean(`N_alt_util_01`)

The default primary construction is `deont_neutral_length_matched`, because it
keeps the persona-vector paper's primed-vs-default design while reducing the
long-instruction-vs-short-instruction confound. The short-neutral
`deont_neutral` construction remains a diagnostic. The anti and utilitarian
constructions are diagnostics, not winner-picked replacements.

## Planned Generation Settings

Recommended first run:

- model: `/models/Qwen/Qwen3-30B-A3B`
- samples: `3` per dilemma x condition
- temperature: `0.7`
- max new tokens: small cap, enough for a terse recommendation
- instruction: no explanation

Expected generation count with current conditions:

- `30` dilemmas x `6` prompt conditions x `3` samples = `540` generations

If budget or runtime pressure matters, drop `P_deont_02` for a `360` generation
four-condition pilot. The preferred pilot keeps `P_deont_02` and
`N_neutral_02` because prompt-variant stability and neutral-length sensitivity
are among the cheapest useful checks.

## Primary Measurement Locus

Primary:

- prompt-final / assistant-colon residual at layer `32`

Diagnostics:

- response-mean residual at layer `32`
- later full scale may sweep layers `0, 4, 8, 16, 28, 36, 40, 44`

Layer `32` is chosen for the pilot because it was the strongest prior
MoReBench prompt-side theory readout layer. It is not a claim that `32` is the
only relevant write/read layer.

## Stage 3 Validation Checks

These checks run before promoting to a larger extraction design.

### Behavioral Divergence

Measure whether theory priming changes terse recommendations. This is a
characterization variable, not a gate. Multiple frameworks may converge on the
same final action while still inducing different internal framework states,
especially at prompt-final.

Report:

- proportion of dilemmas where all conditions converge on the same action
- proportion where `P_deont` differs from `N_neutral`
- proportion where `P_deont` differs from `N_neutral_02`
- proportion where `P_deont` differs from `N_alt_util`

Pre-committed interpretation:

- direction extracts cleanly and divergence `>= 50%`: characterize as a
  behavioral-disposition direction
- direction extracts cleanly and divergence `< 30%`: characterize as a
  framework-state direction, not an action-disposition direction
- direction failure with low divergence: ambiguous; run sensitivity check on a
  behaviorally contested substrate before declaring the approach null
- direction failure with high divergence: stronger negative; the prime changed
  action but no stable readout recovered it

### Synthetic-vs-MoReBench Activation Overlap

Capture a small matched batch of synthetic prompt-final residuals and existing
MoReBench prompt-final residuals at layer `32`.

Report:

- PCA plot or table for combined activations
- silhouette score for `synth` vs `MoReBench` source label

This is a prerequisite for interpreting transfer failure. If synthetic and
MoReBench prompts live in separable activation regions, a transfer null is not
clean evidence against the theory-persona idea.

## Smoke Criteria

Direction stability:

- split-half cosine within each construction, target `>= 0.70`
- cosine between `P_deont_01`-derived and `P_deont_02`-derived directions,
  target `>= 0.50`
- cosine between `deont_neutral` and `deont_neutral_length_matched`, target
  `>= 0.70` if short-neutral length asymmetry is not dominant
- random-label direction cosine with real direction, target `<= 0.30`

Pole comparison:

- `deont_neutral`, `deont_neutral_length_matched`, `deont_anti`, and
  `deont_util` cosine matrix
- if all converge, scale with `deont_neutral_length_matched` as primary
- if only `deont_util` transfers, pivot to theory-vs-theory contrastive vectors
- if only `deont_anti` transfers, treat as suspicious until negation controls pass

Transfer check:

- apply each synthetic direction to existing MoReBench theory-primed prompt-final
  residuals
- primary transfer target: deontology-primed vs not-deontology-primed
- require source-family-aware reporting where feasible

## Confound Checks

Required before scaling:

- negation direction check for `deont_anti`
- generic framework-prime direction check for `deont_neutral`
- length-only direction baseline check for `deont_neutral`
- pair-specificity check for `deont_util`
- char-TFIDF baseline on terse generated answers
- prompt-condition holdout baseline
- response length audit

## Parallel Existing-Capture Diagnostic

The synthetic corpus remains the primary extraction substrate, but there is a
nearly free diagnostic from existing MoReBench contested-case captures. Compute a
deontology-style direction from the existing contested theory captures at L32
prompt-final:

- contested direction: deontology-prime activation mean minus generic/default
  prime activation mean
- compare against `deont_neutral_length_matched` from the synthetic pilot

Pre-committed interpretation:

- cosine `>= 0.70`: strong substrate-convergence evidence
- cosine between `0.40` and `0.70`: partial overlap; report each substrate
  separately
- cosine `< 0.40`: substrate-specific directions; investigate before scaling

This diagnostic does not replace the synthetic extraction design because it uses
MoReBench-like theory captures and weakens the held-out-return framing. Its job
is to distinguish a synthetic-substrate failure from a broader failure of
persona-vector-style moral-theory extraction.

## Decision Rules

Scale the persona-vector plan if:

- `deont_neutral_length_matched` is stable, not dominated by generic framework
  priming or length, and transfers to MoReBench better than controls.

Pivot to contrastive theory vectors if:

- `deont_util` transfers but `deont_neutral_length_matched` does not.

Do not scale on anti-theory alone if:

- `deont_anti` is the only passing construction. That result is likely to encode
  negation, suppression, or instruction-following artifacts unless controls say
  otherwise.

Loop back to synthetic design if:

- no construction is stable or no construction transfers above control.

## Review Questions

1. Are the 30 dilemmas sufficiently contested without importing deontology words?
2. Are the dilemmas MoReBench-shaped enough for transfer to be meaningful?
3. Is `N_neutral_01` too short relative to the positive prompts, and does
   `N_neutral_02` adequately control that asymmetry?
4. Does `N_anti_01` remain useful as a diagnostic, or should it be excluded
   until later?
5. Is `N_alt_util_01` the right first theory-vs-theory comparison for
   deontology?
6. Should any dilemma be removed because one answer is too obviously dominant?
7. Should any domain be added or rebalanced before generation?
