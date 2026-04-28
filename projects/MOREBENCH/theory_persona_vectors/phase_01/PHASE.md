---
project: MOREBENCH
subproject: theory_persona_vectors
phase: 01
artifact: phase_exit
date: 2026-04-26
status: closed_pending_review
---

# Phase 01 Exit — Deontology Persona-Vector Pole Pilot

## Premise

We pivoted back to moral-theory work on MoReBench using a persona-vectors-style
strategy (positive theory-prime vs default-mode pole, difference-in-means
direction extraction). The pilot's job was to settle the pole-construction
question before scaling: theory-vs-anti, theory-vs-theory, or theory-vs-neutral?
A length-matched neutral was added on the suspicion that the natural neutral
("Give your recommendation. Do not explain.") is much shorter than the primed
prompts and was confounding direction extraction with instruction length.

We expected at least one pole construction to extract a deontology-flavored
direction with split-half cosine ≥ 0.70, random-label null cosine ≤ 0.30, and
clean separation across the four candidate constructions. We were prepared for
deont_anti to be contaminated by negation/instruction-suppression and for
deont_util to be theory-pair-specific. The plan tagged
`deont_neutral_length_matched` as the primary construction to scale.

## What we ran

- model: `Qwen/Qwen3-30B-A3B` on Modal H200
- workflow: `morebench_theory_persona_vectors_phase01_deontology_pole_pilot`
- run id: `wr_6354356dc6a6_9918b007`
- generation artifact: `generation_run_1_cdd0020853b6` (180 rows, all `stop`)
- capture artifact: `capture_1_88c42755786d`
- summarize artifact: `transform_1_cc59ebe5`
- report artifact: `report_54943e5ee6fc_5b03205e`
- workflow hash: `6354356dc6a64185c39d14c388d5a141892b3e5eae70bceb4aa5941d9c5b9f3e`

Data:

- 30 synthetic compact dilemmas
  (`outputs/deontology_pole_pilot_synth_dilemmas.jsonl`)
- 6 prompt conditions
  (`specs/deontology_pole_pilot_prompt_conditions.json`):
  P_deont_01, P_deont_02, N_neutral_01, N_neutral_02, N_anti_01,
  N_alt_util_01

Sites:

- `prompt_end_residual` and `generated_sequence_residual` at layers
  `[0, 4, 16, 24, 32, 40]`; primary L32 prompt-end (mean-pooled for
  generated section, identity for prompt-end)

Generation settings: temperature 0.7, top_p 0.95, max_tokens 96,
SAMPLES_PER_CONDITION declared as 3 — see Corrections.

## Primary result

The headline is unflattering. **Direction extraction at the pre-registered
primary locus (L32 prompt_end) is essentially indistinguishable from random
label shuffles.**

Direction stability at L32 prompt_end:

| construction | split_half_cos | null_p95 | null_max | gap (split−p95) |
|---|---:|---:|---:|---:|
| deont_anti | 0.951 | 0.888 | 0.935 | 0.063 |
| deont_neutral_length_matched | 0.946 | 0.868 | 0.928 | 0.078 |
| deont_neutral_short | 0.941 | 0.850 | 0.919 | 0.091 |
| deont_util | 0.890 | 0.786 | 0.857 | 0.104 |

Pre-registered smoke threshold was split-half cos ≥ 0.70 and random-label cos
≤ 0.30. The split-half passes everywhere but the null is also high
everywhere — the gap between real and random labels is 6–10 percentage points
at L32 prompt_end. By the spirit of the smoke (**signal must clear noise**)
none of the four pole constructions is clean at this layer.

Pole-construction cosine matrix at L32 prompt_end:

| | deont_anti | deont_neutral_length_matched | deont_neutral_short | deont_util |
|---|---:|---:|---:|---:|
| deont_anti | 1.000 | 0.535 | 0.561 | 0.269 |
| deont_neutral_length_matched | 0.535 | 1.000 | 0.803 | 0.374 |
| deont_neutral_short | 0.561 | 0.803 | 1.000 | 0.291 |
| deont_util | 0.269 | 0.374 | 0.291 | 1.000 |

Cross-positive (P_deont_01 vs P_deont_02 with neutral_length_matched negative)
at L32 prompt_end: cos = `0.872`. Cross-neutral (deont_neutral_short vs
deont_neutral_length_matched): cos = `0.803`.

The earlier-layer picture is no better at the noise level. Generated-sequence
mean-pool at L4–L32 has a wider real/null gap than prompt_end (e.g. L4
generated split-half 0.69–0.78 vs null_p95 0.36–0.55, gap 0.16–0.32) but the
absolute split-half is lower and the gap shrinks again at L40.

Behavioral divergence at the recommendation level (jaccard < 0.6 on normalized
recommendations):

| pair | diverged_share | avg_jaccard |
|---|---:|---:|
| P_deont_vs_N_neutral_short | 0.833 | 0.369 |
| P_deont_vs_N_neutral_length_matched | 0.867 | 0.338 |
| P_deont_vs_N_anti | 0.700 | 0.370 |
| P_deont_vs_N_alt_util | 0.667 | 0.494 |
| P_deont_01_vs_P_deont_02 | 0.700 | 0.452 |

Per the pre-registered interpretation table, divergence ≥ 50% characterizes the
direction as a behavioral-disposition direction. The label here is moot
because no construction extracts cleanly.

## Qualitative inspection

Reading 30 dilemmas × 6 conditions reveals a confound the pre-reg missed.

**The "neutral" pole is not actually neutral.** Across most dilemmas, the
default-mode response from Qwen3-30B-A3B already lands roughly on the
deontological recommendation. Examples:

- *Software safety flaw (delay vs launch quietly)*: P_deont_01,
  P_deont_02, N_neutral_01, N_neutral_02 all say "delay the launch". Only
  N_anti_01 says "launch quietly".
- *Grade adjustment (refuse vs adjust)*: P_deont_01, P_deont_02, N_neutral_01,
  N_neutral_02 all say "don't adjust". N_anti_01 and N_alt_util_01 say
  "adjust the grade".

If the model's neutral default is already deontology-shaped, then `mean(P) -
mean(N_neutral)` is a small contrast in a direction the model already
occupies. That is exactly the regime where random label-shuffle nulls become
high — the activation variance dominates the mean-shift.

**The positive variants do not behave identically.** On the hospital-supplies
dilemma, P_deont_01 ("guided by duties, rights, promises, and moral
constraints") says "Prioritize the main ward to fulfill the *duty of
maximizing overall patient benefit*" — utilitarian content wearing a duty
label. P_deont_02 ("commitments, boundaries, and standing constraints") says
"Send the supplies to the isolation unit" — a deontology-consistent
commitment-honoring choice. The two positives are correlated at L32
prompt_end (cos 0.87) but they are *not* recommending the same action 70% of
the time.

**N_anti_01 is a real anti-pole, not a negation artifact.** It changes the
recommendation in the cases where deontology and consequentialism actually
conflict. That makes it the most useful diagnostic of the three negatives,
not the least. The pre-reg's worry that "anti" would just encode negation or
instruction-suppression is not borne out at the recommendation level.

**N_alt_util_01 and N_anti_01 frequently converge** on the same flipped
action (grade adjustment, both say adjust). Their direction-space cosines at
L32 prompt_end are still low (0.27) which suggests they encode different
internal trajectories for the same action — interesting but downstream of
fixing the primary problem.

## Corrections

These are the beliefs going in that the pilot revised. Load-bearing.

1. **`deont_neutral_length_matched` is not the right primary
   construction.** The pre-reg picked it because the persona-vectors paper
   uses primed-vs-default and we wanted to control for instruction length. But
   the actual confound is that *default mode is roughly deontology-aligned for
   this model*. Length-matching the neutral does not fix this.

2. **The "primed-vs-default" persona-vector recipe assumes the default is
   theory-neutral.** That assumption fails for moral theories on
   Qwen3-30B-A3B. If the pretrained default already occupies a deontological
   region of activation space, the difference-in-means against any default
   pole will measure a tiny shift floating in a high-variance direction —
   exactly what we see in the null cosines.

3. **High direction-space stability is not the same as low null-cosine
   stability.** Split-half cos 0.94 looked like a pass at L32. It is not. When
   the null is 0.88, the split-half is mostly recovering the dominant
   activation direction shared by all examples, not a theory-specific
   direction. Future smoke gates should report (split-half − null_p95) not
   split-half alone.

4. **Behavioral divergence is informative, not just characterization.** The
   pre-reg framed it as "characterize, don't gate" because frameworks can
   converge on actions while diverging in internal state. That holds in
   principle. In *this* run, divergence is high (83–87% for the primary
   contrasts) and direction extraction still failed — so high divergence does
   not save you from the neutral-pole-is-not-neutral problem.

5. **`SAMPLES_PER_CONDITION = 3` did not produce 3 samples.** The
   `GenerationRunSpec` collapsed identical prompts to a single generation
   per (dilemma, condition), yielding 180 rows instead of 540. The dataset
   builder enumerates samples but the engine deduplicates by prompt content
   before submitting to vLLM. The actual N per condition is 30, not 90. The
   smoke ran on a smaller sample than designed; this affects split-half
   stability estimates but it does not change the headline failure (null is
   high regardless of N).

6. **Theory-vs-theory may be more honest than theory-vs-neutral here.**
   `deont_util` shows the largest separation from the other constructions
   (cos 0.27–0.37 at L32 prompt_end with the neutral and anti versions). If
   "deontology" is mostly the model's default, then *contrastive* moral
   directions (deontology−utilitarian, deontology−virtue, etc.) may be the
   only constructions that meaningfully separate from baseline activation.
   That's the opposite of what the pre-reg expected.

## Running hypothesis

For Qwen3-30B-A3B on terse moral recommendations:

- The default response distribution is approximately deontology-aligned at the
  recommendation level.
- The persona-vector "primed-vs-default" recipe will not extract a stable
  deontology direction from this model because the contrast is too small
  relative to per-example activation variance.
- Theory-vs-theory contrastive constructions (deont vs util, deont vs care,
  etc.) are the more promising path — `deont_util` already shows clear
  separation from the neutral-derived directions even though its split-half
  is the weakest of the four. The reason it looks weakest is that it is
  measuring something *real and different* from the noise floor that the
  other three are riding.
- The L32 prompt-end pre-registered locus is not where this signal lives, if
  it lives anywhere. Generated-sequence mean-pool at L4 has the cleanest
  signal/null gap in the current data.

## Claim boundary

Safe to claim right now:

- We ran the pilot as planned (with the SAMPLES_PER_CONDITION caveat above).
- At L32 prompt-end, none of the four pole constructions extracts a
  deontology direction whose split-half cosine clears the random-label null
  by more than ~0.10.
- P_deont_01 and P_deont_02 produce correlated direction estimates (cos
  0.84–0.91 across layers) — prompt-variant stability is *not* the failing
  axis.
- The model's default-mode (no theory prime) recommendation often coincides
  with the deontology-primed recommendation on this dilemma set. Two
  hand-spot-check dilemmas (software safety, grade adjustment) show
  N_neutral and P_deont giving the same recommendation while N_anti or
  N_alt_util flips it.

Not supported yet, avoid claiming:

- That moral-theory persona vectors don't exist in this model. The pilot
  failed in a specific way (default ≠ neutral) that the persona-vectors
  recipe is silent on; this is not the same as a clean negative.
- That `deont_util` is a real deontology direction. Its low cosine with the
  other constructions could equally mean "it's measuring utilitarian-prime
  activation, not deontology". Pair-specificity controls are still pending.
- That generated-sequence L4 is the right locus. The signal/null gap there is
  wider, but the pre-reg favored prompt-end at L32; we should not relocate
  the primary measurement based on a single underpowered run.
- Anything about transfer to MoReBench. The transfer / contested-capture
  diagnostic is scaffolded in the analysis script but not wired to a
  specific MoReBench artifact yet.

Preferred phrasing for the current state:

- "The persona-vectors primed-vs-default recipe does not extract a clean
  deontology direction on this model because the model's default already sits
  in a roughly-deontological region of activation space. Theory-vs-theory
  contrastive constructions look more promising and should be tested before
  scaling."

## Artifacts

Plan and pre-reg:

- `projects/MOREBENCH/theory_persona_vectors/phase_01/docs/01-deontology-pole-pilot-plan.md`

Specs:

- `projects/MOREBENCH/theory_persona_vectors/phase_01/specs/deontology_pole_pilot_prompt_conditions.json`
- `projects/MOREBENCH/theory_persona_vectors/phase_01/specs/deontology_pole_pilot_workflow.py`
- `projects/MOREBENCH/theory_persona_vectors/phase_01/specs/deontology_pole_pilot_workflow.json`

Data:

- `projects/MOREBENCH/theory_persona_vectors/phase_01/outputs/deontology_pole_pilot_synth_dilemmas.jsonl`

Scripts:

- `projects/MOREBENCH/theory_persona_vectors/phase_01/scripts/analyze_deontology_pole_pilot.py`

Reports:

- `projects/MOREBENCH/theory_persona_vectors/phase_01/reports/deontology_pole_pilot_analysis/summary.json`
- `projects/MOREBENCH/theory_persona_vectors/phase_01/reports/deontology_pole_pilot_analysis/report.md`
- `projects/MOREBENCH/theory_persona_vectors/phase_01/reports/deontology_pole_pilot_report/report_54943e5ee6fc_5b03205e/report.md`

Catalog references:

- run id: `wr_6354356dc6a6_9918b007`
- generation: `generation_run_1_cdd0020853b6`
- capture: `capture_1_88c42755786d`

## Open threads

> **Update 2026-04-26 (post-exit):** the original "pivot to theory-vs-theory"
> recommendation in this phase was over-quick. The data here only shows that
> deont-vs-default is small; it doesn't show that util-, virtue-, or
> contract-vs-default are also small. Phase 02 (all_theories_pole_pilot) reruns
> the same smoke for all four theories on the same 30 dilemmas before any
> pivot. If deont fails and the others pass, the recipe is sound and only
> deontology is default-aligned. If all four fail, then theory-vs-theory is
> the right next step. See `phase_02/docs/01-all-theories-pole-pilot-plan.md`.

These are the questions next phase should pick up.

1. **Is the model's default mode really deontology-aligned?** Hand-coded
   recommendation labels (deontology / utilitarian / mixed / dominated) on
   the 30 dilemmas would let us measure the rate at which N_neutral matches
   each theory. If the rate is ~70%+ for deontology, the persona-vector
   recipe needs a different default. Cheap to do; high information content.

2. **Theory-vs-theory pole construction.** Re-run the pilot with the four
   negatives swapped for theory contrasts: deont vs util, deont vs care,
   deont vs virtue, deont vs contractualist. This tests directly whether
   contrastive moral directions clear the random-label null where
   primed-vs-default does not.

3. **Sample-collapse fix.** The `GenerationRunSpec` deduplicates identical
   prompts. Either inject a sample-index sentinel into the prompt (e.g.
   `<sample-id>k</sample-id>` in a system field that doesn't change the
   model's response distribution) or run multiple replicas of the
   generation step with different seeds. Required before claiming any
   split-half result is well-powered.

4. **Locus sweep.** The pre-registered L32 prompt-end was based on the prior
   MoReBench theory track. Generated-sequence mean-pool at L4 looks cleaner
   in this data. A small layer × site sweep on the next run should be
   pre-registered, not chosen post-hoc.

5. **Transfer to MoReBench held-out captures.** The analysis script has a
   slot for it. We need to identify a MoReBench prompt-end L32 capture
   artifact whose theory-prime labels we trust, then check whether the
   synthetic-derived deont direction has higher cosine with MoReBench
   theory-primed activations than with default-primed. This is the
   ultimate test, deferred until we have a construction that passes the
   synthetic smoke.

6. **Random-label null methodology.** The pilot computes the null by
   permuting positive/negative labels within the union of (P_deont_01,
   P_deont_02, N_*). Since these all share dilemma content, a permutation
   may still respect dilemma structure. A stronger null would also permute
   *across dilemmas* to break the per-dilemma activation correlation.
   Worth doing in the next run.

7. **Behavioral divergence by jaccard is crude.** It catches paraphrase
   collisions but misses cases where two responses recommend the same
   action with different wording. A coarse action-label per response (yes
   / no / depends) would give a cleaner divergence measure for the next
   pilot.
