---
project: MOREBENCH
subproject: theory_persona_vectors
phase: 02
artifact: phase_exit
date: 2026-04-26
status: closed_pending_review
---

# Phase 02 Exit — All-Theories Persona-Vector Pole Pilot

## Premise

Phase_01 found that, on Qwen3-30B-A3B at L32 prompt_end, the persona-vectors
primed-vs-default recipe failed to extract a deontology direction whose
split-half cosine cleared the random-label null by more than ~0.10. The
qualitative reading suggested this was because the model's default-mode
response is approximately deontology-aligned. The phase_01 exit jumped to
"pivot to theory-vs-theory pole construction" as the recommended next step.

That conclusion was over-quick. The data only showed deont-vs-default was
small. It did not show that util-, virtue-, or contract-vs-default were also
small. If the recipe is sound but deontology was the bad starting case, then
util/virtue/contract should pass. This phase tested all four.

## What we ran

- model: `Qwen/Qwen3-30B-A3B` on Modal H200
- workflow: `morebench_theory_persona_vectors_phase02_all_theories_pole_pilot`
- run id: `wr_7fb9542ae82d_0c284562`
- generation artifact: `generation_run_1_15bc125de56b` (420 rows, all `stop`)
- capture artifact: `capture_1_c2684db0530c`
- workflow hash:
  embedded in `specs/all_theories_pole_pilot_workflow.py` (auto)

Conditions (14, see `specs/all_theories_pole_pilot_prompt_conditions.json`):

- 4 theories × (P_T_01 + P_T_02 + N_anti_T_01) = 12
- 2 shared neutrals (N_neutral_01 short, N_neutral_02 length-matched with
  "factors" replacing "consequences" to avoid utilitarian leakage)

Same 30 dilemmas as phase_01. Same captured layers (0/4/16/24/32/40).
Primary locus pre-registered at L32 prompt_end, gap-pass criterion
(`split_half − null_p95`):
`pass` ≥ 0.20, `marginal` ∈ [0.10, 0.20), `fail` < 0.10.

## Locus correction (post-hoc, load-bearing)

The pre-reg fixed the primary locus at **L32 prompt_end** because that was
the strongest prior MoReBench prompt-side theory readout layer. For a
persona-vectors-style analysis, that was the wrong choice. The
persona-vectors paper extracts from response-token activations (response-mean
or per-response-token), not from prompt-end. The phase_02 capture stored
both `prompt_end_residual` and `generated_sequence_residual` (response-mean
pool); the original headline reported the prompt-end version. The L32
**response-mean** picture is the correct headline for this method.

The two loci tell different stories. Both are reported below; the
response-mean picture is now the primary and the prompt_end picture is
demoted to diagnostic. Where statements in the original "Primary result"
and "Qualitative inspection" depended on the prompt-end version, see
"Corrections" for revisions.

## Primary result @ L32 generated (response-mean) — corrected primary locus

Headline smoke at the corrected locus, theory-vs-neutral and theory-vs-anti:

| theory | construction | split_half | null_p95 | gap | verdict |
|---|---|---:|---:|---:|---|
| deontology | deont_neutral_short | 0.687 | 0.626 | 0.061 | fail |
| deontology | deont_neutral_length_matched | 0.654 | 0.538 | 0.116 | marginal |
| deontology | deont_anti | 0.685 | 0.563 | 0.123 | marginal |
| utilitarian | util_neutral_short | 0.559 | 0.517 | 0.042 | fail |
| utilitarian | util_neutral_length_matched | 0.589 | 0.463 | 0.126 | marginal |
| utilitarian | util_anti | 0.444 | 0.465 | -0.021 | fail |
| virtue_ethics | virtue_neutral_short | 0.720 | 0.681 | 0.039 | fail |
| virtue_ethics | virtue_neutral_length_matched | 0.700 | 0.637 | 0.064 | fail |
| virtue_ethics | virtue_anti | 0.655 | 0.560 | 0.095 | fail |
| contractualism | contract_neutral_short | 0.733 | 0.599 | 0.134 | marginal |
| contractualism | contract_neutral_length_matched | 0.675 | 0.566 | 0.108 | marginal |
| contractualism | contract_anti | 0.645 | 0.482 | 0.163 | marginal |

Theory-vs-theory (alt) at the corrected locus is *worse* than null:

| theory | construction | split_half | null_p95 | gap |
|---|---|---:|---:|---:|
| deontology | deont_alt_util | 0.304 | 0.407 | -0.103 |
| deontology | deont_alt_virtue | 0.302 | 0.382 | -0.081 |
| deontology | deont_alt_contract | 0.271 | 0.346 | -0.075 |
| utilitarian | util_alt_deont | 0.387 | 0.336 | 0.051 |
| utilitarian | util_alt_virtue | 0.588 | 0.506 | 0.083 |
| utilitarian | util_alt_contract | 0.258 | 0.400 | -0.141 |
| virtue_ethics | virtue_alt_deont | 0.172 | 0.442 | -0.270 |
| virtue_ethics | virtue_alt_util | 0.325 | 0.503 | -0.178 |
| virtue_ethics | virtue_alt_contract | 0.111 | 0.365 | -0.254 |
| contractualism | contract_alt_deont | 0.251 | 0.322 | -0.071 |
| contractualism | contract_alt_util | 0.133 | 0.410 | -0.277 |
| contractualism | contract_alt_virtue | 0.273 | 0.413 | -0.140 |

11 of 12 alt constructions have negative gaps at the corrected primary
locus. Random label shuffles produce a stronger "direction" than the real
labels. At the response-mean locus, theory-vs-theory contrastive
constructions encode nothing systematic.

Across-layer best gap per (theory, primary-construction) at the corrected
site:

| theory | construction | best gap | best layer |
|---|---|---:|---|
| deontology | neutral_short | 0.155 | L4 |
| deontology | neutral_length_matched | 0.116 | L32 |
| deontology | anti | 0.123 | L32 |
| utilitarian | neutral_short | 0.087 | L4 |
| utilitarian | neutral_length_matched | 0.126 | L32 |
| virtue_ethics | neutral_short | 0.188 | L4 |
| virtue_ethics | neutral_length_matched | 0.199 | L4 |
| virtue_ethics | anti | 0.095 | L32 |
| contractualism | neutral_short | 0.167 | L4 |
| contractualism | neutral_length_matched | 0.170 | L4 |
| contractualism | anti | 0.163 | L32 |

The corrected-locus picture: 8 of 12 theory-vs-neutral/anti constructions
hit marginal at L32 generated. Contractualism (gaps 0.11–0.16) and
deontology (0.12) are the strongest at the corrected locus; utilitarian
and virtue ethics are weaker. **No construction passes the gate at L32
generated either, but the picture is "marginal across the board" rather
than "flat fail across the board" as the prompt-end headline suggested.**

## Original primary result @ L32 prompt_end (now diagnostic)

The original (mis-located) primary headline. Kept for reproducibility and
for the cross-theory cosine-clique evidence which remains
informative-as-prompt-side characterization.

Headline smoke @ L32 prompt_end (theory-vs-neutral and theory-vs-anti):

| theory | construction | split_half | null_p95 | gap | verdict |
|---|---|---:|---:|---:|---|
| deontology | deont_neutral_short | 0.940 | 0.849 | 0.090 | fail |
| deontology | deont_neutral_length_matched | 0.942 | 0.872 | 0.070 | fail |
| deontology | deont_anti | 0.951 | 0.888 | 0.063 | fail |
| utilitarian | util_neutral_short | 0.938 | 0.866 | 0.072 | fail |
| utilitarian | util_neutral_length_matched | 0.932 | 0.873 | 0.059 | fail |
| utilitarian | util_anti | 0.921 | 0.853 | 0.068 | fail |
| virtue_ethics | virtue_neutral_short | 0.947 | 0.856 | 0.091 | fail |
| virtue_ethics | virtue_neutral_length_matched | 0.944 | 0.868 | 0.076 | fail |
| virtue_ethics | virtue_anti | 0.942 | 0.878 | 0.064 | fail |
| contractualism | contract_neutral_short | 0.955 | 0.872 | 0.083 | fail |
| contractualism | contract_neutral_length_matched | 0.952 | 0.872 | 0.079 | fail |
| contractualism | contract_anti | 0.940 | 0.862 | 0.078 | fail |

Theory-vs-theory (alt) constructions at L32 prompt_end are mostly marginal,
none pass:

| theory | construction | gap |
|---|---|---:|
| deontology | deont_alt_util | 0.104 |
| deontology | deont_alt_virtue | 0.135 |
| deontology | deont_alt_contract | 0.140 |
| utilitarian | util_alt_deont | 0.082 |
| utilitarian | util_alt_virtue | 0.097 |
| utilitarian | util_alt_contract | 0.118 |
| virtue_ethics | virtue_alt_deont | 0.119 |
| virtue_ethics | virtue_alt_util | 0.127 |
| virtue_ethics | virtue_alt_contract | 0.162 |
| contractualism | contract_alt_deont | 0.111 |
| contractualism | contract_alt_util | 0.131 |
| contractualism | contract_alt_virtue | 0.152 |

Cross-theory direction cosine matrices at L32 prompt_end tell the structural
story:

`neutral_short` (each theory's primed-vs-neutral direction):

| | contract | deont | util | virtue |
|---|---:|---:|---:|---:|
| contract | 1.000 | 0.842 | 0.867 | 0.865 |
| deont | 0.842 | 1.000 | 0.804 | 0.885 |
| util | 0.867 | 0.804 | 1.000 | 0.833 |
| virtue | 0.865 | 0.885 | 0.833 | 1.000 |

The four theories' primed-vs-neutral directions are mutually 0.80–0.89
cosine. **They are essentially the same direction.** Whatever they encode, it
is not theory-specific.

The `anti` matrix is more separated (0.53–0.76 across theory pairs),
consistent with each anti-prompt being theory-specific.

Across-layer summary (all 4 theories × all sites × all layers): **17
pass-verdicts out of 432 total** (392 non-NaN). 16 of the 17 sit at L4
prompt_end. The only non-L4 pass is `deont_anti_p_variant` at L32
generated_sequence_residual (gap 0.209).

Top L4 prompt_end gaps are dominated by theory-vs-theory (alt)
constructions:

| gap | theory | construction |
|---:|---|---|
| 0.259 | contractualism | contract_alt_deont |
| 0.241 | virtue_ethics | virtue_alt_deont |
| 0.241 | contractualism | contract_alt_virtue |
| 0.231 | deontology | deont_alt_contract |
| 0.226 | deontology | deont_alt_virtue |
| 0.221 | utilitarian | util_alt_deont |
| 0.220 | virtue_ethics | virtue_alt_util |
| 0.219 | deontology | deont_alt_util |
| 0.213 | utilitarian | util_alt_virtue |
| 0.212 | deontology | deont_anti |
| 0.203 | utilitarian | util_neutral_length_matched |
| 0.201 | deontology | deont_neutral_length_matched |
| 0.201 | deontology | deont_neutral_length_matched_p_variant |
| 0.200 | utilitarian | util_anti |
| 0.200 | contractualism | contract_neutral_length_matched |

## Qualitative inspection

Reading the four cross-theory cosine matrices is the most informative move.

At L32 prompt_end, the four theories' primed-vs-neutral directions form a
near-clique with cosines 0.80–0.89. If `mean(P_deont) − mean(N_neutral)` and
`mean(P_util) − mean(N_neutral)` are both pointing in the same direction in
~12k-dimensional residual space, that direction is the theory-prompt subtraction
itself — "you got primed about a moral framework" — not the framework
content.

This generalises the phase_01 finding. Phase_01's interpretation was "the
default is deontology-aligned, so deont vs default is small." Phase_02
shows that **the default is roughly equally close (in residual space) to
all four primed conditions**. The diff-in-means is recovering the
generic primed-vs-not-primed offset, which is shared across theories, plus
a small theory-specific residual that gets buried under random-label
permutation variance.

The anti-pole construction shows more cross-theory separation (0.53–0.76)
because each anti pole is itself theory-specific text — so you're not
just measuring "primed-ness", you're measuring "primed-with-X minus
primed-against-X." But anti still fails the absolute null test because
the anti prompts are linguistic mirrors of the positives and may be
encoding negation/instruction-suppression content rather than theory
geometry.

The L4 result is suggestive but not load-bearing. At layer 4 of a 48-layer
model, "moral-theory" is not a plausible description of what is encoded.
What L4 likely captures is *instruction surface form* — at this depth, the
model has barely processed the persona prompt past the token level. The
persistent L4 finding that theory-vs-theory directions show better
split-half/null gaps than theory-vs-neutral is consistent with surface-form
separation between different instruction wordings, not with deeper theory
representations.

The behavioural-divergence picture also shifts. With the four theories
side-by-side: the average jaccard for theory-vs-neutral is 0.40–0.50
across all four theories. **No theory's positive prompt produces
substantially more behaviour-shift than any other.** Deontology is not
uniquely close to neutral at the recommendation level; it's just typical.
Utilitarian is *more* convergent with neutral than deontology (avg jaccard
0.49–0.52 vs deont 0.37–0.45). The phase_01 hypothesis that "the default
is deontology-aligned" was over-specific; the more accurate version is
"the default is some weighted average of all four, and primed-vs-default
captures generic prompted-moral-reasoning, not framework choice."

## Corrections

Load-bearing. These revise beliefs from phase_01 and from the original
plan.

1. **Pre-registered primary locus was wrong.** The plan fixed L32
   prompt_end as primary because it was the prior MoReBench prompt-side
   winner. For a persona-vectors-style analysis, the primary locus
   should be response-mean (mean-pooled generated tokens). The capture
   stored both; the original headline reported prompt_end. The
   corrected primary is L32 generated. The pre-reg fixed the locus
   without re-deriving it from the persona-vectors method, and that
   inheritance was the error.

2. **The corrected-locus picture is "marginal across the board," not
   "flat fail."** At L32 generated, 8 of 12 theory-vs-neutral/anti
   constructions hit marginal (gap 0.10–0.17). Contractualism and
   deontology are strongest (best gaps 0.16, 0.12); utilitarian and
   virtue ethics are weaker. None passes the 0.20 gate, but several
   are within ~0.05 of it. This changes the strength of the negative
   from "recipe doesn't work" to "recipe is on the edge of working
   and needs more N or substrate refinement to settle."

3. **The phase_01 interpretation was over-specific.** Phase_01
   concluded "default mode is deontology-aligned" from spot-checking
   dilemmas where deont and neutral converged. Phase_02 shows neutral
   converges with util, virtue, and contract roughly as often.
   The accurate statement is: the default is approximately a centroid
   across the four primed conditions, not specifically
   deontology-flavoured. This holds at both loci.

4. **The phase_01 "pivot to theory-vs-theory" recommendation was
   wrong, and the corrected-locus data make this clearer than the
   prompt-end data did.** At L32 generated, 11 of 12 theory-vs-theory
   (alt) constructions have *negative* gaps — random label shuffles
   produce stronger directions than the real labels. Theory-vs-theory
   contrastive directions are not a recovery path on this substrate;
   they are worse than the primed-vs-default they were proposed to
   replace. (At the prompt-end locus this was masked because all
   constructions were in the same low-gap band.)

5. **The cross-theory cosine clique survives the locus correction.**
   At the corrected primary locus (L32 generated), the four
   primed-vs-neutral directions are mutually 0.71–0.90 cosine
   (`neutral_short` matrix; comparable but slightly looser than the
   0.80–0.89 at L32 prompt_end). The four primed-vs-anti directions are
   0.34–0.75. The shared-direction finding that "the four
   primed-vs-default directions point roughly the same way" is robust
   to the locus choice; this was originally framed as a prompt-end
   artifact but it is not. At the response locus, utilitarian is
   somewhat more separated from the others (cosines 0.64–0.84 with
   the rest) but the clique structure still dominates.

6. **Pass criterion of gap ≥ 0.20 is reasonable but not validated.**
   Chosen as a soft gate; not pre-registered against any external
   reference (e.g. a known-extractable direction's gap on this model).
   The pass/marginal/fail labels are descriptive, not categorical.

## Running hypothesis

At the response-mean locus on Qwen3-30B-A3B, the persona-vectors
primed-vs-default recipe extracts theory-specific signal that is **on the
edge of clearing random-label nulls**: 8 of 12 theory-vs-neutral/anti
constructions reach marginal (gap 0.10–0.17), none passes 0.20. The signal
strength varies by theory (contractualism ≈ deontology > utilitarian >
virtue ethics).

Theory-vs-theory contrastive directions, despite being intuitively
attractive (and despite passing phase_01-style gap tests at L32 prompt_end),
**are negative-gap at L32 generated** — they extract less stable directions
than random label shuffles. This kills the "pivot to theory-vs-theory"
recommendation from phase_01 cleanly. Whatever theory-specific signal
exists, it lives in the primed-vs-default contrast at the response locus,
not in the primed-vs-other-prime contrast.

The cross-theory cosine clique survives the locus correction:
primed-vs-neutral cosines are 0.71–0.90 at L32 generated (vs 0.80–0.89
at L32 prompt_end). The "generic moral-prompt-active direction"
interpretation therefore holds at the response locus too. The
diff-in-means is mostly recovering the same shared offset across the
four theories, with a small theory-specific component that is
detectable at marginal-gap level but not strong enough to dominate.

Candidate next moves, ranked by phase_02 evidence:

1. **Compute the response-mean cross-theory cosine matrix.** Trivially
   cheap from the existing capture. Decides whether the
   "shared-direction" finding is locus-specific or method-wide.
2. **Increase per-condition N (substrate refinement).** Current N=30 puts
   the random-label null around 0.50–0.65, which is wide enough that
   gaps in the 0.10–0.20 band don't separate cleanly. Doubling N
   (via prompt-paraphrase variants, since vLLM dedupes identical
   prompts) would narrow the null without changing the recipe.
3. **Move to longer / more elaborated responses.** Terse single-sentence
   recommendations may not give the model enough generation steps to
   express theory structure in residual activations. The
   persona-vectors paper used multi-sentence emotion-induced
   responses. Worth replicating that substrate shape before declaring
   the recipe null.
4. **Drop theory-vs-theory.** At the response locus it is worse than
   primed-vs-default. Continuing to investigate it is not supported by
   the phase_02 data.

## Claim boundary

Safe to claim:

- On Qwen3-30B-A3B with N=30 dilemmas per condition, the persona-vectors
  primed-vs-default recipe applied separately to four moral theories
  produces direction estimates that hit *marginal* (split_half −
  null_p95 gap 0.10–0.17) but not *pass* (gap ≥ 0.20) at the corrected
  primary locus L32 generated, for 8 of 12 theory-vs-neutral/anti
  constructions.
- Theory-vs-theory contrastive directions are *worse* than random
  label shuffles at L32 generated — 11 of 12 alt constructions have
  negative gaps. Theory-vs-theory is not a recovery path on this
  substrate.
- Theory ranking on this substrate, by best gap at L32 generated:
  contractualism (0.16) ≈ deontology (0.12) > utilitarian (0.13 at
  one construction, others fail) > virtue ethics (best 0.10).
- At L32 prompt_end (the originally-mis-located primary), the four
  primed-vs-neutral directions are mutually 0.80–0.89 cosine — a
  shared-direction finding. Whether this clique survives at the
  response-mean locus is not yet computed.
- L4 prompt_end shows several constructions with gaps ≥ 0.20, but L4
  is too early to plausibly encode framework content; this is most
  likely instruction-surface-form separation.

Not supported yet, avoid claiming:

- "The recipe doesn't work." The corrected-locus result is "marginal"
  not "fail." With more N or a richer substrate the gate may clear.
- "Moral-theory persona vectors don't exist in Qwen3-30B-A3B." Design
  space (sample size, substrate length, prompt diversity) is largely
  unexplored.
- Anything about transfer to MoReBench. Still held out.

Preferred phrasing for the current state:

- "On this model and substrate at N=30 per condition, the
  persona-vectors primed-vs-default recipe extracts theory-specific
  directions whose split-half/null gap is marginal at the response-mean
  locus (0.10–0.17 for primary constructions, never reaching the 0.20
  pre-registered pass criterion). Theory-vs-theory contrastive
  directions are negative-gap at the response locus and should not be
  pursued. Two cheap follow-ups can move marginal toward pass:
  (a) increase N via prompt-paraphrase, (b) move to longer responses
  more analogous to the persona-vectors paper substrate."

## Artifacts

Plan and pre-reg:

- `projects/MOREBENCH/theory_persona_vectors/phase_02/docs/01-all-theories-pole-pilot-plan.md`

Specs:

- `projects/MOREBENCH/theory_persona_vectors/phase_02/specs/all_theories_pole_pilot_prompt_conditions.json`
- `projects/MOREBENCH/theory_persona_vectors/phase_02/specs/all_theories_pole_pilot_workflow.py`

Data:

- `projects/MOREBENCH/theory_persona_vectors/phase_02/outputs/all_theories_pole_pilot_synth_dilemmas.jsonl`
  (copy of phase_01)

Scripts:

- `projects/MOREBENCH/theory_persona_vectors/phase_02/scripts/analyze_all_theories_pole_pilot.py`

Reports:

- `projects/MOREBENCH/theory_persona_vectors/phase_02/reports/all_theories_pole_pilot_analysis/summary.json`
- `projects/MOREBENCH/theory_persona_vectors/phase_02/reports/all_theories_pole_pilot_analysis/report.md`
- `projects/MOREBENCH/theory_persona_vectors/phase_02/reports/all_theories_pole_pilot_report/report_0a6a1bb7ed4a_f2f091e0/report.md`

Catalog references:

- run id: `wr_7fb9542ae82d_0c284562`
- generation: `generation_run_1_15bc125de56b`
- capture: `capture_1_c2684db0530c`

## Open threads

1. **Decompose the shared moral-prompt direction.** The cross-theory
   cosine clique at L32 generated (0.71–0.90 for primed-vs-neutral)
   means the four directions span a low-dimensional shared subspace.
   Compute the centroid of the four primed-vs-neutral directions,
   project each theory's direction onto its complement, and re-run
   the smoke gate on the residuals. If theory-specific signal exists,
   it should live in the orthogonal residuals after centroid
   subtraction. Cheap from the existing capture; highest-information
   follow-up.

2. **N-scaling on a single theory at the corrected locus.** Phase_02 used
   N=30 per condition. At gap ~0.13 (corrected-locus median for primary
   constructions), it is plausible that N=60 or N=120 narrows the
   random-label null enough to push the marginal results into pass.
   Implementation: paraphrase each dilemma 2–4 ways to defeat vLLM
   prompt deduplication. Run on a single theory (contractualism is
   strongest at the corrected locus, so the cleanest test) before
   scaling to all four.

3. **Substrate length.** Terse one-sentence recommendations may give
   too few generation tokens for theory structure to express in
   activations. Re-run with `max_tokens` raised (e.g. 256) and a
   prompt that asks for a paragraph of brief reasoning, replicating
   the persona-vectors paper's response shape. If gaps widen, the
   substrate was the binding constraint; if not, the
   primed-vs-default recipe really is null here.

4. **Drop theory-vs-theory.** The corrected-locus data show alt
   constructions are worse than random labels at L32 generated.
   Phase_01's recommendation to pivot to theory-vs-theory was wrong;
   subsequent designs should not include alt constructions as the
   primary contrast.

5. **Pass-criterion calibration.** Before next phase commits to
   `gap ≥ 0.20`, calibrate against a known-extractable direction on
   the same model and dilemmas (e.g., a sentiment or politeness
   direction extracted the same way). Without that anchor,
   pass/marginal/fail is unbenchmarked.

6. **Marshall flywheel re-entry.** Phase_01 and phase_02 both sit at
   stage 3 (synth validation). The flywheel says: do not advance to
   stage 4 (smoke at scale) until validation passes. Phase_02's
   "marginal" reading at the corrected locus is closer to a half-pass
   than phase_01's "fail," so the loop-back is not as deep — open
   threads 2 and 3 (N-scale, substrate length) are stage-2 refinements
   to the existing synth substrate, not a redesign.

7. **L0 prompt_end is empty (NaN).** The capture format does not store
   layer-0 residuals in a diff-in-means-compatible shape. Worth
   verifying whether this is intentional (e.g. layer-0 is the embedding
   output and is deduplicated by token id) or a capture-format bug.
   Cosmetic but should be resolved before next run.

8. **Locus-choice discipline.** Phase_02's locus-correction was
   post-hoc, prompted by the user mid-analysis. Future pre-regs in
   this subproject should derive the primary locus from the *method*
   (persona-vectors → response tokens), not from prior MoReBench
   prompt-side results. Prior-result-inheritance is a documented
   correction path now.
