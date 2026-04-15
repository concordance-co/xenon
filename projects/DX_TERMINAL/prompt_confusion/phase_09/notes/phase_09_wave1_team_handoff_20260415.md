# Phase 09 Wave 1 Team Handoff

Date: 2026-04-15
Project: `DX_TERMINAL / prompt_confusion / phase_09`
Prepared for: team review / Marshall sync

## Why This Note Exists

We now have:
- the completed Phase 09 result
- the richer rerun with exact error accounting
- the Wave 1 follow-up

At this point the project is in a better place than the old family/arbitration versions, but the interpretation has also become more nuanced.

This note is meant to give the team one concise document that covers:
- what Phase 09 established
- what Wave 1 changed
- where my interpretation and Claude's interpretation agree
- the remaining uncertainties
- the best immediate next step

## Executive Summary

The short version is:

- `trade_size` is a very strong, clean result.
  - It looks like a near-canonical binary conflict feature.
  - The probe is almost perfect.
  - Behavior and labels are largely aligned.

- `trading_activity` is also a real result, but it is not the same kind of feature.
  - It is strong within-dimension.
  - It is not well-described as one shared linear conflict axis with `trade_size`.
  - Its errors concentrate in threshold-boundary cells, especially `setting_value=1` with non-weak evidence.

- Wave 1 strongly suggests that the activity probe may be tracking the model's actual internal state better than our synthetic binary label does.
  - This is not a return to the old lexical confound failure mode.
  - It looks more like a label-model mismatch in the hard threshold cells.

- The best next step is to rerun the boundary generation slice with much stricter output control so we can cleanly measure behavior without `<think>` contamination.

## What Phase 09 Already Established

### 1. The old family-confound problem is gone

The earlier failure mode was:
- raw text alone could strongly decode the target
- so the mechanistic story was not trustworthy

Phase 09 changed that:
- text gate on `conflict_present` stayed near chance-ish
- the main conflict signal emerged depth-progressively
- `L0` was at chance

That means:
- this is not a surface-text artifact
- the model is building the relevant internal signal during the forward pass

### 2. The pooled residual probe is strong

Main pooled result at `L36`:
- balanced accuracy: `0.875`
- AUROC: `0.9307`

Exact pooled error accounting from the rerun:
- `TP=186`
- `TN=192`
- `FP=24`
- `FN=30`
- `FPR=0.1111`
- `FNR=0.1389`

### 3. The per-dimension breakdown is the real headline

At `L36`:

`trade_size`
- settings split:
  - balanced accuracy: `0.9948`
  - AUROC: `1.0000`
- strategy split:
  - balanced accuracy: `0.9948`
  - AUROC: `1.0000`

`trading_activity`
- settings split:
  - balanced accuracy: `0.9167`
  - AUROC: `0.9588`
- strategy split:
  - balanced accuracy: `0.8833`
  - AUROC: `0.9663`

So the clean empirical picture from Phase 09 was:
- `trade_size` is almost perfectly linearly readable
- `trading_activity` is also strongly readable, but clearly noisier

## What The Rerun Added

The rerun mattered because it gave us:
- exact `FPR / FNR / TPR / TNR`
- confusion counts
- per-example predictions

That revealed that the activity probe errors were not generic noise.

They concentrated in a small set of threshold-boundary cells, especially:
- `val=1 + solid`
- `val=1 + exceptional`

The important shift was:
- the activity probe did not look weak
- it looked systematically misaligned with the binary label in semantically specific places

This made the problem much more interpretable.

## What Wave 1 Added

Wave 1 had three components:

1. cross-dimension transfer with `compare_direction_similarity=True`
2. matched-pair delta analysis
3. generation on the activity boundary slice

### A. Cross-dimension similarity result

This was the most decisive structural result.

Direction similarity:
- `L24`: `0.0637`
- `L36`: `0.0513`
- best overall: `0.0835` at `L32`

Cross-dimension transfer at `L36`:
- `trade_size -> trading_activity`
  - balanced accuracy: `0.5333`
  - AUROC: `0.6000`
- `trading_activity -> trade_size`
  - balanced accuracy: `0.5312`
  - AUROC: `0.6867`

Within-dimension remained strong:
- `trade_size`: balanced accuracy `1.000`
- `trading_activity`: balanced accuracy `0.9126`, AUROC `0.9661`

Interpretation:
- there is not one shared linear residual-space conflict direction across the two dimensions
- they are better understood as separate dimension-specific features

### B. Pair-delta result

The pair-delta analysis succeeded on:
- `288` matched pairs
- `12` layers

Computed summary:

At `L24`:
- mean cross-dimension pair-delta cosine: `0.1216`

At `L36`:
- mean cross-dimension pair-delta cosine: `0.3691`

At `L36`, delta norms:
- `trade_size`
  - mean delta norm: `0.7386`
  - mean per-pair magnitude: `1.6309`
- `trading_activity`
  - mean delta norm: `0.3180`
  - mean per-pair magnitude: `2.1490`

Interpretation:
- `trade_size` deltas are more coherent and collapse onto a cleaner common direction
- `trading_activity` deltas are individually larger, but much less direction-consistent

This fits the main Phase 09 story very well:
- size conflict is clean and one-dimensional
- activity conflict is real, but more heterogeneous

### C. Boundary generation result

This was the most interpretively important piece, but also the messiest operationally.

Boundary slice:
- `192` rows
- all `trading_activity`
- all `setting_value=1`
- contexts: `solid` or `exceptional`

Problem:
- model frequently emitted visible reasoning or truncated outputs
- only `128 / 192` rows had parseable JSON
- so the exact percentages should not yet be treated as final

Still, the parsed subset was directionally very revealing:

`observe + solid`
- parsed: `33`
- model observed: `31`
- exact: `31`

`trade + solid`
- parsed: `34`
- model observed: `24`
- model bought: `10`
- exact: `24`

`trade + exceptional`
- parsed: `35`
- model bought: `24`
- model observed: `11`
- exact: `21`

`observe + exceptional`
- parsed: `26`
- model observed: `20`
- model bought: `6`
- exact: `6`

The most important cell:
- `observe + exceptional + setting=1`
- label says: `buy`
- model often: `observe`

That is exactly the kind of cell where the activity probe had looked "wrong" relative to the label.
Wave 1 makes it plausible that:
- the probe is not wrong about the model
- the label is wrong about what the model actually computes in that boundary case

## My Current Interpretation

My current view after Phase 09 + rerun + Wave 1 is:

### 1. We are not back to lexical confounds

This does **not** look like the old family-decoding failure mode.

Why:
- text gate on `conflict_present` is still weak
- `L0` is still at chance
- the signal still emerges depth-progressively
- the hard cases are semantically concentrated, not globally lexical

So the problem is not:
- "text gave the answer again"

It is more like:
- the model is building a real internal feature
- but for activity, that feature does not exactly match our binary threshold label

### 2. `trade_size` is a clean success

`trade_size` looks like:
- a clean, nearly one-dimensional conflict representation
- highly linearly readable
- stable across split variants
- behaviorally aligned with the benchmark semantics

This is the strongest result in the project.

### 3. `trading_activity` is real, but not a simple binary conflict feature

The activity signal is still clearly real:
- strong AUROC
- strong within-dimension readout
- depth progression

But the structure appears different:
- more threshold-sensitive
- more semantically graded
- less direction-consistent
- not reducible to the exact same linear axis as size

My best current description is:
- the activity probe is reading something like
  - `restrictive setting relative to evidence quality`
or
  - `policy-source tension under evidence-sensitive gating`

rather than a perfectly crisp binary label in every cell.

### 4. The likely issue is label-model mismatch, not probe failure

In the hard activity boundary cells, especially `val=1 + exceptional`, the model appears to often treat the restrictive setting as still operative even when our threshold logic says the bar is technically cleared.

So the likely situation is:
- our synthetic label implements a discrete threshold rule
- the model often behaves according to a softer semantic reading of the setting
- the probe tracks that softer internal state

That would mean:
- the probe is more faithful to the model than the label is, at least in those cells

## Claude's Interpretation

Claude's latest review is very aligned with the broad direction above, but is more forceful in a few places.

### Where Claude agrees with me

Claude agrees that:
- Wave 1 strengthens the story rather than undermining it
- this is not a return to lexical confounds
- `trade_size` and `trading_activity` are not one shared linear feature
- `trade_size` is clean and coherent
- `trading_activity` is real but more heterogeneous
- the boundary generation result matters a lot
- the next step should be a clean rerun of the boundary generation slice

### Where Claude is sharper / stronger

Claude makes three especially strong claims:

1. The low direction cosine is effectively definitive.
   - `0.0513` at `L36`
   - max `0.0835`
   - interpretation: there is no shared linear conflict direction across the two dimensions

2. The activity probe is likely tracking the model's actual internal state better than the label does.
   - especially in `val=1 + exceptional`
   - the model often does not behave as though the threshold override really happened

3. The best framing is no longer:
   - "the activity probe encodes something slightly fuzzy"
   but:
   - "the probe is aligned with the model, and the synthetic label diverges from the model in threshold-boundary cells"

Claude's summary sentence is basically:
- Wave 1 reveals a deeper finding, not a deeper problem

## Where I Agree vs Where I Am Slightly More Cautious

### I agree strongly on:

- no shared linear direction across size and activity
- no return to old lexical confounds
- activity is multi-directional / heterogeneous
- boundary generation is the most consequential next frontier

### I am slightly more cautious on:

- how hard to lean on the generation result before we fix formatting

I think the qualitative direction is already persuasive:
- `observe + exceptional` really does look like a model/label mismatch cell

But I would still be careful not to make a strong quantitative claim until we rerun that slice with:
- no visible reasoning
- higher `max_tokens`
- cleaner output control

So my current phrasing would be:
- "Wave 1 strongly suggests the probe is more faithful to the model than the label is in those cells"

Claude is comfortable phrasing that a bit more strongly already.

## Synthesis: Best Team-Level Interpretation Right Now

If I had to condense both my view and Claude's into one team-facing interpretation, it would be:

1. `trade_size` and `trading_activity` should no longer be treated as one shared "conflict feature."
   - Both are real.
   - Both are depth-progressive.
   - But they are not one shared linear residual direction.

2. `trade_size` is a clean binary benchmark success.
   - This is our strongest and simplest result.

3. `trading_activity` is also a real representation, but it is richer than the current binary synthetic label.
   - It appears to reflect restrictive-policy-vs-evidence tension.
   - The model may not actually implement our discrete threshold override in the hardest cells.

4. The main outstanding question is no longer lexical confounding.
   - It is behavioral and semantic fidelity.
   - Specifically: when the probe and the synthetic label disagree, which one better matches the model's actual decision process?

## Recommended Immediate Next Step

Both Claude and I think the immediate priority is the same:

### Rerun the activity boundary slice with strict output control

Goal:
- get a clean behavioral read on the exact cells where the probe/label disagreement is concentrated

Recommended changes:
- disable visible reasoning / thinking output
- increase `max_tokens` to at least `256`
- keep `temperature=0.0`
- keep the same boundary slice:
  - `trading_activity`
  - `setting_value=1`
  - `solid` and `exceptional`

Why this is the best next move:
- it directly tests the most important unresolved claim
- it is much cheaper than expanding the benchmark
- it tells us whether the probe is tracking the model or just some internal-but-irrelevant tension feature

## What I Would Want Marshall's Help On

I do think this is a good moment to get Marshall back in the loop, because the project has moved from:
- "did we deconfound the benchmark?"
to
- "what exactly counts as the right target when the model's internal state diverges from our synthetic label?"

The questions that feel especially worth Marshall's take:

1. Should `trading_activity` still be framed as a conflict benchmark with imperfect labels?
   - or should it now be reframed around restrictiveness / tension directly?

2. How strong a claim should we make from the current Wave 1 behavior evidence before the clean rerun?

3. Is the right next milestone:
   - fixing behavioral measurement
   - residualizing the probe against restrictiveness / evidence tier
   - or splitting the project more explicitly into:
     - clean size-conflict benchmark
     - richer activity-tension benchmark

## Proposed Team Headline

If we need one short status line for internal sharing, I would use:

> Phase 09 worked, but not in the simplest possible way: size conflict is a clean near-perfect binary feature, while activity conflict appears to be a stronger-than-expected, semantically richer tension feature whose probe may actually be more faithful to the model than our thresholded synthetic label is.

## Appendix: Concrete Results Referenced

### Main Phase 09 result

At `L36`:
- pooled balanced accuracy: `0.875`
- pooled AUROC: `0.9307`

### Text gate

- settings split:
  - balanced accuracy: `0.5579`
  - AUROC: `0.5237`
- strategy split:
  - balanced accuracy: `0.4977`
  - AUROC: `0.5234`

### Wave 1 direction similarity

- `L24`: `0.0637`
- `L36`: `0.0513`
- max: `0.0835` at `L32`

### Wave 1 cross-dimension transfer at `L36`

- `trade_size -> trading_activity`
  - balanced accuracy: `0.5333`
  - AUROC: `0.6000`
- `trading_activity -> trade_size`
  - balanced accuracy: `0.5312`
  - AUROC: `0.6867`

### Wave 1 pair-delta summary at `L36`

- mean delta cosine across dimensions: `0.3691`
- `trade_size` mean delta norm: `0.7386`
- `trading_activity` mean delta norm: `0.3180`
- `trade_size` mean per-pair magnitude: `1.6309`
- `trading_activity` mean per-pair magnitude: `2.1490`

### Wave 1 boundary generation parsed subset

`observe + solid`
- observed: `31/33`

`trade + solid`
- observed: `24/34`
- bought: `10/34`

`trade + exceptional`
- bought: `24/35`
- observed: `11/35`

`observe + exceptional`
- observed: `20/26`
- bought: `6/26`
