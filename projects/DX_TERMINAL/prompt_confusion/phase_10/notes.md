# Phase 10 Methodology Notes

This file is the running methodology log for `prompt_confusion / phase_10`.

Its purpose is to capture:
- prompt-design changes
- behavioral findings
- benchmark-scope decisions
- interpretation updates
- the handoff point into Phase 11

This is not the main result writeup.
It is the working record of what we learned from `risk_preference`.

## Core Position

Phase 10 was built as the first clean post-Phase-09 expansion beyond
`trade_size`.

The design goal was:
- stay prompt-local
- stay binary and relational
- avoid hidden temporal state
- avoid entry-threshold semantics

That goal was met much more cleanly than `trading_activity`.

Current Phase 10 stance:
- `risk_preference` is a robust conflict-detection benchmark
- it is not a symmetric conflict-resolution benchmark
- it should be treated as the second established prompt-local family
  alongside `trade_size`

## Why Phase 10 Worked

### 1. Risk was framed as asset selection, not gating

The decisive design choice in Phase 10 was:
- treat risk as `ALPHA` vs `BETA`
- not as a threshold for whether trading is allowed

That kept the problem close to `trade_size`:
- always buy
- vary which output field is resolved by the conflict
- keep the benchmark local to the prompt

This avoided the main Phase 09 activity failure mode:
- no hidden trade history
- no evidence-threshold override logic
- no synthetic label that depended on an implicit temporal state

### 2. Both primary assets were kept viable

The benchmark only made sense once:
- `ALPHA` looked like the cleaner / bounded / lower-variance expression
- `BETA` looked like the faster / wider / higher-upside expression
- neither asset looked globally wrong

This took a few prompt iterations.

The important lesson is:
- risk conflicts work better when the contrast is descriptive
- not when one asset sounds obviously superior and the other sounds
  reckless

## What The Final Phase 10 Result Is

### 1. Text gate stayed at chance

The quick text gate and the workflow text baselines both stayed at chance.

This is an important validation:
- the target is still relational
- the benchmark did not collapse into a lexical shortcut

### 2. Standard holdouts were very strong

Standard workflow result:
- XOR split:
  - balanced accuracy `0.9635`
  - AUROC `0.9766`
- strategy holdout:
  - balanced accuracy `0.9844`
  - AUROC `0.9937`
- settings holdout:
  - balanced accuracy `0.9740`
  - AUROC `0.9839`

This puts `risk_preference` in the same broad quality tier as
`trade_size`.

### 3. Strict both-axes holdout still held up

Marshall-style strict holdout result:
- text baseline:
  - balanced accuracy `0.500`
  - AUROC `0.500`
- probe:
  - balanced accuracy `0.8854`
  - AUROC `0.9119`

This matters because it shows the result survives a tougher lexical test
than the original single-axis Phase 09 splits.

### 4. Aligned behavior was clean

Full behavior run on all `384` rows showed:
- valid JSON `1.0`
- aligned rows exact `1.0`

This is the most important behavior fact for benchmark validity.

The benchmark did not reproduce the Phase 09 activity problem where
aligned rows themselves looked semantically unstable.

## What Did Not Fully Clean Up

### 1. Conflict resolution is asymmetric

Conflict rows were not behaviorally symmetric.

The stable pattern is:
- conservative settings bind strongly
- aggressive settings act more like permission than obligation

This means:
- low-risk settings are interpreted like hard prohibitions
- high-risk settings are interpreted like allowable ceilings

Current implication:
- Phase 10 is strong for conflict detection
- Phase 10 is weaker for claims about how the model resolves risk
  conflict

This is an important caveat, but it is not a benchmark failure in the
same way as `trading_activity`.

### 2. Prompt softening experiments hurt more than they helped

We tried trimming some of the more instructional wording.

The important observed lesson was:
- removing too much operational guidance degraded aligned behavior
- keeping the better-resolved wording was the right call

So the settled Phase 10 prompt should be preserved as the baseline for
future comparison.

## Geometry Update

The most important post-Phase-10 development was the direct comparison
between the two trustworthy prompt-local dimensions:
- Phase 09 `trade_size`
- Phase 10 `risk_preference`

That comparison showed:
- max direction similarity `0.5341` at `L36`
- `trade_size -> risk_preference` transfer:
  - balanced accuracy `0.9062`
  - AUROC `0.9832`
- `risk_preference -> trade_size` transfer:
  - balanced accuracy `0.8568`
  - AUROC `0.9773`

Current implication:
- we should no longer describe the project as finding only totally
  separate conflict axes
- the stronger interpretation is:
  - a shared conflict-family component exists across the clean
    prompt-local dimensions
  - with dimension-specific specialization layered on top

This also means the old `trade_size` vs `trading_activity` near-zero
cosine should not be treated as a project-level geometry conclusion.

## What Comes Next

Phase 10 is the checkpoint we intend to come back to for activation
patching once the infrastructure is ready.

That remains a high-priority validation step, especially for:
- the clean `trade_size` direction
- the shared `trade_size` / `risk_preference` conflict-family component

But the immediate next phase is not patching.

The current plan is:
- move to Phase 11 multi-conflict prompts
- combine `trade_size` and `risk_preference` in the same prompt
- test whether both conflicts are simultaneously decodable
- treat that as the intermediate step before returning to causal work on
  Phase 10

## Activation Patching Return

We returned to Phase 10 once the new activation-patching infrastructure was
available.

The first pass was intentionally narrow:
- family: `trade_size`
- rows: conflict rows only
- write site: `L32`
- token: last prompt token
- operator: `SwapMeanPatch`
- target patch:
  - move conflict rows toward the aligned centroid
- controls:
  - same-label centroid patch
  - random-direction norm-matched patch

The concrete question was:
- if the readable `trade_size` conflict representation at `L32` is causally
  load-bearing, does pushing the residual state toward the aligned centroid
  reduce settings-following and increase strategy-following on the `size`
  field?

### First-pass result

Run:
- workflow run id: `wr_89401f421728_dfe90da5`

Outcome:
- effectively null at this site

Key numbers on the `192` conflict rows:
- baseline follows setting:
  - `0.9740` (`187/192`)
- `swap_to_aligned` intended erasure flips:
  - `0.0000` (`0/192`)
- same-label control intended flips:
  - `0.0000` (`0/192`)
- random control intended flips:
  - `0.0052` (`1/192`)
- malformed rate for all three patch variants:
  - `0.0000`

Current interpretation:
- this is not evidence against the existence of the `trade_size`
  representation
- it is evidence that `L32` last-token `SwapMeanPatch` is not enough, by
  itself, to causally move `size` behavior in a useful way

The likely possibilities are:
- the best read layer is not the best write layer
- last-token patching is too lossy at this site
- the representation is causally real but redundant / downstream-consumed
  enough that a single-layer single-token patch has weak leverage

### What this changes

The right next causal step is no longer:
- “prove patching works at `L32`”

It is:
- sweep write layers, especially earlier ones like `L28`
- only move to `risk_preference` after `trade_size` shows a cleaner signal

The null is still valuable because it narrows the causal story:
- strong readout does not automatically imply strong intervention at the same
  site
- we should expect write/read dissociation and subset-local effects rather than
  a clean universal switch

### Layer sweep update

We followed the `L32` null with a last-token layer sweep:
- `L28`
- `L32`
- `L36`
- `L40`

Same setup:
- `trade_size`
- conflict rows only
- `SwapMeanPatch`
- same-label and random controls

Current summary:
- `L32` and `L40` stayed essentially null
- `L28` produced more movement than `L32`, but not a clean directional win over
  controls
- `L36` was the strongest directional-looking layer, but still weak:
  - aligned-centroid intended flips: `3/192`
  - same-label control intended flips: `2/192`

Current causal stance:
- there is still no strong activation-patching result on `trade_size`
- the best current interpretation is not “the feature is fake”
- it is:
  - last-token single-layer `SwapMeanPatch` is too weak / lossy for a clean
    causal demonstration here

So the next meaningful upgrade, if we continue, should be:
- higher-fidelity patching
- or a more position-aware intervention

Not just:
- more of the same low-bandwidth mean-swap sweeps

### Interchange follow-up

We then upgraded from centroid swap to matched donor-row interchange on
`trade_size`.

Setup:
- donor = aligned row
- target = conflict row
- pair key = `matched_group_id`
- write layers:
  - `L28`
  - `L36`
- token:
  - last prompt token

Outcome:
- interchange was stronger than mean-swap in the weak sense that it moved more
  rows
- but it still did not produce a convincing causal effect

Key numbers:
- `L28` interchange:
  - changed rows: `7/192`
  - intended flips: `0/192`
- `L36` interchange:
  - changed rows: `4/192`
  - intended flips: `1/192`

Important pattern:
- at `L28`, the dominant changed outcome was `size = none`
- not the desired strategy-consistent `large`

Current interpretation:
- this supports the view that higher-fidelity patching helps somewhat
- but the current last-token single-layer intervention is still not sufficient
  for a strong causal claim
