# Phase 09 Methodology Notes

This file is the running methodology log for `prompt_confusion / phase_09`.

Its purpose is to capture:
- prompt-design changes
- dataset-design changes
- benchmark-scope changes
- pre-capture and post-capture methodological learnings
- interpretation updates that affect how future work should be run

This is not the main result writeup.
It is the living record of how and why the benchmark changed.

## Core Position

Phase 09 is the first prompt-confusion benchmark version where the main
target looks methodologically defensible:

- the project is no longer organized around `family`
- the benchmark is organized around relational conflict
- prompt structure is more explicit
- market text is more descriptive and less action-coded
- target-setting values are simplified to reduce label ambiguity

The Phase 09 stance is:

- `STRATEGY` expresses preference / style
- `ACTIVE SETTINGS` express binding execution constraints
- `MARKET` describes evidence about the world

The benchmark is intended to measure whether the model internally
represents disagreement between policy sources, not whether it can simply
decode a family identity from text.

## Prompt Philosophy Changes

### 1. Family was removed as the organizing construct

Earlier phases treated strategy family as a meaningful benchmark
variable.

The main lesson from Phase 05 and Phase 07 was:
- family identity was too tightly tied to semantic polarity
- the prompt surface faithfully stated that polarity
- raw text could therefore decode the family too easily

Phase 09 does not attempt to rescue that construct.

Instead, the benchmark asks a narrower and cleaner question:
- do `STRATEGY` and `ACTIVE SETTINGS` agree or disagree?

### 2. Prompt blocks were assigned clearer roles

We explicitly moved toward:

- `STRATEGY`
  - a preference
  - defeasible
  - should matter only within what settings allow

- `ACTIVE SETTINGS`
  - binding constraints
  - the normative execution policy

- `MARKET`
  - descriptive evidence only
  - should not directly tell the model what action to take

This was a deliberate response to earlier prompt language that mixed:
- vague threshold words
- action-coded market descriptions
- and strategy phrasings that were too close to hard prohibitions

### 3. Market wording was made more observational

Older prompt versions used language like:
- `live case`
- `clear live case`
- `strong live case`

These were abandoned because:
- they were semantically vague
- they were hard for us to interpret consistently
- they encouraged leakage between world description and implied action

Phase 09 moved toward descriptive axes like:
- momentum
- confirmation
- uncertainty
- follow-through

The goal was:
- let `MARKET` describe the world
- let `ACTIVE SETTINGS` define the threshold
- force the model to perform the mapping

### 4. Target-setting values were simplified

One major Phase 09 refinement was to use extreme target values only.

For target settings:
- use `1` and `5`
- avoid medium target values in the primary task

Motivation:
- intermediate values introduced threshold ambiguity
- some of the old "behavioral failures" were really benchmark-design
  ambiguity
- a first-pass conflict benchmark should be cleaner and more binary

Important nuance:
- intermediate values can still be useful on non-target settings for
  realism
- but for the target setting they were removed from the primary setup

### 5. `medium` action size was removed

The action schema was simplified to:
- `small`
- `large`
- `none`

Motivation:
- `medium` did not add much analytical value
- it complicated behavior scoring
- it made target labels less crisp

This change improved behavioral interpretability and simplified the
trade-size benchmark.

## Dataset Design Changes

### 1. Conflict is relational by construction

Phase 09 inherits the main Phase 08 pivot:
- conflict should not be a family identity
- conflict should not be a unigram-level property
- conflict should be a relation between prompt spans

That means:
- the same direction tokens can appear in both aligned and conflict rows
- the benchmark target should not be trivially recoverable from raw text

### 2. `trade_size` and `trading_activity` were kept as separate dimensions

We intentionally preserved two policy dimensions:
- `trade_size`
- `trading_activity`

Motivation:
- support replication across multiple conflict types
- test whether the model builds similar or different representations

Important later learning:
- they are not the same kind of conflict feature

### 3. Weak-market rows were removed from `trade_size`

We learned during Phase 09 smoke testing that:
- forcing the model to buy in weak-market `trade_size` rows was a bad
  setup
- those rows created behavioral failures that were really benchmark
  design problems

So the benchmark was simplified:
- do not force trade-size decisions in obviously weak market setups

This made the size dimension much cleaner.

## Pre-Capture Learnings

### 1. Lexical gating is now mandatory

Phase 09 should always be interpreted under the rule:
- run a text-only gate before treating a dataset as capture-ready

This comes directly from the earlier confound failures.

The basic requirement is:
- raw text should be near chance-ish on the main `conflict_present`
  target

If that fails, the benchmark is not ready.

### 2. Behavior sanity is also mandatory

We also learned that passing the text gate is not enough.

The benchmark can still fail if:
- expected outputs are not behaviorally sane
- the model consistently interprets the prompt differently than the
  labeling logic does

So Phase 09 should always be read as requiring:
- lexical gate
- behavior smoke
- only then capture

## Post-Capture Learnings

### 1. `trade_size` is the cleanest success case

Current Phase 09 evidence strongly suggests:
- `trade_size` is a clean, linearly readable, depth-progressive conflict
  feature
- it is highly coherent geometrically
- benchmark labels and model behavior are relatively well aligned

This is the strongest part of the project so far.

### 2. `trading_activity` is real, but more semantically complex

Phase 09 and Wave 1 both suggest:
- activity conflict is not simply a noisier version of size conflict
- it is a different, more threshold-sensitive computation

The current best description is something like:
- restrictiveness / evidence tension
- or policy-source disagreement under evidence-dependent gating

This matters because it means:
- benchmark labels can diverge from the model's actual internal state
  even when the probe is still reading something real

### 3. There is no shared linear conflict direction across dimensions

Wave 1 found:
- very low cosine similarity between the best trade-size and
  trading-activity directions
- weak cross-dimension transfer
- strong within-dimension readouts

Current interpretation:
- there is not one shared linear "conflict axis"
- there are separate dimension-specific features

This should affect future claims:
- do not describe Phase 09 as discovering one universal conflict
  direction
- do describe it as finding strong, depth-progressive, dimension-specific
  policy-disagreement features

### 4. Some activity "probe errors" may really be label-model mismatches

The most important Phase 09 Wave 1 learning is:
- in `setting_value=1` activity boundary cells, especially under strong
  evidence, the model often behaves as though the restrictive setting
  still dominates
- the probe often tracks that state
- the synthetic label often says the threshold has already been cleared

Current implication:
- some activity probe "errors" are likely probe-label mismatches rather
  than probe-model mismatches

This is an important methodological warning:
- do not assume the synthetic label is always the best ground truth for
  the model's internal state

## Current Best Practices For Future Phase 09 Work

If extending or rerunning Phase 09, default to:

1. Preserve the relational conflict framing.
2. Preserve descriptive market wording.
3. Keep `STRATEGY` as preference and `ACTIVE SETTINGS` as constraints.
4. Prefer simple, extremes-only target-setting designs unless there is a
   very specific reason to add intermediate target values.
5. Treat lexical gates and behavior smokes as mandatory.
6. Treat `trade_size` and `trading_activity` as potentially different
   representational objects rather than assuming one shared feature.
7. When probe results disagree with labels in activity boundary cells,
   check behavior before assuming the probe is wrong.

## Open Methodology Questions

These remain unresolved and should be revisited as work continues:

1. Should `trading_activity` still be framed as a binary conflict task,
   or should it be reframed around policy restrictiveness / evidence
   tension?
2. Should future activity benchmarks use a cleaner threshold regime so
   the label matches the model's likely computation more closely?
3. How much of the activity signal should be treated as conflict proper
   versus a broader constraint-satisfaction feature?
4. Should new dimensions like `risk` or `diversification` be added only
   after the activity boundary behavior is fully cleaned up?

## Maintenance Note

Update this file when:
- prompt wording changes materially
- target-setting semantics change
- lexical gate criteria change
- behavior smoke criteria change
- new post-capture findings change how the benchmark should be
  interpreted

The goal is that a future reader can understand not only what Phase 09
is, but how it became that way.
