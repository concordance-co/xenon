# Benchmark-First Mech Interp Failure Modes

**Date:** 2026-04-21

## Purpose

This document is a grounding check for the `benchmark -> mech interp discovery` program.

The goal is not to be pessimistic for its own sake.
The goal is to make sure we are:

- not confusing rich labels with real internal variables
- not overclaiming from probe results
- not mistaking benchmark artifacts for model mechanisms
- not building a lot of infrastructure around weak scientific targets

The intended use is:

- read this before starting a new benchmark-first spec
- use it as a checklist during result review
- use it to decide when to narrow, repair, or reinterpret a benchmark-feature line of work

## The Core Risk

The main failure mode is simple:

`a benchmark can be rich, realistic, and well-labeled, and still fail to produce a clean mechanistic target`

There are many ways this can happen.

## Primary Failure Modes

### 1. The Label Does Not Correspond To A Real Internal Variable

Some labels are too broad, too composite, or too socially constructed to map cleanly onto a latent feature.

Examples:

- “good counseling response”
- “helpfulness”
- “good moral reasoning”

These may bundle many distinct internal processes:

- style
- caution
- knowledge
- empathy
- task competence
- formatting discipline

What this looks like in practice:

- weak or unstable probes
- multiple incompatible high-performing probes
- poor transfer across domains or prompt templates
- hard-to-interpret steering behavior

How to respond:

- narrow the label
- decompose it into smaller sub-labels
- prefer concrete rubric dimensions over global scores

### 2. The Model Solves The Task With Shortcuts

The benchmark may look like it measures one thing, but the model may be using cheap lexical, structural, or formatting cues.

Examples:

- moral theory labels correlate with wording style
- advice quality labels correlate with verbosity
- hierarchy labels correlate with role tokens
- hallucination labels correlate with citation absence

What this looks like:

- bag-of-words or shallow baselines perform surprisingly well
- performance collapses on lexical or template holdouts
- probe signal is very strong but only on the exact benchmark format
- same-content prompt rewrites destroy separability

How to respond:

- run lexical splits early
- run template and domain holdouts
- construct matched same-label controls
- rewrite prompts while preserving labels

### 3. The Signal Is Readable But Not Causally Useful

A probe may decode a variable well without that variable being writable or causally load-bearing.

This is one of the most common mech interp traps if we overclaim mechanism.
It is not automatically a problem if the goal is monitoring or outcome prediction.

What this looks like:

- high AUROC
- weak steering
- weak patching
- late-layer readouts with little intervention leverage

How to respond:

- distinguish read layer from write layer
- use earlier candidate intervention sites
- avoid making mechanism claims from readout alone
- keep indicator-style features if they are robust, transferable, and operationally useful

### 3.1 Readout Features Are Still Valuable

For this program, a feature does not need to be strongly causal to be worthwhile.

There are at least three different success modes:

- `indicator success`
  The feature is a strong, robust predictor of an outcome we care about.
- `mechanistic success`
  The feature is part of the actual computation and survives stronger causal tests.
- `control success`
  The feature can be steered, patched, or otherwise manipulated in a useful way.

These should not be conflated.

A benchmark-first project can still be very successful if it yields:

- a robust monitor
- a transferable warning signal
- a stable internal score

even if the feature is not itself the causal engine of the behavior.

The real requirement is:

- do not oversell indicator features as mechanisms
- do not throw away useful indicators just because they are not clean intervention sites

### 4. The Feature Is Too Distributed To Productize

A real feature may exist, but be too distributed, entangled, or unstable to become a useful probe/vector/monitor.

What this looks like:

- no compact direction
- intervention only works with large, lossy, multi-layer edits
- probe varies a lot across models
- feature collapses under realistic shift

How to respond:

- downgrade from “ship candidate” to “scientific curiosity”
- focus on transfer and robustness before product framing

### 5. The Benchmark Is Too Benchmarky

A benchmark can be well designed but still too unlike real product traffic.

Examples:

- synthetic role conflict that is cleaner than any real agent trajectory
- moral dilemmas that are too explicit
- advice tasks with unusually neat structure

What this looks like:

- beautiful results that fail on real-world logs
- synthetic benchmark transfer is weak
- discovered features are overly tied to benchmark scaffolding

How to respond:

- validate on a messier second dataset
- use synthetic benchmarks as warm-starts, not endpoints
- add realistic perturbations early

### 6. Competence And The Target Variable Are Confounded

Sometimes the benchmark failure mode mixes:

- not understanding the task
- not having the knowledge
- not representing the intended state

This is especially common in:

- agent benchmarks
- legal/medical benchmarks
- long-horizon environments

What this looks like:

- low benchmark performance overall
- poor examples are ambiguous between incompetence and bad internal state
- claimed “safety” feature may really be “task success” feature

How to respond:

- separate competence labels from behavioral labels
- use paired or matched subsets
- audit errors manually before probing

### 7. The Label Is Good, But The Dataset Is Too Small

Some benchmarks are conceptually perfect but too small for reliable discovery work.

What this looks like:

- unstable probe coefficients
- wide variance across splits
- results depend heavily on regularization
- transfer claims are noisy

How to respond:

- treat the dataset as a seed set
- augment with synthetic expansions
- start with simpler contrastive directions before heavy modeling

### 8. The Work Is Scientifically Interesting But Commercially Weak

A feature may be real and publishable, but not convert into a useful product component.

Examples:

- elegant moral-theory representation with no operational use
- subtle geometry result that does not improve monitoring
- benchmark-specific decomposition that no customer cares about

What this looks like:

- strong internal excitement, weak external wedge
- hard to explain why the artifact matters in deployment
- no clear path from vector/probe to a monitor or workflow

How to respond:

- separate `research success` from `product success`
- be explicit which track a project is on

## Anti-Goals

These are things the program should explicitly avoid.

### Anti-goal 1: Probe Theater

Do not treat a decodable probe as equivalent to a mechanism.

Bad pattern:

- train probe
- get good metric
- declare feature “found”

Better pattern:

- verify behavioral sanity
- probe
- localize
- test transfer
- test interventions
- shape claim to evidence level

### Anti-goal 2: Benchmark Prestige Chasing

Do not choose benchmarks because they are famous.

We care about:

- label richness
- latent-variable plausibility
- product relevance
- methodological tractability

not generic benchmark prestige.

### Anti-goal 3: One Benchmark, One Feature

Do not collapse a benchmark into one feature track.

The whole point of benchmark-first work is that one benchmark may yield:

- multiple labels
- multiple feature hypotheses
- multiple artifacts

### Anti-goal 4: One Probe, One Conclusion

Do not make a large scientific or product decision from one probe on one split.

Minimum healthy skepticism:

- alternate split
- alternate prompt form
- control baseline
- manual error inspection

### Anti-goal 5: Confounds As Cleanup

Confound planning is not a final step.
It is part of benchmark design and feature selection.

If a result only survives before the obvious lexical split, it was never strong.

### Anti-goal 6: Product Framing Too Early

Some results should remain:

- exploratory
- scientific
- internal

until they clear transfer and robustness checks.

## A More Realistic Success Model

The wrong success model is:

`every rich benchmark will produce several valuable mechanistic findings`

The better success model is:

`a few rich benchmarks will produce a few reusable internal variables, and the process will help us find which ones are real`

That is enough.

If a benchmark-first program yields:

- one strong probe family
- one strong transfer result
- one strong intervention result
- one product-legible monitor

that is already extremely valuable.

## What Would Make The Program Clearly Worthwhile

The benchmark-first direction is working if we start to see results like:

- the same feature appears across multiple benchmarks or domains
- a benchmark-derived probe transfers into realistic product traffic
- a benchmark-derived direction can be steered or patched causally
- one benchmark yields multiple distinct reusable artifacts
- the process quickly narrows weak ideas, repairs solvable issues, and concentrates effort on strong ones

## MoReBench-Specific Hope

`MoReBench` is still one of the best early validation candidates because:

- the labels are rich
- the benchmark is conceptually legible
- there are multiple feature hypotheses available immediately
- it supports several methodology families

But even there, we should expect some candidate features to fail.

The right goal is not:

`MoReBench must validate everything`

The right goal is:

`MoReBench should tell us quickly whether this overall program can produce clean, reusable internal variables from rich benchmark labels`

## Bottom Line

This effort is not dumb.

The real danger is not that the idea is incoherent.
The real danger is:

- overclaiming from weak readouts
- underestimating confounds
- failing to separate benchmark artifacts from genuine internal structure
- refusing to narrow or reinterpret attractive but weak feature hypotheses

If we stay disciplined about those, the program is a real bet, not a fantasy.
