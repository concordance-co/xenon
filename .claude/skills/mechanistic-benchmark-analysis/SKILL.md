---
name: mechanistic-benchmark-analysis
description: Use when planning or reviewing a full mechanistic benchmark workflow: behavioral sanity checks, probe analysis, span localization, causal patching, attention follow-up, and interpretation. Especially useful for turning an initial benchmark idea into a credible mechanistic result without skipping necessary controls.
---

# Mechanistic Benchmark Analysis

Use this skill when the user wants to:

- run a full benchmark-driven mech-interp workflow
- move from behavioral evaluation to probing to causal tests
- diagnose why a mechanistic story is weak or confusing
- structure a benchmark so later interpretability claims will hold up
- synthesize results from probes, patching, and attention into one story

This skill is the top-level workflow skill. It sits above narrower skills like:

- synthetic benchmark design
- activation patching / causal evals

## Core rule

`Do not jump to mechanism before establishing the task is real.`

The proper order is:

1. define the task
2. verify behavioral sanity
3. probe for the signal
4. localize the signal
5. test causal leverage
6. add mechanism-oriented follow-ups
7. write the claim at the strength the evidence actually supports

## Canonical workflow

### Stage 1: Behavioral sanity

Before any probing:

- inspect real examples from each class
- run the base model on a small slice
- verify output parsing
- verify the benchmark is being solved at a reasonable rate
- inspect errors manually

If the task is not behaviorally sane, stop and repair it first.

### Stage 2: Global readout

Once behavior is sane, test whether the target variable is represented at all.

Typical first pass:

- last-token residual probe
- maybe router probe if using MoE models

### Stage 3: Localize the signal

A strong global readout is not enough.

Ask:

- where in the sequence does the target variable first become locally available?

Good localization strategies:

- span-local probes
- per-section pooling
- earlier vs later token comparisons

### Stage 4: Causal testing

Only after you have a good site hypothesis should you run patching or interchange.

Best practice:

- matched donor-target pairs
- single-layer tests first
- same-label controls
- explicit success criteria

### Stage 5: Mechanism follow-up

If the causal result is strong enough, move to mechanism-focused analysis.

Examples:

- attention to relevant spans
- per-head analysis
- narrower span decomposition
- write vs read layer comparisons

### Stage 6: Claim shaping

At the end, separate what each method supports.

Common structure:

- behavior says the task is real
- probes say the variable is represented
- localization says where it emerges
- patching says whether it is causally load-bearing
- attention says something about routing or mechanism

## Read vs write principle

One of the most important recurring patterns is:

- best read layer != best write layer

Interpretation:

- earlier layers may be where the computation is still malleable
- later layers may be where the result is most linearly separable

## Recommended evidence ladder

### Level 1: Behavioral only

- model behavior on the benchmark is sane

### Level 2: Representational

- a probe reads the target variable

### Level 3: Localized representational

- the signal is localized to a specific span, layer, or position

### Level 4: Causal

- intervention changes behavior beyond controls

### Level 5: Mechanistic

- a plausible computation path is identified

## Common failure patterns

- skipping behavioral audit
- overclaiming from last-token probes
- weak causal methods
- attention too early
- no controls

## Default stance

Mechanistic benchmark work should feel like:

`building a chain of evidence, where each stage earns the next`

not:

`running a pile of analyses and hoping they align.`
