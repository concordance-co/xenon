# Mech Interp Principles

**Date:** 2026-04-22

## Purpose

This document holds the `cross-cutting principles` for benchmark-first mechanistic interpretability work.

These principles should not be duplicated across every phase skill or technique skill.
They are meant to be:

- small
- stable
- reusable
- load-bearing

Phase skills should point here for shared discipline.
Technique skills should point here when the principle cuts across methods.

## 1. Behavioral Sanity Comes First

`Do not jump to mechanism before establishing the task is real.`

Before any probing, localization, or intervention work:

- inspect real examples from each class or label family
- run the probe-target model on a small slice
- verify output parsing
- verify the benchmark is being solved at a reasonable rate
- inspect failures manually

If the task is not behaviorally sane, stop and repair it first.

Behavioral sanity is distinct from benchmark worthiness.

A benchmark can be:

- public
- rich
- important

and still be a bad substrate for a specific model if:

- the model cannot do the task
- outputs are malformed
- labels are behaviorally incoherent
- the task is solved by shortcuts

For response-side labels, the practical version of behavioral sanity is:

- inspect generated model outputs, not just activations

An activation-only smoke test is not enough to confirm task validity.

## 2. Evidence Must Climb A Ladder

Mechanistic benchmark work should feel like:

`building a chain of evidence, where each stage earns the next`

not:

`running a pile of analyses and hoping they align`

### Level 1: Behavioral

- the model behaves sanely on the benchmark

### Level 2: Representational

- a readout or probe detects the target variable

### Level 3: Localized Representational

- the signal is localized to a span, section, token, layer, or position

### Level 4: Causal

- an intervention changes behavior beyond controls

### Level 5: Mechanistic

- a plausible computation path is identified

Do not make Level 5 claims from Level 2 evidence.

Example:

- a high-AUROC probe alone can support a representational claim
- it cannot by itself support a causal or mechanistic claim

## 3. Read Layer Is Not Write Layer

`A readable signal is not automatically a writable signal.`

Common pattern:

- later layers give the strongest probe AUROC
- earlier layers give more causal leverage

Interpretation:

- late layers may contain compressed summary states
- early or middle layers may still contain malleable computation

Do not pick intervention sites only because they are easy to read.

## 4. Grader-Designed Is Not Latent-Designed

Many benchmarks are:

- rubric-designed
- grader-designed
- task-outcome-designed
- preference-designed

Mechanistic work needs labels closer to:

- prompt structure
- internal state
- deliberative process
- objective orientation
- intervention target

The job is often to translate from grading supervision to latent-oriented supervision.

## 5. Benchmark Existence Is Evidence

Start from the benchmark as a crystallized domain artifact.

Its design choices encode:

- what the authors thought mattered
- what distinctions they expected to be meaningful
- what kinds of behavior they believed were worth surfacing

This is evidence of value.

But it is not proof of mechanistic tractability.

The correct posture is:

- derive the implicit question from the benchmark
- then test whether the benchmark supports that question cleanly

## 6. Claim Strength Must Match Evidence Strength

At the end of an analysis, separate what each method supports.

Useful structure:

- behavior says the task is real
- probes say the variable is represented
- localization says where it emerges
- intervention says whether it is causally load-bearing
- attention or routing follow-up says something about mechanism

Always ask:

- what exactly has been shown?
- what has not yet been shown?
- what is still only an interpretation?

## 7. Controls Are Mandatory

Good-looking benchmark results are not enough.

Always ask:

- could a cheap surface baseline do this?
- could the model be using style instead of state?
- could source, length, or role tokens explain the effect?
- could prompt format or role-token placement alone recover the label?
- could a same-label control show the intervention is just destabilizing the model?

Controls should be planned before results, not added as a reaction to skepticism.

## 8. Nuisance-Stratified Cell Size Matters

Any benchmark plan should check post-stratification sample size.

Ask:

- how many examples remain per target label after stratifying on the nuisance variables we actually need to control?
- do the resulting cells still support the planned probe or intervention?

If the answer is no, the correct response is:

- narrow the question
- augment the data
- or stop

not:

- run the analysis anyway and hope regularization hides the problem

## 9. Benchmarks Are Seed Sets, Not Prisons

If a benchmark cannot cleanly support the intended question, the answer is not always "give up."

It may need:

- rewrites
- matched pairs
- counterbalancing
- response generation
- synthetic or semi-synthetic augmentation

But augmentation should repair the experiment, not just enlarge the dataset.
