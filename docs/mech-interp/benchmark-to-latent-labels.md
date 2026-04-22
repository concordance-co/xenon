# Benchmark-To-Latent-Labels

**Date:** 2026-04-22

## Purpose

This document captures a general process for converting `benchmark labels` into `mechanistically useful target labels`.

The core problem is:

- most benchmark labels were designed for `grading`, `ranking`, or `human evaluation`
- mechanistic interpretability needs labels that are closer to:
  - prompt structure
  - internal state
  - deliberative process
  - control policy
  - intervention target

So the job is not just to "use the labels."
The job is to translate from `grader-designed supervision` to `latent-oriented supervision`.

This should be treated as a reusable methodology for the broader `benchmark -> mech interp` program, not just a note about one dataset.

Related documents:
- [benchmark-to-mech-interp.md](/Users/trentelmore/Projects/concordance/knowledge-base/benchmark-to-mech-interp.md)
- [benchmark-first-mech-interp-failure-modes.md](/Users/trentelmore/Projects/concordance/knowledge-base/benchmark-first-mech-interp-failure-modes.md)
- [mech-interp-methodology-roster.md](/Users/trentelmore/Projects/concordance/knowledge-base/mech-interp-methodology-roster.md)

## Core Principle

Do not assume a benchmark's native labels are already mech-interp targets.

Many benchmarks are:

- `grader-designed`
- `rubric-designed`
- `preference-designed`
- `task-outcome-designed`

rather than:

- `latent-designed`
- `control-state-designed`
- `causal-intervention-designed`

This means the first pass should usually ask:

1. what was this label originally designed to measure?
2. what kind of object is it really?
3. what kind of mechanistic question could it support?
4. what relabeling or decomposition is needed before probing?

## Target Output

The output of this process is not just a cleaned spreadsheet.

The target artifact is a `benchmark-specific latent label spec` with:

- candidate mechanistic questions
- prompt-side labels
- response-side labels
- nuisance variables and confounds
- labels that can be used directly
- labels that must be derived
- labels that require new generations or synthetic augmentation
- a recommendation for what to probe first

## Recommended Workflow

### 0. Benchmark Intake

Before touching labels, establish the benchmark frame.

Record:

- dataset size
- public availability
- splits and configs
- prompt structure
- response structure
- whether labels apply to prompt, response, trajectory, or evaluator judgment
- whether labels are human-authored, expert-authored, verifier-generated, or synthetic
- whether a runnable harness exists
- whether activations can be captured at prompt time, generation time, or both

Why this matters:

- some benchmarks mainly support `prompt-side representation` questions
- some mainly support `response-side policy` questions
- some are only useful after generating fresh model outputs

### 1. Inventory Existing Labels

Review all native labels and metadata fields.

For each label, record:

- `Label Name`
- `Where It Lives`
  prompt field, response field, rubric, metadata, trajectory, etc.
- `What It Was Designed To Do`
  grading, taxonomy, filtering, annotation, ranking, etc.
- `Assignment Method`
  human, expert, synthetic, heuristic, verifier, model-generated
- `Granularity`
  benchmark-level, example-level, turn-level, span-level, criterion-level
- `Value Type`
  binary, multiclass, scalar, signed, weighted, free text

Important question:

- is this label naming a stable property, or is it really an evaluator instruction?

This is the first place many candidate labels should be downgraded.

### 2. Classify The Label Type

Do not treat all labels as the same kind of thing.

Useful buckets:

- `prompt-side structure`
  properties of the scenario or task before generation
- `response-side process`
  properties of the model's reasoning or deliberation
- `response-side objective orientation`
  what the model appears to be optimizing for
- `outcome / rubric score`
  evaluator judgments about response quality
- `metadata / nuisance variable`
  source, template, length, domain, formatting, annotator family

This step is especially important because confusion often comes from mixing:

- what the prompt is
- what the model does
- how the grader scores it

Those are different targets.

### 3. Assess Match To Potential Latent Labels

For each native label, ask how well it matches a plausible mechanistic target.

Use rough ratings like:

- `direct`
  already close to a plausible latent or control state
- `derived`
  could become useful after decomposition or relabeling
- `validation-only`
  useful as an eval signal but not as a probe target
- `nuisance-only`
  should be tracked and controlled, not targeted
- `not useful`
  too noisy, composite, case-specific, or confounded

Questions to ask:

- is the label too broad?
- is it obviously composite?
- is it mostly a grading hook?
- does it vary at the right level of abstraction?
- is it likely to transfer across examples?
- could a probe on this label be interpretable if it worked?

### 4. Step Back And Define The Mech Interp Questions

Do not let the benchmark ontology define the whole project.

Before refining labels, articulate the actual mechanistic questions we care about.

Typical forms:

- does the model represent `X` before answering?
- does the model switch between policy modes `A` and `B`?
- are `X` and `Y` separable internal objectives?
- does prompt framing alter early representation, late recommendation, or both?
- is the signal a readout only, or a plausible intervention target?

This is the step where we move from:

- `what labels exist?`

to:

- `what internal distinctions would be interesting if they existed?`

Without this step, the work collapses into relabeling benchmark artifacts.

### 5. Re-Review Labels Against Those Questions

Now revisit the native labels with the mechanistic questions in view.

Pick the labels that are:

- scientifically interesting
- operationally tractable
- most aligned with plausible latent variables
- available or derivable from the benchmark

This is the narrowing step.

The goal is not to preserve the benchmark's full ontology.
The goal is to identify the subset that can support real mech-interp work.

### 6. Separate Direct, Derived, And New Labels

For each promising target, decide whether it is:

- `directly usable`
  available as-is from the dataset
- `derivable from existing fields`
  can be labeled from prompt/response text or metadata
- `derivable from new generations`
  requires generating model responses and labeling those
- `requires new data`
  needs paired examples, synthetic data, rewrites, or targeted collection

This distinction matters because prompt-side and response-side labels often have different data requirements.

Typical pattern:

- prompt-side structure labels can often be derived from `DILEMMA`-like fields
- response-side process labels usually require `fresh generations`
- rubric scores often belong in `validation`, not training targets

### 7. Validate Derivability

Before treating a refined label as real, confirm it can actually be derived with acceptable quality.

Validation options:

- manual spot-checks
- small hand-labeled gold set
- inter-rater agreement
- LLM labeler vs human agreement
- edge-case review
- label-definition revision after disagreement analysis

Important question:

- can this label be assigned consistently from the available evidence?

If not, it is not ready for probe work.

### 8. Audit Confounds And Nuisance Variables

For every candidate label, identify likely shortcuts and confounds.

Always inspect:

- source/template family
- prompt length
- domain/topic
- role tokens
- grammatical person
- answer format expectations
- annotator/rubric family
- theory or ideology leakage
- competence vs target-variable confounds

Then decide which fields must always be:

- stratified on
- regressed out
- held out across splits
- kept only as nuisance variables

This step should happen before serious probing, not after a good-looking result.

### 9. Decide Where The Signal Should Live

For each mechanistic question, explicitly state where we expect the signal.

Examples:

- `prompt-side representation`
  early or mid-pass encoding of scenario structure
- `generation-time deliberation`
  evolving internal state during reasoning
- `commitment / action selection`
  transition from considering to recommending
- `final-token readout`
  compact summary state after the answer is formed

This step prevents a common mistake:

- creating labels that only make sense with generation-time activations
- then trying to study them with prompt-only snapshots

### 10. Choose The Minimal First-Pass Ontology

Do not launch with every plausible label.

Pick a small first-pass set that is:

- high-value
- likely derivable
- confound-aware
- broad enough to learn from

Good first-pass candidates often include:

- one or two prompt-side structure labels
- one or two response-side process labels
- one objective-orientation contrast
- a tracked nuisance-variable set

Everything else should be explicitly postponed rather than left ambiguous.

### 11. Define Validation And Success Modes

Before extraction, decide what success means for each label family.

Possible success modes:

- `indicator success`
  robust readout
- `mechanistic success`
  evidence of causal involvement
- `control success`
  steerable or patchable feature

Also define:

- expected baselines
- required controls
- minimum transfer bar
- what result would cause the label family to be dropped

### 12. Create A Follow-On Data Plan

For labels that are promising but not cleanly present in the benchmark, define what extra work is needed.

Common next steps:

- generate responses from target models
- build matched prompt rewrites
- create theory-paired variants
- synthesize clearer cases
- add same-label controls
- remove or rebalance confounds

This is where the benchmark becomes a seed set rather than a prison.

## Practical Checklist

When converting a benchmark to latent labels, check:

- have we separated prompt-side from response-side labels?
- have we separated target labels from nuisance variables?
- have we separated probe targets from validation signals?
- have we checked whether native labels are evaluator instructions rather than latent names?
- have we written down the actual mechanistic questions?
- have we confirmed where in the forward/generation process signal should appear?
- have we validated that our refined labels can be assigned consistently?
- have we identified which labels require fresh generations?
- have we decided what to control for in splits and evals?
- have we reduced the first pass to a manageable ontology?

## Common Failure Modes

### Treating Grading Labels As Latent Labels

The benchmark may have detailed rubrics, but those can still be:

- case-specific
- evaluator-authored
- response-scoring instructions
- too composite to probe directly

### Mixing Prompt, Response, And Grader Perspectives

Many ontology problems are really viewpoint problems.

A field can describe:

- the scenario
- the answer
- the scorer's standard

Mixing these leads to confusing labels and confusing probes.

### Keeping Too Much Philosophy And Too Little Operationalization

Some distinctions are conceptually interesting but not worth the labeling cost on a given dataset.

Useful question:

- can this distinction be labeled reliably enough, at reasonable cost, to support an actual experiment?

### Forgetting That Some Good Labels Require New Generations

Many of the best response-side labels are not present in the benchmark at all.

They only become available after:

- generating model outputs
- segmenting those outputs
- labeling the resulting reasoning traces

### Confusing Evaluation Signals With Training Targets

A rubric may still be extremely useful even if it should not be probed directly.

Typical use:

- refined label for training or probing
- rubric dimension for evaluation and validation

## Outputs We Should Expect At The End

Each benchmark-specific pass should end with:

1. `Benchmark framing`
2. `Inventory of native labels`
3. `Candidate mechanistic questions`
4. `Refined ontology`
5. `Prompt-side labels`
6. `Response-side labels`
7. `Nuisance variables`
8. `Validation signals`
9. `Confound analysis`
10. `Follow-on data plan`
11. `Recommended first experiments`

## What Came Out Of This Process So Far

The clearest general lessons from this pass are:

- benchmark richness does not imply latent-label readiness
- rubric density is useful, but often as `evaluation scaffolding`
- prompt-side structure labels are usually easier to derive cleanly than response-side reasoning labels
- response-side labels often need to be derived from fresh generations
- some native benchmark dimensions should be demoted from `target` to `validation`
- confounds like source/template/length should be identified before probing
- it is worth explicitly deciding where signal is expected to appear in the computation
- a good first-pass ontology is usually much smaller than the benchmark's full label space

## Suggested Reusable Template

When doing this for a new benchmark, start with:

1. review existing labels and metadata
2. classify each label by type and intended use
3. rate match to plausible latent labels
4. step back and define the most interesting mechanistic questions
5. re-review labels against those questions
6. separate direct labels, derived labels, validation signals, and nuisance variables
7. validate that refined labels can be assigned reliably
8. identify confounds and required controls
9. specify where signal should live in the model or trajectory
10. pick a minimal first-pass ontology
11. define success criteria and required controls
12. create a plan for new generations, synthetic augmentation, or confound removal

This is the process we should likely standardize into a skill.
