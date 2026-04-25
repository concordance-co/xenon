---
name: benchmark-mech-interp
description: Use when evaluating a benchmark as a substrate for mechanistic interpretability discovery, building a benchmark-specific mech interp spec, or mapping benchmark labels to candidate latent features, methods, confound checks, and first experiments.
---

> ARCHIVAL NOTE
>
> This skill has been superseded by the phase-based benchmark-flow skills:
> - [benchmark-validation](../benchmark-validation/SKILL.md)
> - [benchmark-to-latent-labels](../benchmark-to-latent-labels/SKILL.md)
> - [latent-label-data-augmentation](../latent-label-data-augmentation/SKILL.md)
> - [benchmark-mech-interp-analysis](../benchmark-mech-interp-analysis/SKILL.md)
> - [mechanistic-interventions](../mechanistic-interventions/SKILL.md)
>
> Keep this file as an archival/reference artifact while the new flow settles.
> Benchmark-specific worked examples should move into benchmark sidecars such as:
> - [benchmark_context.md](../../../projects/MOREBENCH/benchmark_context.md)

# Benchmark Mech Interp

Use this skill when the task is not “review a mech interp paper,” but rather:

- evaluate whether a benchmark is good for benchmark-first mech interp discovery
- turn a benchmark into a mech interp spec
- map benchmark labels to candidate latent features
- choose methodologies per feature rather than per benchmark
- design a repeatable benchmark-first discovery process

This skill is for `benchmark -> label inventory -> feature hypotheses -> methodologies -> artifacts`.

## Core idea

Do not collapse a benchmark into one project or one method.

The unit of analysis is:

1. `benchmark`
2. `label name`
3. `label type`
4. `potential feature`
5. `best-fit methodologies`
6. `expected artifacts`

One benchmark may expose many labels.
One label may support several feature hypotheses.
One feature hypothesis may support several methods.

## First read

Read these docs before drafting a benchmark spec:

- [benchmark-to-mech-interp.md](../../../methodology/archive/benchmark-to-mech-interp.md)
- [methodology/ROSTER.md](../../../methodology/ROSTER.md)
- [mech-interp-replication-priorities-2026-04-20.md](../../../methodology/archive/mech-interp-replication-priorities-2026-04-20.md)
- [constructing-llm-probes](../constructing-llm-probes/SKILL.md)
- [synthetic-data-generation](../synthetic-data-generation/SKILL.md)
- [benchmark-mech-interp-analysis](../benchmark-mech-interp-analysis/SKILL.md)
- [activation-patching-causal-evals](../activation-patching-causal-evals/SKILL.md)

Use the replication document mainly as a `methodology library`, not as a strict project queue.
Use the Xenon skills above as the default source for confound planning, split design, and causal-eval hygiene.

## Benchmark-first workflow

### 1. Benchmark intake

For the benchmark, identify:

- scale
- public availability
- code / harness availability
- single-turn vs multi-turn structure
- label families as a rough taxonomy
- whether labels are human, expert, rubric-derived, synthetic, or verifier-generated
- obvious product relevance

Do not stop at a benchmark summary. Explicitly inventory the labels.

### 2. Label inventory

Do not stop at broad families. Inventory concrete labels.

Use broad families only as orientation, for example:

- rubric dimensions
- theory labels
- span / claim labels
- belief-vs-statement contrasts
- hierarchy / privilege labels
- failure-mode categories
- stakeholder labels
- role labels
- trajectory or turn-level state labels

But the actual planning unit should be benchmark-specific:

- `Label Name`
- `Label Type`
- `Granularity`
- `How It Is Assigned`
- `Potential Feature`

Examples:

- `Helpful Outcome` -> `rubric dimension`
- `Harmless Outcome` -> `rubric dimension`
- `belief elicitation` -> `belief-report contrast label`
- `security breached` -> `binary outcome`
- `instruction winner` -> `hierarchy-resolution label`
- `hallucination span` -> `span label`

For each concrete label, note:

- what exactly is labeled
- whether it is binary, multiclass, scalar, weighted, or signed
- whether it is per-example, per-span, per-turn, or per-trajectory
- whether matched or contrastive pairs exist
- whether the label is likely to be noisy, composite, or shortcut-prone

### 3. Candidate feature inventory

For each concrete label, propose one or more `feature hypotheses`.

Examples:

- `MoReBench`:
  `Helpful Outcome` -> helpfulness representation
  `Harmless Outcome` -> harm-aversion representation
  `Helpful Outcome x Harmless Outcome` -> helpful-vs-harmless tradeoff
  `ROLE_DOMAIN` -> advisor-vs-agent stance
  theory labels -> framework adherence
- `CounselBench`:
  empathy, reassurance posture, boundary adherence, personalization, crisis sensitivity
- `AgentDojo`:
  `security breached` -> security-state failure
  `utility preserved` -> task-success state
  attack goal tags -> attack-type sensitivity
  source structure -> instruction-source dominance

Do not assume only one feature per benchmark, and do not assume only one feature per label.

Useful structure:

- `Label Name`
- `Potential Feature`
- `Why This Feature Might Exist`
- `Why This Label Supports It`

### 4. Method selection per feature

Choose methods per `feature hypothesis`, not per benchmark.

Use [methodology/ROSTER.md](../../../methodology/ROSTER.md) to select from:

- linear residual probes
- difference-in-means directions
- PCA / geometry recovery
- layer / token sweeps
- claim decomposition
- retrieval-oracle supervision
- logit lens
- head or MLP ablation
- attention blocking
- activation steering
- transfer matrices
- Mahalanobis-cosine portability
- LEACE / INLP erasure
- path patching
- activation scaling
- long-context robustness tests
- cascades

For each feature hypothesis, specify:

- `first-pass readout`
- `stronger follow-up`
- `strongest causal test we can plausibly run`

### 5. Confound audit

Before trusting a feature, plan the confound checks.

Common confounds:

- lexical shortcuts
- prompt-format artifacts
- label leakage from explicit markers
- domain/topic leakage
- length differences
- answer-style differences
- role-token or template-token artifacts
- model-family contamination

Typical controls:

- lexical splits
- domain splits
- prompt-template splits
- same-label controls
- label-balanced train/test splits
- matched-pair comparisons
- shuffled-label sanity checks
- neutral or confound-projected baselines

The Xenon synthetic-data and probe-construction skills contain especially useful guidance here. Reuse that guidance rather than re-deriving it each time.

Always ask:

- could a cheap bag-of-words baseline do this?
- could the probe be reading style instead of state?
- is the signal robust across splits that remove the obvious shortcut?

### 6. Behavioral sanity

Check whether the label itself is behaviorally coherent before doing heavy mech interp.

Examples:

- do high-score examples actually look like the intended behavior?
- do different annotators/frameworks mean the same thing?
- are there subclasses with very different semantics?
- is the benchmark mixing competence failure with the target variable?

If the labels are messy, note that early and narrow the feature scope.

### 7. Artifact planning

For each feature hypothesis, identify the expected artifact:

- probe
- concept direction
- transfer matrix
- per-layer localization map
- patching harness
- steering vector
- monitor component
- benchmark extension recipe

The desired outcome is not just “a result”; it is a reusable artifact.

## Output template

When creating a benchmark-specific spec, use this structure:

### Benchmark

- name
- links
- short purpose

### Label Inventory

- label name
- label type
- exact label structure
- scale and granularity
- how it is assigned

### Feature Hypotheses

- label name
- potential feature
- why it is interesting
- why the label supports it

### Methodology Options

- label name / feature hypothesis
- best first-pass methods
- stronger follow-up methods
- strongest feasible causal test

### Confound Plan

- likely shortcuts
- split strategy
- sanity checks

### First Experiments

- 2-5 concrete experiments in order

### Expected Artifacts

- what reusable outputs should exist if the work succeeds

### Risks

- label ambiguity
- confounds
- transfer concerns
- artifact availability issues

## Default sequencing

Unless there is a strong reason not to, start in this order:

1. label inventory
2. feature inventory
3. confound audit
4. simple readout
5. transfer / geometry
6. intervention
7. deployment hardening

This reduces the chance of overclaiming from a single early probe result.

## Internal process reminders from Xenon skills

- `Behavioral sanity comes first`
  If the benchmark task is malformed, ambiguous, or shortcut-ridden, later mech interp work will be misleading.
- `Protect the real decision bottleneck`
  Especially for synthetic or semi-synthetic tasks, simplify toward the core computation, not toward convenience.
- `Best read layer != best write layer`
  Treat readout and intervention site selection as separate problems.
- `Do not trust last-token readouts by default`
  Prefer span-local or section-aware localization when the benchmark is relational or multi-part.
- `Causal evals need proper controls`
  Same-label controls, matched donor-target pairs, and explicit malformed-output tracking should be standard.

## Worked-example recommendation

If a user wants to develop the process itself, prefer `MoReBench` first because it has:

- rich rubric structure
- theory labels
- several distinct feature families
- strong fit for multiple methodology families
- enough complexity to exercise the workflow without requiring the messiest agent setup

Use `CounselBench` or `HealthBench` next if the goal is to pressure-test the workflow on more product-like advisory data.
