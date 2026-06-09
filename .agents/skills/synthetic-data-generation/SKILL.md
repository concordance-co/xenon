---
name: synthetic-data-generation
description: Use when designing controlled synthetic benchmarks, prompt variants, or data generators where behavioral sanity and experimental validity matter. Covers latent-variable design, lexical controls, split schemes, matched pairs, prompt role placement, agentic task realism, and smoke checks before probing or patching.
---

# Synthetic Data Generation

Use this skill when the user wants to:

- design a new controlled benchmark
- create prompt variants for probing, patching, or behavior checks
- build a synthetic dataset generator
- repair a synthetic benchmark that is behaving strangely
- add lexical, carrier, domain, or split controls to synthetic data

This skill owns **synthetic experiment design craft**. It does not own phase-02
benchmark repair for imported benchmarks. If the task starts from a
benchmark-specific gap list, use
[latent-label-data-augmentation](../latent-label-data-augmentation/SKILL.md)
as the phase owner and load this skill only for craft patterns.

## Core rule

`Behavioral sanity comes before interpretability.`

Before treating probe results, patching results, or attention results as
meaningful, first verify that the benchmark itself is behaviorally sane.

If the base task is malformed, ambiguous, or solved via role-format artifacts,
later interpretability work will be misleading.

## Start here

1. Name the latent variable before writing rows.
2. Define the minimal computation the model must perform.
3. Decide which nuisance dimensions must stay fixed, vary, or be held out.
4. Write a small hand-audited example bank before scaling generation.
5. Run cheap behavioral and lexical smoke checks before moving to probes.

When the task is specifically about agent workflows, role placement, tool
evidence, or instruction authority, also read
[references/AGENTIC_TASKS.md](references/AGENTIC_TASKS.md).

For anti-shortcut repair moves, split schemes, and generator contracts, read
[references/BENCHMARK_CRAFT.md](references/BENCHMARK_CRAFT.md).

## Evidence discipline

Every plan, manifest, or report produced under this skill must include:

- `evidence_rung`
- `claim_boundary`
- `known_shortcuts`
- `smoke_status`

Use `evidence_rung: design_only` for proposed datasets that have not yet passed
behavioral smoke checks. Promote to `behavioral` only after the target model
solves the task for the intended reasons on an inspected sample. Do not claim
`representational`, `localized_representational`, `causal`, or `mechanistic`
from synthetic data design alone; those rungs require downstream analysis per
[methodology/PRINCIPLES.md](../../../methodology/PRINCIPLES.md).

## Required outputs

For a synthetic benchmark or generator, leave behind:

- a design note naming the latent variable, positive/negative classes, and
  nuisance controls
- a row schema or generator contract
- a hand-audited smoke slice, usually 10-50 rows
- split policy, including any lexical, domain, action, or carrier holdouts
- cheap baseline results or a clear reason they are not yet run
- known residual shortcut channels

If materialized data is produced, the manifest should include:

- `row_count`
- `split_counts`
- `generator_version`
- `seed`
- `rows_with_all_placeholders_substituted`
- `evidence_rung`
- `known_bugs`

## Gotchas

- Do not smooth a real workflow until the original decision bottleneck
  disappears.
- Do not let the generator improvise inventories that the design note claimed
  were frozen.
- Do not use role placement, wrapper text, or output format as an accidental
  label carrier.
- Do not call a variant pair a holdout until cheap text baselines fail on the
  train-to-heldout split used for the claim.
- Do not move to probes when the base model is failing because the prompt is
  ambiguous, unnatural, or underspecified.

## Handoff

- Use [latent-label-data-augmentation](../latent-label-data-augmentation/SKILL.md)
  when the work is benchmark phase-02 repair from a gap list.
- Use [constructing-workflows](../constructing-workflows/SKILL.md) when the
  synthetic design is ready to become a `pipelines_v2` workflow.
- Use [constructing-llm-probes](../constructing-llm-probes/SKILL.md) only after
  behavioral sanity and cheap-baseline checks are recorded.
