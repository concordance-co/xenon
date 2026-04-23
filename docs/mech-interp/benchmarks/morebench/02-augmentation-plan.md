---
benchmark: morebench
phase: 02
version: v1
frozen_date: 2026-04-22
input_artifacts:
  - docs/mech-interp/benchmarks/morebench/01-gap-list.md
  - projects/MECH_INTERP/morebench/phase_02/outputs/theory_group_manifest.json
---

# MoReBench 02 Augmentation Plan

## Goal

Repair the benchmark so the phase 01 latent labels become scientifically usable.

## Primary Repair Tracks

- `theory_identity`: The existing theory split already supplies clean five-way matched dilemma sets. The repair move is to make theory explicit in prompt text while preserving the underlying dilemma content.
- `action_locus`: The public split has zero source-controlled mixed-role cells, so the only credible repair is matched advisor/agent rewriting within coherent shared scenario templates.
- response-side labels: keep prompt families clean enough that fresh generations are worth collecting

## Confound-Focused Repair Moves

- `theory_not_prompt_exposed`: Inject explicit theory instructions with framework-specific anchors into matched dilemma groups. (`materialized`)
- `source_role_aliasing`: Materialize starter batch of 10 matched advisor/agent rewrite pairs. (`partially_materialized`)
- `prompt_wrapper_imbalance`: Use a structurally matched neutral control with the same prompt skeleton minus the theory clause. (`materialized`)
- `source_type_aliasing`: Rewrite prompts into matched canonical formats across long-case and expert-case structure. (`not_started`)
- `length_variation`: Build short/medium/long renderings for the same scenario content. (`not_started`)
- `person_grammar_variation`: Create first/second/third-person rewrites that preserve stakes and alternatives. (`not_started`)
- `context_missingness_and_topic_imbalance`: Complete missing context metadata and build balanced evaluation slices. (`not_started`)

## Principle

Augment to repair the experiment, not to make the dataset bigger.
