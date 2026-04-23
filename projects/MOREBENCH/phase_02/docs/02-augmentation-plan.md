---
benchmark: morebench
phase: 02
version: v2
frozen_date: 2026-04-23
input_artifacts:
  - projects/MOREBENCH/phase_01/docs/01-gap-list.md
  - projects/MOREBENCH/phase_02/outputs/theory_group_manifest.json
---

# MoReBench 02 Augmentation Plan

## Goal

Repair the benchmark so the phase 01 latent labels become scientifically usable.

## Primary Repair Tracks

- `theory_identity`: The existing theory split already supplies clean five-way matched dilemma sets, but the first explicit theory family proved shortcut-dominated. The repair move is now to materialize harder prompt-side families that decouple theory identity from a single name token or a fixed anchor sentence.
- `action_locus`: The public split has zero source-controlled mixed-role cells, so the only credible repair is matched advisor/agent rewriting within coherent shared scenario templates.
- response-side labels: keep prompt families clean enough that fresh generations are worth collecting

## Repair Loop Note

The first explicit-theory prompt family is now treated as known shortcut-dominated for `theory_identity` after phase-03 Experiment 1.
This phase therefore reopens theory work as an anti-shortcut repair problem rather than treating the earlier family as phase-03-ready.

## Confound-Focused Repair Moves

- `theory_not_prompt_exposed`: Inject explicit theory instructions with framework-specific anchors into matched dilemma groups. (`materialized`)
- `theory_lexical_shortcuts`: Materialize name-only, alias-only, description-only, and name-plus-description theory families plus cheap-baseline preflight. (`materialized`)
- `source_role_aliasing`: Materialize starter batch of 10 matched advisor/agent rewrite pairs. (`partially_materialized`)
- `prompt_wrapper_imbalance`: Use a structurally matched neutral control with the same prompt skeleton minus the theory clause. (`materialized`)
- `source_type_aliasing`: Rewrite prompts into matched canonical formats across long-case and expert-case structure. (`not_started`)
- `length_variation`: Build short/medium/long renderings for the same scenario content. (`not_started`)
- `person_grammar_variation`: Create first/second/third-person rewrites that preserve stakes and alternatives. (`not_started`)
- `context_missingness_and_topic_imbalance`: Complete missing context metadata and build balanced evaluation slices. (`not_started`)

## Principle

Augment to repair the experiment, not to make the dataset bigger.
