# Phase 01: Self-Interest Conflict Seed Set

## Goal

Create a candidate dataset for an `ethical` versus `self_advantage` activation pole.
This is a new substrate, not a continuation of the named-theory direction work.

## Hypothesis

If moral orientation has a clean polar activation signature in this setting, it is
more likely to appear when the prompt asks the model to choose between:

- acting ethically while accepting personal cost, and
- acting to preserve or improve the actor's own welfare at others' expense.

## Current Artifacts

- `outputs/ethical_advantage_seed_dilemmas.jsonl`: 40 candidate dilemmas.
- `outputs/ethical_advantage_seed_summary.json`: domain and conflict-axis counts.
- `specs/ethical_advantage_prompt_conditions.json`: first-pass prompt conditions.
- `docs/01-seed-dataset-plan.md`: validation and capture plan.

## Gate Before Capture

Do not run activation capture until a behavior smoke confirms that the two poles
actually produce different action recommendations on a filtered subset.

