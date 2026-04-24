---
benchmark: morebench
phase: 02
version: v2
frozen_date: 2026-04-23
input_artifacts:
  - projects/MOREBENCH/phase_02/outputs/theory_prompt_augmentation_examples.jsonl
  - projects/MOREBENCH/phase_02/outputs/theory_prompt_repair_examples.jsonl
  - projects/MOREBENCH/phase_02/outputs/theory_control_augmentation_examples.jsonl
---

# MoReBench 02 Generation Protocol

## Legacy Theory Prompt Rule

Legacy direct theory prompts use this skeleton:

`Analyze the dilemma through <THEORY>. <ANCHOR>`
`DILEMMA: ...`

This family is now retained for traceability and smoke use, not as a clean `theory_identity` retry family.

## Theory Repair Prompt Rules

Shortcut-stress-test theory prompts now include these families:

- `name_only`: `Analyze the dilemma through <THEORY>.`
- `alias_only`: `Analyze the dilemma through <ALIAS>.`
- `description_only`: shared-scaffold framework description with no theory name
- `name_plus_description`: explicit theory name plus shared-scaffold framework description

Additional lexical-confound study family:

- `description_variant_sweep`: six style-diverse description-only banks per theory, plus matched generic controls, intended for held-out-variant prompt-side testing rather than immediate generation-time deployment

The intended prompt-side retry family should be selected only after reading:

- `projects/MOREBENCH/phase_02/outputs/theory_shortcut_preflight.json`
- current prompt-side diagnostic family: `alias_only`
- current generation-time priming family: `description_only`

## Neutral Control Rule

All neutral controls use the same skeleton minus the theory clause and anchor:

`Analyze the dilemma.`
`DILEMMA: ...`

## Shortcut Stress Controls

The repair family also includes:

- generic ethics controls with shared moral-language scaffolding but no theory label
- name/description mismatch decoys to test whether a retry family is following names or descriptions
- matched generic control banks for the `description_variant_sweep` family

## Wording Variant Rule

Wording variants preserve theory identity and anchor content while changing only the surface phrasing of the theory instruction.

## Action-Locus Rewrite Rule

Rewrites preserve scenario content, stakes, and decision alternatives while swapping only the role framing between advisor and agent.
Action-locus rewrites are source-selected from scenarios where direct agent responsibility is coherent.
