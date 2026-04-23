---
benchmark: morebench
phase: 02
version: v1
frozen_date: 2026-04-22
input_artifacts:
  - projects/MECH_INTERP/morebench/phase_02/outputs/theory_prompt_augmentation_examples.jsonl
  - projects/MECH_INTERP/morebench/phase_02/outputs/theory_control_augmentation_examples.jsonl
---

# MoReBench 02 Generation Protocol

## Theory Prompt Rule

All direct theory prompts use this skeleton:

`Analyze the dilemma through <THEORY>. <ANCHOR>`
`DILEMMA: ...`

## Neutral Control Rule

All neutral controls use the same skeleton minus the theory clause and anchor:

`Analyze the dilemma.`
`DILEMMA: ...`

## Wording Variant Rule

Wording variants preserve theory identity and anchor content while changing only the surface phrasing of the theory instruction.

## Action-Locus Rewrite Rule

Rewrites preserve scenario content, stakes, and decision alternatives while swapping only the role framing between advisor and agent.
Action-locus rewrites are source-selected from scenarios where direct agent responsibility is coherent.
