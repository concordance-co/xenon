---
benchmark: morebench
phase: 03
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/MOREBENCH/phase_01/docs/01-latent-label-spec.md
  - projects/MOREBENCH/phase_01/docs/01-frozen-label-set.csv
---

# MoReBench 03 Stakeholder-Tradeoff-Density Gold Slice Spec

## Purpose

`stakeholder_tradeoff_density` stays gated until a concrete validated prompt-side gold slice exists.

## Proposed Gold Slice

- target size:
  `80` prompts
- balancing goal:
  cover multiple source families and dilemma structures rather than sampling one source heavily
- unit:
  prompt-side dilemma text only

## Labeling Procedure

- annotate the number of distinct stakeholder or consequence clusters that are explicitly live in the prompt
- use a short written counting policy so the label does not collapse into prompt length
- include adjudication notes for borderline cases where stakeholders overlap or are only implicit

## Validation Standard

- at least two independent label passes on a shared subset
- disagreements reviewed with written rationale
- the gold slice is not cleared for probing until the counting policy is stable enough to support prompt-side supervision

## Expected Output

- `03-stakeholder-tradeoff-density-gold-slice.csv`
- `03-stakeholder-tradeoff-density-gold-slice-notes.md`
- optional summary:
  `03-stakeholder-tradeoff-density-gold-slice-summary.json`
