---
benchmark: morebench
phase: 02
version: v1
frozen_date: 2026-04-22
input_artifacts:
  - projects/MECH_INTERP/morebench/phase_02/outputs/behavioral_smoke_results.json
  - docs/mech-interp/benchmarks/morebench/02-generation-protocol.md
---

# MoReBench 02 Behavioral Smoke Report

## Setup

- provisional smoke model: `/models/Qwen/Qwen3-30B-A3B`
- protocol: natural freeform answer with post hoc grading for recommendation presence and basic usability
- sampled prompts: `20`
- family distribution: `{'action_locus_rewrite': 5, 'neutral_control': 5, 'theory_direct': 5, 'theory_wording_variant': 5}`

## Summary

- nonempty response rate: `1.0`
- recommendation-present rate: `1.0`
- manual pass rate: `1.0`
- overall decision: `pass`

## Sample Notes

- `smoke_001` [theory_direct] nonempty=`True` manual_pass=`True` note: usable freeform response
- `smoke_002` [theory_direct] nonempty=`True` manual_pass=`True` note: usable freeform response
- `smoke_003` [theory_direct] nonempty=`True` manual_pass=`True` note: usable freeform response
- `smoke_004` [theory_direct] nonempty=`True` manual_pass=`True` note: usable freeform response
- `smoke_005` [theory_direct] nonempty=`True` manual_pass=`True` note: usable freeform response
- `smoke_006` [theory_wording_variant] nonempty=`True` manual_pass=`True` note: usable freeform response
- `smoke_007` [theory_wording_variant] nonempty=`True` manual_pass=`True` note: usable freeform response
- `smoke_008` [theory_wording_variant] nonempty=`True` manual_pass=`True` note: usable freeform response

## Interpretation

The augmented prompt slice cleared the provisional smoke gate, but it should still be rerun once the final target model is frozen.
