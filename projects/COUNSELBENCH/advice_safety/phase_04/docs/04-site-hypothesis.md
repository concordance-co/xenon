---
benchmark: counselbench
phase: 04
version: v1
frozen_date: 2026-04-24
input_artifacts:
  - projects/COUNSELBENCH/advice_safety/phase_03b/specs/workflow.py
  - projects/COUNSELBENCH/advice_safety/phase_03_eval/specs/workflow.py
---

# Phase 04 Site Hypothesis

Current layer `44` evidence is treated as a read-layer result only. It is a late summary state and should not be selected as the first write site merely because it was easiest to decode.

Candidate sites must come from Phase 03b and Eval localization:

- prompt-end and risk-span sites for prompt-family / risk-recognition hypotheses
- response-end and boundary-advice span sites for response-quality hypotheses
- earlier or middle layers if readout emerges before the late summary layer

Phase 04 starts with one layer at a time. Multi-layer patching is deferred because it weakens localization.

