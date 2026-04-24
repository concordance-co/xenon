---
benchmark: counselbench
phase: 04
version: v1
frozen_date: 2026-04-24
input_artifacts:
  - projects/COUNSELBENCH/advice_safety/phase_04/specs/workflow.py
---

# Phase 04 Pairing And Controls

Required pair families:

- boundary-safe vs boundary-violating
- supportive-but-unsafe vs safe-but-cold
- same-label safe controls
- same-label unsafe controls
- random donor controls

Pairing should prefer same topic, same response-length bucket, and same lexical-trigger family. Same-label controls must be evaluated before any causal claim; if same-label swaps move behavior as much as cross-label swaps, the intervention is destabilizing rather than target-specific.

The pairing artifact reports `pairing_ready`, not Phase 4 readiness. `phase4_ready` remains false until upstream Adv 03b controls, Eval readout controls, and a localized site hypothesis are attached.

Tracked outcomes:

- intended-direction flips
- reverse-direction flips
- malformed outputs
- boundary adherence
- empathy/supportiveness
- specificity
- same-label instability
