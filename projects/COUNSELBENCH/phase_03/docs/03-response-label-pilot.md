---
benchmark: counselbench
phase: 03
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/COUNSELBENCH/phase_02/docs/02-behavioral-smoke-report.md
---

# CounselBench Response Label Pilot

Generated Adv responses need a label pilot before response-side probes can support strong claims.

Pilot protocol:

- sample responses from every Adv failure mode
- manually inspect unsafe advice, boundary adherence, empathy, and specificity
- compare manual labels against lexical and LLM labelers if used
- freeze accepted labels with version and disagreement metadata
- only then run response-side probes as more than smoke scaffolding

