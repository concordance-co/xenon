---
benchmark: counselbench
phase: 03
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/COUNSELBENCH/phase_02/docs/02-generation-protocol.md
---

# CounselBench Execution Targets

- Primary model: `/models/Qwen/Qwen3-30B-A3B`
- Runtime: vLLM on Modal H200 GPU runner
- Dataset: full CounselBench-Adv, 120 prompts
- Generation cap: `15000` tokens
- Model context window: `30000` tokens
- Generation batch size: `16`
- Model-bound Modal runner: four shards / four max containers for generation and residual capture
- Capture layers: `0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44`
- Capture sites: residual post-activation at prompt end and generated-answer end
- Analysis runner: Modal CPU
- Report runner: local
