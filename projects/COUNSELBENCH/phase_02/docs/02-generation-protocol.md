---
benchmark: counselbench
phase: 02
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/COUNSELBENCH/phase_01/docs/01-latent-label-spec.md
---

# CounselBench Generation Protocol

- Model: `/models/Qwen/Qwen3-30B-A3B`
- Dataset: balanced `CounselBench-Adv` slice, default `4` prompts per failure mode for first pass
- Prompt format: raw CounselBench question as a single `user` chat message; no project-local system prompt or hand-written `User:` / `Assistant:` completion wrapper
- Decoding: `temperature=0.0`, `top_p=1.0`, `top_k=-1`
- Max tokens: `15000`
- Model context window: `30000`
- Claims: generation outputs are for behavioral smoke and provisional readouts only until response-side labels are validated
