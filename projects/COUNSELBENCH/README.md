# CounselBench Mechanistic Interpretability

This project turns CounselBench into a phase-based benchmark-first mech-interp program.

Canonical sources:

- [CounselBench project page](https://llm-eval-mental-health.github.io/counselbench-2025/)
- [CounselBench GitHub repo](https://github.com/llm-eval-mental-health/CounselBench)
- [CounselBench-Eval](https://huggingface.co/datasets/izi-ano/CounselBench-Eval)
- [CounselBench-Adv](https://huggingface.co/datasets/izi-ano/CounselBench-Adv)

Current executable workflows:

```bash
uv run python -m pipelines_v2.cli workflow plan --file projects/COUNSELBENCH/advice_safety/phase_03/specs/workflow.py
uv run python -m pipelines_v2.cli workflow run --file projects/COUNSELBENCH/advice_safety/phase_03/specs/workflow.py --logging INFO

uv run python -m pipelines_v2.cli workflow plan --file projects/COUNSELBENCH/advice_safety/phase_03b/specs/workflow.py
uv run python -m pipelines_v2.cli workflow plan --file projects/COUNSELBENCH/advice_safety/phase_03_eval/specs/workflow.py
uv run python -m pipelines_v2.cli workflow plan --file projects/COUNSELBENCH/advice_safety/phase_04/specs/workflow.py
```

The phase-03 workflow starts with the full CounselBench-Adv table: 20 source rows x six failure-mode prompt columns = 120 prompts. It runs deterministic generation with a 15000-token cap and 30000-token model window on a four-shard H200 Modal GPU runner with generation batch size 16, checks generated-response label support, captures prompt-end and generation-end residual states, compares prompt probes against text baselines, and emits PCA geometry views for prompt-risk and response-posture structure. Response-side classifier probes are gated until the generated-response labels have both classes in grouped train/test splits.

Follow-up workflow roles:

- `phase_03b`: Adv control/localization follow-up with grouped readouts, text/topic/length/lexical/source-row controls, nuisance probes, residualized probes, risk-span capture, quantitative geometry metrics, and control triage.
- `phase_03_eval`: CounselBench-Eval expert-label response-context readouts after question-response aggregation, with response-text and nuisance baselines plus boundary-vs-quality direction overlap checks.
- `phase_04`: intervention-readiness/pairing artifacts only; causal interchange remains blocked until 03b and Eval gates pass.

Prior smoke: `wr_299cf3c365e4_a3d3ea13` completed on 2026-04-23. Treat those results as diagnostic only: 17 of 24 generations hit the old 800-token length cap, and the replayable generated-boundary label was one-class. The current phase-03 workflow supersedes it.
