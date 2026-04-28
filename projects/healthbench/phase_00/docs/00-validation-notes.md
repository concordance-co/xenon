---
benchmark: healthbench
phase: 00
version: v1
frozen_date: 2026-04-28
input_artifacts:
  - projects/healthbench/phase_00/docs/00-validation-memo.md
---

# HealthBench Validation Notes

## Source Notes

- `healthbench_eval.py` defines three source blobs: full HealthBench,
  HealthBench Hard, and HealthBench Consensus.
- The eval constructs a candidate conversation by appending the model response
  to the prompt and grading each rubric item independently.
- The grader returns `criteria_met`; for negative point criteria, this means the
  undesirable criterion was met.
- Overall per-example score divides achieved criterion points by total positive
  criterion points.
- Aggregate metrics are clipped to `[0, 1]`.

## Operational Notes

- Do not commit source JSONL, generated-response JSONL, parquet exports, or
  activation dumps.
- HealthBench Consensus has been loaded into Neon table
  `healthbench_consensus_v1` with 3,671 rows.
- Use `Dataset.from_postgres(...)` for later workflow specs.
- Reports should reference prompt ids and aggregate slices, not raw examples.

## Phase 01 Labeling Notes

First-pass ontology should stay small. Good candidates to discuss:

- prompt-side `context_needed`
- prompt-side `triage_urgency`
- prompt-side `audience_framing`
- response-side `scope_calibration`
- response-side `context_seeking_before_advice`

Likely nuisance variables:

- topic or specialty
- language
- turn count
- prompt length bucket
- rubric count
- obvious emergency terms
- clinician vs layperson persona

## Open Concerns

- The HealthBench labels may be too response-outcome-oriented for clean
  prompt-side probes unless we derive narrower labels.
- Response-side labels will require stacked confound controls: at minimum a
  viewport decision plus either text-baseline residualization or validated
  training-distribution variation.
- Qwen generation behavior on HealthBench is not yet known; behavioral sanity
  comes before capture.
