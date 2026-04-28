# HealthBench Phase 00

## Premise

Start HealthBench as a benchmark-first mech-interp project. The goal of this
phase is substrate validation and project framing, not activation capture or
mechanistic claim-making.

Inputs:

- benchmark: HealthBench from OpenAI `simple-evals`
- initial slice: HealthBench Consensus
- probe-target model: `/models/Qwen/Qwen3-30B-A3B`
- research mode: causality-aware framing, with the first concrete causal
  hypothesis deferred until after this validation pass
- durable data surface: HealthBench Consensus metadata in Neon table
  `healthbench_consensus_v1`

Success means we can say whether HealthBench Consensus is worth advancing to
latent-label formation and what risks must be controlled before any probe or
intervention work.

Planned sequence after Phase 00 framing:

1. Freeze a minimal latent-label ontology in Phase 01.
2. Run Qwen behavioral sanity on a small Consensus slice before any capture.
3. Build text baselines and nuisance-stratified checks before probe work.

## What we ran

## Primary result

## Qualitative inspection

## Corrections

## Running hypothesis

## Claim boundary

## Artifacts

- `projects/healthbench/REAL_DATA.md`
- `projects/healthbench/benchmark_context.md`
- `projects/healthbench/phase_00/docs/00-validation-memo.md`
- `projects/healthbench/phase_00/docs/00-validation-summary.json`
- `projects/healthbench/phase_00/docs/00-validation-notes.md`
- `projects/healthbench/phase_00/scripts/upload_healthbench_consensus_to_neon.py`
- `projects/healthbench/phase_00/specs/workflow.py`
- Neon table: `healthbench_consensus_v1`

## Open threads

- Choose the first hypothesis family before Phase 01 ontology freeze.
- Decide whether first labels should be prompt-side, response-side, or both.
- Define Qwen generation protocol and grader protocol.
- Design text baselines and nuisance stratification before any activation
  capture.
