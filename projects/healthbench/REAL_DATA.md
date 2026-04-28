# HealthBench Data Context

**Date:** 2026-04-28
**Version:** `v0`

This project uses the public HealthBench release from OpenAI's `simple-evals`
repository. The source data should not be committed locally as JSONL or parquet.
Durable benchmark tables belong in Neon, loaded by repeatable uploader scripts.

## Source Surfaces

- Repository: https://github.com/openai/simple-evals
- HealthBench announcement: https://openai.com/index/healthbench/
- Primary eval implementation:
  https://github.com/openai/simple-evals/blob/main/healthbench_eval.py
- Consensus source blob:
  `https://openaipublic.blob.core.windows.net/simple-evals/healthbench/consensus_2025-05-09-20-00-46.jsonl`
- Full source blob:
  `https://openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl`
- Hard source blob:
  `https://openaipublic.blob.core.windows.net/simple-evals/healthbench/hard_2025-05-08-21-00-10.jsonl`

## Active Neon Tables

- `healthbench_consensus_v1`
- Source: HealthBench Consensus blob above.
- Loader: `projects/healthbench/phase_00/scripts/upload_healthbench_consensus_to_neon.py`
- Purpose: canonical shared source for HealthBench Consensus prompt/rubric
  metadata, not generated model outputs or activation artifacts.
- Uploaded: 2026-04-28.
- Row count at upload: 3,671.

## Handling Rules

- Do not commit raw HealthBench rows to the repository.
- Do not include raw prompt/rubric examples in public-facing reports.
- Keep generated responses and activations in the normal workflow artifact
  stores, not local JSONL dumps.
- Use Neon as the durable benchmark metadata surface once ingestion begins.
