# Llama 3.3 70B Emotion Generation Status

Date: 2026-05-26

This document tracks where the Llama 3.3 70B emotion-vector replication stands
before launching the full generated-row job. The immediate objective is to
produce paper-shaped story and neutral dialogue rows for the Sofroniew et al.
emotion-vector workflow, persist them in Neon, and then use those rows for the
activation capture/vector-space run.

## Current State

- Prompt shape is aligned with the paper appendix excerpts we have: generated
  prompts request `n_stories` different dialogues/stories for a topic, use the
  appendix scenario/dialogue structure, and keep the `Person`/`AI` conversion as
  post-processing rather than asking the model to rewrite the format.
- Full mode is configured for all 171 emotions from `data/emotions.txt`, 100
  story topics, 100 neutral topics, and 12 generated rows per source prompt.
- Generation is separated from activation capture. The first full job only
  writes durable generated text rows to Neon; capture and vector construction
  come after generated-row QA.
- The generation workflow uses `meta-llama/Llama-3.3-70B-Instruct` on the
  transitional `yora-models` Modal volume and writes reusable assets under
  `model-assets/vectors`, not product-specific storage.
- Strict persistence QA is wired so a full run fails before writing rows if any
  prompt yields fewer than the requested 12 parseable outputs.

## Calibration Runs

| Run | Scope | Settings | Result |
| --- | --- | --- | --- |
| `wr_4a37c7ffd513_45ec6346` | 10 emotions x 5 story topics plus 5 neutral topics | `H100:4`, tensor parallel 4, `max_tokens=4096` | 600/600 story rows parsed. Neutral rows exposed parser/headroom issues. Runtime was about 10.5 minutes including startup. |
| `wr_cd914fcd8f4f_23e35982` | 5 neutral prompts only | `H100:4`, tensor parallel 4, `max_tokens=6144`, strict QA | 60/60 neutral rows parsed, zero length caps, persisted to `emo70b_neutral6144_20260526`. |

Smoke-only tables from earlier format tests were dropped. The useful calibration
tables left in Neon are `emo70b_calib_20260526` and
`emo70b_neutral6144_20260526`.

## Full Run Size

Full generation has:

- 171 emotions
- 100 story topics
- 12 stories per emotion/topic prompt
- 100 neutral topics
- 12 neutral dialogues per topic prompt

That means:

- Story prompts: `171 * 100 = 17,100`
- Neutral prompts: `100`
- Total generation prompts: `17,200`
- Expected story rows: `17,100 * 12 = 205,200`
- Expected neutral rows: `100 * 12 = 1,200`
- Expected generated rows total: `206,400`

The generation workflow records prompt-level metadata, parse counts, finish
reasons, raw generation text, and parsed row text so failed or suspicious source
prompts can be audited without rerunning the entire job blindly.

## Runtime Estimate

The best current estimate is based on the 55-prompt mixed calibration run:

- 55 prompts took about 10.5 minutes wall clock including Modal startup.
- The run generated 109,948 output tokens.
- Naive prompt-count scaling gives about 55 hours for 17,200 prompts.

This is only a rough planning number. Small-run startup overhead inflates the
per-prompt estimate, while very long neutral generations and batching behavior
can push the other direction. The full workflow timeout is therefore set to 72
hours. A full run on the current Xenon/vLLM path should be treated as a
multi-day job unless later batching measurements prove otherwise.

## Current Generation Defaults

The generation-only workflow defaults are:

- Workflow: `papers/voice/emotions/replication/specs/llama70b_generation_workflow.py`
- Mode: `EMOTION_ASSET_MODE=full`
- Model: `meta-llama/Llama-3.3-70B-Instruct`
- Model volume: `yora-models`
- Model mount: `/models`
- GPU: `H100:4`
- Tensor parallel size: `4`
- Max model length: `8192`
- Max output tokens: `6144`
- Max concurrent sequences: `8`
- Temperature: `0.8`
- Top-p: `0.95`
- Timeout: `72h`
- Destination table: `papers_voice_emotions_llama70b_generated_rows_v1`

## Launch Command

```bash
export XENON_NEON_DATABASE_URL="$(sed -n 's/^XENON_NEON_DATABASE_URL=//p' ../xenon/.env)"

EMOTION_ASSET_GENERATED_ROWS_TABLE=papers_voice_emotions_llama70b_generated_rows_v1 \
EMOTION_ASSET_STRICT_PERSISTENCE_QA=1 \
EMOTION_ASSET_MODE=full \
uv run python -m pipelines_v2.cli workflow run \
  --file papers/voice/emotions/replication/specs/llama70b_generation_workflow.py \
  --logging INFO
```

The matching plan command has been checked locally and resolves to two steps:
`generate_rows` followed by `persist_generated_rows`.

## Acceptance Before Capture

Before starting the activation capture/vector-space run, verify:

- Neon row count is exactly 206,400 for the full generated-row table.
- Prompt-level QA shows 12 parsed rows for every story and neutral source
  prompt.
- Finish reasons are not dominated by `length`; any capped prompts are inspected
  manually.
- Representative examples still match the paper-shaped scenario/dialogue
  contracts.
- No obvious direct-target leakage or parser artifacts are severe enough to
  require prompt changes and regeneration.

## Next Step After Generation

Once generated rows pass QA, mirror the Assistant Axis asset flow for emotions:
capture Llama 70B residual activations, build the 171-concept emotion vector
space with neutral projection and token-50+ pooling, write geometry and heldout
score artifacts under `model-assets/vectors/emotions/llama-3.3-70b`, then update
the emotion manifest from `planned` to `pilot` or `validated` with run and
artifact IDs.
