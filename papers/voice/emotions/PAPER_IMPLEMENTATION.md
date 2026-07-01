# Emotion Vectors Implementation Status

Status: planned Llama 3.3 70B asset package.

This directory mirrors the Assistant Axis asset shape while keeping the
emotion-paper methodology separate. The package entrypoint is
`papers.voice.emotions.assets`, the Llama 70B manifest is
`papers/voice/model_assets/vectors/emotions/llama-3.3-70b/sofroniew-2026/v1/manifest.toml`,
and the paper-scale workflow is
`papers/voice/emotions/replication/specs/llama70b_vector_workflow.py`.

## Asset Target

- Model: `meta-llama/Llama-3.3-70B-Instruct`
- Current model volume: `yora-models` mounted at `/models`
- Artifact root: `/data/artifacts/model-assets/vectors/emotions/llama-3.3-70b/sofroniew-2026/v1`
- Generated text table: `papers_voice_emotions_llama70b_generated_rows_v1`
- Concepts: all 171 emotions from `replication/data/emotions.txt`
- Pilot concepts: `happy`, `sad`, `angry`, `calm`
- Generation compute: `H100:4`, tensor parallel `4`, max model length `16384`,
  `max_num_seqs=8`
- Full capture compute: `H200:2`, tensor parallel `2`, four capture shards,
  `max_num_seqs=512`, 128 GiB memory, and 24-hour timeout
- Default capture max model length: `16384` tokens; override with
  `EMOTION_ASSET_MAX_MODEL_LEN` if inspected generations require more headroom
- Full-mode activation capture stores residual stream tokens 50+ after mean
  pooling; pilot mode keeps full-sequence capture for smoke-test visibility
- Neutral control: project out neutral transcript PCs up to variance threshold `0.5`

## Workflow Shape

The Llama 70B generation workflow first writes durable story and neutral rows to
Neon. After those rows are inspected, full-mode vector runs consume the Neon
table directly for activation capture, vector-space derivation, geometry
diagnostics, heldout scoring, and one steering direction artifact per emotion.
Pilot and full vector runs both consume inspected Neon rows; generation belongs
in `llama70b_generation_workflow.py`.

Pilot mode validates the plumbing with four emotions. Full mode uses all 171
emotions and is the required mode for demo-grade assets because each vector is
centered by the across-concept mean.

## Current Boundary

No Llama 70B emotion vector-space artifact has been promoted yet. The manifest
therefore remains `planned`, and `assets.precomputed_vector_space_spec()` raises
until the manifest records a materialized vector-space path.
