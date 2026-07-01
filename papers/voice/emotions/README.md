# Emotion Vectors

Goal: reproduce the Xenon-native core of the Transformer Circuits emotions work
for Llama 3.3 70B: derive emotion concept vectors, score heldout sections,
inspect geometry, and export one direction per emotion for steering.

This package now follows the same public asset shape as Assistant Axis:

- `assets.py`: manifest loader and future precomputed vector-space helper.
- `PAPER_IMPLEMENTATION.md`: current Llama 70B asset status.
- `replication/specs/llama70b_generation_workflow.py`: generation-only workflow
  that writes parsed Llama 70B story/neutral rows to Neon.
- `replication/specs/llama70b_vector_workflow.py`: full-scale workflow for the
  Sofroniew et al. emotion vector space.
- `../model_assets/vectors/emotions/llama-3.3-70b/sofroniew-2026/v1/manifest.toml`:
  reusable asset manifest.

Smoke command:

```bash
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/emotions/specs/workflow.py
```

Llama 70B asset pilot plan:

```bash
XENON_NEON_DATABASE_URL=postgresql://placeholder \
EMOTION_ASSET_MODE=pilot \
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/emotions/replication/specs/llama70b_generation_workflow.py
```

Llama 70B full generation run:

```bash
export XENON_NEON_DATABASE_URL="..."
EMOTION_ASSET_GENERATED_ROWS_TABLE=papers_voice_emotions_llama70b_generated_rows_v1 \
EMOTION_ASSET_STRICT_PERSISTENCE_QA=1 \
EMOTION_ASSET_MODE=full \
uv run python -m pipelines_v2.cli workflow run --file papers/voice/emotions/replication/specs/llama70b_generation_workflow.py --logging INFO
```

Full generation defaults are paper-shaped: 171 emotions, 100 story topics, 12
stories per topic/emotion, 100 neutral topics, and 12 neutral dialogues per
topic. Llama 70B generation uses `H100:4`, tensor parallel `4`, max model
length `16384`, `max_num_seqs=8`, `max_tokens=8192`, and a 24-hour generation
timeout. Strict persistence QA fails the run before Neon writes if any source
prompt yields fewer than 12 parseable stories/dialogues.

Llama 70B vector/capture pilot plan:

```bash
XENON_NEON_DATABASE_URL=postgresql://placeholder \
EMOTION_ASSET_MODE=pilot \
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/emotions/replication/specs/llama70b_vector_workflow.py
```

Full 171-emotion plan:

```bash
XENON_NEON_DATABASE_URL=postgresql://placeholder \
EMOTION_ASSET_MODE=full \
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/emotions/replication/specs/llama70b_vector_workflow.py
```

The vector workflow consumes the inspected Neon table
`papers_voice_emotions_llama70b_generated_rows_v1` directly. It does not
regenerate stories or rebuild capture datasets from generation artifacts. The
default full capture shape is cost-insensitive: `H100:4`, tensor parallel `4`,
`max_num_seqs=16`, chunked prefill enabled, `gpu_memory_utilization=0.92`, eight
capture shards, eight max containers, 192 GiB container memory, and a 24-hour
capture timeout. Analysis steps default to 16 CPU cores, 256 GiB memory, and a
12-hour timeout.

Expected artifacts: capture, emotion vector space, emotion scores, one exported
direction per emotion, geometry diagnostics, and report summary.

Historical smoke default: `Qwen/Qwen3-8B`. The demo asset target is now
`meta-llama/Llama-3.3-70B-Instruct` on the transitional `yora-models` volume,
with reusable artifacts under
`/data/artifacts/model-assets/vectors/emotions/llama-3.3-70b/sofroniew-2026/v1`.
Generation uses `H100:4`, tensor parallel `4`, max model length `16384`, and
`max_num_seqs=8`; the parsed text checkpoint is table
`papers_voice_emotions_llama70b_generated_rows_v1` in Neon.

Generation calibration evidence:

- Estimates and launch checklist:
  `replication/LLAMA70B_GENERATION_STATUS.md`.
- `wr_4a37c7ffd513_45ec6346`: 50 story prompts plus 5 neutral prompts at
  `max_tokens=4096`; all 600 story rows parsed, but neutral required stronger
  parsing and more headroom.
- `wr_cd914fcd8f4f_23e35982`: 5 neutral prompts at `max_tokens=6144` with
  strict QA; 60/60 neutral rows parsed, zero length caps.

Data hooks:

- `emotion_probe_story_dataset(limit=...)` points at a public generated
  story/dialogue probe mirror with `real_emotion`, `displayed_emotion`, `topic`,
  and `text` columns.
- `emotion_contrast_dataset(records=...)` maps arbitrary agent logs, stories,
  or transcript sections into the same labeled vector-space workflow.
- `EmotionPrecomputedVectorSpaceSpec(...)` remains the path for released or
  user-owned precomputed emotion-space artifacts.

Paper-scale scaffold:

- `replication/` contains TODO-marked prompts, config, data manifest, report
  directories, and workflows for recreating the paper's story-vector recipe
  without treating the work as a normal Xenon research phase.

Claim boundary: the Qwen smoke vectors are fixture-only, and the Llama 70B
asset is still planned until the pilot and full runs record artifact IDs in the
manifest. Paper-level claims require large labeled story/dialogue data,
naturalistic transcript checks, preference or behavior evaluations, and
intervention controls.
