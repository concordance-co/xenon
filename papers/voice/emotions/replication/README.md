# Emotion Vector Paper Replication Scaffold

This directory is for recreating the emotion-vector dataset and vector
extraction recipe from the paper inside Xenon. It is intentionally not a
`projects/.../phase_XX` research phase.

The checked-in smoke workflow remains at `../specs/workflow.py`. This scaffold
is the paper-scale workbench: fill the TODOs from the paper, generate/capture
through normal Xenon storage, and keep raw datasets/artifacts out of git.

## Fill First

1. `configs/replication.todo.toml`
   - Paper citation, emotion/topic lists, prompt paths, cheap-model defaults,
     storage settings, and remaining TODOs.
2. `prompts/emotional_stories.md`
   - Story-generation prompt from the paper appendix excerpt.
3. `prompts/neutral_transcripts.md`
   - Neutral dialogue prompt and 50% neutral-PC projection note.
4. `prompts/emotional_dialogues.md`
   - Basic emotional-dialogue generation prompt.
5. `specs/replication_workflow_outline.py`
   - Non-default workflow outline for paper-scale execution.

## Intended Flow

1. Generate a small story smoke set.
2. Inspect the rows by eye and verify parser/label contracts.
3. Write durable generated rows to Neon or use a deferred external dataset.
4. Capture residual activations on Modal.
5. Build `EmotionVectorSpaceSpec` from story captures, with neutral projection.
6. Run `EmotionGeometrySpec`, `EmotionScoreSpec`, and `EmotionDirectionSpec`.
7. Only then run steering with controls.

## Runnable Workflows

Tiny model-cache smoke:

```bash
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/emotions/replication/specs/qwen_smoke_workflow.py
```

Handwritten four-emotion pilot:

```bash
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/emotions/replication/specs/happy_vector_pilot_workflow.py
```

Tier A generated-data vector space:

```bash
export XENON_NEON_DATABASE_URL="$(sed -n 's/^XENON_NEON_DATABASE_URL=//p' .env)"
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/emotions/replication/specs/tier_a_generated_vector_workflow.py
uv run python -m pipelines_v2.cli workflow run --file papers/voice/emotions/replication/specs/tier_a_generated_vector_workflow.py --logging INFO
```

Tier A generates `happy`, `sad`, `angry`, and `calm` story datasets with Qwen,
captures layers `8,16,24,32`, computes a neutral-projected vector space with
full-sequence pooling, scores heldout stories at layer `24`, and exports all
four directions. The parser counts direct target-word violations but keeps rows
for this first fidelity run. Switch back to token-50+ pooling and stricter
filtering after inspecting generated story lengths and text quality.

Latest completed Tier A run:

- run id: `wr_ca397883191d_e59ec1b8`
- report:
  `papers/voice/emotions/replication/reports/tier_a_generated_vectors/report_621f284bbd6b_c5ab4054/report.md`
- vector space: `emotion_vector_space_1_5bd3a454`
- directions:
  `happy=emotion_direction_1_5165556a`,
  `sad=emotion_direction_1_2537811b`,
  `angry=emotion_direction_1_d3089c72`,
  `calm=emotion_direction_1_43e2d0d5`
- heldout score summary: 35 examples, 10 correct, 28.6% top-1 vs 25% chance.

This is an end-to-end plumbing validation, not a fidelity claim. The generated
story parser recovered 70/96 target train rows and 35/48 target heldout rows,
and counted 40 exact target-emotion word mentions.

## Locality Rules

- Do not commit generated story rows, neutral transcripts, activations, model
  outputs, vector dumps, or safetensors.
- Small config files, prompt templates, and report summaries are fine.
- Use `data/README.md` as the data manifest, not as a place to store data.

## Modal / Neon Defaults

- Cheap first-pass target: `Qwen/Qwen3-8B`.
- Repo-standard larger target: `/models/Qwen/Qwen3-30B-A3B`.
- Model/cache volume: `xenon-models` mounted at `/models`.
- Artifact volume: `xenon-data` under
  `/data/artifacts/model-assets/vectors/emotions/...`.
- Shared catalog: `PostgresCatalog` from `XENON_NEON_DATABASE_URL`.
- Remote Neon secret: Modal secret `xenon-neon`.

Persistent storage avoids repeated model downloads after a model is present on
the mounted volume. It does not guarantee the model remains resident in GPU
memory after Modal scales a container down.
