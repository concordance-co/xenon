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
  `/data/artifacts/papers_voice_emotions_replication`.
- Shared catalog: `PostgresCatalog` from `XENON_NEON_DATABASE_URL`.
- Remote Neon secret: Modal secret `xenon-neon`.

Persistent storage avoids repeated model downloads after a model is present on
the mounted volume. It does not guarantee the model remains resident in GPU
memory after Modal scales a container down.
