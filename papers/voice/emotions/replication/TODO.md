# TODO Checklist

## Source Details

- [x] Paper URL/title are in `configs/replication.todo.toml`.
- [x] Full citation is in `configs/replication.todo.toml`.
- [x] BibTeX is in `paper.bib`.
- [ ] TODO: record exact model studied in the paper as a source fact, separately
  from the Qwen replication target.
- [ ] TODO: record exact generator model used for synthetic stories/dialogues.
- [x] Exact list of emotion concepts is in `data/emotions.txt`.
- [x] Exact topic list is in `data/topics.txt`.
- [x] Story count is recorded as 12 stories per topic per emotion.
- [ ] TODO: record exact neutral-dialogue and emotional-dialogue counts if they
  differ from the story count.

## Dataset

- [x] `prompts/emotional_stories.md` filled from the appendix excerpt.
- [x] `prompts/neutral_transcripts.md` filled from the appendix excerpt.
- [x] `prompts/emotional_dialogues.md` filled from the appendix excerpt.
- [ ] TODO: decide whether downstream dialogue variants beyond the basic emotional-dialogue prompt are in scope.
- [ ] TODO: implement/upload durable dataset rows behind a loader. Proposed
  Neon tables are in `configs/replication.todo.toml`.
- [ ] TODO: inspect a smoke slice before scaling.
- [ ] TODO: verify target emotion word/synonym bans.
- [ ] TODO: verify parser output columns:
  `example_id`, `emotion`, `topic`, `story_index`, `text`.

## Captures

- [x] Cheap first-pass target is set to `Qwen/Qwen3-8B`.
- [ ] TODO: prewarm `Qwen/Qwen3-8B` into `xenon-models` at
  `/models/Qwen/Qwen3-8B`, or switch to an already-mounted model.
- [x] Initial cheap-model capture sweep is set to layers `8, 16, 24, 32`.
- [x] Story token selector starts at token 50 to match the paper recipe.
- [ ] TODO: set neutral token selector.
- [x] Residual site is set to `resid_post`.

## Vector Space

- [x] `min_examples_per_concept` is set in config.
- [x] Formula is set by `EmotionVectorSpaceSpec`:
  `mean(concept examples) - mean(concept means)`.
- [x] Neutral-PC projection threshold is set to 50% explained variance.
- [x] `EmotionVectorSpaceSpec` exports raw and normalized vectors in the vector-space payload.

## Validation

- [ ] TODO: run geometry diagnostics.
- [ ] TODO: run held-out scoring on non-training text.
- [ ] TODO: compare raw vectors vs neutral-projected vectors.
- [ ] TODO: inspect high-projection snippets.
- [ ] TODO: document claim boundary in `reports/`.

## Steering

- [x] Initial target emotions are set in config.
- [x] Initial write-layer sweep is set in config separately from readout layer.
- [x] Random-direction controls are listed in config.
- [x] Same-topic neutral controls are listed in config.
- [ ] TODO: define success/failure before running steering.

## Storage And Indexing

- [x] Artifacts are configured for Modal volume `xenon-data`.
- [x] Model/cache volume is configured as `xenon-models` at `/models`.
- [x] Shared catalog uses `XENON_NEON_DATABASE_URL`.
- [x] Remote jobs should bind Modal secret `xenon-neon`.
- [ ] TODO: local shell needs `XENON_NEON_DATABASE_URL` exported from `.env`
  before local Neon-backed CLI runs.
