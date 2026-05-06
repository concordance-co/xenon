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
- [x] Tier A generated-data workflow now parses generated story/dialogue blocks
  into artifact-backed capture datasets.
- [ ] TODO: inspect Tier A generated rows before scaling. Latest e2e run
  `wr_ca397883191d_e59ec1b8` recovered 70/96 target train rows and 35/48
  target heldout rows.
- [ ] TODO: inspect Tier A target emotion word violations; the current parser
  counts exact target-word mentions but keeps rows for the first fidelity run.
- [x] Tier A parser output columns include:
  `example_id`, `emotion`, `topic`, `story_index`, `text`.

## Captures

- [x] Cheap first-pass target is set to `Qwen/Qwen3-8B`.
- [ ] TODO: prewarm `Qwen/Qwen3-8B` into `xenon-models` at
  `/models/Qwen/Qwen3-8B`, or switch to an already-mounted model.
- [x] Initial cheap-model capture sweep is set to layers `8, 16, 24, 32`.
- [ ] TODO: restore story token selector to token 50+ after Tier A generated
  story lengths are inspected.
- [x] Tier A neutral projection uses generated neutral-dialogue captures with
  full-sequence residual pooling.
- [x] Residual site is set to `resid_post`.

## Vector Space

- [x] `min_examples_per_concept` is set in config.
- [x] Formula is set by `EmotionVectorSpaceSpec`:
  `mean(concept examples) - mean(concept means)`.
- [x] Neutral-PC projection threshold is set to 50% explained variance.
- [x] `EmotionVectorSpaceSpec` exports raw and normalized vectors in the vector-space payload.

## Validation

- [x] Tier A workflow includes geometry diagnostics.
- [x] Tier A workflow includes held-out scoring on non-training topics.
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
- [x] Local Neon-backed CLI runs work when the shell exports
  `XENON_NEON_DATABASE_URL` from `.env`.
