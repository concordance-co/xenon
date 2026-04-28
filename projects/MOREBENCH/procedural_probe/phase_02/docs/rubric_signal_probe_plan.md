# MoReBench Phase 2 Raw Dimension Probe Plan

Status note: this workflow has now produced a useful negative result. See
`projects/MOREBENCH/procedural_probe/phase_02/docs/run_results_and_pivot.md`
for the latest run interpretation and the recommended latent-label pivot.

Grounding:

- local skill: `benchmark-validation`
- local skill: `benchmark-to-latent-labels`
- local skill: `benchmark-mech-interp-analysis`
- local skill: `constructing-llm-probes`
- methodology roster: `methodology/ROSTER.md`
- benchmark methodology: `methodology/PRINCIPLES.md`, `methodology/CHECKS.md`
- paper: https://arxiv.org/pdf/2510.16380
- official repo: https://github.com/morebench/morebench

## Purpose

Phase 2 no longer probes aggregate rubric-weight burden from the prompt alone.
That was the wrong level of abstraction for the benchmark.

The revised question is:

- using the raw MoReBench rubric dimensions, with criterion weights ignored, are
  dimension-family labels decodable at the end of the prompt and at the end of
  the model's generated-answer context?

This separates two surfaces that should not be conflated:

- prompt-side labels: what can the model infer while reading the dilemma?
- generation-context labels: what state is present in generated tokens that the
  model has actually fed back through later decode steps?

Most MoReBench rubric criteria are fundamentally about generated responses:
whether the answer identifies the dilemma, follows a clear/logical process, gives
a helpful outcome, or avoids harmful outcomes. Prompt-only probing is still a
useful baseline, but it is not sufficient.

## Raw Dimension Targets

The workflow builds dilemma-level labels from criterion rows by counting rubric
dimensions and ignoring criterion weights.

Primary target:

- `dominant_dimension_by_count`
  - multiclass label
  - value is the rubric dimension with the largest raw criterion count
  - candidate feature: raw rubric-dimension family state

One-vs-rest targets:

- `dominant_identifying_by_count`
- `dominant_clear_process_by_count`
- `dominant_logical_process_by_count`
- `dominant_helpful_outcome_by_count`

These are not official MoReBench scores. They are first-pass labels that collapse
many sparse criterion titles into a smaller set of rubric families.
`dominant_harmless_outcome_by_count` is intentionally not a runnable target in
the current public split because the label artifact is single-class (`no=500`),
so a classifier baseline or probe would be ill-defined.

## Workflow

The checked-in workflow is:

1. `build_raw_rubric_dimension_labels`
2. `generate_dilemma_responses`
3. `build_successful_generation_capture_dataset`
4. `capture_prompt_generated_residual`
5. for each raw dimension target:
   - `probe_prompt_*_residual`
   - `text_baseline_prompt_*`
   - `probe_generation_*_residual`
   - `text_baseline_generation_*`
6. `report`

Generation and capture are split:

- `generate_dilemma_responses` writes generated text and finish reasons to the
  Modal artifact volume.
- `GenerationSpec.max_tokens=None` leaves generation uncapped, bounded by vLLM's
  stop/context behavior rather than a project-local token cap.
- Rows with `finish_reason=length` are excluded before capture for now.
- `build_successful_generation_capture_dataset` emits prompt/generated/full
  token sections plus endpoint sections.
- `capture_prompt_generated_residual` replays prompt plus generated answer text
  and captures `prompt_end`, `generated_end`, and `full_end`.

Important capture semantics:

- This phase does not rely on generation-time hidden state capture.
- Generated-answer activations are from the replayed prompt+answer context.
- The current Qwen3-30B-A3B model config has `max_position_embeddings=40960`
  with no rope scaling; running above that needs explicit rope-scaling support.

No generated-answer text is materialized to Postgres for this phase.

## Metric Interpretation

For prompt-end residual probes:

- balanced accuracy or AUROC near `0.5` means little decodable prompt-side signal
- strong metrics mean the prompt state predicts raw rubric dimension structure
- strong metrics must beat prompt-text baselines and shuffled/selectivity controls
- for the multiclass `dominant_dimension_by_count` activation probes, AUROC is
  omitted and accuracy/balanced accuracy are interpreted against majority and
  shuffled-label controls

For generation-context residual probes:

- balanced accuracy or AUROC near `0.5` means little decodable post-answer signal
- stronger metrics than prompt-end probes suggest the answer text/state contains
  more rubric-dimension information than the dilemma alone
- this is still a decodability claim, not causal evidence

For text baselines:

- prompt baselines use dilemma text
- generation baselines use generated answer text
- high text-baseline metrics mean the label is available from surface wording or
  answer style, so activation readouts need to be interpreted conservatively

## Methodology Fit

From the methodology roster, this phase uses:

- linear residual probes
- layer sweep
- text-only baselines
- shuffled/selectivity controls
- prompt-vs-generation surface comparison

It intentionally does not use:

- reusable LLM-as-judge abstractions in `pipelines_v2`
- official criterion-fulfillment scoring
- activation patching
- steering
- LEACE/INLP

Those belong after we know whether raw rubric-family labels are recoverable and
which surface is more informative.

## Confounds

Primary confounds:

- raw dimension labels are still aggregates, not atomized concepts
- dominant-dimension labels can be class-imbalanced
- generated answers may leak labels through style rather than internal state
- domain/topic may correlate with rubric dimensions

Current controls:

- prompt text baseline for each target
- generated-answer text baseline for each target
- shuffled-label/selectivity control on residual probes
- grouped evaluation by `base_dilemma_id`
- no source/type/context targets in the main workflow

Likely next controls:

- intermediate labels within each rubric dimension, especially for sparse
  identifying/logical criterion titles
- fine-grained domain labels for daily dilemma and AI-risk examples
- project-local criterion-fulfillment judging against generated answers

## Success Criteria

The phase is useful if it produces:

- raw dimension label support and class balance checks
- one prompt-end residual capture
- one generated-context residual capture from the same pass as answer generation
- prompt-end versus generation-context probe metrics for each target
- matching prompt-text and generated-answer text baselines
- a clear decision about whether later work should focus on prompt
  comprehension, generated-answer state, intermediate labels, or judge-scored
  criterion fulfillment
