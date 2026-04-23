# MoReBench Phase 1 Rubric Validation Plan

## Core Correction

The rubric entries are not independent supervised examples. They are multiple expert annotations attached to one query. A single query can have many criteria across dimensions, with positive and negative weights.

That means Phase 1 should not ask:

- can a criterion title predict its rubric dimension?
- can a criterion title predict its own weight?

Those are mostly schema/wording checks. The useful validation object is the full rubric set for a query.

## Validation Unit

Phase 1 now derives one `rubric_profile` per `base_dilemma_id`.

The profile includes:

- dominant weighted rubric dimension
- whether any negative `harmless outcome` penalty exists
- whether both positive `helpful outcome` mass and negative `harmless outcome` mass exist
- low/medium/high rubric complexity
- low/medium/high count of high-weight criteria
- per-dimension criterion counts and signed/absolute weight mass

This preserves the fact that MoReBench gives a set of criteria per query.

## Current Workflow

The checked-in workflow expands Hugging Face `RUBRIC` rows at remote runtime, then collapses them into scenario-level profiles:

1. `build_rubric_profiles`
2. `dilemma_text_to_dominant_dimension_baseline`
3. `dilemma_text_to_harmless_penalty_baseline`
4. `dilemma_text_to_helpful_harmless_tension_baseline`
5. `dilemma_text_to_rubric_complexity_baseline`

All text baselines use query/dilemma text as the input and profile attributes as labels.

Metric interpretation:

- High metrics mean a profile attribute is predictable from query topic alone.
- Low metrics mean the attribute likely depends on the expert rubric set, not just surface query text.
- These are validation/leakage checks, not mechanistic evidence.

## Why This Is Better

The original criterion-title baselines tested the rubric against itself. The revised workflow tests whether the query text already determines the shape of the expert evaluation rubric.

This is the right pre-capture question because later activation work should ask whether a model internally represents:

- the rubric profile implied by the query
- the constraints in the rubric set
- criterion fulfillment while generating an answer
- helpful-vs-harmless tension

## Next Experiments

### Response Fulfillment Labels

Generate answers, then judge each answer against each criterion:

- criterion fulfilled: binary or scalar
- signed weighted contribution: `criterion_weight * fulfillment`
- dimension-level score
- total rubric score

This is closer to MoReBench scoring than metadata prediction.

### Rubric-Conditioned Capture

Prompt the model with the query plus either:

- the full rubric set
- selected criteria
- a compact rubric-profile description

Do not reveal profile labels. Capture activations across query, rubric, and answer sections.

Probe targets:

- profile attributes from Phase 1
- criterion fulfillment
- dimension-level scores
- helpful-vs-harmless tension

### Helpful-Harmless Tradeoff Geometry

Use profile labels and fulfillment labels to build:

- helpful-vs-harmless directions
- PCA/LDA geometry views
- transfer tests across dilemma families
- causal steering or activation-patching interventions

## Controls

- Use `base_dilemma_id` as the validation unit.
- Keep query text, criterion text, and generated answer text separate.
- Do not include profile labels in activation-capture prompts.
- Use query-text baselines as leakage checks before activation probes.
- Treat high predictability as a warning that profile labels may be topic shortcuts.

## Missing Implementation Pieces

- criterion-fulfillment labeling op for generated answers
- signed rubric score aggregation op
- prompt metadata builder for query/rubric/answer token sections
- report template for profile distributions and profile-predictability baselines
- activation workflow over profile and fulfillment labels
