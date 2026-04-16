# Prompt Confusion Phase 09 Design

## Goal

Phase 09 is a fresh rebuild of the prompt-confusion benchmark around the part
of the project that still looks real:

- conflict detection between policy sources
- lexically non-trivial relational structure
- prompt designs that are behaviorally legible enough to support reliable
  binary labels

It is **not** an attempt to rescue:

- family identity
- generic arbitration as a headline variable

## Why this phase exists

Phase 04 and Phase 05 jointly showed:

- conflict signals are real
- family-level interpretations were too confounded
- lexical harmonization alone cannot rescue a structurally wrong target

Later redesign work also showed:

- the activity prompt language was too vague
- weird phrases like `live case` made both human and model interpretation
  unstable
- the benchmark likely had prompt-induced label ambiguity, which contributed
  to the mismatch between strong AUROC and weaker balanced accuracy

The prompt redesign handoff note is the main conceptual reference:

- [prompt_redesign_handoff_20260415.md](../../notes/prompt_redesign_handoff_20260415.md)

## New prompt philosophy

### Roles

- `STRATEGY`
  - preference or style
- `ACTIVE SETTINGS`
  - binding execution constraints
- `MARKET`
  - descriptive evidence about assets
- `PORTFOLIO`
  - available resources / current state

### Execution order

The prompt should make it natural for the model to reason in this order:

1. do ACTIVE SETTINGS permit entry?
2. if yes, which asset is best?
3. if yes, what size is permitted?

This is meant to reduce ambiguous blended reasoning where strategy, market,
and settings are all treated as equally negotiable.

### Descriptive, not behavior-coded, market text

`MARKET` should describe:

- momentum
- confirmation
- follow-through
- uncertainty
- signal noisiness

It should not directly imply:

- whether the model should enter
- whether the evidence is already sufficient for the settings threshold

This is the most important prompt-language change in the rebuild.

## Dataset design

Phase 09 keeps the relational conflict framing:

- `target_dimension ∈ {trade_size, trading_activity}`
- `strategy_direction` varies within the dataset
- `setting_value` varies within the dataset
- `setting_implied_direction` is derived from the setting and context
- `conflict_present` is defined relationally

The builder in this phase is only a scaffold, but it already follows the new
principles:

- descriptive market contexts
- explicit settings-constrain framing
- reduced weird terminology
- reusable strategy shells
- full settings block
- balanced lexical split over strategy-shell and settings-shell combinations
- explicit phrase-family metadata for activity and size phrasing

### Primary benchmark vs stress-test rows

Phase 09 now distinguishes between:

- `main_benchmark_row = true`
  - rows used for the primary text gate / probe / transfer results
- `stress_test_slice != ""`
  - rows kept for targeted behavioral or interpretive follow-up

The main excluded slice is:

- `trading_activity`
- `setting_value = 1`
- `evidence_tier = exceptional`

These rows are preserved because they are scientifically interesting, but they
are no longer part of the primary binary benchmark. In practice they behave
more like a threshold-boundary stress test than a clean label regime.

## Evaluation intent

Once the rebuilt dataset is published, the `pipelines_v2` workflow should
cover:

1. text-only gate on `conflict_present`
2. prompt-EOS residual capture
3. residual conflict probe
4. cross-dimension transfer probe
5. report packaging

Behavior sanity should still happen before capture, but that remains outside
the first `pipelines_v2` workflow skeleton for now.

## Behavior sanity plan

Before any expensive capture, Phase 09 should run a small balanced behavior
smoke covering:

- both dimensions
- both strategy directions
- aligned / edge / strong-conflict rows
- at least two lexical variants per cell

The smoke should explicitly score:

- valid JSON rate
- exact expected rate
- strategy-following vs setting-following vs neither

Capture should only proceed if the behavior smoke shows that:

- activity settings reliably gate entry
- trade-size settings reliably gate size once entry is allowed
- edge rows are not dominating errors in a way that makes the binary target
  incoherent

Later cleanup work sharpened this further:

- `trade_size` is clean enough for primary benchmarking
- `trading_activity` is mostly clean, but the `value=1 + exceptional` cell is
  boundary-sensitive enough that it is now treated as stress-test-only

## Why `pipelines_v2`

We are moving this phase onto `pipelines_v2` because the new library now
supports:

- checked-in Python workflow definitions
- run planning
- run history
- resume / rerun-step / rerun-from-step
- deferred Postgres datasets
- artifact-bound text baselines and probes

That is a better fit for iterative benchmark work than the old one-off
scripts.

## Open items

Phase 09 is not done after scaffolding. The next instance still needs to:

1. refine the descriptive market templates
2. tighten the activity threshold semantics
3. publish the rebuilt dataset table
4. run the text gate
5. run a balanced behavior smoke
6. only then start expensive capture
