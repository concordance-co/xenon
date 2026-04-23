# Prompt Confusion Phase 08 Design

## Goal

Phase 08 is a relational redesign for the part of the prompt-confusion
story that still looks real:

- conflict detection
- lexical-holdout transfer
- depth-progressive signal formation

It is **not** trying to rescue `family` as an interpretable variable.

## Why a new design is needed

Phase 05 and Phase 07 jointly suggest that:

- `family` was mostly a proxy for semantic polarity on a target dimension
- raw text could decode that variable because the prompt faithfully stated
  the strategy direction and the active setting dimension
- more shell harmonization does not solve this, because the content itself
  carries the label

So the benchmark should stop asking:

- can the model represent `family`?

and instead ask:

- can the model represent whether two policy sources agree or disagree?

## Core idea

Conflict becomes a crossed relational variable.

For each dimension:

- `strategy_direction` varies within the dataset
- `setting_value` varies within the dataset
- `setting_implied_direction` is derived from the setting value
- `conflict_present` is true iff the strategy direction and setting-implied
  direction disagree on canonical aligned/conflict rows
- `edge_conflict` rows are retained in the dataset but excluded from the
  primary binary target

This means a token like `large` or `small` appears in both:

- aligned rows
- conflict rows

The conflict label is therefore a relation between two spans, not a
unigram-level property of the prompt.

## Dimensions

Phase 08 keeps two dimensions:

1. `trade_size`
   - strategy direction in `{small, large}`
   - setting value in `{1,2,3,4,5}`
   - implied direction:
     - `1,2 -> small`
     - `3 -> medium / edge`
     - `4,5 -> large`

2. `trading_activity`
   - strategy direction in `{observe, trade}`
   - setting value in `{1,2,3,4,5}`
   - implied direction depends on pressure bucket, as in prior phases

This gives replication across two policy dimensions without using
family-locked semantics.

## Row structure

Each row should include:

- `target_dimension`
- `strategy_direction`
- `setting_value`
- `setting_implied_direction`
- `conflict_present`
- `edge_conflict`
- `conflict_strength`
- `conflict_band`
- `matched_group_id`
- `matched_pair_id` for canonical aligned-vs-strong rows
- strategy template id
- settings template id
- context variant id

## Prompt structure

### System

Keep the Phase 07 framing:

- `STRATEGY` = directional plan
- `SETTINGS` = execution policy constraints
- if they disagree, `SETTINGS` constrain final execution

### User

Keep the same overall shape:

- `TASK`
- `STRATEGY`
- `ACTIVE SETTINGS`
- `PORTFOLIO`
- `MARKET`

### Important prompt design rule

The strategy text must no longer define a dataset family.

Instead:

- both strategy directions for a dimension must appear across the same
  lexical shells
- both directions must appear in both aligned and conflict rows

Example for size:

- strategy: `use the large size tier when a trade is taken`
- settings: `Trade Size: 5/5`
  - aligned
- settings: `Trade Size: 1/5`
  - conflict

And also:

- strategy: `use the small size tier when a trade is taken`
- settings: `Trade Size: 1/5`
  - aligned
- settings: `Trade Size: 5/5`
  - conflict

So `large` and `small` are balanced across labels.

The same principle applies to activity:

- strategy says `trade`
- settings imply `trade`
  - aligned
- strategy says `trade`
- settings imply `observe`
  - conflict
- strategy says `observe`
- settings imply `observe`
  - aligned
- strategy says `observe`
- settings imply `trade`
  - conflict

## Contexts and nuisance settings

Phase 08 keeps the useful Phase 07 improvements:

- full settings block always present
- nuisance settings vary instead of being pinned to `3`
- contexts are shared and repeated within pressure bucket

But nuisance values should be chosen so they do not reveal the target
dimension through obvious range restrictions.

Phase 08 should therefore draw nuisance settings from the same broad range
across dimensions, ideally the full `1..5` support unless a specific value
would change the task semantics in a way we do not want.

Recommended default:

- nuisance `Trading Activity` values sampled from `1..5`
- nuisance `Trade Size` values sampled from `1..5`
- nuisance `Risk`, `Holding`, `Diversification` values sampled from `1..5`

The nuisance values should be deterministic within a `matched_group_id` so
the only intended moving axis inside a group is `setting_value`.

The only intended label signal should be the relationship between
`strategy_direction` and `setting_implied_direction`.

## Value=3 handling

The middle setting value needs explicit treatment.

For `trade_size`, `setting_value=3` implies `medium`, which agrees with
neither `small` nor `large`. If we label these rows as ordinary conflict,
the tokens `3/5` and `standard size tier` become a bounded lexical shortcut.

So Phase 08 should:

- keep `setting_value=3` rows in the dataset
- mark them as `conflict_band = edge`
- set `edge_conflict = true`
- set primary binary `conflict_present = NULL` on these rows

The primary binary gate and primary probe target should operate on the
canonical subset:

- aligned rows
- strong conflict rows

Edge rows remain available for secondary graded analyses.

For `trading_activity`, `setting_value=3` can still be retained and labeled
according to the pressure-bucket logic, but the spec should keep the same
`edge_conflict` field so the builder can represent ambiguity consistently.

## Grouping

Group structure should be explicit.

A `matched_group_id` fixes:

- `target_dimension`
- `strategy_direction`
- `strategy_template`
- `settings_template`
- `context_variant_id`
- all nuisance setting values

and sweeps only:

- `setting_value`

So each group contains 5 rows.

A canonical `matched_pair_id` should pair the most-aligned and
most-conflicted rows inside each group:

- for size: the two extreme values relative to `strategy_direction`
- for activity: the most clearly aligned vs most clearly conflicted rows
  under the current pressure bucket

This preserves pair-based analyses without pretending the whole 5-row
sweep is a pair dataset.

## Lexical split

The lexical split should also be explicit.

Recommended default:

- `strategy_template v0,v1 -> train`
- `strategy_template v2,v3 -> test`
- `settings_template v0,v1 -> train`
- `settings_template v2,v3 -> test`

Both strategy directions must appear in both split halves.

That should happen by construction because `strategy_direction` is crossed
with template choice rather than nested inside it, but it should be stated
as an invariant in the builder and checked in dataset summaries.

## Lexical gate

The pre-capture gate changes.

Old gate:

- `user_text -> strategy_family`

New gate:

- `user_text -> conflict_present`

Expectation:

- CountVectorizer + LogisticRegression should be near chance
- if balanced accuracy is much above `0.55`, something is wrong in the
  generator

Primary gate:

- run on the canonical binary subset where `conflict_present IS NOT NULL`

Recommended reporting:

- pooled gate on all canonical rows
- per-dimension gates for `trade_size` and `trading_activity`

Optional secondary checks:

- `user_text -> target_dimension`
- `user_text -> strategy_direction`

These may remain decodable and are not themselves disqualifying.
The critical point is that `conflict_present` should no longer be a
bag-of-words property.

## Behavioral sanity

Behavior comes before interpretability.

Before capture:

1. run the lexical gate on `conflict_present`
2. run a small balanced behavior slice across:
   - both dimensions
   - both strategy directions
   - aligned, edge, and strong-conflict rows
3. verify the model usually follows the settings-constrained output

If the model does not behaviorally honor the synthetic semantics, stop and
repair before probing.

Recommended smoke target:

- at least a few rows per cell of
  `(target_dimension, strategy_direction, conflict_band)`

Recommended acceptance rule:

- the model should follow the settings-constrained expected output on a
  clear majority of canonical conflict rows before activation capture

## Analysis note

Phase 08 does not inherit the old family-residualization logic from
Phase 05.

That control existed because `family` was a major confound. In Phase 08,
the corresponding control is:

- the BoW gate on `conflict_present`
- plus the relational crossed design itself

Writeups should make that substitution explicit.

## Intended next steps

1. implement the builder with the crossed relational design
2. build and inspect the dataset locally
3. run the raw-text `conflict_present` gate on the canonical subset
4. run a small behavior sanity slice
5. if both look healthy:
   - upload to Neon
   - publish workflow
   - capture prompt-only activations
   - rerun the existing conflict-detection analysis stack
