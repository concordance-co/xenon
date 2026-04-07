# Conflict Probe Dataset v1

## Why v0 underperformed

The current `conflict_probe_examples_v0` dataset is too small and too cleanly
templated to give the probe many chances to learn a general conflict feature.

Observed issues from the live table:

- Only `375` rows total.
- Only `140` conflict rows.
- Only `10` rows at the strongest conflict level (`conflict_strength = 4`).
- Only `5` base prompts / market contexts.
- Only `3` distinct prompt prefixes in the first 500 user-prompt characters.
- Every strategy is marked `high` priority.
- Most strategy templates only realize strengths `0, 1, 2`; only
  `diversify_top5` and `partial_profits` ever hit `4`.
- The dataset is dominated by near-duplicate lexical surface forms:
  15 fixed strategy strings x 5 base prompts x 5 slider settings.

This creates two failure modes:

1. The probe can overfit lexical or prompt-template regularities instead of
   conflict.
2. Many rows are "theoretically conflicting" but may not cause the model to
   behave differently, which means there may be no usable signal to detect.

## Design goals for v1

1. Increase the number of strong conflicts, not just total rows.
2. Increase prompt diversity across base market contexts.
3. Split by lexical family so success means semantic generalization.
4. Stop paying for low-information rows that are unlikely to move behavior.
5. Preserve some graded sweeps for analysis, but do not make the whole dataset
   a full 1..5 sweep.

## Recommended dataset shape

### 1. Use semantic families plus lexical variants

Replace the single fixed string per strategy with semantic families and multiple
paraphrases per family.

Suggested family structure:

- `trade_size_force_large`
- `trade_size_force_small`
- `activity_force_trade`
- `activity_force_observe`
- `holding_force_hold`
- `holding_force_exit`
- `diversification_force_concentrate`
- `diversification_force_spread`
- `risk_force_safe`
- `risk_force_degen`

For each family, write `3-5` paraphrases with different lexical surface forms:

- imperative: "Go all-in immediately."
- policy framing: "Capital should be deployed in maximum size."
- prohibition framing: "Do not split capital across names."
- motivational framing: "Concentrate into the best idea."

Every example should carry:

- `strategy_family`
- `strategy_variant_id`
- `lexical_split`

`lexical_split` should be assigned at generation time so train/test can be split
by paraphrase family instead of random row.

### 2. Replace full 1..5 sweeps with targeted severity buckets

Do not sweep all five values for every strategy/context pair by default.

Use three buckets for the main dataset:

- `aligned`
- `edge_conflict`
- `strong_conflict`

For example:

- `trade_size_force_large`
  - aligned: `trade_size in {4, 5}`
  - edge_conflict: `trade_size = 3`
  - strong_conflict: `trade_size in {1, 2}`

- `activity_force_observe`
  - aligned: `trading_activity in {1, 2}`
  - edge_conflict: `trading_activity = 3`
  - strong_conflict: `trading_activity in {4, 5}`

This keeps the contrast you care about while dropping many low-value near-copy
rows.

Recommendation:

- Main training corpus: targeted 3-bucket design.
- Small calibration subset: retain full 1..5 sweeps for `10-20%` of the data.

### 3. Increase base prompt diversity aggressively

Move from `5` base prompts to at least `24-40`.

Each base prompt should be sampled from a different context bucket:

- cash rich vs cash constrained
- no holdings vs concentrated holdings
- winning position vs losing position
- quiet market vs high-volatility market
- strong trend vs mixed market
- reap pressure vs no reap pressure

The key is not just more rows. It is more situations where the same conflict
family appears under different surrounding prompt structure.

### 4. Stop using a single priority regime

v0 uses `high` priority everywhere. That makes the strategy block too uniform.

For v1, vary:

- `high`
- `medium`

Keep `low` optional. The main goal is to avoid teaching the model that conflict
means "high-priority strategy text appears in a fixed format."

### 5. Sample non-swept sliders from realistic configs

Do not hard-code all non-target sliders to one extreme default.

Instead, for each base prompt:

- inherit the source config from the original `interp_examples_v0` row, or
- sample a realistic config from empirical slider combinations.

Then only override the slider(s) needed for the target conflict condition.

This keeps the dataset closer to real prompt geometry and avoids obvious
synthetic artifacts.

### 6. Add a behavioral viability filter before capture

This is the highest-leverage change.

Before running expensive activation capture:

1. Generate model outputs for the aligned and strong-conflict variants.
2. Parse the action (`buy` / `sell` / `observe`, target asset, size).
3. Keep examples where the behavior differs across the contrast.

Examples that do not move behavior are still useful for a separate analysis, but
they should not dominate the probe-training set.

Recommended split:

- `behavior_live = true`: aligned vs conflict changes action or materially
  changes size / asset.
- `behavior_live = false`: no behavioral change.

Primary probe training should use `behavior_live = true` rows first.

## Recommended size

A good first v1 target:

- `30` base prompts
- `10` semantic families
- `3` lexical variants per family
- `3` severity buckets

This gives:

- `30 x 10 x 3 x 3 = 2,700` rows

Optional calibration add-on:

- full 1..5 sweeps for `10` families x `10` contexts = `500` more rows

That lands in the `2.7k-3.2k` range, which is still manageable for capture and
much better than `375`.

## Label structure for v1

Keep the simple binary label, but add richer metadata:

- `conflict_binary`
- `conflict_severity_bucket`
- `conflict_strength_ordinal`
- `conflicting_slider`
- `strategy_family`
- `strategy_variant_id`
- `lexical_split`
- `base_context_id`
- `behavior_live`
- `behavior_delta_type`

This supports:

- binary probe
- ordinal probe
- lexical holdout evaluation
- family holdout evaluation
- behavior-live filtering

## Recommended evaluation splits

Do not use only random row splits.

Add explicit evaluations for:

- lexical holdout:
  train on some paraphrases, test on unseen paraphrases
- context holdout:
  train on some base prompts, test on unseen contexts
- family holdout:
  train on some conflict families, test on unseen families

If the probe only works on random rows, it is still mostly memorizing prompt
surface patterns.

## Concrete v1 build plan

1. Create a new dataset builder, preferably as prompt-confusion-specific code.
2. Generate `~2.7k` rows using the family / paraphrase / severity design.
3. Run cheap behavioral inference on aligned vs strong-conflict pairs.
4. Mark `behavior_live`.
5. Publish the dataset as a new workflow-backed relation.
6. Capture activations on:
   - all `behavior_live = true` rows
   - a smaller matched sample of `behavior_live = false` rows
7. Evaluate:
   - random split
   - lexical split
   - context split
   - family split

## Recommendation

The highest-value change is:

- fewer full sweeps
- more contexts
- more paraphrases
- more strong conflicts
- behavioral prefiltering before activation capture

If we only scale v0 by adding more rows of the same pattern, we will mostly buy
more templated negatives and weak positives, not more useful signal.
