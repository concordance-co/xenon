# Free-Ranch Research Sweep

Date: 2026-03-21

## Goal

Run a broader research scan across the Xenon domain, not just continue the current manifold line. The target is a program where:

1. the core hypothesis is crisp,
2. the main behavior is natural in the prompt and common in the logs,
3. a synthetic dataset can isolate the variables cleanly,
4. and the synthetic result can be validated against real DX-style prompts.

## Domain Audit

Live DB facts that matter for question selection:

- `decision_capture_base_v1` has `121,352` rows.
- Tool mix is heavily observe-dominant:
  - `record_observation`: `103,476`
  - `buy_token`: `11,179`
  - `sell_token`: `6,697`
- Actionability regimes are strongly imbalanced but still diverse:
  - `zero_eth=f, buy=t, sell=t`: `92,244`
  - `zero_eth=t, buy=f, sell=t`: `15,116`
  - `zero_eth=f, buy=t, sell=f`: `9,660`
- Policy-tension observe cases are abundant and mostly fully actionable:
  - `buy=t, sell=t`: `28,500`
  - `buy=f, sell=t`: `1,373`
  - `buy=t, sell=f`: `242`
- Blocked-observe cases are abundant but dominated by a noisy bucket:
  - `high_strategy_present`: `53,559`
  - `strategy_blocks_both`: `2,466`
  - `strategy_blocks_buys`: `1,998`
  - `strategy_blocks_sells`: `1,445`
- High-level strategy structure is common:
  - no high strategies: `46,284`
  - one or more high strategies: the remainder, with many recurring profiles
  - immediate / triggered high-action profiles are present but smaller:
    - `imm=1, trig=0`: `11,379`
    - `imm=0, trig=1`: `3,505`
- Extreme settings are not rare:
  - `extreme_settings_count=5`: `30,036`
  - `extreme_settings_count=4`: `14,254`
  - `extreme_settings_count=3`: `15,265`
- Memory is probably a poor first research target because it is nearly always deep:
  - `memory_depth >= 11`: `114,087`

## Candidate Questions

### 1. Preference vs Permission Algebra

Question:
Does the model form a stable market preference first, then map it through permissions, strategy overrides, and settings into an executable action?

Why it is attractive:

- It directly connects to the most important behavioral question: what the model wants to do versus what it is allowed to do.
- The prompt semantics are explicit and natural.
- The current real-data results already hint at it:
  - top buy/sell asset identity stayed fixed across the corrected settings reruns,
  - while downstream action state moved.
- It is easy to isolate synthetically by holding the market rows fixed and changing only policy text.

Synthetic path:

- fixed market preference
- vary permission mode, strategy override, and risk mode
- label:
  - market-best asset
  - policy-best asset
  - final action type
  - final action asset

Real validation path:

- settings-twist reruns
- direct block-mode reruns (`strategy_blocks_buys`, `strategy_blocks_sells`, `strategy_blocks_both`)

### 2. Strategy Priority Compliance

Question:
Does the model implement the explicit HIGH-strategy > settings ordering as a separable mechanism?

Why it is interesting:

- Very natural for the prompt.
- Many real examples contain high-priority strategy structure.
- Could yield a crisp override circuit if isolated well.

Why it is not first:

- It is probably a special case of the broader preference-vs-permission algebra.
- Focusing only on strategy hierarchy risks missing the affordance side.

### 3. Direct Block-Type Latent Valence

Question:
Do direct block modes (`buys blocked`, `sells blocked`, `both blocked`) hide stable bullish / bearish states better than the generic blocked-observe pool?

Why it is interesting:

- The current generic blocked pool underperformed.
- The DB shows thousands of more specific block cases.

Why it is not first:

- On its own it is narrower and less mechanistic than the preference-vs-permission decomposition.
- Best treated as the real-data validation leg of candidate 1.

### 4. Per-Slider Settings Semantics

Question:
Do the five sliders each produce distinct downstream changes, or are they mostly one coarse “risk appetite” axis?

Why it is interesting:

- Extreme settings are abundant.
- The current all-1 / all-5 reruns show real downstream movement.

Why it is not first:

- It is narrower than the broader policy algebra.
- Better pursued after the base decomposition is clearer.

### 5. Hold / Portfolio Gating

Question:
How much of final action is driven by portfolio state, hold-floor logic, and sellability constraints rather than market preference?

Why it is interesting:

- Portfolio token counts are broad.
- Sell-only and buy-blocked regimes are clearly present.

Why it is not first:

- It is another slice of permission algebra, not a broader story by itself.

### 6. Triggered Sell and Immediate-Action Strategies

Question:
Are immediate / triggered action strategies represented distinctly from passive restrictions?

Why it is interesting:

- The DB has thousands of triggered or immediate profiles.

Why it is not first:

- Smaller and more heterogeneous.
- Probably better as a second-wave policy study.

### 7. Observe Taxonomy

Question:
Can observe be decomposed into neutral, blocked, waiting, and uncertainty states?

Why it is interesting:

- Observe dominates the corpus.

Why it is not first:

- Very likely too broad and noisy as a starting point.
- Better to decompose it via permission algebra first.

### 8. Memory Inertia

Question:
How strongly do previous decisions and recent memory bias hold / re-entry / observe behavior?

Why it is interesting:

- Potentially important for agent behavior over time.

Why it is not first:

- The real dataset is almost always at high memory depth, so clean counterfactual isolation is harder.

### 9. Sell-Side Asymmetry

Question:
Is bearish / sell behavior mechanistically distinct from bullish / buy behavior, or mainly a permission / portfolio asymmetry?

Why it is interesting:

- Sell-side was the weakest part of the earlier decision-structure work.

Why it is not first:

- Important, but better answered after the policy algebra is explicit.

### 10. Token Identity Priors vs Abstract Market Reasoning

Question:
How much of real-data asset targeting is just token-level prior versus abstract market-state reasoning?

Why it is interesting:

- It remains an open concern after the early buy-target results.

Why it is not first:

- It is a strong support track, but not the main behavioral decomposition target.

## Ranking

1. Preference vs Permission Algebra
2. Strategy Priority Compliance
3. Direct Block-Type Latent Valence
4. Per-Slider Settings Semantics
5. Hold / Portfolio Gating
6. Triggered Sell and Immediate-Action Strategies
7. Observe Taxonomy
8. Sell-Side Asymmetry
9. Token Identity Priors vs Abstract Market Reasoning
10. Memory Inertia

## Chosen Direction

The best candidate is **Preference vs Permission Algebra**.

Why:

- It is the shortest path to a real decision mechanism.
- It is common in the data and explicit in the prompt.
- It admits a clean synthetic dataset.
- It naturally supports real-data validation through both settings reruns and refined blocked cohorts.
- It generalizes several lower-ranked candidates instead of competing with them.

## Experiment Plan

### Synthetic

Build a synthetic policy dataset where market rows stay fixed while policy text changes:

- `permission_grid`
  - `buy_and_sell`
  - `buy_only`
  - `sell_only`
  - `observe_only`
- `strategy_override_grid`
  - `none`
  - `no_new_buys`
  - `force_sell_held`
  - `force_observe`
- `risk_gate_grid`
  - `low_risk`
  - `high_risk`

Core synthetic labels:

- `market_best_asset`
- `policy_best_asset`
- `expected_action_type`
- `expected_action_asset`
- `scenario_group`

### Real Validation

Two real-data validation legs:

1. reuse corrected settings-twist results to test stable asset preference vs moving downstream action state
2. run a refined direct block-mode rerun on:
   - `strategy_blocks_buys`
   - `strategy_blocks_sells`
   - `strategy_blocks_both`

## Running Notes

### 2026-03-21

- Built and uploaded `policy_algebra_v1` to `synthetic_market_examples_v0`
- Dataset size:
  - `120` prompts
  - `48` permission-grid
  - `48` strategy-override-grid
  - `24` risk-gate-grid
- First H200 smoke stalled at startup with no logs or metadata writes.
- A100 smoke with `--max-model-len 16384` succeeded cleanly on the first 8 prompts.
- Full synthetic capture completed successfully:
  - `120 / 120` prompts captured
  - dedicated synthetic Modal volume
  - A100 fallback worked cleanly after H200 startup stalls
- Sharded synthetic structure pooling completed and produced the pooled residual files used by analysis.
- The first synthetic policy-algebra analysis finished successfully after a small Modal `scikit-learn` API compatibility patch.

## Findings From The Chosen Direction

### Synthetic Result: `policy_algebra_v1`

The synthetic result is intentionally simple, but it does cleanly test the algebraic decomposition.

Strongly supported:

- `market_best_asset` is perfectly recoverable from the row states across the full layer range.
  - best: `row_mean`, layer `0`, `AUROC=1.0`, `hit@1=1.0`
  - `row_eos` is also perfect across all layers
- `permission_mode` is perfectly recoverable from `active_settings_eos`.
  - best: `active_settings_eos`, layer `0`, `accuracy=1.0`
- `expected_action_type` is perfectly recoverable from the settings/action section.
  - best: `active_settings_eos`, layer `0`, `accuracy=1.0`
- `policy_best_asset` is *not* best read directly from the settings section.
  - best settings-only read: `active_settings_eos`, layer `1`, `accuracy=0.833`
  - best downstream read: `last_token`, layer `27`, `accuracy=0.967`

Interpretation:

- The model can keep the market preference fixed while permission / policy information changes the executable action.
- The choice of *what the market likes* is already stably available in the row states.
- The choice of *what should actually be done under policy* sharpens later than the raw permission labels themselves.

### Repeated-Split Robustness

The synthetic slice is easy, so repeated splits matter more than single best numbers.

- `permission_top_symbol_invariance`
  - mean `1.0`
  - std `0.0`
- `strategy_top_symbol_invariance`
  - mean `1.0`
  - std `0.0`
- `risk_pair_policy_accuracy`
  - mean `0.416`
  - std `0.368`
  - unstable across repeated splits

Interpretation:

- The *permission* and *strategy* branches are robust.
- The *risk gate* branch is not yet robust, which likely reflects a synthetic design problem rather than a decisive failure of the overall preference-vs-permission framing.
- So the chosen direction looks correct, but the synthetic suite should treat risk gating as a second-wave refinement, not as proof that the entire algebra is already solved.

### Real-Data Validation Status

Existing corrected real-data reruns are directionally aligned with the synthetic result.

- In `blocked_valence + settings twist v2`, top buy/sell asset identity stayed fixed across all settings triplets.
- But downstream action state still moved on a meaningful minority:
  - `17 / 120` settings valence flips
  - `13 / 120` strong trade-probability shifts
- Generic blocked-observe remains noisy:
  - only `3 / 34` blocked pairs reveal directional valence after strategy clearing

Interpretation:

- Real data supports the same qualitative split:
  - stable preference
  - downstream permission / policy movement
- The generic blocked pool is still too noisy to serve as the main validation set.

## Updated Recommendation

The best research track remains **Preference vs Permission Algebra**, but with a narrower emphasis:

1. permission and strategy overrides are the strongest immediate target,
2. risk gating should be redesigned synthetically before it is used as a central claim,
3. real-data follow-up should target direct block modes rather than generic `high_strategy_present`.

## Next Concrete Steps

1. Build `policy_algebra_v2` with a harder risk branch.
   - Reduce lexical shortcutting.
   - Make safe-vs-risky preference require more genuine composition of row features.
2. Add a synthetic “distributed policy text” variant.
   - Avoid having all policy information concentrated in one settings block.
3. Run a real direct-block rerun.
   - prioritize:
     - `strategy_blocks_buys`
     - `strategy_blocks_sells`
     - `strategy_blocks_both`
4. Keep the current corrected settings-twist rerun as the main real validation anchor until the direct-block cohort is captured.
