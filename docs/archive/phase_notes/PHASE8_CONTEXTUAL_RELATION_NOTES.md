# Synthetic Market Phase 8: Contextual Relation

## Purpose

Phase 8 tests a harder relation-first hypothesis than Phase 7.

Instead of letting the anchor-pair relation move alongside obvious raw-factor deltas, this phase:

- keeps the anchor pair numerically fixed
- changes the surrounding roster so the anchor pair moves through different contextual rank regimes
- preserves the same nuisance axes as Phase 7:
  - style
  - layout
  - roster
  - global magnitude scale

The question is no longer “can the model preserve a simple anchor-pair relation?” but “does the relation survive when the surrounding market context changes what that pair means?”

## Dataset

- `384` market-only prompts
- `4` scenario families
  - `generic_duel_context`
  - `momentum_shadow_context`
  - `flow_shadow_context`
  - `paired_cluster_context`
- `2` surface styles
- `4` layouts
- `4` contextual roster variants
- `3` global magnitude scales

## Main findings

### 1. Primitive factors are still trivial

Primitive row factors remain almost perfectly explicit:

- `pct_5m`: `R² 0.999917` at `row_mean @ L1`
- `net_flow_5m`: `R² 0.999907` at `row_mean @ L1`
- `unique_traders_5m`: `R² 0.999913` at `row_mean @ L1`
- `top20_holder_pct`: `R² 0.999897` at `row_mean @ L1`
- `attractiveness_score`: `R² 0.999926` at `row_mean @ L1`
- `risk_adjusted_score`: `R² 0.999925` at `row_mean @ L1`

So the model still clearly sees the market factors.

### 2. Direct focal pairwise labels are still trivial

The synthetic pair labels remain easy:

- `a_beats_b_on_attractiveness`: `AUROC 1.0` at `row_mean @ L0`
- `a_beats_b_on_risk_adjusted`: `AUROC 1.0` at `row_mean @ L0`

That means Phase 8 did **not** make the direct comparison task hard.

### 3. Contextual relation identity mostly collapses

The stronger Phase 8 target was relation identity under contextual pressure.

This is the important result: it is mostly weak.

Best relation margins by scenario:

- `paired_cluster_context`
  - best relation margin: `0.032727`
  - best mode: `style_only`
  - best state: `row_mean @ L47`
- `generic_duel_context`
  - best relation margin: `0.028740`
  - best mode: `style_only`
  - best state: `row_mean @ L0`
- `momentum_shadow_context`
  - best relation margin: `0.021264`
  - best mode: `style_only`
  - best state: `row_mean @ L47`
- `flow_shadow_context`
  - best relation margin: `0.010645`
  - best mode: `style_only`
  - best state: `row_mean @ L44`

These are much smaller than Phase 7.

### 4. Layout and contextual controls are the real failure modes

The strongest degradation shows up in the harder control axes:

- `layout_only`
  - `flow_shadow_context`: `-0.001644`
  - `momentum_shadow_context`: `-0.001990`
  - `generic_duel_context`: `0.006379`
  - `paired_cluster_context`: `0.005041`
- `roster_only`
  - mostly near zero except `paired_cluster_context`
  - `paired_cluster_context`: `0.023692`
- `rank_ctrl`
  - near zero for three scenarios
  - only `paired_cluster_context` reaches `0.004591`
- `scale_ctrl`
  - weak or negative in three scenarios
  - only `paired_cluster_context` reaches `0.013974`

So the contextual market meaning of the pair is not being preserved cleanly in most families.

## Interpretation

Phase 7 was real, but too easy. Phase 8 shows why.

What survives:

- primitive market factors
- direct pairwise preference labels

What does **not** broadly survive:

- a stable anchor-pair relation identity once the surrounding roster changes what the pair means

That means the research should not treat “anchor-pair relation retrieval” as the final market representation object.

## Best reading

The cleanest reading is:

- the model has very explicit local market-factor representations
- it can make direct pairwise comparisons easily
- but the broader contextual meaning of a fixed pair is not represented as one robust invariant object across harder market contexts

`paired_cluster_context` is the one promising exception. That family likely deserves follow-up because it is the only case where contextual relation survives roster pressure by a nontrivial margin.

## Next steps

The next experiment should stop asking whether a fixed pair has a globally stable identity.

Better directions:

1. Set-level market geometry
- compare whole-snapshot structure, not just one pair

2. Factor-difference objects
- represent `delta momentum`, `delta participation`, `delta concentration` explicitly
- test whether those difference directions survive contextual changes better than pair identity

3. Context-conditioned relation families
- instead of one fixed pair, ask whether the model preserves a *type* of relation
- e.g. “momentum shadowed by stronger roster context”

4. Focus on `paired_cluster_context`
- it is the only scenario that still shows a meaningful contextual relation signal

