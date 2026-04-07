# Phase 6 Profile Invariance Notes

Date: 2026-03-21

## Question

After the Phase 5 symbol-permutation control, does a profile-level market abstraction survive a harder invariance setting that combines:

- symbol alias changes
- row-order permutation
- row-surface/style variation

And if it weakens, which nuisance is the main failure mode?

## Dataset

- Phase: `phase6_profile_invariance_v1`
- Context: `market_only`
- Captures: `48`
- Families:
  - `participation_concentration_tiebreak`
  - `momentum_flow_tiebreak`
- Construction:
  - `4` surface styles
  - `6` row-layout permutations
  - `2` scenario families

## Main Findings

### 1. Primitive factor decode remains trivial

Best held-out primitive regression remains extremely high:

- `pct_5m`: `row_mean @ L1`, `R² 0.9997`
- `net_flow_5m`: `row_mean @ L1`, `R² 0.9996`
- `unique_traders_5m`: `row_mean @ L1`, `R² 0.9991`
- `top20_holder_pct`: `row_mean @ L1`, `R² 0.9998`
- `attractiveness_score`: `row_mean @ L1`, `R² 0.9998`
- `risk_adjusted_score`: `row_mean @ L1`, `R² 0.9998`

Interpretation:

- the harder invariance slice does not challenge primitive-factor representation
- the challenge is higher-order abstraction, not raw factor storage

### 2. Full profile-control survival is modest, but still favors participation/concentration

Best full-control results:

- `momentum_flow_tiebreak`: `row_eos @ L43`, margin `0.0203`, NN accuracy `0.625`
- `participation_concentration_tiebreak`: `row_eos @ L16`, margin `0.0322`, NN accuracy `0.7708`

Interpretation:

- the Phase 5 participation/concentration advantage survives the harder Phase 6 control
- but the effect is much smaller once all nuisance factors are combined

### 3. The main failure mode is layout, not surface style

Best style-only retrieval:

- `momentum_flow_tiebreak`: `row_eos @ L25`, margin `0.1176`, NN accuracy `1.0`
- `participation_concentration_tiebreak`: `row_eos @ L15`, margin `0.1208`, NN accuracy `1.0`

Best layout-only retrieval:

- `momentum_flow_tiebreak`: `row_eos @ L4`, margin `0.0064`, NN accuracy `0.6667`
- `participation_concentration_tiebreak`: `row_eos @ L21`, margin `0.0108`, NN accuracy `0.7188`

Interpretation:

- changing wording / format / symbol alias barely hurts profile retrieval
- moving profiles across row layouts is what mostly breaks invariance
- participation/concentration still has the stronger layout-sensitive signal

## Updated Read

The current best representation story is:

- the model preserves primitive market factors very explicitly
- profile-level abstraction is not globally robust
- participation/concentration remains the strongest abstraction candidate
- the abstraction failure is driven much more by row-layout sensitivity than by surface wording sensitivity

## Best Next Step

Run a targeted layout-invariance phase next:

- keep wording/style fixed
- vary roster position and distractor composition more aggressively
- test whether profile retrieval can be improved by using row-difference / pairwise-relative representations instead of raw row retrieval

That is more useful than adding more paraphrase-only variants, because Phase 6 already shows paraphrase/surface variation is not the primary bottleneck.
