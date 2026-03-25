# Phase 17 Market Axis Decomposition

This phase is a decomposition pass built on top of the existing Phase 15 market-only captures and nuisance-residualized PCA basis.

Relevant artifacts:
- [axis decomposition results](/Users/brockelmore/concordance/xenon/data/analysis_results/synthetic_market_axis_decomposition/phase15_market_basis_discovery_v1/prompt_visible_v2/results.json)
- [Phase 15 residualized PCA results](/Users/brockelmore/concordance/xenon/data/analysis_results/synthetic_market_discovery/phase15_market_basis_discovery_v1/residualized_nuisance_v1/results.json)
- [analysis code](/Users/brockelmore/concordance/xenon/pipelines/interp/synthetic_market_axis_decomposition_analysis.py)

## Scope

The goal here is narrower than Phase 15:

- take the strongest discovered market axes
- ask what prompt-derived market formulas best explain them
- keep only prompt-visible or prompt-derived market aggregates
- drop nuisance variables from the candidate feature bank
- compare simple formulas, small linear mixtures, and mild nonlinear mixtures

Two target axes were analyzed:

- `leader_axis`: `market_mean`, layer `4`, `PC1`
- `dispersion_axis`: `market_mean`, layer `35`, `PC1`

## Sanity Checks

These passed cleanly:

- target-PC nuisance correlations are near zero after residualization
  - leader axis: `seq_len rho ~= 0.006`
  - dispersion axis: `seq_len rho ~= 0.022`
- shuffled controls collapse close to zero
  - leader: best shuffled single `R² ~= 0.024`, best shuffled quadratic pair `R² ~= 0.061`
  - dispersion: best shuffled single `R² ~= 0.00004`, best shuffled quadratic pair `R² ~= 0.025`
- no candidate feature was degenerate

So the readout below is not being driven by prompt length or the earlier nuisance confound.

## Leader Axis

### Best simple explanations

- Best single prompt-derived feature:
  - `vol_1h_max`
  - `CV R² = 0.459`
- Best linear pair:
  - `pct_1h_cv_abs + vol_5m_range`
  - `CV R² = 0.629`
- Best quadratic pair:
  - `pct_1h_max + vol_5m_max`
  - `CV R² = 0.672`

### Family-level readout

Cross-validated ridge using all 13 aggregates from one metric family:

- `vol_1h`: `R² = 0.621`
- `pct_1h`: `R² = 0.595`
- `vol_5m`: `R² = 0.364`
- `pct_5m`: `R² = 0.298`
- `net_flow_5m`: `R² = 0.213`
- `unique_traders_5m`: `R² = 0.068`
- `top20_holder_pct`: `R² ~= 0`

### Aggregate-type readout

Cross-validated ridge using one aggregate type across all metric families:

- `cv_abs`: `R² = 0.616`
- `median`: `R² = 0.556`
- `max`: `R² = 0.534`
- `min`: `R² = 0.506`
- `mean`: `R² = 0.402`
- `top2_mean`: `R² = 0.367`

### Main interpretation

The leader axis is not well-described as just "highest 1h return."

The strongest explanation is:

- a leader-prominence signal
- carried mainly by `1h` return and `1h`/`5m` volume
- with some sensitivity to relative spread rather than only the absolute maximum

Useful evidence for that:

- the best single feature is `vol_1h_max`, not `pct_1h_max`
- the best family-level fits are `vol_1h` and `pct_1h`, very close together
- the best quadratic pair combines one return-leader term and one volume-leader term
- the single-feature winner is stable across all `5/5` folds: `vol_1h_max`

So the safest statement is:

- the leader axis looks like a **prominent, high-throughput leader** axis
- not a pure "top return" axis

## Dispersion Axis

### Best simple explanations

- Best single prompt-derived feature:
  - `pct_1h_mad`
  - `CV R² = 0.523`
- `pct_1h_std` is weaker:
  - `CV R² = 0.347`
- Best linear pair:
  - `vol_5m_max_minus_rest_mean + vol_1h_top1_minus_median`
  - `CV R² = 0.814`
- Best quadratic pair:
  - `vol_5m_mean + vol_1h_median`
  - `CV R² = 0.843`

### Family-level readout

Cross-validated ridge using all 13 aggregates from one metric family:

- `vol_1h`: `R² = 0.818`
- `pct_1h`: `R² = 0.810`
- `vol_5m`: `R² = 0.779`
- `unique_traders_5m`: `R² = 0.597`
- `net_flow_5m`: `R² = 0.488`
- `top20_holder_pct`: `R² = 0.437`
- `pct_5m`: `R² = 0.362`

### Aggregate-type readout

Cross-validated ridge using one aggregate type across all metric families:

- `mad`: `R² = 0.768`
- `mean`: `R² = 0.759`
- `std`: `R² = 0.728`
- `leader_zscore`: `R² = 0.714`
- `top2_mean`: `R² = 0.690`
- `top1_minus_median`: `R² = 0.666`
- `median`: `R² = 0.593`
- `max_minus_rest_mean`: `R² = 0.503`

### Main interpretation

This axis is not best described as literal standard deviation.

The cleanest single proxy is:

- `pct_1h_mad`

And at the aggregate-family level:

- `mad` beats `std`
- `std` is still strong, but not dominant

So the safest statement is:

- this is an **unevenness / spread / roster-dispersion** axis
- `std` is one useful proxy, but not the best one
- the model seems to care more about **average deviation from the market center** and **leader-versus-rest structure** than about the exact standard-deviation formula

Important caveat:

- the best multivariate fits bring in strong volume terms
- family-level fits show `pct_1h` and `vol_1h` are almost equally powerful
- that suggests the dataset couples return-dispersion and volume structure in a way the axis is using

So this is not evidence that "dispersion is really volume." It is better read as:

- the dispersion axis is multivariate
- with strong return-spread structure
- and strong co-moving volume structure

## Phase 15 Subspace Readout

This is still based on the existing Phase 15 residualized PCA run, which only keeps the top `5` PCs per layer and state.

### `market_mean`

Across layers:

- top-5 cumulative variance:
  - mean `0.523`
  - min `0.464`
  - max `0.661`
- top-5 participation ratio:
  - mean `4.164`
  - min `3.747`
  - max `4.505`

Best-compressed layers by top-5 cumulative variance:

- `L1`: `0.661`
- `L2`: `0.598`
- `L42`: `0.566`
- `L43`: `0.559`
- `L0`: `0.557`

Interpretation:

- `market_mean` is not close to one-dimensional
- even the strongest layers still need multiple PCs
- the market-only pooled market summary looks like a fairly broad `~4D` top-PC subspace, not one dominant axis

### `market_eos`

Across layers:

- top-5 cumulative variance:
  - mean `0.613`
  - min `0.448`
  - max `0.836`
- top-5 participation ratio:
  - mean `3.150`
  - min `2.256`
  - max `4.188`

Best-compressed layers by top-5 cumulative variance:

- `L5`: `0.836`
- `L6`: `0.809`
- `L7`: `0.752`
- `L8`: `0.745`
- `L2`: `0.742`

Interpretation:

- `market_eos` is more compressed than `market_mean`
- especially in early-to-mid layers
- but even here, top-5 variance never reaches `0.90`

So the correct summary is:

- there is real low-dimensional structure
- but not a near-complete collapse into one or two PCs

## Bottom Line

The prompt-derived decomposition supports three claims:

1. The leader axis is a **return-plus-volume leader** axis, not just `pct_1h_max`.
2. The dispersion axis is **not** best described as literal standard deviation.
   - `pct_1h_mad` is the best single visible proxy.
   - aggregate-family results say `mad > std`.
3. The underlying market subspace is multi-dimensional.
   - `market_mean` looks broad and roughly `4D` in its top PCs.
   - `market_eos` is more compressed, but still not close to fully captured by one or two PCs.
