# Market Counting Manifolds Plan

## Purpose

This document defines the next research step after the first decision-structure runs.

The key shift is methodological:

- stop treating `buy/sell/observe` as the primary object of study
- stop relying on weak prompt ablations as the main route to insight
- move to a hypothesis-first, synthetic-data-first program modeled on the workflow in *When Models Manipulate Manifolds: The Geometry of a Counting Task*

The goal is to identify the clean intermediate quantities the model may actually represent when it "sees" a market, and then study those quantities as manifolds before bringing the result back to noisy DX-terminal prompts.

This should not be read as assuming that the *entire* market state reduces to one universal one-dimensional manifold. The intended sequence is:

- first, search for 1D or near-1D structure in isolated scalar factors
- then, search for low-dimensional coupled-factor geometry
- finally, characterize the joint market-state space built from those pieces


## Why This Shift Is Necessary

The current decision-structure work was useful, but it mostly told us **where** action-relevant information lives:

- buy-target is strongly decodable from early row states
- target-asset and sell-target get modest downstream sharpening

That is a staging result, not yet a manifold result.

Recent sanity checks strengthen the case for a change in emphasis:

- the best buy probe is moderately correlated with short-horizon momentum, participation, and flow
- several simple raw-feature heuristics match or beat the probe on top-1 asset selection within held-out buy snapshots

Interpretation:

- the current report is useful as localization
- it does not yet show that the model's market conception is richer than structured salience
- we may be probing a fused downstream quantity rather than the cleaner internal variables the model actually represents

This is exactly the failure mode described in the counting-manifolds paper: probing the wrong fused quantity before understanding the model's cleaner intermediate representation.


## Core Analogy to Counting Manifolds

The linebreak paper decomposes the final decision into a sequence of intermediate quantities:

- current character count
- line width constraint
- characters remaining
- next token length
- final newline decision

The analogous Xenon decomposition is:

- per-asset state
- market-relative attractiveness
- fee-adjusted edge
- risk-adjusted acceptability
- actionability / legality
- final buy / sell / observe decision

The important lesson is that the final decision is probably not the cleanest quantity to probe first.


## Research Question

What quantities does the model represent cleanly when reading a market snapshot, and how are those quantities transformed by settings, portfolio state, and constraints before they become a final trading action?


## Geometric Target

The counting-manifolds analogy is still useful, but in a staged way.

Stage 1 target:

- find individual scalar market variables that may admit clean 1D or near-1D structure when isolated

Stage 2 target:

- find coupled low-dimensional structures for variables that naturally interact, such as momentum + participation or signal + concentration

Stage 3 target:

- characterize the higher-dimensional joint market-state space that combines those simpler pieces

So the 1D search remains a core part of the plan. It is simply no longer the full expected endpoint for the market as a whole.


## Candidate Latent Variables

These are the first variables to test as possible "counting-like" or "market manifold" objects.

Some of these may be best studied as scalar 1D candidates, while others may only become clean when treated as small coupled spaces.

### 1. Asset Attractiveness

A latent scalar or low-dimensional variable encoding how attractive an asset is relative to the rest of the current roster.

Expected ingredients:

- short-horizon momentum
- flow
- participation
- concentration risk
- token age / newness
- reap role

Why it matters:

- this is the best analogue to a clean market-side perceptual variable

### 2. Fee-Adjusted Edge

A latent variable for whether expected upside clears the round-trip fee barrier.

Expected ingredients:

- magnitude of current signal
- estimated continuation vs reversal
- noise / conviction
- fee threshold

Why it matters:

- buy/sell decisions should care about not just attractiveness, but attractiveness net of cost

### 3. Pairwise Preference

A latent relation `A > B` under the current market and policy context.

Expected ingredients:

- relative momentum
- relative flow surprise
- relative participation
- relative risk

Why it matters:

- trading is inherently relative
- pairwise preference may be cleaner than absolute asset quality

### 4. Risk-Adjusted Acceptability

A latent variable for whether an asset is acceptable under current settings, apart from whether it is the best.

Expected ingredients:

- volatility profile
- concentration profile
- token age
- settings such as `asset_risk_preference` and `holding_style`

Why it matters:

- settings may twist the same base market manifold into a different acceptability manifold

### 5. Actionability / Affordance

A latent variable for whether the preferred action is actually feasible now.

Expected ingredients:

- ETH balance
- held vs unheld
- strategy restrictions
- hold floors
- cooldown state
- price impact limits

Why it matters:

- this should be separated from attractiveness rather than mixed into it


## Initial Hypotheses

### H1. The model represents market quality more cleanly than final action.

Prediction:

- `asset attractiveness` and `pairwise preference` will yield cleaner manifolds than `buy/sell/observe`

### H2. The earliest market representation is more relational than absolute.

Prediction:

- pairwise and rank-based labels will explain row geometry better than raw magnitudes alone

### H3. Settings do not create preference from scratch; they reweight an existing market manifold.

Prediction:

- adding settings will rotate, shift, or sharpen a market manifold rather than replace it with an unrelated one

### H4. Actionability is partly orthogonal to attractiveness.

Prediction:

- legality and blocked-action labels will occupy a distinct downstream subspace

### H5. Simple raw metrics explain part, but not all, of the relevant geometry.

Prediction:

- raw-feature baselines will recover some ranking structure
- activation geometry will retain signal after residualizing out basic raw metrics and ranks

### H6. The global market state is likely composed from several simpler geometric pieces rather than one universal 1D manifold.

Prediction:

- some isolated scalar factors will admit cleaner 1D structure than others
- some factors will only become clean in 2D or small coupled spaces
- the joint market representation will be low-dimensional relative to hidden size, but not globally 1D


## Synthetic Dataset Design

The synthetic dataset should not try to simulate all of DX terminal immediately.

It should instead isolate a few latent variables at a time.

### General Principles

- use neutral symbols like `A`, `B`, `C`, `D` instead of real token identities
- start with small rosters, ideally 3 to 6 assets
- control a small number of metrics at first
- sweep systematically rather than randomly
- attach labels by construction, not by post-hoc heuristic
- add downstream context back in gradually

### Phase 1 Prompt Skeleton

Use a minimal prompt with:

- a trading-bot role
- allowed actions
- fee statement
- market snapshot only

No portfolio, no strategies, no prior decisions, no reap section, no real symbols.

The goal is to expose the base market perceptual manifold with minimal confounds.

### Phase 1 Asset Fields

Keep the first version narrow:

- `pct_5m`
- `pct_1h`
- `net_flow_5m`
- `vol_5m`
- `vol_1h`
- `unique_traders_5m`
- `top20_holder_pct`
- `age_bucket`

### Phase 1 Archetypes

Define a small library of interpretable asset archetypes.

Recommended starting set:

- Stable winner
- Short-term momentum burst
- Flow-backed continuation
- Noisy pump
- Fading leader
- Illiquid spike
- Crowded concentration risk
- Fresh but uncertain launch
- Mean-reversion candidate

### Phase 1 Dataset Families

#### A. Scalar Sweeps

Vary one latent factor continuously while holding others fixed.

Examples:

- increase `pct_5m` from negative to positive in small increments
- increase `net_flow_5m` while holding price changes fixed
- vary concentration risk while holding momentum fixed

Purpose:

- test whether a clean 1D manifold exists for individual factors

#### B. Pairwise Tradeoff Grids

Construct two-asset or four-asset grids where one asset wins on one factor and loses on another.

Examples:

- high momentum / weak flow vs moderate momentum / strong flow
- strong participation / high concentration vs weaker participation / low concentration

Purpose:

- test whether the model builds a pairwise preference relation rather than just reading the largest single number

#### C. Archetype Families

Instantiate the same archetype over many small perturbations.

Purpose:

- test whether the model groups assets by behavior type rather than by exact numeric values

#### D. Context Ladder

Start from the same market snapshot and add context back in stepwise:

- Market only
- Market + settings
- Market + portfolio
- Market + constraints
- Market + strategies
- Full DX-style prompt

Purpose:

- cleanly measure how each context block transforms the base market manifold

#### E. Confusion Sets

Construct cases where the highest single metric should not win.

Examples:

- highest `pct_5m` but terrible participation and high concentration
- best flow but obviously below fee-adjusted edge threshold
- strong raw market signal but blocked by actionability

Purpose:

- directly challenge the "big numbers = buy" story


## Labels by Construction

The synthetic dataset should include labels that are known because we designed the prompt, not because we inferred them afterward.

### Asset-Level Labels

- `true_archetype`
- `attractiveness_rank`
- `risk_adjusted_rank`
- `pairwise_preference_to_each_other_asset`
- `edge_gt_fee`
- `acceptable_under_risk_setting`
- `buyable_if_unconstrained`
- `sellable_if_held`

### Snapshot-Level Labels

- `best_asset`
- `buy_any`
- `observe_vs_act`
- `blocked_preference_present`
- `which_context_block_changed_the_outcome`

### Context Ladder Labels

For each base market state, define:

- `market_only_best_asset`
- `settings_adjusted_best_asset`
- `portfolio_adjusted_best_asset`
- `final_action`

This makes the pre/post transformation measurable by construction.


## Capture Strategy

Use the same pooled positions we already built, but apply them to synthetic prompts.

### Pre-Context Positions

- `row_mean_i`
- `row_eos_i`
- `market_mean`
- `market_eos`

### Post-Context Positions

- `active_settings_eos`
- `portfolio_eos`
- `constraints_eos`
- `active_strategies_eos`
- `prev_decisions_eos`
- `last_token`

### Routing

Keep router indices/logits wherever possible.

The paper's lesson is relevant here:

- discrete features and geometric structure are dual views
- routing may expose the model's computational partition of market concepts


## Exact Experiment Program

### Experiment 1. Single-Factor Manifold Discovery

Question:

- does the model represent a clean manifold for one scalar market factor at a time?

Method:

- use scalar sweeps for one factor while holding all else fixed
- compute mean activations by sweep step
- run PCA / low-rank approximation
- train probes for the scalar value and neighboring bins
- test whether the resulting representation has rippled / curved structure

Validation:

- high probe performance on the intended variable
- low intrinsic dimension relative to the full hidden size
- smooth neighborhood structure across adjacent sweep values
- causal patching along the manifold should move outputs monotonically

Failure condition:

- the representation is indistinguishable from raw lexical scale or does not vary smoothly

### Experiment 2. Pairwise Preference Geometry

Question:

- does the model represent `A > B` cleanly, even when no single raw metric dominates?

Method:

- use pairwise tradeoff grids
- label pairwise winner by construction
- probe on differences or concatenations of row states
- compare activation performance to raw-metric baselines

Validation:

- pairwise probes beat simple single-metric rules on confusion sets
- the same pairwise direction transfers across multiple archetypes

Failure condition:

- preference collapses to whichever asset has the maximum simple metric

### Experiment 3. Coupled-Factor Geometry

Question:

- do important market variables become cleaner when studied as small coupled spaces rather than as isolated 1D sweeps?

Method:

- generate 2D sweeps such as:
  - momentum × participation
  - signal × concentration
  - flow × fee-adjusted edge
- compute PCA / local dimension / neighborhood preservation on the resulting grids
- compare against the corresponding 1D isolated sweeps

Validation:

- coupled-factor geometry explains more variance or cleaner neighborhood structure than the corresponding isolated scalar sweeps
- the geometry is stable across prompt templates and background rosters

Failure condition:

- the 2D or small coupled space adds no clarity beyond the isolated scalars

### Experiment 4. Archetype Manifolds

Question:

- does the model cluster assets by behavioral archetype rather than by exact numeric identity?

Method:

- generate many perturbed examples of each archetype
- run RSA against:
  - raw metric space
  - rank space
  - archetype space
- inspect row-state geometry and neighborhood preservation

Validation:

- archetype labels explain row geometry better than raw metrics alone in deeper layers

Failure condition:

- geometry stays almost entirely aligned with raw numeric ordering

### Experiment 5. Settings as Twists or Reweightings

Question:

- does adding settings twist the same market manifold, or replace it?

Method:

- use the context ladder on identical market states
- compare pre and post representations with:
  - CKA
  - subspace overlap
  - Procrustes alignment
  - drift decomposition into parallel vs orthogonal movement

Validation:

- meaningful but structured post shift
- some preserved geometry under alignment
- predictable changes in asset rank under different settings

Failure condition:

- either no change at all or fully unrelated post geometry

### Experiment 6. Affordance Separation

Question:

- is actionability represented separately from attractiveness?

Method:

- take the same base market state
- vary only ETH balance, hold state, or restriction status
- compare row and downstream states

Validation:

- attractiveness probes remain stable pre-context
- downstream legality probes change strongly
- blocked-preference cases preserve target preference while changing final action

Failure condition:

- legality and attractiveness are inseparable even in controlled synthetic settings

### Experiment 7. Joint Market-State Geometry

Question:

- once the cleanest scalar and coupled-factor candidates are known, what does the joint market-state space look like?

Method:

- construct snapshots spanning combinations of the best-supported scalar and coupled-factor variables
- measure effective dimension, RSA, clustering, and neighborhood structure at the full-snapshot level
- compare against raw-metric, rank, and archetype baselines

Validation:

- the full market state is low-dimensional relative to hidden size
- the geometry is better explained by latent factor structure than by raw lexical or single-metric ordering

Failure condition:

- the joint space is too entangled to improve on raw-metric baselines even after the lower-level factors are characterized

### Experiment 8. Transfer Back to DX Data

Question:

- do synthetic manifold directions transfer to real noisy prompts?

Method:

- train probes or subspace readouts on synthetic data
- evaluate them on real DX decision-structure captures
- focus on:
  - trade ticks
  - blocked observe cases
  - policy-tension cases

Validation:

- synthetic directions recover meaningful signal on real data
- especially on confusion and blocked cases

Failure condition:

- strong synthetic result with no transfer to real DX prompts


## Minimal Validation Criteria

We should treat a synthetic market variable as "real" only if it passes multiple lenses.

### Representation Criteria

- high held-out decode performance
- smooth local geometry across adjacent values
- low-dimensional structure in PCA or related embeddings
- stable behavior across paraphrases or prompt templates

### Causal Criteria

- subspace ablation changes the downstream readout selectively
- mean-vector substitution or manifold patching moves the model's preference in the predicted direction
- confusion-set interventions break the correct behavior in interpretable ways

### Baseline Criteria

- activation-based readout must beat:
  - single raw metrics
  - within-snapshot ranks
  - simple multivariate raw-feature baselines

If it does not, then the result is probably not yet beyond "big numbers = good buy."


## Implementation Sequence

### Phase 0. Hypothesis Lock

Deliverable:

- finalize the first 3 to 5 latent variables
- finalize the first 6 to 10 asset archetypes

### Phase 1. Synthetic Generator

Deliverable:

- market-only prompt generator with neutral symbols
- scalar sweeps
- pairwise tradeoff grids
- archetype family generator
- labels by construction

### Phase 2. Market-Only Captures

Deliverable:

- capture a clean synthetic activation set
- focus on row states and market section first

### Phase 3. Geometry Analysis

Deliverable:

- PCA / intrinsic dimension for isolated scalar sweeps
- RSA against raw, rank, pairwise, and archetype spaces
- first 1D manifold candidates

### Phase 4. Coupled-Factor Geometry

Deliverable:

- targeted 2D or small coupled-factor sweeps
- comparison of isolated 1D vs coupled-factor geometry
- first non-1D factor-space candidates

### Phase 5. Joint Market-State Geometry

Deliverable:

- full-snapshot synthetic geometry built from the strongest factor candidates
- estimate of the intrinsic dimension of the joint market space

### Phase 6. Context Ladder

Deliverable:

- same synthetic markets under settings / portfolio / constraints / strategies
- pre/post transformation analysis

### Phase 7. Causal Tests

Deliverable:

- subspace ablations
- mean-vector substitution
- rank vs magnitude corruption
- pairwise swap interventions

### Phase 8. Transfer to DX

Deliverable:

- evaluate synthetic manifold directions on real captures
- especially blocked and policy-tension cohorts


## Immediate Next Build

The first concrete build should be intentionally small.

Recommended initial scope:

- 4 assets per snapshot
- 6 archetypes
- 3 scalar sweep families
- 2 pairwise tradeoff families
- market-only prompt first
- then settings-only context ladder on the same market states

This is enough to test whether there is a clean 1D candidate manifold for:

- attractiveness
- pairwise preference
- risk-adjusted acceptability

without paying the complexity tax of simulating the entire DX environment immediately.


## Success Condition for This Program

This program is working if, within a few focused synthetic families, we can say something like:

- "the model represents one or more scalar variables on low-dimensional curved manifolds in early row states"
- "some market concepts only become clean in small coupled spaces rather than isolated 1D sweeps"
- "the joint market state is higher-dimensional than those scalar manifolds, but still structured and compressible"
- "settings rotate or reweight that market space rather than replacing it"
- "actionability lives on a partly distinct downstream subspace"
- "these synthetic directions transfer back to real DX captures"

If we cannot say those things, then we should conclude that the current market state representation is either:

- too entangled for this methodology as currently framed, or
- still being probed at the wrong variable level


## Relationship to Existing Root Plans

This document does not replace:

- `MARKET_MANIFOLD_RESEARCH_PLAN.md`
- `MARKET_MANIFOLD_IMPLEMENTATION_PLAN.md`

It sharpens them.

Those documents define the overall manifold program. This document defines the next concrete methodology change:

- from broad probing to hypothesis-first probing
- from noisy production data to synthetic variable isolation
- from final-action labels to cleaner intermediate quantities

This should be treated as the v2 experimental design for discovering Xenon's market manifolds.
