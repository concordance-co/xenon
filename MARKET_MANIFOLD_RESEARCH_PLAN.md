# Market Manifold Research Plan

## Purpose

This document defines a research program for understanding how the Xenon trading model represents the market internally, how that representation changes after settings and downstream context are integrated, and what information is actually necessary for the model to make a decision.

The goal is not just to ask whether a probe can decode a feature from activations. The goal is to characterize the structure of the model's internal market space:

- what the basic representation units are
- what geometry organizes them
- which factors are preserved across layers
- how policy and context transform that geometry
- which parts of the representation are causally necessary for action

This plan assumes the current Xenon architecture:

- source prompts and raw payloads stored in Neon
- activation capture on Qwen3-30B-A3B surrogate
- production decision prompts originating from Qwen3-235B-A22B
- pooled capture positions already available for market rows and downstream sections


## Why This Needs a New Plan

The initial counterfactual experiment was useful as a pipeline pilot, but it is not a strong answer to the market-geometry question.

Known issues from the current `init` experiment:

- The stored Dataset A low-risk vs high-risk preamble contrast is only a one-line difference in the market scanning heuristic.
- The stored `counterfactual_prompts` rows contain Dataset A `low_raw` and `high_raw`, but not explicit `low_pad` and `high_pad` prompt rows.
- Dataset B uses settings-digit edits that are too weak to induce measurable downstream differences.
- `question_b_results.json` shows `settings_eos` as insufficient data across layers.
- Downstream positions in Dataset B are effectively identical under the current settings edit.
- `decision.json` is internally inconsistent: the top-level decision says `late_reinterpretation`, while the metric flags indicate `b_reinterprets: false`.

Interpretation:

- v1 should be treated as a pipeline validation and prompt-auditing exercise.
- It should not be treated as a strong result about policy-driven reinterpretation.
- The existing captures are still valuable for learning the base market manifold.


## Core Research Questions

### 1. What is the model's market representation before policy and downstream context are integrated?

This is the "market manifold" question in the narrow sense.

We want to know whether the model organizes market rows by:

- raw metrics
- within-snapshot ranks
- pairwise relative comparisons
- latent factors such as momentum, flow, participation, and concentration risk
- sparse MoE routing patterns

### 2. How is that market representation transformed after settings and downstream context are integrated?

Because the model is causal:

- market tokens can see the preamble that comes before them
- market tokens cannot see the `ACTIVE SETTINGS` section that comes after them
- downstream positions can attend to both market and settings

So the pre/post comparison is:

- pre: market-row and market-section activations while the model is reading the market
- post: downstream section activations after settings, portfolio, constraints, and memories are available

### 3. What information is necessary for the model to choose an action?

This is the causal question.

We want to know:

- what the model needs to decide whether any action is legal
- what it needs to decide whether action is worthwhile
- what it needs to choose a target asset
- what it needs to choose direction and size


## Representation Stack

The system should not be modeled as a single manifold. It is better understood as a stack of coupled manifolds.

### A. Asset-in-Context Manifold

Unit:

- one asset row inside one market snapshot

Source features:

- `Market.Tokens[*].Metrics`
- token age from `CreatedTimestamp`
- reap role: source, target, neither
- held vs unheld
- pairwise relation to the rest of the snapshot

Research purpose:

- characterize how the model sees one candidate asset relative to other candidates

### B. Market-State Manifold

Unit:

- one full market snapshot

Source features:

- all asset rows together
- distribution of factor scores across assets
- pairwise comparison graph
- dominance structure
- reap configuration

Research purpose:

- characterize the model's geometry over market regimes, not just over isolated assets

### C. Policy / Settings Manifold

Unit:

- the effective policy state for the tick

Source features:

- `Agent.Options`
- prompt-level policy text
- `ACTIVE SETTINGS`
- parsed strategy clauses
- hard limits from constraints

Research purpose:

- separate soft preferences from actual legal-action constraints

### D. Affordance Manifold

Unit:

- what actions are actually available now

Source features:

- `AllowedTools`
- ETH balance
- held token balances
- high-priority restrictions
- hold floors
- price impact limits
- cooldown state

Research purpose:

- distinguish market interpretation from action legality

This matters because many prompts are dominated by hard restrictions rather than by market reasoning.

### E. Portfolio / Memory Manifold

Unit:

- agent-specific state entering the tick

Source features:

- portfolio holdings
- unrealized PnL
- time held
- time since last interaction
- `Memories`
- `PREVIOUS DECISIONS`

Research purpose:

- capture historical state that shapes decision policy independently of current market features

### F. Asset-Valence Manifold

Unit:

- asset-conditioned downstream preference state

Source features:

- final tool call and target asset
- tool arguments
- reasoning content
- matched observe examples
- deconstrained or strategy-removed counterfactual labels

Research purpose:

- separate asset preference from executed action
- learn what "bullish on asset i" and "bearish on asset i" look like internally
- test whether asset-specific valence forms before or after settings and downstream context are integrated

This is different from the final action space. A model can be bullish on an asset and still observe because the action is blocked by policy, constraints, or lack of ETH.

### G. Decision Manifold

Unit:

- the final integrated decision state

Source features:

- tool call
- tool arguments
- reasoning content

Research purpose:

- measure how pre-market structure survives into action


## Primary Hypotheses

### H1. The model's pre-market asset representation is more relational than absolute.

Prediction:

- within-snapshot ranks and pairwise deltas will explain more representational variance than raw metrics alone

### H2. The model compresses market rows into a low-dimensional latent factor space in middle and late layers.

Prediction:

- effective dimension drops by layer after controlling for snapshot identity
- derived factors outperform raw features in RSA at deeper layers

### H3. MoE routing is part of the market representation, not just an implementation detail.

Prediction:

- expert specialization emerges for momentum, flow, concentration, newness, reap roles, and legality regimes

### H4. Post-settings shifts can be decomposed into two types.

Prediction:

- parallel-to-market-subspace drift indicates reinterpretation or reweighting of market information
- orthogonal drift indicates policy overlay without rewriting market structure

### H5. In many prompts, affordance structure dominates decision geometry.

Prediction:

- strategy restrictions, no-ETH states, and hold rules create distinct downstream manifolds that can swamp market-driven variation unless prompts are stratified

### H6. Asset-conditioned bullish/bearish valence can be studied as a distinct downstream space.

Prediction:

- hard buy and sell labels at `last_token` will recover a useful decision-valence space
- the cleanest signal will come from asset-conditioned labels rather than a single pooled `buy_vs_sell` axis
- observation examples will contain both neutral cases and blocked-latent-sentiment cases, so observe must be decomposed rather than treated as uniformly neutral

### H7. Decision formation is staged.

Prediction:

- legality and actionability become decodable before target asset and size
- target asset becomes decodable before precise trade size


## Data Sources

### Existing sources

- `full_logs.raw_payload` in Neon
- `interp_examples_v0`
- `counterfactual_prompts`
- `counterfactual_snapshots`
- `counterfactual_templates`
- counterfactual activations on Modal

### Fields available from raw payloads

At minimum, the research dataset can derive:

- market metrics per token
- token age and newness
- reap source and target roles
- strategy texts and priorities
- slider settings
- portfolio holdings and unrealized PnL
- memory notes and recent tool history
- actual final tool call and reasoning
- asset-conditioned hard sentiment labels from executed buy/sell actions
- asset-conditioned weak sentiment labels from observation reasoning text
- asset-conditioned pseudo-labels from deconstrained counterfactual reruns

### Existing activation units

From the current counterfactual capture pipeline:

- `row_mean_{i}`
- `row_eos_{i}`
- `market_mean`
- `market_eos`
- `last_token`
- downstream section pools such as `active_settings_eos`, `portfolio_eos`, `constraints_eos`, `prev_decisions_eos`
- router indices at market and downstream positions


## Required Prompt Stratification

All analyses should be stratified before any manifold claims are made.

Minimum regime splits:

- unconstrained prompts
- prompts with active high-priority restriction strategies
- prompts with active high-priority hold rules
- zero-ETH prompts
- prompts with immediate-action or triggered-action strategies
- prompts with reap source exposure
- prompts with reap target exposure

Reason:

- if these regimes are mixed, the dominant geometry may reflect legality and compliance rather than market perception


## Measurements

## 1. Geometry of the Asset-in-Context Manifold

Use `row_mean_{i}` and `row_eos_{i}` as primary units.

Measurements:

- intrinsic dimension per layer
- PCA participation ratio
- TwoNN dimension estimate
- neighborhood preservation across layers
- CKA between layers

Controls:

- center activations within snapshot before dimensionality estimation
- report metrics separately for held and unheld tokens
- report metrics separately for constrained and unconstrained regimes

## 2. Representational Similarity Analysis

Build several representational dissimilarity matrices:

- raw metric space
- within-snapshot z-score space
- rank space
- pairwise comparison space
- derived factor space
- legality / affordance space
- portfolio-context space

Measure:

- layerwise correlation between model RDM and each hypothesis RDM

Research value:

- tells us what kind of structure the model preserves at each depth

## 3. Per-Metric and Per-Factor Decodability

Regression targets:

- individual market metrics
- derived continuous factors

Classification targets:

- leader labels
- top-k buckets
- rank buckets
- new launch / mature token
- source / target / neither in reap
- held vs unheld

Key question:

- what is linearly decodable, where it emerges, and whether raw metrics are replaced by higher-order factors deeper in the network

## 4. Pairwise Relational Encoding

Construct pair units from pairs of assets in the same snapshot.

Probe tasks:

- is asset A stronger than B on 5m momentum?
- is A more flow-surprising than B?
- is A safer than B on concentration?
- is A more attractive under the current risk setting?

This is critical because trading decisions are inherently relative.

## 5. Router Geometry and Expert Specialization

Use router indices and router statistics as first-class signals.

Measurements:

- expert frequency by label and factor
- routing entropy by layer and regime
- expert-label mutual information
- expert specialization for legality, momentum, flow, concentration, newness, reap roles, and strategy regimes

Key hypothesis:

- routing may encode the model's computational partition of market concepts more directly than residual vectors do

## 6. Whole-Market-State Geometry

Aggregate market rows into snapshot-level descriptors:

- top-1, top-2, and rank ordering summaries
- dispersion of factor scores
- pairwise comparison graph
- dominance entropy
- reap configuration

Measure:

- clusterability of market snapshots by regime
- CKA/RSA between market-section activations and snapshot-level factor spaces

## 7. Pre/Post Settings and Context Transformation

Compare pre-market units with downstream units:

- pre: `row_mean_{i}`, `row_eos_{i}`, `market_mean`, `market_eos`
- post: `active_settings_eos`, `portfolio_eos`, `constraints_eos`, `prev_decisions_eos`, `last_token`

Measurements:

- cross-decoding from pre to post
- subspace overlap and retained variance
- projection of post activations onto pre-market subspace
- orthogonal drift magnitude
- preservation of asset ordering and nearest neighbors

Interpretation:

- mostly parallel drift -> settings reweight market content
- mostly orthogonal drift -> settings add policy without rewriting market representation

## 8. Asset-Conditioned Valence Space

This analysis turns the user's observation into a first-class research object.

The core idea is to learn a downstream bullish/bearish space, then test when that space becomes asset-specific and whether settings apply it before or after downstream context is integrated.

Important caution:

- `buy = bullish` and `sell = bearish` is only approximately true
- `sell` can mean bearishness, rotation, risk reduction, or strategy execution
- `observe` can mean neutral, constrained, no-ETH, hold-rule, or latent bullishness blocked by policy

So the correct object is not a single raw `buy_vs_sell` sentiment axis. The correct object is asset-conditioned valence plus actionability.

### Label families

Hard labels:

- `bullish_on_asset_i` from executed `buy(asset_i)`
- `bearish_on_asset_i` from executed `sell(asset_i)`

Weak labels:

- `weak_bullish_on_asset_i` from observation reasoning that clearly expresses positive preference without action
- `weak_bearish_on_asset_i` from observation reasoning that clearly expresses negative preference without action

Pseudo-labels:

- `blocked_bullish_on_asset_i` from deconstraint or strategy-removal counterfactual runs where the model buys `asset_i`
- `blocked_bearish_on_asset_i` from deconstraint or strategy-removal counterfactual runs where the model sells `asset_i`

### Measurements

- train `last_token` probes for `bullish_on_asset_i` and `bearish_on_asset_i`
- learn contrastive directions from `buy(asset_i) - matched observe` and `sell(asset_i) - matched observe`
- learn contrastive directions from `bullish_on_asset_i - bearish_on_asset_i` on the same asset where possible
- project pre and post activations onto those directions
- compare valence decodability at market rows vs downstream sections

### Research questions

- is asset-specific bullishness already visible in `row_mean_i` or `row_eos_i` before settings?
- does settings/context sharpen that valence without changing its sign?
- does settings/context suppress, rotate, or reverse asset-specific valence?
- does a generic bullish/bearish pressure appear pre-market while target-asset binding happens later?

### Interpretation

- valence visible pre-market and preserved downstream -> the model forms market preference early, then policy gates execution
- valence weak pre-market and strong post-settings -> preference is assembled downstream with policy/context integrated
- generic valence early but asset binding late -> the model forms action pressure early and binds it to a target after downstream integration

## 9. Decision Crystallization

Downstream decision probes:

- `act_vs_observe`
- `buy_vs_sell`
- `forced_observe`
- `strategy_driven_vs_slider_driven`
- `target_asset`
- `size_bucket`
- `edge_gt_fee`
- `bullish_on_asset_i`
- `bearish_on_asset_i`
- `blocked_bullish_on_asset_i`
- `blocked_bearish_on_asset_i`

Measure:

- first layer at which each target becomes decodable
- whether legality emerges before target selection
- whether target selection emerges before size


## Causal Tests

Decodability alone is insufficient. We need targeted interventions.

### A. Market feature ablations

- remove or neutralize one metric family at a time
- examples: momentum-only removal, flow-only removal, concentration-only removal

### B. Rank-preserving vs magnitude-preserving corruptions

- preserve ranks and scramble magnitudes
- preserve magnitudes and scramble ranks

Purpose:

- test whether the model is primarily ordinal or metric-sensitive

### C. Pairwise swaps

- swap two assets' flow or momentum profiles
- preserve roster and formatting

Purpose:

- test relational dependence

### D. Affordance interventions

- toggle restriction strategies
- set ETH to zero vs funded
- hold portfolio constant while changing legality

Purpose:

- measure whether market geometry survives when legal action space collapses

### E. Stronger policy interventions

- full low-risk vs high-risk policy rewrites
- strategy clause insertion/removal
- policy text families that differ materially, not just in one sentence

### F. Valence interventions

- remove or relax action constraints while holding market constant
- rerun constrained prompts with the same market and portfolio but modified legality state
- compare whether latent bullish/bearish asset preferences become executable

Purpose:

- distinguish true asset preference from action blocking


## Dataset Program

### Phase A: Existing captures only

Use current Dataset A captures to learn:

- base market geometry
- factor organization
- pairwise structure
- routing specialization

Do not use current Dataset A to make broad claims about materially different policies.

### Phase B: Stronger post-settings dataset

Build a new Dataset B that includes:

- full settings rewrites
- stronger policy swaps
- strategy on/off contrasts
- funded vs zero-ETH contrasts
- memory truncation or controlled history variants

### Phase C: Causal perturbation suite

Build synthetic market edits that preserve prompt format but perturb specific market properties.


## Success Criteria

The project succeeds if it can answer all three of the following with clean evidence.

### 1. What is the market manifold made of?

Example acceptable answers:

- "The manifold is primarily organized by pairwise ordering and latent factors, not raw values."
- "Middle layers compress market rows into a low-dimensional momentum-flow-concentration space."

### 2. What changes after settings and context are integrated?

Example acceptable answers:

- "Settings mainly add an orthogonal policy offset."
- "Settings reweight the market manifold inside the same subspace."
- "Asset-conditioned bullishness is already present on market rows, but settings determine whether it becomes executable."

### 3. What is necessary for decision?

Example acceptable answers:

- "Legality is resolved first, target asset second, and size last."
- "The model is ordinal with respect to asset ranking but metric-sensitive for edge-vs-fee decisions."
- "The model forms asset-specific bullish/bearish valence pre-market, then portfolio and policy gates determine whether that valence becomes a buy, sell, or observe."


## Risks and Failure Modes

- Confounding legality with market interpretation
- Treating weak prompt interventions as policy tests
- Over-reading pooled activations without stronger causal tests
- Failing to separate held-token and unheld-token regimes
- Using only residual geometry and ignoring router behavior
- Building labels that collapse to token identity instead of market structure


## Deliverables

- a structured market-state dataset derived from raw payloads
- layerwise market manifold analysis for existing Dataset A captures
- router specialization report
- pre/post settings transformation report using stronger interventions
- intervention-based causal analysis of what the model needs to decide
- revised counterfactual report that cleanly separates perception, policy overlay, legality, and final decision formation
