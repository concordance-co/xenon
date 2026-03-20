# Market Manifold Implementation Plan

## Purpose

This document translates the market-manifold research agenda into concrete changes to the Xenon codebase, data model, and execution flow.

It assumes the current system described in:

- [README.md](/Users/brockelmore/concordance/xenon/README.md)
- [MARKET_ANALYSIS_PLAN.md](/Users/brockelmore/concordance/xenon/MARKET_ANALYSIS_PLAN.md)
- [pipelines/interp/counterfactual.py](/Users/brockelmore/concordance/xenon/pipelines/interp/counterfactual.py)
- [pipelines/interp/counterfactual_capture.py](/Users/brockelmore/concordance/xenon/pipelines/interp/counterfactual_capture.py)
- [pipelines/interp/counterfactual_analysis.py](/Users/brockelmore/concordance/xenon/pipelines/interp/counterfactual_analysis.py)


## Guiding Principles

- Keep the research pipeline auditable. Every result should be traceable back to a specific prompt family, intervention family, and capture run.
- Separate feature extraction from activation analysis. Raw payload parsing should produce a stable tabular dataset that all later analyses use.
- Treat legality and policy context as explicit objects, not hidden confounds.
- Keep the existing counterfactual capture stack, but broaden what it pools and how it records provenance.


## Current Gaps

The current implementation has four immediate weaknesses that should be fixed before deeper analysis.

### 1. Prompt provenance is too weak

Observed issues:

- `counterfactual_prompts` does not include an `experiment_id`
- stored prompt variants do not fully match the report narrative
- prompt-template provenance is not explicit enough for prompt audits

Required fix:

- make every prompt and every capture self-describing

### 2. Raw payloads are not yet normalized into a research dataset

Observed issues:

- the information needed for market, policy, affordance, memory, and decision manifolds exists in `raw_payload`
- there is no canonical derived feature table that unifies those views

Required fix:

- add a stable feature extraction layer

### 3. Capture positions are incomplete for the intended research

Observed issues:

- current pooling covers market rows and some downstream sections
- it does not explicitly pool strategy blocks, reaps blocks, or current-state positions

Required fix:

- expand pooled positions and metadata

### 4. Analysis logic conflates pipeline health with scientific verdicts

Observed issues:

- automated verdict output can disagree with metric flags
- settings experiments can silently degenerate into identity comparisons

Required fix:

- add stricter validation and explicit "insufficient intervention" reporting


## Deliverable Overview

This implementation plan produces six concrete outputs.

- a structured market-state dataset derived from raw payloads
- a richer counterfactual prompt and capture schema
- market geometry analysis on current captures
- stronger policy/settings intervention datasets
- pre/post transformation analyses
- causal intervention suite for necessity testing


## Workstream 1: Provenance and Schema Fixes

### Goal

Make prompts, captures, and analyses auditable.

### Changes

#### `counterfactual_prompts`

Add columns:

- `experiment_id`
- `prompt_family`
- `intervention_family`
- `intervention_strength`
- `template_hash`
- `source_prompt_hash`
- `user_text_hash`
- `system_text_hash`
- `metadata_json`

Purpose:

- support exact prompt audits
- distinguish weak and strong policy interventions
- make reports reproducible

#### `counterfactual_snapshots`

Add or confirm fields:

- `market_features_json`
- `regime_labels_json`
- `stratification_tags`
- `roster_hash`
- `snapshot_family`

Purpose:

- support later stratified analysis without reconstructing everything on the fly

#### Activation metadata

Extend `metadata.parquet` or counterfactual metadata rows with:

- `experiment_id`
- `dataset`
- `variant`
- `prompt_family`
- `intervention_family`
- `section_keys_present`
- `router_keys_present`
- `boundary_quality_flags`
- `capture_version`

### Files

- [pipelines/interp/counterfactual.py](/Users/brockelmore/concordance/xenon/pipelines/interp/counterfactual.py)
- [pipelines/interp/counterfactual_capture.py](/Users/brockelmore/concordance/xenon/pipelines/interp/counterfactual_capture.py)
- migration scripts under `scripts/`


## Workstream 2: Structured Feature Extraction from Raw Payloads

### Goal

Create a canonical research dataset from `full_logs.raw_payload`.

### New module

- `pipelines/interp/manifold_dataset.py`

### Outputs

#### Tick-level table

One row per prompt / tick.

Fields:

- identifiers: `log_id`, `vault_address`, `request_id`, `created_at`
- policy/settings: sliders, parsed policy tags, strategy counts by type
- affordance state: can_buy, can_sell_any, forced_observe, zero_eth, restriction_active
- portfolio summaries: held-token count, concentration, total unrealized PnL summary, time-held summaries
- memory summaries: recent action counts, repeated token mentions, recent same-token churn features
- decision labels: tool name, target token, size bucket, strategy-driven vs slider-driven
- valence labels: generic bullish pressure, generic bearish pressure, target-conditioned bullish/bearish state when available
- regime labels: unconstrained, restriction, hold-rule, immediate-action, reap-exposed

#### Asset-row table

One row per asset per prompt.

Fields:

- token identifiers and raw metrics
- within-snapshot z-scores and ranks
- derived factors
- pairwise aggregate summaries
- reap role
- held vs unheld
- feasible buy / feasible sell labels
- chosen / not chosen labels
- hard bullish / bearish labels from executed action
- weak bullish / bearish labels from observation reasoning
- blocked bullish / blocked bearish pseudo-labels from deconstraint reruns

#### Pairwise table

One row per ordered pair of assets in a snapshot.

Fields:

- pairwise metric deltas
- pairwise factor deltas
- relation labels such as `a_beats_b_on_5m`

### Parsing requirements

- parse strategy clauses into immediate-action, triggered-action, restriction, hold-rule
- parse active settings and current state from user prompt blocks when needed
- derive legality labels directly from payload state, not just from reasoning text
- derive asset-conditioned valence labels separately from executed action labels

### Validation

- unit tests for strategy classification
- unit tests for feature extraction from representative raw payloads
- invariants such as one chosen token at most for buy/sell actions
- invariants that blocked-valence labels are only created from explicit pseudo-label procedures

### Asset-valence label program

The implementation should treat asset-conditioned bullish/bearish space as a separate labeling layer.

Hard labels:

- `buy(asset_i)` -> `bullish_on_asset_i`
- `sell(asset_i)` -> `bearish_on_asset_i`

Weak labels:

- observation reasoning that clearly prefers an asset without acting
- observation reasoning that clearly rejects or wants to exit an asset without acting

Pseudo-labels:

- rerun the same prompt under relaxed legality or strategy conditions
- use resulting action as a latent valence readout

Reason:

- raw `buy_vs_sell_vs_observe` is too entangled with legality, constraints, and cash state to serve as the only sentiment label family


## Workstream 3: Capture Expansion

### Goal

Extend the capture pipeline to better align with the research questions.

### Existing pooled positions

Already available:

- `row_mean_{i}`
- `row_eos_{i}`
- `market_mean`
- `market_eos`
- `last_token`
- downstream section pools

### New pooled positions

Add:

- `active_strategies_mean`
- `active_strategies_eos`
- `reaps_mean`
- `reaps_eos`
- `current_state_mean`
- `current_state_eos`
- optional `symbol_prefix_mean_{i}` for explicit leakage controls

### Router capture extensions

Add router indices for:

- active strategies section
- reaps section
- current state section

### Boundary quality

Every capture should record:

- whether each expected boundary was found
- start and end token indices
- fallback behavior taken if a boundary failed

### Files

- [pipelines/interp/counterfactual_capture.py](/Users/brockelmore/concordance/xenon/pipelines/interp/counterfactual_capture.py)
- [pipelines/interp/modal_vllm_capture.py](/Users/brockelmore/concordance/xenon/pipelines/interp/modal_vllm_capture.py)


## Workstream 4: Dataset A Refresh

### Goal

Rebuild the base market-geometry dataset so the stored prompt spec matches what analysis claims it uses.

### Tasks

- audit current Dataset A variant generation
- explicitly store every generated prompt variant
- if padded variants are used, persist them in `counterfactual_prompts`
- if padded variants are not used, remove padding language from report generation and analysis assumptions
- record exact template hashes for low/high policies

### Additional improvements

- ensure row-order randomization is recorded per prompt
- record roster hash and market snapshot hash
- add strong and weak policy families as separate prompt families

### Files

- [pipelines/interp/counterfactual.py](/Users/brockelmore/concordance/xenon/pipelines/interp/counterfactual.py)


## Workstream 5: Stronger Dataset B Redesign

### Goal

Replace the current too-weak settings edit with intervention families that can meaningfully test pre/post transformation.

### New intervention families

#### 1. Full settings rewrite

Replace entire `ACTIVE SETTINGS` block with materially different values and explanatory notes.

#### 2. Strong policy rewrite

Swap in policy text families that differ materially in:

- sell rules
- cooldown rules
- market scanning heuristics
- rotation policy
- risk framing

#### 3. Strategy interventions

- remove active high-priority restrictions
- insert controlled restrictions
- convert restriction to hold-rule
- add immediate-action directives

#### 4. Affordance interventions

- zero ETH vs funded ETH
- held-token vs unheld-token states
- price impact limit tightening

#### 5. Memory interventions

- truncate previous decisions
- inject controlled recent action sequences

### Dataset structure

Each family should have:

- matched prompt pairs
- clear metadata describing what changed
- enough prompts per regime to estimate downstream shifts with confidence

### Files

- [pipelines/interp/counterfactual.py](/Users/brockelmore/concordance/xenon/pipelines/interp/counterfactual.py)


## Workstream 6: Market Geometry Analysis Module

### Goal

Analyze the structure of the pre-market manifold using existing captures.

### New module

- `pipelines/interp/market_geometry.py`

### Responsibilities

- intrinsic dimension estimation
- PCA participation ratio
- TwoNN dimension
- CKA across layers
- neighborhood preservation
- RSA against raw, z-scored, ranked, pairwise, factor, and affordance spaces
- router specialization summaries

### Inputs

- structured asset-row dataset
- pooled market-row activations
- router indices

### Outputs

- JSON summaries
- plots for dimension curves and RSA curves
- optional embedding snapshots for selected layers


## Workstream 7: Pre/Post Transformation Analysis Module

### Goal

Measure how settings and downstream context transform the market manifold.

### New module

- `pipelines/interp/setting_shift.py`

### Responsibilities

- fit pre-market subspaces per layer
- project downstream activations into pre-market subspaces
- measure retained variance
- measure orthogonal drift
- compare same-setting and cross-setting decoding
- measure asset-order preservation and nearest-neighbor preservation
- project downstream asset-valence directions back onto pre-market rows
- measure where generic valence becomes asset-specific

### Inputs

- strong Dataset B captures
- structured prompt-level and asset-level datasets

### Outputs

- subspace-overlap curves
- parallel-vs-orthogonal drift plots
- pre/post decoding curves
- regime-stratified reports


## Workstream 8: Decision Architecture Analysis Module

### Goal

Map when legality, actionability, asset-conditioned bullish/bearish valence, target selection, and sizing crystallize across layers.

### New module

- `pipelines/interp/decision_architecture.py`

### Probe targets

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
- generic `bullish_pressure`
- generic `bearish_pressure`

### Additional responsibilities

- fit contrastive directions such as `buy(asset_i) - matched observe`
- fit contrastive directions such as `sell(asset_i) - matched observe`
- fit same-asset `bullish_on_asset_i - bearish_on_asset_i` directions where possible
- compare whether asset-conditioned valence appears first on market rows or only after downstream integration
- report whether target binding happens before or after generic bullish/bearish pressure appears

### Outputs

- layerwise probe curves
- first-decodable-layer summaries
- target-vs-size ordering report


## Workstream 9: Causal Intervention Suite

### Goal

Test necessity, not just decodability.

### New module

- `pipelines/interp/intervention_suite.py`

### Intervention families

- metric-family ablations
- rank-preserving corruptions
- magnitude-preserving corruptions
- pairwise swaps
- asset masking
- held/unheld flips
- strong policy and strategy rewrites

### Requirements

- preserve prompt format
- preserve roster unless intentionally changed
- annotate intervention type and target property


## Workstream 10: Reporting and Guardrails

### Goal

Prevent scientific claims from outrunning what the data actually supports.

### Changes

#### Decision logic

Replace binary verdict shortcuts with explicit states:

- `supported`
- `underpowered`
- `insufficient_data`
- `intervention_too_weak`
- `schema_mismatch`

#### Report generator

Make reports display:

- exact prompt-family difference summary
- exact intervention family
- capture boundary health
- sufficient-data checks
- whether verdict logic and metric flags agree

### Files

- [pipelines/interp/modal_analysis.py](/Users/brockelmore/concordance/xenon/pipelines/interp/modal_analysis.py)
- report generation logic under the counterfactual analysis flow


## Execution Plan

### Phase 1: Foundation

Scope:

- provenance fixes
- structured raw-payload dataset
- basic regime stratification
- tests for extraction and strategy parsing

Expected output:

- stable feature tables for tick, asset-row, and pairwise data

### Phase 2: Existing Capture Analysis

Scope:

- market geometry on current Dataset A captures
- router specialization
- base manifold plots

Expected output:

- a defensible description of the base market manifold from current data

### Phase 3: Stronger Counterfactual Recapture

Scope:

- rebuild Dataset A and Dataset B with stronger intervention families
- recapture on Modal
- validate boundaries and provenance

Expected output:

- new capture run with usable pre/post settings contrasts

### Phase 4: Pre/Post and Decision Architecture

Scope:

- setting-shift analysis
- decision crystallization probes
- asset-conditioned valence probes
- causal interventions

Expected output:

- a complete report on pre-market representation, post-settings transformation, and decision necessity


## Testing Plan

### Unit tests

- raw payload parsing
- strategy clause classification
- legality / affordance label derivation
- section-boundary detection
- capture metadata completeness

### Integration tests

- build small synthetic prompt families and ensure every expected section key exists
- verify prompt-family provenance is persisted
- verify analyses fail loudly on insufficient-data settings

### Scientific sanity checks

- symbol-only and row-index baselines
- constraint-only baselines
- portfolio-only baselines
- memory-only baselines
- random-label controls


## Suggested File Additions

- `pipelines/interp/manifold_dataset.py`
- `pipelines/interp/market_geometry.py`
- `pipelines/interp/setting_shift.py`
- `pipelines/interp/decision_architecture.py`
- `pipelines/interp/intervention_suite.py`
- `tests/test_manifold_dataset.py`
- `tests/test_strategy_parsing.py`
- `tests/test_counterfactual_boundaries.py`


## Suggested File Modifications

- [pipelines/interp/counterfactual.py](/Users/brockelmore/concordance/xenon/pipelines/interp/counterfactual.py)
- [pipelines/interp/counterfactual_capture.py](/Users/brockelmore/concordance/xenon/pipelines/interp/counterfactual_capture.py)
- [pipelines/interp/counterfactual_analysis.py](/Users/brockelmore/concordance/xenon/pipelines/interp/counterfactual_analysis.py)
- [pipelines/interp/modal_analysis.py](/Users/brockelmore/concordance/xenon/pipelines/interp/modal_analysis.py)
- scripts that reconcile, migrate, or summarize counterfactual captures


## Minimal Acceptance Criteria

This implementation plan is complete when all of the following are true.

- Every prompt used in an experiment has explicit provenance and intervention metadata.
- Raw payloads can be converted into stable tick-level, asset-level, and pairwise research tables.
- Existing Dataset A captures can produce a market-geometry report without relying on weak policy claims.
- New Dataset B captures produce non-degenerate pre/post settings contrasts.
- Reports explicitly label weak or failed interventions instead of turning them into narrative conclusions.


## Final Deliverable

The final deliverable is a reproducible analysis pipeline that can answer three questions with auditable evidence:

- what the model's market manifold looks like
- how settings and downstream context transform it
- what information is necessary for the model to decide
