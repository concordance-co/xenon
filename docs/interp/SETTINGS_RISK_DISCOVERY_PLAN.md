# Settings And Risk Discovery Plan

## Why This Is The Parallel Track

The meeting direction was:

- main track: causal work on the market subspace
- parallel track: settings / risk representation discovery

This second track matters because the model may already be building an internal market-risk representation from the market itself:

- dispersion-like market structure
- leader-versus-rest spread
- concentration or fragility

That internal market-risk picture may then interact with explicit user settings such as:

- risk slider
- trading activity
- size
- diversification

So the goal is not just “decode the setting token.”

The goal is:

- identify explicit settings representation
- identify market-derived risk representation
- ask where they meet or drift apart


## Core Questions

1. Where is explicit settings information represented?
2. Is explicit risk represented differently from easier settings like trading activity?
3. Does the model already build a market-risk representation from the market block before reading settings?
4. When settings arrive, do they alter that representation or just gate later decisions?
5. Is the explicit risk setting aligned with the same internal directions as market dispersion / riskiness?


## Phase A: Settings Discovery

### A1. Reuse The Phase 15 Method

This should largely reuse the Phase 15 discovery pipeline:

- repeated prompt families
- pooled section states
- nuisance residualization
- PCA + probe/correlation analysis

But the target sections shift from the market block to the settings-related blocks.


### A2. Target States

Primary:

- `active_settings_mean`
- `active_settings_eos`

Secondary:

- `portfolio_context_eos`
- `constraints_eos`
- `price_impact_limits_eos`
- post-market integration deltas, e.g.
  - `active_settings_eos - market_eos`
  - `constraints_eos - market_eos`


### A3. Prompt Families

Use matched prompts where the market stays fixed and settings vary.

At minimum:

1. Risk ladder
   - `risk = 1..5`

2. Trading activity ladder
   - `TA = 1..5`

3. Trade size ladder
   - `size = 1..5`

4. Diversification ladder
   - `div = 1..5`

Trading activity is a good control:

- easier to define
- less semantically fuzzy than risk

Risk is the main target:

- more interesting
- but fuzzier
- likely partly mediated by market-derived risk signals


## Phase B: External Labels

Risk cannot stay fuzzy.

Build explicit external labels for both:

### B1. Explicit settings labels

- the literal slider value
- one-hot or ordinal form

### B2. Market-derived riskiness labels

These should be computed from prompt-visible market rows only.

Candidate families:

- return dispersion
  - `mad`, `std`, `gap`
- volume dispersion / fragility
- participation breadth
- holder concentration
- combined riskiness indices
  - e.g. high dispersion + thin breadth + high concentration

### B3. Decision-risk labels

For later validation, define external “riskiness of the chosen action” measures, such as:

- buying the most volatile asset
- concentration of the resulting portfolio
- size relative to the most risky available asset

These are secondary and should come after representation discovery.


## Phase C: Main Analyses

### C1. Section-local discovery

For each settings-related state:

- run nuisance-residualized PCA
- compute top correlated setting features
- compute top correlated market-risk features

This tells us whether the state is primarily:

- lexical settings readout
- market-risk readout
- or a mixed integration state


### C2. Easier-setting control

Use `TA` as the first control.

Why:

- if we cannot find `TA`, the pipeline is weak
- if we can find `TA` cleanly but not risk, that says risk is genuinely fuzzier rather than the method being broken


### C3. Risk-vs-market alignment

For risk specifically:

- compare the explicit risk-setting subspace to the market-dispersion / market-risk subspace

Methods:

- subspace overlap
- CCA / PLS between:
  - market-risk feature space
  - settings-state activations
- directional regression

Main question:

- does explicit risk recruit the same internal axes as market-derived riskiness?


### C4. Order-position comparison

Use the Phase 16 A/B/C logic for settings.

Conditions:

- market only
- market then settings
- settings then market

Main tests:

1. If settings come after market, market perception should stay fixed.
2. If settings come before market, does the market encoding itself shift?
3. Which settings do this most strongly?
   - `TA`?
   - `risk`?
   - `size`?


## Settings / Risk Sanity Checks

### S1. Exact-prefix sanity

If the prefix up to `market_eos` is identical, the state should be identical.

If not, fix the prompt construction or section indexing first.


### S2. Easier-setting positive control

The pipeline should recover at least one easier setting, ideally `TA`.

If not, do not over-interpret a null risk result.


### S3. Prompt-visible restriction

When making semantic claims, use only:

- literal setting values
- prompt-visible market-derived features

Do not label axes using hidden synthetic sidecar fields.


### S4. Paraphrase / format robustness

Settings representation should survive:

- wording changes
- row reordering
- slider formatting variants

If the signal vanishes under mild paraphrase, it is probably mostly lexical.


### S5. Null-setting control

Include a condition where settings change position but stay semantically neutral.

Expected result:

- minimal effect

This helps distinguish meaningful risk/settings signal from generic section-order effects.


### S6. Risk-specificity check

A risk axis should not just be a generic “do more / do less” action axis.

Compare it to:

- TA
- size
- diversification

If the “risk” signal is fully explained by those, it is not a distinct representation yet.


## Success Criteria

This track is successful if we can show at least one of the following:

1. A clean explicit settings subspace exists.
2. Risk is represented differently from simpler settings like `TA`.
3. The explicit risk setting aligns with a market-derived risk / dispersion subspace.
4. Settings-before-market measurably warps market perception.

The strongest result would be:

- a market-derived risk direction
- an explicit risk-setting direction
- and evidence that they meet in later integration states


## Immediate Next Steps

1. Build matched settings ladders with fixed market snapshots.
2. Start with `TA` and `risk`.
3. Run the Phase 15-style residualized discovery pipeline on:
   - `active_settings_mean`
   - `active_settings_eos`
   - `constraints_eos`
4. Compare explicit settings decode against prompt-derived market-risk features.
5. Add the A/B/C order-position variant once the local discovery states are working.
