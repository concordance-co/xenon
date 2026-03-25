# Market Representation Discovery Proposal

## Why We Should Shift The Program

The current program has produced one strong result:

- post-market section states in real DX prompts still carry context-shaped geometry changes, and affordance is the clearest case

But it has also exposed two important weaknesses in the current synthetic setup:

1. The synthetic prompt surface is too artificial.
   - It says `SYNTHETIC MARKET SCENARIO`
   - It includes fields like `Archetype`
   - Its section order and formatting differ from real DX prompts
   - The context edits are easier and cleaner than the real prompt distribution

2. The current latent coordinates are too hand-designed.
   - We used a 2D latent space like `strength` and `quality`
   - That was useful as a scaffold
   - But it is a human factorization, not evidence that these are the model’s real axes
   - In particular, `quality` bundles several different things together: participation, concentration, stability, and freshness penalties

So the next phase should not be “more of the same synthetic ladder with slightly different prompts.”

It should be:

- first, discover what directions the model actually uses to encode the market
- second, test how context changes that encoding
- third, rebuild synthetic prompts so they match real DX much more closely


## Concrete Prompt Spot Check

Before talking about PCA, context ordering, or latent axes, it helps to look at the actual prompt surface.

The key point is simple:

- the current synthetic prompts are easy to read and easy to control
- but they also differ from real DX prompts in obvious ways that could matter to the model

The examples below are proposed prompt text for the rebuilt synthetic dataset, followed by a real DX excerpt that shows the target surface we want to imitate.

### Proposed Synthetic Pair

These are draft prompts for the next generator. They are not meant to be “toy” examples. They are meant to look like real DX prompts while still letting us control the hidden market factors underneath.

System prompt:

```text
You are an autonomous trading agent in a 21-day onchain tournament. Your owner gave you ETH to deploy into tournament tokens and maximize returns. ETH sitting idle earns nothing, but overtrading burns fees, so allocate capital deliberately.

Each tick, you MUST respond with exactly ONE tool call: buy_token, sell_token, or record_observation.
Do not output any non-tool text.

Decision hierarchy (resolve conflicts in this order):

1) Hard constraints & tool schema (one-tool rule, available tokens only, ETH balance).
2) ACTIVE STRATEGIES with priority HIGH:
   - override ALL slider constraints, max trade size, and max price impact while directives are unfulfilled.
   - if a HIGH directive is unfulfilled and feasible, execute with buy_token or sell_token now.
   - use record_observation under HIGH only when objective is fulfilled or hard-blocked.
   - persistent HIGH directives remain active until their explicit end condition; do not mark them complete after one action.
3) ACTIVE STRATEGIES with priority MEDIUM - override slider preferences when they conflict, but respect TA/HS constraints.
4) User sliders.
5) ACTIVE STRATEGIES with priority LOW (suggestions only).

Inside every tool call, include a brief reasoning note (1-2 lines). Mention the strategy or slider(s) that drove your decision and the key market signal. Keep it conversational but specific so your owner and your future self can audit decisions.
```

`market_only` user prompt:

```text
## MARKET SNAPSHOT

- NERA
  - Price: 0.00000182 ETH
  - 5m change: +8.1%
  - 1h change: +16.8%
  - Net flow 5m: +2.11 ETH
  - Volume 5m: 6.42 ETH
  - Volume 1h: 23.70 ETH
  - Unique traders 5m: 31
  - Top 20 holder pct: 27.8%
  - Time since launch: 18h

- VEXA
  - Price: 0.00000231 ETH
  - 5m change: +6.0%
  - 1h change: +13.7%
  - Net flow 5m: +1.58 ETH
  - Volume 5m: 5.21 ETH
  - Volume 1h: 18.46 ETH
  - Unique traders 5m: 27
  - Top 20 holder pct: 31.9%
  - Time since launch: 19h

- MORI
  - Price: 0.00000141 ETH
  - 5m change: +4.8%
  - 1h change: +11.1%
  - Net flow 5m: +1.12 ETH
  - Volume 5m: 4.67 ETH
  - Volume 1h: 16.94 ETH
  - Unique traders 5m: 23
  - Top 20 holder pct: 35.4%
  - Time since launch: 20h

- LUMA
  - Price: 0.00000096 ETH
  - 5m change: +3.4%
  - 1h change: +8.6%
  - Net flow 5m: +0.71 ETH
  - Volume 5m: 3.92 ETH
  - Volume 1h: 14.21 ETH
  - Unique traders 5m: 20
  - Top 20 holder pct: 38.7%
  - Time since launch: 18h

- KIRO
  - Price: 0.00000109 ETH
  - 5m change: +2.8%
  - 1h change: +7.9%
  - Net flow 5m: +0.53 ETH
  - Volume 5m: 3.48 ETH
  - Volume 1h: 12.87 ETH
  - Unique traders 5m: 18
  - Top 20 holder pct: 40.1%
  - Time since launch: 21h

- TAVO
  - Price: 0.00000166 ETH
  - 5m change: +1.9%
  - 1h change: +6.8%
  - Net flow 5m: +0.39 ETH
  - Volume 5m: 3.22 ETH
  - Volume 1h: 11.75 ETH
  - Unique traders 5m: 17
  - Top 20 holder pct: 42.8%
  - Time since launch: 20h

------------------------------

## ACTIVE STRATEGIES (CURRENT ONLY)

No active strategies.

------------------------------

## ACTIVE SETTINGS

- Trading Activity (TA): 5 / 5 - How often you trade when there is fresh edge (1=very patient, 5=highly active). TA does NOT require a trade every tick.
- Asset Risk Preference (Risk): 2 / 5 - Which tokens you consider (1=prefer least volatile available, 5=embrace high volatility). Risk determines which tokens to consider, not whether to trade at all.
- Trade Size (Size): 3 / 5 - Maximum position sizing per trade (1=up to ~15%, 2=up to ~30%, 3=up to ~50%, 4=up to ~70%, 5=up to ~90% of available ETH). These are UPPER BOUNDS, not minimums.
- Holding Style (Hold): 2 / 5 - Minimum hold time before considering an exit (1=about ~30 minutes minimum; 2=about ~1 hour; 3=hours; 4=many hours; 5=days).
- Diversification (Div): 4 / 5 - Portfolio spread (1=concentrated in 1-2 tokens; 2=focused 2-3; 3=balanced 3-5; 4=spread 4-6; 5=wide 5+).

------------------------------

## PORTFOLIO CONTEXT

- ETH: Balance: 0.240000
- No current token holdings.

## CONSTRAINTS

- Max Trade Amount (Percent): 100.00% of available ETH

## PRICE IMPACT LIMITS (max 1500 bps)

- NERA: BUY max 100.00% of ETH
- VEXA: BUY max 100.00% of ETH
- MORI: BUY max 100.00% of ETH
- LUMA: BUY max 100.00% of ETH
- KIRO: BUY max 100.00% of ETH
- TAVO: BUY max 100.00% of ETH
```

`affordance_4` user prompt:

```text
## MARKET SNAPSHOT

- NERA
  - Price: 0.00000182 ETH
  - 5m change: +8.1%
  - 1h change: +16.8%
  - Net flow 5m: +2.11 ETH
  - Volume 5m: 6.42 ETH
  - Volume 1h: 23.70 ETH
  - Unique traders 5m: 31
  - Top 20 holder pct: 27.8%
  - Time since launch: 18h

- VEXA
  - Price: 0.00000231 ETH
  - 5m change: +6.0%
  - 1h change: +13.7%
  - Net flow 5m: +1.58 ETH
  - Volume 5m: 5.21 ETH
  - Volume 1h: 18.46 ETH
  - Unique traders 5m: 27
  - Top 20 holder pct: 31.9%
  - Time since launch: 19h

- MORI
  - Price: 0.00000141 ETH
  - 5m change: +4.8%
  - 1h change: +11.1%
  - Net flow 5m: +1.12 ETH
  - Volume 5m: 4.67 ETH
  - Volume 1h: 16.94 ETH
  - Unique traders 5m: 23
  - Top 20 holder pct: 35.4%
  - Time since launch: 20h

- LUMA
  - Price: 0.00000096 ETH
  - 5m change: +3.4%
  - 1h change: +8.6%
  - Net flow 5m: +0.71 ETH
  - Volume 5m: 3.92 ETH
  - Volume 1h: 14.21 ETH
  - Unique traders 5m: 20
  - Top 20 holder pct: 38.7%
  - Time since launch: 18h

- KIRO
  - Price: 0.00000109 ETH
  - 5m change: +2.8%
  - 1h change: +7.9%
  - Net flow 5m: +0.53 ETH
  - Volume 5m: 3.48 ETH
  - Volume 1h: 12.87 ETH
  - Unique traders 5m: 18
  - Top 20 holder pct: 40.1%
  - Time since launch: 21h

- TAVO
  - Price: 0.00000166 ETH
  - 5m change: +1.9%
  - 1h change: +6.8%
  - Net flow 5m: +0.39 ETH
  - Volume 5m: 3.22 ETH
  - Volume 1h: 11.75 ETH
  - Unique traders 5m: 17
  - Top 20 holder pct: 42.8%
  - Time since launch: 20h

------------------------------

## ACTIVE STRATEGIES (CURRENT ONLY)

No active strategies.

------------------------------

## ACTIVE SETTINGS

- Trading Activity (TA): 5 / 5 - How often you trade when there is fresh edge (1=very patient, 5=highly active). TA does NOT require a trade every tick.
- Asset Risk Preference (Risk): 2 / 5 - Which tokens you consider (1=prefer least volatile available, 5=embrace high volatility). Risk determines which tokens to consider, not whether to trade at all.
- Trade Size (Size): 3 / 5 - Maximum position sizing per trade (1=up to ~15%, 2=up to ~30%, 3=up to ~50%, 4=up to ~70%, 5=up to ~90% of available ETH). These are UPPER BOUNDS, not minimums.
- Holding Style (Hold): 2 / 5 - Minimum hold time before considering an exit (1=about ~30 minutes minimum; 2=about ~1 hour; 3=hours; 4=many hours; 5=days).
- Diversification (Div): 4 / 5 - Portfolio spread (1=concentrated in 1-2 tokens; 2=focused 2-3; 3=balanced 3-5; 4=spread 4-6; 5=wide 5+).

------------------------------

## PORTFOLIO CONTEXT

- ETH: Balance: 0.006000
- MORI: Balance: 18422000.000 | Avg Entry: 0.00000126 ETH | Unrealized PnL: +4.90% | Time Since Last Interaction: 35m | Time Held: 2h 10m

## CONSTRAINTS

- Max Trade Amount (Percent): 12.00% of available ETH

## PRICE IMPACT LIMITS (max 400 bps)

- NERA: BUY max 0.00% of ETH
- VEXA: BUY max 0.00% of ETH
- MORI: BUY max 12.00% of ETH, SELL max 100.00% of MORI
- LUMA: BUY max 0.00% of ETH
- KIRO: BUY max 0.00% of ETH
- TAVO: BUY max 0.00% of ETH
```

What these proposed prompts are trying to fix:

- no `synthetic` marker
- no `Archetype` field
- same broad section order as DX
- six-asset rosters instead of the earlier four-asset toy format
- affordance changes expressed through realistic balance and price-impact limits rather than explicit “Asset A is blocked” narration

### Real DX Pair

Below is a verbatim excerpt from a matched real affordance ladder pair. This is the kind of surface we ultimately need the synthetic prompts to imitate much more closely.

`market_only` excerpt:

```text
## ACTIVE SETTINGS

- Trading Activity (TA): 5 / 5 - How often you trade when there is fresh edge (1=very patient, 5=highly active). TA does NOT require a trade every tick.
- Asset Risk Preference (Risk): 2 / 5 - Which tokens you consider (1=prefer least volatile available, 5=embrace high volatility). Risk determines which tokens to consider, not whether to trade at all.
- Trade Size (Size): 3 / 5 - Maximum position sizing per trade (1=up to ~15%, 2=up to ~30%, 3=up to ~50%, 4=up to ~70%, 5=up to ~90% of available ETH). These are UPPER BOUNDS, not minimums. A small trade can still be valid.
- Holding Style (Hold): 2 / 5 - Minimum hold time before considering an exit (1=about ~30 minutes minimum; 2=about ~1 hour; 3=hours; 4=many hours; 5=days). Do NOT sell before minimum hold unless an exceptional reason exists (thesis broken, stop-loss, or explicit [HIGH] directive). When in doubt about whether to sell, holding is usually correct — positions need time to develop.
- Diversification (Div): 4 / 5 - Portfolio spread (1=concentrated in 1-2 tokens; 2=focused 2-3; 3=balanced 3-5; 4=spread 4-6; 5=wide 5+). Div=1-2 should usually add to existing positions instead of opening many new ones.

**If any slider is 0, treat it as 3 (balanced) and mention in reasoning that the slider was not configured.**

**Sell sizing**: Trade Size guides sells as well as buys. Size=1-2: small trims (10-30%). Size=3: moderate trims (20-50%). Size=4-5: larger exits (30-70%) when conviction is high. Prefer smaller trims over larger exits — you can always sell more later, but you cannot undo a sell. Full 100% exits should be rare — they require either a [HIGH] directive that explicitly says "sell all" or "liquidate", or a genuine fundamental thesis break. A price drop — even a large, fast one — is NOT a thesis break in a tournament where 10-20% swings are normal.

**Active + low-risk note (TA=5, Risk=2):** Low risk means prefer relatively less volatile options among available tokens. Avoid sell-rebuy oscillations caused by fabricated volatility thresholds.

**Fresh-signal gate (TA=5):** High activity means acting on fresh information. If your planned trade repeats a recent same-token/same-direction action without meaningful new evidence, OBSERVE.

**Maximum activity note (TA=5):** You are very active, but still avoid reflexive every-tick churn. Under active [HIGH], execute immediately. Under slider-only logic, require fresh edge.

**Low ETH note (0.0054 ETH):** Low ETH means you are deployed — that is a valid state. Monitor positions for genuine exit signals (stop-loss, thesis broken), not to restore a buffer. Do NOT sell just because ETH is low.

------------------------------

## PORTFOLIO CONTEXT

ETH sitting idle earns nothing — your job is to find opportunities and deploy into tokens. Once deployed, focus on HOLDING and monitoring for genuine exit signals — not continuously trading in and out. Positions need time to develop; the best returns come from conviction holds, not from constant rotation. If ETH is near zero and all positions have a valid thesis, that is a healthy state — do NOT sell just to rebuild an ETH buffer. If mostly ETH, look for quality entries.

**Note**: Unrealized PnL shown below is per-token only. You do NOT have vault-level total PnL. Do not treat one token's unrealized gain as your overall performance.

- ETH: Balance: 0.250000
- HOTDOGZ: Balance: 25520894187694874165248.000 | Avg Entry: 0.000000210156227185 ETH | Unrealized PnL: +9.51% | Time Since Last Interaction: 49m | Time Held: 1d 17h

## CONSTRAINTS

- Max Trade Amount (Percent): 100.00% of available ETH - **[HIGH] strategies may exceed this limit.**

## PRICE IMPACT LIMITS (max 1500 bps)

Max sizes that stay within your price impact tolerance. **[HIGH] strategies may exceed these limits** if needed to fulfill explicit directives.

- AIGF: BUY max 100.00% of ETH
- HOLE: BUY max 100.00% of ETH
- HOTDOGZ: BUY max 100.00% of ETH, SELL max 100.00% of HOTDOGZ
- LMAO: BUY max 100.00% of ETH
- LOOKSMAX: BUY max 100.00% of ETH
- POOPCOIN: BUY max 100.00% of ETH
```

`affordance_4` excerpt:

```text
## ACTIVE SETTINGS

- Trading Activity (TA): 5 / 5 - How often you trade when there is fresh edge (1=very patient, 5=highly active). TA does NOT require a trade every tick.
- Asset Risk Preference (Risk): 2 / 5 - Which tokens you consider (1=prefer least volatile available, 5=embrace high volatility). Risk determines which tokens to consider, not whether to trade at all.
- Trade Size (Size): 3 / 5 - Maximum position sizing per trade (1=up to ~15%, 2=up to ~30%, 3=up to ~50%, 4=up to ~70%, 5=up to ~90% of available ETH). These are UPPER BOUNDS, not minimums. A small trade can still be valid.
- Holding Style (Hold): 2 / 5 - Minimum hold time before considering an exit (1=about ~30 minutes minimum; 2=about ~1 hour; 3=hours; 4=many hours; 5=days). Do NOT sell before minimum hold unless an exceptional reason exists (thesis broken, stop-loss, or explicit [HIGH] directive). When in doubt about whether to sell, holding is usually correct — positions need time to develop.
- Diversification (Div): 4 / 5 - Portfolio spread (1=concentrated in 1-2 tokens; 2=focused 2-3; 3=balanced 3-5; 4=spread 4-6; 5=wide 5+). Div=1-2 should usually add to existing positions instead of opening many new ones.

**If any slider is 0, treat it as 3 (balanced) and mention in reasoning that the slider was not configured.**

**Sell sizing**: Trade Size guides sells as well as buys. Size=1-2: small trims (10-30%). Size=3: moderate trims (20-50%). Size=4-5: larger exits (30-70%) when conviction is high. Prefer smaller trims over larger exits — you can always sell more later, but you cannot undo a sell. Full 100% exits should be rare — they require either a [HIGH] directive that explicitly says "sell all" or "liquidate", or a genuine fundamental thesis break. A price drop — even a large, fast one — is NOT a thesis break in a tournament where 10-20% swings are normal.

**Active + low-risk note (TA=5, Risk=2):** Low risk means prefer relatively less volatile options among available tokens. Avoid sell-rebuy oscillations caused by fabricated volatility thresholds.

**Fresh-signal gate (TA=5):** High activity means acting on fresh information. If your planned trade repeats a recent same-token/same-direction action without meaningful new evidence, OBSERVE.

**Maximum activity note (TA=5):** You are very active, but still avoid reflexive every-tick churn. Under active [HIGH], execute immediately. Under slider-only logic, require fresh edge.

**Low ETH note (0.0054 ETH):** Low ETH means you are deployed — that is a valid state. Monitor positions for genuine exit signals (stop-loss, thesis broken), not to restore a buffer. Do NOT sell just because ETH is low.

------------------------------

## PORTFOLIO CONTEXT

ETH sitting idle earns nothing — your job is to find opportunities and deploy into tokens. Once deployed, focus on HOLDING and monitoring for genuine exit signals — not continuously trading in and out. Positions need time to develop; the best returns come from conviction holds, not from constant rotation. If ETH is near zero and all positions have a valid thesis, that is a healthy state — do NOT sell just to rebuild an ETH buffer. If mostly ETH, look for quality entries.

**Note**: Unrealized PnL shown below is per-token only. You do NOT have vault-level total PnL. Do not treat one token's unrealized gain as your overall performance.

- ETH: Balance: 0.006000
- HOTDOGZ: Balance: 25520894187694874165248.000 | Avg Entry: 0.000000210156227185 ETH | Unrealized PnL: +9.51% | Time Since Last Interaction: 49m | Time Held: 1d 17h

## CONSTRAINTS

- Max Trade Amount (Percent): 12.00% of available ETH - **[HIGH] strategies may exceed this limit.**

## PRICE IMPACT LIMITS (max 400 bps)

Max sizes that stay within your price impact tolerance. **[HIGH] strategies may exceed these limits** if needed to fulfill explicit directives.

- AIGF: BUY max 0.00% of ETH
- HOLE: BUY max 0.00% of ETH
- HOTDOGZ: BUY max 12.00% of ETH, SELL max 100.00% of HOTDOGZ
- LMAO: BUY max 0.00% of ETH
- LOOKSMAX: BUY max 0.00% of ETH
- POOPCOIN: BUY max 0.00% of ETH
```

Full verbatim real raw prompt files:

- [Real `market_only` system prompt](/Users/brockelmore/concordance/xenon/data/report_assets/postmarket_context_geometry_evidence/raw_prompts/real_market_only_system.txt)
- [Real `market_only` user prompt](/Users/brockelmore/concordance/xenon/data/report_assets/postmarket_context_geometry_evidence/raw_prompts/real_market_only_user.txt)
- [Real `affordance_4` system prompt](/Users/brockelmore/concordance/xenon/data/report_assets/postmarket_context_geometry_evidence/raw_prompts/real_affordance_4_system.txt)
- [Real `affordance_4` user prompt](/Users/brockelmore/concordance/xenon/data/report_assets/postmarket_context_geometry_evidence/raw_prompts/real_affordance_4_user.txt)


## Main Hypothesis

The model does not rely on one single global market axis or one single universal market manifold.

Instead, it likely does three separable things:

1. It builds a market representation when it reads the market block
2. It modifies or reweights that representation depending on context
3. It carries the changed representation forward into downstream section states

The cleanest open questions are:

- what axes structure the market representation itself?
- are those axes about raw metrics, roster-relative structure, or both?
- does context change perception of the market, later integration of the market, or both?


## Proposed Next Program

### Phase 1: Market-Basis Discovery

Goal:

- infer the major directions the model uses for market encoding instead of assuming them in advance

Setup:

- run many `market-only` prompts
- use real DX prompt structure wherever possible
- no context edits in this phase

Captures:

- `market_mean`
- `market_eos`
- optionally row-level pooled states as a secondary read

Analysis:

1. PCA across prompts
2. variance explained by top components
3. correlation of top components with nuisance variables:
   - prompt length
   - token count
   - formatting or section-length variation
   - roster width
4. correlation of top components with market variables:
   - price changes
   - flow
   - volume
   - participation
   - concentration
   - spreads / route limits where applicable
5. correlation with roster-relative variables:
   - within-roster rank
   - z-score within roster
   - gap from leader
   - gap from second-best
6. metric-to-activation fitting:
   - ridge / PLS / CCA from market metrics to activations

Why this matters:

- PCA tells us where the variance is
- correlation and regression tell us what those directions mean
- this gives us candidate axes that are closer to the model’s own representation than `strength/quality`


### Phase 2: Context Comparison With Order Controls

Goal:

- separate market perception from later market-context integration

For each matched market, create three conditions:

- `A`: market only
- `B`: market then context after
- `C`: context before then market

Capture points:

- `market_eos` for `A`, `B`, and `C`
- `last_token` for `B` and `C`
- optionally `active_settings_eos`, `portfolio_eos`, and `constraints_eos`

Core tests:

1. Sanity check: `market_eos(A)` vs `market_eos(B)`
   - These should be near-identical if the prefix up to `market_eos` is truly the same

2. Perception test: `market_eos(A)` vs `market_eos(C)`
   - If context-before-market changes `market_eos`, then context is changing the encoding of the market as it is being read

3. Integration test: `last_token(B)` vs `last_token(C)`
   - If these converge, then order matters less by the end
   - If they do not, then context order leaves a persistent representational difference

4. Projection into the Phase 1 basis
   - Use the discovered Phase 1 basis as a visualization frame
   - Ask where the differences live

Important note:

- the Phase 1 basis should be treated as a discovery and visualization basis, not automatically as the “true semantic basis”


### Phase 3: Rebuild The Synthetic Dataset

Goal:

- keep the control advantages of synthetic data, but stop making it look obviously synthetic

Changes:

1. Remove explicit synthetic markers
   - no `SYNTHETIC MARKET SCENARIO`
   - no `Archetype`

2. Match real DX structure closely
   - same system prompt style
   - same section order
   - same slider language
   - same constraint phrasing
   - same route and price-impact style

3. Move from hand-bundled axes to discovered or unbundled factors
   - separate momentum, flow, participation, concentration, volatility/freshness
   - do not hide several concepts inside one “quality” axis

4. Prefer realistic roster widths
   - use `6`-asset rosters for the main bridge-oriented synthetic work

Why:

- the current synthetic dataset is still useful as a controlled scaffold
- but it is not realistic enough to support strong synthetic-to-real transfer claims


### Phase 4: Re-run The Strongest Ladder In The New Basis

Goal:

- test the strongest current family with better prompts and better axes

Recommended priority:

1. affordance
2. risk
3. portfolio

Why this order:

- affordance is already the clearest real result
- risk is weaker and should be revisited only after the basis and prompt surface are improved


## What We Expect To Learn

This program should answer five high-value questions:

1. What directions dominate market representation before any context is added?
2. Are those directions mostly raw-metric directions, roster-relative directions, or both?
3. Does context alter market perception itself, or mostly later integration?
4. Are affordance and risk modifying the same market basis in different ways, or different bases entirely?
5. Do better synthetic prompts and learned axes improve synthetic-to-real alignment?


## Why This Is Better Than The Current Synthetic Basis

The current `strength/quality` basis was a good bootstrap.

It gave us:

- controlled latent geometry
- same-rank / different-shape comparisons
- a clean testbed for the analysis pipeline

But it is not the right long-term object because:

- it is too human-designed
- it bundles too much
- it may not match how the model itself organizes the market

The proposed program keeps the good part:

- controlled comparisons

And replaces the weak part:

- hand-imposed axes that may not be the model’s own


## Practical Execution Order

1. Run Phase 1 discovery on market-only prompts
2. Build the A/B/C context-order experiment
3. Inspect whether the discovered basis is dominated by nuisance or by meaningful market variables
4. Rebuild synthetic prompts to match DX
5. Re-run the affordance ladder first in the new setup
6. Compare old synthetic basis vs discovered basis directly


## Bottom Line

The next phase should move from:

- “we define the latent axes, then see whether geometry appears”

to:

- “we discover the model’s market basis first, then test how context changes it”

That is the cleanest way to address both current concerns:

- the synthetic prompts are too artificial
- and the current latent coordinates are too hand-designed
