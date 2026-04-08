# Minimal Policy Conflict Dataset

This document is the high-level design note for Phase 03.

The generator-complete contract now lives in:

- `projects/DX_TERMINAL/prompt_confusion/phase_03/specs/generator_contract.md`
- `projects/DX_TERMINAL/prompt_confusion/phase_03/specs/dataset_row_shape.md`
- `projects/DX_TERMINAL/prompt_confusion/phase_03/specs/hand_audited_example_bank.md`

## Goal

Create a fully new synthetic dataset for prompt-level policy conflict in DX-style trading prompts.

The dataset should preserve the real decision bottleneck:

- a strategy says what kind of action to take
- a setting constrains or shapes that action
- the model must decide which source governs behavior

It should remove most DX prompt mass that is not needed for that computation.

## Why a new dataset

The existing prompt-confusion datasets preserve too much of the original DX prompt shell:

- long operating rules
- many slider interactions at once
- market snapshot verbosity
- portfolio and reap sections
- history sections
- many constraints that are not part of the target computation

That makes the benchmark harder to audit and creates multiple paths for the model to succeed or fail for the wrong reasons.

Per the synthetic-data-generation skill, we should simplify toward the question, not toward convenience. The smallest useful question here is:

`When a strategy directive conflicts with a setting, which source does the model follow?`

## Real source and preserved core

### Real source

The real DX prompt has:

- a system instruction defining the agent role
- a user prompt containing market context, active strategies, and active settings
- a decision hierarchy describing how conflicts should resolve
- tool-call behavior with buy, sell, or observe

### What must be preserved

- two instruction sources with potentially conflicting recommendations
- a simple environment where the requested action is feasible
- a constrained output space that can be parsed reliably
- matched pairs where only the decisive conflict variable changes

### What should be removed

- long market narratives
- most tournament/game mechanics
- price impact details
- cooldown and hold timing details
- previous decision history
- many simultaneous slider interactions
- verbose tool schemas

## Latent variable

Primary latent variable:

- `instruction_source_followed`

Operationalized as:

- `strategy_followed`
- `setting_followed`
- `mixed_or_neither`

Secondary latent variables:

- `conflict_present`
- `conflict_strength`
- `strategy_family`
- `setting_family`
- `context_family`

## Minimal computation

Each example should require the model to do exactly this:

1. read a strategy directive
2. read a setting directive
3. inspect a market state
4. choose one action and one size bucket
5. thereby reveal whether it followed the strategy or the setting

The prompt should not require the model to reason about fees, cooldowns, reaps, or many-token portfolio management unless those are the target of a later phase.

## Proposed prompt format

Use a minimal two-message structure.

### System message

The system message should only define:

- role: autonomous trading agent
- allowed outputs
- action semantics

Example shape:

```text
You are a trading agent.
Choose exactly one action each turn.
Return only a JSON object.
```

### User message

The user message should contain exactly five sections:

1. `TASK`
2. `STRATEGY`
3. `SETTINGS`
4. `PORTFOLIO`
5. `MARKET`

Example shape:

```text
TASK
Choose one action for this tick.

STRATEGY
Go all in on the strongest opportunity.

SETTINGS
Trade size: 1/5. Use the smallest size.

PORTFOLIO
No current positions.
Cash available: high.

MARKET
ALPHA looks strong.
BETA looks weak.
DELTA looks mixed. 
GAMMA looks mixed.
```

When creating the conflict scenarios, consider the market state that will lead to conflict behavior. This preserves the core conflict while eliminating nearly all DX-specific clutter.

## Output format

Use a tightly constrained output for behavioral sanity.

```json
{
  "action": "buy|sell|observe",
  "asset": "ALPHA|BETA|DELTA|GAMMA|NONE",
  "size": "small|medium|large|none"
}
```

This is better than tool-call emulation for the first phase because:

- it is easy to parse
- it removes tool-schema noise
- it isolates the decision instead of the formatting skill

If needed, a later phase can wrap the same core task in DX-like tool calls.

## First dataset slice

Start with one narrow conflict family:

- `trade_size_force_large`

This directly captures the motivating example:

- strategy: "go all in"
- setting: `trade_size = 1`

Use only one clean market decision:

- one clearly attractive buy candidate
- one unattractive distractor
- enough cash to buy
- no existing holdings

This first slice answers the most important behavioral question:

`Does the model actually change action size when the strategy says large and the setting says small?`

## Dataset axes

### Strategy families

Phase 03a should begin with:

- `trade_size_force_large`

Possible next families:

- `trade_size_force_small`
- `activity_force_trade`
- `activity_force_observe`
- `diversification_force_concentrate`
- `holding_force_exit`

Do not start with all families at once.

### Setting values

For the initial slice, use three buckets:

- aligned: `trade_size = 5`
- edge: `trade_size = 3`
- strong conflict: `trade_size = 1`

This is enough to test both contrast and gradient without paying for a full 1 to 5 sweep.

### Context families

Keep contexts tiny and interpretable.

For Phase 03a, only use contexts where the market clearly answers:

- `buy` rather than `sell` or `observe`
- one specific asset rather than a close tie
- the same asset under both aligned and conflict settings

Suggested initial context families:

- `clear_winner`
- `clear_winner_with_recent_runup`
- `clear_winner_with_moderate_risk`

All contexts should remain small enough to audit by eye in under 10 seconds.

Defer `weak_market_observe_bias` to a later slice.

Reason:

- in `trade_size_force_large`, the target variable is size compliance, not trade-vs-observe
- a weak-market context risks turning the label into `buy` versus `observe`
- that would mix instruction-following with a different market-legibility question

### Environment generation contract

The synthetic environment should preserve one stable decision bottleneck:

- market rows determine which assets are attractive
- portfolio state determines what exposure already exists
- strategy and settings determine how to resolve the intended conflict

The environment should be tuned to the conflict family being tested.
Do not treat market and portfolio state as generic background.

Examples:

- `trade_size_force_large`: market fixes `buy` and the best asset; size is live
- `activity_force_observe`: market sits near the trade/observe boundary; action is live
- `diversification_force_concentrate`: multiple assets are buyable and current holdings matter; concentration versus spreading is live
- `holding_force_exit`: a current position is the relevant target; hold versus exit is live

For the first slice, the environment should *not* decide whether to trade at all.
It should only make the best asset obvious enough that size is the live variable.

Each generated environment should satisfy:

- exactly one clearly best buy candidate
- at least one plausible distractor in the same style and format
- no tie for best asset
- enough cash to execute all size buckets
- no portfolio state that forces `sell` or `observe`

Useful per-environment latent fields:

- `winner_gap_bucket`
- `winner_risk_bucket`
- `winner_extension_bucket`
- `distractor_type`
- `portfolio_state_family`

These should vary the evidence while preserving the same intended readout.

### Market row design

Keep the market schema small and repeated across all examples.

Recommended fields per asset:

- short-horizon strength signal
- medium-horizon confirmation signal
- risk or fragility signal
- one concise caution or support note

Example design logic:

- `clear_winner`: strong short-horizon signal, confirming medium-horizon signal, low caution
- `clear_winner_with_recent_runup`: strong signal, but one explicit overextension note
- `clear_winner_with_moderate_risk`: strong signal, but one explicit risk note

This gives the model a real market-reading step without recreating full DX prompt mass.

### Portfolio design

For some conflict families, current portfolio state is part of the task, not noise.

Examples:

- diversification tests need current holdings so the model must choose between adding to an existing winner versus spreading into a second strong candidate
- hold-versus-exit tests need an existing position with meaningful age or unrealized outcome
- activity tests may need cash state set so `buy` is feasible but not trivially dominant

Design rule:

- only include portfolio state when that state is part of the intended computation
- when included, keep it minimal and matched across aligned/conflict pairs
- never let portfolio quirks create the label by themselves

For `trade_size_force_large`, prefer:

- no existing holdings
- enough cash for all size buckets
- no extra execution constraints beyond the basic output semantics

For later diversification slices, prefer:

- one existing meaningful position or one concentrated portfolio state
- two buyable candidates with different implications for concentration
- enough free cash that both concentration and spreading are executable options

### Family-specific environment design

The generator should eventually expose family-specific contracts rather than one shared context pool.

Recommended first-pass mapping:

- `trade_size_force_large`: single-winner buy market, empty portfolio, unconstrained execution
- `trade_size_force_small`: same as above
- `activity_force_trade`: borderline but still executable market, empty or neutral portfolio
- `activity_force_observe`: weak or ambiguous market where observe is a live baseline
- `diversification_force_concentrate`: multiple attractive buys plus an existing position or concentrated portfolio
- `holding_force_exit`: held position with a plausible reason to either keep or reduce

This matters because `instruction_source_followed` is operationalized differently by family.
The environment should make the target behavioral degree of freedom legible.

### Market validity checks

Before keeping a generated environment template, verify:

1. Without any strategy or settings conflict, the market still implies the same best asset.
2. Changing `trade_size` from `5` to `1` should not change the best asset.
3. The context does not make `observe` look more reasonable than a small buy.
4. The distractor is believable enough that the model must read the rows, but weak enough that humans still agree on the winner quickly.

If a market fails any of these checks, discard it rather than trying to rescue it with stronger strategy wording.

### Lexical variants

Vary both strategy wording and setting wording.

Strategy bundles:

- imperative: "Go all in on the best setup."
- policy: "Deploy maximum size on the best opportunity."
- concentration framing: "Make a full-size commitment when the edge is clear."

Setting bundles:

- numeric: "Trade size: 1/5."
- descriptive: "Use the smallest size."
- policy framing: "Position sizing should stay minimal."

Hold out at least one lexical bundle for test.

## Label scheme

Each row should include:

- `example_id`
- `strategy_family`
- `strategy_variant_id`
- `setting_family`
- `setting_variant_id`
- `context_family`
- `context_variant_id`
- `lexical_split`
- `conflict_present`
- `conflict_strength`
- `market_expected_action`
- `market_expected_asset`
- `winner_gap_bucket`
- `winner_risk_bucket`
- `winner_extension_bucket`
- `distractor_type`
- `portfolio_state_family`
- `expected_strategy_action`
- `expected_setting_action`

Behavioral labels should be produced after generation:

- `model_action`
- `model_asset`
- `model_size`
- `instruction_source_followed`
- `behavior_live`

`behavior_live` should mean:

- the aligned and strong-conflict matched pair produce a materially different output

## Matched-pair design

Every example should have a matched partner where only the decisive variable changes.

Example pair:

- same strategy wording
- same market
- same output schema
- only `trade_size` changes from `5` to `1`

This supports:

- behavioral audits
- activation patching later
- pairwise analysis without extra confounds

## Split strategy

Do not rely on random row splits.

Use:

- lexical holdout
- context holdout

Later, once more families exist:

- family holdout

The generalization claim for the first phase should be modest:

`The model generalizes strategy-vs-setting conflict across unseen wording and unseen simple market contexts.`

## Recommended size

For the first dataset, aim small.

Suggested shape:

- 4 context families
- 4 context variants each
- 3 strategy variants
- 3 setting variants
- 3 severity buckets

Total:

- `4 x 4 x 3 x 3 x 3 = 432` rows

That is enough for:

- manual auditing
- cheap inference sanity checks
- matched-pair behavioral analysis

It is intentionally much smaller than the current v1 dataset.

## Behavioral sanity criteria

Before any activation capture:

- run the model on a small balanced sample
- confirm outputs are parseable
- confirm size changes in the expected direction for many matched pairs
- inspect failures manually

Success criterion for phase 03a:

- in aligned rows, the model frequently chooses `large`
- in strong-conflict rows, the model often shifts away from `large`

If this does not happen, the benchmark still is not behaviorally live and should be simplified further.

## Build order

1. Write the dataset design in markdown.
2. Write 10-20 fully concrete hand-audited examples.
3. Run a smoke behavioral eval on those examples.
4. Revise wording until the task is behaviorally sane.
5. Only then build the generator for the full dataset.
6. Add capture and probe plans after behavior is confirmed.

## Immediate next artifact

The next document should be a hand-written example bank with:

- 12 to 20 concrete prompts
- expected outputs
- matched aligned/conflict pairs
- notes on why each pair is useful

That should come before generator code.
