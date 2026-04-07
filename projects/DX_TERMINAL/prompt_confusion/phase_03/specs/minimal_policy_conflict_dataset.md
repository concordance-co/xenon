# Minimal Policy Conflict Dataset

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

1. read a simple strategy directive
2. read a simple setting directive
3. inspect a tiny market state
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

The user message should contain exactly four sections:

1. `TASK`
2. `STRATEGY`
3. `SETTINGS`
4. `MARKET`

Example shape:

```text
TASK
Choose one action for this tick.

STRATEGY
Go all in on the strongest opportunity.

SETTINGS
Trade size: 1/5. Use the smallest size.

MARKET
ALPHA looks strongest.
BETA looks weak.
Current cash: high.
```

This preserves the core conflict while eliminating nearly all DX-specific clutter.

## Output format

Use a tightly constrained output for behavioral sanity.

```json
{
  "action": "buy|sell|observe",
  "asset": "ALPHA|BETA|NONE",
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

Do not start with all families at once.

### Setting values

For the initial slice, use three buckets:

- aligned: `trade_size = 5`
- edge: `trade_size = 3`
- strong conflict: `trade_size = 1`

This is enough to test both contrast and gradient without paying for a full 1 to 5 sweep.

### Context families

Keep contexts tiny and interpretable.

Suggested initial context families:

- `clear_winner`
- `clear_winner_with_recent_runup`
- `clear_winner_with_moderate_risk`
- `weak_market_observe_bias`

All contexts should remain small enough to audit by eye in under 10 seconds.

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
