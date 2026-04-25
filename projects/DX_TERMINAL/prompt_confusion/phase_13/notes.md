# Phase 13 Notes

Working notes for the real-transfer signal read. This is intentionally less
polished than the report. It keeps the precise interpretation, caveats, and
example shapes we do not want to lose.

## Current Best Read

Synthetic probes trained on clean policy-source conflict in controlled prompts
transfer to real DX Terminal prompts at L32 `settings_end`.

The strongest current claim is not "the probe detects complaint rows." It is:

- `trade_size` fires on current-prefix, concrete sized-action conflict:
  wrong buy/sell side, wrong allocation, or explicit size mismatch.
- `shared_mean` fires on a broader policy-tension component, but its top rows
  currently overlap the `trade_size` top rows in the medium read.
- Low projection rows can still be legitimate complaints. They often are not
  clean `trade_size` probe targets because the conflict evidence lives in
  history, strategy lifecycle state, bookkeeping, or interpretation.

Plain version:

> High projection is not just "complaint exists." It seems to prefer conflicts
> that are current, concrete, and action/size-shaped.

## Primary Medium Result

Medium run:

- run id: `wr_14f78308dbac_dbc78513`
- corpus table: `dx_terminal_signal_discovery_phase13_v1`
- tier: aggressive
- sites: ends-only
- primary cell: L32 `settings_end`

Primary means:

| Direction | Anchor | Complaint | Structure control | Anchor-control | Complaint-control |
| --- | ---: | ---: | ---: | ---: | ---: |
| `trade_size` | 4.425 | 3.803 | 3.278 | +1.147 | +0.526 |
| `shared_mean` | 3.462 | 3.137 | 2.760 | +0.703 | +0.377 |

This is a positive transfer read at the cohort level.

Important family-specific caveat:

- `risk_preference` was weaker at this exact cell.
- `diversification_preference` did not show clean ordering at this exact cell.
- Therefore the result is not "all synthetic vectors light up on real
  complaints."

## Top/Bottom Read

Review artifact:

`reports/signal_discovery/report_dd8c8ac3385c_7e82ff1b/results/l32_settings_top25_complaint_review.json`

Compact handoff artifact:

`reports/signal_discovery/report_dd8c8ac3385c_7e82ff1b/results/l32_settings_trade_size_audit_packet.json`

Top/bottom means:

- top = highest complaint projections
- bottom = complaints closest to the structure-control mean

Counts:

| Direction | Top action/size | Top strategy ignored | Bottom action/size | Bottom strategy ignored |
| --- | ---: | ---: | ---: | ---: |
| `trade_size` | 20/25 | 5/25 | 15/25 | 10/25 |
| `shared_mean` | 20/25 | 5/25 | 9/25 | 16/25 |

Top `trade_size` complaint types:

- `UNWANTED_BUY`: 10/25
- `UNWANTED_SELL`: 6/25
- `WRONG_SIZE`: 4/25
- combined concrete action/size: 20/25
- `STRATEGY_IGNORED`: 5/25

The high `shared_mean` top-25 is the same row set as high `trade_size`, but
with lower magnitude. The clearest top/bottom contrast is `shared_mean` bottom,
where `STRATEGY_IGNORED` dominates.

## Important Correction: Root Cause Labels Were The Wrong Proxy

The preregistered proxy was too coarse:

- expected high: `USER_CONFIG_CONFLICT` / `config_conflict_like`
- expected low: `RULE_FABRICATION` / non-config

That was not the right target for the `trade_size` probe.

Why:

- `USER_CONFIG_CONFLICT` and `RULE_FABRICATION` diagnose why the complaint
  happened.
- The `trade_size` probe target is closer to visible prompt conflict shape:
  does the current prompt contain a concrete buy/sell/size mismatch?
- A `RULE_FABRICATION` row can still contain a wrong-size trade.
- A `USER_CONFIG_CONFLICT` row can be a stale-history complaint with no active
  strategy visible in the current prompt prefix.

Observed counts make this obvious:

- `trade_size` top-25: 17/25 `USER_CONFIG_CONFLICT`
- `trade_size` bottom-25: 20/25 `USER_CONFIG_CONFLICT`

So the root-cause split did not validate the signal. It was the wrong semantic
slice for this probe.

## Shape Of The Bottom Rows

Reading the bottom rows manually, the conflict shape is mostly not:

> "agent simply ignored a clear current instruction."

It is more like:

> temporal / bookkeeping / stale-strategy conflict.

Main patterns:

### 1. No active strategy visible, but complaint references an old or expected strategy

Example:

- complaint: "why didn't you lock in some gains when i asked you to?"
- prompt says: `No active strategies.`
- decision: sells HOTDOGZ 100%.
- shape: complaint is about prior/history expectation, but the current prompt
  prefix does not expose a live strategy conflict.

Why this matters:

At `settings_end`, the probe has seen the active strategy section and settings.
If the active strategy section says no active strategies, the conflict is not a
clean current-prefix policy-source mismatch even if the user complaint is valid
historically.

### 2. The agent is taking the requested action, so the conflict is retrospective

Example:

- complaint: "why you not buying AIGF as I said?"
- active strategy: allocate all available ETH to AIGF and hold indefinitely.
- decision: buys AIGF 100%.
- shape: not a current-tick mismatch. The agent may have failed earlier, but
  this row's visible action is aligned.

Why this matters:

This kind of row can be a real complaint, but it is not a clean readout target
for a current-prefix conflict probe.

### 3. Multi-step strategy execution, where current action is only one phase

Example:

- complaint: "wanted HOLE"
- active strategy says: sell POOPCOIN first, then deploy into HOLE.
- decision: sells POOPCOIN.
- shape: user complaint is about the final desired state, while the current
  tick is an intermediate step.

Why this matters:

The model may be following a reasonable first step. The conflict depends on
whether the system later completes the strategy.

### 4. Strategy/rule interpretation conflict, not pure size conflict

Examples:

- "sell every time you have a profit"
- "why is strategy 30 minutes, not 30 seconds"
- "you bought higher than my entry"

Shape:

- disagreement is about how to interpret strategy conditions,
  time windows, entry prices, fulfillment state, or prior decisions.
- not a crisp "strategy says X size/action, current visible prompt implies Y"
  mismatch.

### 5. Partial execution exists, but it is not the dominant bottom shape

Example:

- complaint: "SELL IT FULL"
- strategy: sell all POOPCOIN
- decision: sells 50%.

This is a real size/action mismatch. It belongs closer to the trade-size target
than many other bottom rows. But many bottom rows are not this clean, and the
bottom set as a whole is dominated by history/lifecycle/interpretation shape.

## What The Probe May Be Reading

Hypothesis after the medium read:

- `trade_size` is most sensitive to prompt-local sized-action tension.
- It prefers cases where the current visible prompt makes the buy/sell/size
  issue legible.
- It is less sensitive to complaints where the relevant evidence is outside
  the current prefix or depends on lifecycle bookkeeping.

The practical distinction:

- high projection: "this prompt currently contains a concrete trading action or
  sizing conflict."
- low projection: "this may be a valid complaint, but the conflict is not
  primarily a current-prefix size/action conflict."

This is good news. It means the probe is not just reading the complaint stratum.
It is selecting for something closer to the synthetic training geometry.

## High Rows: Qualitative Shape

The top rows are more often immediate, concrete action complaints.

Common shapes:

- "why did you buy HOTDOGZ?"
- "why did you buy so much POOPCOIN?"
- "Buy available balance 30%, not 10 ETH"
- "Quit buying tokens. Liquidate..."
- "You are under allocated to POOPCOIN"

These are closer to the synthetic `trade_size` family because the conflict is
about concrete action or allocation. Some are still messy. A few top rows have
label/context oddities where the visible decision appears aligned or the
complaint seems to refer to a prior tick. But the top slice is much more
action/size-shaped than the low `shared_mean` slice.

## Shared Direction

The current `shared_mean` read is subtle:

- Top-25 is the same row set as `trade_size` top-25.
- That means in this medium slice, `shared_mean` did not discover a distinct
  high-projection family.
- But its bottom slice is more strongly enriched for diffuse
  `STRATEGY_IGNORED` rows: 16/25.

Interpretation:

- high `shared_mean`: still largely concrete action-conflict rows in this cell.
- low `shared_mean`: especially avoids diffuse strategy narrative rows.
- if we want to prove `shared_mean` captures broader policy tension, we need a
  separate audit that looks for non-size high-shared rows outside the same
  top-k overlap or at neighboring cells.

## Current Claim Boundary

Safe claim:

> Fixed synthetic directions recover a real production signal at L32
> `settings_end`. `trade_size` is selective for current-prefix concrete
> sized-action conflict. `shared_mean` tracks broader policy tension, while
> firing less on temporal, bookkeeping, and interpretation-layer complaints.

Avoid for now:

- "The probe correctly classifies complaints."
- "The probe detects all real conflicts."
- "The bottom rows are non-conflicts."
- "The shared direction has already proven a separate broad-conflict top-k
  family."

Better phrasing:

- "fires less strongly"
- "tends to downweight"
- "selects for current-prefix conflict shape"
- "root-cause labels are metadata, not gold labels for this readout"

## Proposed Hand Label Schema

For the next audit, label top/bottom rows by conflict shape:

1. `current_action_size_conflict`
   - visible prompt/action contains a concrete wrong side, wrong token, wrong
     allocation, or explicit size mismatch.

2. `retrospective_history_conflict`
   - complaint appears valid only by reference to prior ticks, old directives,
     or a pattern over time.

3. `strategy_fulfillment_conflict`
   - disagreement is about whether a multi-step strategy is fulfilled, blocked,
     in-progress, or complete.

4. `interpretation_or_rule_conflict`
   - disagreement depends on interpreting thresholds, entry prices, timing,
     cooldowns, sell rules, or strategy language.

5. `unclear_or_label_mismatch`
   - visible prompt/action does not make the complaint legible, or the metadata
     label appears inconsistent with the row.

Likely confirmatory pattern:

- high `trade_size` enriched for `current_action_size_conflict`
- low `shared_mean` enriched for `retrospective_history_conflict`,
  `strategy_fulfillment_conflict`, and `interpretation_or_rule_conflict`

## Files To Keep In Mind

Report:

- `reports/PHASE13_REAL_TRANSFER_SIGNAL_BRIEF_2026_04_24.pdf`
- `reports/PHASE13_REAL_TRANSFER_SIGNAL_BRIEF_2026_04_24.typ`

Primary review JSON:

- `reports/signal_discovery/report_dd8c8ac3385c_7e82ff1b/results/l32_settings_top25_complaint_review.json`

Agent-facing audit packet:

- `reports/signal_discovery/report_dd8c8ac3385c_7e82ff1b/results/l32_settings_trade_size_audit_packet.json`

Medium report:

- `reports/signal_discovery/report_dd8c8ac3385c_7e82ff1b/report.md`

Pre-registration:

- `medium_settings_validation_prereg.md`

## Open Threads

- Need true non-complaint production baseline controls eventually. Current
  `structure_matched_control` is useful but not the same thing.
- Need hand labels before making a strong within-complaint semantic separation
  claim.
- Need to decide whether to inspect neighboring cells after hand-labeling:
  likely L28/L32/L36 `settings_end`, not a broad grid search.
- Need to test whether `shared_mean` has a distinct high-projection family, or
  whether it is mostly a lower-magnitude version of `trade_size` in this cell.
