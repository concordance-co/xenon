# Phase 13 Medium Settings Validation Prereg

Date: 2026-04-24

## Purpose

Validate the smoke-run finding that the synthetic conflict directions separate
real complaint rows within the complaint corpus after settings have been read.

This is a targeted confirmation run, not a new grid search.

## Run Scope

- Dataset: `dx_terminal_signal_discovery_phase13_v1`
- Prompt tier: `aggressive`
- Capture sites: end-of-section only
- Primary cell: `L32 settings_end`
- Primary directions:
  - `trade_size`
  - `shared_mean`

The run may still compute the existing end-site grid because that is how the
workflow is currently structured, but interpretation is preregistered around the
primary cell above.

## Primary Readout

Rank complaint rows by projection at `L32 settings_end`.

For each primary direction, compare:

- top-25 complaints by projection
- bottom-25 complaints closest to the structure-matched-control mean

Use available Neon metadata plus prompt/decision read-through.

## Confirmation Criteria

Confirm if both primary directions show a clear top-vs-bottom semantic delta:

- top-25 is enriched for `USER_CONFIG_CONFLICT` and/or
  `config_conflict_like=true`, target threshold >= 60% of rows.
- bottom-25 is enriched for `RULE_FABRICATION` or other non-config-conflict
  complaint types, target threshold >= 60% of rows.
- top examples read as settings/action-governor or strategy/settings conflict
  failures more often than bottom examples.

The exact threshold is not a p-value. It is a guardrail against interpreting a
weak or post-hoc split as confirmation.

## Kill Criteria

Kill this finding if top-25 and bottom-25 have similar category distributions
and prompt reads do not show a meaningful semantic contrast.

## Ambiguous Criteria

Treat as ambiguous if one primary direction confirms and the other does not, or
if the category enrichment is partial but prompt reads show a plausible
semantic difference.

Ambiguous result means do not scale to the full Phase 13 grid yet. Next move
would be label audit and possibly a larger targeted run, not causal claims.

## Label Assumption To Check First

Spot-check `RULE_FABRICATION` complaint rows before running. The intended
interpretation is that most `RULE_FABRICATION` rows are not direct
strategy-settings conflicts. If many are actually strategy/settings conflicts
with a secondary rule-fabrication label, the top-vs-bottom category story is
muddier.

## Label Spot-Check Outcome

Spot-checked the first 20 aggressive-tier complaint rows labeled
`RULE_FABRICATION` in `dx_terminal_signal_discovery_phase13_v1`.

Interpretation:

- The spot-check does not support treating `RULE_FABRICATION` as a pure
  "non-conflict complaint" bucket.
- It does support treating `RULE_FABRICATION` as mostly distinct from direct
  slider/settings conflict: sampled rows were marked `config_conflict_like=false`.
- Many sampled rows were strategy-execution or invented-rule failures:
  unwanted buys/sells, strategy ignored, holding violation, cooldown/position
  logic, or active-strategy execution failures.

Therefore the medium run's top-vs-bottom contrast should be interpreted as:

- high projection enriched for direct config/settings conflict, if confirmed;
- low projection enriched for rule/strategy-execution failure modes, not
  necessarily "no conflict" in the broad sense.
