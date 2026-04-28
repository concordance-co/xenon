# Phase 15: Mid-Prompt Direction Real Transfer

## Premise

Phase 13 found the strongest real-transfer read at real `settings_end`, but it
used synthetic prompt-EOS directions. Phase 14 showed that the settled
three-family synthetic benchmark has strong section-local probes and directions
at `settings_end`, `portfolio_end`, and `market_end`.

Question:

> Do Phase 14 section-local synthetic directions transfer to real DX Terminal
> prompts better than the older prompt-EOS direction bank?

## Design

Reuse the completed Phase 13 medium real capture:

- Run: `wr_14f78308dbac_dbc78513`
- Capture: `capture_1_bbfed191794c`
- Corpus: `dx_terminal_signal_discovery_phase13_v1`
- Tier: `aggressive`
- Sites: end-of-section real sites

Use Phase 14 direction artifacts as the synthetic direction bank:

- `strategies_end`
- `settings_end`
- `portfolio_end`
- `market_end`
- `prompt_eos`

Evaluation surface:

- project each Phase 14 bank site onto each available real end position
- include `trade_size`, `risk_preference`, `diversification_preference`, and
  normalized `shared_mean`
- report stratum means and cohort deltas:
  - `anchor_positive - structure_matched_control`
  - `complaint - structure_matched_control`
- flag matched-site cells separately:
  - `settings_end -> settings_end`
  - `portfolio_end -> portfolio_end`
  - `market_end -> market_end`
  - `prompt_eos -> prompt_im_end`

## Claim Boundary

This is still a projection transfer read, not a trained real classifier. It can
say which fixed synthetic direction bank gives cleaner cohort separation on the
same real capture. It cannot by itself prove real-data ground-truth accuracy or
causal leverage.

## What We Ran

Completed run:

- Run: `wr_13b5e0c84804_b0cafa71`
- Phase 14 direction bank: `transform_1_f363eb6e`
- Real-transfer grid: `transform_1_2a9fc514`
- Report: `report_5e9dd8308398_4940ce8a`
- Report path:
  `projects/DX_TERMINAL/prompt_confusion/phase_15/reports/mid_prompt_real_transfer/report_5e9dd8308398_4940ce8a/report.md`

Focused audit rerun:

- Run: `wr_13b5e0c84804_7e83dad4`
- Phase 14 direction bank: `transform_1_f363eb6e`
- Real-transfer grid: `transform_1_3a98514b`
- Report: `report_49d8d251c445_aed45e9a`
- Report path:
  `projects/DX_TERMINAL/prompt_confusion/phase_15/reports/mid_prompt_real_transfer/report_49d8d251c445_aed45e9a/report.md`

Command:

```bash
PHASE13_SIGNAL_DISCOVERY_TABLE=dx_terminal_signal_discovery_phase13_v1 \
PHASE13_SIGNAL_DISCOVERY_TIERS=aggressive \
uv run python -m pipelines_v2.cli workflow run \
  --file projects/DX_TERMINAL/prompt_confusion/phase_15/specs/workflow.py \
  --logging INFO
```

Grid size:

- `1080` cells
- 5 Phase 14 bank sites x 6 real positions x 9 layers x 4 directions

## Primary Result

The strongest matched-site transfer by raw cohort separation did not come from
`settings_end`; it came from Phase 14 `market_end` directions projected onto
real `market_end`, mainly at L44. That headline is suspect because L44 is the
last captured layer and repeats the earlier failure mode where a final-layer
projection looked strong before closer reading found more useful structure in
the mid layers.

Top clean matched-site cells, using the Phase 13 ordering criterion
`anchor_positive > complaint > structure_matched_control`:

| Bank site -> real site | Direction | Layer | Anchor | Complaint | Control | Complaint-control |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `market_end -> market_end` | `risk_preference` | 44 | -9.803 | -11.502 | -20.089 | 8.587 |
| `market_end -> market_end` | `shared_mean` | 44 | -8.666 | -10.160 | -18.027 | 7.867 |
| `market_end -> market_end` | `trade_size` | 44 | 0.846 | -2.752 | -6.815 | 4.063 |
| `settings_end -> settings_end` | `risk_preference` | 44 | 1.050 | -6.248 | -8.084 | 1.837 |
| `settings_end -> settings_end` | `trade_size` | 44 | 0.141 | -5.652 | -6.843 | 1.191 |

For the old Phase 13 primary cell, L32 real `settings_end`:

| Bank site -> real site | Direction | Layer | Anchor | Complaint | Control | Complaint-control |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `settings_end -> settings_end` | `trade_size` | 32 | -1.893 | -1.471 | -1.527 | 0.056 |
| `settings_end -> settings_end` | `shared_mean` | 32 | 0.637 | 0.827 | 0.424 | 0.403 |

For comparison, Phase 13's old prompt-EOS direction bank at L32 real
`settings_end` had:

- `trade_size`: complaint-control `0.526`
- `shared_mean`: complaint-control `0.377`

So the Phase 14 `settings_end` bank does not improve the L32 `trade_size`
primary read, but the L32 `shared_mean` read is comparable/slightly higher. The
much larger new effect is the L44 `market_end` matched-site result.

The same-site shared-mean mid-layer cells remain live and should not be hidden
by the L44 sort:

| Bank site -> real site | Layer | Anchor | Complaint | Control | Complaint-control | Clean order? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `settings_end -> settings_end` | 28 | 3.873 | 3.544 | 3.372 | 0.172 | yes |
| `settings_end -> settings_end` | 32 | 0.637 | 0.827 | 0.424 | 0.403 | no, complaint > anchor |
| `settings_end -> settings_end` | 36 | 0.620 | 0.650 | 0.343 | 0.307 | no, complaint > anchor |
| `market_end -> market_end` | 32 | 2.608 | 1.792 | 1.118 | 0.674 | yes |
| `market_end -> market_end` | 36 | 1.619 | 0.674 | -0.306 | 0.980 | yes |
| `portfolio_end -> portfolio_end` | 28 | -1.585 | -2.694 | -4.386 | 1.691 | yes |
| `portfolio_end -> portfolio_end` | 36 | -4.282 | -4.176 | -6.136 | 1.960 | no, complaint > anchor |

Layer caveat:

- Phase 15 evaluated every captured layer, not only the synthetic probe-best
  layer.
- The strongest real-transfer cells mostly moved later than the Phase 14
  synthetic probe optima.
- For `market_end risk_preference`, the synthetic probe was best by balanced
  accuracy at L32 and best by AUROC at L28; the strongest real matched-site
  transfer was L44.
- For `market_end trade_size`, the synthetic probe was best by balanced
  accuracy at L40 and best by AUROC at L36; the strongest real matched-site
  transfer was L44.
- This makes the L44 `market_end` result a transfer/localization finding, not a
  simple reuse of the synthetic probe's best layer.

## Interpretation

The immediate answer is:

- `market_end` Phase 14 directions look best by raw cohort separation, but the
  L44 winner should be treated as a late-layer amplification warning, not the
  main scientific result.
- `settings_end` Phase 14 directions do not cleanly beat the old Phase 13
  prompt-EOS trade-size result at the original L32 `settings_end` cell.
- `settings_end` Phase 14 `shared_mean` at L32 does reproduce the earlier
  same-site shared-mean transfer scale: Phase 13 prompt-EOS shared mean was
  `0.377`; Phase 14 `settings_end -> settings_end` shared mean is `0.403`.
- `portfolio_end` has moderate matched-site transfer, but several high
  complaint-control cells invert anchor/complaint ordering, so they are weaker
  evidence under the Phase 13 criterion.
- Focused top/bottom audit makes the L36 `market_end -> market_end`
  `shared_mean` cell more interesting than the L44 headline: high-scoring
  complaint prompts emphasize restrictive/exclusive HIGH-strategy handling and
  fallthrough prevention, while low-scoring complaint prompts emphasize more
  immediate-execution/default BUY/SELL policy language. That is a concrete
  policy-lifecycle axis, not just generic complaintness.
- The L32 `settings_end -> settings_end` shared-mean audit is less clean from
  prompt previews alone; it appears to mix prompt-policy/lifecycle wording
  variants. It remains live because the transfer magnitude matches Phase 13,
  but it needs a hand-labeled packet before promotion.

## Artifacts

- Workflow: `projects/DX_TERMINAL/prompt_confusion/phase_15/specs/workflow.py`
- Result:
  `projects/DX_TERMINAL/prompt_confusion/phase_15/reports/mid_prompt_real_transfer/report_49d8d251c445_aed45e9a/results/phase14_real_transfer_grid_results.json`
