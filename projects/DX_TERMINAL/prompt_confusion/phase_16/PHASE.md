# Phase 16: Phase 13 Split Audit

## Premise

Phase 15 showed plausible mid-layer same-site transfer, but the Phase 13 real
data buckets are messy. The original `complaint`, `anchor_positive`, and
`structure_matched_control` strata are useful for coarse discovery, but they are
not clean enough to treat as ground truth controls.

Question:

> Can we semi-programmatically derive cleaner Phase 13 buckets before rerunning
> transfer summaries?

## Design

Read the existing Phase 13 Neon table:

- Default table: `dx_terminal_signal_discovery_phase13_v1`
- Source workflow: Phase 13 real signal discovery
- No new model capture

Derive:

- `phase16_case_id`
  - complaints group by `trace_id`
  - synthetic controls and anchors group by `source_example_id`
- `phase16_bucket`
  - `strict_system_conflict`
  - `user_config_conflict_control`
  - `synthetic_template_control`
  - `anchor_aligned_real`
  - `anchor_conflict_like`
  - `ambiguous_mixed`
  - `review_or_exclude`
- `phase16_dimension`
  - `trade_size`
  - `holding`
  - `strategy_lifecycle_activity`
  - `general_performance`
  - `unknown`
- `phase16_action_polarity`
  - derived from `complaint_type`

This phase is a split audit, not a new representational claim.

## Claim Boundary

This phase can say whether the existing Phase 13 data can support cleaner
control buckets and what should be routed to manual review. It cannot prove that
the derived labels are correct without hand inspection.

## What We Ran

Completed run:

- Run: `wr_1f8e8cbad157_5bd854a6`
- Split audit artifact: `transform_1_7fc6f5f2`
- Report: `report_da428a86e40c_cbd16c6e`
- Report path:
  `projects/DX_TERMINAL/prompt_confusion/phase_16/reports/split_audit/report_da428a86e40c_cbd16c6e/report.md`

Command:

```bash
uv run python -m pipelines_v2.cli workflow run \
  --file projects/DX_TERMINAL/prompt_confusion/phase_16/specs/workflow.py \
  --logging INFO
```

Cleaned transfer rerun:

- Run: `wr_ae051002ca6f_e8ecd370`
- Split audit artifact: `transform_1_2bfc3206`
- Phase 14 direction bank: `transform_1_4f8a2509`
- Case-averaged bucket transfer: `transform_1_968fa3ad`
- Report: `report_00814e1d96ce_b6478dbb`
- Report path:
  `projects/DX_TERMINAL/prompt_confusion/phase_16/reports/split_audit/report_00814e1d96ce_b6478dbb/report.md`

The cleaned transfer step reuses the Phase 13 real capture and Phase 14
direction bank. It averages projection scores within `phase16_case_id` before
taking bucket means, so repeated ticks and prompt tiers do not dominate a
complaint trace.

## Results

The first conservative split pass found:

| Bucket | Unique cases |
| --- | ---: |
| `synthetic_template_control` | 300 |
| `strict_system_conflict` | 109 |
| `user_config_conflict_control` | 99 |
| `anchor_conflict_like` | 70 |
| `anchor_aligned_real` | 48 |
| `ambiguous_mixed` | 10 |

The core complaint-vs-real-control comparison is viable:

- `strict_system_conflict`: 109 unique complaint traces
- `user_config_conflict_control`: 99 unique complaint traces
- `ambiguous_mixed`: 10 unique traces, mostly `STRATEGY_SLIDER_LOCKOUT`

The current `anchor_positive` bucket should not be treated as clean by default:
it splits into 48 aligned-real cases and 70 conflict-like cases under the
Phase 16 rules.

Dimension coverage among unique cases:

| Bucket | Dimension | Unique cases |
| --- | --- | ---: |
| `strict_system_conflict` | `strategy_lifecycle_activity` | 91 |
| `user_config_conflict_control` | `strategy_lifecycle_activity` | 85 |
| `strict_system_conflict` | `trade_size` | 12 |
| `user_config_conflict_control` | `trade_size` | 12 |
| `strict_system_conflict` | `holding` | 6 |
| `user_config_conflict_control` | `holding` | 2 |

Immediate takeaway:

- the cleanest quick rerun should compare `strict_system_conflict` against
  `user_config_conflict_control`, grouped by `phase16_case_id`
- `synthetic_template_control` should remain a separate synthetic baseline, not
  the primary real-data control
- `STRATEGY_SLIDER_LOCKOUT` should go to review/mixed, not either side of the
  main split
- trade-size has only 12 cases per side, so the most powered real-data split is
  strategy-lifecycle/activity rather than the old three-family synthetic split

## Cleaned Transfer Test

Using `strict_system_conflict - user_config_conflict_control` as the primary
real-control contrast changes the picture materially.

Best matched-site cells on aggressive prompts:

| Contrast | Bank site -> real site | Direction | Layer | Strict | User-control | Delta |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| all dimensions | `portfolio_end -> portfolio_end` | `diversification_preference` | 44 | -4.866 | -5.353 | 0.487 |
| shared mean, all dimensions | `portfolio_end -> portfolio_end` | `shared_mean` | 40 | -5.346 | -5.611 | 0.265 |
| shared mean, strategy lifecycle | `portfolio_end -> portfolio_end` | `shared_mean` | 40 | -5.340 | -5.650 | 0.310 |
| shared mean, trade size | `market_end -> market_end` | `shared_mean` | 44 | -9.196 | -10.500 | 1.304 |

Mid-layer same-site shared-mean cells under the cleaned real-control split:

| Bank site -> real site | Layer | Dimension | Strict | User-control | Delta |
| --- | ---: | --- | ---: | ---: | ---: |
| `settings_end -> settings_end` | 32 | all | 0.861 | 0.782 | 0.079 |
| `settings_end -> settings_end` | 32 | strategy lifecycle | 0.872 | 0.763 | 0.109 |
| `settings_end -> settings_end` | 36 | all | 0.677 | 0.595 | 0.082 |
| `settings_end -> settings_end` | 36 | strategy lifecycle | 0.679 | 0.581 | 0.097 |
| `market_end -> market_end` | 32 | all | 1.804 | 1.739 | 0.065 |
| `market_end -> market_end` | 36 | all | 0.679 | 0.727 | -0.048 |
| `portfolio_end -> portfolio_end` | 36 | all | -4.132 | -4.210 | 0.078 |
| `portfolio_end -> portfolio_end` | 40 | strategy lifecycle | -5.340 | -5.650 | 0.310 |

Interpretation:

- The old large `market_end` L44 shared-mean result is mostly a
  synthetic-template-control effect. On all strict vs user-config cases it is
  negative: `-0.546`; on strategy-lifecycle cases it is `-0.679`.
- The trade-size-only `market_end` L44 shared-mean delta remains large
  (`1.304`), but it has only 12 cases per side and should not be promoted yet.
- The cleaner, better-powered signal is smaller: `portfolio_end` L40
  `shared_mean` over strategy-lifecycle/activity cases has delta `0.310`
  across 91 strict cases vs 85 user-config controls.
- `settings_end` L32/L36 shared mean survives in the expected direction but is
  modest (`0.08` to `0.11`) under real controls.
- This supports using Phase 16 buckets for further transfer reads and weakens
  the L44 market headline from Phase 15.

## Artifacts

- Workflow: `projects/DX_TERMINAL/prompt_confusion/phase_16/specs/workflow.py`
- Result:
  `projects/DX_TERMINAL/prompt_confusion/phase_16/reports/split_audit/report_da428a86e40c_cbd16c6e/results/phase13_split_audit_results.json`
- Cleaned transfer result:
  `projects/DX_TERMINAL/prompt_confusion/phase_16/reports/split_audit/report_00814e1d96ce_b6478dbb/results/case_averaged_bucket_transfer_results.json`
