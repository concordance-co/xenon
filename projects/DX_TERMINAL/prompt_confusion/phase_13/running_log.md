# Phase 13 Running Log

## 2026-04-24

- Created phase 13 scaffold for real signal discovery.
- Kept the corpus source canonical in Neon via
  `dx_terminal_signal_discovery_phase13_v1`.
- Confirmed available Neon sources include Phase 12 anchor/bridge tables and the
  real complaint transfer tick table.
- Did not find a non-complaint production tick table in the current Neon schema.
  Baseline controls therefore need an explicit source table before the corpus is
  complete.
- Obvious-aligned controls are intentionally optional because the spec says to
  drop them if they are not sourceable.
- Dry-run corpus build succeeded with 651 base prompts / 1,953 tiered rows:
  118 Stage 1b strict anchors, 33 strict buy-only anchors, and 500 complaint
  tick rows. Baseline and obvious-aligned rows were skipped because no source
  table was supplied.
- Workflow planning succeeded for capture, coarse projection grid, and report.
- Replaced the local `probe_directions.json` handoff with a Modal/catalog
  `build_synthetic_direction_bank` transform step. Raw direction vectors should
  live in the Modal-backed transform artifact only, not in local files.
- Added `structure_matched_control` as a fallback control stratum sourced from
  aligned Stage 1a real-template controls. This is intentionally distinct from
  `baseline_control`, which remains reserved for real non-complaint production
  ticks.
- Materialized `dx_terminal_signal_discovery_phase13_v1` in Neon with 951 base
  prompts / 2,853 tiered rows:
  118 Stage 1b strict anchors, 33 strict buy-only anchors, 500 complaint ticks,
  and 300 structure-matched controls. True `baseline_control` and
  `obvious_aligned` remain absent.
- Materialized `dx_terminal_signal_discovery_phase13_smoke_v1` in Neon with 90
  base prompts / 270 tiered rows:
  20 Stage 1b strict anchors, 10 strict buy-only anchors, 30 complaint ticks,
  and 30 structure-matched controls.
- Ran the smoke workflow on the aggressive prompt tier with end-of-section
  sites only. Completed run: `wr_83f9740dda21_d7ba7df1`.
- Smoke artifacts:
  capture `capture_1_1da847b70a39`, direction bank `transform_1_20e79296`,
  coarse projection grid `transform_1_0a089d56`, report
  `report_922d1299ea2c_c7599a0a`.
- Smoke result covered 216 cells:
  9 layers x 6 end positions x 1 prompt tier x 4 directions. There were 73
  cells where anchor and complaint means both exceeded the structure-matched
  control mean, and 49 cells with clean ordering
  `anchor_positive > complaint > structure_matched_control`.
- Strongest clean smoke cells were concentrated at L44 `strategies_end`,
  especially `risk_preference`, `trade_size`, and `shared_mean`. This is
  positive smoke evidence, but not a full verdict because it used only the
  aggressive tier, end sites, a small sample, and structure-matched controls
  rather than true non-complaint production baselines.
- Built a focused top/bottom complaint review artifact for L44
  `strategies_end`:
  `reports/signal_discovery/report_922d1299ea2c_c7599a0a/results/l44_strategies_topk_complaint_review.json`.
  The review found that the top complaint rows are mostly `WRONG_SIZE` rows, but
  many are not explicit strategy-vs-settings conflicts. Several top rows have no
  active strategies and appear to involve fee/cooldown/repeat-trade or sizing
  rule-fabrication behavior. There are also many exact projection ties,
  suggesting this smoke cell may be partly reading strategy-protocol/template
  variants rather than only unique active strategy content.
- Built matching top/bottom complaint reviews for L32 and L36 `settings_end`:
  `l32_settings_topk_complaint_review.json` and
  `l36_settings_topk_complaint_review.json`. L32 `settings_end` is more
  encouraging than L44 `strategies_end`: both `trade_size` and `shared_mean`
  have clean cohort ordering, and top complaint rows are enriched for
  `USER_CONFIG_CONFLICT` relative to the bottom rows. However, the top examples
  still mix direct strategy/settings conflicts with broader action-governor,
  sizing, cooldown, and rule-fabrication failures.
- Preregistered a medium validation run in
  `medium_settings_validation_prereg.md`, then ran
  `dx_terminal_signal_discovery_phase13_v1` with aggressive tier and ends-only
  sites. Completed run: `wr_14f78308dbac_dbc78513`.
- Medium artifacts:
  direction bank `transform_1_1a87e6d1`, capture `capture_1_bbfed191794c`,
  coarse projection grid `transform_1_a3a50795`, report
  `report_dd8c8ac3385c_7e82ff1b`.
- Medium result replicated clean cohort separation at the preregistered primary
  cell:
  L32 `settings_end` `trade_size` means were anchor `4.425`, complaint `3.803`,
  structure control `3.278`;
  L32 `settings_end` `shared_mean` means were anchor `3.462`, complaint
  `3.137`, structure control `2.760`.
- Built top/bottom-25 complaint review for L32 `settings_end`:
  `report_dd8c8ac3385c_7e82ff1b/results/l32_settings_top25_complaint_review.json`.
  The preregistered category-confirmation criterion did not cleanly pass.
  `trade_size` top-25 was `17/25` `config_conflict_like`, but the
  control-like bottom-25 was `20/25` `config_conflict_like`. `shared_mean`
  top-25 was `17/25` `config_conflict_like`, while bottom-25 was `14/25`.
  Interpret this as ambiguous: scalar cohort separation replicated, but the
  within-complaint category split is not simply config conflict versus
  rule-fabrication.

## Open Questions

1. Shared direction should start as the mean of trade size, risk preference, and
   diversification preference directions. PCA is a follow-up if the mean is
   incoherent.
2. Baseline control source table is still needed.
3. Obvious-aligned source is still needed, or we drop that stratum.
4. Section parsing is implemented heuristically for DX Terminal headings and
   should be inspected on a sample before launching the full capture.
