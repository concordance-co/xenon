# Marshall's Battery Results on Phase 09

Ran 2026-04-15 against Trent's Phase 09 dataset (`phase_09_dataset.jsonl`,
864 rows), fresh capture on Modal A100-80GB with generations enabled.
Workflow and audit script live under
`projects/DX_TERMINAL/prompt_confusion/phase_09/scripts/marshalls_battery/`.

## Bottom line for Trent's findings

**Only one result updates the Phase 09 headline:**

- **Strict combined both-axes lexical holdout** (train iff BOTH strategy
  and settings are v0/v1; test iff BOTH are v2/v3): probe on
  `conflict_present` hits **0.764 bal_acc / 0.844 AUROC at L36**, vs
  your XOR-split `lexical_split` number of 0.875 pooled / 0.995 on
  trade_size. The strict holdout is ~20pt harder because your XOR
  split lets the probe see (test,test) cells at train time, which
  exposes v2/v3 wordings on both axes before evaluation. 0.764 is
  still well above the text baseline (0.574), so the detection signal
  is real, just softer under the toughest generalization test.

**Everything else confirms the Phase 09 story:**

- Behavioral audit (864 generations): trade_size aligned 100% correct,
  97.4% follow_setting on trade_size conflict uniformly across all
  16 variant pairs. No variant-authority asymmetries (unlike our
  Phase 06 v4). Trading_activity aligned 92.9% overall with the known
  muddy cell (`strategy=trade + setting_value=1` at 67%) consistent
  with your Wave 1 boundary finding. Essentially 0 refusals across
  the dataset.
- Text baseline under strict combined holdout: 0.574 — slight
  leakage, consistent with your single-axis 0.56 settings split.
- Setting_value probe: 0.901 bal_acc. The model strongly encodes
  what the setting says, as expected. Setting_value is balanced 50/50
  across `conflict_present` at the dataset level so this is not a
  confound for the conflict probe.

## What NOT to take from this battery

I also ran a `strategy_direction` cross-cohort transfer probe on
trade_size (train on rows where strategy says small, test on rows
where strategy says large). That probe collapses to chance with
near-zero direction cosine.

**This does not invalidate your probe.** Your probe is trained on rows
with both strategy_directions mixed together. If the residual stream
only encoded `setting_value` direction-specifically, a probe trained on
mixed directions would cap at 0.50 (because the same setting value is
aligned in one direction and conflict in the other). Your probe hitting
0.995 on mixed-direction data **requires** a direction-invariant linear
feature in the residual stream.

My cross-cohort probe fails because it is trained on a restricted
cohort (one direction only), where it has no incentive to find the
relational feature -- it takes the easier setting-value shortcut,
which then gets flipped on the held-out cohort. That is a probe
methodology artifact, not a reframe of your finding.

## Files

### Top-level

- `behavioral_audit.md` -- generations-level audit: outcome
  distribution, per-dimension/per-pair breakdown, aligned-correctness
  per cell, variant x variant resolution heatmaps.
- `figures/` -- charts referenced by the audit.
  - `fig_outcome_by_dim_band.png` -- outcome stacked bars by
    (target_dimension, conflict_band).
  - `fig_variant_heatmap_trade_size.png` -- strategy_variant x
    settings_variant follow_setting rate on trade_size conflict rows.
  - `fig_variant_heatmap_trading_activity.png` -- same for activity.
- `results/` -- raw `result.json` files pulled from the Modal
  capture artifacts, plus the capture's generations.
  - `probe_conflict_strict_combined_holdout.json` -- **the headline
    result that updates Phase 09.** Per-layer bal_acc, AUROC,
    selectivity, shuffled-label + majority baselines.
  - `probe_conflict_grouped_cv_selectivity.json` -- same probe under
    5-fold grouped CV (apples-to-apples selectivity number).
  - `probe_setting_value_grouped_cv.json` -- setting_value reference
    probe.
  - `transfer_probe_direction_trade_size.json` -- cross-cohort
    direction transfer probe (see caveats above).
  - `text_baseline_conflict_strict_combined.json` -- text gate under
    strict holdout.
  - `generations.json` -- raw capture generations, 864 rows, all JSON
    parseable. Used by `behavioral_audit.py` to produce the audit.

## How to regenerate

The workflow and audit script:

- `phase_09/scripts/marshalls_battery/workflow.py` -- `pipelines_v2`
  workflow (capture with generation + 4 probes + 1 text baseline +
  report).
- `phase_09/scripts/marshalls_battery/behavioral_audit.py` --
  standalone audit that takes a `generations.json` and a dataset
  `jsonl` and emits `behavioral_audit.md` + `figures/`.

Commands:

```bash
# Plan and run (capture runs on Modal A100-80GB with residual-only
# sites and GenerationSpec enabled; analyses on Modal CPU).
uv run python -m pipelines_v2.cli workflow plan \
  --file projects/DX_TERMINAL/prompt_confusion/phase_09/scripts/marshalls_battery/workflow.py

uv run python -m pipelines_v2.cli workflow run \
  --file projects/DX_TERMINAL/prompt_confusion/phase_09/scripts/marshalls_battery/workflow.py

# Pull the generations from Modal volume (substitute the real capture
# artifact id from the workflow output):
uv run modal volume get xenon-data \
  /data/artifacts/prompt_confusion_phase_09_marshalls_battery/capture_.../generations.json \
  /tmp/phase_09_battery_generations.json --force

# Behavioral audit:
uv run python projects/DX_TERMINAL/prompt_confusion/phase_09/scripts/marshalls_battery/behavioral_audit.py \
  --generations /tmp/phase_09_battery_generations.json \
  --dataset projects/DX_TERMINAL/prompt_confusion/phase_09/outputs/phase_09_dataset/phase_09_dataset.jsonl \
  --output-dir projects/DX_TERMINAL/prompt_confusion/phase_09/reports/marshalls_battery
```

## Honest post-mortem

Of the five battery tests, two were genuinely useful (strict combined
holdout, behavioral audit). Three were redundant or based on a flawed
analysis on my part:

- `probe_setting_value_grouped_cv` is interesting in isolation but
  doesn't tell us anything about the conflict probe because
  `setting_value` is balanced across `conflict_present` at the dataset
  level.
- `probe_conflict_grouped_cv_selectivity` duplicates what your main
  probe already reports.
- `transfer_probe_direction_trade_size` tests a methodology artifact
  (what a restricted-cohort probe learns) rather than the underlying
  representational claim. It does not falsify your result.

Keeping them here for completeness and in case the raw data is useful
for something else later.
