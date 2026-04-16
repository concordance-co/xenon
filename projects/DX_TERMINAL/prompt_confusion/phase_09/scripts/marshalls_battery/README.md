# Marshall's Battery for Phase 09

Extra confound + sanity tests on top of Trent's
`phase_09/specs/workflow.py`.

## What this runs

`workflow.py` is a standalone `pipelines_v2` workflow file. Steps:

| Step | Purpose |
|---|---|
| `text_baseline_conflict_strict_combined` | Text gate under the strict both-axes holdout (train iff BOTH strategy and settings are train variants; test iff BOTH are test). Trent's main text gate uses per-axis splits. |
| `capture_residual_with_generation` | Fresh capture with the same residual sites Trent used, **plus `GenerationSpec(enabled=True, max_tokens=256)`** so the behavioral audit has outputs to parse. Trent's main capture ran with generation disabled. |
| `probe_conflict_strict_combined_holdout` | Probe on `conflict_present` under strict both-axes holdout. Answers "does the 0.995 survive when neither axis is seen during fit?" Trent's `conflict_probe` uses the XOR-derived `lexical_split` column, which still exposes all individual variants at train time. |
| `probe_setting_value_grouped_cv` | Probe on `setting_value` (1 vs 5) directly, grouped by `matched_group_id`. If this decodes near-perfectly, some of the conflict signal may route through a direct setting-value encoding rather than a comparison. `setting_value` is balanced across conflict at the dataset level, so this probe alone cannot decode `conflict_present` -- but it tells us what the residual stream knows about the setting dial independent of the comparison. |
| `probe_direction_transfer_trade_size` | `TransferProbeSpec` on `trade_size` with `cohort_by=strategy_direction`. Trains on rows where STRATEGY says `small`, tests on rows where STRATEGY says `large`, and vice versa. Transfer >> chance distinguishes real relational conflict from "small-word plus large-word both appear" lexical shortcuts. |
| `probe_conflict_grouped_cv_selectivity` | Belt-and-suspenders conflict probe with explicit `shuffled_label` baseline and `selectivity` metric. Trent already has selectivity in his main probe; this reruns it alongside the strict holdout so the selectivity delta is apples-to-apples. |
| `report` | Local materialization of all of the above. |

`behavioral_audit.py` is a standalone script run after the workflow
finishes. It consumes the capture's `generations.json` and emits a
markdown audit + figures:

- Overall outcome distribution (`aligned_match` / `follow_strategy` /
  `follow_setting` / `refuse` / `other` / `malformed`).
- Outcome by (`target_dimension`, `conflict_band`).
- Aligned correctness by (`dim`, `strategy_direction`, `setting_value`).
  Cells below ~90% aligned_match are red flags.
- `strategy_variant x settings_variant` heatmaps of `follow_setting`
  rate on conflict rows, per dimension. Identifies variant-wording
  authority asymmetries (the Phase 06 v4 pattern).

## How to run

### 1. Plan and run the workflow

```bash
uv run python -m pipelines_v2.cli workflow plan \
  --file projects/DX_TERMINAL/prompt_confusion/phase_09/scripts/marshalls_battery/workflow.py

uv run python -m pipelines_v2.cli workflow run \
  --file projects/DX_TERMINAL/prompt_confusion/phase_09/scripts/marshalls_battery/workflow.py
```

The capture runs on Modal A100-80GB. Analyses run on Modal CPU. The
report materializes locally under
`projects/DX_TERMINAL/prompt_confusion/phase_09/reports/marshalls_battery/`.

Expected wall clock: capture ~10 min, analyses 1-2 min each, report
local.

### 2. Pull the generations file from Modal

Find the capture artifact id from the workflow output, then:

```bash
uv run modal volume get xenon-data \
  /data/artifacts/prompt_confusion_phase_09_marshalls_battery/capture_<artifact_id>/generations.json \
  /tmp/phase_09_battery_generations.json --force
```

### 3. Run the behavioral audit

```bash
uv run python projects/DX_TERMINAL/prompt_confusion/phase_09/scripts/marshalls_battery/behavioral_audit.py \
  --generations /tmp/phase_09_battery_generations.json \
  --dataset projects/DX_TERMINAL/prompt_confusion/phase_09/outputs/phase_09_dataset/phase_09_dataset.jsonl \
  --output-dir projects/DX_TERMINAL/prompt_confusion/phase_09/reports/marshalls_battery
```

Writes `behavioral_audit.md` and `figures/` into the battery's report
directory.

## What answers we are looking for

- **Strict combined holdout.** If the probe drops noticeably from Trent's
  single-axis ~0.995 -- say, below 0.9 -- single-axis numbers were
  inflated by partial wording overlap at train time. If it holds near
  0.99, the detection claim is stronger than anyone has yet shown.
- **Setting-value probe.** Expect high bal_acc (it's a scalar encoded
  in the setting block). The question is just whether the *magnitude*
  is comparable to or lower than the conflict probe. If setting_value
  decodes at 0.99 too, that's fine -- the conflict probe has to do
  something relational on top.
- **Strategy-direction cross-transfer.** This is the sharpest test for
  "relational comparison vs lexical co-occurrence." If transfer is
  strong (bal_acc >> chance both directions), the probe is doing
  relational work. If it collapses to chance on the opposite cohort,
  the probe memorized "small + large both in prompt."
- **Selectivity.** Probe bal_acc minus shuffled-label bal_acc. Already
  in Trent's main probe; re-run for the strict holdout version for
  apples-to-apples comparison.
- **Behavioral audit.** Aligned correctness per cell should be near
  ceiling (the Phase 06 v4 failure was aligned cells producing
  refusals at 77% in specific combinations). Variant heatmaps should
  not show one cell at 100% follow_setting while others are at 0% --
  that was the numeric-scale authority confound on v4.
