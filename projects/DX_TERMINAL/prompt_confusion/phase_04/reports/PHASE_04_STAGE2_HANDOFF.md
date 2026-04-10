# Phase 04 Stage 2 Handoff

Date: 2026-04-10

This note summarizes the follow-up work after the initial Phase 04 checkpoint report. It focuses on three questions:

1. Is the causal harness real now, or was it still broken?
2. How strong are the confounds on the conflict-side arbitration label?
3. Does a simple section-aggregated attention pass add much mechanistic traction yet?

## Short Version

- Yes, the causal harness is fixed. The old `<think>` failure mode is gone.
- The conflict-side arbitration label is strongly confounded by dataset structure, especially `strategy_family` and `strategy_family + environment_pressure_bucket`.
- The causal patch is behaviorally active, but interpretation is still limited because the same arbitration label is heavily explained by family and pressure metadata.
- The section-aggregated attention pass runs and saves usable outputs, but it is weak as a decoder so far.

## What Changed

Code changes:

- [modal_vllm_engine.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/interp/modal_vllm_engine.py)
- [modal_conflict_arbitration_analysis.py](/Users/marshallvyletel/repos/concordance/xenon/projects/DX_TERMINAL/prompt_confusion/phase_04/scripts/modal_conflict_arbitration_analysis.py)

Important fixes:

- The shared vLLM engine now respects `capture_reasoning=false` instead of auto-enabling Qwen reasoning parsing whenever the model id contains `qwen3`.
- The causal generation path now explicitly uses `enable_thinking=false`, matching the validated behavioral generation surface used in the main Phase 04 behavior run.
- New stage-2 analyses were added without touching stage-1 outputs:
  - causal rerun
  - confound baselines
  - section-aggregated attention summary

## Stage 1 Reference Point

These are the main reference numbers from the earlier arbitration work:

- Best conflict-side residual probe:
  - layer `20`
  - grouped balanced accuracy `0.7124`
  - source: [results.json](/Users/marshallvyletel/repos/concordance/xenon/projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/analysis_conflict_readout_residual/6924c2b36f43/results.json)
- Best conflict-side router probe:
  - layer `16`
  - grouped balanced accuracy `0.6476`
  - source: [results.json](/Users/marshallvyletel/repos/concordance/xenon/projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/analysis_conflict_readout_router/4aeb882aea9d/results.json)
- Best section-local policy readout:
  - `SETTINGS mean @ layer 24`
  - grouped balanced accuracy `0.7647`
  - source: [summary.json](/Users/marshallvyletel/repos/concordance/xenon/projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage1/section_attribution/summary.json)

That section-local readout was the basis for the stage-2 causal rerun.

## Stage 2 Outputs

Local stage-2 artifact root:

- [conflict_arbitration_stage2](/Users/marshallvyletel/repos/concordance/xenon/projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage2)

Main files:

- causal summary: [summary.json](/Users/marshallvyletel/repos/concordance/xenon/projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage2/causal_check/summary.json)
- causal row-level outputs: [row_level.parquet](/Users/marshallvyletel/repos/concordance/xenon/projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage2/causal_check/row_level.parquet)
- confound summary: [summary.json](/Users/marshallvyletel/repos/concordance/xenon/projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage2/confound_checks/summary.json)
- confound baseline table: [baseline_results.parquet](/Users/marshallvyletel/repos/concordance/xenon/projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage2/confound_checks/baseline_results.parquet)
- attention summary: [summary.json](/Users/marshallvyletel/repos/concordance/xenon/projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage2/attention_summary/summary.json)
- attention row-level table: [row_level.parquet](/Users/marshallvyletel/repos/concordance/xenon/projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage2/attention_summary/row_level.parquet)
- attention plot: [prompt_eos_attention_delta.png](/Users/marshallvyletel/repos/concordance/xenon/projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage2/attention_summary/prompt_eos_attention_delta.png)

## Causal Rerun

The crucial technical result is that the causal rerun is now valid.

Row-level JSON validity:

- `baseline`: `1.0`
- `setting_push`: `1.0`
- `strategy_push`: `1.0`

So the old stage-1 failure mode is gone. The previous all-`neither` result was a generation-surface bug, not a mechanistic null.

### Aggregate Causal Result

Target:

- section: `settings`
- pooling: `mean`
- layer: `24`
- strength: `1.0`

Overall rates:

| Condition | Strategy Rate | Setting Rate | Neither Rate |
|---|---:|---:|---:|
| `baseline` | `0.5366` | `0.4553` | `0.0081` |
| `setting_push` | `0.4878` | `0.4878` | `0.0244` |
| `strategy_push` | `0.6260` | `0.3659` | `0.0081` |

Interpretation:

- The intervention is behaviorally active.
- `strategy_push` clearly moves the model toward strategy-following behavior.
- `setting_push` moves behavior toward setting-following, but more weakly and with a small increase in `neither`.
- So this is not a null result anymore. It is a real directional causal effect, but asymmetric.

### By Family

Counts by family and condition:

#### `baseline`

- `activity_force_observe`: `30 strategy / 6 setting`
- `activity_force_trade`: `14 strategy / 22 setting`
- `trade_size_force_large`: `0 strategy / 21 setting`
- `trade_size_force_small`: `22 strategy / 7 setting / 1 neither`

#### `setting_push`

- `activity_force_observe`: `31 strategy / 5 setting`
- `activity_force_trade`: `13 strategy / 23 setting`
- `trade_size_force_large`: `0 strategy / 21 setting`
- `trade_size_force_small`: `16 strategy / 11 setting / 3 neither`

#### `strategy_push`

- `activity_force_observe`: `31 strategy / 5 setting`
- `activity_force_trade`: `16 strategy / 20 setting`
- `trade_size_force_large`: `6 strategy / 15 setting`
- `trade_size_force_small`: `24 strategy / 5 setting / 1 neither`

Practical read:

- The clearest shift is in the size families, especially `trade_size_force_small` and `trade_size_force_large`.
- The activity families move less.
- That asymmetry is useful and probably worth highlighting in the next report rather than averaging away.

## Confound Baselines

This is the most important new interpretation result.

Grouped baseline results:

| Baseline | Grouped Balanced Accuracy |
|---|---:|
| `family_plus_pressure` | `0.8035` |
| `family_only` | `0.7633` |
| `user_text_ngram` | `0.7010` |
| `metadata_all` | `0.6740` |
| `lexical_ids` | `0.6507` |
| `pressure_only` | `0.5728` |
| `length_position_numeric` | `0.4873` |

Comparison to stage-1 probes:

- best conflict-side residual probe: `0.7124`
- best conflict-side router probe: `0.6476`
- best policy-source section probe: `0.7647`

Interpretation:

- The arbitration label is strongly confounded by `strategy_family`.
- `family_only = 0.7633` is already better than the best residual arbitration probe (`0.7124`).
- `family_plus_pressure = 0.8035` is better than both the residual arbitration probe and the selected section-local policy probe.
- So the current conflict-side readout is not clean evidence of a source-selection mechanism by itself.

Important negative result:

- `length_position_numeric = 0.4873`

That means the confound story is **not** “section length alone explains everything.” The stronger confounds are family semantics and pressure structure, not just raw token-count differences.

### Plain-Language Explanation

What this means in simpler terms:

- We can predict `which side wins` surprisingly well just from high-level dataset metadata like family and pressure bucket.
- Because of that, a probe that predicts `which side wins` is not automatically discovering a clean latent arbitration circuit.
- It may be picking up family-dependent prompt regularities instead.

So the current state is:

- conflict detection looks real
- arbitration signal also looks real
- but the arbitration signal is heavily entangled with structured dataset variables

That does **not** kill the project. It just lowers the strength of the mechanism claim until family-controlled or residualized analyses are added.

## Attention Summary

The section-aggregated attention pass now runs end to end and saves usable outputs, but it is weak as a decoder.

Grouped balanced accuracy by anchor and layer:

| Layer | Anchor | Balanced Accuracy |
|---|---|---:|
| `20` | `settings_eos` | `0.5870` |
| `24` | `prompt_eos` | `0.5538` |
| `36` | `strategy_eos` | `0.5476` |
| `36` | `prompt_eos` | `0.5436` |
| `24` | `settings_eos` | `0.5351` |
| `20` | `prompt_eos` | `0.5318` |
| `24` | `strategy_eos` | `0.5134` |
| `20` | `strategy_eos` | `0.5033` |
| `36` | `settings_eos` | `0.5032` |

Interpretation:

- Section-aggregated attention mass is not yet a strong decoder of `strategy` versus `setting`.
- The best result is only `0.5870`.
- So attention is not yet carrying the same evidential weight as residual readout or causal intervention.
- This pass is still useful as descriptive support, especially through the saved plot, but not as a main result.

Relevant artifact:

- [prompt_eos_attention_delta.png](/Users/marshallvyletel/repos/concordance/xenon/projects/DX_TERMINAL/prompt_confusion/phase_04/outputs/conflict_arbitration_stage2/attention_summary/prompt_eos_attention_delta.png)

## Recommended Report Framing

Suggested framing for the updated report:

1. The causal harness is fixed, and the intervention now moves behavior in the expected direction.
2. That is encouraging, but the arbitration target is not clean: family and pressure metadata are stronger predictors than the residual arbitration probe.
3. Therefore:
   - conflict detection remains the cleanest result
   - the arbitration result should be framed as promising but confounded
   - the causal result should be framed as real but not yet mechanism-clean

## Recommended Next Step

The next high-value analysis is **family-controlled arbitration**, not more generic probing.

Concretely:

- run within-family arbitration probes
- or regress out family and pressure metadata before probing
- or evaluate causal deltas separately within each family and compare where the patch is actually active

That is the step that would tell us whether the stage-2 causal effect is acting on a real cross-family arbitration feature, or just nudging family-specific structures that already dominate the label.
