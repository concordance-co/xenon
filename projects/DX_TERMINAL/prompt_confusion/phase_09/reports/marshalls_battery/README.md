# Marshall's Battery Results on Phase 09

Fresh capture on Modal A100-80GB + per-dimension probes under a strict
both-axes lexical holdout. Run against
`main_benchmark_row=True` rows only (i.e., excluding the 96 behaviorally
muddy boundary rows on `trading_activity`).

## Headline table

| | Pooled main-only | trade_size | trading_activity |
|---|---|---|---|
| Trent's XOR-split `lexical_split` | 0.875 | **0.995** | 0.92 |
| Strict combined both-axes holdout (peak) | 0.812 / 0.852 (L40) | **0.990 / 1.000** (L40) | **0.635 / 0.662** (L36) |
| Text baseline (same holdout) | 0.536 / 0.554 | 0.500 / 0.500 | 0.542 / 0.586 |

All numbers are `balanced_accuracy / AUROC`. Strict holdout means the
probe trains on rows where both `strategy_lexical_split=='train'` AND
`settings_lexical_split=='train'`, and evaluates on rows where both
are `'test'`. Trent's `lexical_split` column is an XOR of the two
per-axis splits and exposes v2/v3 wordings on each axis during
training.

## What this updates in Phase 09

- **trade_size detection survives strict holdout essentially intact.**
  0.990 / 1.000 at L40 vs Trent's 0.995 -- within a single error on
  192 held-out rows. The finding is robust.
- **trading_activity detection is meaningfully softer than the XOR-split
  numbers suggest.** Trent's 0.92 drops to 0.635 / 0.662 peak under
  strict holdout. The gap above the text baseline (0.542) is only
  ~10 points. Consistent with Trent's Wave 1 pair-delta finding that
  activity is heterogeneous / less direction-consistent than size.
  The strict holdout magnifies that heterogeneity.
- The pooled number (0.812) averages a clean result (trade_size) with
  a soft one (trading_activity). It is arithmetically correct but
  hides the per-dimension asymmetry, which is the actual interesting
  thing here.

## Files

Raw `result.json` payloads pulled from the capture and probe artifacts
on the `xenon-data` Modal volume:

- `results/probe_conflict_strict_combined_holdout.json` -- pooled
  main-only probe, n=768.
- `results/probe_conflict_strict_combined_trade_size.json` --
  trade_size-only probe, n=384.
- `results/probe_conflict_strict_combined_trading_activity.json` --
  trading_activity-only probe (excluding boundary rows), n=384.
- `results/text_baseline_conflict_strict_combined.json` -- pooled
  text baseline under the same strict holdout, n=768.
- `results/text_baseline_conflict_strict_combined_trade_size.json` --
  per-dim text baseline, n=384.
- `results/text_baseline_conflict_strict_combined_trading_activity.json` --
  per-dim text baseline, n=384.

## How to reproduce

Workflow and audit script live at
`projects/DX_TERMINAL/prompt_confusion/phase_09/scripts/marshalls_battery/`.

```bash
uv run python -m pipelines_v2.cli workflow plan \
  --file projects/DX_TERMINAL/prompt_confusion/phase_09/scripts/marshalls_battery/workflow.py

uv run python -m pipelines_v2.cli workflow run \
  --file projects/DX_TERMINAL/prompt_confusion/phase_09/scripts/marshalls_battery/workflow.py
```

Capture ran with `enforce_eager=False`, `max_num_seqs=16`,
`enable_prefix_caching=True`, `enable_thinking=False`, residual-only
sites at layers {0,4,8,12,16,20,24,28,32,36,40,44}.
