# Trade Size Activation Patch Layer Sweep

Run:
- workflow: `prompt_confusion_trade_size_activation_patch_layer_sweep`
- run id: `wr_b05b536729e5_8587a67e`
- summary json: `trade_size_activation_patch_layer_sweep_summary.json`

Setup:
- family: `trade_size`
- scope: conflict rows only (`n = 192`)
- token: last prompt token
- operator: `SwapMeanPatch`
- write layers: `L28`, `L32`, `L36`, `L40`
- controls:
  - same-label `conflict` centroid patch
  - random-direction norm-matched patch

Baseline:
- valid JSON: `1.0000` (`192/192`)
- follows setting: `0.9740` (`187/192`)
- follows strategy: `0.0000` (`0/192`)

## Results

### L28

- `swap_to_aligned`
  - size changed: `0.0365` (`7/192`)
  - intended erasure flip: `0.0052` (`1/192`)
  - follows setting: `0.9375` (`180/192`)
  - follows strategy: `0.0052` (`1/192`)

- same-label control
  - size changed: `0.0312` (`6/192`)
  - intended erasure flip: `0.0000` (`0/192`)
  - follows setting: `0.9427` (`181/192`)
  - follows strategy: `0.0000` (`0/192`)

- random control
  - size changed: `0.0156` (`3/192`)
  - intended erasure flip: `0.0104` (`2/192`)
  - follows setting: `0.9583` (`184/192`)
  - follows strategy: `0.0104` (`2/192`)

Read:
- `L28` is the only layer where the aligned-centroid patch produces more
  movement than the `L32/L40` null-like layers
- but the directional effect is still tiny and does **not** clearly beat the
  random control

### L32

- `swap_to_aligned`
  - size changed: `0.0052` (`1/192`)
  - intended erasure flip: `0.0000` (`0/192`)

- same-label control
  - size changed: `0.0052` (`1/192`)
  - intended erasure flip: `0.0000` (`0/192`)

- random control
  - size changed: `0.0104` (`2/192`)
  - intended erasure flip: `0.0052` (`1/192`)

Read:
- effectively null
- same conclusion as the original first-pass run

### L36

- `swap_to_aligned`
  - size changed: `0.0208` (`4/192`)
  - intended erasure flip: `0.0156` (`3/192`)
  - follows setting: `0.9635` (`185/192`)
  - follows strategy: `0.0156` (`3/192`)

- same-label control
  - size changed: `0.0208` (`4/192`)
  - intended erasure flip: `0.0104` (`2/192`)
  - follows setting: `0.9635` (`185/192`)
  - follows strategy: `0.0104` (`2/192`)

- random control
  - size changed: `0.0052` (`1/192`)
  - intended erasure flip: `0.0000` (`0/192`)

Read:
- `L36` is the strongest directional-looking layer in this sweep
- but the effect is still weak:
  - only `3/192` intended flips
  - same-label control already produces `2/192`
- this is suggestive, not clean causal evidence

### L40

- `swap_to_aligned`
  - size changed: `0.0052` (`1/192`)
  - intended erasure flip: `0.0000` (`0/192`)

- same-label control
  - size changed: `0.0000` (`0/192`)
  - intended erasure flip: `0.0000` (`0/192`)

- random control
  - size changed: `0.0052` (`1/192`)
  - intended erasure flip: `0.0000` (`0/192`)

Read:
- effectively null again

## Overall Interpretation

This sweep does **not** produce strong causal evidence from last-token
single-layer `SwapMeanPatch`.

Best layer in the tested range:
- `L36`

But the effect remains very small:
- intended erasure flips at `L36`: `3/192`
- same-label control at `L36`: `2/192`

Current conclusion:
- `trade_size` remains strongly readable
- the representation may still be causally relevant
- but this patch format is too weak or too lossy to show a convincing effect
  on its own

Most defensible read:
- strong readout does not imply strong writable leverage at the same token/site
- if we continue causal work, we should upgrade fidelity rather than just keep
  sweeping the same low-bandwidth intervention

## Operational Note

All expensive patch runs completed successfully.

The workflow failed only in the final `compare_patch_runs` step because the
remote row-evaluator import re-executed the workflow module and hit a relative
dataset path issue.

This was fixed in the workflow spec by resolving the Phase 09 dataset path from
the workspace root, and the sweep metrics above were aggregated directly from
the completed patch artifacts.
