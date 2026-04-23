# Trade Size Activation Patch First Pass

Run:
- workflow: `prompt_confusion_trade_size_activation_patch_first_pass`
- run id: `wr_89401f421728_dfe90da5`
- comparison artifact: `patch_comparison_1_478e9593`

Setup:
- family: `trade_size`
- scope: conflict rows only (`n = 192`)
- write site: `L32`
- token: last prompt token
- operator: `SwapMeanPatch`
- target patch: move conflict rows toward the `aligned` centroid
- controls:
  - same-label `conflict` centroid patch
  - random-direction norm-matched patch

Question:
- if the `trade_size` conflict state at `L32` is causally load-bearing, does
  moving conflict rows toward the aligned centroid reduce settings-following
  and increase strategy-following on the `size` field?

Result:
- baseline conflict-row behavior:
  - valid JSON: `1.0000` (`192/192`)
  - follows setting: `0.9740` (`187/192`)
  - follows strategy: `0.0000` (`0/192`)

- `swap_to_aligned`:
  - valid JSON: `1.0000` (`192/192`)
  - size changed: `0.0052` (`1/192`)
  - follows setting: `0.9688` (`186/192`)
  - follows strategy: `0.0000` (`0/192`)
  - intended erasure flip: `0.0000` (`0/192`)
  - malformed: `0.0000` (`0/192`)

- `same_label_control`:
  - valid JSON: `1.0000` (`192/192`)
  - size changed: `0.0052` (`1/192`)
  - follows setting: `0.9688` (`186/192`)
  - follows strategy: `0.0000` (`0/192`)
  - intended erasure flip: `0.0000` (`0/192`)
  - malformed: `0.0000` (`0/192`)

- `random_control`:
  - valid JSON: `1.0000` (`192/192`)
  - size changed: `0.0104` (`2/192`)
  - follows setting: `0.9635` (`185/192`)
  - follows strategy: `0.0052` (`1/192`)
  - intended erasure flip: `0.0052` (`1/192`)
  - malformed: `0.0000` (`0/192`)

Interpretation:
- this is effectively a null result for `L32` last-token `SwapMeanPatch`
- the aligned-centroid patch did not outperform either control
- the representation is strongly readable at this site, but this first write
  did not produce meaningful directional control over `size`

Most likely read:
- this does **not** falsify the `trade_size` representation
- it says the current intervention site/operator is too weak, too late, or too
  lossy to move behavior
- the next best move is to sweep write layers, especially earlier layers like
  `L28`, before making stronger causal claims

Operational note:
- the first run hit a Modal artifact-localization collision when two patch
  steps shared the same centroid artifact
- the fix was to materialize separate centroid steps for the aligned-patch and
  same-label-control branches
- no core `pipelines_v2` code was changed; this was handled entirely in the
  project workflow spec
