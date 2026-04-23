# Trade Size Interchange First Pass

Run:
- workflow: `prompt_confusion_trade_size_interchange_first_pass`
- run id: `wr_e2a6251b826a_416e4d18`
- comparison artifact:
  - `patch_comparison_605f0ee39d09_fe4b7521`

Core setup:
- family: `trade_size`
- scope: conflict rows only (`n = 192`)
- paired donor-target patch:
  - donor = aligned row
  - target = conflict row
  - pair key = `matched_group_id`
- token: last prompt token
- write layers:
  - `L28`
  - `L36`
- controls:
  - random norm-matched patch at `L28`
  - random norm-matched patch at `L36`

Why this was run:
- the mean-swap sweeps were too weak / lossy
- interchange preserves the actual donor row state instead of a class centroid

Baseline:
- valid JSON: `1.0000` (`192/192`)
- follows setting: `0.9740` (`187/192`)
- follows strategy: `0.0000` (`0/192`)

## Results

### Interchange at `L28`

- valid JSON: `1.0000` (`192/192`)
- size changed: `0.0365` (`7/192`)
- patched follows setting: `0.9375` (`180/192`)
- patched follows strategy: `0.0000` (`0/192`)
- intended erasure flip: `0.0000` (`0/192`)
- malformed: `0.0000`

Main qualitative pattern:
- `L28` interchange moves more rows than mean-swap
- but every changed row fell to `size = none`, not to strategy-consistent
  `large`
- so this is perturbational movement, not clean causal control

### Interchange at `L36`

- valid JSON: `1.0000` (`192/192`)
- size changed: `0.0208` (`4/192`)
- patched follows setting: `0.9844` (`189/192`)
- patched follows strategy: `0.0052` (`1/192`)
- intended erasure flip: `0.0052` (`1/192`)
- malformed: `0.0000`

Main qualitative pattern:
- `L36` produced exactly one clean intended flip:
  - `small -> large`
- the other changed rows mostly repaired baseline `none` outputs back to
  settings-consistent `small`

### Random controls

`L28` random:
- intended erasure flip: `0.0052` (`1/192`)

`L36` random:
- intended erasure flip: `0.0000` (`0/192`)

## Interpretation

The interchange version is somewhat better than the mean-swap version, but it
still does not deliver strong causal leverage.

What improved:
- more rows changed at `L28`
- `L36` produced one clean intended semantic flip

What did not improve enough:
- the dominant `L28` failure mode is `size = none`, not `small -> large`
- the clean intended flip count remains tiny
- control-subtracted evidence is still weak

Current best read:
- the `trade_size` representation is probably real
- donor-row interchange is stronger than centroid swap
- but last-token single-layer patching is still not enough for a convincing
  causal result on this benchmark

Most likely lesson:
- we need either:
  - a more position-aware intervention
  - or a more structured path-style patch
- not just better averaging
