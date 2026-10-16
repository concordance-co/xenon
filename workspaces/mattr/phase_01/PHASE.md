# MAttr second-model plumbing

## Premise

Verify that Xenon's pinned official MAttr/MIB integration runs the supported
Qwen2.5-0.5B IOI cell as well as GPT-2. The user's current scope is platform
plumbing, not reproduction quality or method superiority.

## Design

One seed, two fitting steps, two validation examples, original train split,
official ten-point sparsity curve. Qwen uses upstream BF16; GPT-2 uses upstream
FP32. Preserve that difference. No cross-model average, ranking victory or
causal explanation follows from this check. CPU execution after explicit asset
preparation; bounded subprocess timeout, RSS and output size.

## Artifacts

`workflow.py` is executable source. Runtime/model/data pins and producer hashes
travel with the request/result. The existing GPT-2 recipe remains in
`workspaces/mattr_ioi_smoke/`. Actual run evidence is recorded after execution.

## Plumbing result

`wr_t11_qwen25_mattr_smoke` completed on CPU with 361 scored nodes and ten official
curve points; `external_mattr_1_c6345d93a805` passed C03 verification. A separate
analysis-only run produced `transform_1_8c2d2571cf93` (42,797-byte report summary).
Generic C01 remains unresolved for file-input/numerical-policy observation.
Scientific verdict is `not_tested`; no success criterion depends on the ranking.

## Deferred research

The original T11 reference matrix (three seeds, baselines, scientific controls
and reference reproduction) remains unexecuted. It is not needed to establish
that this second supported model passes through the platform.
