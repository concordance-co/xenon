# Residual capture and write coordinates

## The bug

Xenon previously forwarded `ResidualSite(site="resid_post", layers=[L])`
directly to vLLM's `eagle_aux_hidden_state_layer_ids`, while patching indexed
`model.layers[L]` and modified its **output**. These are not the same coordinate
system in the reviewed backends:

```text
embedding                 auxiliary index 0
decoder block 0 output    auxiliary index 1   / write block 0
decoder block 31 output   auxiliary index 32  / write block 31
decoder block 32 output   auxiliary index 33  / write block 32
final block N-1 output    auxiliary index N   / write block N-1
final normalization       NOT resid_post of the final block
```

Thus nominal capture 32 -> write 32 transplanted an earlier boundary into a
later one. This was an adapter/indexing bug, not an inherent distinction
between reading and writing neural representations. A research-motivated
cross-layer transplant remains possible, but must not be mistaken for same-site
interchange.

Two proposed corrections change different parts of an experiment:

| Preserve | Raw auxiliary capture -> physical write |
| --- | --- |
| Captured representation | 32 -> 31 |
| Physical write location | 33 -> 32 |

Neither correction guarantees behavioral effects. Tensor identity, score
movement, and actual selected-answer changes are distinct endpoints. This PR
does not relabel past experiments as same-site or claim that indexing explains
their behavioral outcomes.

## New contract

New Python `ResidualSite` instances default to `layer_indexing="decoder_block"`.
For `resid_post`, layer L is the **combined, unnormalized residual after physical
decoder block L**, zero-based, matching `ResidualInterventionSite`.

The write adapter also combines vLLM's `(branch, residual)` output before
applying an edit. It replaces the combined state. It does not add the old
residual to a replacement value. Unchanged components keep their original
split values, so an exact self-patch preserves the computation.

PR #5 fixes the index translation. This branch also fixes the split-state
write helper. Earlier writes changed only the branch tensor. An index offset
alone cannot correct those historical interventions. Re-run them before
making a same-site claim. Legacy capture mode preserves capture requests;
it does not restore the old write bug.

```python
capture_site = ResidualSite(name="options", site="resid_post", layers=[32])
write_site = ResidualInterventionSite(site="resid_post", layers=(32,))
# Xenon asks the reviewed vLLM backend for auxiliary index 33 internally.
# No offset in source_layer_map is needed for this new capture.
```

Both standalone and reusable-session capture use the translation. A reusable
session stores its complete ordered auxiliary-index union; each operation
selects its own requested boundaries from that union. Feature keys remain the
requested public layer labels, not translated backend indices.

The convention enters spec serialization/hashing. Feature payloads and derived
activation banks preserve `residual_coordinates`, including coordinate schema,
public-to-backend mapping, and runtime audit metadata. Persisted artifacts are
never automatically renumbered.

## Compatibility and migration

1. **Already stored captures, banks, directions and bases:** their tensors and
   labels are unchanged. Missing coordinate metadata means *unknown legacy
   provenance*, not proof of physical-block indexing. Consult the original
   runtime. This fix does not rewrite old artifacts or upstream catalog entries.
2. **Old serialized capture specs without `layer_indexing`:** deserialize as
   `"vllm_aux"`, preserving their raw-index requests on vLLM. They emit a warning
   when executed on vLLM; serialization makes the legacy convention explicit.
3. **Old Python workflow files:** constructor defaults now change their
   capture semantics. To reproduce their old requests, explicitly add
   `layer_indexing="vllm_aux"` to each residual capture site. Do this especially
   when a workflow already compensates with `source_layer_map`.
4. **Migrating to block coordinates:** remove the old offset compensation only
   when using a new block-indexed source. For example, to keep the old captured
   auxiliary-32 tensor's physical location, recapture block 31 and write block
   31. To keep the old physical write location 32, recapture block 32 and write
   block 32. These preserve different things; choose deliberately.
5. **Mixing sources:** explicit cross-layer mappings remain supported. A newly
   produced legacy-tagged bank requires `source_layer_map` for every write
   block. Banks with no provenance retain compatibility but emit a warning;
   Xenon cannot infer their original semantics safely.

```python
# Reproduce an old capture and explicitly patch the matching boundary in the
# audited Llama runtime. This is NOT the recipe for a new block-indexed bank.
legacy_site = ResidualSite(
    name="options", site="resid_post", layers=[32], layer_indexing="vllm_aux",
)
# InterchangePatch(..., write_site=ResidualInterventionSite(
#     site="resid_post", layers=(31,)), source_layer_map={31: 32})
```

Legacy auxiliary index 0 can represent embeddings; it is not a writable
decoder block -1. Do not globally subtract one from every stored layer label.
Derived historical projections need the same provenance review as full-state
banks; this PR does not infer coordinates for arbitrary custom transforms.

## Supported boundary adapters and validation

Canonical capture fails closed on unreviewed runtime/model combinations and
pipeline parallelism, rather than assuming that every backend uses the same
offset. Initial source-reviewed adapters:

- vLLM 0.19.0: `LlamaModel`.
- vLLM 0.25.1 (current pinned runtime): `LlamaModel`, `Qwen2Model`,
  `Qwen3Model` (inherits Qwen2's forward), and `Qwen3MoeModel`.

Before generation each worker validates the actual decoder class/version,
complete block range and requested bounds, checks the reviewed forward's
post-block auxiliary call, and records its source digest. Raw `vllm_aux` mode
is available for deliberate backend-specific work; it asserts no portable
block semantics. `resid_pre` is not implemented by this translation.

Source references:

- [Llama forward, v0.25.1](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/models/llama.py)
- [Qwen2 forward, v0.25.1](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/models/qwen2.py)
- [Qwen3 inherited forward, v0.25.1](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/models/qwen3.py)
- [Qwen3 MoE forward, v0.25.1](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/model_executor/models/qwen3_moe.py)
- [Auxiliary configuration pass-through, v0.25.1](https://github.com/vllm-project/vllm/blob/v0.25.1/vllm/v1/worker/gpu_model_runner.py)

Regression tests cover first/middle/final block outputs against independent
toy residual states, mixed legacy/canonical sites, session supersets,
serialization/hashing, provenance, and fail-closed guards.

Run the opt-in live check from the Xenon checkout:

```bash
XENON_RUN_MODAL_RESIDUAL_BOUNDARIES=1 .venv/bin/python -m pytest -q \
  tests/pipelines_v2/engine/test_modal_residual_boundaries.py
```

This check uses one Modal L4, Qwen3-0.6B, eager BF16 execution, and float32
capture storage. It compares first, middle, and final captures with independent
block-output sums. It then checks a self-patch through the intervention path.
It checks activation deltas, generated tokens, and patch execution status.
The standard vLLM image includes the C compiler required by Triton's driver.
Other model families and compiled execution require their own live checks.
