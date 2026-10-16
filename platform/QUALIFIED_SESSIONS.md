# Measured SDXL workflow sessions

`ModalResources(enable_workflow_sessions=True)` opts an SDXL runner into one
workflow-lifetime Modal app. Existing DAG dependencies and artifact references
remain unchanged. Set one shard/worker and explicit GPU, memory and timeout
bounds. Other engines retain their existing execution path.

Before each resolved SDXL step's reuse decision, this scope obtains fresh GPU
admission: actual model bytes, source, locked image/runtime and numerical/device
identity. This allocates a GPU even when every artifact is reused; it does not
load model weights for reuse-only work. CPU preflight occurs before the scope's
first GPU call. The prepared model volume supplies `HF_HOME`; offline model
preparation remains a separate recorded operation.

The first useful image in each exact operation/hook profile is its canary.
After ordinary image publication, an ordinary C03 `qualification` artifact
records actual loading/execution process-tree RSS, CUDA reserved-memory peaks,
latency, device/driver, exact envelope, hook coverage and ten-percent memory
headroom. Only a durable measured pass permits the next image. Generation alone
cannot qualify an edited profile. Unknown measurements fail closed. Current
scope accepts homogeneous dimensions/steps/CFG within each operation and one
GPU worker; it does not authorize multi-worker fanout or profile subsumption.
Failed measurements preserve the useful image and diagnostic receipt, while
preventing further submission. One exact-profile writer lock protects ownership.

Compatible dependent steps share a private SDXL session. Each image resets its
Euler scheduler, RNG, latents and owned hooks. Compatibility binds observed
source/model/runtime/numerics, resources/environment and the versioned reset
contract; incompatible state closes the previous model before loading another.
The worker closes models after at most 300 idle seconds, on operation failure,
and at scope exit. Provider eviction may still cause another load. Image
metadata records session/worker IDs and cumulative model loads; runner
`session_events` reports fresh observations, replacements, completions and close.
A changed worker never silently counts as a warm continuation.

Live qualification is recorded separately from fixture tests. SDXL receipts do
not qualify GPT-2 CUDA, backward/training, vLLM or external-tool profiles. The
existing vLLM batch/session behavior is unchanged. The option remains opt-in. A two-step SDXL run shared one model load; a reuse-only repeat loaded no weights. See the [support matrix](research_hub_support.json) for the exact evidence and [backburner](BACKBURNER.md) for future LLM session support.
