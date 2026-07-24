# vLLM Model Runner V2

## Decision

On this experimental branch, use Model Runner V2 for ordinary generation,
structured output, routing-only capture, and request-scoped activation
interventions. Keep residual capture on Model Runner V1 until vLLM supports
Xenon's hidden-state extraction contract under V2.

Do not promote this runner policy to the production branch yet. The fixed
Qwen3-30B-A3B benchmark preserves output exactly but regresses throughput by
16.3% and model-ready time by 80.7%. The MRv2 intervention port is functional;
the global MRv2 default is not currently a performance win for Xenon's primary
MoE workload.

This is a hybrid runner policy, not a silent fallback:

| Workload | Runner | Reason |
| --- | --- | --- |
| Generation | V2 | Current vLLM execution path |
| Structured output | V2 | Uses the ordinary generation path |
| MoE routing-only capture | V2 | The routing observer is independent of residual extraction |
| Request-scoped interventions | V2 | Xenon now implements the V2 request lifecycle |
| Residual capture | V1 | vLLM 0.25.1 rejects `extract_hidden_states` under V2 |
| Mixed session containing residual capture | V1 | One loaded engine cannot mix runner implementations |

`VLLMEngine.runtime_spec()` selects V2 by default. Residual-capture entry
points override that selection before constructing `VllmConfig`; intervention
entry points assert that the constructed worker actually uses Xenon's V2
bridge.

## Why Residual Capture Was on V1

Xenon does not obtain residuals from an incidental Python forward hook. It uses
vLLM's `extract_hidden_states` speculative method with
`ExampleHiddenStatesConnector` so that prompt and generated-token activations
are transferred and cleaned up through vLLM's request lifecycle.

In vLLM 0.25.1, `_get_v2_model_runner_unsupported_features()` rejects that
speculative method. Leaving runner selection automatic makes vLLM choose V1;
forcing `VLLM_USE_V2_MODEL_RUNNER=1` raises:

```text
Model Runner V2 does not yet support: speculative method 'extract_hidden_states'
```

Model Runner V2 has internal auxiliary-hidden-state tensors for supported
speculative decoders, but it does not expose the connector protocol Xenon's
capture artifacts rely on. A direct Xenon worker could collect those tensors,
but that would be a new capture implementation with its own batching,
generated-token alignment, cleanup, tensor-parallel, and pipeline-parallel
contracts. It is not equivalent to changing an environment variable.

The smallest credible direct-capture follow-up is:

1. Start with eager TP=1/PP=1 and split V2's flattened `InputBatch` outputs by
   request and absolute token span.
2. Reproduce the current prompt/generated token-alignment artifact exactly.
3. Validate concurrent requests, preemption, and cleanup before compiled mode.
4. Add TP coverage. PP remains blocked until non-last stages can return their
   local auxiliary residuals rather than discarding them.

Until those checks pass, V1 residual capture is the functionality-preserving
choice.

### How large would an MRv2 residual adapter be?

Changing Xenon's persisted artifact expectations is unnecessary. MRv1 and
MRv2 expose the same model-level auxiliary residual values at the requested
Qwen3-MoE boundaries. The current connector returns
`hidden_states: [tokens, layers, hidden]`; Xenon permutes that to
`[layers, tokens, hidden]`, converts to CPU float32 NumPy, applies selectors and
pooling, and only then writes feature artifacts. It ignores the connector's
stored token ids after loading.

The important existing alignment convention is positional: prompt rows first,
then generated tokens that became inputs to another forward pass. The final
sampled token is absent because it never ran through the model. Preserving this
post-load tensor contract leaves all selectors, sections, pooling, and
safetensors storage unchanged.

The moderate-sized option is a Xenon-local MRv2 collector, estimated at
roughly 200–350 lines plus tests:

- enable the model's existing auxiliary hidden-state layers;
- truncate padded output to MRv2's real scheduled-token count;
- split flattened rows with `query_start_loc_np`;
- key chunks by request id and place them at
  `num_computed_tokens_np + local_offset`;
- use absolute-position last-write-wins across preemption/replay;
- drain per-request `[tokens, layers, hidden]` tensors after generation and
  feed the existing Xenon conversion code.

That prototype should initially require TP=1, PP=1 and
`enable_prefix_caching=False`. Prefix caching can leave unexecuted holes in a
direct collector, while PP non-final stages currently discard their local
auxiliary outputs. TP=2 is a later validation target.

A file-backed collector that recreates today's lock/cleanup lifecycle is
approximately 350–650 lines. A native vLLM cache-aware MRv2 port that preserves
prefix reuse, asynchronous connector writes, TP behavior, and eventual PP
transport is much larger—roughly 800–1,500 lines across an upstream patch or
fork.

So the artifact adaptation is not massive; lifecycle-equivalent capture is.
The recommended next experiment is the bounded in-memory collector with exact
MRv1/MRv2 tensor and artifact comparisons.

## Why Interventions Were on V1

The previous Xenon worker subclassed
`vllm.v1.worker.gpu_model_runner.GPUModelRunner` and depended on its
`_update_states()` and `_prepare_inputs()` hooks. Model Runner V2 is a separate
class with different lifecycle boundaries, so forcing V2 would have bypassed
request registration, flat batch-row mapping, and cleanup.

The V2 bridge now maps those responsibilities onto:

- `add_requests()` for request-scoped patch registration;
- `prepare_inputs()` for scheduled-token to flat-query row mapping;
- `execute_model()` for patch-stat collection and per-step cleanup;
- `finish_requests()` for completed and preempted request cleanup.

MRv2's replay-prefill length can include generated tokens after a preempted
request resumes, so Xenon reads the immutable original prompt length from
`req_states.prompt_len` when classifying prompt versus decode rows. Replacing a
streaming request under the same id also clears any old patch payload before
registration.

The custom worker installs the selected V1 or V2 runner only while vLLM creates
the runner, restores the upstream class afterward, and fails if vLLM builds a
different class. The model and artifacts record `model_runner=v2`, so a passing
intervention test proves actual runner selection rather than merely proving an
environment setting.

Compiled interventions require one extra precaution. vLLM compiles and CUDA
graph-captures dummy requests before real patch payloads exist. FULL graphs can
therefore bake in an inactive operator branch. Xenon explicitly uses PIECEWISE
CUDA graphs for patched runtimes, keeps the activation-patch custom op present,
and preselects the operator family. FULL graph support remains out of scope
until patch buffers are allocated before dummy capture and mutated in place.

## Compatibility Evidence

The v0.25.1 V2 intervention validation used
`/models/Qwen/Qwen3-30B-A3B` on A100-80GB, TP=1, PP=1:

- PIECEWISE-compiled, decode-only `project_out`: two concurrent requests,
  nonzero patch norms, prompt patch count zero, decode patch count positive,
  and `compiled_custom_op` dispatch.
- One reused V2 session ran `project_out`, `random_control`, `add_direction`,
  `swap_mean`, and `swap_components`, exercising registration and cleanup
  across operator families.
- The paired V2 matrix passed both request interchange and residual-path
  interventions under the final PIECEWISE graph policy.
- All six inspected intervention manifests record `model_runner=v2`.
- Routing-only capture discovered and recorded Qwen3-MoE router data at layers
  0 and 24 under V2.
- Structured JSON-schema generation returned the exact expected object under
  V2.
- The generated-token residual plus MoE contract passed under the hybrid
  policy and its artifact recorded `model_runner=v1`, proving the explicit
  fallback still overrides the runtime-wide V2 setting.

Evidence:

- Compiled smoke workflow: `wr_b58934939548_4471a065`
- Final PIECEWISE compiled smoke Modal app: `ap-7lwcuSks6rbVX2lzplzDKk`
- Five-operator Modal app: `ap-dvefYRjMRXDbJA58frNXr5`
- Paired interchange/residual-path Modal app: `ap-61t5pJcxlRoV25mPPa168l`
- Routing-only V2 Modal app: `ap-guj4qzBpRtoPckyeHdPSsZ`
- Structured-output V2 Modal app: `ap-TZc85svu6qwfLki1rDIdtj`
- Hybrid residual-capture Modal app: `ap-Y7Puyofx9m2LQmzmOoFpeL`

## Generation Correctness and Performance

The paired compiled benchmark generated 2,048 tokens across 16 deterministic
requests on A100-80GB. Every V1 and V2 output digest is identical, and every
request reached the same 128-token length limit.

| Measurement | MRv1 | MRv2 | Change |
| --- | ---: | ---: | ---: |
| Generated tokens/second | 753.90 | 630.92 | -16.3% |
| Model ready, end to end | 91.39 s | 165.14 s | +80.7% |
| Engine init/profile/compile | 90.55 s | 164.02 s | +81.1% |
| Generation time, 2,048 tokens | 2.72 s | 3.25 s | +19.5% |

These are paired single runs intended to catch large regressions, not a broad
serving benchmark. The compile/startup difference is large enough that it
should be reproduced and understood before MRv2 becomes the default for
Qwen3-MoE.

- MRv1 workflow: `wr_ce3e878246b0_a9f9d191`
- MRv1 Modal app: `ap-U0OgufR0untawq15Byl5Ns`
- MRv2 workflow: `wr_4181c20fc028_5371281f`
- MRv2 Modal app: `ap-pk7vx0QcCcHAwldA96EJKO`

## Known Boundaries

- Qwen3-MoE is not in vLLM 0.25.1's default V2 architecture allowlist. Xenon
  forces V2 only after validating its required paths; this remains an
  experimental integration boundary.
- GPU validation currently covers TP=1 and PP=1. Revalidate each target
  multi-GPU topology.
- Patched runtimes explicitly use PIECEWISE CUDA graphs. FULL graphs are not
  supported because dummy capture occurs before request-scoped patch state
  exists.
- A mixed reusable session containing residual capture uses V1 for every
  operation in that session. Split the workflow if V2 generation or
  intervention execution is required independently.
- Runner selection is process-global inside vLLM, so every Xenon engine
  construction selects explicitly and invalidates vLLM's environment cache.
  Loaded sessions retain their own runner metadata.
- Compiled patch buffers support at most 64 simultaneously patched requests;
  larger batches now fail explicitly rather than silently skipping requests.
- Residual capture remains on V1 by design; this is the only known runner
  exception in the current hybrid policy.

## Reproduction

Run the fixed benchmark once per runner:

```bash
PYTHONDONTWRITEBYTECODE=1 \
XENON_VLLM_BENCHMARK_LABEL=v0251-mrv1 \
XENON_VLLM_BENCHMARK_MODEL_RUNNER=v1 \
uv run python -m pipelines_v2.cli workflow run \
  --file scripts/pipelines_v2_vllm_upgrade_benchmark.py \
  --logging INFO

PYTHONDONTWRITEBYTECODE=1 \
XENON_VLLM_BENCHMARK_LABEL=v0251-mrv2 \
XENON_VLLM_BENCHMARK_MODEL_RUNNER=v2 \
uv run python -m pipelines_v2.cli workflow run \
  --file scripts/pipelines_v2_vllm_upgrade_benchmark.py \
  --logging INFO
```

Run the compiled intervention and operator contracts:

```bash
PYTHONDONTWRITEBYTECODE=1 \
XENON_RUN_MODAL_VLLM_GPU_SMOKE=1 \
XENON_RUN_MODAL_VLLM_PATCH_OPERATOR_CONTRACTS=1 \
XENON_RUN_MODAL_VLLM_ROUTING_V2_CONTRACT=1 \
XENON_RUN_MODAL_VLLM_OUTPUT_CONTRACTS=1 \
XENON_MODAL_VLLM_ENGINE_CONTRACT_SHARD_COUNT=1 \
XENON_MODAL_VLLM_ENGINE_CONTRACT_MAX_CONTAINERS=1 \
uv run python -m pytest -q \
  tests/pipelines_v2/engine/test_modal_vllm_gpu_smoke.py -s
```
