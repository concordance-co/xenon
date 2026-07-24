# vLLM 0.25.1 Upgrade

## Decision

Adopt vLLM 0.25.1 for Xenon's Modal runtime. The upgrade preserves the
ground-truth behavior of the vLLM 0.19.0 runtime across generation, generated
token residual capture, MoE routing capture, request-scoped activation
interventions, compiled patch operators, and structured output. The fixed
Qwen3-30B-A3B/A100 benchmark found no output or performance regression.

The runtime stack is pinned as a unit:

```text
vllm==0.25.1
torch==2.11.0
transformers>=5.5.3
safetensors>=0.6.2
```

This is a deliberate compatibility boundary. Do not bump one of these packages
independently without rerunning the tests below.

## Upstream Changes That Affect Xenon

The upstream comparison from v0.19.0 to v0.25.1 spans 3,154 commits and 4,167
files. The relevant integration changes are:

- Model Runner v2 was introduced and progressively enabled. It does not support
  the `extract_hidden_states` path that Xenon uses for residual capture.
- vLLM now ships an `ExampleHiddenStatesConnector` with synchronized
  `load_hidden_states()` and `cleanup_hidden_states()` calls. Generated-token
  hidden-state capture is requested through
  `kv_transfer_params.include_output_tokens`.
- GPU worker/model loading changed substantially. Xenon's previous copied
  `load_model` implementation was coupled to v0.19 internals.
- Fused MoE dispatch moved toward factory-created `MoERunner` objects, and
  `select_experts` gained additional arguments. Kernels can also be
  monolithic, with no observable router-selection boundary.
- The supported dependency floor moved to Torch 2.11 and Transformers 5.
- Reasoning parser marker attributes changed names.
- v0.25.1 adds two targeted fixes over v0.25.0: startup no longer requires
  system FFmpeg when TorchCodec is unused, and mixed-dtype allreduce/RMSNorm
  quantization fusions are guarded.

Primary upstream references:

- https://github.com/vllm-project/vllm/compare/v0.19.0...v0.25.1
- https://github.com/vllm-project/vllm/releases/tag/v0.25.1

## Xenon Compatibility Work

- Keep residual capture on Model Runner v1 because vLLM 0.25.1 rejects
  `extract_hidden_states` under Model Runner v2. A follow-on compatibility
  branch ports request-scoped interventions to V2 and uses a documented hybrid
  runner policy; see `platform/VLLM_MODEL_RUNNER_V2.md`.
- Use vLLM's native hidden-state connector protocol, including output-token
  opt-in and synchronized cleanup. Retain a direct safetensors fallback for
  CPU-only tests and controlled compatibility failures.
- Delegate worker model loading to vLLM and install Xenon's request-scoped
  intervention hook after the model is constructed. This removes the copied
  v0.19 loader.
- Support both v0.19 and v0.25 `select_experts` call signatures and discover
  routing boundaries through either `FusedMoE` or its `MoERunner`. Missing,
  monolithic, or observation-free routing paths now fail loudly.
- Disable only the FlashInfer sampler with
  `VLLM_USE_FLASHINFER_SAMPLER=0`. In the slim Modal image, its JIT path
  requires an unavailable `nvcc`; attention remains on FlashAttention and MoE
  remains on Triton.
- Accept both old and new reasoning-parser marker attributes.
- Record tokenizer, preflight, engine initialization, model readiness,
  generation, token counts, and token throughput in capture/generation
  artifact metadata.
- Retry transient Modal volume-download RPC failures with bounded backoff.
  Permanent errors still fail immediately.

## Validation

The v0.19.0 baseline was run from commit `5564d89` in a detached worktree. The
upgrade was run from the same commit plus only the compatibility changes on an
A100-80GB with `/models/Qwen/Qwen3-30B-A3B`.

### Correctness

| Contract | v0.19.0 | v0.25.1 |
| --- | ---: | ---: |
| Local suite | ground truth | 378 passed, 6 skipped |
| Model-bound basic, unpaired, and paired patch matrix | 5 passed | passed |
| Generated-token residual plus MoE routing | passed | passed |
| Compiled custom-op activation patching | passed | passed |
| Structured JSON-schema output | supported | passed; exact expected object |
| Fixed benchmark output digests | reference | 16/16 identical |

The paired benchmark generated 2,048 tokens across 16 deterministic requests;
every request hit the 128-token length limit in both versions.

### Performance

These are single-run upgrade checks, not a general serving benchmark. They are
useful for detecting a large regression under an identical Xenon workload.

| Measurement | v0.19.0 | v0.25.1 | Change |
| --- | ---: | ---: | ---: |
| Generated tokens/second | 496.17 | 753.30 | +51.8% |
| Model ready, end to end | 201.71 s | 109.45 s | -45.7% |
| Checkpoint weights | 57.86 s | 29.73 s | -48.6% |
| vLLM core model load | 60.62 s | 32.53 s | -46.3% |
| Engine init/profile/compile | 122.03 s | 57.18 s | -53.1% |

The old runtime did not emit Xenon's new structured timing metadata, so its
numbers were taken from vLLM progress/log timestamps. The new throughput number
comes from artifact metadata. Workflow evidence:

- v0.19.0 throughput: `wr_a6855c9b5945_105ef9ae`
- v0.25.1 throughput: `wr_21d48586c0d8_725e326c`
- v0.25.1 generated residual/MoE capture: Modal app
  `ap-LTlMFo9TkghU1m6DRAvDdM`
- v0.25.1 structured output: Modal app `ap-5nYAQz5n2euanwFvmeR80j`

## Reproduction

Run the local suite:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -m pytest -q tests
```

Run the fixed throughput workflow:

```bash
PYTHONDONTWRITEBYTECODE=1 \
XENON_VLLM_BENCHMARK_LABEL=v0251 \
uv run python -m pipelines_v2.cli workflow run \
  --file scripts/pipelines_v2_vllm_upgrade_benchmark.py \
  --logging INFO
```

Run the real GPU compatibility matrix:

```bash
PYTHONDONTWRITEBYTECODE=1 \
XENON_RUN_MODAL_VLLM_GPU_SMOKE=1 \
XENON_RUN_MODAL_VLLM_ENGINE_CONTRACTS=1 \
XENON_RUN_MODAL_VLLM_PATCH_OPERATOR_CONTRACTS=1 \
XENON_RUN_MODAL_VLLM_PAIRED_PATCH_CONTRACTS=1 \
XENON_RUN_MODAL_VLLM_CAPTURE_CONTRACTS=1 \
XENON_RUN_MODAL_VLLM_OUTPUT_CONTRACTS=1 \
uv run python -m pytest -q tests/pipelines_v2/engine/test_modal_vllm_gpu_smoke.py -s
```

## Known Boundaries

- Xenon residual capture intentionally remains on Model Runner v1. The
  follow-on V2 integration covers generation, structured output, routing-only
  capture, and activation patching while retaining V1 for the unsupported
  hidden-state connector path.
- Monolithic fused-MoE kernels are rejected for routing capture because they do
  not expose the router-selection observations required by Xenon's artifact
  contract.
- The GPU validation used tensor parallel size 1 and pipeline parallel size 1.
  Revalidate target multi-GPU topologies before treating them as release-gated.
- The performance table is one controlled paired run. Repeat it when GPU image,
  model, quantization, or compile settings change.
