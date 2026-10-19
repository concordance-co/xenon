# Configurable Qwen2.5 crosscoder workflows

The supported pair is the immutable original Qwen2.5-0.5B base/instruct checkpoints in `model_lock.json`. Capture aligns document keys, exact token IDs and residual coordinates; fit uses the pinned upstream reference; import, encode, decode and feature edits produce ordinary artifacts. Feature edits cover every configured held-out row in both original models. Identity no-op, feature delta and full reconstruction are separate arms. Feature quality and scientific hypotheses are **not tested**.

## Configure and run

Use `pipelines_v2.representation.crosscoder.workflow.build_workflow` from Python, or provide a JSON config to the installed CLI:

```bash
python -m pipelines_v2.representation.crosscoder.workflow --config config.json --plan
python -m pipelines_v2.representation.crosscoder.workflow --config config.json
```

The checked-in `phase_00/workflow.py` reads `XENON_CROSSCODER_CONFIG`; it also works with ordinary `pipelines_v2.cli workflow plan/run/resume/rerun-step` commands. The small legacy environment-variable example remains available. `modal_smoke.json` is a six-prompt, width-64, six-step CUDA configuration; its paths, volumes and output directory are explicit examples to review before running.

Config accepts `prompts` or a serialized `Dataset`, ordered `model_paths`, `reference_path`, `fit`, `execution`, `limits`, explicit `train_keys` (or `train_count`), `layer`, `checkpoint_step`, `feature_index`, `amount`, `output`, and `mode`. `mode` is `local-scratch` or `modal`. All non-training rows are held out. Materialized datasets and bounded streaming Postgres preparation are supported. Other deferred sources must first be materialized within the caller's limits. Chat templates and image rows are rejected.

Fit width, steps and batch size are configurable; there are no demo-specific ceilings. Explicit limits bound examples, per-document/total tokens, prepared activation bytes, parameter/gradient/Adam bytes, output bytes, process RSS, device memory and wall time. Preparation rejects oversized work before model allocation; fitting gathers sampled token batches without concatenating the corpus. The prepared corpus is still held in host memory, bounded by `max_activation_bytes`; this is not an out-of-core trainer. CPU scratch applies aggregate process and publication limits; Modal applies worker allocation/time limits and the fitter has an independently supervised subprocess.

Execution supports FP32 eager on macOS arm64 CPU or Linux x86_64 CUDA, with separate exact locks. Device, deterministic algorithms, TF32 policy, actual GPU and model/reference content are recorded. CUDA requires an explicit GPU allocation; no CPU fallback occurs. Cross-device checkpoint resume is rejected. Automatic recovery from arbitrary interrupted optimizer steps is not claimed: `checkpoint_step` creates an ordinary finalized checkpoint and subsequent fit verifies alignment, source/runtime/policy, optimizer/scheduler, sampler and CPU/CUDA RNG before continuing.

## Prepare explicitly

Local preparation downloads pinned public bytes and checks the bundled SHA256 manifests. Workflows themselves stay offline.

```bash
uv venv --python 3.13 /private/tmp/xenon-crosscoder-venv
uv pip sync --python /private/tmp/xenon-crosscoder-venv/bin/python pipelines_v2/representation/crosscoder/requirements-darwin-arm64.lock
PYTHONPATH=. /private/tmp/xenon-crosscoder-venv/bin/python -m pipelines_v2.representation.crosscoder.prepare --reference /private/tmp/xenon-crosscoder-reference --cache /private/tmp/xenon-model-cache --models /private/tmp/xenon-crosscoder-reference/models.json
```

Modal preparation uses one CPU worker, the separate Linux lock and an existing volume (no implicit volume creation). It verifies package pins and `pip check`, then prepares `/models/hub` and `/models/xenon-crosscoder/reference`:

```bash
python -m pipelines_v2.representation.crosscoder.modal_prepare --volume xenon-models
python -m pipelines_v2.representation.crosscoder.workflow --config examples/qwen_crosscoder/modal_smoke.json --plan
python -m pipelines_v2.representation.crosscoder.workflow --config examples/qwen_crosscoder/modal_smoke.json
```

Compute stays on the mounted artifact volume; the report is local and consumes compact summaries. `summary.json` records the run and artifact IDs. A successful local proof used six prompts, 64 features, step-3 checkpoint resumed to step 6, feature 40 and two held-out rows: all eight manifests had resolved provenance, all four original-model no-ops and direct feature-delta oracle errors were exactly zero. This is a plumbing result, not a feature-quality result. The same eight-stage CUDA workflow also passed on one NVIDIA L4, with resolved provenance and zero no-op/direct-delta errors. Historical run receipts are outside the checkout; current supported scope is recorded in the [support matrix](../../platform/research_hub_support.json).

The six public specs are `CrosscoderPairCaptureSpec`, `CrosscoderFitSpec`, `CrosscoderImportSpec`, `CrosscoderEncodeSpec`, `CrosscoderDecodeSpec`, and `CrosscoderFeatureSpliceSpec`. Import accepts completed safe checkpoints from this pinned fitter; arbitrary tensor conventions are not inferred.

Reference: [science-of-finetuning/crosscoder_learning at d1fd6f3](https://github.com/science-of-finetuning/crosscoder_learning/tree/d1fd6f3fea41fc1123699cc09f4cdad47ca35160), MIT. The trainer and dictionary execute unchanged after verification; only the unused NNsight collector initializer is bypassed. Codes use `CrossCoder.encode`, not norm-scaled `forward(output_features=True)` output. Observed provenance substitutes verified asset content for machine-local paths, while unobserved identities remain ineligible for reuse.

A separate final-source two-step CUDA gate verified persisted admission records. A fresh macOS process then reused both capture and fit after two fresh CPU readiness calls, with GPU execution methods guarded to fail if invoked; neither GPU method ran. Source or input changes still invalidate reuse.
