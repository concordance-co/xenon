# Qwen3 PLT circuit-tracer plumbing

This opt-in workflow runs the official circuit-tracer **v0.5.3** on the documented public Qwen3-0.6B / `mwhanna/qwen3-0.6b-transcoders-lowl0` pair. These are **per-layer transcoders (PLTs)**. Live CLT capability is not claimed.

Prepare assets separately on a CPU worker, into the existing `xenon-models` Modal volume:

```bash
python -m pipelines_v2.operations.circuit_tracer prepare-modal
```

Preparation downloads only the pinned model/tokenizer and 28 transcoder weight files plus configuration (~20.3 GB total). Optional feature-visualization assets are excluded. Every file is verified against the checked-in content lock. Normal execution is offline and fails on missing or changed files.

Run the public workflow:

```bash
export XENON_TRACER_OUTPUT=/tmp/xenon-circuit-trace-report
python -m pipelines_v2.cli workflow plan --file examples/circuit_trace_smoke/workflow.py
python -m pipelines_v2.cli workflow run --file examples/circuit_trace_smoke/workflow.py --logging INFO
```

The separate tool interpreter has its own fully pinned dependencies; it does not modify Xenon's control interpreter or the MAttr environment. Modal CPU readiness checks the interpreter, source/package identities and prepared assets before GPU work. The bounded profile uses one A100-80GB, 64 GiB host memory, a 20-minute worker ceiling, a 25-minute client ceiling, FP32, one short prompt (`Hello`), at most two tokens including the special prefix, two selected feature nodes, one logit target and backward batch size one.

Artifacts preserve the raw official `.pt` graph and JSON export, content hashes, reversible node mapping, signed matrix weights, all error nodes, feature-selection omissions, numerical settings and measured GPU memory. The importer never unpickles upstream objects in the control interpreter. `graph.html` and `graph.json` are written locally after the trace artifact completes.

The report clearly separates local linear replacement-flow edges from native causal effects. Attention patterns and normalization scales are frozen during attribution. One same-prompt no-error-correction forward is a plumbing diagnostic, not held-out fidelity. A tiny original-model feature edit is attempted only if the pinned HF and TransformerLens forward agree at the declared FP32 tolerance; otherwise it stays explicitly unavailable. Even a successful edit remains **unvalidated** without matched controls. Scientific verdict: **not tested**.

Pins and sources:

- [Official release v0.5.3](https://github.com/decoderesearch/circuit-tracer/tree/255d989e93be1299d53d67b59e9fd76a0e274b14)
- [Supported pair listing](https://github.com/decoderesearch/circuit-tracer/blob/255d989e93be1299d53d67b59e9fd76a0e274b14/README.md#available-transcoders)
- [Model snapshot](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca)
- [PLT snapshot](https://huggingface.co/mwhanna/qwen3-0.6b-transcoders-lowl0/tree/28aefe686c09e9bc1a862195b51e145f10868d87)

For a different workload, verify its resource envelope with a small real run. Original-model parity and broader model coverage are on the [backburner](../../platform/BACKBURNER.md). Fixture-only CLT format round-trips are not live CLT evidence.
