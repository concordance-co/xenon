# Native neuron microscope

One reusable workflow captures four signed Qwen MLP quantities for layer-0 neurons 0–2 on four original text fixtures. It retains bounded top/bottom, random and weak contexts, builds three evidence-linked dossiers, and catalogs a local ReportSpec artifact. One prompt has baseline, identity and single-neuron gate-zero arms. Two neurons deliberately remain unexplained.

This is a plumbing fixture. The stored description is unverified. It does not measure explanation quality, paper findings, neuron cleanliness, or scientific causality.

## Prepare once

From the repository root, follow the exact isolated **core-local** lock installation in [local runtime setup](../../local_research/phase_00/README.md#explicit-preparation-network-allowed-here-only), exporting `XENON_LOCAL_PYTHON` and `PYTHONPATH`. Use the Python 3.13.6 environment from that complete lock, not an arbitrary shared `.venv`.

Prepare the exact nine Qwen files using the separate [native model preparation command](../../../platform/NATIVE_SITES.md#prepare-and-run-one-real-example). Set `XENON_NATIVE_MODEL_CACHE` to that directory. The pinned revision is `c1899de289a04d12100db370d81485cdf75e47ca`; execution is CPU FP32 eager, plain text, no chat template or thinking wrapper. Execution never downloads a model. Actual model/tokenizer bytes and execution-host runtime identity are recorded by C01; the cache directory is part of the workflow recipe, so relocating it changes the recipe hash.

## Run and read

```bash
export XENON_HOME="$HOME/.xenon-neuron-microscope"
export PYTHONPATH="$PWD"
"$XENON_LOCAL_PYTHON" -m pipelines_v2.cli workflow plan --file workspaces/native_neurons/phase_00/workflow.py
"$XENON_LOCAL_PYTHON" -m pipelines_v2.cli workflow run \
  --file workspaces/native_neurons/phase_00/workflow.py --profile local-scratch \
  --scratch-model-cache "$XENON_NATIVE_MODEL_CACHE" --run-id wr_neurons_001 > "$XENON_HOME-run.json"
"$XENON_LOCAL_PYTHON" -m workspaces.native_neurons.phase_00.inspect_run \
  --run-json "$XENON_HOME-run.json" --output "$XENON_HOME/REPORT.md"
```

The strict local-scratch profile enforces no network/cloud, 8GiB aggregate process-tree RSS, 600s and 256MiB new artifacts. This workflow uses four examples and two generated tokens. Artifacts and FileCatalog are local under XENON_HOME; shared Postgres defaults are ignored. Rerun with a fresh run ID. Interrupted whole-profile compute resume is not qualified.

The inspector verifies every C03 artifact, requires complete model-step C01 identities, checks the actual transform envelope/coverage, and renders a bounded dossier reading view. The canonical ReportSpec consumes only the dossier artifact. Inspect each artifact’s provenance before requesting reuse; older fixture artifacts may retain unresolved fields.

## Reuse

`pipelines_v2.operations.neurons.summarize_observations(iterator, SummaryLimits(...))` accepts bounded original text/token-window references and all four signed values. Selection uses product rank, not absolute magnitude. Histograms include under/overflow. Quantiles and empirical percentiles use a deterministic seeded min-hash reservoir; they are approximate when that reservoir is smaller than the observed count. Retained IDs are order-independent; floating-point sums may differ at roundoff precision. Duplicate tracking is bounded by `max_observations` and retained contexts by `max_neurons`, `k`, and reservoir size.

`summarize_capture` adapts existing selected native features through TransformSpec. It checks four-way model/site/token/unit alignment and an explicit `max_feature_bytes` budget (32 MiB by default). It loads those bounded features, not an arbitrary large capture stream. Original text and document references are caller-supplied and tied to the workflow recipe; no character-to-token offsets are inferred.

`StoredExplanationProvider` / `ManualExplanationProvider` accept versioned `ExplanationCandidate` records. A provider receives only the discovery reservoir, in either product-only or all-signed-values view. Candidates must reference visible discovery context IDs; heldout rows never cross that boundary. Missing candidates remain in denominators. The interface does not evaluate candidates. Predictiveness, blinded/shuffled controls and human audit require a new study; see [deferred research](../../../platform/BACKBURNER.md#research-is-its-own-workstream).
