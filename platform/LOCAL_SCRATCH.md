# Bounded offline local scratch

Run a small ordinary WorkflowSpec without Modal, Postgres, API providers or
network access:

```bash
uv run python -m pipelines_v2.cli workflow run --profile local-scratch --file /absolute/path/workflow.py
```

Workflow files keep `build_dataset()` and `build_workflow(dataset)`. Their import
and builders execute inside the same supervised child as the workflow. Runner
specs, if supplied, must be local. No external default runner or catalog is
constructed. Existing shared workflow commands retain their defaults.

| Limit | Default | Enforcement |
|---|---:|---|
| Distinct examples across workflow | 64 | Structural preflight and resolved-step checks |
| Input tokens per example | 256 | Exact Toy sequence length or prepared adapter tokenizer |
| Generated tokens per example | 32 | GenerationSpec preflight |
| New artifact bytes across workflow | 256 MiB | Synchronous guarded store writes, direct-write polling, final exact check |
| Worker wall time | 600 seconds | Whole subprocess tree supervision, including imports/builders |
| Aggregate process-tree RSS | 8 GiB | psutil sampling every 20 ms; terminate tree on observed excess |

RSS includes descendant processes and is observed RSS, not device-memory
reservation or a guarantee about unsampled transient peaks. CPU is required;
strict CUDA/MPS profiles fail explicitly. Parent publication follows worker
acceptance and is bounded by the accepted artifact-byte ceiling. This phase
verifies payloads and atomically renames the run directory; it does not execute
models. Logs/cache/temp have a separate fixed 16 MiB allowance above the artifact
budget. Attempt diagnostics retain actual duration, observed peak RSS and bytes.

`--scratch-limits-file limits.json` accepts ScratchLimits fields, for example
`{"max_examples": 32, "timeout_seconds": 120}`. Omitted fields use the defaults.
Increases must appear explicitly here; the full resulting profile is saved in
the ordinary WorkflowRunRecord and request JSON. `--scratch-artifact-root` and
`--scratch-catalog-root` override local roots; each generated root carries a
`*` gitignore rule, including custom roots inside a checkout. Keep executable
workflow sources outside these output roots. Do not commit generated data.

The default artifact directory is `$XENON_HOME/pipelines_v2/artifacts` and the
catalog is the existing `$XENON_HOME/pipelines_v2/catalog` (XENON_HOME defaults to
`~/.xenon`). Artifact files are grouped under run ID and indexed through the same
FileCatalog. `workflow runs` and `workflow show --run-id ...` discover defaults;
use FileCatalog with its explicit root for a custom catalog.

Only local fixture datasets and local artifact refs with known bounded example
coverage are accepted. Deferred datasets, remote refs even if cached, secret
providers, cloud runners and non-Torch real-model engines fail before execution.
A real adapter must expose exact offline `scratch_token_counts(examples)`, a CPU
device and model_id inside `--scratch-model-cache`. Prepare that cache separately
with an immutable revision, for example `hf download <model> --revision <commit>
--local-dir <cache>`. Missing cache is an actionable error, never a download.
Toy fixtures and prepared GPT-2 CPU capture/interventions have been exercised; see the [local recipe](../examples/local_research/phase_00/README.md).

macOS requires sandbox-exec; Linux requires bubblewrap with a working unshared
network namespace. Missing/unsupported OS isolation fails explicitly. The OS
boundary denies network access and writes outside the private attempt, including
native child programs. An inherited Python audit sentinel additionally blocks
sockets and Modal/Postgres/API-client imports. This is an offline research
boundary, not a general security sandbox for hostile workflow code.

A worker failure retains diagnostic logs and private checkpoints, marks the
ordinary run failed and publishes no canonical artifact directory. Successful
outputs receive verified C03 completion, report presentation paths are rebased
and rehashed, and all artifacts become visible in one same-filesystem rename.
C01 scientific identities and tensor bytes survive relocation unchanged.
Catalog writes follow durable publication. If indexing is interrupted, recover
without rerunning model work:

```python
from pipelines_v2.runtime.scratch import LocalScratchProfile
profile = LocalScratchProfile()  # Use the original saved roots and limits.
result = profile.recover("wr_scratch_...")
```

Recovery validates worker acceptance and every canonical C03 payload before
reindexing the existing FileCatalog. It refuses failed supervision, incomplete
compute, changed profile or missing/corrupt payloads. Scratch compute always
starts a new run; the shared resume/reuse options are not silently translated.
