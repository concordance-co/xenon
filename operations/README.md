# Operations guide

Use this guide to configure, execute, inspect, and recover Xenon workflows.
Use the requested outcome and budget to choose the workload. Example sizes are not limits.

## How the parts fit

```text
workflow.py → plan → run → artifacts + catalog records → local report
```

- A **workflow** defines steps and their dependencies.
- A **spec** defines a capture, analysis, intervention, or report operation.
- A **runner** executes a step locally or on Modal.
- An **artifact** stores an output, its manifest, and its execution identity.
- A **catalog** indexes runs, steps, and artifacts. It does not store the bulk tensors.
- A **report** presents selected artifact results. It does not repeat model execution.

The workflow defines `build_dataset()` and `build_workflow(...)`.
It can define `build_runner_specs()` to select compute, stores, and catalogs.
Use `StepRef` to connect steps. See the [API](../platform/API.md) for signatures.

## Choose the entry point

| Need | Start with |
| --- | --- |
| Select a method and supported model | [Capability map](../platform/RESEARCH_HUB.md) |
| Configure residual edits, native sites, SAE edits, or MAttr | [LLM guide](../platform/LLM_RESEARCH.md) |
| Fit or use a crosscoder | [Crosscoder recipe](../examples/qwen_crosscoder/README.md) |
| Trace a pretrained transcoder | [Tracing recipe](../examples/circuit_trace_smoke/README.md) |
| Work with images | [Recognition](../examples/recognition/phase_00/README.md), [generation](../examples/sdxl_steering/README.md) |
| Extend an operation or adapter | [Architecture](../platform/ARCH.md), [spec conventions](../platform/SPECS.md) |

Check the supported model, revision, runtime, and operation before adapting a recipe.
A new configuration does not add support for an arbitrary architecture.
Keep actual study files in the consuming research repo.

## Prepare the environment

Run setup from the Xenon checkout. For CLI, Modal control, report charts, and tests:

```bash
uv sync --extra modal --extra analysis --extra dev
uv run python -m pipelines_v2.cli workflow --help
```

Use the method's preparation instructions for local model execution.
Some methods need a separate Python environment with exact package versions.
The control environment above does not replace those environments.
Prepare model weights and external tools before execution. Use pinned revisions.
Modal requires account credentials and access to the selected volumes and secrets.

`xenon.toml` supplies workspace defaults. Runner specs supply workflow settings.
The CLI loads a nearby `.env` without replacing variables already set in the environment.
Keep credentials out of source files and logs.

| Setting | Effect |
| --- | --- |
| `XENON_WORKSPACE_ROOT` | Selects the workspace root for config and source mounts. |
| `XENON_HOME` | Selects local state. Default: `~/.xenon`. |
| `XENON_NEON_DATABASE_URL` | Enables the configured shared Postgres catalog when available. |
| `XENON_LLM_CONFIG` | Selects the JSON config for the generic LLM recipe. Other recipes can use different variables. |

See [catalog selection](INDEXING.md#catalog-selection) before relying on shared discovery.
An explicit runner catalog can replace the shared default.

## Select compute and storage

| Mode | Compute | Storage and isolation |
| --- | --- | --- |
| Ordinary local | Local runner; the generic LLM profile uses CPU. | Local artifacts. Normal catalog defaults apply. No OS network isolation or general hard resource cap. |
| Modal | Remote CPU or GPU runners. | Mounted volumes for bulk artifacts and models. Reports run locally. |
| `local-scratch` | Supported local CPU operations. | Local store and catalog. Enforced limits and OS network isolation. Prepared offline assets only. |

The CLI's default `--profile shared` does not mean GPU execution.
The runner specs select compute. A local run can still contact the shared catalog.
Use `local-scratch` when the task requires enforced offline execution.
See [storage rules](LOCALITY.md) and [scratch limits](../platform/LOCAL_SCRATCH.md).

For Modal, set the GPU, CPU, `memory_mb`, timeout, and worker count in the runner config.
`memory_mb` is host RAM, not GPU memory. Set checkpoint paths to paths visible on the worker.
A model path such as `/models/...` needs the corresponding volume mount.
Use the method's resource profile as a starting point; check it against the actual workload.

## Plan and execute

In the commands below, replace the absolute paths with the selected interpreter and workflow.
Run from the Xenon checkout, or use the sibling-repo procedure below.

```bash
export XENON_PYTHON=/absolute/path/to/runtime/bin/python
export XENON_WORKFLOW=/absolute/path/to/research-repo/workflow.py

"$XENON_PYTHON" -m pipelines_v2.cli workflow plan --file "$XENON_WORKFLOW"
"$XENON_PYTHON" -m pipelines_v2.cli workflow run --file "$XENON_WORKFLOW" --logging INFO
```

`plan` checks declared capabilities and configuration without model execution.
Read its errors, warnings, effective config, and workload limits.
Workflow imports and builders still execute; they can read data or perform user-defined work.
Keep builders free of compute launches and unrelated writes.

When runtime or mounted assets need a target check, run readiness:

```bash
"$XENON_PYTHON" -m pipelines_v2.cli workflow readiness --file "$XENON_WORKFLOW"
```

Readiness can launch Modal CPU workers. It does not load model weights.
Steps with unresolved upstream inputs are reported as unresolved.
The command returns nonzero unless every step is ready. Inspect each reason.
An unresolved dependency is not the same as a broken runtime or missing asset.
Readiness does not prove that model execution fits in GPU memory.
See [admission](../platform/ADMISSION.md) for the checks and their limits.

Choose further checks for the task's risk. Use an actual model run to check a new execution path.
Use relevant prior results for an unchanged path. Continue to the requested workload after checks pass.
A successful plumbing check is not the completion of a larger research task.

`--logging INFO` writes progress to stderr. The command writes its JSON result to stdout.
Keep the run ID and any Modal app IDs. Use them to inspect the run:

```bash
"$XENON_PYTHON" -m pipelines_v2.cli workflow runs --file "$XENON_WORKFLOW"
"$XENON_PYTHON" -m pipelines_v2.cli workflow show --run-id wr_YOUR_RUN
```

For offline scratch, use `workflow run --profile local-scratch` with the method's
prepared-cache and limit flags. Do not add reuse or resume flags to scratch execution.

## Reuse or recover work

Inspect the failed step and its diagnostics before choosing a recovery action.

| Intent | Command |
| --- | --- |
| Reuse compatible completed work | `workflow run --file "$XENON_WORKFLOW" --reuse-completed` |
| Resume a failed run | `workflow resume --file "$XENON_WORKFLOW" --run-id wr_YOUR_RUN` |
| Select the latest failed run | `workflow resume --file "$XENON_WORKFLOW" --latest-failed` |
| Rerun one step in a new run | `workflow rerun-step --file "$XENON_WORKFLOW" --run-id wr_YOUR_RUN --step STEP_NAME` |
| Rerun a step and its dependents | `workflow rerun-from-step --file "$XENON_WORKFLOW" --run-id wr_YOUR_RUN --step STEP_NAME` |

Prefix these commands with `"$XENON_PYTHON" -m pipelines_v2.cli`.
Reuse requires compatible execution identity and verified completed payloads.
Changed source, inputs, runtime, or numeric settings can prevent reuse.
Do not edit hashes or manifests to force a match.

Resume does not guarantee recovery from any GPU crash or partial checkpoint.
Scratch publication recovery and crosscoder checkpoint continuation have separate procedures.
See [durability](../platform/DURABILITY.md), [scratch](../platform/LOCAL_SCRATCH.md),
and [crosscoders](../examples/qwen_crosscoder/README.md).

## Inspect, report, and share

Use [discovery](INDEXING.md) to find run and artifact IDs.
Use [reporting](REPORTING.md) to rebuild a report or view selected results.
Use [publication](../platform/PUBLICATION.md) to copy a supported artifact into shared storage.
Publication is explicit. A local run does not upload its outputs automatically.

Keep the executed config, run IDs, result, and next action with the study.
Use enough detail to continue work without repeating discovery. A fixed report template is optional.

## Use a workflow from another repo

Set the Xenon root explicitly. This selects its `xenon.toml` and source mount base.
Add the research repo to Python's import path if the workflow imports its modules.

```bash
export XENON_WORKSPACE_ROOT=/absolute/path/to/xenon
export PYTHONPATH="/absolute/path/to/research-repo:$XENON_WORKSPACE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export XENON_WORKFLOW=/absolute/path/to/research-repo/workflow.py
"$XENON_PYTHON" -m pipelines_v2.cli workflow plan --file "$XENON_WORKFLOW"
```

`PYTHONPATH` controls local imports. It does not upload source files to Modal.
Declare all worker-side source directories in `local_python_sources`.
The resolver accepts directories inside the selected workspace or detected library root.
An absolute path does not bypass that restriction.

If the worker needs study helpers outside Xenon, use an editable Xenon install.
Run from the research repo without `XENON_WORKSPACE_ROOT`.
Keep its runner settings and `xenon.toml` there.
The source resolver can then use the research root and the editable Xenon source root.
Declare both the required study packages and `pipelines_v2`.
Check that each source resolves to the intended directory before a remote run.

Keep those directories narrow. Do not mount the whole repo with `"."`.
See [workspace mounts](../platform/WORKFLOW.md#workspace-root-and-modal-source-mounts).

## Resolve common failures

| Symptom | Check first |
| --- | --- |
| `local dir ... does not exist` | Workspace root and each declared source directory. |
| Missing checkpoint or dictionary | Preparation step, pinned revision, worker path, and volume mount. |
| Runtime or tool mismatch | The method's lock and selected Python executable. Do not upgrade packages blindly. |
| Out of memory | GPU memory versus host RAM, sequence length, batch size, capture sites, and worker count. |
| Unknown run or artifact | Original catalog root, workspace config, and shared catalog access. |
| Source lock mismatch | Recipe edits and pack metadata. Update locks deliberately for changed source. |
| Reuse rejected | Observed identity and completion diagnostics. See [provenance](../platform/PROVENANCE.md). |
| No report file | Report step, `ReportSpec.output_dir`, and selected artifact. See [reporting](REPORTING.md). |
| Failed or stalled Modal step | Stored progress and `modal app logs APP_ID`. |

## Code references

Use these when the guide does not resolve a behavior:

- [CLI](../pipelines_v2/cli.py): command options, catalog selection, and recovery routing.
- [Workflow orchestrator](../pipelines_v2/workflow/orchestrator.py): dependency execution and reuse.
- [Config](../pipelines_v2/core/config.py), [paths](../pipelines_v2/core/paths.py): workspace defaults and local state.
- [Runtime](../pipelines_v2/runtime/): local, Modal, readiness, and scratch implementations.
- [Storage](../pipelines_v2/storage/): artifacts, catalogs, completion, and publication.
