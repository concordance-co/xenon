# Start an LLM experiment

The supported adapters now accept caller-owned datasets and explicit workload budgets. The prepared model families remain GPT-2 (residuals, native heads, pretrained SAE), Qwen3-0.6B (native MLP, PLT tracer), and the Qwen2.5-0.5B base/instruct pair (crosscoder); official MAttr supports GPT-2 and Qwen2.5 IOI-schema inputs. Adding an arbitrary model architecture still requires an adapter.

The four completed implementation areas are:

1. **Configurable workloads:** bring datasets and set fitting, tracing, token, and edit budgets within the supported adapters.
2. **Reusable results:** bind actual inputs/runtime/source, verify completed payloads, and reuse matching work across runs.
3. **Shared LLM artifacts:** publish typed graphs, attribution results, dictionaries, and summaries with explicit transfer budgets and preserved lineage.
4. **Explicit compute:** local CPU and Modal GPU configurations, early asset/resource checks, and bounded real-model verification.

## First session or new thread

1. Read [AGENTS.md](../AGENTS.md), choose a method from the [capability map](RESEARCH_HUB.md), and inspect its recipe. Read [locality](../operations/LOCALITY.md) before handling shared data. For research design, also load the methodology flywheel, principles and checks.
2. Create `workspaces/<study>/<phase>/`. Save the question, dataset/split, model revision, method, budget and success check in `PHASE.md` using the [template](../methodology/templates/PHASE.md). Keep `workflow.py` as executable source of truth. Add an [experiment pack](EXPERIMENT_PACKS.md) when preset/recipe-lock validation is useful.
3. Prepare immutable model/tool assets separately. Confirm the exact runtime and asset paths from the relevant recipe. Do not assume old `/private/tmp` paths or ignored `.planning/` logs exist on a new machine.
4. Plan, then run one small real example through capture/intervention, saved artifact and local report. Check shapes, finite values, example alignment and an obvious effect or no-op control. Expand only after this path works.
5. Save run/artifact IDs, the executed config, outcome and next step in the phase record. Use the catalog for discovery; leave large tensors on their compute store.

For plain local CPU execution, install the exact isolated runtime in the [GPT-2 setup](../workspaces/local_research/phase_00/README.md#explicit-preparation-network-allowed-here-only). For Modal control, use the repo environment with `uv sync --extra modal`; target images install their own locks. Official MAttr and crosscoders have separate preparation instructions. Keep those tool dependencies out of the shared control environment.

## Start with a configuration

Copy [local.example.json](../workspaces/llm_research/local.example.json) or [modal.example.json](../workspaces/llm_research/modal.example.json) into the new phase. Set the prepared checkpoint and artifact/report paths, examples, layers and generation budget. The checked-in [workflow loader](../workspaces/llm_research/workflow.py) reads this file; copy or adapt it into the phase when adding steps.

Use `dataset_file` instead of `examples` for a JSON array of Example records (`key`, `prompt`, optional labels/metadata). Python workflows can pass a `Dataset` directly. Local JSON is for explicit fixtures and scratch data; shared datasets retain Neon/locality conventions.

```bash
# Choose the prepared local runtime, or the Modal control environment's Python.
export XENON_EXPERIMENT_PYTHON=/absolute/path/to/runtime/bin/python
export PYTHONPATH="$PWD"
export XENON_LLM_CONFIG="$PWD/workspaces/my_study/phase_00/config.json"

"$XENON_EXPERIMENT_PYTHON" -m pipelines_v2.cli workflow plan --file workspaces/llm_research/workflow.py
"$XENON_EXPERIMENT_PYTHON" -m pipelines_v2.cli workflow run --file workspaces/llm_research/workflow.py --logging INFO
"$XENON_EXPERIMENT_PYTHON" -m pipelines_v2.cli workflow runs --file workspaces/llm_research/workflow.py
"$XENON_EXPERIMENT_PYTHON" -m pipelines_v2.cli workflow show --run-id wr_YOUR_RUN
```

`backend: local` uses local CPU compute without Modal. This ordinary local runner does not provide OS network isolation or a hard process budget, and CLI catalog defaults remain active. For a strictly offline test with no cloud/catalog access, use [the local-scratch profile](LOCAL_SCRATCH.md) and its prepared-cache and limits flags. For Modal, `workflow readiness --file ...` explicitly runs CPU target checks before the first GPU job. Readiness does not guarantee arbitrary workloads fit.

After a successful shared-profile run, `workflow run --file ... --reuse-completed` reuses compatible completed steps. Use `workflow resume --file ... --latest-failed` for supported failed-run recovery. Scratch whole-run recovery and external-tool optimizer recovery have narrower boundaries; see [durability](DURABILITY.md). Source/input/runtime changes can invalidate reuse legitimately.

## Available operations

| Method | Configurable scope | Fixed scientific contract |
|---|---|---|
| `residual` | examples, layers, residual site, generation length | GPT-2 eager FP32, CPU/CUDA |
| `native_capture` | typed native sites, examples; optional `native_edit` with site/mode/value | GPT-2 heads or Qwen3 MLP; original model recomputes downstream |
| `native_route` | complete serialized `operation`: pairs, sites, IG intervals, token/work budgets | explicit target/foil and control pairs; selected head effects, not exhaustive circuits |
| `sae` | examples, any distinct valid feature edits, prompt/example budgets, optional independent oracle | pinned layer-6 GPT-2 standard ReLU dictionary and verified coordinate map |
| `mattr` | caller-owned IOI-schema train/dev files, fitting steps, evaluation examples, process budgets | pinned official tool/evaluator, model, mask policy, ten-point curve |

SAE config supplies `dictionary: {cache_root, model_path, coordinate_evidence}` and `feature_edits: [[index, amount], ...]`. The splice preserves the original residual and adds centered decoded feature deltas. Reconstruction remains a separate diagnostic arm. `verify_oracle` is optional validation work, not a workload restriction.

MAttr config supplies `tool_paths` for a prepared local interpreter/source checkout and `attribution: {model, model_path, dataset_path, steps, eval_examples, seed, ...budgets}`. The two JSON files must follow the pinned upstream IOI schema. Their bytes become the data identity; custom files cannot masquerade as the official benchmark revision. For Modal, prepare the request locally and supply its serialized `request` with mounted model/dataset paths. Runtime source/environment provisioning occurs during image preparation. No broader task-format compatibility is implied.

Crosscoder uses its own configurable builder in `workspaces/qwen_crosscoder`: datasets, explicit train/heldout keys, dictionary width, steps, token batches, feature edits, exact checkpoint continuation, and CPU/CUDA profiles. It processes model activations sequentially and bounds activation/parameter/output/device allocations. See that workspace's README.

`CircuitTraceSpec` accepts positive feature-node/token/time/memory/output budgets instead of the former fixed two-node/two-token fixture. Each operation traces one prompt; ordinary workflow steps compose multiple prompts. The PLT-to-original-model parity limitation remains explicit and deferred.

## Modal profiles

Change `backend` to `modal`, set the engine's `device` to `cuda`, use mounted `/models/...` paths, and supply:

```json
{
  "artifact_store": {"name": "xenon-data", "root": "/data/my-experiment"},
  "resources": {
    "gpu": "L4", "cpu": 4, "memory_mb": 16384,
    "timeout_seconds": 900, "client_timeout_seconds": 1200,
    "readiness_timeout_seconds": 300,
    "max_containers": 1, "shard_count": 1,
    "volumes": [{"name": "xenon-models", "mount_path": "/models"}]
  }
}
```

`pipelines_v2.research.profiles.llm_modal_profile('small'|'large')` supplies named Python declarations (L4/16GiB or A100-80GB/64GiB). These are starting allocations, not guarantees that arbitrary workloads fit. Planning checks device/allocation consistency and mounted asset paths; readiness verifies assets/runtime and prompt bounds before weights load. External process budgets must fit the runner allocation. Reports run locally; bulk captures stay on Modal. Independent workflow steps may launch separate workers; shared warm sessions remain deferred.

## Identity and sharing

Supported operations bind observed model/dictionary/data/reference bytes, source, actual runtime and numeric policy. Unresolved identities remain ineligible for reuse; arbitrary Python transforms are not automatically certified. CUDA identities include observed GPU/runtime policy, so different GPU models do not claim equivalence. Changing inputs, upstream payload bytes, numeric settings or source invalidates reuse.

See [PROVENANCE.md](PROVENANCE.md) for identities and [PUBLICATION.md](PUBLICATION.md) for typed LLM result/dictionary publication. Publication preserves scientific identity and verifies a new storage completion record. Explicit transfer budgets allow dictionary weights; bulk capture publication remains separate. Small runs validate execution and artifact plumbing, not scientific claims.

For remote-only native checkpoints, `engine.site(..., model_identity=checkpoint_sha256)` constructs a typed site without a local model copy. The worker verifies that digest against the actual mounted checkpoint before execution. Eager Torch/native adapters use two intra-op threads and one inter-op thread; initialize them before unrelated parallel Torch work in a local process.

A catalog admission is evidence for reuse after fresh target checks. If reuse misses, a new worker observes its own runtime and GPU facts; historical admissions do not force a fresh execution to match a previous machine. Explicitly supplied execution admissions remain strict. Generic LLM artifacts persist the worker admission alongside their completion record.

External-tool readiness has its own declared child-process memory/deadline (`readiness_memory_bytes`, `readiness_timeout_seconds`): defaults are 8 GiB and 120 seconds. Modal reserves another 2 GiB for the control process, and planning rejects insufficient worker memory/deadlines. Failed readiness retains bounded diagnostics. MAttr summaries use CPU workers; reports remain local.

## Handoff to the next thread

Copy this into a new chat and fill in the concrete values:

```text
Repo: /absolute/path/to/xenon
Read AGENTS.md and platform/LLM_RESEARCH.md, then my phase's PHASE.md.
Question and desired output: ...
Phase directory / workflow / config: ...
Model revision and prepared local or mounted paths: ...
Runtime interpreter / local CPU or Modal GPU profile: ...
Dataset source, split and current sample size: ...
Budget: examples/tokens/steps, memory, timeout, GPU and worker count.
Latest run/artifact IDs and result: ...
Next action and verification condition: ...
Start with a small real plumbing check. Continue routine implementation
without asking. Keep shared captures on Modal and build reports locally.
Do not pick up platform/BACKBURNER.md unless this experiment requires it.
```

The [platform index](README.md) points to deeper APIs and operational guides. Historical T00–T25 plans are archived; they are not a startup checklist for new experiments.
