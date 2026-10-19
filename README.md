# Xenon

Xenon is Concordance's `pipelines_v2` platform for mechanistic
interpretability workflows: capture, analysis, intervention, workflow
orchestration, artifact storage, and local report generation.

The canonical flow is:

`workflow.py -> workflow run -> artifacts -> local report`

Capture and analysis jobs can run on Modal. Reports are built locally from
workflow artifacts.

Start with the [agent guide](AGENTS.md) and [operations guide](operations/README.md).
Use the [LLM guide](platform/LLM_RESEARCH.md) for model and method configuration.
The [research hub map](platform/RESEARCH_HUB.md) covers:
local patching, neuron inspection, official attribution, SAE edits, paired-model
crosscoders, and pretrained transcoder traces. Its [support matrix](platform/research_hub_support.json)
distinguishes demonstrated plumbing from unverified models and research claims.

## Setup

```bash
uv sync
uv sync --extra interp --extra analysis --extra dev --extra modal
```

Optional environment:

```bash
XENON_NEON_DATABASE_URL=postgresql://...
```

## CLI

```bash
uv run python -m pipelines_v2.cli workflow plan --file path/to/workflow.py
uv run python -m pipelines_v2.cli workflow run --file path/to/workflow.py --logging INFO
uv run python -m pipelines_v2.cli workflow runs --file path/to/workflow.py
uv run python -m pipelines_v2.cli workflow show --run-id wr_...
uv run python -m pipelines_v2.cli workflow resume --file path/to/workflow.py --latest-failed
uv run python -m pipelines_v2.cli workflow rerun-step --file path/to/workflow.py --run-id wr_... --step report
uv run python -m pipelines_v2.cli workflow rerun-from-step --file path/to/workflow.py --run-id wr_... --step capture
```

For long-running jobs, prefer `--logging INFO`; structured progress and Modal
app ids are printed to stderr while the JSON result stays on stdout.

## Repo Map

```text
pipelines_v2/
  api.py
  cli.py
  data/
  engine/
  operations/
  reporting/
  runtime/
  storage/
  workflow/

examples/                maintained runnable recipes

platform/
  API.md
  WORKFLOW.md
  SPECS.md
  ARCH.md
  examples/

methodology/
operations/
tests/
```

The package build publishes `pipelines_v2` and the compatibility `papers.voice`
helpers used by downstream repos; their import paths are preserved. New research
studies belong in a separate research repo, using the maintained [examples](examples/README.md).
Operations and platform docs explain how to use and extend the system.
[Methodology](methodology/README.md) offers optional research prompts and illustrative examples.

## Tests

Run the local platform suite:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest -q tests --ignore=tests/test_pipelines_v2_dashboard.py
```

Modal/vLLM GPU tests are opt-in because they start GPU containers and load real
model weights. Use them when changing vLLM capture, generation, activation
patching, Modal execution, sharding, or workflow batching.

```bash
PYTHONDONTWRITEBYTECODE=1 \
XENON_RUN_MODAL_VLLM_GPU_SMOKE=1 \
XENON_RUN_MODAL_VLLM_ENGINE_CONTRACTS=1 \
XENON_RUN_MODAL_VLLM_PATCH_OPERATOR_CONTRACTS=1 \
XENON_RUN_MODAL_VLLM_PAIRED_PATCH_CONTRACTS=1 \
XENON_MODAL_VLLM_ENGINE_CONTRACT_SHARD_COUNT=2 \
XENON_MODAL_VLLM_ENGINE_CONTRACT_MAX_CONTAINERS=2 \
uv run pytest -q tests/pipelines_v2/engine/test_modal_vllm_gpu_smoke.py -s
```

For a cheaper GPU sanity check, run only the smoke test:

```bash
PYTHONDONTWRITEBYTECODE=1 \
XENON_RUN_MODAL_VLLM_GPU_SMOKE=1 \
uv run pytest -q \
  tests/pipelines_v2/engine/test_modal_vllm_gpu_smoke.py::test_modal_vllm_compiled_project_out_smoke \
  -s
```

The Modal tests print app ids while running. If a run hangs or fails inside the
remote container, inspect it with:

```bash
modal app logs <app-id>
```

GitHub can run the Modal suite without making it part of every PR. Use the
`Modal vLLM` workflow from the Actions tab for an explicit run against a ref or
SHA. Available suites are:

- `smoke`: one compiled vLLM GPU sanity test.
- `contracts`: smoke plus the reusable engine contract suite.
- `full`: smoke, engine contracts, patch-operator contracts, paired-patch
  contracts, and sharded `run_many` coverage.

Trusted repo members can also comment on an in-repo PR branch:

```text
/modal-smoke
/modal-contracts
/modal-full
```

Store `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the protected `modal-gpu`
GitHub environment.

## Docs

- [Operations guide](operations/README.md)
- [Configure an LLM experiment](platform/LLM_RESEARCH.md)
- [Capabilities and recipes](platform/RESEARCH_HUB.md)
- [Documentation index](platform/README.md)
- [Deferred work](platform/BACKBURNER.md)

- [platform/API.md](platform/API.md)
- [platform/WORKFLOW.md](platform/WORKFLOW.md)
- [platform/SPECS.md](platform/SPECS.md)
- [platform/ARCH.md](platform/ARCH.md)
