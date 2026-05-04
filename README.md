# Xenon

Xenon is the active `pipelines_v2` platform plus the live DX Terminal
prompt-confusion work that still runs on top of it.

The canonical flow is:

`workflow.py -> workflow run -> artifacts -> local report`

## Operator Surface

Use the `pipelines_v2` CLI:

```bash
uv run python -m pipelines_v2.cli workflow plan --file projects/.../specs/workflow.py
uv run python -m pipelines_v2.cli workflow run --file projects/.../specs/workflow.py --logging INFO
uv run python -m pipelines_v2.cli workflow runs --file projects/.../specs/workflow.py
uv run python -m pipelines_v2.cli workflow show --run-id wr_...
uv run python -m pipelines_v2.cli workflow resume --file projects/.../specs/workflow.py --latest-failed
uv run python -m pipelines_v2.cli workflow rerun-step --file projects/.../specs/workflow.py --run-id wr_... --step report
uv run python -m pipelines_v2.cli workflow rerun-from-step --file projects/.../specs/workflow.py --run-id wr_... --step capture_prompt_eos_residual
```

For real jobs, capture and analysis run on Modal. Reports are built locally
from workflow artifacts.

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

dashboard/
  src/

projects/DX_TERMINAL/
  prompt_confusion/
```

Older runtime code and archived project work were moved out of the live repo
surface. Treat `xenon` as the v2 platform repo.

## Setup

```bash
uv sync
uv sync --extra interp --extra analysis --extra dev --extra dashboard
```

Environment:

```bash
XENON_NEON_DATABASE_URL=postgresql://...
```

## Tests

Run the local suite first:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q tests
```

Modal/vLLM GPU tests are opt-in because they start GPU containers and load real
model weights. Use them when changing `pipelines_v2` vLLM capture, generation,
activation patching, Modal execution, sharding, or workflow batching.

The full Modal contract file uses the shared-session fast path: it loads the
model once per compatible batch, then runs capture, generation, patched
generation, mixed patch operators, and sharded `run_many` coverage against that
loaded engine.

```bash
PYTHONDONTWRITEBYTECODE=1 \
XENON_RUN_MODAL_VLLM_GPU_SMOKE=1 \
XENON_RUN_MODAL_VLLM_ENGINE_CONTRACTS=1 \
XENON_RUN_MODAL_VLLM_PATCH_OPERATOR_CONTRACTS=1 \
XENON_RUN_MODAL_VLLM_PAIRED_PATCH_CONTRACTS=1 \
XENON_MODAL_VLLM_ENGINE_CONTRACT_SHARD_COUNT=2 \
XENON_MODAL_VLLM_ENGINE_CONTRACT_MAX_CONTAINERS=2 \
.venv/bin/pytest -q tests/pipelines_v2/engine/test_modal_vllm_gpu_smoke.py -s
```

For a cheaper GPU sanity check, run only the smoke test:

```bash
PYTHONDONTWRITEBYTECODE=1 \
XENON_RUN_MODAL_VLLM_GPU_SMOKE=1 \
.venv/bin/pytest -q \
  tests/pipelines_v2/engine/test_modal_vllm_gpu_smoke.py::test_modal_vllm_compiled_project_out_smoke \
  -s
```

The Modal tests print app ids while running. If a run hangs or fails inside the
remote container, inspect it with:

```bash
modal app logs <app-id>
```

Keep `-s` enabled for Modal tests so app ids and structured remote progress are
visible in pytest output. Do not enable the GPU flags in ordinary local test
runs or CI jobs without Modal credentials and budget.

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

The comment trigger intentionally ignores forked PRs so Modal secrets are not
exposed to untrusted code. Store `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in
the protected `modal-gpu` GitHub environment, and require environment approval
if a human should explicitly approve GPU spend before the job starts.

## Docs

- [platform/API.md](platform/API.md)
- [platform/WORKFLOW.md](platform/WORKFLOW.md)
- [platform/SPECS.md](platform/SPECS.md)
- [platform/ARCH.md](platform/ARCH.md)

For long-running `pipelines_v2` jobs, prefer:

```bash
uv run python -m pipelines_v2.cli workflow run --file projects/.../specs/workflow.py --logging INFO
```

That prints structured step progress and Modal app ids to stderr while the JSON
result remains on stdout.
