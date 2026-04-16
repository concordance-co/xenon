# Xenon

Xenon is the active `pipelines_v2` platform plus the live DX Terminal
prompt-confusion work that still runs on top of it.

The canonical flow is:

`workflow.py -> workflow run -> artifacts -> local report`

## Operator Surface

Use the `pipelines_v2` CLI:

```bash
uv run python -m pipelines_v2.cli workflow plan --file projects/.../specs/workflow.py
uv run python -m pipelines_v2.cli workflow run --file projects/.../specs/workflow.py
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

## Docs

- [docs/PIPELINES_V2_API.md](/Users/marshallvyletel/repos/concordance/xenon/docs/PIPELINES_V2_API.md)
- [docs/WORKFLOW.md](/Users/marshallvyletel/repos/concordance/xenon/docs/WORKFLOW.md)
- [docs/SPECS.md](/Users/marshallvyletel/repos/concordance/xenon/docs/SPECS.md)
- [docs/ARCH2.md](/Users/marshallvyletel/repos/concordance/xenon/docs/ARCH2.md)
