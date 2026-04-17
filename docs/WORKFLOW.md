# Workflow

The canonical Xenon workflow is:

`workflow.py -> workflow run -> artifacts -> local report`

For real jobs, capture and analysis run on Modal. Reports are built locally
from workflow artifacts.

## Operator Surface

Typical commands:

```bash
uv run python -m pipelines_v2.cli workflow plan --file projects/.../specs/workflow.py
uv run python -m pipelines_v2.cli workflow run --file projects/.../specs/workflow.py --logging INFO
uv run python -m pipelines_v2.cli workflow resume --file projects/.../specs/workflow.py --latest-failed
uv run python -m pipelines_v2.cli workflow rerun-step --file projects/.../specs/workflow.py --run-id wr_... --step report
uv run python -m pipelines_v2.cli workflow rerun-from-step --file projects/.../specs/workflow.py --run-id wr_... --step capture_prompt_eos_router
uv run python -m pipelines_v2.cli workflow runs --file projects/.../specs/workflow.py
uv run python -m pipelines_v2.cli workflow show --run-id wr_...
```

Operational notes:

- `--logging INFO` streams structured progress to stderr, including remote
  stages such as Modal launch, app start, and step heartbeats.
- `workflow show --run-id ...` includes the latest persisted run- and
  step-level progress snapshots from the local registry in addition to workflow
  step metadata.
- For builder-backed remote workflows, keep `local_python_sources` minimal and
  explicit. Mounting `"."` into Modal will package the whole repo and can slow
  startup or fail if local generated files change during upload.

Workspace defaults can live in the repo-root [`xenon.toml`](/Users/brockelmore/concordance/xenon/xenon.toml).
That file is git-committable and is the right place for shared defaults such as
the external catalog env var and dashboard static dir. CLI flags still win when
you pass them explicitly, and workflow runner specs still win when they set
their own catalog directly.

## Source Of Truth

The executable source of truth is the checked-in Python workflow file:

```text
projects/<project>/<subproject>/<phase>/specs/workflow.py
```

Optional checked-in snapshots can live alongside it:

```text
projects/<project>/<subproject>/<phase>/specs/workflow.json
```

The usual pattern is:

1. author `workflow.py`
2. `workflow plan`
3. `workflow run`
4. inspect with `workflow runs` and `workflow show`
5. recover with `workflow resume`, `workflow rerun-step`, or `workflow rerun-from-step`

## Canonical Interfaces

- [pipelines_v2/cli.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines_v2/cli.py)
- [pipelines_v2/api.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines_v2/api.py)
- [pipelines_v2/workflow/orchestrator.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines_v2/workflow/orchestrator.py)
