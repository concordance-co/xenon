# AGENTS.md

Orientation for agents working in this repo.

## What this is

Xenon is Concordance's mech interp platform. The open-source repo centers on
`pipelines_v2`: workflow specs, runners, storage, operation specs, vLLM
integration, reporting, and deployment runtime helpers.

## Orientation

Three reference trees. Load the relevant one for the work.

- **`methodology/`** — research flywheel, principles, checks, hypothesis catalog spec, method roster, templates.
- **`operations/`** — data locality, indexing, and report-building conventions.
- **`platform/`** — `pipelines_v2` API, workflow authoring, specs, architecture, and examples.

Archive directories (`*/archive/`) are historical. Treat them as effectively
deleted unless explicitly digging through old work.

## Agent Protocol

For research-method work, load `methodology/` by default. The flywheel,
principles, and checks are part of the expected reasoning context.

For operational work touching data, captures, or reports, load `operations/` by
default. The rules in `LOCALITY.md` are enforced in code and will fail if
ignored.

For workflow authoring, load `platform/`.

Decision points that trigger `methodology/CHECKS.md`:

- picking a measurement locus
- promoting a claim up the evidence ladder
- committing to a synth design
- scaling synth generation
- crossing into real data
- designing an intervention
- closing a phase

Each research phase should keep a canonical `PHASE.md` per
`methodology/templates/PHASE.md`. It starts as the phase's
premise/design/orientation surface, stays current as work runs, and becomes the
closure artifact when the phase ends.

## Canonical Execution Model

```text
workflow.py -> workflow run -> artifacts -> local report
```

Capture and analysis run on Modal when the workflow uses Modal runners. Reports
build locally. See `operations/LOCALITY.md`.

## CLI Cheat Sheet

```bash
# plan a workflow
uv run python -m pipelines_v2.cli workflow plan --file <workflow.py>

# run with structured progress
uv run python -m pipelines_v2.cli workflow run --file <workflow.py> --logging INFO

# discover runs for a workflow
uv run python -m pipelines_v2.cli workflow runs --file <workflow.py>

# inspect one run
uv run python -m pipelines_v2.cli workflow show --run-id wr_...

# recover a failed run
uv run python -m pipelines_v2.cli workflow resume --file <workflow.py> --latest-failed

# rerun a step (new run, reuses upstream artifacts)
uv run python -m pipelines_v2.cli workflow rerun-step --file <workflow.py> --run-id wr_... --step <name>

# rerun a step and all downstream dependents
uv run python -m pipelines_v2.cli workflow rerun-from-step --file <workflow.py> --run-id wr_... --step <name>
```

Full CLI reference: `platform/API.md`.

## Repo Map

```text
methodology/             research substrate
operations/              data locality, indexing, reporting
platform/                pipelines_v2 reference and examples
pipelines_v2/            active platform code
tests/                   platform tests
```

Dashboard and paper workspaces are outside the public package surface for this
release.

## Conventions

- Checked-in `workflow.py` is the executable source of truth for a workflow.
- `workflow.json` snapshots are optional, for reviewability.
- When using `PromptMetadataBuilder.from_function(...)`, pass narrow
  `local_python_sources` like `("pipelines_v2", "scripts")`. Do not rely on
  `"."`.
- Helpers that repeat across downstream workspaces should be promoted into
  `pipelines_v2/`; keep one-off workflow code next to the workflow file.

## Workspace Defaults

Defaults live in repo-root `xenon.toml`:

- `[pipelines_v2.modal]` — Modal volume and vLLM cache defaults
- catalog — local `FileCatalog` at `~/.xenon/pipelines_v2/catalog` is always on; external Postgres catalog defaults to the shared team catalog

Don't override the catalog with `NullCatalog` unless you mean to be invisible
to the team.
