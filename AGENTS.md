# AGENTS.md

Orientation for agents working in this repo.

## What this is

Xenon is Concordance's mech interp platform: `pipelines_v2` infrastructure plus the projects that live on top of it. It operationalizes mech interp against real agent data and evals.

## Orientation

Three reference trees. Load the relevant one for the work.

- **`methodology/`** — how to do the research. Flywheel, principles, checks, hypothesis catalog spec, method roster, templates.
- **`operations/`** — where data lives, how to discover it, how reports are built. Locality, indexing, reporting.
- **`platform/`** — `pipelines_v2` API, workflow authoring, specs, architecture.

Project work lives under `projects/<project>/`. Phases live directly under the project, or grouped into subprojects when a project has multiple research questions. `REAL_DATA.md` is added when stage-1 work begins.

Archive directories (`*/archive/`) are historical. Treat as effectively deleted unless explicitly digging through old work.

## Agent protocol

For any research work, load `methodology/` by default. Agents that don't carry the flywheel, principles, and checks produce incompetent mech interp.

For any operational work — touching data, authoring captures, building reports — load `operations/` by default. The rules in `LOCALITY.md` are enforced in code and will fail if ignored.

For workflow authoring, load `platform/`.

Decision points that trigger `methodology/CHECKS.md`:

- picking a measurement locus
- promoting a claim up the evidence ladder
- committing to a synth design
- crossing into real data
- designing an intervention
- closing a phase

Each phase has a canonical `PHASE.md` per `methodology/templates/PHASE.md`. It starts as the phase's premise/design/orientation surface, stays current as work runs, and becomes the closure artifact when the phase ends. The next phase's premise inherits from the closing phase's open threads.

## Canonical execution model

```
workflow.py → workflow run → artifacts → local report
```

Capture and analysis run on Modal. Reports build locally. See `operations/LOCALITY.md`.

Workflow files live at:

```
projects/<project>/[<subproject>/]<phase>/specs/workflow.py
```

## CLI cheat sheet

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

## Repo map

```
methodology/             research substrate
  FLYWHEEL.md            how a subproject moves from question to finding
  PRINCIPLES.md          always-true research values
  CHECKS.md              decision-point triggers
  HYPOTHESES.md          per-subproject hypothesis catalog spec
  ROSTER.md              method catalog (29 families)
  templates/
    PHASE.md             canonical phase orientation/design/closure shape
    REAL_DATA.md         project real-data living doc shape
  archive/               historical methodology references

operations/              data locality, indexing, reporting
  LOCALITY.md            where data lives, what's allowed local
  INDEXING.md            how to discover what exists
  REPORTING.md           the local report build flow

platform/                pipelines_v2 reference
  API.md                 public API and spec surface
  WORKFLOW.md            workflow authoring and CLI
  SPECS.md               checked-in spec conventions
  ARCH.md                architecture notes
  patching_best_practices.md
  examples/

pipelines_v2/            active platform code
dashboard/               dashboard code
tests/                   platform tests

projects/
  <project>/
    [REAL_DATA.md]       optional — living real-data context doc
    [shared/]            optional — project-local code
    phase_XX/
      PHASE.md           canonical phase orientation/design/closure artifact
      specs/
        workflow.py      executable source of truth for the phase
      docs/              phase-internal artifacts
      reports/           local report sources and assets
    [<subproject>/]      optional — when a project has multiple research questions
      phase_XX/
        ... same shape
```

## Starting new work

When the user opens with a concrete task, follow `methodology/FLYWHEEL.md`'s "Starting a new project" flow: read methodology + operations, scan any provided source, scaffold `projects/<name>/phase_00/` per the layout there, report the picture back. Phase 0 is the framing pass before stage 1.

Existing projects (`MOREBENCH/`, `COUNSELBENCH/`, `DX_TERMINAL/`) predate the current methodology and templates. Don't pattern-match new work against them — build against the docs. Reference old projects only when explicitly asked.

**New phase.** A new phase begins when a loopback is triggered or the subproject commits to a meaningfully new direction. See `methodology/FLYWHEEL.md`. Create the phase directory with `specs/workflow.py` and a `PHASE.md` initialized from `methodology/templates/PHASE.md`. Keep `PHASE.md` current as the phase runs; finalize the same file when the direction shifts again.

## Conventions

- Checked-in `workflow.py` is the executable source of truth for every phase.
- `workflow.json` snapshots are optional, for reviewability.
- When using `PromptMetadataBuilder.from_function(...)`, pass narrow `local_python_sources` like `("pipelines_v2", "scripts")`. Do not rely on `"."`.
- Project-specific helpers (`paths.py`, `neon.py`, `catalogs.py`) live under the project. They are project-local by design. If a pattern repeats across projects, propose promotion to `pipelines_v2/`.

## Workspace defaults

Defaults live in repo-root `xenon.toml`:

- `[pipelines_v2.modal]` — Modal volume and vLLM cache defaults
- catalog — local `FileCatalog` at `~/.xenon/pipelines_v2/catalog` is always on; external Postgres catalog defaults to the shared team catalog

Don't override the catalog with `NullCatalog` unless you mean to be invisible to the team.
