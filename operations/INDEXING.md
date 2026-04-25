# Indexing

How to discover what exists without downloading it.

The catalog is the primary index for anything workflow-produced. For data uploaded outside workflows, naming conventions plus project-local helpers. For Modal volumes, CLI listing.

## The catalog is the index

All workflow runs, workflow steps, and artifacts are recorded in a Postgres catalog that is shared across the team. A local `FileCatalog` mirrors state at `~/.xenon/pipelines_v2/catalog`. `CompositeCatalog` reads from both.

Default behavior is configured in repo-root `xenon.toml`. Don't override with `NullCatalog` unless you mean to be invisible to the team.

### Discovery through the CLI

```bash
# list runs for a specific workflow file
uv run python -m pipelines_v2.cli workflow runs --file projects/<project>/<sub>/<phase>/specs/workflow.py

# inspect one run (steps, statuses, artifact ids, progress snapshots)
uv run python -m pipelines_v2.cli workflow show --run-id wr_...
```

This is the default agent-facing discovery path. When an agent wants to know "what has been run for this phase," use these commands.

### Direct catalog queries

For cross-workflow or cross-project discovery, query the Postgres catalog directly. See `pipelines_v2/storage/postgres.py` for table names and schema. Typical questions:

- recent runs across all workflows
- artifacts of a given kind across a time window
- which workflows have produced a particular artifact kind (probe, direction, etc.)

The catalog holds artifact manifests (which include `workflow_context`, input refs, storage refs), workflow run records, and step records. Everything needed to trace lineage without reading any data.

## Neon tables outside the catalog

Not everything in Neon goes through workflows. Ad-hoc uploads land in plain tables:

- uploaded real datasets (complaint exports, trace exports, benchmark imports)
- seed corpora
- project-level reference tables

These are not catalog-indexed. Discovery is via SQL against Neon's information_schema:

```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
```

### Naming convention

Current observed pattern: `<project>_<purpose>_v<N>`. Example: `dx_terminal_complaint_dataset_enriched_v1`. A new version produces a new table, never an in-place mutation.

Keep using this pattern for new uploads. If you need something that doesn't fit, propose a convention extension before creating the table.

### Uploaders

Ad-hoc uploads go through repeatable uploader scripts at the project level:

```
projects/<project>/<sub>/<phase>/scripts/upload_<what>_to_neon.py
```

Example: `projects/DX_TERMINAL/prompt_confusion/phase_12/scripts/upload_complaint_dataset_to_neon.py`.

When you upload a new table, document it in the project's `REAL_DATA.md` — table name, purpose, one-line schema note.

## Modal volumes

Two volumes of relevance:

- `xenon-data` — artifacts produced by workflows
- `xenon-models` — model weights and vLLM compile cache

### Listing without mounting

```bash
modal volume ls xenon-data <path_prefix>
```

This is the default agent-facing path for "is this artifact present on the volume?" Don't mount the volume to check.

### Reading artifacts

Programmatic access goes through `ModalVolumeStore`. Use the catalog to get an `artifact_id`, then use the typed artifact handle (`CaptureArtifact`, `OperationArtifact`) to read refs. Don't shell into the volume to grep for files.

## Project-local helpers

Project-specific discovery helpers live under the project:

- `projects/<project>/<sub>/paths.py` — path helpers (export roots, phase-local paths, env-driven overrides)
- `projects/<project>/<sub>/neon.py` — Neon connection and query helpers
- `projects/<project>/<sub>/catalogs.py` — project-local catalog surface (if needed beyond the shared catalog)

These are project-local by design for now. If a helper pattern starts repeating across projects, it's a candidate for promotion into `pipelines_v2/` — but default to project-local until the pattern is repeated.

Open question: some of what's currently in `prompt_confusion/paths.py` (dataset export roots, registry paths) may generalize. Worth revisiting during the next infrastructure pass.

## Cross-project discovery

The catalog is cross-project by default. Neon tables and Modal volumes are not indexed cross-project.

Each project's `REAL_DATA.md` should list:

- active Neon tables (with purpose and version)
- active Modal volume prefixes relevant to the project
- catalog identity if the project uses a non-default catalog

This makes the project discoverable to an agent landing on it cold.

## Default agent workflow when orienting to a phase

1. Read the phase's `PHASE.md` and surrounding `project/subproject/README.md`.
2. `workflow runs --file <phase workflow.py>` to see what's been run.
3. `workflow show --run-id <latest>` for the most recent run's shape.
4. If real data is involved, read the project's `REAL_DATA.md` for Neon table names.
5. Query Neon or the catalog directly for anything else.

Don't discover by listing directories. Don't discover by downloading.
