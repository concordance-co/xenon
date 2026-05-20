# Indexing

How to discover what exists without downloading it.

The catalog is the primary index for anything workflow-produced. For data uploaded outside workflows, use naming conventions plus workspace-local helpers. For Modal volumes, use CLI listing.

## The catalog is the index

All workflow runs, workflow steps, and artifacts are recorded in a Postgres catalog that is shared across the team. A local `FileCatalog` mirrors state at `~/.xenon/pipelines_v2/catalog`. `CompositeCatalog` reads from both.

Default behavior is configured in repo-root `xenon.toml`. Don't override with `NullCatalog` unless you mean to be invisible to the team.

### Discovery through the CLI

```bash
# list runs for a specific workflow file
uv run python -m pipelines_v2.cli workflow runs --file path/to/workflow.py

# inspect one run (steps, statuses, artifact ids, progress snapshots)
uv run python -m pipelines_v2.cli workflow show --run-id wr_...
```

This is the default agent-facing discovery path. When an agent wants to know "what has been run for this phase," use these commands.

### Direct catalog queries

For cross-workflow discovery, query the Postgres catalog directly. See `pipelines_v2/storage/postgres.py` for table names and schema. Typical questions:

- recent runs across all workflows
- artifacts of a given kind across a time window
- which workflows have produced a particular artifact kind (probe, direction, etc.)

The catalog holds artifact manifests (which include `workflow_context`, input refs, storage refs), workflow run records, and step records. Everything needed to trace lineage without reading any data.

## Neon tables outside the catalog

Not everything in Neon goes through workflows. Ad-hoc uploads land in plain tables:

- uploaded real datasets (complaint exports, trace exports, benchmark imports)
- seed corpora
- workspace-level reference tables

These are not catalog-indexed. Discovery is via SQL against Neon's information_schema:

```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
```

### Naming convention

Recommended pattern: `<workspace>_<purpose>_v<N>`. A new version produces a new table, never an in-place mutation.

Keep using this pattern for new uploads. If you need something that doesn't fit, propose a convention extension before creating the table.

### Uploaders

Ad-hoc uploads go through repeatable uploader scripts near the workflow or data manifest:

```
workflows/<workflow_name>/scripts/upload_<what>_to_neon.py
```

When you upload a new table, document the table name, purpose, and one-line schema note in the workspace's data inventory.

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

## Workspace-local helpers

Workspace-specific discovery helpers should live near the workflow code:

- `paths.py` — path helpers (export roots, phase-local paths, env-driven overrides)
- `neon.py` — Neon connection and query helpers
- `catalogs.py` — workspace-local catalog surface (if needed beyond the shared catalog)

These are workspace-local by design. If a helper pattern starts repeating across workspaces, it is a candidate for promotion into `pipelines_v2/`.

## Cross-workflow discovery

The catalog is cross-workflow by default. Neon tables and Modal volumes are not indexed by workspace unless the workspace records that metadata.

Each real-data workspace should list:

- active Neon tables (with purpose and version)
- active Modal volume prefixes relevant to the work
- catalog identity if the workflow uses a non-default catalog

This makes the work discoverable to an agent landing on it cold.

## Default agent workflow when orienting to a workflow

1. Read the workflow's README, `PHASE.md`, or equivalent context doc.
2. `workflow runs --file <workflow.py>` to see what's been run.
3. `workflow show --run-id <latest>` for the most recent run's shape.
4. If real data is involved, read the workspace data inventory for Neon table names.
5. Query Neon or the catalog directly for anything else.

Don't discover by listing directories. Don't discover by downloading.
