# Find runs and artifacts

Start with the catalog. It records runs, steps, artifact IDs, and storage refs.
Use storage inspection to diagnose a known path, not to infer the full run history.

## Catalog selection

Normal CLI execution adds a local `FileCatalog` mirror.
Its default root is `$XENON_HOME/pipelines_v2/catalog`, usually `~/.xenon/pipelines_v2/catalog`.
`--local-catalog-root` or workspace config can select a different root.

The repo's `xenon.toml` names `XENON_NEON_DATABASE_URL` for the shared Postgres catalog.
The CLI uses this default when the environment variable is available.
An explicit runner catalog takes precedence over this default.
Runners in one workflow must resolve to a common catalog identity.

An explicit local catalog can keep a workflow out of the team catalog.
`NullCatalog` also behaves differently in a direct runner and through the CLI:
the CLI attaches its local mirror and can apply the shared default.
Inspect the effective setup rather than using `NullCatalog` as an offline switch.
Use `local-scratch` for enforced offline execution.

## Discover and inspect

Use the selected runtime's Python. These examples use `python` as shorthand.

```bash
python -m pipelines_v2.cli workflow runs --file /absolute/path/workflow.py
python -m pipelines_v2.cli workflow runs --status failed --limit 10
python -m pipelines_v2.cli workflow show --run-id wr_YOUR_RUN
```

`runs --file` imports the workflow and uses its name and runner catalog settings.
Keep its builders free of unrelated side effects. Without `--file`, discovery uses workspace defaults.
`show` returns run data, step records, and local progress snapshots.

For an alternate local catalog:

```bash
python -m pipelines_v2.cli workflow show --run-id wr_YOUR_RUN --local-catalog-root /absolute/path/catalog
```

Use the same workspace config and catalog roots as the original run.
An unknown ID can mean the wrong catalog, not a missing output.

## Read selected results

The evidence CLI defaults to the local mirror. It does not silently query Postgres.

```bash
python -m pipelines_v2.evidence runs --study steering --limit 10
python -m pipelines_v2.evidence show wr_YOUR_RUN
python -m pipelines_v2.evidence artifact ARTIFACT_ID --result
python -m pipelines_v2.evidence --catalog-root /absolute/path/catalog show wr_YOUR_RUN
```

Place `--catalog-root` before the evidence subcommand.
The reader limits metadata scans and result sizes. It reports missing or oversized inputs.
A truncated result is not a complete inventory. Remote result reads need access to their store.
See [evidence](../platform/EVIDENCE.md) for Python queries and comparison rules.

## Data outside workflow catalogs

An independently uploaded Neon table is not automatically a workflow artifact.
Keep its table name, purpose, version, and selection with the study config.
Query `information_schema.tables` when the table name is unknown.
`<study>_<purpose>_v<N>` is a useful naming convention, not a code requirement.

For a known Modal path, inspect its volume without downloading the contents:

```bash
modal volume ls xenon-data PATH_PREFIX
```

Use the artifact manifest to resolve its volume and path.
Do not substitute a directory listing for completion or lineage checks.
For custom queries, see [catalog implementations](../pipelines_v2/storage/).
