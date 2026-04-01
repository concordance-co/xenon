# Next Steps

This file is the current handoff note for the Xenon cleanup and workflow rebuild.

## Current State

The canonical workflow is now:

`workflow spec -> published Neon dataset relation -> capture run -> analysis run -> report`

The canonical operator surface is:

```bash
uv run -m pipelines.cli ...
```

Implemented:

- `workflow_specs`, `workflow_runs`, `dataset_publications` in Neon schema
- workflow registry and publication logic in [pipelines/workflows.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/workflows.py)
- CLI support in [pipelines/cli.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/cli.py) for:
  - `spec create|update|show|list|delete`
  - `dataset build`
  - `capture run`
  - `analysis run`
  - `report build`
  - `run show|list`
  - `publication list`
- capture can now read from a published Neon relation via `--source-relation`
- analysis can now use exported `workflow_label` parquet labels
- generic workflow report builder in [pipelines/reporting.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/reporting.py)
- major research code moved out of `pipelines/interp/` into `research/`
- legacy `prep-targets` API removed from the backend
- README/docs now point at the CLI-first workflow

## Non-Negotiable Architecture Point

Modal is not being replaced.

Brock's existing Modal-backed capture and analysis machinery remains the intended
execution plane for heavy jobs. The cleanup goal is:

- preserve the strong/flexible Modal worker path
- preserve the fast execution behavior Brock built
- simplify the control plane on top of it

So the intended architecture is:

- `workflow specs + pipelines.cli` = control plane
- existing Modal workers/runtime = execution plane

The remaining work is to make the CLI/spec flow drive that existing Modal layer
cleanly, not to replace it.

Latest test status:

```bash
uv run --extra interp --extra analysis --extra dev -m pytest tests -q
```

Result:

- `322 passed`

## Explicitly Legacy

These are no longer the recommended operator path:

- [pipelines/backend/](/Users/marshallvyletel/repos/concordance/xenon/pipelines/backend/)
- [pipelines/dashboard-ui/](/Users/marshallvyletel/repos/concordance/xenon/pipelines/dashboard-ui/)
- [pipelines/dashboard.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/dashboard.py)
- [scripts/xenon_backend.sh](/Users/marshallvyletel/repos/concordance/xenon/scripts/xenon_backend.sh)

They still exist, but the repo should be treated as CLI-first.

## Remaining Work

### High Priority

1. Real Modal workflow validation

We have strong local test coverage, but the important next check is the actual remote workflow:

- create or update a real workflow spec in Neon
- publish a dataset relation
- run Modal capture against that publication
- run analysis against the resulting capture
- build a report

This is the main unresolved validation step.

2. Capture defaults for real remote runs

The CLI wiring is in place, but real runs should verify:

- model id handling
- Modal/local path expectations
- output directory conventions
- layer parsing and pooling behavior
- router/residual flags on the real Modal path

3. Finish CLI -> Modal orchestration

This is the most important remaining architectural task.

What needs to happen:

- `pipelines.cli capture run` should cleanly drive the existing Modal capture path
- heavy analysis paths that should remain Modal-backed should also be reachable from the CLI
- run metadata should still be recorded in `workflow_runs`
- outputs should still land in the expected locations for downstream analysis/reporting

Important clarification:

- this is not a replacement of Modal
- this is not a rewrite of Brock's execution layer
- this is finishing the connection between the new CLI/spec control plane and the existing Modal execution plane

4. Report quality

`report build` is now real, but generic. It should be evaluated for:

- whether the default Typst output is sufficient for operator use
- whether workflow-specific summary fields should be expanded
- whether to include charts or richer result summaries later

### Medium Priority

5. Backend/UI deletion pass

If we are committed to CLI-only:

- delete [pipelines/backend/](/Users/marshallvyletel/repos/concordance/xenon/pipelines/backend/)
- delete [pipelines/dashboard-ui/](/Users/marshallvyletel/repos/concordance/xenon/pipelines/dashboard-ui/)
- delete [pipelines/dashboard.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/dashboard.py)
- delete [scripts/xenon_backend.sh](/Users/marshallvyletel/repos/concordance/xenon/scripts/xenon_backend.sh)

This is repo-pruning, not architectural work.

6. Old wrapper removal

There are still many compatibility shim modules under `pipelines/interp/`.

Those should remain until:

- Modal workflows are validated
- scripts/tests no longer need the old imports

After that, remove them in a cleanup pass.

7. Legacy docs pruning

There are still historical plans and report-era docs in the repo that are no longer onboarding-critical. Those can be reduced further later.

### Lower Priority

8. Spec ergonomics

Potential follow-ups:

- add a checked-in example spec under `research/<effort>/specs/`
- add a `spec validate` command
- add helper templates for common spec patterns

9. Publication management

Potential follow-ups:

- publication cleanup/drop command
- explicit refresh/rebuild semantics
- stronger versioning discipline for publications

## Tomorrow Morning: Real Modal Test Checklist

### 1. Create a real spec file

Put it under:

```text
research/<effort>/specs/workflow.json
```

At minimum include:

- `name`
- `dataset.source`
- `dataset.label`
- `capture`
- `analysis`
- `report`

### 2. Register the spec

```bash
uv run -m pipelines.cli spec create --file research/<effort>/specs/workflow.json
uv run -m pipelines.cli spec list
```

Record the returned `spec_id`.

### 3. Publish the dataset relation

```bash
uv run -m pipelines.cli dataset build --spec <spec_id>
uv run -m pipelines.cli publication list --spec <spec_id>
```

Check:

- publication exists
- relation name looks correct
- row count is plausible

### 4. Run capture

Use the canonical CLI path first. If Modal-specific glue is needed, use it only as an implementation detail.

Example:

```bash
uv run -m pipelines.cli capture run \
  --spec <spec_id> \
  --output-dir data/activations/<run_name>
```

Check:

- run recorded in `workflow_runs`
- activations landed where expected
- `metadata.parquet` exists

### 5. Run analysis

```bash
uv run -m pipelines.cli analysis run \
  --capture-run <capture_run_id> \
  --output-dir data/analysis_results/<run_name>
```

Check:

- label parquet export exists
- analysis result files are written
- `workflow_runs` entry is marked succeeded

### 6. Build report

```bash
uv run -m pipelines.cli report build --analysis-run <analysis_run_id>
```

Check:

- `summary.json` exists
- `report.typ` exists
- `report.pdf` exists if `typst` is installed

### 7. Record any breakpoints

If anything fails, capture:

- exact CLI command
- returned `spec_id` / `run_id`
- whether failure happened in dataset, capture, analysis, or report stage
- whether failure is local-only or Modal-only

## Suggested First Follow-Up After Tomorrow

If tomorrow’s Modal run succeeds:

- remove more legacy wrappers
- prune backend/UI code
- add one or two example effort specs to `research/`

If tomorrow’s Modal run fails:

- fix the concrete execution-path issue first
- preserve and reuse the existing Modal worker path while fixing it
- do not spend time on more repo pruning until the real workflow is stable
