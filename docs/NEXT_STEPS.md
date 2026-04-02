# Next Steps

This is the short handoff note for the cleanup branch.

## Current State

- Canonical flow: `spec -> published Neon dataset relation -> capture run -> analysis run -> report`
- Canonical interface: `uv run -m pipelines.cli ...`
- Specs/runs/publications are wired through:
  - [pipelines/workflows.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/workflows.py)
  - [pipelines/cli.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/cli.py)
  - [pipelines/reporting.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/reporting.py)
- Repo shape is now:
  - `pipelines/` for platform code
  - `research/` for effort-specific work
  - `docs/` for live operator docs
  - `docs/archive/` and `scripts/archive/` for old material

Latest local test status:

```bash
uv run --extra interp --extra analysis --extra dev -m pytest tests -q
```

Result: `322 passed`

## Runtime Model

The intended model is:

- `pipelines.cli` plus workflow specs = control plane
- Modal = execution path for capture and analysis

The remaining work is to validate and tighten that connection.

## What Still Matters

1. Run a real end-to-end Modal workflow.
2. Confirm the CLI is driving the Modal capture and analysis path correctly.
3. Confirm outputs and run metadata line up with:
   - `workflow_runs`
   - `dataset_publications`
   - local output directories

## Tomorrow's E2E Checklist

1. Create a real spec under `research/<effort>/specs/workflow.json`.
2. Register it:

```bash
uv run -m pipelines.cli spec create --file research/<effort>/specs/workflow.json
```

3. Publish the dataset:

```bash
uv run -m pipelines.cli dataset build --spec <spec_id>
uv run -m pipelines.cli publication list --spec <spec_id>
```

4. Run capture:

```bash
uv run -m pipelines.cli capture run --spec <spec_id> --output-dir data/activations/<run_name>
```

5. Run analysis:

```bash
uv run -m pipelines.cli analysis run --capture-run <run_id> --output-dir data/analysis_results/<run_name>
```

6. Build report:

```bash
uv run -m pipelines.cli report build --analysis-run <analysis_run_id>
```

## If It Fails

Capture these before changing code:

- exact command
- error output
- whether failure was local CLI, Modal dispatch, capture output, or analysis/report chaining
- relevant `workflow_runs` / `publication list` state

## Legacy Surfaces

Archive folders and legacy shell helpers are historical only. Treat them as effectively deleted unless you are explicitly digging through old work.
