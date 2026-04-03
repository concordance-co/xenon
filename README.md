# Xenon

Xenon is a spec-driven workflow for Terminal Markets interpretability work.

The canonical flow is:

`spec -> published Neon dataset object -> capture run -> analysis run -> report`

The canonical operator surface is the CLI:

```bash
uv run -m pipelines.cli spec create --file projects/<effort>/specs/workflow.json
uv run -m pipelines.cli dataset build --spec <spec_id>
uv run -m pipelines.cli capture run --spec <spec_id> --output-dir data/activations/<run_name>
uv run -m pipelines.cli analysis run --capture-run <run_id> --output-dir data/analysis_results/<run_name>
uv run -m pipelines.cli report build --analysis-run <run_id>
```

For real jobs, capture and analysis run on Modal. The CLI/spec layer is the control surface.

## Repo Map

```text
pipelines/
  cli.py                 # canonical operator interface
  workflows.py           # workflow spec/run/publication registry
  reporting.py           # generic workflow report builder
  ingest/                # Terminal Markets API -> Neon
  datasets/              # dataset build, labeling, publication, synthetic generation
  interp/                # reusable capture/analysis runtime code

projects/
  <effort>/
    specs/               # checked-in workflow spec snapshots
    notes/
    outputs/
    reports/
    scripts/
```

Rules:

- `pipelines/` is reusable platform code.
- `projects/` is effort-local code and deliverables.
- Backend/UI are legacy surfaces and are no longer the documented path.
- `archive/` folders are historical only. Treat them as effectively deleted unless you are explicitly digging through old work.

## Workflow Commands

Create or update a workflow spec in Neon:

```bash
uv run -m pipelines.cli spec create --file projects/<effort>/specs/workflow.json
uv run -m pipelines.cli spec list
uv run -m pipelines.cli spec show --id <spec_id>
```

Publish a dataset relation in Neon:

```bash
uv run -m pipelines.cli dataset build --spec <spec_id>
uv run -m pipelines.cli publication list --spec <spec_id>
```

Run capture against the published relation:

```bash
uv run -m pipelines.cli capture run \
  --spec <spec_id> \
  --output-dir data/activations/<run_name> \
  --layers 16,24,32 \
  --pool-on-capture mean_pool
```

Run analysis using the capture run and workflow labels:

```bash
uv run -m pipelines.cli analysis run \
  --capture-run <run_id> \
  --output-dir data/analysis_results/<run_name>
```

Build a workflow report:

```bash
uv run -m pipelines.cli report build --analysis-run <run_id>
```

This writes `summary.json` and `report.typ`, and compiles a PDF if `typst` is installed.

## Setup

```bash
uv sync
uv sync --extra interp --extra analysis --extra dev
```

Environment:

```bash
XENON_NEON_DATABASE_URL=postgresql://...
```

## Legacy Surfaces

`pipelines.cli` is the only recommended path.
