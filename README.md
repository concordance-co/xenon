# Xenon

Xenon is a spec-driven workflow for Terminal Markets interpretability work.

The canonical flow is:

`spec -> published Neon dataset object -> capture run -> analysis run -> report`

There are currently two operator surfaces in the repo:

- `pipelines.cli`
  - the older Neon/publication-driven runtime documented below
  - deprecated for new development; keep using it only for existing workflows that have not moved yet
- `pipelines_v2`
  - the newer Python workflow / artifact-oriented runtime
  - documented in [docs/PIPELINES_V2_API.md](/Users/brockelmore/concordance/xenon/docs/PIPELINES_V2_API.md)
  - supports local run tracking under `~/.xenon/pipelines_v2/catalog`, workflow resume, `rerun-step`, and `rerun-from-step`

The canonical operator surface is the CLI:

```bash
uv run -m pipelines.cli project init --project DX_TERMINAL
uv run -m pipelines.cli spec create --file projects/<project>/phases/<phase>/specs/workflow.json
uv run -m pipelines.cli dataset build --spec <spec_id>
uv run -m pipelines.cli capture run --spec <spec_id> --output-dir projects/<project>/phases/<phase>/outputs/<capture_run>
uv run -m pipelines.cli analysis run --capture-run <run_id> --output-dir projects/<project>/phases/<phase>/outputs/<analysis_run>
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
  <project>/
    project_spec.json    # umbrella project spec
    docs/
    shared/
    phases/
      <phase>/
        phase_spec.json
        specs/
          workflow.json  # checked-in executable workflow snapshot
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

Initialize a new umbrella project and first phase:

```bash
uv run -m pipelines.cli project init --project DX_TERMINAL
```

By default this creates a starter phase at `projects/DX_TERMINAL/phases/phase_01/`.
Pass `--phase <name>` if you want a different first phase name.

Create or update a workflow spec in Neon:

```bash
uv run -m pipelines.cli spec create --file projects/<project>/phases/<phase>/specs/workflow.json
uv run -m pipelines.cli spec list
uv run -m pipelines.cli spec show --id <spec_id>
```

Publish a dataset relation in Neon:

```bash
uv run -m pipelines.cli dataset build --spec <spec_id>
uv run -m pipelines.cli publication list --spec <spec_id>
```

When defining a workflow dataset, set `dataset.identity.column` if the source
uses a durable row identifier other than `log_id`. The published relation now
exposes `workflow_row_key` and `workflow_prompt_hash`, and capture reuse is
validated against that stable row identity plus prompt hash.

Run capture against the published relation:

```bash
uv run -m pipelines.cli capture run \
  --spec <spec_id> \
  --output-dir projects/<project>/phases/<phase>/outputs/<capture_run> \
  --layers 16,24,32 \
  --pool-on-capture mean_pool
```

Run analysis using the capture run and workflow labels:

```bash
uv run -m pipelines.cli analysis run \
  --capture-run <run_id> \
  --output-dir projects/<project>/phases/<phase>/outputs/<analysis_run>
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

## Operator Docs

For the older Neon/publication runtime, use the `pipelines.cli` docs above.

For new `pipelines_v2` workflow authoring and operation, start with:

- [docs/PIPELINES_V2_API.md](/Users/brockelmore/concordance/xenon/docs/PIPELINES_V2_API.md)
- [docs/ARCH2.md](/Users/brockelmore/concordance/xenon/docs/ARCH2.md)
