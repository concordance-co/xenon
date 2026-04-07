# Workflow

The canonical Xenon workflow is:

`workflow spec -> published Neon dataset relation -> capture run -> analysis run -> report`

For real jobs, capture and analysis run on Modal. The CLI/spec layer is the control surface.

## 1. Create Or Update A Spec

To scaffold a new umbrella project and first phase:

```bash
uv run -m pipelines.cli project init --project DX_TERMINAL
```

This creates a default first phase at `projects/DX_TERMINAL/phases/phase_01/`.
Pass `--phase <name>` to choose a different initial phase name.

Checked-in spec snapshots should live under:

```text
projects/<project>/phases/<phase>/specs/workflow.json
```

Register the spec in Neon:

```bash
uv run -m pipelines.cli spec create --file projects/<project>/phases/<phase>/specs/workflow.json
uv run -m pipelines.cli spec list
```

## 2. Publish A Dataset

Build the dataset publication from the workflow spec:

```bash
uv run -m pipelines.cli dataset build --spec <spec_id>
uv run -m pipelines.cli publication list --spec <spec_id>
```

By default this publishes a named Neon view. The capture step targets that
published relation, not internal storage tables.

If the source dataset has a durable row identifier, declare it in
`dataset.identity.column`. Published workflow relations expose
`workflow_row_key` and `workflow_prompt_hash`, and capture reuse is checked
against both the stable row identity and the prompt hash.

## 3. Run Capture

```bash
uv run -m pipelines.cli capture run \
  --spec <spec_id> \
  --output-dir projects/<project>/phases/<phase>/outputs/<capture_run> \
  --layers 16,24,32
```

## 4. Run Analysis

```bash
uv run -m pipelines.cli analysis run \
  --capture-run <run_id> \
  --output-dir projects/<project>/phases/<phase>/outputs/<analysis_run>
```

This exports workflow labels from the published Neon relation into a local
parquet file and feeds that into the existing analysis toolkit.

## 5. Build A Report

```bash
uv run -m pipelines.cli report build --analysis-run <run_id>
```

This writes:

- `summary.json`
- `report.typ`
- `report.pdf` if `typst` is installed

## Canonical Interfaces

- [pipelines/cli.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/cli.py)
- [pipelines/workflows.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/workflows.py)
- [pipelines/reporting.py](/Users/marshallvyletel/repos/concordance/xenon/pipelines/reporting.py)
