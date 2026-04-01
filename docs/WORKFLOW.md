# Workflow

The canonical Xenon workflow is:

`workflow spec -> published Neon dataset relation -> capture run -> analysis run -> report`

## 1. Create Or Update A Spec

Checked-in spec snapshots should live under:

```text
research/<effort>/specs/workflow.json
```

Register the spec in Neon:

```bash
uv run -m pipelines.cli spec create --file research/<effort>/specs/workflow.json
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

## 3. Run Capture

```bash
uv run -m pipelines.cli capture run \
  --spec <spec_id> \
  --output-dir data/activations/<run_name> \
  --layers 16,24,32
```

## 4. Run Analysis

```bash
uv run -m pipelines.cli analysis run \
  --capture-run <run_id> \
  --output-dir data/analysis_results/<run_name>
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
