# Prompt Confusion

This effort directory holds the conflict-probe experiment and its workflow-era
operator surface.

## Workflow Spec

The canonical spec lives at:

```text
research/prompt_confusion/specs/workflow.json
```

That spec wraps the legacy Neon table `conflict_probe_examples_v0`, which
predates the workflow architecture and does not expose the required `log_id`
column. The workflow source SQL generates a stable synthetic `log_id` with
`row_number() over (order by example_id)`.

That is safe here because:

- `example_id` is unique for every row in `conflict_probe_examples_v0`
- the dataset is already fixed at 375 rows

If the underlying table contents change, re-run dataset publication before
capture or analysis so the synthetic ids stay aligned.

## Canonical Commands

Register the spec in Neon:

```bash
uv run -m pipelines.cli spec create --file research/prompt_confusion/specs/workflow.json
```

Publish the workflow dataset relation:

```bash
uv run -m pipelines.cli dataset build --spec conflict_probe_v0
uv run -m pipelines.cli publication list --spec conflict_probe_v0
```

Run capture:

```bash
uv run -m pipelines.cli capture run \
  --spec conflict_probe_v0 \
  --output-dir data/activations/conflict_probe_v0
```

Run analysis:

```bash
uv run -m pipelines.cli analysis run \
  --spec conflict_probe_v0 \
  --activations-dir data/activations/conflict_probe_v0 \
  --output-dir data/analysis_results/conflict_probe_v0
```

Build a report:

```bash
uv run -m pipelines.cli report build --spec conflict_probe_v0
```

For a faster smoke run, override the capture configuration at runtime instead
of editing the checked-in spec, for example with `--limit`, `--model-id`, and
`--layers`.
