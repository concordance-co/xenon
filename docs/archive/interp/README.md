# Interp Docs

Planning notes and longer design documents for reusable `pipelines/interp`
runtime code live here so the package directory stays code-focused.

The canonical workflow is CLI-first:

```bash
uv run -m pipelines.cli spec create --file research/<effort>/specs/workflow.json
uv run -m pipelines.cli dataset build --spec <spec_id>
uv run -m pipelines.cli capture run --spec <spec_id>
uv run -m pipelines.cli analysis run --capture-run <run_id>
uv run -m pipelines.cli report build --analysis-run <run_id>
```

Backend/UI notes in this repo are legacy and should not be treated as the
primary operator path.
