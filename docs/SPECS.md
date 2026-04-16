# Specs

`pipelines_v2` workflow files are the canonical checked-in authoring surface in
this repo.

## Workflow Files

Typical file contract:

```python
def build_dataset() -> Dataset: ...
def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec: ...
def build_runner_specs() -> dict[str, RunnerSpec]: ...  # optional but recommended
```

Typical checked-in locations:

```text
projects/<project>/<subproject>/<phase>/specs/workflow.py
projects/<project>/<subproject>/<phase>/specs/workflow.json
```

Where:

- `workflow.py`
  - the executable authoring source
- `workflow.json`
  - an optional checked-in snapshot for reviewability and reproducibility

The runtime entrypoint for these files is:

```bash
uv run python -m pipelines_v2.cli workflow plan --file projects/.../specs/workflow.py
uv run python -m pipelines_v2.cli workflow run --file projects/.../specs/workflow.py
```

See [PIPELINES_V2_API.md](/Users/marshallvyletel/repos/concordance/xenon/docs/PIPELINES_V2_API.md)
for the current `pipelines_v2` API and CLI surface.

## Notes

- Prefer `workflow.py` as the source of truth.
- Keep `workflow.json` only when a phase wants a checked-in snapshot.
- Use `workflow runs`, `workflow show`, `workflow resume`, `workflow rerun-step`,
  and `workflow rerun-from-step` for follow-up operations rather than manual
  reconstruction.
