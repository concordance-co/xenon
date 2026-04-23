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
projects/<project>/<phase>/specs/workflow.py
projects/<project>/<phase>/specs/workflow.json
projects/<project>/<subproject>/<phase>/specs/workflow.py
projects/<project>/<subproject>/<phase>/specs/workflow.json
```

Both layouts are valid:

- use `projects/<project>/<phase>/...` when the phase is part of the main project flow
- use `projects/<project>/<subproject>/<phase>/...` when the phase belongs to a true nested subproject

Where:

- `workflow.py`
  - the executable authoring source
- `workflow.json`
  - an optional checked-in snapshot for reviewability and reproducibility

The runtime entrypoint for these files is:

```bash
uv run python -m pipelines_v2.cli workflow plan --file projects/.../specs/workflow.py
uv run python -m pipelines_v2.cli workflow run --file projects/.../specs/workflow.py --logging INFO
```

See [PIPELINES_V2_API.md](/Users/brockelmore/concordance/xenon/docs/PIPELINES_V2_API.md)
for the current `pipelines_v2` API and CLI surface.

Authoring notes for Python workflow files:

- Prefer checked-in `build_dataset()`, `build_workflow(...)`, and
  `build_runner_specs()` functions over ad hoc CLI-only construction.
- When a workflow uses `PromptMetadataBuilder.from_function(...)` or other
  runtime-imported local helpers, pass explicit narrow `local_python_sources`
  such as `("pipelines_v2", "scripts")` or one project-local package root.
- Do not rely on the default `"."` source root for Modal-backed workflows
  unless you intentionally want the whole workspace mounted into the remote
  image.

## Notes

- Prefer `workflow.py` as the source of truth.
- Keep `workflow.json` only when a phase wants a checked-in snapshot.
- Use `workflow runs`, `workflow show`, `workflow resume`, `workflow rerun-step`,
  and `workflow rerun-from-step` for follow-up operations rather than manual
  reconstruction.
