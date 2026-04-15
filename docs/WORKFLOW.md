# Workflow

The canonical Xenon workflow is:

`workflow spec -> published Neon dataset relation -> capture run -> analysis run -> report`

For real jobs, capture and analysis run on Modal. The CLI/spec layer is the control surface.

## `pipelines_v2` Operator Surface

In parallel with the older `pipelines.cli` path, the repo now has a Python
workflow / artifact-oriented runtime in `pipelines_v2`.

Use it when:

- you want one Python workflow file that defines dataset + runners + workflow
- you want artifact-driven multi-step orchestration
- you need workflow resume or targeted reruns (`rerun-step`, `rerun-from-step`)
- you want a local run registry under `~/.xenon/pipelines_v2/catalog`

Typical commands:

```bash
uv run python -m pipelines_v2.cli workflow plan --file projects/.../specs/workflow.py
uv run python -m pipelines_v2.cli workflow run --file projects/.../specs/workflow.py
uv run python -m pipelines_v2.cli workflow resume --file projects/.../specs/workflow.py --latest-failed
uv run python -m pipelines_v2.cli workflow rerun-step --file projects/.../specs/workflow.py --run-id wr_... --step report
uv run python -m pipelines_v2.cli workflow rerun-from-step --file projects/.../specs/workflow.py --run-id wr_... --step capture_prompt_eos_router
uv run python -m pipelines_v2.cli workflow runs --file projects/.../specs/workflow.py
uv run python -m pipelines_v2.cli workflow show --run-id wr_...
```

Workspace defaults can live in the repo-root [`xenon.toml`](/Users/brockelmore/concordance/xenon/xenon.toml).
That file is git-committable and is the right place for shared defaults such as
the external catalog env var and dashboard static dir. CLI flags still win when
you pass them explicitly, and workflow runner specs still win when they set
their own catalog directly.

The detailed library and CLI contract is documented in
[PIPELINES_V2_API.md](/Users/brockelmore/concordance/xenon/docs/PIPELINES_V2_API.md).

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

For new synthetic datasets, do not scale immediately to the full target size.
The recommended pattern is:

1. publish a small smoke dataset first, usually a few hundred rows
2. inspect slices manually
3. run early behavioral and leakage checks
4. only then scale to the larger publication

This avoids spending capture and analysis cycles on a large dataset with obvious
construction errors or trivial shortcuts.

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

For workflow-driven analysis on a narrower slice, prefer passing an explicit
publication relation or view:

```bash
uv run -m pipelines.cli analysis run \
  --capture-run <run_id> \
  --publication <slice_relation> \
  --execution modal \
  --output-dir projects/<project>/phases/<phase>/outputs/<analysis_run>
```

This is the canonical way to analyze:

- aligned vs strong-conflict slices
- family-restricted slices
- strong-conflict-only source-following slices

Analysis compaction is slice-aware: it now compacts only the rows present in
the exported label slice, not the entire capture by default.

If the benchmark has dependent rows, prefer grouped evaluation:

```bash
uv run -m pipelines.cli analysis run \
  --capture-run <run_id> \
  --publication <slice_relation> \
  --group-column matched_pair_id \
  --execution modal \
  --output-dir projects/<project>/phases/<phase>/outputs/<analysis_run>
```

Use `--group-column` whenever related rows should stay in the same fold. This
is especially important for matched prompt pairs, template families, or other
structured variants where random row-level CV would leak information.

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
