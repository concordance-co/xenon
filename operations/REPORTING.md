# Reports and shared results

Build reports from existing artifacts. Report generation does not require another model run.
Use the format that serves the task. Xenon does not require Typst, PDF, or a notebook ban.

## Rebuild a workflow report

The source run must contain a `ReportSpec` step with accessible input artifacts.
Use the selected runtime's Python. These examples use `python` as shorthand.

```bash
python -m pipelines_v2.cli workflow report wr_YOUR_RUN
python -m pipelines_v2.cli workflow report wr_YOUR_RUN --step REPORT_STEP_NAME
```

`--step` is required when the run contains more than one report step.
Use `--local-catalog-root /absolute/path/catalog` for an alternate local catalog.
This command loads the saved workflow and runs only the selected report locally.
It does not repeat capture or analysis.

The command returns the report artifact ID, location, and summary.
When `ReportSpec.output_dir` is set, files are written under `output_dir/ARTIFACT_ID/`:

- `report.md`, `report.json`, and `summary.json`;
- selected input results and generated chart assets, where supported.

Without `output_dir`, the report remains an artifact result; no presentation files are generated there.
Use `workflow rerun-step` when a changed workflow definition must execute a new report step.
The `workflow report` command uses the original saved definition.

## Build an evidence view

Use the evidence reader for selected artifacts or when no report step exists:

```bash
python -m pipelines_v2.evidence report ARTIFACT_ID --run-id wr_YOUR_RUN --output /absolute/path/report.html
```

The default catalog is the local mirror. For another catalog, put
`--catalog-root /absolute/path/catalog` before `report`.
The viewer reads supported summaries within explicit limits. It does not fetch raw captures.
A comparison also requires compatible evidence records; a matching metric name is not sufficient.
See [evidence views](../platform/EVIDENCE.md) for supported data and limits.

## Report inputs and code

Use artifact IDs and typed refs to select report inputs.
Keep large remote tensors on Modal. Compute a reduction there when a report needs one.
Check the size of result payloads before transferring them.
An operation result is not necessarily small because it is JSON.

Route report rendering to a local runner. Keep reporting imports inside the functions
that need them when the module is also used on Modal.
The chart dependencies are not part of every worker image.

Keep report source and inputs sufficient to rebuild the output.
A study can choose its own output directory and deliverable format.
Do not commit generated artifact stores to Xenon.

## Share an artifact

Use the [publication procedure](../platform/PUBLICATION.md) to plan a transfer.
Select the source artifact, destination volume, dependencies, and byte budget.
Inspect the plan before publication. Existing authorization for the specified transfer is sufficient.

Publication copies supported outputs and indexes the shared result. It does not rerun the model.
Some artifact types support only a summary view; bulk captures are not local publication candidates.
Publication preserves source identity. It does not prove research quality or compatibility with a different GPU.
