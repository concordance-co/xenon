---
name: constructing-workflows
description: Use when designing or editing `pipelines_v2` workflow files in this repo. Covers workflow file structure, runner specs, dataset patterns, row alignment, section metadata, report inputs, and the decision boundary between first-class specs and project-local transforms.
---

# Constructing Workflows

Use this skill when the task is to author or significantly reshape a
`pipelines_v2` workflow.

This skill is for workflow construction, not day-to-day operation. For running,
resuming, rerunning, and inspecting existing workflows, use
`pipelines-v2-run-ops`.

## Start Here

1. Read:
   - [docs/PIPELINES_V2_API.md](/Users/brockelmore/concordance/xenon/docs/PIPELINES_V2_API.md)
   - [docs/ARCH2.md](/Users/brockelmore/concordance/xenon/docs/ARCH2.md)
2. Inspect the nearest existing phase workflow under `projects/.../specs/workflow.py`.
3. Prefer first-class `pipelines_v2` specs over `TransformSpec`.

## Preferred Workflow File Shape

Prefer:

```python
def build_dataset() -> Dataset: ...
def build_workflow(dataset: Dataset | None = None) -> WorkflowSpec: ...
def build_runner_specs() -> dict[str, RunnerSpec]: ...
```

Keep a checked-in JSON snapshot alongside the Python source:

- `projects/.../specs/workflow.py`
- `projects/.../specs/workflow.json`

Within a project phase, default to a single checked-in workflow as the
executable source of truth. If the work no longer fits cleanly in one workflow,
that is usually a sign the phase boundary should change rather than adding
multiple competing workflows to the same phase.

## Runner Construction

- Prefer `build_runner_specs()` over ad hoc CLI flag-only construction.
- Default every non-report step to a remote runner. Reports should be local by
  default.
- Use named runners such as:
  - `capture_gpu`
  - `analysis_cpu`
  - `report_local`
- In practice, this usually means:
  - capture / model-bound execution on a remote GPU runner
  - analysis / derive / readout / geometry / other artifact-bound compute on a
    remote CPU runner
  - report packaging on `report_local`
- Put engines on model-bound specs, not on runners.
- If a workflow uses a shared external catalog, keep runner catalog identities aligned.

## Dataset Construction

- For local examples: `Dataset.from_examples(...)`
- For remote sources: prefer `Dataset.from_postgres(...)`
- For deferred Postgres datasets, use `PostgresSource.from_env(...)`, not raw URLs
- If a workflow mixes feature rows from one dataset with labels/grouping from another, declare `rows=...`

Do not rely on accidental key overlap.

## Token Sections

- `TokenSelector.section(...)` requires explicit token-section metadata
- Use `PromptMetadataBuilder.from_function(...)`
- Do not rely on inferred sections

## vLLM Routing Constraints

For `MoERoutingSite` with `VLLMEngine`, currently require:

- `enforce_eager=True`
- `enable_prefix_caching=False`
- `max_num_seqs=1`

Do not assume router capture can share the same batching envelope as residual
capture.

If a workflow needs both residual activations and MoE router features, prefer
separate workflow steps. Keep the normal residual capture on its own step, and
make MoE capture a distinct step with its own runtime envelope.

## Statistical Methods Available Today

Use the narrowest first-class spec that matches the scientific question.

### Probe / Readout Surface

- `ProbeSpec`
  - standard linear activation probe across layers
  - best for: "is label X linearly decodable from this feature?"
  - supports:
    - grouped CV
    - fixed train/test splits
    - residual and router features
    - metrics: `accuracy`, `balanced_accuracy`, `auroc`, `selectivity`
  - implementation uses `SGDClassifier(loss=\"log_loss\")`

- `TransferProbeSpec`
  - cross-cohort transfer probe for activations
  - best for:
    - family transfer
    - lexical holdout
    - regularization sweeps
  - supports:
    - within-vs-transfer comparisons
    - grouped CV where possible
    - residual and router features
  - implementation uses standardized `LogisticRegression`

- `TextBaselineSpec`
  - text-only baseline
  - best for:
    - lexical leakage checks
    - "how much can raw text alone explain?"
  - current model family:
    - `countvectorizer_logreg`

- `ResidualizedProbeSpec`
  - nuisance-removal before probing
  - best for:
    - "does the target signal survive after removing family / lexical subspace?"
  - currently removes a learned linear nuisance subspace, then re-runs the target probe

### Representation / Geometry Surface

- `DirectionSpec`
  - simple positive-vs-negative difference vector
  - best for:
    - quick contrast directions
    - steering-style exploratory work

- `BasisSpec`
  - reusable basis learning surface
  - current implementation:
    - PCA only
  - use when you want a decomposition artifact, not just a plotted view

- `GeometrySpec`
  - low-dimensional projection for inspection/reporting
  - current methods:
    - `pca`
    - `lda`
  - use when you want structured projection outputs for reports, not a reusable basis object

- `PairDeltaSpec`
  - paired contrast feature construction
  - best for:
    - matched-pair workflows
    - conflict-vs-aligned deltas within one case structure

### Derived Labels

- `LabelMapSpec`
  - taxonomy collapse / vocabulary remapping
- `LabelFieldsSpec`
  - extract structured fields from a composite label payload

## Statistical Sharp Edges And Best Practices

### 1. Prefer grouped evaluation when rows are related

If rows share a matched-pair id, template id, or any other dependency structure,
use `group_by=...`.

Do not use row-level CV for matched prompt pairs unless you explicitly want leakage.

### 2. Use `rows=...` whenever the analysis row universe is not the full feature row set

If:

- features come from capture on dataset A
- labels/grouping/cohorts come from dataset B

declare `rows=B` explicitly.

This is not optional workflow hygiene; it changes the semantics of the analysis.

### 3. Check class support before choosing a smoke slice

Small `LIMIT N` slices can easily become one-class datasets.

That breaks:

- `ProbeSpec`
- `TransferProbeSpec`
- `TextBaselineSpec`
- `ResidualizedProbeSpec`
- `GeometrySpec(method=\"lda\")`

If you need a smoke run, prefer an explicit balanced SQL slice instead of raw `LIMIT`.

### 4. Use `balanced_accuracy` by default for skewed binary tasks

Plain accuracy is often misleading on imbalanced tasks. In most workflow design
cases here, `balanced_accuracy` should be the default headline metric.

Use `auroc` when:

- the problem is binary
- class ranking matters
- you want a threshold-free view

### 5. Treat `selectivity` as a stronger claim than raw accuracy

`selectivity` is probe performance relative to a shuffled-label control.

Use it when you want to argue:

- the representation contains real decodable signal
- not just that a flexible classifier can fit the dataset

### 6. Separate activation readouts from text baselines

Do not fold lexical controls into activation probes.

Use:

- `TransferProbeSpec` / `ProbeSpec` for activations
- `TextBaselineSpec` for text-only controls

Then compare them in the report.

### 7. Router features are not just "another residual"

Router features currently have different runtime constraints and a different
payload shape.

When designing a workflow:

- keep router capture as its own workflow step when it is needed
- do not assume batching behavior matches residual capture
- validate the requested router record type is what the downstream analysis needs

### 8. `BasisSpec` and `GeometrySpec` are not interchangeable

- `BasisSpec`
  - reusable learned basis/decomposition
- `GeometrySpec`
  - report/inspection projection view

If you just need projected coordinates for a report, prefer `GeometrySpec`.
If you need a basis object that another step should conceptually consume later,
prefer `BasisSpec`.

### 9. `GeometrySpec` emits structured payloads, not plots

Do not assume geometry steps create figures. They produce projection payloads
that reports can consume.

If a workflow needs rendered charts, make that an explicit reporting concern.

### 10. Keep `TransformSpec` out of the critical path unless it is genuinely project-specific

If a statistical pattern is clearly reusable, add the first-class spec instead
of burying core methodology in a transform.

## Report Construction

- `ReportSpec` should take explicit direct inputs
- direct `OperationArtifact` inputs will have their `result.json` copied into:
  - `report_<id>/results/{step_name}_results.json`
- capture artifacts are summarized from manifests and are not localized just to build the report

If a report needs more than summary/provenance + direct analysis results, add
the missing reporting behavior deliberately instead of sneaking it into CLI glue.

### Reports run locally, not on Modal

Report steps must be assigned to a local runner (for example, `report_local`
backed by `LocalRunnerSpec` + `LocalArtifactStore`). The chart/plot stack
(`matplotlib`, `pipelines_v2.reporting`) is intentionally not part of the
Modal capture or analysis runtime image.

Rules of thumb:

- Do not assign a `ReportSpec` step to a Modal runner.
- Do not add `matplotlib` or `pipelines_v2.reporting` to capture/analysis
  runtime dependencies just to make a remote report step work.
- If a new chart or plotting behavior is needed, add it under
  `pipelines_v2.reporting` and keep any imports of it lazy in modules that
  load on Modal (capture engines, analysis runtime paths).

If you find yourself wanting matplotlib on a Modal runner, stop — the
artifact that report step needs should be pulled back locally first and
rendered in a `report_local` step.

## When To Use `TransformSpec`

Use `TransformSpec` only when:

- the logic is genuinely project-specific
- there is not already a first-class spec that matches the intent

Do not use `TransformSpec` just because the old script did a lot inline.

If the same transform pattern appears repeatedly, stop and add the missing
first-class spec to `pipelines_v2`.

## Construction Checklist

Before calling a workflow “done”, verify:

1. the workflow plans cleanly
2. row alignment is explicit where needed
3. prompt section metadata is explicit where needed
4. runner specs match the intended resource split
5. the phase has one checked-in workflow source of truth
6. the checked-in JSON snapshot matches the Python builder
7. the report step only references what it really needs
