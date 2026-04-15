# Pipelines v2 API Reference

This document describes the current user-facing `pipelines_v2` library surface.
It is a reference for what each public spec or method means, how to use it, and
what is actually implemented today.

The public import surface is:

```python
from pipelines_v2.api import ...
```

The intended execution model is:

```text
dataset/source refs -> capture spec -> capture artifact -> analysis/report specs -> workflow orchestration
```

## Status

Current execution support:

- implemented:
  - `CaptureSpec`
  - `ProbeSpec`
  - `TransferProbeSpec`
  - `TextBaselineSpec`
  - `ResidualizedProbeSpec`
  - `DirectionSpec`
  - `BasisSpec`
  - `GeometrySpec`
  - `PairDeltaSpec`
  - `LabelMapSpec`
  - `LabelFieldsSpec`
  - `TransformSpec`
  - `ReportSpec`
- schema exists but execution is not implemented yet:
  - `ActivationPatchSpec`

Current engine support:

- `ToyEngine`: local deterministic test engine
- `VLLMEngine`: real capture engine

Current runner support:

- `LocalRunner`: local execution
- `ModalRunner`: remote execution on Modal
- `LocalRunnerSpec` / `ModalRunnerSpec`: serializable runner-profile definitions for Python workflow files

Current workflow persistence support:

- workflow runs and workflow step records are persisted when runners share one non-null catalog
- `WorkflowOrchestrator.run(..., resume_run_id=..., reuse_completed=...)` is implemented
- `WorkflowOrchestrator.run(..., reuse_from_run_id=..., force_rerun_steps=..., parent_run_id=...)` is implemented
- artifact manifests carry workflow provenance in `workflow_context`
- workflow step records capture `runtime_app_id` when a runner reports one back during execution
- the CLI always mirrors workflow state into a local file catalog under `~/.xenon/pipelines_v2/catalog` by default

Current operation-module layout:

- `pipelines_v2.operations.capture`
  - model-bound capture specs and capture-site definitions
- `pipelines_v2.operations.derive`
  - derived labels, paired contrasts, and function-backed transforms
- `pipelines_v2.operations.readouts`
  - readout and probe specs
- `pipelines_v2.operations.representation`
  - direction and decomposition specs
- `pipelines_v2.operations.interventions`
  - intervention specs such as patching
- `pipelines_v2.operations.reports`
  - report packaging specs
- `pipelines_v2.operations.execution`
  - artifact-bound execution dispatch split by operation family

Compatibility note:

- `pipelines_v2.operations.specs` and `pipelines_v2.operations.execute` still exist as stable import shims
- new code should prefer the family modules above

## Core Concepts

- `Example`
  - one prompt invocation
- `Dataset`
  - a collection of examples, either materialized in memory or deferred to a source
- `CaptureSpec`
  - model-bound operation that produces activations, routing data, and/or generations
- `CaptureArtifact`
  - stored outputs from a capture run
- artifact-bound analysis spec
  - consumes feature or label refs from existing artifacts instead of talking to a model
- `Runner`
  - executes one spec under one resource profile
- `RunnerSpec`
  - serializable execution-profile definition that materializes a concrete runner
- `WorkflowOrchestrator`
  - coordinates multiple steps across named runners

## Data Layer

### `Example`

What it is:
- one user-facing unit of model input

Important fields:
- `key`
  - stable example id
- `prompt`
  - either a plain string or a chat-message sequence
- `prompt_hash`
  - optional explicit hash; auto-derived if omitted
- `labels`
  - named supervision/grouping metadata
- `metadata`
  - extra non-label fields used by engines or downstream code
- `cases`
  - named case/group memberships
- `case_key`
  - convenience primary case id; also mirrored into `cases["case_key"]`

Common methods:
- `to_dict()`
- `from_dict(payload)`

### `Dataset`

What it is:
- a dataset abstraction that can be:
  - materialized: examples already present locally
  - deferred: a source + fetch plan to resolve inside a runtime

Constructors:
- `Dataset.from_examples(examples, *, id=None, name=None)`
  - use for hardcoded examples in Python
- `Dataset.from_records(records, *, prompt_column=..., example_key_column=..., ...)`
  - convert row dicts into examples
- `Dataset.from_json(path, **kwargs)`
- `Dataset.from_parquet(path, **kwargs)`
- `Dataset.from_source(source=..., **kwargs)`
  - generic adapter over any `Source`
- `Dataset.from_postgres(source=..., table=... | sql=..., **kwargs)`
  - Postgres-specific deferred dataset constructor
  - requires exactly one of `table` or `sql`

Common methods:
- `dataset.select(limit=..., keys=...)`
  - narrows the dataset
  - if the dataset is deferred, this updates the deferred selection instead of fetching locally
- `dataset.labels(name)`
  - returns a `LabelSet`
- `dataset.cases(name="case_key")`
  - returns a `CaseSet`
- `dataset.example_keys()`
  - returns example keys if materialized, or selected keys if explicitly constrained
- `dataset.coverage()`
  - summary used in manifests
- `dataset.resolve()`
  - fetches a deferred dataset inside the current runtime
- `dataset.to_dict()`
- `Dataset.from_dict(payload)`

Usage:

```python
dataset = Dataset.from_postgres(
    source=PostgresSource.from_env("XENON_DATABASE_URL"),
    sql="""
        SELECT example_id, prompt_messages_json, conflict_present
        FROM conflict_probe_examples_v3
        WHERE lexical_split in ('train', 'test')
    """,
    prompt_column="prompt_messages_json",
    example_key_column="example_id",
    label_columns=["conflict_present", "lexical_split"],
)
```

### `LabelSet`

What it is:
- a named mapping from example key -> value

Common methods:
- `label_set.equals(value)`
  - returns a `LabelPredicate`
- `label_set.for_examples(example_keys)`
- `label_set.resolve_values()`
  - resolves even for deferred datasets
- `label_set.values`
  - only available when the underlying dataset is already materialized

### `CaseSet`

What it is:
- a named mapping from example key -> case/group id

Common methods:
- `case_set.for_examples(example_keys)`
- `case_set.resolve_values()`
- `case_set.values`
  - only available for materialized datasets

### `LabelPredicate`

What it is:
- a filter over a label source

Currently supported predicate:
- `equals(value)`

Common methods:
- `matches(example_key)`
- `resolve_example_keys()`

### Sources

#### `InMemorySource`

Use when:
- the records already exist locally

Constructors/methods:
- `InMemorySource.from_records(records)`
- `identity()`
- `runtime_secrets()`
- `fetch_dataset(**kwargs)`

#### `PostgresSource`

Use when:
- examples live in Postgres

Constructors/methods:
- `PostgresSource.from_env("XENON_DATABASE_URL")`
  - preferred for deferred/runtime execution
- `PostgresSource(url="postgresql://...")`
  - fine for immediate local fetches, not allowed for deferred serialization
- `identity()`
  - only works when the source was built from `url_env_var`
- `runtime_secrets()`
- `fetch_dataset(...)`
  - supports:
    - `table="schema.relation"`
    - `sql="SELECT ..."`
  - exactly one is required
- `connection_url()`
  - resolves the actual runtime connection string

## Engine Layer

### `RuntimeSecret`

What it is:
- a declaration that a runtime must provide a given env var

### `RuntimeSpec` and `PythonRuntimeSpec`

What they are:
- runner-agnostic runtime requirements

`PythonRuntimeSpec` fields:
- `python_version`
- `pip_packages`
- `env`
- `secrets`
- `local_python_sources`

### `Engine`

Protocol for model-bound execution.

Methods:
- `identity()`
- `capabilities()`
- `runtime_spec()`
- `planning_errors(spec)`
- `capture(spec)`

### `ToyEngine`

What it is:
- deterministic fake engine for tests and local bring-up

Use when:
- validating workflow shape
- testing downstream analysis
- writing local smoke tests

### `VLLMEngine`

What it is:
- real capture backend for vLLM

Important fields:
- `model_id`
- `max_model_len`
- `tensor_parallel_size`
- `gpu_memory_utilization`
- `enforce_eager`
- `max_num_seqs`
- `enable_prefix_caching`
- `enable_chunked_prefill`
- `add_generation_prompt`
- `reasoning_parser`
- `enable_thinking`
  - pass-through to the tokenizer's chat template for models that
    support an enable_thinking kwarg (e.g., Qwen3). `None` leaves the
    template default; set `False` to skip the reasoning prefix and
    make short `max_tokens` generation usable.
- `extra`

Important methods:
- `identity()`
- `capabilities()`
- `runtime_spec()`
- `planning_errors(spec)`
  - currently rejects MoE routing capture with `enable_prefix_caching=True`
  - allows compiled mode (`enforce_eager=False`) and batched routing capture
    when prefix caching is disabled
- `capture(spec)`

## Capture Surface

### `TokenSelector`

What it is:
- selects which captured token positions to keep from a captured sequence

Constructors:
- `TokenSelector.last()`
- `TokenSelector.full_sequence()`
- `TokenSelector.slice(start, stop=None)`
- `TokenSelector.section(name)`

Notes:
- `section(name)` requires explicit `token_sections` metadata
- there is no automatic section inference anymore
- for remote/deferred datasets, the usual pattern is to attach a `PromptMetadataBuilder` to the capture spec
- capture stores the selected token states
- analysis specs can further select and pool from the stored token axis

### `PromptMetadataBuilder`

What it is:
- a serializable reference to a user-defined Python function that derives prompt metadata at capture time

What the function must do:
- accept one positional argument: the rendered prompt text
- return a mapping
- if you want `TokenSelector.section(name)`, return metadata with `token_sections`

Supported `token_sections` shapes:
- `{name: [token_index, ...]}`
- `{name: {"token_positions": [...]}}`
- `{name: {"char_start": ..., "char_end": ...}}`

Important constructors/methods:
- `PromptMetadataBuilder.from_function(fn)`
- `build(rendered_prompt)`

Important note:
- the function must be importable in the runtime where capture runs
- `from_function(...)` records the module path and local source roots needed by Modal
- if you do not pass `local_python_sources=...`, `from_function(...)` treats the
  workspace root as the source root and records `"."`

### `TokenPooling`

What it is:
- reduces selected token states to one vector per example

Constructors:
- `TokenPooling.mean()`
- `TokenPooling.last()`
- `TokenPooling.first()`

Typical use:

```python
ProbeSpec(
    feature=cap.feature("resid_strategy_section"),
    labels=dataset.labels("conflict_present"),
    tokens=TokenSelector.full_sequence(),
    pooling=TokenPooling.mean(),
)
```

### `TensorStorage`

What it is:
- per-site tensor storage policy

Current fields:
- `dtype`
- `format`

Current practical format:
- `safetensors`

### `GenerationSpec`

What it is:
- generation settings attached to a capture

Fields:
- `enabled`
- `max_tokens`
- `temperature`
- `capture_reasoning`
- `structured_output`

### `ResidualSite`

What it is:
- capture request for residual activations at one named site

Fields:
- `name`
- `site`
- `layers`
- `tokens`
- `storage`

### `RoutingRecord`

What it is:
- one requested MoE router output family

Constructors:
- `RoutingRecord.gate_logits(dtype="float16")`
- `RoutingRecord.gate_probs(dtype="float16")`
- `RoutingRecord.routing_decisions(required=True)`
- `RoutingRecord.topk_from_gate(k=..., include_weights=True)`
- `RoutingRecord.expert_load(source=...)`

### `MoERoutingSite`

What it is:
- capture request for MoE routing information

Fields:
- `name`
- `layers`
- `tokens`
- `record`

### `CaptureSpec`

What it is:
- the main model-bound execution spec

Fields:
- `engine`
- `dataset`
- `sites`
- `generation`
- `prompt_metadata_builder`

Important methods:
- `required_capabilities()`
- `runtime_secrets()`
- `bound_engine()`
- `runtime_spec()`
- `resolve_dataset()`
- `uses_section_token_selector()`
- `provides_token_sections()`
- `to_dict()`
- `from_dict(payload)`
- `from_file(path)`

## Artifact-Bound Analysis and Derived Ops

These specs run against existing features/labels/generations instead of running
a model directly.

### `ProbeSpec`

What it does:
- trains and evaluates a linear probe over one feature family across layers

Fields:
- `feature`
  - `FeatureRef`, `FeatureLayerRef`, or `StepFeatureRef`
- `rows`
  - optional explicit row universe for analysis
  - use this when feature rows come from one capture dataset but labels/grouping come from a narrower dataset or subset
- `labels`
  - `LabelSet`, `ArtifactLabelRef`, or `StepLabelRef`
- `group_by`
  - optional grouping ref to keep related rows together
- `split`
  - optional fixed train/test split label
- `tokens`
- `pooling`
- `train_values`
- `test_values`
- `folds`
- `baselines`
- `metrics`

Current metrics:
- `accuracy`
- `balanced_accuracy`
- `auroc`
- `selectivity`

Current baselines:
- `majority`
- `shuffled_label`

Implementation notes:
- uses `SGDClassifier(loss="log_loss")`
- supports grouped or fixed split evaluation
- supports residual and MoE-router feature payloads
- if `rows` is provided, probe execution first aligns feature rows to that declared row universe
- if `rows` is omitted, workflow planning treats the feature/text rows as the default row universe
- `selectivity` is computed against a shuffled-label control even if
  `shuffled_label` is not requested as an exposed baseline metric

### `TransferProbeSpec`

What it does:
- evaluates cross-cohort transfer for activation features
- can also run fixed split holdouts such as lexical holdout within a cohort

Fields:
- `feature`
- `rows`
- `labels`
- `group_by`
- `cohort_by`
- `cohort_values`
- `split_by`
- `train_values`
- `test_values`
- `regularization`
- `metrics`
- `compare_within_baseline`
- `compare_direction_similarity`
- `tokens`
- `pooling`

Current implementation:
- supports residual and MoE-router feature payloads
- uses `LogisticRegression` with standardization for activation readouts
- grouped CV uses stratified-group splits when possible
- cross-cohort transfer reports test-side delta versus within-cohort baseline
- regularization sweeps are supported through repeated `C` values
- when `rows` is provided, transfer analysis runs over that explicit row universe instead of assuming full feature coverage

### `TextBaselineSpec`

What it does:
- evaluates text-only baselines over raw text labels such as `user_text`

Fields:
- `text`
- `rows`
- `labels`
- `group_by`
- `cohort_by`
- `cohort_values`
- `split_by`
- `train_values`
- `test_values`
- `model`
- `regularization`
- `metrics`

Current implementation:
- supports `model="countvectorizer_logreg"`
- supports grouped CV, cross-cohort transfer, and fixed split holdouts
- uses `CountVectorizer(ngram_range=(1, 2)) + LogisticRegression`
- `rows` can restrict text-baseline evaluation to a declared subset or secondary dataset row universe

### `ResidualizedProbeSpec`

What it does:
- removes a nuisance linear subspace, then reruns the target probe

Fields:
- `feature`
- `rows`
- `labels`
- `residualize_against`
- `group_by`
- `metrics`
- `tokens`
- `pooling`

Current implementation:
- fits a logistic family/nuisance classifier on raw activations
- projects into the orthogonal complement of the learned coefficient row space
- reports both raw and residualized probe metrics plus deltas

### `DirectionSpec`

What it does:
- computes a difference vector between positive and negative groups

Fields:
- `feature`
- `positive`
- `negative`
- `group_by`
- `layers`
- `tokens`
- `pooling`

### `BasisSpec`

What it does:
- learns a basis over activations

Fields:
- `feature`
- `rows`
- `method`
- `by`
- `layers`
- `components`
- `tokens`
- `pooling`

Current implementation:
- only `method="pca"`

### `GeometrySpec`

What it does:
- projects activations into a low-dimensional geometry view for inspection or reporting

Fields:
- `feature`
- `method`
- `layers`
- `label`
- `color_by`
- `subset`
- `normalize`
- `components`
- `tokens`
- `pooling`

Current implementation:
- supports `method="pca"` and `method="lda"`
- supports subset filtering with `LabelPredicate`
- supports `normalize="rms_per_row"` or `None`
- emits structured projection payloads, not rendered plots
- `rows` can declare the base row universe before any `subset` predicate is applied

### `PairDeltaSpec`

What it does:
- computes paired deltas between positive and negative members of the same case
- emits a new derived residual feature plus propagated labels

Fields:
- `feature`
- `case`
- `positive`
- `negative`
- `layers`
- `tokens`
- `pooling`
- `output_feature_name`
- `labels`
- `propagate_from`

### `LabelMapSpec`

What it does:
- remaps one label vocabulary into another

Fields:
- `source`
- `mapping`
- `output_name`
- `strict`
- `default_value`

Use when:
- you want taxonomy collapse or label derivation to live in the workflow graph

### `LabelFieldsSpec`

What it does:
- extracts named fields from a structured label payload

Fields:
- `source`
- `fields`
- `strict`

Typical use:
- split `expected_output_json` into `action`, `asset`, `size`

### `TransformSpec`

What it does:
- runs a user-defined transform function in the analysis runtime over named inputs

Fields:
- `builder`
- `inputs`

Builder surface:
- `TransformBuilder.from_function(fn)`
- `TransformResult(...)`

Function contract:
- the function must be importable in the runtime
- it is called with keyword arguments from `inputs`
- it returns either:
  - `TransformResult`, or
  - a mapping with keys like `payload`, `labels`, `metadata`, and `example_keys`

Typical use:
- project-specific behavioral relabeling
- generation postprocessing
- custom derived labels that do not belong as first-class core specs

### `ActivationPatchSpec`

What it is:
- the planned model-bound intervention spec

Fields:
- `engine`
- `dataset`
- `basis`
- `site`
- `layers`
- `tokens`
- `mode`
- `components`
- `strengths`
- `controls`
- `metrics`

Status:
- serializable now
- execution not implemented yet

### `ReportSpec`

What it does:
- packages existing artifacts/steps into a report artifact

Fields:
- `inputs`
- `template`
- `output_dir`

Current implementation:
- emits a structured summary artifact
- `LocalRunner` can also materialize report files under `output_dir`

Runtime boundary:
- `ReportSpec` steps are intended to run on a local runner.
- The chart/plot stack (`matplotlib`, `pipelines_v2.reporting`) is
  deliberately not part of the Modal capture or analysis runtime image.
- Do not assign a `ReportSpec` step to a Modal runner.
- Any new code that lives on the Modal import path must keep
  `pipelines_v2.reporting` imports lazy (inside the function that needs
  them), so capture/analysis containers do not pull matplotlib at
  container import time.

## Runner Layer

### `ExecutionPlan`

What it is:
- preflight result for one spec on one runner

Fields/properties:
- `spec_kind`
- `required_capabilities`
- `engine_capabilities`
- `artifact_kinds`
- `checks`
- `errors`
- `warnings`
- `missing_capabilities`
- `valid`

Methods:
- `estimated_artifacts()`
- `validate()`
  - raises on missing capabilities or preflight errors

### `LocalResources`

Current field:
- `device`

### `LocalRunner`

What it does:
- executes capture or artifact-bound specs locally

Fields:
- `resources`
- `artifacts`
- `catalog`

Methods:
- `identity()`
- `plan(spec)`
- `run(spec)`

Behavior:
- capture runs through `spec.engine.capture(...)`
- analysis/report runs through artifact-bound execution helpers

### `LocalRunnerSpec`

What it is:
- serializable definition of one local execution profile

Fields:
- `resources`
- `artifacts`
- `catalog`

Methods:
- `to_runner()`
- `to_dict()`
- `from_dict(payload)`

### `ModalVolumeMount`

Fields:
- `name`
- `mount_path`
- `create_if_missing`
- `commit_on_success`

### `ModalSecret`

What it is:
- binding from Modal secret config to env vars required by a runtime

Methods:
- `ModalSecret.from_env_var(env_var, secret_name=None)`
- `to_dict()`

### `ModalResources`

Fields:
- `gpu`
- `cpu`
- `memory_mb`
- `timeout_seconds`
- `secrets`
- `volumes`

### `ModalRunner`

What it does:
- executes capture or artifact-bound specs on Modal

Fields:
- `resources`
- `artifacts`
- `catalog`

Methods:
- `identity()`
- `plan(spec)`
- `run(spec)`

Behavior:
- validates required secret bindings at plan time
- ships one spec to a remote runtime

### `ModalRunnerSpec`

What it is:
- serializable definition of one Modal execution profile

Fields:
- `resources`
- `artifacts`
- `catalog`

Methods:
- `to_runner()`
- `to_dict()`
- `from_dict(payload)`

Use when:
- a Python workflow file should fully describe its named runners via `build_runner_specs()`
- you want one checked-in workflow definition to materialize concrete runners without CLI-only glue

## Storage and Artifacts

### Stores

#### `ArtifactStore`

Protocol methods:
- `make_artifact_dir(artifact_id)`
- `has_local_artifact(artifact_id)`
- `write_safetensors(...)`
- `write_json(...)`
- `has_local_ref(ref)`
- `read_safetensors_ref(ref)`
- `read_json_ref(ref)`
- `localize(artifact_id)`
- `estimate_download_bytes(ref)`
- `validate_transfer(bytes=..., label=...)`

#### `LocalArtifactStore`

Use when:
- artifacts live on local disk

Methods:
- `identity()`
- all `ArtifactStore` methods

#### `ModalVolumeStore`

Use when:
- artifacts live in a Modal volume

Important fields:
- `name`
- `root`
- `local_cache_root`
- `transfer_policy`

Behavior:
- writes inside the mounted volume path
- localizes artifacts into a cache when needed
- blocks unexpectedly large downloads unless overridden

### Catalogs

#### `Catalog`

Protocol methods:
- `record_artifact(manifest)`
- `load_artifact(artifact_id)`
- `record_workflow_run(record)`
- `load_workflow_run(run_id)`
- `list_workflow_runs(...)`
- `record_workflow_step(record)`
- `list_workflow_steps(run_id)`
- `find_latest_reusable_step(step_name=..., step_semantic_hash=..., input_artifact_refs=...)`

#### `CompositeCatalog`

What it does:
- mirrors catalog writes into multiple backends
- reads from the first backend that has the requested record
- used by the CLI to combine the default local run registry with an external catalog such as Postgres

#### `FileCatalog`

What it does:
- writes one manifest JSON per artifact to a local directory
- also stores workflow runs and workflow step records
- does not currently implement alias resolution or spec snapshot storage

#### `PostgresCatalog`

What it does:
- upserts artifact manifests into `pipelines_v2_artifacts`
- stores workflow runs and workflow step records in Postgres
- does not currently implement alias resolution or spec snapshot storage

#### `NullCatalog`

What it does:
- no-op catalog

### `ArtifactManifest`

What it is:
- canonical summary of one produced artifact

Important fields:
- `operation_spec_hash`
- `operation_semantic_hash`
- `input_artifact_refs`
- `example_coverage`
- `storage_refs`
- `workflow_context`

Methods:
- `to_dict()`
- `from_dict(payload)`

### `CaptureArtifact`

What it is:
- typed handle over a capture manifest + store

Methods:
- `manifest()`
- `manifest_dict()`
- `localize()`
- `estimated_local_transfer_bytes()`
- `feature(name)`
- `load_feature(name)`
- `label(name)`
- `load_label(name)`
- `generations()`

### `OperationArtifact`

What it is:
- typed handle over a non-capture artifact

Methods:
- `manifest()`
- `manifest_dict()`
- `localize()`
- `estimated_local_transfer_bytes()`
- `result()`
- `summary()`
- `feature(name)`
- `load_feature(name)`
- `label(name)`
- `load_label(name)`
- `uri`

### Feature and Label Refs

#### `FeatureRef`

Methods:
- `load()`
- `estimated_transfer_bytes()`
- `layer(layer_index)`

#### `FeatureLayerRef`

Methods:
- `load()`

#### `ArtifactLabelRef`

Methods:
- `load()`
- `resolve_values()`
- `equals(value)`
- `estimated_transfer_bytes()`

## Workflow Layer

### `StepRef`

What it is:
- reference to a prior workflow step result

Methods:
- `feature(name)`
- `label(name)`
- `to_dict()`
- `from_dict(payload)`

### `StepFeatureRef`

What it is:
- reference to one named feature produced by a prior step

Methods:
- `layer(layer_index)`
- `to_dict()`
- `from_dict(payload)`

### `StepLabelRef`

What it is:
- reference to one named label produced by a prior step

Methods:
- `equals(value)`
- `to_dict()`
- `from_dict(payload)`

### `WorkflowStep`

What it is:
- one named step in a workflow

Fields:
- `name`
- `runner`
- `spec`
- `depends_on`

Methods:
- `resolved_depends_on()`
- `to_dict()`
- `from_dict(payload)`

### `WorkflowSpec`

What it is:
- a DAG of steps

Fields:
- `steps`
- `name`
- `schema_version`

Methods:
- `to_dict()`
- `from_dict(payload)`
- `ordered_steps()`

Behavior:
- validates duplicate names
- validates missing dependencies
- validates dependency cycles
- automatically infers dependencies from embedded `StepRef`/`StepFeatureRef`/`StepLabelRef`

### `WorkflowPlan`

What it is:
- plan output for a whole workflow

### `WorkflowResult`

What it is:
- result bundle for a finished workflow

Fields:
- `run_id`
- `workflow_hash`
- `step_results`

Methods:
- `step(name)`

### `WorkflowOrchestrator`

What it does:
- coordinates a workflow over named runners

Fields:
- `runners`
- `max_parallelism`

Methods:
- `plan(workflow)`
- `run(workflow, *, resume_run_id=None, reuse_completed=False, reuse_from_run_id=None, force_rerun_steps=frozenset(), parent_run_id=None)`

Behavior:
- resolves step refs before execution
- can run independent ready steps in parallel
- plan-time validation rejects artifact-bound analysis steps that mix feature/text rows from one dataset with label/group refs from another dataset unless `rows=...` is declared explicitly
- persists workflow runs and workflow step records when runners share one non-null catalog
- `resume_run_id=...` reloads completed prior step artifacts from the shared catalog
- `reuse_completed=True` reuses latest completed step artifacts whose semantic lineage matches
- `reuse_from_run_id=...` reuses matching completed step artifacts from one specific prior run
- `force_rerun_steps={...}` prevents those named steps from being reused
- the intended pattern is:
  - `resume`: keep the same `run_id` and continue a failed/interrupted run
  - `rerun-step` / `rerun-from-step`: create a new run, reuse upstream artifacts from a prior run, and force the selected step set to execute again
- resume/reuse require every participating runner to point at the same catalog identity

### `WorkflowRunRecord`

What it is:
- persisted record for one workflow execution attempt

Important fields:
- `run_id`
- `workflow_name`
- `workflow_hash`
- `workflow_spec_hash`
- `workflow_payload`
- `status`
- `started_at`
- `parent_run_id`
- `finished_at`
- `error`

### `WorkflowStepRecord`

What it is:
- persisted record for one workflow step within one workflow run

Important fields:
- `run_id`
- `workflow_hash`
- `workflow_step_key`
- `step_name`
- `step_index`
- `runner`
- `status`
- `step_semantic_hash`
- `step_spec_hash`
- `input_artifact_refs`
- `artifact_id`
- `artifact_kind`
- `started_at`
- `finished_at`
- `reused_from_run_id`
- `reused_from_artifact_id`

### `WorkflowStepContext`

What it is:
- workflow provenance attached to one concrete runner execution and written into the artifact manifest

Important fields:
- `run_id`
- `workflow_name`
- `workflow_hash`
- `workflow_spec_hash`
- `step_name`
- `step_index`
- `runner`
- `step_semantic_hash`
- `step_spec_hash`

## CLI Surface

Current entrypoint:

```bash
uv run python -m pipelines_v2.cli workflow plan --file path/to/workflow.py
uv run python -m pipelines_v2.cli workflow run --file path/to/workflow.py
uv run python -m pipelines_v2.cli workflow runs --file path/to/workflow.py
uv run python -m pipelines_v2.cli workflow show --run-id wr_...
uv run python -m pipelines_v2.cli workflow resume --file path/to/workflow.py --latest-failed
uv run python -m pipelines_v2.cli workflow rerun-step --file path/to/workflow.py --run-id wr_... --step report
uv run python -m pipelines_v2.cli workflow rerun-from-step --file path/to/workflow.py --run-id wr_... --step capture_prompt_eos_router
```

Current scope:
- the CLI currently loads Python workflow definition files
- JSON workflow loading through the CLI is not implemented yet, even though many individual specs are serializable

Workflow file contract:
- `build_dataset() -> Dataset`
- `build_workflow(dataset: Dataset | None = None) -> WorkflowSpec`
- optional `build_runner_specs() -> dict[str, RunnerSpec]`

If `build_runner_specs()` is present:
- the CLI materializes those runner specs directly
- CLI resource flags become unnecessary for that file

If `build_runner_specs()` is absent:
- the CLI builds conventional `capture_gpu`, `analysis_cpu`, and optional `report_local` runners from flags

Default local run tracking:
- the CLI always mirrors workflow state into a local `FileCatalog`
- default location:
  - `~/.xenon/pipelines_v2/catalog`
- override with:
  - `--local-catalog-root /path/to/catalog`
- if an external catalog is also configured, the CLI uses a `CompositeCatalog` so local tracking and external provenance both update together

Workspace config:
- the CLI and dashboard also read repo-root `xenon.toml` when present
- shared defaults such as `catalog_postgres_env` belong there
- CLI flags override `xenon.toml`
- workflow runner specs that already set a catalog override the workspace default
- dashboard defaults such as `static_dir` can also live there

Important run flags:
- `--resume-run-id <run_id>`
- `--reuse-completed`
- `--catalog-postgres-env <ENV_VAR>`
- `--local-catalog-root <PATH>`

Additional workflow commands:
- `workflow runs`
  - list locally tracked runs
- `workflow show --run-id <run_id>`
  - show one persisted run plus its step records
- `workflow resume`
  - resume a failed run in place
- `workflow rerun-step --run-id <run_id> --step <name>`
  - create a new run that reuses upstream artifacts from the source run and reruns only the named step
- `workflow rerun-from-step --run-id <run_id> --step <name>`
  - create a new run that reuses upstream artifacts from the source run and reruns the named step plus downstream dependents

## Practical Patterns

### 1. Hardcoded local smoke

```python
dataset = Dataset.from_examples(
    [
        Example(key="a", prompt="hello", labels={"class": "positive"}),
        Example(key="b", prompt="world", labels={"class": "negative"}),
    ]
)

capture = LocalRunner().run(
    CaptureSpec(
        engine=ToyEngine(),
        dataset=dataset,
        sites=[ResidualSite(name="resid_last", site="resid_post", layers=[0])],
    )
)

probe = LocalRunner().run(
    ProbeSpec(
        feature=capture.feature("resid_last"),
        labels=dataset.labels("class"),
        tokens=TokenSelector.full_sequence(),
        pooling=TokenPooling.mean(),
    )
)
```

### 2. Deferred Postgres capture on Modal

```python
dataset = Dataset.from_postgres(
    source=PostgresSource.from_env("XENON_DATABASE_URL"),
    table="conflict_probe_examples_v3",
    prompt_column="prompt_messages_json",
    example_key_column="example_id",
    label_columns=["conflict_present", "lexical_split"],
)

runner = ModalRunner(
    resources=ModalResources(
        gpu="A100-80GB",
        secrets=(ModalSecret.from_env_var("XENON_DATABASE_URL", secret_name="xenon-db"),),
    ),
    artifacts=ModalVolumeStore(name="xenon-data", root="/data/artifacts/example_run"),
)

artifact = runner.run(
    CaptureSpec(
        engine=VLLMEngine(model_id="/models/Qwen/Qwen3-30B-A3B", max_num_seqs=2, enforce_eager=False),
        dataset=dataset,
        sites=[ResidualSite(name="resid_last", site="resid_post", layers=[12, 24, 36], tokens=TokenSelector.last())],
    )
)
```

### 3. Multi-step workflow

```python
workflow = WorkflowSpec(
    name="capture_then_probe",
    steps=(
        WorkflowStep(
            name="capture",
            runner="gpu",
            spec=CaptureSpec(...),
        ),
        WorkflowStep(
            name="probe",
            runner="cpu",
            spec=ProbeSpec(
                feature=StepRef("capture").feature("resid_last"),
                labels=dataset.labels("conflict_present"),
            ),
        ),
    ),
)

result = WorkflowOrchestrator({"gpu": capture_runner, "cpu": analysis_runner}).run(workflow)
probe_artifact = result.step("probe")
```

## Current Limits To Know About

- `ActivationPatchSpec` is not executable yet.
- `BasisSpec` only supports PCA today.
- `LabelPredicate` only supports equality today.
- artifact-bound analysis supports residual and MoE router features, but broader capture families still need dedicated ops.
- `VLLMEngine` currently requires `enable_prefix_caching=False` for MoE routing capture, because cache hits can skip prompt execution and leave router rows incomplete.
- large remote reads are blocked by default unless the store transfer policy allows them.
- workflow resume/reuse require one shared non-null catalog across all runners in the workflow.
- step reruns now exist through the CLI, but there is not yet a richer workflow-edit API for mutating a checked-in workflow spec and automatically planning invalidation from the diff.
- catalog aliases and spec snapshots described in `ARCH2` are not implemented yet.
- the orchestrator supports dependency-aware parallelism and failure cancellation, but not a configurable retry policy yet.
