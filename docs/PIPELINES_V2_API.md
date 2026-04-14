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
  - `DirectionSpec`
  - `BasisSpec`
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
- `extra`

Important methods:
- `identity()`
- `capabilities()`
- `runtime_spec()`
- `planning_errors(spec)`
  - currently rejects MoE routing capture with `max_num_seqs > 1`
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
- `selectivity` is computed against a shuffled-label control even if
  `shuffled_label` is not requested as an exposed baseline metric

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
- `method`
- `by`
- `layers`
- `components`
- `tokens`
- `pooling`

Current implementation:
- only `method="pca"`

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

Protocol method:
- `record_artifact(manifest)`

#### `FileCatalog`

What it does:
- writes one manifest JSON per artifact to a local directory

#### `PostgresCatalog`

What it does:
- upserts manifests into `pipelines_v2_artifacts`

#### `NullCatalog`

What it does:
- no-op catalog

### `ArtifactManifest`

What it is:
- canonical summary of one produced artifact

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
- `run(workflow)`

Behavior:
- resolves step refs before execution
- can run independent ready steps in parallel

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
- artifact-bound analysis currently expects residual-style features.
- `VLLMEngine` currently rejects MoE routing capture with `max_num_seqs > 1`.
- large remote reads are blocked by default unless the store transfer policy allows them.
