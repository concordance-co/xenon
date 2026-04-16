# ARCH2: Mech Interp Operations API

Status: working draft.

This document defines the desired architecture for a Python-first library for
operationalizing mechanistic interpretability research. The CLI is an optional
entry point over the library, not the primary design surface.

## Purpose

Build a foundation for running mech interp operations at scale and speed:

- activation capture
- attention and KV-cache inspection
- router and expert analysis
- probe construction
- persona vectors and representation directions
- SAE feature extraction and analysis
- CLT/circuit workflows
- activation patching, steering, ablations, and interventions
- statistical testing and evaluation
- reproducible artifact and report generation

The library should make common research operations composable while preserving
enough low-level access to support new methods from papers.

## Principles

1. Python API first.
   Every meaningful operation should be available through importable Python
   objects.

2. Single-operation runners plus workflow orchestration.
   Operation specs are passed to a runner for one execution under one resource
   profile. Multi-step workflows are passed to an orchestrator. A runner
   executes one spec; an orchestrator composes many steps, dependencies, and
   parallel fanout.

3. Capability-aware engines.
   Model runtimes support different hooks. The API should represent this
   explicitly.

4. Concrete integrations are implementations.
   `Source`, `ArtifactStore`, and `Catalog` are abstractions.
   `PostgresSource`, `ModalVolumeStore`, `LocalArtifactStore`, and
   `PostgresCatalog` are concrete implementations.
   `Runner` is the abstraction for single-op execution; `ModalRunner` is one
   implementation. `Orchestrator` is the abstraction for multi-step execution.
   `Engine` is the abstraction; `VLLMEngine` is one implementation.

5. Domain vocabulary beats storage vocabulary.
   Public APIs should talk about examples, cases, labels, features, metrics,
   sites, operations, and artifacts. Storage terms like rows stay behind source
   and storage adapters.

6. Artifacts are first-class.
   Persisted operations produce artifacts with manifests, provenance, storage
   refs, and typed helper methods.

7. Catalog is optional.
   File manifests are the portable contract. A catalog is useful for discovery,
   lineage, and aliases, but local-only workflows should still work.

8. Specs are serializable.
   Operation specs should round-trip to JSON/YAML/TOML so they can be checked
   into project folders and used by agents.

9. Low-level access remains possible.
   High-level operation specs should compile to lower-level engine operations
   rather than hiding the machinery so completely that new methods become
   awkward.

10. Tests are part of the public contract.
   New engines, runners, sources, artifact stores, and catalogs should be able
   to prove compatibility by running shared conformance tests.

## Vocabulary

### Example

One model input/invocation.

An example usually contains:

- an example key
- a prompt or chat transcript
- optional metadata
- optional labels
- optional case membership

An example is the unit that gets passed through a model for generation, capture,
or intervention.

### Case

A grouped experimental unit containing related examples.

Examples:

- clean/corrupt prompt pair
- matched prompt family
- persona contrast set
- prompt variants sharing the same latent scenario

Cases matter for grouped evaluation, leakage prevention, intervention design,
and paired statistical tests.

### Dataset

A collection of examples. Construction of datasets can be a separate layer, but
the operations API consumes a `Dataset` as the source of examples, labels,
cases, and metadata.

Storage may be a database relation, SQL query, parquet file, in-memory list, or
synthetic source.

### Engine

A model runtime backend.

Potential engines:

- vLLM
- SGLang
- transformers
- xLLM
- mlx-lm
- custom partner runtimes

An engine exposes capabilities: generation, logprobs, residual hooks, attention
hooks, KV-cache access, patching, structured outputs, batching, streaming, etc.

### Runner

Execution environment for a single operation submission under one resource
profile.

Examples:

- local Python worker
- Modal GPU worker
- Modal CPU worker
- future cluster/batch worker

The engine says what model/runtime behavior is available. The runner says where
and how one operation runs.

Runners should not be treated as whole-workflow schedulers. If a capture step
needs GPU resources and a probe step needs CPU resources, those are different
runner configurations, even if both happen to use the same underlying platform.

### Orchestrator

Workflow coordinator above runners.

Examples:

- shard one capture across N GPU child runtimes
- wait for capture completion
- fan out 6 probe jobs in parallel on CPU child runtimes
- aggregate outputs into a report step

An orchestrator chooses runners for steps and handles dependencies, sharding,
parallelism, retries, child runtimes, and fan-in/fan-out structure.

### Operation Binding

Operations are not all bound to the same execution shape.

- model-bound operations:
  capture, generation, activation patching, and other model-intervention work
  that requires an engine and often GPU resources
- artifact-bound operations:
  probes, directions, bases, reports, and similar analysis work that consumes
  artifacts, labels, and cases and should default to cheaper runtimes

The API should preserve this distinction. Artifact-bound operations should not
inherit capture-time GPU assumptions by default.

### Source

Connection and read interface for examples, labels, cases, and metadata.

Examples:

- Postgres source
- parquet source
- in-memory source
- generated synthetic source

A source answers "where do examples and labels come from?"

### ArtifactStore

Read/write interface for artifact payloads.

Examples:

- local filesystem store
- Modal volume store
- S3-compatible object store
- database-backed store for small tabular artifacts

An artifact store answers "where do tensors, parquet files, JSON summaries, and
reports live?"

### Catalog

Discovery and lineage interface for specs, artifacts, aliases, and run records.

Implementations may use Postgres, files, object storage metadata, or another
backend. A catalog answers "what artifacts exist, what produced them, and how
do I resolve a name or id?"

The catalog may use the same database connection as a `Source`, but it is a
separate role. The API should not depend on a specific hosted database product.

### CaptureSite

A model-internal location to record.

Examples:

- residual stream
- attention outputs
- attention patterns
- MLP outputs
- MoE routing evidence and expert assignments
- KV cache
- logits
- SAE features
- backend-specific hooks

### MoE Routing

MoE routing is not just "router logits".

For each routed token at each MoE layer, there may be several related but
distinct quantities:

- pre-routing logits over experts
- normalized router probabilities or scores over experts
- selected expert ids after top-k or backend-specific routing
- selected expert weights used to combine expert outputs
- derived expert load/frequency summaries
- routing policy metadata such as top-k, score function, normalization, expert
  count, and any backend-specific constraints

The API should preserve this distinction. A backend may expose only some of
these quantities. If selected experts or selected weights are recomputed from
logits rather than observed directly from the backend, the artifact manifest
must say so.

Useful routing record types:

```python
RoutingRecord.gate_logits(...)
RoutingRecord.gate_probs(...)
RoutingRecord.routing_decisions(...)  # observed selected experts/weights
RoutingRecord.topk_from_gate(...)     # derived from gate logits/probs
RoutingRecord.expert_load(...)
```

### TokenSelector

A description of which token positions or spans an operation applies to.

Examples:

- last token
- full sequence
- fixed slice
- named prompt section
- regex-delimited section
- custom selector

### Feature

A captured or derived representation used by downstream analysis.

Examples:

- residual vectors
- attention matrices
- MoE routing distributions
- probe-ready feature matrices
- PCA bases
- persona vectors
- SAE activations
- CLT circuit features

### Label

An outcome or annotation used for supervised or statistical analysis.

### Metric

A measured behavioral, statistical, or mechanistic quantity.

Examples:

- probe accuracy
- selectivity over shuffled labels
- logprob margin
- choice flip rate
- KL divergence
- grouped bootstrap interval

### InterventionSite

A model-internal location to modify.

Capture sites and intervention sites often overlap, but they should be named
separately because reading and writing can have different backend constraints.

### Artifact

A persisted result with a manifest and typed accessors.

Examples:

- capture artifact
- feature artifact
- probe artifact
- basis artifact
- SAE artifact
- intervention artifact
- evaluation artifact
- report artifact

### OperationSpec

A serializable description of a unit of work.

Examples:

- `CaptureSpec`
- `ProbeSpec`
- `DirectionSpec`
- `BasisSpec`
- `SAEEncodeSpec`
- `ActivationPatchSpec`
- `EvaluationSpec`
- `ReportSpec`

## Layer Model

### 1. Engine Runtime Layer

Owns backend-specific model execution.

Responsibilities:

- load model/tokenizer
- apply chat templates
- generate outputs
- compute logprobs
- expose backend capabilities
- run forward passes
- provide hooks where supported

Example:

```python
engine = VLLMEngine(
    model_id="Qwen/Qwen3-30B-A3B",
    max_model_len=8192,
)

caps = engine.capabilities()
```

### 2. Capability And Hook Layer

Represents what an engine can do.

Capability examples:

```python
EngineCapability.GENERATION
EngineCapability.LOGPROBS
EngineCapability.RESIDUAL_CAPTURE
EngineCapability.ATTENTION_CAPTURE
EngineCapability.KV_CACHE_CAPTURE
EngineCapability.MOE_ROUTING_CAPTURE
EngineCapability.ACTIVATION_PATCHING
EngineCapability.REQUEST_SCOPED_INTERVENTIONS
EngineCapability.STRUCTURED_OUTPUT
```

Operations should validate capability requirements before running. Some
capabilities will need parameters, not just booleans. For example, an engine may
support residual capture only at layer outputs, or patching only in eager mode.

### 3. Dataset And Case Layer

Represents model inputs and experimental grouping.

Example:

```python
dataset = Dataset.from_postgres(
    source=PostgresSource.from_env("XENON_DATABASE_URL"),
    table="conflict_examples_v2",
    prompt_column="prompt_messages_json",
    example_key_column="example_id",
    prompt_hash_column="prompt_hash",
    label_columns=["conflict_label"],
    case_key_column="matched_pair_id",
)

small = dataset.select(limit=100)
labels = dataset.labels("conflict_label")
cases = dataset.cases("matched_pair_id")
```

For source-backed datasets, `labels(...)` and `cases(...)` should be
serializable refs, not eager local fetches. If a downstream operation runs on a
remote runtime, dataset materialization and label/case resolution should happen
inside that runtime.

`Dataset` should represent:

- database-backed examples
- local parquet examples
- in-memory examples
- synthetic/generated examples
- grouped case structures
- multiple labels per example

### 4. Storage And Artifact Layer

Defines how operation outputs are persisted and reused.

Artifact manifest requirements:

- artifact id
- artifact kind
- operation spec hash
- created time
- engine identity and version
- runner identity
- input artifact refs
- example coverage
- storage refs
- format details
- schema/version
- summary metadata

Example:

```python
cap.manifest()
cap.localize()
cap.feature("resid_post_last")
cap.generations()
```

Storage implementations can include:

- local filesystem
- Modal volume
- S3-compatible object storage
- database tables
- safetensors
- parquet
- JSON manifests

### 5. Execution Profile Layer

Defines where a single operation runs and what resources it gets.

Example:

```python
db = PostgresSource.from_env("XENON_DATABASE_URL")

capture_runner = ModalRunner(
    resources=ModalResources(
        gpu="A100-80GB",
        secrets=(
            ModalSecret.from_env_var("XENON_DATABASE_URL", secret_name="xenon-db"),
        ),
    ),
    artifacts=ModalVolumeStore(name="xenon-data", root="/data/artifacts"),
    catalog=PostgresCatalog(source=db),
)

analysis_runner = ModalRunner(
    resources=ModalResources(
        cpu=6,
        memory_gb=24,
        secrets=(
            ModalSecret.from_env_var("XENON_DATABASE_URL", secret_name="xenon-db"),
        ),
    ),
    artifacts=ModalVolumeStore(name="xenon-data", root="/data/artifacts"),
    catalog=PostgresCatalog(source=db),
)

report_runner = LocalRunner()
```

A runner executes one operation under one resource profile. It should not
implicitly define a whole workflow resource plan.

### 6. Capture And Generation Layer

Records model internals and model outputs for examples.

Example:

```python
cap = capture_runner.run(
    CaptureSpec(
        engine=VLLMEngine(
            model_id="Qwen/Qwen3-30B-A3B",
            max_model_len=8192,
        ),
        dataset=dataset,
        sites=[
            ResidualSite(
                name="resid_post_last",
                site="resid_post",
                layers=[0, 4, 8, 12, 16, 20, 24],
                tokens=TokenSelector.last(),
                storage=TensorStorage(dtype="float16"),
            ),
            MoERoutingSite(
                name="moe_routing_last",
                layers=[0, 4, 8, 12, 16, 20, 24],
                tokens=TokenSelector.last(),
                record=[
                    RoutingRecord.gate_logits(dtype="float16"),
                    RoutingRecord.routing_decisions(required=False),
                    RoutingRecord.topk_from_gate(k=8, include_weights=True),
                    RoutingRecord.expert_load(source="topk_from_gate"),
                ],
            ),
        ],
        generation=GenerationSpec(
            enabled=True,
            max_tokens=512,
            temperature=0.0,
            capture_reasoning=False,
        ),
    )
)
```

Capture should produce a `CaptureArtifact`.

Generation may be part of capture when it shares examples and engine settings,
but it should appear as a typed output inside the artifact rather than being
confused with activation tensors.

### 7. Representation Operation Layer

Turns raw captures into reusable derived representations.

Examples:

```python
persona = analysis_runner.run(
    DirectionSpec(
        feature=cap.feature("resid_post_last"),
        positive=dataset.labels("risk_style").equals("risk_seeking"),
        negative=dataset.labels("risk_style").equals("risk_averse"),
        group_by=dataset.cases("contrast_case_id"),
        layers=[16, 24],
    )
)

basis = analysis_runner.run(
    BasisSpec(
        feature=cap.feature("resid_post_last"),
        method="pca",
        by=dataset.labels("conflict_label"),
        layers=[16, 20, 24],
        components=8,
    )
)
```

These operations are artifact-bound by default. They should not require a model
engine or GPU resources unless the specific method truly needs them.

Future representation operations:

- SAE feature extraction
- SAE feature statistics
- CLT circuit feature extraction
- expert routing feature tables
- attention pattern summaries
- cross-model representation alignment

### 8. Analysis And Statistics Operations

Runs supervised and statistical analyses over features, labels, and cases.

Example:

```python
probe = analysis_runner.run(
    ProbeSpec(
        feature=cap.feature("resid_post_last"),
        labels=dataset.labels("conflict_label"),
        group_by=dataset.cases("matched_pair_id"),
        folds=5,
        metrics=["accuracy", "balanced_accuracy", "selectivity"],
        baselines=["majority", "shuffled_label"],
    )
)
```

Analysis operations should be explicit about:

- feature source
- label source
- grouping/case structure
- split strategy
- baselines
- statistical tests
- output summaries

Candidate operation specs:

```python
ProbeSpec(...)
ExpertRoutingSpec(...)
PCASpec(...)
GroupComparisonSpec(...)
BootstrapSpec(...)
```

### 9. Intervention And Evaluation Layer

Runs model modifications and measures effects.

Example:

```python
patch = capture_runner.run(
    ActivationPatchSpec(
        engine=VLLMEngine(
            model_id="Qwen/Qwen3-30B-A3B",
            max_model_len=8192,
        ),
        dataset=dataset,
        basis=basis,
        site=InterventionSite.residual("resid_post"),
        layers=[16, 20, 24],
        tokens=TokenSelector.section("MARKET"),
        mode="project_out",
        components=["conflict_pc1"],
        strengths=[0.0, 0.5, 1.0, 2.0],
        metrics=[
            Metric.choice_flip_rate(),
            Metric.logprob_margin(labels=["aligned", "conflict"]),
            Metric.kl_divergence(),
        ],
    )
)
```

`TokenSelector.section(...)` assumes the upstream capture stored explicit
`token_sections` metadata, typically via `prompt_metadata_builder=...` on the
capture step.

Intervention families:

- activation patching
- steering vector addition
- projection/removal
- mean/resample ablation
- SAE feature clamp
- attention ablation
- KV-cache edits
- path patching
- request-scoped interventions

### 10. Workflow Orchestration And Reporting Layer

Composes operations and renders artifacts across heterogeneous runners.

Example:

```python
orchestrator = WorkflowOrchestrator(
    runners={
        "capture_gpu": capture_runner,
        "analysis_cpu": analysis_runner,
        "report_local": report_runner,
    }
)

workflow_artifacts = orchestrator.run(
    WorkflowSpec(
        steps=[
            WorkflowStep(
                name="capture",
                runner="capture_gpu",
                spec=CaptureSpec(
                    engine=VLLMEngine(model_id="Qwen/Qwen3-30B-A3B"),
                    dataset=dataset,
                    sites=[ResidualSite(name="resid_post_last", site="resid_post", layers=[16, 24])],
                ),
            ),
            WorkflowStep(
                name="probe_conflict",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("capture").feature("resid_post_last"),
                    labels=dataset.labels("conflict_label"),
                    group_by=dataset.cases("matched_pair_id"),
                ),
            ),
            WorkflowStep(
                name="probe_risk_style",
                runner="analysis_cpu",
                spec=ProbeSpec(
                    feature=StepRef("capture").feature("resid_post_last"),
                    labels=dataset.labels("risk_style"),
                    group_by=dataset.cases("matched_pair_id"),
                ),
            ),
            WorkflowStep(
                name="report",
                runner="report_local",
                spec=ReportSpec(
                    inputs=[
                        StepRef("capture"),
                        StepRef("probe_conflict"),
                        StepRef("probe_risk_style"),
                    ],
                    template="workflow_summary",
                    output_dir="projects/DX_TERMINAL/prompt_confusion/reports",
                ),
            ),
        ]
    )
)
```

The orchestrator owns:

- step dependency ordering
- heterogeneous runner selection
- sharding and child runtime fanout
- parallel analysis branches
- retry/cancellation policy
- artifact lineage between steps

Reports should consume explicit artifact refs. Catalog lookup may help users
resolve aliases, but reports should not depend on implicit global state.

## Testing And Verification Strategy

The test suite should validate the library contract, not incidental
implementation choices. The goal is to make each layer independently testable
while still preserving enough integration coverage to trust real Modal/vLLM
runs.

### Test Surface

1. Spec model tests.
   Every `OperationSpec` should have tests for validation, defaulting,
   serialization, deserialization, stable hashing, migration between schema
   versions, and useful error messages.

2. Adapter contract tests.
   `Engine`, `Runner`, `Orchestrator`, `Source`, `ArtifactStore`, and `Catalog`
   should each have reusable contract tests. A new implementation should be
   able to run the same suite and know which capabilities it actually
   supports.

3. Operation semantic tests.
   Each operation should have tiny deterministic fixtures that check meaning,
   not just execution. Examples:
   - `CaptureSpec` returns tensors with the expected example keys, token
     positions, layers, dtype, and manifest refs.
   - `ProbeSpec` recovers a synthetic linear signal and fails selectivity on
     shuffled labels.
   - `DirectionSpec` has a stable sign convention and separates known positive
     and negative examples.
   - `BasisSpec` produces deterministic components for a seeded synthetic
     matrix.
   - `SAEEncodeSpec` maps activations through a toy SAE with known sparse
     features.
   - `ActivationPatchSpec` changes a toy model outcome only when the target
     site and token selector match.

4. Artifact invariant tests.
   Artifacts should always include manifest schema version, operation spec
   hash, engine identity, runner identity, input refs, example coverage,
   storage refs, shape/dtype metadata, and provenance. Tests should verify that
   typed accessors such as `cap.feature(...)`, `cap.generations()`, and
   `artifact.localize()` resolve real payloads.

5. Dataset and alignment tests.
   Dataset views, labels, and cases should stay aligned by example key and
   prompt hash. Tests should cover filtered datasets, grouped cases, missing
   labels, duplicate prompt hashes, and joins between external labels and
   captured artifacts.

6. CLI adapter tests.
   CLI tests should prove that a serialized spec reaches the same library path
   as Python usage. The CLI should not need separate semantics tests beyond
   argument parsing, config loading, exit codes, and artifact id reporting.

7. Compatibility and regression tests.
   Golden manifests, schema migrations, and small artifact fixtures should make
   it obvious when a change breaks old checked-in workflow specs or old
   artifacts.

8. Workflow orchestration tests.
   Heterogeneous workflow tests should verify dependency ordering, step-level
   runner selection, fanout/fan-in, artifact handoff between steps, retry
   behavior, cancellation, and failure propagation.

### Conformance Test Helpers

There should be a developer-facing test helper package for backend authors and
future integrations:

```python
from pipelines_v2.testing import (
    ArtifactStoreContractSuite,
    CatalogContractSuite,
    EngineContractSuite,
    OrchestratorContractSuite,
    RunnerContractSuite,
    SourceContractSuite,
    assert_artifact_manifest_valid,
    make_toy_dataset,
)
```

These helpers are not the main user API, but they should be stable enough that
a new backend can claim support by running them.

Expected contract examples:

```python
def test_engine_contract(engine_factory):
    EngineContractSuite(engine_factory).run(
        required_capabilities=[
            EngineCapability.GENERATION,
            EngineCapability.LOGPROBS,
        ],
        optional_capabilities=[
            EngineCapability.RESIDUAL_CAPTURE,
            EngineCapability.MOE_ROUTING_CAPTURE,
        ],
    )


def test_runner_contract(runner_factory):
    runner = runner_factory()
    artifact = runner.run(make_toy_capture_spec())

    assert_artifact_manifest_valid(artifact.manifest())
    assert artifact.manifest().runner.kind in {"local", "modal", "batch"}
    assert artifact.localize().exists()
```

### Test Engines And Fixtures

Testing should not require a large model by default.

Useful fixtures:

- `ToyEngine`.
  A deterministic fake engine with known tokens, logits, activations, attention
  maps, and optional MoE routing records. This drives most unit, operation, and
  runner contract tests.

- `GoldenEngine`.
  A fixture engine that reads expected outputs from small checked-in files. This
  is useful for artifact compatibility and schema migration tests.

- `TinyTransformersEngine`.
  A real tiny local model used only where actual tokenizer/model behavior
  matters.

- `TinyVLLMEngine`.
  A real vLLM smoke fixture used only in GPU-capable integration tests.

Fixture datasets should be tiny, versioned, non-sensitive, and designed to
exercise labels, cases, prompt hashes, prompt sections, and malformed examples.

### Modal Integration Tests

Modal should be tested in layers so normal development is fast and cloud tests
remain intentional.

1. Local runner contract tests.
   Always run in PRs. These use `ToyEngine`, `LocalRunner`,
   `LocalArtifactStore`, and `FileCatalog`.

2. Modal transport smoke tests.
   Opt-in or nightly. These use `ModalRunner` with tiny toy/golden operations,
   not a large model. They verify packaging, environment propagation, secrets
   wiring, artifact upload/download, status updates, cancellation behavior, and
   catalog records.

3. Modal GPU engine smoke tests.
   Nightly or release-gated. These use a tiny model through vLLM and a dataset
   with one or two examples. They verify that the engine starts, generation
   works, requested hooks either produce valid artifacts or fail with
   capability errors, and payloads can be localized.

4. Workflow orchestrator smoke tests.
   Opt-in or nightly. These verify that a tiny capture step can run on a GPU
   runner, hand off artifacts to one or more CPU analysis steps, and complete a
   small fanout/fan-in workflow with correct lineage.

5. Production rehearsal tests.
   Manual or scheduled. These run a representative capture/probe/patch/report
   workflow against a small real dataset slice and assert artifact lineage end
   to end.

Modal tests should be explicitly marked and gated:

```python
pytestmark = [
    pytest.mark.modal,
    pytest.mark.integration,
]
```

Suggested gates:

- `XENON_RUN_MODAL_TESTS=1` enables Modal transport tests.
- `XENON_RUN_GPU_TESTS=1` enables GPU/vLLM smoke tests.
- PR CI runs no cloud jobs by default.
- nightly CI runs Postgres integration and Modal transport tests.
- release CI runs the tiny vLLM GPU smoke suite.

Modal tests should have strict limits: tiny datasets, short timeouts, explicit
GPU types, stable artifact prefixes, cleanup tags, and no dependency on private
research datasets.

### Postgres, Catalog, And Storage Integration Tests

`PostgresSource` and `PostgresCatalog` should be tested against an ephemeral
Postgres instance seeded from small fixtures. Tests should prove that:

- source reads do not require a catalog
- catalog writes do not define example data access
- the same connection object may be reused for both roles
- migrations are idempotent
- aliases resolve to immutable artifact ids
- artifact manifests survive round-trip through the catalog

Artifact stores should have the same behavioral tests across local, Modal
volume, and future object-store implementations:

- write/read/delete or tombstone semantics
- atomic manifest writes
- tensor payload integrity
- partial write cleanup
- path traversal prevention
- concurrent writes with unique artifact ids

### Preflight And Dry Runs

Runners and orchestrators should expose non-mutating preflight paths so
expensive jobs can be validated without submitting cloud work. The likely shape
is:

```python
plan = runner.plan(CaptureSpec(...))
plan.require_capabilities()
plan.estimated_artifacts()
plan.validate()

workflow_plan = orchestrator.plan(WorkflowSpec(...))
workflow_plan.validate()
```

Preflight tests should verify capability checks, dataset availability, artifact
store reachability, catalog write permissions, workflow dependencies, and Modal
packaging assumptions.

`runner.run(spec)` remains the single-op execution API. `orchestrator.run(...)`
is the workflow execution API. Preflight exists so tests, agents, and users can
understand what would run before paying for it.

### CI Tiers

Recommended CI split:

- PR: lint, type checks, spec serialization tests, operation semantic tests,
  conformance tests over toy/local implementations, example syntax checks, and
  local artifact/catalog tests.
- Nightly: ephemeral Postgres integration, Modal transport smoke tests, schema
  migration tests, and compatibility tests against golden artifacts.
- Release: tiny vLLM GPU smoke tests, production rehearsal on a tiny dataset
  slice, and backend conformance summaries.

## Public API Shape

Preferred top-level import style:

```python
from pipelines_v2.api import (
    ActivationPatchSpec,
    BasisSpec,
    CaptureSpec,
    Dataset,
    DirectionSpec,
    GenerationSpec,
    MoERoutingSite,
    ModalSecret,
    ModalRunner,
    ModalResources,
    ModalVolumeStore,
    PostgresCatalog,
    PostgresSource,
    ProbeSpec,
    ReportSpec,
    RoutingRecord,
    VLLMEngine,
    WorkflowOrchestrator,
    WorkflowSpec,
    WorkflowStep,
)
```

The API should support three levels:

### Low-Level Engine API

For custom experiments and backend work:

```python
engine.generate(dataset, generation_spec)
engine.capture(dataset, capture_sites)
engine.intervene(dataset, intervention_spec)
```

### Mid-Level Operation API

For most research code:

```python
cap = capture_runner.run(CaptureSpec(engine=VLLMEngine(...), ...))
probe = analysis_runner.run(ProbeSpec(feature=cap.feature("resid_post_last"), labels=labels))
patch = capture_runner.run(ActivationPatchSpec(engine=VLLMEngine(...), ...))
```

The same underlying platform may back both runners, but they should be treated
as different runner configurations if they need different resource profiles.

### High-Level Workflow API

For reusable checked-in heterogeneous workflows:

```python
workflow = WorkflowSpec.from_file("workflow.json")
artifacts = orchestrator.run(workflow)
```

The single-op API remains important, but a workflow/orchestrator layer becomes
necessary once capture and analysis no longer share the same runtime profile.

## CLI Relationship

The CLI should be an adapter over serialized operation specs and workflow specs:

```bash
uv run python -m pipelines_v2.cli workflow plan --file workflow.py
uv run python -m pipelines_v2.cli workflow run --file workflow.py
```

Equivalent Python:

```python
db = PostgresSource.from_env("XENON_DATABASE_URL")
capture_runner = ModalRunner(
    resources=ModalResources(
        gpu="A100-80GB",
        secrets=(
            ModalSecret.from_env_var("XENON_DATABASE_URL", secret_name="xenon-db"),
        ),
    ),
    artifacts=ModalVolumeStore(name="xenon-data", root="/data/artifacts"),
    catalog=PostgresCatalog(source=db),
)
artifact = capture_runner.run(CaptureSpec.from_file("capture.json"))
print(artifact.id)
```

The CLI should not invent a second orchestration model. Single-op commands
should call the runner API; workflow commands should call the orchestrator API.

## Example Review Notes

The usage sketches exposed several design constraints that should be treated as
part of the API, not left implicit:

1. Dataset construction needs an explicit source.
   `Dataset.from_postgres(...)` must receive a `PostgresSource` or equivalent
   connection object. It should not secretly reach into a catalog or process
   environment.

2. Catalog is not source and not storage.
   A catalog indexes manifests, aliases, spec snapshots, and run records. It
   does not define where examples come from or where tensor payloads are stored.

3. Artifact storage must be explicit for remote runners.
   A `ModalRunner` should know which `ArtifactStore` it writes to. Local runners
   may eventually have ergonomic defaults, but remote execution should avoid
   hidden output locations.

4. Runner resources are not engine identity.
   GPU type, CPU count, RAM, Modal limits, and local device are
   runner/resource concerns. The engine describes model/backend behavior and
   capabilities.

5. Model-bound operations should not force artifact-bound operations onto the
   same runner.
   Capture and patch steps may need GPU/model runtimes. Probe, direction,
   basis, and report steps usually should not.

6. Engine ownership should be operation-bound for model-bound steps.
   A workflow may contain both model-bound and artifact-bound steps. Engine
   configuration should therefore attach to the relevant operation or workflow
   step, not implicitly to every runner in the workflow.

7. Dataset views must stay aligned.
   If capture uses `dataset.select(limit=...)`, later labels/cases should come
   from that same selected dataset or from a label set explicitly joined by
   example key and prompt hash.

8. Token scope must be explicit for derived representations.
   A basis or direction fit over full-sequence residuals needs its own
   `TokenSelector`; otherwise it is unclear whether the method uses last token,
   mean pooling, a section, or all token states.

9. MoE routing records must say whether they are observed or derived.
   Gate logits, observed routing decisions, top-k derived from logits, selected
   expert weights, and expert load summaries are different records with
   different causal interpretations.

10. Heterogeneous multi-step execution needs a first-class orchestrator.
   Scheduling child runtimes, step fanout, retries, and cross-step artifact
   wiring should live in a workflow/orchestration layer, not be smuggled into a
   single-op runner API.

11. Prompt rendering still needs a first-class API.
   The examples assume the dataset prompt column plus engine defaults are
   enough. For reproducibility, capture and generation specs likely need an
   explicit prompt/template policy.

## Initial Design Decisions

These need to be decided before implementation:

1. Public package name:
   - A. `pipelines_v2.api`
   - B. `pipelines_v2`
   - C. `xenon`

2. Primary runner API:
   - A. `runner.run(CaptureSpec(...))`
   - B. `runner.capture(CaptureSpec(...))`
   - C. both, with operation-specific convenience methods delegating to
        `runner.run(...)`

3. Dataset labels:
   - A. labels are accessors on `Dataset`
   - B. labels are separate `LabelSet` objects
   - C. both, where `Dataset.labels(...)` returns a `LabelSet`

4. Spec model implementation:
   - A. dataclasses plus explicit validation
   - B. Pydantic models
   - C. plain typed dicts

5. Capability model:
   - A. simple enum capabilities
   - B. typed capability objects with parameters
   - C. start with enum, evolve selected capabilities into typed objects

6. Capture site model:
   - A. high-level helper classes only
   - B. backend-independent hook IR only
   - C. high-level helper classes compile to hook IR

7. Token selectors in first API slice:
   - A. last token and full sequence only
   - B. add fixed slices
   - C. include named prompt sections from the start

8. Catalog scope:
   - A. catalog only stores artifact manifests and aliases
   - B. catalog also stores run status/history
   - C. catalog stores manifests, aliases, run records, and spec snapshots

9. Artifact store defaulting:
   - A. runner always requires an explicit artifact store
   - B. runners choose a conventional default store
   - C. local runner may default; remote runners require explicit stores

10. Test helper surface:
   - A. keep all test helpers private to the repo
   - B. expose a stable `pipelines_v2.testing` conformance package
   - C. expose conformance suites, keep project-specific fixtures private

11. Runner preflight API:
   - A. `runner.validate(spec)`
   - B. `runner.plan(spec)`
   - C. `runner.run(spec, dry_run=True)`

12. Modal test policy:
   - A. all Modal tests are opt-in only
   - B. Modal transport tests run nightly, GPU/vLLM tests run for releases
   - C. Modal transport tests run in PR CI, GPU/vLLM tests run nightly

13. Engine ownership for model-bound ops:
   - A. runner owns engine and resources
   - B. model-bound spec or workflow step owns engine; runner owns resources
   - C. separate model-runner and analysis-runner types

14. Workflow orchestration:
   - A. library supports only single-op execution
   - B. add a first-class `WorkflowOrchestrator`
   - C. keep orchestration in CLI/project code only

15. Parallel workflow fanout:
   - A. orchestrator only supports explicit DAG steps
   - B. orchestrator supports DAG steps plus simple fanout helpers
   - C. hide fanout inside individual operation implementations

## Current Lean

- `Example`, `Case`, `Dataset`, `CaptureSite`, `Feature`, `Label`, `Metric`,
  `Artifact`, `OperationSpec`, `Engine`, `Runner`, `Orchestrator`, `Source`,
  `ArtifactStore`, and `Catalog` are the core public vocabulary.
- Use `Dataset.labels(...) -> LabelSet`, so labels are convenient but still
  become explicit analysis inputs.
- Use `runner.run(spec)` as the primary single-op execution API.
- Start with vLLM as the first engine, but make engine capabilities explicit so
  the API can admit other backends without pretending they are equivalent.
- Use `PostgresSource` for database-backed examples and labels.
- Use `PostgresCatalog` for database-backed manifests, lineage, aliases, and
  run records.
- Keep artifact payload storage explicit with `LocalArtifactStore`,
  `ModalVolumeStore`, or later object-store implementations.
- Treat capture and patch as model-bound operations.
- Treat probe, direction, basis, and report as artifact-bound operations.
- Do not make artifact-bound operations inherit GPU/model runtimes by default.
- Put engine selection on model-bound operations or workflow steps, not on
  every runner in a heterogeneous workflow.
- Add a first-class `WorkflowOrchestrator` for multi-step heterogeneous runs.
- Add a developer-facing `pipelines_v2.testing` conformance suite for engines,
  runners, orchestrators, sources, artifact stores, catalogs, artifact
  manifests, and toy operation fixtures.
- Prefer `runner.plan(spec)` for single-op preflight and `orchestrator.plan(...)`
  for workflow preflight.
- Keep cloud tests out of normal PR flow by default: run local contract tests
  always, Modal transport tests nightly or explicitly, and tiny vLLM GPU smoke
  tests for release gates.
