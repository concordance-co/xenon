# Locality

Where data lives, what's allowed local, what's not.

Hard rules first, then reference by data class.

## Hard rules

1. Shared or durable datasets live in Postgres (Neon), not local files.
2. Capture activations and generations live in Modal volumes, not local.
3. Workflow runs and artifacts are indexed through the catalog, not discovered by listing files.
4. Capture and analysis run on Modal. Reports and plots build locally. No exceptions.
5. Large remote reads are blocked by `ModalVolumeStore.transfer_policy` unless explicitly overridden — don't override casually.
6. No data artifacts (jsonl, parquet, safetensors, activation dumps) get committed to the repo.
7. Local files are for smoke fixtures, audit exports, report inputs, and scratch. Nothing else.
8. Synth generation runs that write to Neon must surface to the user first — show the spec, a sample of rows, and the planned row count. Smoke first (typically <100 rows); scale only after user approval and behavioral sanity against the criteria specified.

## Data class reference

For each class: canonical home, how to read it, what's allowed locally.

### Synth datasets

- **Home.** Neon table.
- **Read.** `Dataset.from_postgres(source=PostgresSource.from_env("XENON_NEON_DATABASE_URL"), table=..., ...)`.
- **Local.** Small fixtures for smoke (`Dataset.from_examples(...)`). No committed parquet or jsonl. `Dataset.from_parquet(...)` and `Dataset.from_json(...)` exist for temporary imports and scratch only.

### Real / production traces

- **Home.** Neon table. Uploaded via repeatable workspace-level uploader scripts.
- **Read.** Same as synth datasets.
- **Local.** Same rules. Uploader scripts should live next to the workflow or data manifest they support; they are the only sanctioned path for getting real data into Neon.

### Captured activations

- **Home.** Modal volumes via `ModalVolumeStore`.
- **Read.** Don't download. Reference via `FeatureRef` or `FeatureLayerRef` from the capture artifact. Analysis specs consume these on Modal.
- **Local.** Do not localize full activation tensors. `transfer_policy` blocks by default. If you believe you need activations locally, you are almost certainly doing the wrong analysis locally instead of as an analysis spec on Modal.

### Captured generations

- **Home.** Stored as part of the capture artifact in the Modal volume.
- **Read.** Reference via the artifact. For downstream analysis, use `StepRef` / `GenerationRunSpec` results.
- **Local.** Do not dump full generation sets to local jsonl for inspection. If you need to read generations by eye, sample with a SQL query (generations persisted to Neon) or use a `result.summary()` payload.

### Probes, directions, subspaces, centroids

- **Home.** Artifact outputs in Modal volume, indexed by catalog.
- **Read.** `OperationArtifact` with typed refs. Metadata and small result payloads can come local; full activations cannot.
- **Local.** Probe weights, direction vectors, and result JSON are small and fine to localize for reporting. Activation banks are not.

### Workflow runs and step records

- **Home.** The catalog — shared Postgres catalog plus the local `FileCatalog` mirror at `~/.xenon/pipelines_v2/catalog`.
- **Read.** `workflow runs`, `workflow show`, or direct queries against the catalog. See `INDEXING.md`.
- **Local.** The local mirror already exists and is authoritative for the local machine. Don't build parallel tracking.

### Reports

- **Home.** Local, under the workflow workspace's `reports/` tree.
- **Read.** Locally.
- **Local.** This is the sanctioned local build surface. See `REPORTING.md`.

## What NOT to do locally

- Don't commit jsonl, parquet, safetensors, or activation files.
- Don't download activations for inspection.
- Don't dump full generation results to a local file for reading.
- Don't construct a local file as the primary dataset surface when Neon is available.
- Don't print thousands of rows to stdout or notebooks. Sample or query.
- Don't mount entire Modal volumes to read one artifact. Use the catalog to get the artifact id, then read via the store.
- Don't bypass `transfer_policy` without a reason you can state in one sentence.

## Enforcement surfaces

Code already enforces most of this. When an agent fights an enforcement, the enforcement is right.

- `Dataset.from_postgres(...)` is the canonical dataset constructor for shared data. File-backed constructors are allowed but marked for fixtures and scratch only.
- `ModalVolumeStore.transfer_policy` blocks large downloads.
- `CaptureArtifact.localize()` and `OperationArtifact.localize()` estimate bytes before transferring.
- `ReportSpec` is the only spec family that runs local by design. Any new code on the Modal import path must keep `pipelines_v2.reporting` imports lazy.
- Repo `.gitignore` should cover data artifact extensions. Confirm before committing new data-adjacent directories.

## When in doubt

If you are about to write a local file that looks like production data, stop. Say what you're trying to do and ask whether it should be a Neon query, a workflow spec, or a `result.summary()` read instead.
