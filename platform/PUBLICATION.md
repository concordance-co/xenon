# Publish an existing artifact

Publication copies existing bytes and indexes a new immutable artifact in the shared catalog and normal FileCatalog mirror. It preserves the original C01 scientific identity, records source lineage, and computes a truthful C03 completion for destination bytes. It never reruns a model or qualifies CPU results for GPU reuse.

Plan a local result with an explicit transfer budget:

```bash
python -m pipelines_v2.storage.publication \
  --manifest /absolute/local/artifacts/ARTIFACT_ID/manifest.json \
  --volume xenon-data --root /data/publications/my-project \
  --max-bytes 1048576 --plan-output /tmp/publication-plan.json
```

Inspect the plan, then add `--publish --receipt /tmp/publication-receipt.json` when publication is authorized. The public API remains `PublicationTarget`, `plan_publication(...)`, and `publish_artifact(...)`. The CLI uses normal workspace environment configuration, including `XENON_WORKSPACE_ROOT`; `--catalog-env` defaults to `XENON_NEON_DATABASE_URL`. Credentials stay on the client.

For an artifact already on Modal, select its shared catalog ID:

```bash
python -m pipelines_v2.storage.publication \
  --artifact-id ARTIFACT_ID \
  --volume xenon-data --root /data/publications/my-project \
  --max-bytes 268435456 --plan-output /tmp/publication-plan.json
```

The primary source and existing shared dependencies must currently use the **same named Modal volume** as the destination. The CPU worker copies their bytes between namespaces on that mount. Different volume names are rejected explicitly. The client may read bounded JSON metadata to validate schemas and rewrite references; it never downloads remote tensor payloads. Even a dry plan verifies remote C03 using a CPU worker when shared sources are selected.

## Eligible data and explicit views

| Artifact | Publication behavior |
| --- | --- |
| Native or replacement attribution graph | Validate C07 and publish typed JSON; graph semantics stay distinct. |
| Official MAttr attribution | Validate C06. `--result-only` strips raw producer base64 and machine-local `effective_compute.argv`, retaining metrics, request identity and raw-file hashes. |
| SAE dictionary; complete crosscoder fit/import | Publish the named typed safetensors dictionary/checkpoint feature and metadata. Weights are file-backed on the client, not copied into Python byte buffers. |
| SAE/crosscoder encode or decode | Require `--result-only`; publish measurements as `*_summary`, excluding per-example codes/reconstructions. |
| SAE/crosscoder feature splice | Publish bounded result rows with dictionary identity and closed input lineage. |
| Known graph-report and neuron-dossier transforms | Validate the known result shape. Graph reports require `--result-only` and preserve a declared source graph link while omitting local HTML/JSON presentation paths. Arbitrary transforms fail. |
| Built-in summary ReportSpec | Require `--result-only`; publish `report_summary`, retaining its pack/records and source references. C05 measurements are reverified against supplied linked artifacts. Rendered files and local runtime/cache presentation descriptors are explicitly omitted. |
| Existing recognition results, generation results, probes, directions, centroids, geometry, bases and subspaces | Retain their existing bounded publication support. |

A partial view is never labeled a complete rendered report. Its distinct artifact kind and `publication_v1.view=result_summary` describe omitted fields/files; the original artifact and original C03 remain unchanged. Its source completion digest and original scientific identity remain in lineage. A known typed view may remove specific presentation diagnostics; arbitrary absolute scientific input strings are never silently stripped.

Captures, activation banks, paired crosscoder captures, unknown kinds and opaque files are not local publication candidates. Bulk activations stay on Modal. A shared capture can remain an upstream dependency: provide its existing catalog ID, verify its C01/C03 on the volume, and reference it in place without uploading or localizing it. Unresolved C01 fails admission; publication never clears unresolved fields or invents a qualification.

## Budgets and dependency closure

`--max-bytes` is required and has no artificial 16 MiB hard ceiling. A larger explicit budget can admit actual pretrained dictionary weights. The exact actual byte inventory must fit it both before transfer and inside the finalizer. Local safetensors stream through Modal upload; remote safetensors use mounted copies. Default additional bounds are configurable:

- `--max-json-bytes`: 16 MiB for individual and aggregate source JSON reads and published JSON. Increase explicitly when projecting an existing large official-output JSON into a small summary.
- `--max-artifacts`: 128 artifacts across the closed dependency graph.
- `--max-metadata-bytes`: 1 MiB serialized request metadata.

Every upstream artifact must be supplied with repeated `--dependency-manifest` or `--dependency-artifact-id` flags. Missing, cyclic and unused dependencies fail. Existing shared dependencies must already be registered with matching C01/C03; publication never flattens them into an independent summary. Selected payloads must be declared in C03. Unknown references, escaping paths, symlinks and undeclared sidecars fail. JSON pointers remain selectors rather than being mistaken for filesystem paths. Tensor bytes remain unchanged; relocated JSON refs and companion sizes receive new hashes. Unstable reference-size cycles fail closed.

Policy version 3 participates in new publication IDs, separating this behavior from earlier publication plans. Repeating the same current plan is idempotent. IDs do not depend on the operator's chosen budget when the published content is identical.

## Durability and reads

Client uploads use content-addressed staging. The finalizer is CPU-only (1 CPU, 512 MiB, 600 seconds; client deadline 660 seconds). It verifies all staged or mounted-source bytes before any final manifest, preserves immutable destinations, commits the volume, finalizes C03, then returns manifests for catalog registration. Shared source verification uses the existing bounded integrity worker. Large transfers that exceed these execution deadlines fail explicitly; they do not relax payload limits or publish unchecked completions.

Interrupted transfer leaves staging only. Conflicting final bytes/manifests are never overwritten. A catalog failure leaves `storage_complete_catalog_pending`; rerun the same command to verify completed data and retry catalog writes without model work. Completed receipts say `published`. Staging remains for retry; cleanup is targeted, never a broad volume deletion. Local audit paths stay in the local plan/receipt and are not uploaded.

Use `python -m pipelines_v2.evidence artifact PUBLISHED_ID --result` for a bounded summary, or `evidence report PUBLISHED_ID --output /tmp/evidence.html`. Large dictionary tensors are consumed on Modal through their normal typed artifact feature refs. C01 equality proves preserved execution identity, not fresh hardware qualification. Publication lineage records original run/hash/step separately from new workflow execution context.

Local-to-shared graph publication, bounded readback and same-volume mounted copying have been exercised without rerunning a model. Use a new publication plan for the actual source and destination; saved proof-run paths are not required.

Republishing preserves the original run/hash/step separately from the copy operation. The evidence reader follows at most 16 prior-publication hops for older copies, and only accepts source workflow lineage when C01 matches the preserved source identity, the preservation flag is true, and the bounded result checksum verifies. Partial views retain their declared summary kind and omissions; original step kinds are used only to validate lineage. This is one-hop C05 verification of reader-produced views, not authentication of arbitrary Python dictionaries. The recorded live proofs used policy 2; policy 3 adds flattened original-workflow lineage without changing scientific identity.
