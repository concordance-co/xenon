# Scientific provenance and reuse

New artifact manifests carry `metadata.provenance_v1`. It records source bytes,
observed runtime/package identity, immutable model and tokenizer components,
input content/order/splits, effective numerical policy, method versions and seeds.
`ProvenanceV1` and `compare_provenance` are available from `pipelines_v2.api`.

`identity_complete` means the scientific inputs are identified. It does **not**
mean a model/device configuration passed a measured runtime qualification.
The separate `worker_compatibility` descriptor is reserved for process reuse;
its initial reset policy is `not_qualified`. Payload checksums and durable
completion checks are separate from both identities.

## Current behavior

- Existing operation payloads and their spec/semantic hashes are unchanged.
  Scientific identity is an additional reuse gate.
- Legacy manifests remain readable (`manifest.provenance_status == "legacy"`).
  They cannot certify a new identified workload. Unsupported future provenance
  versions fail with a reader-upgrade error.
- Workflow resume, explicit prior-run reuse and latest-completed reuse resolve
  current identity after upstream references are available, then compare it with
  the artifact. Rejection logs identify changed sections and both hashes.
- Unknown facts permit execution, but prevent reuse of final or partial outputs.
  Unresolved remote attempts use separate output directories. Historical outputs
  are retained. Partial generation rows require matching complete provenance.
- Local and ordinary remote model calls hash resolved input contents before
  looking for reusable output. Streaming Postgres capture keeps batched reads;
  it cannot reuse an unresolved data query and records content hashes from the
  actual batches it executes. Data is not fetched locally for Modal admission.
- Individual shards have distinct identities containing their actual input
  subset and partition. Their common execution parent identity is separate: it
  describes the whole workload with producer-observed execution facts, not the
  submitting host's environment. Merge checks
  request/source/input identity, unanimous producer execution facts, unique full
  partition coverage and child-to-parent links. The merger preserves producer
  runtime facts rather than substituting its own process environment. Deferred
  stream assemblies record an explicitly tagged ordered-partition digest.

## Resolution boundaries

The default resolver is offline. It observes an exact installed-package inventory
(`lock_kind=observed_environment`), rather than claiming the current environment
was installed from a particular lockfile. Package inventory includes bundled
distributions declared by installed wheel RECORD files before imports expose
their directories on `sys.path` (`package_inventory_policy=owned-distributions-v1`).
Repeated paths to the same physical provider count once; distinct providers and
real package or bundled-package version changes remain visible. RECORD ownership
scans are cached by file metadata, while package metadata is checked on each
observation. It hashes explicitly mounted source
content, including non-Python computational assets, and records HEAD and relevant
dirty-content digests when Git is available. Standard documentation names/cache
folders are excluded; other mounted assets count. Secret files and symlinked or
excessively broad source mounts cannot silently yield complete identity. Raw
source, prompts, environment secrets and credential-bearing URLs are not embedded
in provenance. Numeric environment settings use a small explicit allowlist.

Built-in Toy model/tokenizer identities and pure label operations are identified.
Real model weights/tokenizers, container images, unsupported numerical policies,
mutable external assets and unobserved file inputs remain explicitly unresolved
until admission supplies immutable facts. In particular, unseeded PCA is not
silently declared deterministic. Previously reusable numerical analysis may run
again until its effective policy has been supplied. Incidental import state, such
as whether a warm process has imported Torch, does not determine identity.

Tests or admission can inject a `LocalRunner.provenance_resolver(spec)` returning
`ProvenanceV1`. The standard resolver also accepts explicit source, runtime,
component, numerical, method and seed observations. Such injection is a trusted
boundary: an asserted fixture is not proof of a live model/runtime measurement.
The worker always observes its own runtime; submitting-host provenance transports
source context and does not substitute host packages for GPU-worker packages.

Paths, artifact IDs, display names, timestamps and requested revision aliases do
not establish scientific equivalence. Artifact inputs are represented by their
upstream scientific identity and logical selectors. Actual prompt bytes are
hashed even when the caller supplies a `prompt_hash`. Requested and resolved model
revisions remain separate; mutable names such as `main` cannot be an immutable
resolved revision. No equivalence across different runtimes is assumed.
