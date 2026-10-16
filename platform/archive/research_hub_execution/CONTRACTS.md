# Shared execution contracts

These are required v1 field meanings and invariants for the planned implementation. They are **not APIs that already exist**. Producers may choose idiomatic Python names/module layout while preserving serialized meaning. Register codecs through T24. Additive optional fields must preserve legacy defaults/hashes; scientific changes explicitly invalidate reuse. Unknown major versions fail early.

## Ownership and consumers

| Contract | Producer | Main consumers |
| --- | --- | --- |
| C01 Provenance and identity | T01 | All later tasks |
| C02 Admission and qualification | T02 static/readiness; T06 measured | All model and external-tool tasks |
| C03 Durable completion and checkpoints | T05 | T06, T08–T20, T23 |
| C04 Sites, capabilities and hooks | T24 codecs; T04 text session; T12 native | T13–T20 |
| C05 Packs and evidence | T07 | All study packs, T21, T22 |
| C06 External attribution results | T10 | T11, T14, T23 |
| C07 Graphs and route tests | T14 | T20, T21 |
| C08 Image and latent data | T25 | T15, T16, T17, T21 |
| C09 Dictionary and model alignment | T18 dictionary; T19 paired extension | T19, T20, T21 |

A producer ships a minimal valid fixture, invalid fixtures, round-trip codec and validator with its implementation. Consumers test against those fixtures. Changing a required field or its meaning requires a version change plus affected consumer tests in the same integration change. Schemas live with code; this document is the semantic contract, not a second competing runtime schema.

## C01 Provenance and identity

`provenance_v1` resides in manifest metadata, with required fields:

- `source`: repository ID, HEAD, relevant dirty-content digest, mounted-source digest; never secrets or the entire home directory.
- `runtime`: runtime family, lock digest, Python/platform, installed-package digest, image identity (nullable for local with explicit reason).
- `model_components[]`: role, repository/checkpoint ID, requested revision, resolved immutable revision, file-set digest. Tokenizer, scheduler, preprocessing and chat-template are explicit components or versioned configs.
- `data`: dataset/content digest, ordered example-key digest, split digest, tokenization/preprocessing identity; unavailable material is identified, not replaced by a path hash.
- `numerics`: dtype, attention implementation, device/backend, determinism/compile settings. Record matmul/TF32 settings where relevant.
- `methods`: operation version, hook schema/adapter version, evaluator version/config; `seeds`: named seeds/RNG descriptors.
- `scientific_identity`: canonical hash over relevant resolved inputs/config; `worker_compatibility`: separate descriptor/hash for safe process reuse.

Paths, labels and timestamps are not scientific inputs. Conservative identity includes runtime differences until a qualified equivalence policy explicitly permits reuse. Artifact content SHA-256 is separate. An old manifest without this payload is readable but marked legacy, not silently granted current qualification. Immutable revisions must resolve before reuse selection. Duplicate model names with different weights are not equal.

## C02 Admission and qualification

`AdmissionCheck = {schema_version, check_id, phase, status, field_path, message, remediation, evidence_ref}`. Phase is static/readiness/qualification. Status is passed/failed/unresolved/not_applicable. An estimate or declaration is never a passed measurement. `ExecutionPlan.valid` retains legacy compatibility but a qualified run also requires all applicable gates.

`WorkloadEnvelope = {model_identity, runtime_lock, hardware_topology, dtype, attention_impl, operation_kind, hook_set, max_input_shape, max_output_shape, batch_size, capture_shape, backward_or_optimizer, compile_config, host_memory_limit, device_memory_limit, timeout}`. Include TP/PP per worker separately from independent shards. Unknown quantities are explicit.

`qualification` is an ordinary cataloged artifact, not a new Postgres table. It includes the envelope hash, source/admission IDs, measured loading peak RSS, execution peak RSS/VRAM, useful-work latency, sample shape/coverage, headroom policy, device/driver/runtime IDs, status and failure diagnostics. Match exact envelopes initially; profile dominance/subsumption is not assumed. Stale/legacy profiles cannot authorize fanout.

Gate flow: static pass → readiness pass → one canary owner → persisted measured pass → remaining shard submission. Failed canary publishes diagnostics, not successful qualification. Canary continues its assigned useful partition. RNG and row allocation prevent duplicate examples. Session compatibility also includes runtime env, hooks, mutable-state reset rules and numerical settings. Idle warm lifetime is explicit and finite; default maximum five minutes between compatible calls.

## C03 Durable completion

Partial journal fields: `schema_version`, `artifact_id`, `attempt_id`, `operation_semantic_hash`, `provenance_hash`, `expected_coverage`, `completed_units[]`, `checkpoint_refs`, `state=incomplete`, `resume_semantics`.

Each completed unit carries stable example/image/method/seed key, shard identity, token/site coverage, payload refs, byte counts and checksums. Expected coverage must distinguish legitimate zero outputs from omitted records. Capture key includes example, site and selected token; image key includes prompt, initial-noise identity and intervention arm.

Ordered protocol:

1. Write immutable payload to attempt-local temporary name; finish checksum/integrity validation.
2. Atomically publish payload/journal unit; commit mounted volume visibility as needed.
3. Validate full planned coverage and all durable refs.
4. Atomically write the canonical final manifest, explicitly complete in new metadata.
5. Index the final manifest in catalogs; a failed indexing step is retriable without model work.

**No partial result writes the canonical final manifest.** Resume never treats directory existence as completion. A final manifest left unindexed can be repaired; a corrupt payload cannot be made valid by reindexing. Handle concurrent attempts with scoped ownership/idempotence and unique temporary filenames. No assumption of cross-host atomic filesystem rename without the store's durability boundary.

Learned-method checkpoints additionally include weights/masks, optimizer, schedule, sampler/order position, relevant RNG and runtime identity. Exact continuation requires an executed oracle; otherwise declare restart-at-unit behavior. Diffusion starts with image-boundary recovery.

## C04 Sites, capabilities and hooks

New site payloads have `kind`, `schema_version`, `model_identity`, `adapter_version`, `module_path`, `layer`, `quantity`, `selection`, `axes`, `read_or_write`, `phase`, `dtype`. Physical module path is diagnostic; quantity/axes/adapter define meaning. Existing untagged residual/router payloads keep their old interpretation.

Quantities are explicit: residual pre/post block, gate preactivation, gate activation, up/value, product before down projection, attention-head output before output projection, or expert-local equivalent. `selection` includes example/token/head/neuron/expert/spatial selectors as applicable. Router-selected token coverage and mixture weighting are recorded; absent routing is not measured zero. Image/timestep additions use C08.

Hooks must preserve autograd for advertised differentiable operations, apply only selected coordinates and clean up on all exits. Model weights are frozen unless a separately scoped fitting operation declares otherwise. No-op, replacement and gradient oracles are required. Capabilities are granular per adapter/device/operation; merely having an enum entry does not establish support. Do not add universal training to the existing Engine protocol.

Patch pairing includes donor/recipient keys, token map, site identities, direction/sign, baseline/source orientation and operator semantics. A pooled mean is distinct from a full-state tokenwise replacement. Unsupported operators fail at plan, never silently degrade.

The initial Torch residual operator set is InterchangePatch, AddDirectionPatch and the new finite-valued ScalePatch. Scale one is identity and scale zero ablation; donor controls use explicit interchange pairings. T04 owns optional target-token logprob readout: last unpadded prompt position before first generation, full-vocabulary log-softmax normalization, retained values keyed by declared token IDs. Omitted targets preserve legacy rows/serialization/hashes. LOGPROBS capability alone is not an implementation: nonempty target requests require the new granular TARGET_LOGPROBS capability, initially Torch-only. Every engine also declares supported patch operators; unsupported operators fail at plan, including ScalePatch on current vLLM/Toy.

## C05 Packs and evidence

One `pack.json` per phase contains `schema_version`, study/phase IDs, source paper links, one `workflow.py` path, source/model/data locks, question/hypothesis IDs, preset definitions, limits, split policy, metrics, controls, qualification requirements and expected output kinds. `workflow.py` remains executable truth; metadata is validated against its built WorkflowSpec. Environment selector `XENON_EXPERIMENT_PRESET` chooses smoke/reference/scale; default smoke. No independent pack scheduler.

Evidence record contains run/artifact IDs, exact selectors for examples/sites/graph elements, metric/evaluator identity, split, control arms, effect/uncertainty, methodology evidence level, limitations and verdict. `engineering_status` (tested/qualified/pending/failed) is separate from `scientific_verdict` (supported/refuted/inconclusive/not_tested). A supported claim requires the planned controls and held-out evidence; success of code alone earns no causal claim.

PHASE.md follows the repository template. Premise/Artifacts before running; result sections only after evidence. Preserve hypothesis lifecycle rather than inventing statuses for it. Reports accept direct artifact inputs and summaries; data stays in its assigned store. In-repo phase metadata/source are committed; model/data payloads are not.

## C06 External attribution

Tool lock includes repo URL, immutable commit, submodule commits, Python/runtime lock, entrypoint and command template, evaluator identity and licenses/access requirements. External request includes request/schema version, resolved task/model/basis/data/split identity, method/seed, base/source orientation, input-node coverage, fitting/sparsity/optimizer config, budgets and output directory. Shell interpolation is forbidden. Different Python versions communicate through files. T10 owns external_tool runtime dispatch/building: a locked Xenon control interpreter invokes a separately locked tool interpreter/venv, provisions pinned upstream/submodule checkouts, and validates imports independently. An old upstream runtime does not install/import Xenon by force. Local/Modal dispatch and batching must understand this descriptor explicitly.

Normalized result includes node ID mapping, ranking/score values and units, basis, masking convention, sparsity grid and per-point metrics, uncertainty/seed scope, full evaluator config, raw-output refs, effective compute, checkpoint/resume semantics and reproduction label. Validate uniqueness, shapes, finite values and undefined-metric handling. Upstream undefined CPR is represented as undefined with reason, not fabricated zero; nonfinite ordinary scores are invalid.

Use official evaluator for `paper_reproduction`; record setting deviations. A subset smoke is integration evidence, not full reproduction. MAttr node ordering is not an edge contribution. Endpoint tests compare explicit interventions under the same input/basis coverage. Native gradient/evaluator rewrites are optional follow-on unless needed by a scoped native graph task; any native MAttr fitting needs separate parity qualification.

## C07 Attribution graphs

Graph payload: `schema_version`, `graph_id`, `source_run_ids`, target output, model/site identity, algorithm/evaluator config, semantics enum, nodes, edges, approximations, normalization/attention assumptions, pruning, omitted contributions/error nodes, evidence refs.

Node = stable ID + typed site/feature/token/unit identity + optional ranking annotation. Edge = source/target IDs + metric/units/sign + intervention/baseline semantics + score. Semantics distinguish native measured intervention effect, ranking annotation and replacement-model attribution flow. Only compute conservation/omitted-mass quantities when the algorithm defines them. Preserve unknown/error contributions instead of assigning them to features.

Route evidence names proposed subgraph, exact original-model interventions, held-out split, controls, target/collateral effects and verdict. Viewer layout is not scientific evidence. Graph import validates every endpoint and preserves upstream-to-Xenon ID mappings.

## C08 Images and latents

Image input contains `image_id`, content digest, store ref, media type, decoded shape/layout/color space/value range, preprocessing identity and text conditioning where applicable. Logical example identity includes ordered input content hashes and preprocessing, not just path/URI. Keep legacy text prompt hashes unchanged. New sharding uses a discriminated resolved-input identity.

Latent contains content ref/digest, shape/axes/dtype, model component identity and origin. Diffusion adds prompt/negative prompt, initial-noise ref/hash, generator state, scheduler name/config, step index, scheduler timestep value, denoising window and CFG state/branch. Branch values: conditional/unconditional/both; CFG-disabled input validates appropriate choices.

Spatial selectors state input/preprocessed/latent/feature coordinates, axis order, mask shape and interpolation. Image boundaries are default recovery units. Summaries/thumbnails have independent byte limits and link to originals. T25 owns these codecs; T15 and T16 own their adapter semantics.

## C09 Dictionaries and alignment

Dictionary metadata contains type (SAE/crosscoder/transcoder), tool/checkpoint hashes, input/output model/site identities, architecture/activation/normalization, model-loading transformations (folding/centering/bias conventions), validated coordinate map, feature dimensions, encoder/decoder/bias refs, training provenance if known, and fidelity metrics. Missing provenance is explicit; dimensions alone cannot establish checkpoint compatibility.

Paired alignment contains ordered model identities, row/document/token maps, tokenizer/template settings, chosen sites/layers and split hashes. Mismatched tokenization requires an explicit scientifically justified map, not array truncation. Crosscoder model-side order is meaningful. Transcoder graphs carry reconstruction/error contributions through C07 and distinguish PLT from CLT with source/destination layer mappings. A live PLT qualification cannot certify CLT support.

Feature intervention specifies original-plus-delta or reconstruction replacement and preserves/reports reconstruction residual. Report reconstruction and behavioral changes separately. Feature IDs are local to their checkpoint; cross-model alignment is an evaluated mapping, never integer equality.
