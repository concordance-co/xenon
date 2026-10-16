# Durable artifacts and interrupted work

New execution results publish `metadata.completion_v1` only after every planned
unit and final payload has passed coverage and SHA-256 verification. The order is
payload write, durable journal, payload visibility, atomic final manifest,
manifest visibility, catalog update. A catalog failure can be repaired by retrying
the same workflow step: a valid orphan manifest is reused without model work.

`finalize_manifest(store, manifest)` implements the final publication boundary;
`validate_completion(manifest, store=store)` verifies identity, coverage, refs and
actual bytes. Runners validate before reuse; the orchestrator also validates
fresh results before declaring a step complete. Typed handles expose
`completion_status` as a declaration, not proof. Legacy manifests remain readable;
`legacy_policy="read_only"` returns false and never authorizes workflow reuse.

`CheckpointJournal` records semantic hash, complete C01 identity, shard layout,
expected units, immutable payload refs, byte counts and checksums. Capture units
contain examples, named sites, layers and selected tokens. Intervention units
contain complete paired cases or unpaired targets, target-token coverage and
explicit skip reasons. Generation preserves distinct example keys even when
prompts repeat and rejects duplicate producer rows. A complete zero-row result
has an explicit zero count; an unfinished journal cannot substitute for it.

On retry, verified completed units are retained, missing units run once, and final
output follows stable input/case order. Unresolved source/model/runtime/data
identity never authorizes continuation. Deferred streamed capture preserves
checksummed progress but restarts a new attempt until its input identity can be
resolved safely. Changed batch boundaries or shard layout reject incompatible
journals rather than guessing how to combine them.

`CheckpointJournal.set_checkpoint(refs)` accepts future optimizer, scheduler,
sampler or RNG payloads. The supported continuation mode is `restart_at_unit`.
Exact optimizer/RNG continuation requires a separately implemented and executed
adapter oracle; this API does not assert bitwise replay.

Local writers use OS-released file locks. Modal writers require the existing
named Dict `xenon-artifact-writers-v1`; readiness must explicitly provision or
verify it. Offline planning never creates it. Ownership is scoped by volume,
root and execution identity. Atomic immutable ownership slots carry call/input
IDs and release markers. Raw attempt IDs are retained and matched to Modal's
canonical input IDs. A killed owner requires an immutable cancellation certificate
in that Dict plus a fresh matching terminal call graph. A graph failure alone is
insufficient: Modal may still be automatically retrying that input. The owning
coordinator calls `cancel_modal_writer_call(call_id, expected_input_id=raw_id)`;
this returns a certificate only after acknowledged
`cancel(terminate_containers=True)` and a terminal graph for the exact call/input.
Unavailable proof raises `IntegrityError` and can be retried within the coordinator's
deadline. A certificate cannot authorize a different call or canonical input;
new or pending graph inputs block recovery. Writer acquisition never initiates
cancellation. Normal completion or Python exception unwinding releases ownership
without a certificate. Elapsed time and generic exceptions never steal ownership.
Interrupted recovery owners use the same rule, so no recovery mutex can orphan.
After 1,000 retained slots, execution fails with a diagnostic requiring a fresh
attempt after old owners are verified stopped. Ownership metadata is not a run
catalog or scheduler.

Atomic payload writes use unique temporary files, file/directory fsync and
replacement. Modal commits payload visibility before publishing journals/finals;
lock acquisition reloads the volume. Full integrity checks for remote refs run
in a small CPU worker on the mounted volume, without downloading activations.
This worker has a 60-second execution limit; client transport deadlines and CPU
merge resource placement are T02 integration responsibilities. A worker timeout
alone does not bound allocation, queueing or client result streaming.

New capture codecs must register a coverage verifier with
`register_capture_coverage_validator(kind, verifier)` when their layout is not the
legacy example/layer/token structure. The verifier must attest expected axes and
keys before any complete manifest can be published.

Validation evidence and remaining live gates are in the T05 receipt. Filesystem
failure tests exercise actual process exits around journal/final boundaries;
fixture tests do not claim live GPU recovery qualification.
