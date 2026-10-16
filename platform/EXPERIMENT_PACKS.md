# Experiment packs and evidence (C05 v1)

`pipelines_v2.experiments.ExperimentPack` is immutable versioned metadata around
an ordinary `WorkflowSpec`. There is no pack scheduler. Load `pack.json` inside
`build_workflow`, then call `pack.validate_workflow(workflow, phase_dir=...)`
before returning it. The existing plan/run/runs/show/resume/report commands
continue to work. See `workspaces/toy_pack/workflow.py` and its checked-in pack.

`XENON_EXPERIMENT_PRESET` selects `setup`, `smoke`, `reference`, or `scale`;
the default is smoke. All four have separate enabled flags, workflow hashes,
finite limits, paper-setting notes and qualification refs. A builder can use
the same selector to construct different ordinary specs. Reference metadata
must state matched settings and deviations. Smoke is engineering evidence.
Scale is disabled until explicitly enabled and an injected qualification
verifier checks actual upstream C02 records for the built envelope. The pack
cannot grant qualification. Limits feed admission; they are not measurements
or a new scheduling policy.

Source locks hash the exact relative source-file bytes. `workflow_locks()`
returns model configuration and dataset recipe digests to place in the pack
named lock registries. Each preset has `model_bindings` and `data_bindings`
mapping workflow paths to registry names, so smoke and reference may use
different data or model configurations. `validate_workflow` checks the selected
bindings and the full workflow semantic hash. These are
recipe locks: deferred data contents and resolved checkpoint revisions must
still be supplied by C01 provenance and C02 admission. The validator does not
fetch remote data. Expected outputs map every step name to its artifact kind.
Unsupported versions, unknown fields, malformed budgets, unbound outputs,
source changes and recipe changes fail before the workflow is returned.

`split_policy.digest` is a study-supplied commitment to the split assignment
procedure or manifest. The study must document and audit what it hashes; this
validator checks its format, not the assignment procedure or underlying rows.

Evidence uses `EvidenceRecord.from_dict(record, pack=pack)`. A record declares
question/hypothesis, preset, metric/evaluator identity, split, controls, effect,
uncertainty, methodology level, limitations and two separate statuses:

- `engineering_status`: tested, qualified, pending, failed.
- `scientific_verdict`: supported, refuted, inconclusive, not_tested.

A refuted or inconclusive result can be a completed engineering experiment.
Scientific support and refutation require a non-smoke run, declared required
controls and an observed held-out split. Explicit engineering fixture refutations
remain engineering-only; support always needs a level above engineering. These structural checks do not substitute
for scientific review or automatically promote a claim up the evidence ladder.
The held-out link only verifies that the selected artifact field equals the
declared split name. It does not establish disjoint membership, prevent leakage,
or verify that the reported metric was computed on that split; the study's
evaluator and review must establish those properties.

Each link contains immutable artifact/run IDs, C01 scientific identity and an
exact RFC6901 selector `{domain, pointer, expected}`. Domains are `manifest`
or bounded JSON `result`. Example/site selectors can point into manifest
coverage/features; graph selectors point to exact node/edge fields in bounded
results. Values are checked, not accepted as free-text links. Effect references
one numeric result selector; control arms reference distinct observed links.
A supported verdict requires `split_link` pointing to the observed held-out split,
and numeric uncertainty requires its own matching link; unavailable uncertainty
uses null value/link with an explicit reason.

Pass `ReportSpec(evidence={"pack": pack.to_dict(), "records": [...]}, inputs=(...))`
with explicit direct artifact handles. The report validates IDs, provenance,
selectors and sizes before rendering. No capture tensors or generation sets
are downloaded. Each result is capped at 256 KiB, combined downloads/report at
1 MiB, each record at 128 KiB, and reports at 100 records. Unknown result sizes
fail before reading. Reports show controls, verdicts, provenance limitations and
diagnostics. Legacy artifacts remain readable elsewhere but cannot supply an
unverified C01 evidence link. Qualification status is a record, never a grant.

The report's evidence field is omitted by default from serialization and hashes,
preserving existing report specs. Report modules stay lazy on execution paths.
PHASE.md starts with Premise/Artifacts only; populate results from actual runs.
