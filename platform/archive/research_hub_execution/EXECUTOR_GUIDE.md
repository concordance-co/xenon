# Instructions for an implementation model

Implement exactly one assigned task and its verification, using its plan plus this guide and CONTRACTS.md. The user has delegated routine implementation decisions and does not want a plan-review checkpoint. This planning package itself has not launched implementation or GPU jobs. When dispatched to implement, proceed through the task's bounded engineering work and tests; do not ask the user to approve an ordinary class name, test fixture or reversible design choice.

## Start and finish rules

**Current user scope:** implement and verify the research plumbing, stopping once
the capability works end to end. Research-heavy task plans below describe future
experiments, not a requirement to run full reference campaigns now. Use the
smallest real-model example that verifies the new execution path, tensor/hook
semantics, controls, artifact persistence and usable workflow/report. Do not spend
additional time improving scientific effects, tuning hypotheses, running seed
sweeps or polishing research conclusions. Keep reduced runs labeled engineering
smokes; paper reproduction, scientific validation and unrun larger-model cells
remain explicitly unclaimed. Independent code review and relevant correctness
checks still apply.

1. Read AGENTS.md, the assigned task, CONTRACTS.md and predecessor handoffs. Load methodology for study work, operations for storage/reporting and platform for workflow construction. Relevant workflow/causal-evaluation skills apply; the explicitly requested local path overrides old remote-only wording within the documented bounded mode.
2. Verify actual branch/source state against T00. Preserve all unrelated dirty changes. Never reset or bundle them into your patch. Reuse a suitable attached worktree when available; if isolation is needed, establish one from the agreed baseline, including required uncommitted changes through a reviewed scoped patch. Record the exact base. Do not assume remote default HEAD contains local hardening.
3. Check graph start dependencies. You may develop against fixtures while integration dependencies are pending, but cannot mark the task accepted until both sets and all required live gates pass. Do not recursively implement another owner's task to get unblocked.
4. Follow task steps in small coherent changes. Registration, serialization, plan validation, local/remote dispatch, artifact readers and public exports are part of each new operation. Keep optional imports lazy and isolated dependency locks separate. No silent fallback or mock result advertised as real.
5. Run meaningful local tests first. Then named integration gates within the plan's budget. Record exact commands, exit codes, dependency locks, source digest and skips. A fake test does not establish live GPU/model qualification.
6. Obtain an independent capable-model code/contract review before acceptance, focused on correctness, scientific semantics and recovery. Resolve findings; do not require user review of routine changes. Do not send messages outside the authorized collaborator/task scope. Committing/publishing/merging follows the dispatch's repository authorization; this plan alone does not authorize merging unrelated work.
7. Supply the completion receipt below. The integrating model validates it and updates task acceptance in its existing task/PR system; graph.json remains a versioned plan, not a new mutable run database.

## Status and handoff

Task states in receipts: `not_started`, `in_progress`, `code_complete_integration_pending`, `accepted`, `blocked_external`. `accepted` requires start and integration predecessors accepted, tests passed, required evidence present and review findings resolved. Scientific refutation is compatible with accepted engineering. A missing model/GPU/credential is not a passed test. Do not use these statuses to overwrite the repository's hypothesis lifecycle.

Completion receipt (save with the PR or task evidence; only compact pointers go into repository docs):

```json
{
  "task_id": "Txx",
  "status": "code_complete_integration_pending",
  "base_revision": "...",
  "result_revision_or_diff_digest": "...",
  "dependency_receipts": [],
  "contracts_produced": [],
  "contracts_consumed": [],
  "changed_paths": [],
  "checks": [{"command": "...", "environment_lock": "...", "exit_code": 0,
              "result": "passed", "evidence_ref": "..."}],
  "live_evidence": [{"run_id": "...", "app_id": "...", "artifact_ids": []}],
  "qualification_cells": [],
  "scientific_verdict": "not_tested",
  "review": {"reviewer": "...", "resolved_findings": [], "open_findings": []},
  "limitations": [],
  "next_action_if_pending": "..."
}
```

Do not invent run IDs, resolved model commits, passing tests or human annotations. Never put secrets in receipts. Resource metrics and scientific outputs belong in versioned artifacts referenced by the receipt.

## Incremental live GPU gates

User execution steering: prioritize working research capabilities over expanding
the verification framework. Deliver a real model workflow and a readable result
for each capability as early as its required interfaces allow. During development,
run focused checks for the behavior being changed; run the full regression suite
once per integrated capability milestone, not after every small fix. Fold live GPU
checks into useful capture, steering, attribution, or generation runs. Fix concrete
failures before retrying, but defer unrelated hardening instead of delaying the
next usable capability. Required acceptance evidence and independent review still
apply; keep pending gates explicit.

User execution steering: exercise the real Modal/model process throughout development, not only at T22. Run a bounded existing-backend smoke against the preserved baseline first. Repeat affected live checks after registration/provenance integration and after changes to dependency images, resource admission, dispatch, completion/recovery or session reuse. Each new model adapter gets a real single-worker load plus smallest useful operation before dependent research lanes are released. Use the same reviewed source snapshot for the live gate and its receipt; ongoing edits must not change a submitted source mount.

Pure codecs and metadata still use deterministic local tests. GPU-sensitive successors remain integration-pending until their relevant live gate passes. Missing access blocks only the affected gate; it never turns a fixture into GPU evidence. Use the task's finite resource envelope, record app/model/source IDs and diagnostics, and cancel exhausted attempts. Investigate deterministic failures before another cold start. Once T06 exists, its persisted qualification is required before fanout; until then, keep these engineering smokes to one worker without scientific qualification claims.

## Shared-file integration discipline

Own your task's new modules and tests. The integrator serializes changes to these hotspots; same integration lock means no simultaneous edits/merges to the listed shared surface, although independent module/fixture development can continue:

| Lock | Shared surface | Primary sequence |
| --- | --- | --- |
| `core-registration` | core/types, api, engine/operation registries, runtime/artifacts, execution dispatch, site loaders, feature codecs | T24 first; later tasks add registrations through its interfaces |
| `runtime-config` | engine/base runtime specs, runtime/planning/specs, core/config, CLI plan/profile commands | T02 and T03 coordinate; profile and lock fields must compose |
| `artifact-lifecycle` | storage/artifacts/local/modal, workflow records/reuse, manifest readers | T01 provenance before T05 finalization; T09 publication afterward |
| `remote-execution` | runtime/modal/modal_worker/remote_executor, orchestrator batching/fanout | T05 durable changes before T06; T10/T16 external dispatch rebase on both |
| `reporting-cli` | CLI public commands, reporting/workflow/assets, report specs | T07 minimal evidence first; later panels via T21/shared helpers |
| `image-contract` | image inputs, spatial selectors and codecs | T25 first; T15/T16 consume, do not fork. T25 also acquires remote-execution for the shard identity call |

No giant registry refactor beyond what T24's test-only extension proves necessary. Runtime authors provide hooks/interfaces for later tasks, not speculative universal abstractions. Task plans name primary source anchors, not exclusive file lists; small necessary adjacent edits are in scope with review.

## Tests and numerical policy

New test paths in task plans are **deliverables to create**, not commands assumed to work today. Use the task's runtime lock/environment, especially for external tools with another Python version. Default local CI command comes from T00. Avoid collecting GPU tests into default CPU CI; mark and explicitly select network/Modal/integration checks.

FP32 same-implementation CPU tensor/logit oracle starts at `rtol=1e-4, atol=1e-5`; direct exact metadata/ID checks remain exact. Gradient finite differences use deterministic tiny doubles/FP32 with epsilon/error justified by an analytical fixture. For cross-backend/GPU tests, preregister candidate tolerances on calibration fixtures (initial FP32 1e-4/1e-5; FP16/BF16 2e-2/2e-2), verify meaningful task effects exceed noise, and freeze tolerances before held-out scientific comparisons. A tolerance failure triggers investigation or unqualified status, not silent widening. Store actual error distributions and numerical settings; passing coarse tolerances alone cannot validate a tiny causal effect.

Test no-op equivalence, known-effect direct-forward oracles, shape/alignment errors, serialization/legacy behavior, stale reuse, partial recovery and failure cleanup. Do not write tests merely asserting the implementation's own formula; use independent tiny forwards or analytical fixtures. Full regression suite after integration, targeted suite while developing.

## External assets and finite compute

Runtime/model versions must be resolved to immutable commits during implementation; mutable names in these plans are candidate targets, not qualification claims. Use official upstream docs, inspect actual installed signatures, record chosen versions, and keep code pin changes explicit. Do not guess available GPU memory, model permissions, dictionary compatibility or upstream checkpoint contents.

Missing credentials/access: complete unaffected code/fixture tests, record exact failed readiness check, and leave only the dependent gate pending. Do not replace the requested model/tool silently, create service accounts, send external messages, train a fallback model or buy compute. Missing official MAttr evaluator means reproduction remains pending; an independent implementation needs a different label and separately scoped work.

Task-specific budgets are initial safety bounds, not runtime estimates or spending forecasts. In absence of a stricter task bound: one worker, max two canary attempts of 20 minutes each, <=1 GPU-hour of further verification. Aggregate accelerator time is GPU count × wall time. Cancel jobs at the cap, preserve progress and record pending cells. Higher-budget scientific campaigns are separate dispatches with an explicit finite envelope; do not silently shrink a paper reference and call it reproduced. No GPU jobs run just to prepare these plans.

Public/local fixtures and existing licensed datasets avoid a new synthetic Neon write campaign. If later work truly requires scaled synthetic data or private assets, keep it outside these defaults and honor the applicable data/workspace authorization. User-delegated plan review does not supply unavailable credentials or authorize publishing private data. Human-audit exports are allowed; absent annotations remain absent and cannot support human-validated claims.

## Failure and rollback

Keep new capability opt-in until qualified. On regression, disable its registration/profile or revert only the task's patch; never delete user artifacts or rewrite catalog history. Versioned malformed output is rejected, not coerced into valid schema. Preserve failed-run diagnostics and verified chunks. Deterministic errors do not cold-start retry; transport retries are bounded and visible. Explain the exact remaining external blocker in the receipt, along with the single smallest action that would unblock it.
