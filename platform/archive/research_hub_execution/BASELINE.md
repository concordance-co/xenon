# T00 execution baseline

Baseline captured 2026-10-16 on the existing `main` working tree at HEAD `6df58cdfda43c4b17dcb35abc9eb2bfdbc571ce6`. The active Codex chat owns integration; Claude Fable reviews acceptance. Preserve the pre-existing changes below. They are inputs to this implementation, not changes authored by this task.

## Source identity and preservation

The scoped inventory contains 323 files from `pipelines_v2`, `tests`, packaged `papers/voice`, platform/operations/methodology docs, `.github`, AGENTS.md, pyproject.toml, uv.lock and xenon.toml, including the two untracked hardening tests. Digest: `b9a3566bc45b48299831cde37c902c6c009bcc6ba686c9f5ad44087837596afc`.

Algorithm: SHA-256 of UTF-8 JSON of the path-sorted list `{path, sha256(file_bytes)}`, with sorted object keys and separators `(',', ':')`. Excludes credentials, caches, generated data, unrelated files and new execution receipts. The ignored `.planning/baselines/T00/` contains `inventory.json`, copied file bytes, `preexisting.patch`, and the original status listing. Nothing has been reset, stashed or committed. Parallel worktrees must carry this scoped baseline plus accepted implementation changes; clean HEAD alone is insufficient.

## Local verification

Executed using the existing Python 3.13.6 environment without installing/upgrading dependencies:

```sh
UV_CACHE_DIR=/private/tmp/xenon-uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --frozen --offline --no-sync pytest -q tests --ignore=tests/test_pipelines_v2_dashboard.py
python3 platform/research_hub_execution/validate_plan.py
```

Result: **368 passed, 6 skipped, 11 warnings in 21.20 seconds; exit 0**. Six real Modal/vLLM tests are opt-in and were not launched. This is the repository CI test selection on macOS using the recorded existing environment, not a fresh Linux environment qualification. Raw output: `/private/tmp/xenon-research-hub-evidence/T00-pytest.log`; a retained copy is under the ignored baseline directory.

Installed versions: pytest 9.0.2, torch 2.10.0, transformers 5.2.0, modal 1.3.4, numpy 2.4.2, scikit-learn 1.8.0, pyarrow 23.0.1. GPU readiness, model access, Neon access and target Linux/CUDA dependency locks were not checked. T02/T06 own those gates. Existing warnings include numerical empty-slice behavior and local CPU-count detection; no failed tests were waived.

Plan validator: 26 tasks, 80 dependency edges, nine contracts, five milestones and 320 local links checked; passed. T24 is the next executable task, followed by T01. C01–C09 field meanings are frozen as in CONTRACTS.md; implementation corrections require producer/consumer tests.

## Pre-existing change dispositions

Supplemental live gate after the user's request for incremental GPU verification: the existing `test_modal_vllm_structured_output_contract` ran from the preserved T00 bytes with one A100-80GB and a 1,200-second worker limit. **1 passed in 174.26 seconds, exit 0**. Modal app `ap-kybYc2E07nAh3W9bt9Nk5U` loaded cached `/models/Qwen/Qwen3-30B-A3B`, generated the required JSON, and terminated normally. Logs are retained under `.planning/baselines/T00/T00-modal-{output,app}.log`. The worker reported 56.88 GiB model memory. This adds real loading/generation evidence to the initial local baseline; it does not qualify every backend, capture/patch path, or immutable model revision.

| Paths | Disposition and coverage |
| --- | --- |
| `.github/workflows/modal-vllm-reusable.yml` | Preserve GPU CI settings; actual Modal jobs remain unrun here |
| `pipelines_v2/cli.py`, `core/config.py` | Preserve strict config/cancellation changes; config and cancellation tests pass |
| `engine/vllm/capture.py`, `engine.py`, `session.py` | Preserve batching, validation and pre-load checks; session/capture helpers and preflight tests pass locally; real GPU parity unverified |
| `runtime/modal.py`, `modal_deployments.py`, `modal_worker.py` | Preserve resource/topology/mount/secret and source-resolution checks; preflight/deployment/runtime tests pass |
| `runtime/remote_executor.py` | Preserve generation identity and streaming/recovery changes; generation-identity/runtime tests pass |
| `tests/pipelines_v2/engine/test_modal_vllm_gpu_smoke.py`, `test_vllm_session.py` | Preserve existing coverage; smoke remains opt-in/skipped |
| `tests/pipelines_v2/runtime/test_contracts.py`, `test_deployments.py`, untracked `test_generation_identity.py`, `test_preflight.py` | Baseline tests to retain and extend, not claim as new |
| `tests/test_pipelines_v2_basics.py`, `test_pipelines_v2_config.py`, `test_workflow_cancel.py` | Preserve baseline regression checks |
| `platform/API.md`, `WORKFLOW.md`, `ARCH.md` | Preserve existing docs, including separate activation-proxy study material; new documentation edits must be surgical |
| `activation-proxy-experiment-ladder.md` | Unrelated study artifact; untouched |
| `platform/RESEARCH_HUB_PLAN.md`, `platform/research_hub_execution/` | Prior agreed planning deliverables; retained as implementation instructions |

Source paths abbreviated in the table after `pipelines_v2/` refer to that package; test file abbreviations refer to the preceding test directory.

## Integration map and carried concerns

- Serialization/identity: `core/types.py` (`to_primitive`, `to_semantic_primitive`, OperationSpec), engine identity methods, operation `from_dict` methods. Preserve default legacy hashes; intentional invalidation is separate.
- Registration: `engine/registry.py`, `operations/registry.py`; capture site key-based loader and intervention site loader; patch recipe loaders; `storage/features.py` codecs.
- Execution: `operations/execution/__init__.py`, `runtime/artifacts.py`, LocalRunner, ModalRunner and remote_executor. Frozen imported artifact-spec tuples must not make late registrations invisible. T24 must test a third extension through both local and fake-remote paths.
- Artifacts: storage manifest/typed accessors, local/Modal stores, catalogs, workflow records and resume lookup. Final-manifest presence currently implies completion; T05 owns atomic completion/recovery changes.
- Public surface: `pipelines_v2/api.py`, family exports, compatibility modules; optional imports must remain lazy.
- Runtime composition: spec `runtime_spec()` methods rebuild PythonRuntimeSpec field-by-field; T02 must preserve new lock/build fields across every wrapper and merge.
- Carried validation gaps: ModalResources.from_dict coerces some numeric/string-bool inputs before validation; add malformed serialized-input tests in T02. PatchedGenerationSpec is not included in existing pre-weight prompt-length checks; T02/T04 must cover its actual inputs. These are recorded baseline gaps, not newly introduced regressions.

## Acceptance

Implementation code remains unchanged in T00. Independent source audit completed. Claude Fable (`claude-fable-5-1`, Planbridge request `b845cd07-57de-467b-bcd5-d9fb2925d633`) accepted T00 on 2026-10-16. Its one required verification is resolved: `git check-ignore -v .planning/baselines/T00/inventory.json` reports `.planning/.gitignore:1:*`. T24 is dispatched, followed by T01; subsequent lanes follow graph.json and shared-file locks.

## Reproduction and parallel checkout handoff

`python3 .planning/baselines/T00/verify.py` verifies the copied bytes against the inventory and recomputes its digest. The snapshot is evidence, not an alternate source tree. Local deterministic integration tests join the existing CPU suite; live/network/GPU tests remain opt-in and are selected explicitly.

Before a parallel checkout is dispatched, create it at the HEAD above, apply `preexisting.patch`, copy the two untracked hardening tests from `files/tests/pipelines_v2/runtime/`, and copy the execution planning package. Then apply accepted implementation diffs, including their new files. Verify the resulting source inventory against the integrator's frozen dispatch inventory. Never use clean HEAD as a substitute. Record each lane's exact base digest and restrict its final diff to its assigned work. Integrate one reviewed patch at a time and run affected cross-lane tests.

Future verification logs include the process exit code explicitly. T00's exit code was observed directly from the command tool; its retained pytest log itself contains only pytest output.
