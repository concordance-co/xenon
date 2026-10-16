# Planning review

## Decisions from independent Codex and Claude Fable decomposition

- Preserve existing architecture and dirty hardening; establish a reproducible baseline before implementation.
- Add T24 for legacy hash/codec compatibility and narrow extension registries. New native/latent tensors must not fall through to JSON storage.
- Make image/latent identity a shared producer task T25; recognition and SDXL consume it independently.
- Keep MAttr external with its pinned MIB fork and its own Python environment; a native rewrite is deferred.
- Distinguish start dependencies from acceptance-time integration dependencies and serialize shared-file edits.
- Keep qualifications/dossiers as existing catalog artifacts, with no new catalog database.
- Preserve atomic final manifests and honest incomplete states across kill/resume.
- Bound every live check and retain separate engineering and scientific verdicts. No required research winner or promised compact top-percent circuit.
- Select one initial recognition model, a narrow Torch patch capability set and separately qualified optional model cells; avoid inflating prerequisites to all architectures/operators.
- Human audit is an explicitly unfilled scientific extension when annotations are absent; user plan review is not required.

## Full package review and dispositions

Claude read all 26 task plans plus the graph, guide, contracts and supporting docs, then cross-checked runtime and patching code. Its first full-package verdict was revise with seven concrete blockers; all are addressed:

| Finding | Resolution |
| --- | --- |
| T08 had no target-logprob producer | T04 owns optional target-token readout, precise position/normalization and direct-forward oracle; legacy defaults omit the new field |
| Ablation/control operator names were undefined | T04 owns ScalePatch; T08 uses explicit scale/interchange operators and donor pairings |
| Modal HF cache setup had no owner | T06 owns mounted HF_HOME, secret binding, readiness and preparation timing |
| Image sharding could conflict with runtime work | T25 takes remote-execution lock and adds only a pure identity function plus call site |
| Report visual review could be read as a human gate | T21 explicitly assigns rendered-panel inspection to the independent reviewer model |
| MAttr had no executable external runtime kind | T10 owns external_tool descriptor and separate locked control/tool interpreters on local/Modal |
| Workspace report outputs were not ignored | T07 owns report/artifact ignore patterns |

Further refinements incorporated: native lock ownership; preservation through runtime-spec wrappers; explicit C06/C09 dependencies; subprocess memory qualification; client-side publication upload versus mounted finalization; candidate corpora; primary image-metric mask fallback; validator location; Qwen2.5 adapter ownership; native gradient ranking oracles; pretrained SAE coordinate transformations; and explicit PLT versus CLT qualification. Source checks confirmed the Qwen3-0.6B PLT pair in the official circuit-tracer list and the named GPT-2 SAE release in the official SAELens catalog.

The Goodfire essay.md source was already fetched and read during the initial paper review; it is retained as a verified source link rather than marked missing based on the reviewer's lack of web tools.

## Final review

Claude Fable's final verdict was **agree**, conditional only on two exact plan-time guards; both have been inserted as requested and Claude explicitly stated no further review pass was needed:

1. T02/T04/C04 require engine-specific patch operator sets, rejecting the new ScalePatch on current vLLM/Toy before worker submission.
2. T04/T24/C04 require a granular TARGET_LOGPROBS capability for nonempty target-token requests; the existing LOGPROBS enum alone cannot pass this check.

The final pass read all blocker fixes and the additionally changed task plans/contracts/graph. The full-package pass had read all 26 plans. Planbridge response metadata confirmed `claude-fable-5-1` on all three execution-planning turns. The independent decomposition, full review and final dispositions are preserved in the ignored local transcript; this document contains the implementation-relevant decisions.

Codex and Claude agree the package is ready for task dispatch without another user design review. Real credentials, inaccessible assets, exhausted compute bounds and merge authorization remain execution conditions, not unresolved plan choices. The suggested direct Hugging Face pilot shortcut was not adopted as a bypass of current shared-data locality; bounded local fixtures and shared Neon inputs remain explicitly separated.

Planning validation: 26 unique tasks, 80 dependency edges, 9 shared contracts, 5 milestones, acyclic topological order, matching task/graph/diagram dependencies, and all local document/source links resolved. Run `python3 platform/research_hub_execution/validate_plan.py` to recheck. No implementation or GPU qualification has occurred in this planning task.
