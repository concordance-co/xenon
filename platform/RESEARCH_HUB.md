# Research hub: capabilities and recipes

Start a new experiment or thread with [LLM_RESEARCH.md](LLM_RESEARCH.md). It covers preparation, configurable local/Modal runs, verification, and handoff. The supported-model platform is ready for experiments; [BACKBURNER.md](BACKBURNER.md) contains deferred infrastructure and extensions.

## Choose a capability

| Goal | Entry point | Supported scope |
|---|---|---|
| Quick local hypothesis or residual steering | [Configurable LLM workflow](LLM_RESEARCH.md), [GPT-2 recipe](../workspaces/local_research/phase_00/README.md), [Torch API](TORCH_ENGINE.md) | Pinned GPT-2; local CPU or Modal CUDA; capture, generation, interchange and direction edits |
| Inspect or edit native neurons and heads | [Native sites](NATIVE_SITES.md), [neuron microscope](../workspaces/native_neurons/phase_00/README.md) | GPT-2 heads and dense Qwen3-0.6B MLP coordinates; signed activation summaries and selected-coordinate edits |
| Run official MAttr attribution | [MAttr guide](MATTR.md) | Pinned GPT-2 and Qwen2.5-0.5B; caller-owned IOI-schema data; configurable fitting/evaluation |
| Measure head effects and inspect a graph | [Native graph recipe](../workspaces/native_route_smoke/README.md) | GPT-2 paired interventions, controls, gradient candidates and typed graphs; explicit work/token budgets |
| Encode/decode or steer SAE features | [SAE guide](../pipelines_v2/representation/sae/README.md) | Pinned GPT-2 layer-6 dictionary; configurable examples and valid feature edits |
| Compare representations across models | [Crosscoder guide](../workspaces/qwen_crosscoder/README.md) | Qwen2.5-0.5B base/instruct; configurable fitting, exact checkpoint continuation, encode/decode and original-model splices |
| Trace pretrained transcoder flows | [Circuit tracing recipe](../workspaces/circuit_trace_smoke/README.md) | Qwen3-0.6B PLTs; configurable node/token budgets; replacement-model evidence, original-model parity deferred |
| Browse or share results | [Evidence](EVIDENCE.md), [publication](PUBLICATION.md) | Catalog discovery, bounded reports, typed LLM result/dictionary publication with checksums and preserved lineage |
| Explore vision | [ViT recognition](../workspaces/recognition/phase_00/README.md), [SDXL steering](../workspaces/sdxl_steering/README.md) | Existing image capture/edit/report paths; image expansion is currently lower priority |

Recipe example counts and training steps are small verification choices. Configurable APIs accept workloads within explicit budgets and supported model/dictionary contracts. Adding an arbitrary architecture requires an adapter.

## What has been verified

Small real runs exercised GPT-2 residual edits, Qwen3 native edits, pretrained SAE splices, MAttr, and the complete crosscoder capture/train/resume/encode/decode/splice chain. Modal L4 checks ran during implementation. Fresh-process crosscoder reuse needed two CPU readiness checks and **zero GPU executions**. Local SAE/MAttr reuse and shared graph publication/readback also passed.

The integrated suite passed **930 tests, 10 skipped**; a subsequent mount-error reporting fix passed nine focused checks. A core-only wheel install verified 20 runtime/pin assets and lazy optional imports. Claude Fable reviewed the implementation and its final hardware conditions passed. Exact revisions, run IDs and evidence are in the [support matrix](research_hub_support.json). These checks establish working plumbing; study-specific research conclusions still need their own evidence.

## Operational rules

- Prepare immutable model/tool caches separately; execution uses verified offline assets.
- Plan the workflow before compute; use readiness checks before a costly first run.
- Keep shared datasets in Neon, bulk captures on Modal, and reports local. Follow [locality](../operations/LOCALITY.md).
- Reuse requires matching observed provenance and verified completed payloads. Changed inputs, code, runtime or numeric policy may legitimately trigger execution.
- Use the standard catalog for discovery and [publication](PUBLICATION.md) for deliberate local-to-shared transfer.

For detailed runtime behavior, use [admission](ADMISSION.md), [local scratch](LOCAL_SCRATCH.md), [provenance](PROVENANCE.md), and [durability](DURABILITY.md). The [original design and execution plans](archive/research_hub_execution/README.md) are historical reference.
