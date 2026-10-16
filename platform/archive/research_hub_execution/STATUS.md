# Implementation status

The current target is usable capabilities with small real-model plumbing checks.
Scientific reproduction and larger evaluations remain separate follow-on work.

| Capability | Implemented and exercised | Remaining boundary |
| --- | --- | --- |
| Local experiments | Offline GPT-2 capture, generation, intervention, artifact and report workflow | Broader architectures/devices are separately qualified |
| Native neurons and heads | GPT-2 head coordinates; real Qwen3-0.6B SwiGLU capture and selected-coordinate edit | Real MoE and large-model support remain unqualified |
| Official attribution | Actual official MAttr GPT-2 and Qwen2.5/IOI runs, 157/361-node rankings, ten-point evaluator output, cataloged summaries | Caller-owned IOI files and complete provenance verified locally and on Modal L4; other task schemas need adapters |
| Image generation and steering | Real pinned SDXL generation, internal capture/edit, same-noise replay, remote artifacts and bounded local previews | Strong signed direction comparison completed: modest color shift, pronounced distortion; broader study unexecuted |
| Pretrained SAEs | Real GPT-2 dictionary import, encode/decode, original-model feature splice, independent forward comparison and report | Complete provenance/reuse and nine-example/three-edit Modal L4 run verified; dictionary coordinate remains pinned |
| Neuron microscope | Bounded contexts, signed Qwen3 summaries, three neuron dossiers and a real capture/edit/report workflow | Larger corpus studies and scientific validation unexecuted |
| Attribution graphs | Typed rankings/effects, original-model GPT-2 route checks, token-gradient candidates and bounded graph report | Cross-framework coordinate joins and larger scientific validation remain pending |
| Vision recognition | Pinned ViT class/patch capture, identity and edit; real two-image CPU workflow with independent forward parity | Natural-image research and GPU qualification unexecuted |
| Evidence workbench | Catalog discovery, bounded neuron/native-graph/transcoder/MAttr panels, verified shared publication readback | C05 pack/link verification integrated; independent visual review pending |
| Crosscoders | Real Qwen2.5 base/instruct capture, pinned reference fit/checkpoint/resume, encode/decode and original-model feature edits | Configurable CPU/CUDA workloads and full eight-step L4 workflow verified; scientific validation deferred |
| Pretrained circuit tracing | Real Qwen3 PLT trace on A100-80GB, 61 nodes/90 edges including all 56 error nodes | Strict HF/TransformerLens parity gate failed; original-model edit skipped; live CLT unverified |
| Publication | Approved local-to-Modal publication and shared catalog readback, exact checksums and source identity preserved | Explicit bounded result kinds and complete provenance required |
| Modal run efficiency | Runtime admission and image-boundary recovery exercised; scoped SDXL session implementation integrated | Two dependent SDXL steps used one model load; fresh-admission reuse ran zero model loads. General LLM session support remains unqualified |

Start with the [research hub guide](../../RESEARCH_HUB.md) and its [exact support matrix](../../research_hub_support.json).

Current usable entry points:

- [Local GPT-2 steering](../../TORCH_ENGINE.md)
- [Native sites](../../NATIVE_SITES.md)
- [Official MAttr](../../MATTR.md)
- [SDXL workflow and gallery](../../../workspaces/sdxl_steering/README.md)
- [SAE import and splice](../../../pipelines_v2/representation/sae/README.md)
- [ViT recognition](../../../workspaces/recognition/phase_00/README.md)
- [Evidence retrieval and workbench](../../EVIDENCE.md)

Claude Fable accepted the bounded LLM capability increment, evidence, publication and T05 recovery closure. Full foundation/vision review and broader qualification remain separate. Passing a small model run does not imply acceptance of every gate in the larger task plans.

Earlier capability-increment checks: **907 passed, 8 skipped**; fresh core-only wheel install and all 17 runtime/pin assets verified. CPU Modal crash/resume was exercised on the integrated recovery fix; real-model GPU hard-death recovery remains unqualified.

## Generalization follow-up (G01)

The four current priorities are configurable supported-model workloads, observed provenance/reuse, typed LLM sharing, and explicit Modal profiles. The [configurable LLM guide](../../LLM_RESEARCH.md) is the current entry point; the table above includes the current supported boundaries.

Implemented: Dataset-based SAE inputs and unrestricted valid edit lists; bounded native route/IG workloads; caller-owned IOI inputs and configurable MAttr fitting/evaluation; configurable crosscoder data/width/batches/checkpoint continuation; CPU/CUDA Torch/native adapters; positive configurable tracer node/token budgets. Supported architectures and dictionary coordinates remain explicit.

Verified: GPT-2 CUDA capture/generation, Qwen3 CUDA native capture/edit, nine-example/three-edit SAE CUDA, local SAE and MAttr cross-run reuse, fresh-process Modal reuse with zero GPU execution, and local/remote LLM graph publication/readback. Integrated suite: 930 passed, 10 skipped; core-only wheel: 20 assets, optional imports stay lazy.

All four priorities are complete. MAttr GPU execution and the full crosscoder CUDA workflow passed. A fresh process reused crosscoder capture and fit after two CPU readiness checks with zero GPU executions. Claude Fable resolved all eight review findings; both stated hardware conditions passed. The minor mount-error reporting follow-up passed nine focused checks. Warm session sharing, GPU crash recovery and tracer-to-original parity remain deferred, as requested. Research campaigns are outside this increment.
