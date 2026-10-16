# Xenon research hub execution plan

> Historical build plan (2026-10-16). Start new work from the [experiment guide](../LLM_RESEARCH.md). Current deferrals are in [BACKBURNER.md](../BACKBURNER.md). Task checklists here are design history, not a current queue.

Make Xenon a dependable place to move from a research question to a small experiment, then to a reproducible study across models. Keep the existing workflow, runner, catalog, and artifact architecture; extend its research capabilities and qualify its execution paths.

Detailed handoff: [execution graph and task plans](research_hub_execution/README.md), including implementation contracts, verification gates and model-ready briefs.

## The plan at a glance

1. **Make expensive runs fail early and recover cleanly.** Add measured resource profiles, complete dependency locks, staged preflight, representative canaries, and durable checkpoints. Preserve loaded models across compatible experiments.
2. **Add real local experiments and a gradient-capable research runtime.** Run small models on CPU or qualified MPS without Modal; use the same research operations on Modal CUDA when scaling. Keep this stack separate from vLLM.
3. **Build a native-neuron microscope.** Inspect and intervene on MLP neurons, gate/value sign regimes, heads, and experts. Retrieve representative examples and test explanations rather than merely label top activations.
4. **Add attribution, circuit analysis, and MAttr.** Reproduce Matryoshka Attribution through its released implementation; produce attributable, testable graphs and compare methods across models, tasks, and sparsity levels.
5. **Make image generation steerable.** Qualify one open-weight Diffusers model, initially SDXL, with internal activation capture and interventions across modules and denoising steps; evaluate changes across prompts and seeds.
6. **Add SAEs, crosscoders, and transcoder-based tracing.** Integrate established tools, starting with pretrained dictionaries. Support model differences and learned-feature circuits alongside native-neuron baselines.
7. **Turn papers into reusable experiment packs and an evidence workbench.** Standardize smoke/reference/scale workflows, comparison reports, provenance, and links from claims to runs, neurons, and intervention evidence.

**Priority:** 1–2 are the foundation; 3–5 deliver the motivating research capabilities; 6 follows those baselines. The minimal version of 7 ships with the first experiment, while its richer interface grows with the research tools.

**First useful delivery:** a real local GPT-2 capture/intervention report, followed by a small MAttr replication and the same recipe on one qualified Modal worker. Do not wait for a complete dashboard or universal backend before producing research results.

## What already exists

This assessment uses the current working tree, including its in-progress runtime hardening, rather than treating architecture proposals as shipped capabilities.

| Area | Existing foundation | Extension needed |
| --- | --- | --- |
| Execution | Local and Modal runners; workflow DAGs; resume and completed-artifact reuse | Real local research engine, differentiable operations, qualified workload profiles |
| Reliability | Strict vLLM/config validation, GPU-count checks, mounts/secrets checks, prompt-length checks before model loading, reusable vLLM sessions, partial generation recovery | Capacity prediction and measurement, complete environment locks, pre-fanout canaries, checkpointing for all new long-running operations |
| Interpretation | Residual capture, MoE routing, probes, directions, geometry, steering, interchange and path-patch machinery | Native neuron/head sites, gradients, attribution algorithms, graph artifacts |
| Learned features | PCA and other representation analysis | SAE module is a placeholder; no implemented crosscoder/transcoder suite |
| Research records | Catalogs, typed artifacts, reports, phase and hypothesis conventions | Paper packs, evidence-linked neuron/graph browsing, cross-model comparisons |
| Vision | General runner/storage abstractions | Image inputs, spatial/timestep coordinates, recognition and generative-model adapters |

**Integrate:** pinned Hugging Face/PyTorch, the official MAttr harness, Diffusers, SAELens, and circuit-tracer for validated model/checkpoint pairs. Keep their incompatible environments separate.

**Defer:** large dictionary training, RL weight attribution, layer-crush retraining, full checkpoint-development studies, FLUX qualification, frontier-scale enumeration, and a broad dashboard rebuild.

## Delivery order and completion gates

| Slice | Deliverable | Gate to the next slice |
| --- | --- | --- |
| A | Minimal admission/provenance, local PyTorch research path, one local capture/intervention pack | Real model produces a useful report with cloud calls disabled; invalid settings fail before execution |
| B | Small MAttr pack and one Modal reference profile | Official evaluation is reproduced within declared tolerance or discrepancy is explained; interrupted work resumes; compatible operations reuse one model load |
| C | Native-neuron audit, second model, basic graph explorer | Held-out explanation tests and causal controls are recorded; cross-model comparisons state their limitations |
| D, alongside C | SDXL steering pack | Internal intervention has a repeatable effect across held-out prompts/seeds, with controls and quality assessment |
| E | Learned features, model differences, wider circuit/weight attribution and checkpoint studies | Reconstruction and intervention fidelity are measured; new models are admitted through the same qualification process |

Before slice A, reconcile and verify the existing uncommitted hardening with its owner; this plan does not authorize committing or merging it. These are acceptance milestones, not calendar estimates. Each slice is now decomposed into the [execution graph and task briefs](research_hub_execution/README.md); these specify outcomes, dependencies, implementation steps and acceptance evidence.

## Implementation scope

### 1. Run admission, resources, and recovery

Extend the existing planner rather than adding a second launcher. Admission has three stages:

- **Static checks before cloud submission:** validate the entire operation/workflow configuration, graph dependencies, model and tokenizer revisions, hook support, dtype, context/output lengths, capture sizes, GPU topology, CPU RAM, storage, secrets, source mounts, and dependency compatibility. Reject unknown options and unsupported combinations. Show the effective configuration and where overrides came from. Distinguish checks actually performed from requirements merely declared.
- **Environment readiness before loading weights:** verify the pinned runtime imports, model access, checkpoint completeness, source packaging, dataset/schema availability, and store/catalog access. Produce a structured diagnostic with the exact failed check.
- **One bounded GPU canary before fanout:** load the intended model and exercise the actual capture/intervention/backward path at the representative maximum workload shape. Measure host RAM, VRAM, storage throughput, and runtime. The orchestrator must hold all remaining shards until a persisted, matching qualification result passes; this extends current eager fanout. Continue useful work in the canary worker and deduplicate any examples used for calibration. The session timeout includes both canary and continuation.

A profile is keyed by model/component hashes, runtime lock, hardware/topology, dtype, operation, hooks, sequence/image shape, batching, and compilation settings. Record estimate uncertainty and measured headroom; an inference profile cannot certify an attribution backward pass. Include KV cache, captured tensors, donor buffers, optimizer state, loading peaks, and host memory in the resource envelope. Install per-runtime fully pinned dependency/constraints files resolved for the target Linux/CUDA environment, separately from local host locks, and record the built image identity. Publish an explicit model × operation × device support matrix with qualified, unverified, and unsupported states.

Preserve the existing session reuse and generation checkpoints. Extend the vLLM-only batch-compatibility key and remote session cache to qualified PyTorch operations; external MAttr runs retain their own process/session boundary. Add checksum-verified partial capture/intervention records and learned-method checkpoints, including optimizer, sampler, and RNG state where required. Partial records carry incomplete status and never publish a final artifact manifest. Write that manifest atomically only after expected example/token/shard coverage and payload integrity validate; current resume logic treats its presence as completion. Deterministic configuration errors stop immediately. Transient transport failures use bounded retries. Workload changes invalidate qualification and reuse as appropriate. Any batch-size adaptation must be visible and tested for the experiment's semantics.

Track cold-start count, model-load count, time to first useful result, peak memory, recovered work, and failed compute. Warm sessions have an explicit idle lifetime and budget. Provider eviction and arbitrary workload OOMs remain possible; the goal is preventing predictable failures and preserving completed work.

**Acceptance:** seeded bad-config cases submit no GPU work; a canary failure prevents fanout; a killed run resumes without duplicate or missing examples; an incomplete capture cannot be mistaken for a completed artifact; changed model/data/runtime identity prevents stale reuse; compatible experiments incur one model load within their session.

**Main surfaces:** `core/config.py`, `runtime/planning.py`, `runtime/modal.py`, `runtime/modal_worker.py`, `runtime/remote_executor.py`, `engine/vllm/session.py`, workflow records/orchestrator, batch compatibility keys, and runtime tests.

### 2. Local research and differentiable execution

Add a research backend based on PyTorch/Hugging Face, with narrowly qualified model adapters. The official MAttr harness retains TransformerLens inside its own external environment; a native TransformerLens integration is not needed for the first slices. Preserve vLLM for high-throughput inference. Use the existing mounted model volume for the HF cache, with explicit model-access secrets where needed, to avoid repeated downloads. Different dependency stacks run in separate locked environments/images; neither a shared Python process nor the current vLLM dependency set is assumed.

Introduce explicit capabilities for differentiable forward passes, native sites, supported read/write operations, and mask fitting. Fitting is an operation contract; do not force a universal training API into inference engines. General model fine-tuning remains isolated in a sibling runtime and exchanges versioned artifacts. Mask/dictionary fitting can be orchestrated by Xenon without expanding core into a general training framework.

Add an explicit **local scratch** execution profile: bounded rows, tokens, artifact bytes, and wall time; local fixture inputs, LocalArtifactStore under `XENON_HOME/pipelines_v2/artifacts`, and FileCatalog; no Modal, Neon, or external judge dependency. Keep artifacts outside the repo by default and ignore any configured in-repo artifact path. Update `operations/LOCALITY.md` and authoring guidance to authorize this bounded mode. Shared datasets and scaled captures retain their existing durable homes. Add a small publish operation to upload eligible local artifacts, validate their hashes, and register a new remote manifest with local lineage; this is new functionality. Rerun when execution semantics require it. Production activation banks remain remote.

CPU is the correctness baseline; qualify MPS separately and expose unsupported operations rather than silently offloading them. A named GPT-2 capture/patch recipe of eight examples, at most 128 input tokens and 16 generated tokens each, should target under one minute once its model is loaded; report installation, weight download, and cold-load time separately. A full mask-training study is a separate budget.

**Acceptance:** the local recipe works with cloud clients disabled and no cloud credentials. Revision, dtype, attention implementation, and hook version enter semantic identity; changing dtype invalidates reuse. Identity/no-op and known-effect hook tests pass. Local versus Modal and PyTorch versus vLLM comparisons use declared numerical tolerances for logits, aligned activations, and intervention effects.

**Main surfaces:** new `engine/torch/`, engine and operation registries, capabilities, runner dispatch, remote executor, paths, artifact publication, locality docs, and shared engine contracts. General fine-tuning and native TransformerLens integration are outside the first slice.

### 3. Native-neuron investigation

Add typed coordinates for MLP gate preactivation, activated gate, value/up projection and product, attention heads, and expert-local neurons. Preserve signed activations and routed-token coverage. Target the PyTorch research engine first; fused vLLM kernels do not guarantee access to these internals. A module name that happens to contain “MLP” is not sufficient evidence that the intended neuron quantity was captured.

Build streaming top/bottom examples, raw-magnitude and percentile distributions, random/weak-activation samples, and full-context retrieval. Store bounded summaries and indexed references by default. Dossiers link explanations, counterexamples, upstream/downstream evidence, interventions, and uncertainty to an immutable model/site identity.

The first [Goodfire-inspired](https://already-clean.goodfire.com/) pack preregisters a depth-stratified random neuron sample, evaluation rubric, corpus strata and held-out explanation tests. Compare product-only versus sign-aware views, short versus long context, and weak-activation removal versus count/magnitude-matched controls. Use [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) for tool development and [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) for a qualified larger-model comparison; freeze thinking/template settings. These are adapter targets, not claims that small-model behavior replicates the larger model. Use a pluggable, versioned explanation provider with a blinded human-audit subset; local smoke requires no API judge. Report task-specific behavior and unexplained cases; a local pilot cannot establish a model-wide cleanliness percentage.

Add a recognition-model microscope on one small CNN/ViT, with natural examples and feature visualization checked against real-image responses. Developmental claims require actual same-lineage checkpoints; unrelated models or endpoint similarity do not establish how a neuron grew.

**Acceptance:** native quantities are verified against direct forwards; neuron edits produce the expected tensor changes; blinded controls and held-out results are available; failure to explain a neuron is retained rather than relabeled away.

**Main surfaces:** capture/intervention sites, research-engine hooks, streaming summaries, dataset/context references, and report components. Whole-model enumeration and developmental retraining are deferred.

### 4. Attribution, graphs, and MAttr

Drive the [official MAttr implementation](https://github.com/aryamanarora/matryoshka-attribution) and its pinned MIB fork/submodules through a narrow external-tool operation in its own locked local environment or Modal image. Pass a versioned config and import validated scores/curves as Xenon artifacts. This preserves its TransformerLens dependencies outside core. A native PyTorch adapter comes later and must match stored reference fixtures. The official evaluator is required to call a run a paper reproduction; if unavailable, label the result an independent reimplementation. Start with GPT-2 IOI node attribution; add Qwen2.5-0.5B, then qualified Gemma/Llama cells and more granular bases. Record base/source orientation, masking convention, input-node treatment, optimizer, sparsity schedule, and train/validation/test separation, following the [paper's evaluation definitions](https://arxiv.org/html/2609.25518v1).

Compare random rankings, input-times-gradient, integrated gradients, direct interventions on a tractable subset, and a mask-learning baseline. Report full sparsity curves, CPR and accuracy/compactness, seed variation, compute, and memory. Keep paper-matched and compute-matched comparisons distinct. Test learned rankings on held-out examples and related tasks.

Add graph artifacts with typed nodes/edges, target output, algorithm and intervention semantics, signed scores, approximation labels, fixed-attention/normalization assumptions, pruning policy, and omitted nodes or contributions where defined. MAttr ordering scores are not additive edge-flow values. Native graphs and transcoder graphs share a viewer and evidence links while retaining their different meanings. Validate candidate routes through original-model interventions, with same-label, random, identity, and resampled/zero controls as appropriate.

A useful cross-paper extension ranks task-relevant neurons with MAttr, then audits their sign regimes, explanations, and causal roles using workstream 3. ViT patch attribution is an optional method check, not a substitute for image-generation steering.

A later milestone attributes weight differences between compatible checkpoints. Start with a benign spelling/case change and measure target behavior, collateral effects, and capabilities. Weight masks require exact parameter mappings and model lineage; they are not cross-architecture feature alignment. The paper's RL behavior-attribution experiments form a separate later study.

**Acceptance:** soft-mask endpoints match explicit interventions under the same basis/input coverage; analytical fixtures validate curve calculations without assuming a universal random-ranking score. Adapter outputs agree with the pinned reference on a small test; a graph can be traced back to its run and re-tested; a reported circuit survives held-out evaluation or is explicitly rejected. A failed paper finding still yields a reproducible result.

**Main surfaces:** new attribution specs/adapters, operation registries and all runner dispatch paths, graph artifacts/evaluators, optional graph viewer, and isolated upstream runtime. RL and a native rewrite are not prerequisites.

### 5. Vision generation and steering

Qualify [SDXL base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) through Diffusers as the first open-weight generator. Freeze the UNet, text encoders, VAE, scheduler, preprocessing, and exact revisions. SDXL is the initial engineering target; FLUX is a later adapter requiring separate qualification.

Add image/latent artifacts and selectors for module, denoising timestep, spatial position, and conditional/unconditional branch. Support baseline generation, internal capture, direction addition, ablation, and matched activation replacement. [Step callbacks](https://huggingface.co/docs/diffusers/using-diffusers/callback) are useful orchestration points, but internal steering needs verified module hooks. CFG selectors must correctly distinguish conditional, unconditional, and both branches, including their concatenated batch layout. Record initial noise, all seeds, guidance settings, prompts, dimensions, and intervention strength.

Derive a color/style direction from paired training prompts with matched seeds; evaluate fixed prompts with unchanged initial noise, then held-out prompts and seeds. Include no-op and norm-matched random directions, strength sweeps, attribute scoring, blinded visual review, image quality, and off-target changes. Preserve scheduler/RNG state for supported resumption boundaries; otherwise resume between images rather than claiming exact mid-trajectory recovery.

**Acceptance:** one end-to-end Modal recipe produces a reproducible baseline/steering gallery and metrics. It demonstrates internal control beyond prompt editing. Generation steering is a separate extension from the papers' image-recognition experiments.

**Main surfaces:** new Diffusers adapter, image examples/artifacts, timestep/spatial selectors, runner dispatch, and gallery/evaluation reports. Local SDXL performance and full visual circuit tracing are not first-slice requirements.

### 6. Learned feature and model-difference tools

Integrate pretrained SAEs through [SAELens](https://github.com/decoderesearch/SAELens) first: checkpoint/site compatibility, encoding/decoding, reconstruction error, sparsity, and causal fidelity. Follow with one paired base/fine-tuned crosscoder, using a pinned [reference integration](https://github.com/ckkissane/crosscoder-model-diff-replication), using aligned inputs/tokens and explicit normalization. Compare native neurons, PCA/random baselines, and learned features under matched evaluation budgets.

Treat crosscoders and cross-layer transcoders as distinct capabilities. Crosscoders support joint representations and model differences. Transcoders approximate model computations and support replacement-model attribution graphs; integrate [circuit-tracer](https://github.com/decoderesearch/circuit-tracer) only for validated model/transcoder combinations. Preserve reconstruction/error contributions and validate findings in the original model.

Small fitting jobs use isolated research workers; larger dictionary-training campaigns receive their own scoped plan. No arbitrary cross-model neuron-ID correspondence is assumed.

**Acceptance:** imported dictionaries round-trip, mismatched checkpoints/sites fail early, reconstruction effects are reported, and one model-difference hypothesis is tested causally rather than inferred from decoder norms alone.

**Main surfaces:** representation specs, dictionary checkpoint adapters, paired-feature alignment, research fitting workers, graph imports, and fidelity evaluators. Large dictionary campaigns remain separately scoped.

### 7. Experiment packs and evidence workbench

Create packs under in-repo `workspaces/<study>/phase_XX/`, following `methodology/FLYWHEEL.md`; keep reusable library code in `pipelines_v2/` and existing `papers/voice` untouched. Creating this currently absent directory is part of slice A. Keep checked-in `workflow.py` as executable truth. Packs supply frozen paper/code/model/data versions, hypotheses, splits, baselines, controls, budget limits, and smoke/reference/scale presets that compile to existing WorkflowSpecs. New research phases use `PHASE.md`; claim-shaped hypotheses use the existing catalog convention. Do not introduce another scheduler or run database.

Every result records source revision plus dirty-tree identity, runtime lock/image digest, model/tokenizer/component revisions, data and row identities, random seeds, hook schema, method/evaluator version, and artifact hashes. Scientific reuse and live-worker compatibility are separate keys. Preserve local catalog visibility, with explicit later publication of local scratch results to shared storage.

Start with CLI/Python discovery, capability inspection, budgeted planning, execution/resume, evidence retrieval, and local reports. Add an optional workbench for comparing runs, inspecting neurons and graphs, viewing steering galleries, and seeing failure diagnostics and resumable state. Every displayed claim links to its supporting examples and intervention results. Keep the existing hypothesis lifecycle and record confirmation/refutation/uncertainty as evidence verdicts. UI views consume summaries and capped samples, not bulk remote activations.

**Acceptance:** a fresh session can discover and reproduce a selected experiment from its saved record; changing a split, checkpoint, evaluator, or hook implementation cannot silently reuse an incompatible result.

**Main surfaces:** workflow helpers, catalogs and manifests, reporting, new study workspaces, phase/hypothesis templates, and an optional UI package. No new scheduler, parallel tracking database, or mandatory cloud UI.

## Scope boundaries and research questions

The initial release qualifies a few model families and three useful research paths. It does not promise universal models, a new scheduler, full model training in core, frontier-scale neuron enumeration, or failure-free cloud compute.

The [Goodfire essay](https://already-clean.goodfire.com/) is a working hypothesis, and [MAttr](https://arxiv.org/html/2609.25518v1) is an attribution method with explicit task/basis/evaluation choices. Neither should become a platform assumption. Exact reproduction of Goodfire's custom training studies requires its checkpoints, data, and settings; until available, label our work as an independent test or conceptual replication. Keep preregistered tests capable of rejecting both native-neuron and learned-feature explanations.

Useful follow-on questions: when do sign/context-aware neuron explanations improve held-out prediction and causal control; which weak activations carry meaningful rare behavior; when do MAttr rankings transfer across tasks; which graph motifs remain stable across checkpoints; and when do dictionaries add explanatory or intervention value beyond native neurons?

## Sources and review record

- [Neural Networks Are Already Clean](https://already-clean.goodfire.com/), including its methodological appendices.
- [Matryoshka Attribution](https://arxiv.org/html/2609.25518v1), especially representation attribution, weight attribution, evaluation caveats, and the vision appendix.
- [Official MAttr implementation](https://github.com/aryamanarora/matryoshka-attribution), including its separate dependency groups and parameter-attribution sibling.
- [SDXL model card](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [Diffusers callbacks](https://huggingface.co/docs/diffusers/using-diffusers/callback).
- [SAELens](https://github.com/decoderesearch/SAELens), [circuit-tracer](https://github.com/decoderesearch/circuit-tracer), and [crosscoder model-diffing reference](https://github.com/ckkissane/crosscoder-model-diff-replication).

Repository evidence: [engine registration](../../pipelines_v2/engine/registry.py), [engine contract](../../pipelines_v2/engine/base.py), [capture sites](../../pipelines_v2/operations/capture/sites.py), [SAE placeholder](../../pipelines_v2/operations/representation/sae.py), [local runner](../../pipelines_v2/runtime/local.py), [Modal runner](../../pipelines_v2/runtime/modal.py), [remote execution](../../pipelines_v2/runtime/remote_executor.py), [artifact manifests](../../pipelines_v2/storage/artifacts.py), and [runtime tests](../../tests/pipelines_v2/runtime). The [locality rules](../../operations/LOCALITY.md), [research methodology](../../methodology/FLYWHEEL.md), and [architecture](../ARCH.md) define the policy changes identified above.

Planning baseline: 2026-10-16, HEAD `6df58cdfda43c4b17dcb35abc9eb2bfdbc571ce6` plus existing working-tree changes. No experiments or GPU jobs were run for this planning review.

Agreed by the active Codex chat (Astra High) and Claude Fable through three Planbridge review turns. Claude CLI confirmed `claude-fable-5-1`; final verdict: “agree,” with no remaining blocker. The local [planning transcript](../../.planning/tasks/research-hub/transcript.md) preserves the independent assessment, challenges, and dispositions; `.planning/` is ignored and is not part of the distributable plan.
