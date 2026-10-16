# Backburner

Current supported-model experiments can proceed. These items are deferred work, not prerequisites for starting a study. Promote an item only when an experiment needs it, with its own bounded implementation and verification scope.

## Revisit when needed

| Priority | Work | When to pick it up | Completion check |
|---|---|---|---|
| 1 | **Tracer-to-original-model parity** | A study needs to transfer a PLT tracing result into a causal claim about the original HF model | Pin both implementations, quantify logits/activation differences, verify coordinate mappings and representative original-model interventions. Preserve replacement-model-only labels until the gate passes. |
| 2 | **Warm LLM session sharing** | Repeated model loads dominate iteration time | Reuse a compatible worker/model across dependent steps; reset hooks/RNG/request state; compare cold and warm results; record actual model-load counts. Existing SDXL sessions do not establish LLM support. |
| 3 | **GPU crash recovery** | Longer expensive runs justify recovery engineering | Kill a real worker mid-run, recover only verified durable coverage/checkpoints, and show no missing/duplicated work. Crosscoder continuation from an explicit finalized checkpoint already works; arbitrary interruption recovery is separate. |

## Extensions driven by future studies

- **More models and devices:** add architecture-specific adapters and immutable checkpoints, then small capture/edit checks before larger jobs. MPS, real MoE, larger Qwen native models, and broad GPU coverage are not currently established.
- **Additional learned-feature tooling:** live CLT tracing, additional pretrained dictionaries, cross-framework graph/coordinate joins, and broader crosscoder model pairs. Each needs explicit coordinate and artifact compatibility checks.
- **Broader attribution tasks:** new MAttr dataset/evaluator adapters beyond the supported IOI schema, native MAttr implementations, or benign checkpoint-delta attribution. Preserve official-method comparisons where applicable.
- **Scale and collaboration:** out-of-core crosscoder training, multi-worker admission/reuse, cross-volume publication, richer evidence views, and a hosted dashboard. Choose these from a demonstrated workflow bottleneck.
- **Vision expansion:** additional recognition/generation models and cleaner steering experiments, after higher-priority LLM needs.

## Research is its own workstream

Paper reproduction, baseline comparisons, held-out generalization, large dictionary studies, and strong scientific claims were intentionally outside the infrastructure checks. Start those as experiments with [methodology/FLYWHEEL.md](../methodology/FLYWHEEL.md) and a [phase record](../methodology/templates/PHASE.md). A small successful plumbing run is not a paper result.

The [original plan](archive/RESEARCH_HUB_PLAN.md) and [task graph](archive/research_hub_execution/README.md) preserve detailed design history. They include larger ambitions and old acceptance checklists; this file is the current deferral list. Current runnable scope lives in [RESEARCH_HUB.md](RESEARCH_HUB.md).
