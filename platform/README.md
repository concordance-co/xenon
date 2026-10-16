# Platform documentation

## Start an experiment

- [LLM_RESEARCH.md](LLM_RESEARCH.md) — new-thread handoff, setup, configurable inputs, local/Modal execution and verification.
- [RESEARCH_HUB.md](RESEARCH_HUB.md) — capability map and runnable method recipes.
- [BACKBURNER.md](BACKBURNER.md) — deferred infrastructure and optional extensions.
- [Support matrix](research_hub_support.json) — exact observed models, run IDs, tests and limitations.

## Author and operate workflows

- [API.md](API.md) — public Python types and CLI commands.
- [SPECS.md](SPECS.md) — workflow authoring conventions.
- [WORKFLOW.md](WORKFLOW.md) — plan, run, inspect, resume and rerun.
- [ARCH.md](ARCH.md) — execution architecture.
- [Examples](examples/) — runnable usage snippets.
- [EXPERIMENT_PACKS.md](EXPERIMENT_PACKS.md) — optional presets, recipe locks and evidence links.
- [Patching guidance](patching_best_practices.md) — intervention design and implementation.

## Runtime, storage and reporting

- [ADMISSION.md](ADMISSION.md) — static checks, target readiness and measured resource envelopes.
- [LOCAL_SCRATCH.md](LOCAL_SCRATCH.md) — bounded offline CPU experiments.
- [PROVENANCE.md](PROVENANCE.md) — observed scientific identity and safe reuse.
- [DURABILITY.md](DURABILITY.md) — artifact completion and recovery boundaries.
- [PUBLICATION.md](PUBLICATION.md) — verified artifact sharing.
- [EVIDENCE.md](EVIDENCE.md) — retrieval and bounded evidence reports.
- [QUALIFIED_SESSIONS.md](QUALIFIED_SESSIONS.md) — measured SDXL session behavior.
- [Data locality](../operations/LOCALITY.md), [indexing](../operations/INDEXING.md), [reporting](../operations/REPORTING.md).

## Model-specific guides

- [Torch residual engine](TORCH_ENGINE.md), [native sites](NATIVE_SITES.md), [MAttr](MATTR.md).
- [Image contracts](IMAGE_DATA.md), [Diffusers](DIFFUSION.md).
- [vLLM upgrade notes](VLLM_0_25_1_UPGRADE.md) — runtime compatibility and validation evidence.

[Archived research-hub build plans](archive/research_hub_execution/README.md) preserve implementation history. New experiments should begin with the current guides above.
