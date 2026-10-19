# Agent guide

Xenon is Concordance's platform for mechanistic interpretability.
It provides workflow execution, model adapters, analysis, interventions,
artifact storage, and reports.

## Start here

Read the [operations guide](operations/README.md) for setup and execution.
Then use the reference that fits the task:

| Task | Reference |
| --- | --- |
| Develop a research idea | [Research prompts and examples](methodology/README.md) |
| Choose a capability or model | [Capability map](platform/RESEARCH_HUB.md) |
| Configure an LLM experiment | [LLM guide](platform/LLM_RESEARCH.md) |
| Adapt a runnable recipe | [Examples](examples/README.md) |
| Write a workflow | [Workflow API](platform/API.md), [spec conventions](platform/SPECS.md) |
| Inspect data or stored results | [Storage](operations/LOCALITY.md), [discovery](operations/INDEXING.md) |
| Build or share a report | [Reporting](operations/REPORTING.md) |
| Change the platform | [Architecture](platform/ARCH.md), relevant code and tests |

Read further references when the task needs them. Do not load every guide.

## Execute the requested scope

- Use the user's objective, budget, and completion criteria to set the scope.
- Resolve routine details from the task, code, and existing recipes.
  Ask when missing information prevents a sound decision.
- Choose checks for the risk of the change. A smoke test is a check, not the goal.
- Use relevant existing evidence. Repeat a check when a change or failure warrants it.
- Check changes to GPU execution on the actual model path during implementation.
  Do not leave all live GPU checks until the end.
- Separate working infrastructure from research findings. State what the result supports.
- Keep a concise record of the config, run IDs, result, and next action.
  Use the research project's format. No fixed phase sequence is required here.

[Methodology](methodology/README.md) offers prompts and illustrative examples for research planning.
Use it when it helps. It does not require a plan template or a fixed research process.
[Backburner work](platform/BACKBURNER.md) is outside the task unless the task needs it.

## Repository boundaries

```text
pipelines_v2/       Reusable platform code
examples/           Maintained recipes and integration fixtures
platform/           API and capability references
operations/         Setup, execution, storage, and reports
tests/              Platform tests
papers/voice/       Packaged compatibility helpers
scripts/            Infrastructure tools
dashboard/          Frontend source; outside the Python package
methodology/        Research prompts and illustrative examples
```

Keep new research studies in a separate repo that uses Xenon.
Preserve `papers.voice` imports and bundled assets; downstream repos use them.
Promote reusable helpers into `pipelines_v2/`. Keep study code with its study.
Historical plans and outputs are outside this checkout. They are not setup inputs.

## Implementation and writing

- Treat `workflow.py` as the executable source of truth.
- Set `local_python_sources` to explicit source directories. Do not mount `"."`.
- Keep shared data in its assigned store. Use the catalog to find workflow results.
- Preserve the user's changes. Do not include credentials or generated data in commits.
- Use ASD-STE100 Simplified Technical English for technical instructions.
  Use short sentences, active voice, and consistent terms. Keep exact API names.
  Separate required actions, recommended practices, and code-enforced limits.

## Running workflows from a sibling repo

For workflows that need only Xenon worker sources, set `XENON_WORKSPACE_ROOT` to the Xenon checkout.
This selects its config and Modal source mount base.
Use absolute workflow paths. Include the research repo on `PYTHONPATH` when needed.
See [the complete procedure](operations/README.md#use-a-workflow-from-another-repo).
