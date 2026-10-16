# Emotion-vector workflow recipes

This directory contains generation, capture, vector construction and validation recipes. It does not maintain a current research task queue. For a new study, use the [experiment guide](../../../../platform/LLM_RESEARCH.md), save a phase record, and inspect existing cataloged results and dataset tables before launching new generation.

## Choose a workflow

| Recipe | Workflow file | Purpose |
|---|---|---|
| Small Qwen check | `specs/qwen_smoke_workflow.py` | Model-cache and execution plumbing |
| Authored pilot | `specs/happy_vector_pilot_workflow.py` | Small four-emotion vector workflow |
| Generated-data pilot | `specs/tier_a_generated_vector_workflow.py` | Generate, capture, derive and score a bounded Qwen dataset |
| Llama generation | `specs/llama70b_generation_workflow.py` | Persist parsed stories and neutral rows in Neon |
| Llama vector construction | `specs/llama70b_vector_workflow.py` | Capture prepared rows and derive vectors |
| Llama validation | `specs/llama70b_validation_workflow.py` | Validate derived assets |
| Additional probes | `specs/llama70b_dosage_context_workflow.py`, `specs/llama70b_numeric_semantics_workflow.py` | Study-specific follow-up measurements |

Start by planning the chosen file:

```bash
uv run python -m pipelines_v2.cli workflow plan --file papers/voice/emotions/replication/specs/qwen_smoke_workflow.py
uv run python -m pipelines_v2.cli workflow runs --file papers/voice/emotions/replication/specs/qwen_smoke_workflow.py
```

Use `workflow show --run-id ...` to inspect an existing run. Before execution, inspect the selected workflow's mode, model, resource allocation, table and sample limits. The larger Llama workflows have materially different compute budgets from the Qwen smoke. Asset preparation and shared-data writes follow the normal [locality rules](../../../../operations/LOCALITY.md).

## Inputs and study decisions

- [Data assets](data/README.md) identifies the checked-in label lists and durable-data lookup conventions.
- `prompts/*.md` are executable generation templates; preserve their placeholders when editing.
- `configs/replication.todo.toml` and `specs/replication_workflow_outline.py` are legacy planning scaffolds. Use each runnable workflow's actual configuration as execution truth.
- Choose the study's model, token pooling, parser/filter policy, split, neutral projection and validation controls explicitly. Earlier pilot defaults are not automatically faithful paper settings.

Generated rows, captures and vector payloads stay in their assigned Neon/Modal stores. Local reports are generated outputs. Old run IDs and launch estimates have been removed from these instructions; current execution state comes from the catalog and asset manifest. Methodological conclusions require study-specific evidence.
