# Phase 13: Real Signal Discovery

Question: does the synthetic conflict direction fire anywhere on real DX Terminal prompts?

This phase is the coarse search after the Phase 12 bridge work. The goal is not
to train a classifier or tune thresholds. The goal is to fill the layer x
position x tier x direction grid, read the extremes, and decide whether there
is a real readout worth pursuing.

## Scope

In scope: coarse projection grid only — stratum-mean heatmap across layer × position × tier × direction, plus top/bottom prompt notes for any interesting cells. Deliverable is a kill/partial/continue verdict.

Out of scope:

- classifier training
- threshold tuning
- AUROC
- Stage 2 span localization
- bridge label cleaning
- interventions

## Canonical Surfaces

- Corpus builder: `scripts/build_signal_discovery_corpus.py`
- Workflow: `specs/workflow.py`
- Running log: `running_log.md`
- Neon table: `dx_terminal_signal_discovery_phase13_v1`
- Direction bank: Modal/catalog transform artifact produced by
  `build_synthetic_direction_bank`

## Current Data Assumptions

- Anchor rows come from:
  - `dx_terminal_trade_size_stage1b_adapter_strict_v1`
  - `dx_terminal_trade_size_stage1b_adapter_strict_buy_only_v1`
- Complaint rows come from:
  - `dx_terminal_real_complaint_transfer_ticks_v1`
- Fallback structure-matched controls come from aligned rows in:
  - `dx_terminal_trade_size_stage1a_template_control_v1`
- Baseline controls require an explicit non-complaint production tick source.
- Obvious-aligned controls require an explicit source and should be skipped if
  not sourceable.

## Operator Sequence

1. Build or refresh the Neon corpus table.
2. Plan the workflow.
3. Recompute/index the synthetic direction bank on Modal from the Phase 12
   capture artifact.
4. Run real-prompt capture on Modal.
5. Run the coarse projection grid on Modal.
6. Read top/bottom prompts for interesting cells.
7. Record a kill/partial/continue verdict.

```bash
uv run python -m projects.DX_TERMINAL.prompt_confusion.phase_13.scripts.build_signal_discovery_corpus
uv run python -m pipelines_v2.cli workflow plan --file projects/DX_TERMINAL/prompt_confusion/phase_13/specs/workflow.py
uv run python -m pipelines_v2.cli workflow run --file projects/DX_TERMINAL/prompt_confusion/phase_13/specs/workflow.py --logging INFO
```

Raw direction vectors are not a local file artifact. The workflow stores them as
the `build_synthetic_direction_bank` transform result in the Modal-backed
artifact catalog.

`structure_matched_control` is not a true production baseline. It is a fallback
control stratum for aligned, real-template prompt structure while
`baseline_control` remains reserved for actual non-complaint production ticks.
