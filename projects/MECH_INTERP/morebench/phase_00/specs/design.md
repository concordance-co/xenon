# MoReBench Phase 00 Design

## Goal

Validate whether `MoReBench` is a credible substrate for benchmark-first mechanistic interpretability work before moving into probing or interventions.

## Deliverables

- `outputs/benchmark_framing.json`
- `outputs/benchmark_snapshot_detail.json`
- `outputs/native_label_inventory.json`
- `outputs/theory_pairing_audit.json`
- `outputs/confound_analysis.json`
- `outputs/action_locus_probeability_audit.json`
- `reports/phase_00_benchmark_validation.md`

## Scope

- inspect the actual public and theory CSVs
- verify schema, scale, and rubric structure
- identify readiness blockers and confounds
- separate benchmark worthiness from behavioral sanity
- make probeability failures explicit when post-stratified usable N collapses

## Explicit Non-Goals

- no probe training
- no intervention claims
- no theory-conditioned claims unless prompt exposure is verified
