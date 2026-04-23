---
benchmark: morebench
phase: 02
version: v2
frozen_date: 2026-04-23
input_artifacts:
  - projects/MOREBENCH/phase_01/docs/01-gap-list.md
  - projects/MOREBENCH/phase_02/outputs/action_locus_rewrite_pairs.jsonl
---

# MoReBench 02 Gap List Resolution

## Gap To Repair Mapping

- `theory_identity` not clean prompt-side -> legacy explicit-theory family now marked shortcut-dominated; new shortcut-stress-test families materialized for repair and preflight
- `action_locus` not probeable -> partially resolved with a 10-pair matched rewrite starter batch
- response-side labels need fresh generations -> unresolved in this phase; next step remains generation capture
- `stakeholder_tradeoff_density` needs gold validation -> unresolved in this phase; remains a phase 03 gate item

## Materialized Data Snapshot

{
  "benchmark": "morebench",
  "phase": "02",
  "status": "partial_repair_materialized",
  "phase_status": "partial_repair_materialized",
  "behavioral_smoke_status": "not_run",
  "datasets": [
    {
      "name": "theory_prompt_augmentation_examples",
      "path": "projects/MOREBENCH/phase_02/outputs/theory_prompt_augmentation_examples.jsonl",
      "row_count": 150,
      "rows_with_all_placeholders_substituted": 150,
      "controls_structurally_matched_to_target": null,
      "known_bugs": [
        "canonical theory name plus fixed per-theory anchor sentence creates a one-to-one label fingerprint",
        "phase-03 Experiment 1 showed the prompt family is shortcut-dominated for theory_identity"
      ],
      "purpose": "legacy theory exposure family retained for traceability, not for clean theory_identity probing"
    },
    {
      "name": "theory_control_augmentation_examples",
      "path": "projects/MOREBENCH/phase_02/outputs/theory_control_augmentation_examples.jsonl",
      "row_count": 30,
      "rows_with_all_placeholders_substituted": 30,
      "controls_structurally_matched_to_target": 30,
      "known_bugs": [
        "paired legacy theory family remains shortcut-dominated because the target rows still contain fixed per-theory anchors"
      ],
      "purpose": "legacy neutral controls retained for traceability"
    },
    {
      "name": "theory_wording_variant_examples",
      "path": "projects/MOREBENCH/phase_02/outputs/theory_wording_variant_examples.jsonl",
      "row_count": 150,
      "rows_with_all_placeholders_substituted": 150,
      "controls_structurally_matched_to_target": 150,
      "known_bugs": [
        "preserves the same canonical theory name and fixed anchor sentence as the broken legacy family"
      ],
      "purpose": "legacy same-label wording variants retained for traceability"
    },
    {
      "name": "theory_prompt_repair_examples",
      "path": "projects/MOREBENCH/phase_02/outputs/theory_prompt_repair_examples.jsonl",
      "row_count": 2250,
      "rows_with_all_placeholders_substituted": 2250,
      "controls_structurally_matched_to_target": null,
      "known_bugs": [],
      "purpose": "harder theory prompt families for shortcut stress testing and prompt-side retry selection"
    },
    {
      "name": "theory_prompt_repair_controls",
      "path": "projects/MOREBENCH/phase_02/outputs/theory_prompt_repair_controls.jsonl",
      "row_count": 180,
      "rows_with_all_placeholders_substituted": 180,
      "controls_structurally_matched_to_target": 180,
      "known_bugs": [],
      "purpose": "generic ethics and mismatch decoy controls for shortcut diagnosis"
    },
    {
      "name": "theory_shortcut_preflight",
      "path": "projects/MOREBENCH/phase_02/outputs/theory_shortcut_preflight.json",
      "row_count": 2250,
      "rows_with_all_placeholders_substituted": 2250,
      "controls_structurally_matched_to_target": null,
      "known_bugs": [],
      "purpose": "cheap prompt-side baseline suite for theory shortcut diagnosis and retry gating"
    },
    {
      "name": "action_locus_rewrite_pairs",
      "path": "projects/MOREBENCH/phase_02/outputs/action_locus_rewrite_pairs.jsonl",
      "row_count": 10,
      "rows_with_all_placeholders_substituted": 10,
      "controls_structurally_matched_to_target": 10,
      "known_bugs": [],
      "purpose": "starter matched advisor/agent rewrite batch built only from coherent agent-owned scenarios"
    }
  ],
  "residual_repairs_needed": [
    "another theory prompt-side anti-shortcut iteration if a stronger alias-style diagnostic family becomes phase-03-retryable; current prompt-side retry gate remains closed",
    "generation-time theory-persistence experiment design using the semantically functional description_only family as a priming family",
    "expanded advisor/agent rewrite dataset",
    "structure-normalized prompt variants",
    "length-matched controls",
    "person-grammar controls",
    "fresh generation dataset for response-side labels",
    "behavioral smoke on augmented prompt slice"
  ],
  "prompt_side_retry_gate": {
    "artifact": "projects/MOREBENCH/phase_02/outputs/theory_shortcut_preflight.json",
    "recommended_family": "alias_only",
    "readout_type": "diagnostic_alias_family",
    "retry_ready": false
  },
  "generation_prime_recommendation": {
    "artifact": "projects/MOREBENCH/phase_02/outputs/theory_shortcut_preflight.json",
    "recommended_family": "description_only",
    "read": "description_only remains semantically text-decodable and is better treated as a generation-time priming family than as a clean prompt-side retry family"
  },
  "behavioral_smoke_summary": null,
  "behavioral_smoke_artifact": null,
  "generation_protocol_artifact": "projects/MOREBENCH/phase_02/docs/02-generation-protocol.md"
}
