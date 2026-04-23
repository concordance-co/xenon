# MoReBench Phase 02 Augmentation Scaffold

## Bottom Line

Phase 02 now materializes real augmentation data instead of only artifact shells.
It remains a partial repair phase, but the materialized slice is now usable:

- `150` theory-exposed prompts with framework-specific anchors
- `30` structurally matched neutral controls
- `150` same-label wording variants
- `10` action-locus rewrite pairs from coherent agent-owned scenarios

## Why This Is Still Partial

- theory is now meaningfully augmented
- action-locus has a repaired starter batch, not a complete repair
- structure, length, and person-grammar controls are still pending
- fresh generations are still pending

- behavioral smoke completed on `20` prompts with manual pass rate `1.0`

## Current Repair Matrix

- `theory_not_prompt_exposed` -> `theory_prompt_exposure` (`materialized`)
- `source_role_aliasing` -> `advisor_agent_role_swaps` (`partially_materialized`)
- `prompt_wrapper_imbalance` -> `wrapper_normalization_controls` (`materialized`)
- `source_type_aliasing` -> `structure_normalization` (`not_started`)
- `length_variation` -> `length_matched_controls` (`not_started`)
- `person_grammar_variation` -> `person_grammar_controls` (`not_started`)
- `context_missingness_and_topic_imbalance` -> `context_completion_and_balancing` (`not_started`)

## Artifact Pointers

- theory group manifest: `phase_02/outputs/theory_group_manifest.json`
- theory prompts: `phase_02/outputs/theory_prompt_augmentation_examples.jsonl`
- theory wording variants: `phase_02/outputs/theory_wording_variant_examples.jsonl`
- theory neutral controls: `phase_02/outputs/theory_control_augmentation_examples.jsonl`
- action-locus rewrite pairs: `phase_02/outputs/action_locus_rewrite_pairs.jsonl`
- behavioral smoke raw results: `phase_02/outputs/behavioral_smoke_results.json`
