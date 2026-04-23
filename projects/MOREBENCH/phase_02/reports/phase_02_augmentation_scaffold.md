# MoReBench Phase 02 Augmentation Scaffold

## Bottom Line

Phase 02 now materializes real augmentation data instead of only artifact shells.
It remains a partial repair phase, but the materialized slice is now usable:

- `150` legacy theory-exposed prompts with framework-specific anchors
- `30` legacy structurally matched neutral controls
- `150` legacy same-label wording variants
- `2250` new shortcut-stress-test theory prompt rows
- `180` new shortcut-stress-test theory controls
- `10` action-locus rewrite pairs from coherent agent-owned scenarios

## Why This Is Still Partial

- the legacy theory family is now explicitly treated as shortcut-dominated rather than phase-03-ready
- theory has a new repair family and prompt-side preflight, but not yet a proven clean retry slice
- action-locus has a repaired starter batch, not a complete repair
- structure, length, and person-grammar controls are still pending
- fresh generations are still pending
- behavioral smoke is still pending


## Shortcut Preflight

- legacy cue-text bag-of-words score: `1.0`
- recommended prompt-side diagnostic family: `alias_only`
- recommended generation-time priming family: `description_only`
- current read: alias_only is the strongest prompt-side diagnostic family, but the prompt-side retry gate remains closed until its strongest held-out text baseline falls further; description_only is better treated as a generation-time priming family

## Current Repair Matrix

- `theory_not_prompt_exposed` -> `theory_prompt_exposure` (`materialized`)
- `theory_lexical_shortcuts` -> `shortcut_stress_test` (`materialized`)
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
- theory repair prompts: `phase_02/outputs/theory_prompt_repair_examples.jsonl`
- theory repair controls: `phase_02/outputs/theory_prompt_repair_controls.jsonl`
- theory shortcut preflight: `phase_02/outputs/theory_shortcut_preflight.json`
- action-locus rewrite pairs: `phase_02/outputs/action_locus_rewrite_pairs.jsonl`
- behavioral smoke raw results: `phase_02/outputs/behavioral_smoke_results.json`
