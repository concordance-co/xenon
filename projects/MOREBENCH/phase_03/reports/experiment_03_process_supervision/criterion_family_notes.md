# Criterion Family Taxonomy Notes

Frozen date: 2026-04-24

Scope: MoReBench phase 03 process-supervision annotation pass.

Inputs used:

- `annotation_packet/criteria.jsonl`
- `annotation_packet/annotation_guidelines.md`
- `docs/03-process-probe-precommitment.md`

Inputs not used: row-level annotation results, reviewer merge results, probe results, or any downstream performance signal.

## Taxonomy Summary

The frozen taxonomy contains 25 families:

- 11 dilemma/process content families requested for this pass: `close_relationship_obligation`, `institutional_policy_constraint`, `privacy_monitoring_conflict`, `public_harm_resource_tradeoff`, `disclosure_transparency_conflict`, `fairness_access_conflict`, `authority_constraint`, `loyalty_relationship`, `public_safety`, `autonomy_boundary`, `uncertainty_incomplete_info`.
- 13 rubric/process families for coverage structure: `summarize_core_dilemma`, `identify_stakeholders`, `identify_options`, `explain_consequences`, `weigh_tradeoffs`, `decision_procedure`, `concrete_recommendation`, `tailored_next_steps`, `risk_mitigation`, `respect_values_autonomy`, `preserve_relationships`, `legal_policy_compliance`, `epistemic_uncertainty`.
- 1 residual family: `other_process`.

The family IDs are semantic, stable, and intended to be broad enough for support while still separating process structure relevant to process-supervision probing.

## Support Estimation

Estimated support values in `criterion_families.json` are approximate counts from simple semantic and keyword screens over criterion titles and dimensions. The screens were intentionally pre-outcome and transparent. They are not merged annotation labels, and they are not probe eligibility decisions.

Because a title can contain several valid cues, support screens are not a claim that every raw criterion has already been assigned exclusively. During annotation, each raw criterion should receive one primary family. The support estimates are intended to indicate likely coverage scale and representative examples, not final class priors.

## Primary Assignment Guidance

Use semantic fit rather than literal keyword matching. If several families appear plausible, choose the family that captures the criterion's main annotative demand.

- Prefer `summarize_core_dilemma` when the criterion primarily asks the response to restate the central conflict, even if the conflict mentions a domain family.
- Prefer `identify_options`, `explain_consequences`, `weigh_tradeoffs`, `decision_procedure`, or `concrete_recommendation` when the criterion is mainly about the reasoning operation or answer structure.
- Prefer content families when the criterion is about a specific moral consideration, such as privacy, authority, public safety, autonomy, fairness, loyalty, or institutional constraints.
- Use `uncertainty_incomplete_info` for missing-fact or unresolved-context considerations in the dilemma.
- Use `epistemic_uncertainty` for response-quality requirements such as avoiding assumptions, stating confidence, seeking information, or conditioning advice on evidence.
- Use `institutional_policy_constraint` for organizational and professional setting constraints.
- Use `legal_policy_compliance` for law, formal rules, regulatory compliance, confidentiality, and illegal-action avoidance.
- Use `loyalty_relationship` when the central issue is relational trust, friendship, or loyalty.
- Use `preserve_relationships` when the response is asked to protect, repair, or avoid damaging a relationship as an action-quality consideration.
- Use `autonomy_boundary` for consent, rights, boundaries, and agency as the dilemma content.
- Use `respect_values_autonomy` for respectful handling of values, beliefs, dignity, and preferences as response quality.
- Use `other_process` only when no more specific family captures the criterion.

## Caveats

The largest support screens are broad process families such as `explain_consequences` and `decision_procedure`; this reflects the rubric style, where many criteria ask for causal reasoning and explicit process. These broad families should not absorb more specific criteria when a more diagnostic family is available.

`other_process` has a high residual estimate because the packet includes meta-rubric, prompt-quality, source-use, and unusual case-specific criteria. Annotators should treat it as a last resort rather than a miscellaneous default.

All family names, descriptions, and support estimates are frozen before annotation merge and probing. They must not be changed based on probe performance, labelability, or downstream model behavior.
