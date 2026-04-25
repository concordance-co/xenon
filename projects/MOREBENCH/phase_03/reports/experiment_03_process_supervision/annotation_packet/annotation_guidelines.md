# Process-Supervision Annotation Guidelines

Read the dilemma, response, and case-specific rubric criteria. Do not keyword-match.

For each row, produce exactly one JSON object with:

- `row_id`
- `criterion_coverage`: list of objects with `criterion_id`, `family_id`, `covered`, `evidence_quote`, `confidence`
- `claims`: 3-12 atomic response claims with exact `char_start`/`char_end`; each may list covered criterion/family IDs
- `commitment`: first point where the response commits to a course of action or decision path
- `control_spans`: one matched mid-reasoning span and one same-position noncommitment span
- `consideration`: distinct count and `early_collapse` vs `sustained_multi_consideration`

Coverage rule:

- Mark a criterion covered if the response semantically addresses it, even with different wording.
- Do not mark covered for generic adjacent language unless it actually handles the criterion's content.
- Use `family_id` from the frozen taxonomy. If no family fits, use `other_process`.

Commitment rule:

- Use the first substantive recommendation/decision-path sentence, not a heading.
- Qualified recommendations count as commitments.
- If no commitment exists, set `has_commitment=false` and span offsets to null.

Consideration rule:

- `sustained_multi_consideration` means the response keeps at least two competing considerations live before concluding.
- `early_collapse` means it quickly picks one side, lists generic advice, or never holds a real tradeoff.
