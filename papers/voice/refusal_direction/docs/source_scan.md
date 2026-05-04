# Source Scan

Source: `andyrdt/refusal_direction`.

The source pipeline samples harmful and harmless instructions, optionally
filters by baseline refusal scores, computes mean activation differences at
end-of-instruction positions, selects the most effective direction on
validation rows, then evaluates baseline, ablation, and activation-addition
completions plus CE loss.

Xenon mapping:

- `RefusalDirectionSpec` computes harmful-minus-harmless directions.
- `RefusalScoreSpec` scores validation rows.
- `RefusalDirectionSelectionSpec` selects a direction layer by projection gap.
- `RefusalAblationSubspaceSpec` converts the direction for `ProjectOutPatch`.
- `AddDirectionPatch` models activation addition.

Deviation: the smoke selector uses projection separation, not external
jailbreak-evaluator scores. Real runs should add behavior evaluators before
promoting causal claims.
