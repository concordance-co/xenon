# Experiment 02 Deontology vs Virtue Group Holdout

- capture dataset artifact: `transform_1_4a60e2ca`
- capture artifact: `capture_1_34cdfd7923d9`
- prime family: `description_only`
- target: `deontology` vs `virtue_ethics`
- usable complete groups after old strict filtering: `21`
- usable rows: `42`

## Group Coverage
- complete groups: `theory_group_002, theory_group_004, theory_group_005, theory_group_006, theory_group_007, theory_group_008, theory_group_009, theory_group_010, theory_group_011, theory_group_013, theory_group_016, theory_group_017, theory_group_019, theory_group_020, theory_group_021, theory_group_024, theory_group_025, theory_group_026, theory_group_027, theory_group_028, theory_group_030`
- class counts: `{'deontology': 21, 'virtue_ethics': 21}`

## Group-Holdout Baselines
- char-TF-IDF text AUROC: `1.0`
- char-TF-IDF text balanced accuracy: `1.0`
- length-only AUROC: `0.6259`
- length-only balanced accuracy: `0.5952`

## Layer Results
- layer `0`: probe AUROC `1.0`, BA `1.0`, delta vs text `0.0`, delta vs length `0.3741`
- layer `4`: probe AUROC `1.0`, BA `1.0`, delta vs text `0.0`, delta vs length `0.3741`
- layer `8`: probe AUROC `1.0`, BA `1.0`, delta vs text `0.0`, delta vs length `0.3741`
- layer `16`: probe AUROC `1.0`, BA `1.0`, delta vs text `0.0`, delta vs length `0.3741`
- layer `24`: probe AUROC `1.0`, BA `1.0`, delta vs text `0.0`, delta vs length `0.3741`
- layer `32`: probe AUROC `1.0`, BA `1.0`, delta vs text `0.0`, delta vs length `0.3741`
- layer `40`: probe AUROC `1.0`, BA `1.0`, delta vs text `0.0`, delta vs length `0.3741`
- layer `44`: probe AUROC `1.0`, BA `1.0`, delta vs text `0.0`, delta vs length `0.3741`

## Best Layer
- best layer: `0`
- probe AUROC: `1.0`
- probe BA: `1.0`
- delta vs text AUROC: `0.0`
- delta vs length AUROC: `0.3741`

## Read
- This is a true dilemma-group holdout over the surviving complete groups, not the earlier bank-only split.
- It only answers the question on the old strict description-only capture substrate.
- It does not test name_only or full 30-group coverage because those activations are not available in usable form.
