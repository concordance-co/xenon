# Experiment 02 Deontology vs Virtue Type Holdout

Type-stratified text diagnostic on `deontology` vs `virtue_ethics`, comparing source-family and context holdouts against the earlier random dilemma-group holdout text baseline of `1.00` AUROC.


## Coverage

- cue rows: `360`

- raw response rows from generation artifact `generation_run_1_3d4009fb21d8`: `60`

- response minimum test count per class: `5`


### Cue Source-Family x Prime

| Source family | Deontology | Virtue |
| --- | ---: | ---: |
| `ai_risk_dilemmas` | `60` | `60` |
| `daily_dilemmas` | `60` | `60` |
| `expert_written_collab` | `60` | `60` |

### Cue Context x Prime

| Context | Deontology | Virtue |
| --- | ---: | ---: |
| `Animal & Environment` | `12` | `12` |
| `Bioethics & Healthcare` | `18` | `18` |
| `Business & Workplace` | `18` | `18` |
| `Education` | `24` | `24` |
| `Entertainment` | `6` | `6` |
| `Interpersonal relationship` | `36` | `36` |
| `Others` | `12` | `12` |
| `Professional Ethics` | `12` | `12` |
| `Right & Duty & Justice` | `18` | `18` |
| `Science & Techonology` | `6` | `6` |
| `Sports` | `6` | `6` |
| `Transport` | `12` | `12` |

### Cue Source-Family Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `ai_risk_dilemmas` | `240` | `120` | `d=120, v=120` | `d=60, v=60` | `1.0` | `1.0` |
| `daily_dilemmas` | `240` | `120` | `d=120, v=120` | `d=60, v=60` | `1.0` | `1.0` |
| `expert_written_collab` | `240` | `120` | `d=120, v=120` | `d=60, v=60` | `1.0` | `1.0` |

- mean AUROC across evaluated holdouts: `1.0`

### Cue Context Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `Animal & Environment` | `336` | `24` | `d=168, v=168` | `d=12, v=12` | `1.0` | `1.0` |
| `Bioethics & Healthcare` | `324` | `36` | `d=162, v=162` | `d=18, v=18` | `1.0` | `1.0` |
| `Business & Workplace` | `324` | `36` | `d=162, v=162` | `d=18, v=18` | `1.0` | `1.0` |
| `Education` | `312` | `48` | `d=156, v=156` | `d=24, v=24` | `1.0` | `1.0` |
| `Entertainment` | `348` | `12` | `d=174, v=174` | `d=6, v=6` | `1.0` | `1.0` |
| `Interpersonal relationship` | `288` | `72` | `d=144, v=144` | `d=36, v=36` | `1.0` | `1.0` |
| `Others` | `336` | `24` | `d=168, v=168` | `d=12, v=12` | `1.0` | `1.0` |
| `Professional Ethics` | `336` | `24` | `d=168, v=168` | `d=12, v=12` | `1.0` | `1.0` |
| `Right & Duty & Justice` | `324` | `36` | `d=162, v=162` | `d=18, v=18` | `1.0` | `1.0` |
| `Science & Techonology` | `348` | `12` | `d=174, v=174` | `d=6, v=6` | `1.0` | `1.0` |
| `Sports` | `348` | `12` | `d=174, v=174` | `d=6, v=6` | `1.0` | `1.0` |
| `Transport` | `336` | `24` | `d=168, v=168` | `d=12, v=12` | `1.0` | `1.0` |

- mean AUROC across evaluated holdouts: `1.0`

## Response Diagnostic: `filter_on`

- viewport `full` row count: `50`
- viewport `last_75` row count: `50`
- viewport `last_25` row count: `50`
- viewport `mid_50` row count: `50`

### `filter_on` / `full` Source-Family x Prime

| Source family | Deontology | Virtue |
| --- | ---: | ---: |
| `ai_risk_dilemmas` | `9` | `8` |
| `daily_dilemmas` | `9` | `7` |
| `expert_written_collab` | `8` | `9` |

### `filter_on` / `full` Context x Prime

| Context | Deontology | Virtue |
| --- | ---: | ---: |
| `Animal & Environment` | `2` | `1` |
| `Bioethics & Healthcare` | `2` | `2` |
| `Business & Workplace` | `2` | `3` |
| `Education` | `3` | `4` |
| `Entertainment` | `1` | `1` |
| `Interpersonal relationship` | `5` | `4` |
| `Others` | `2` | `2` |
| `Professional Ethics` | `2` | `2` |
| `Right & Duty & Justice` | `3` | `3` |
| `Science & Techonology` | `1` | `1` |
| `Sports` | `1` | `0` |
| `Transport` | `2` | `1` |

### `filter_on` / `full` Response Source-Family Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `ai_risk_dilemmas` | `33` | `17` | `d=17, v=16` | `d=9, v=8` | `1.0` | `1.0` |
| `daily_dilemmas` | `34` | `16` | `d=17, v=17` | `d=9, v=7` | `1.0` | `1.0` |
| `expert_written_collab` | `33` | `17` | `d=18, v=15` | `d=8, v=9` | `1.0` | `1.0` |

- mean AUROC across evaluated holdouts: `1.0`

### `filter_on` / `full` Response Context Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| _none_ |  |  |  |  |  |  |

Skipped holdouts:
- `Animal & Environment`: `test_class_count_below_5` (train d/v `24/23`, test d/v `2/1`)
- `Bioethics & Healthcare`: `test_class_count_below_5` (train d/v `24/22`, test d/v `2/2`)
- `Business & Workplace`: `test_class_count_below_5` (train d/v `24/21`, test d/v `2/3`)
- `Education`: `test_class_count_below_5` (train d/v `23/20`, test d/v `3/4`)
- `Entertainment`: `test_class_count_below_5` (train d/v `25/23`, test d/v `1/1`)
- `Interpersonal relationship`: `test_class_count_below_5` (train d/v `21/20`, test d/v `5/4`)
- `Others`: `test_class_count_below_5` (train d/v `24/22`, test d/v `2/2`)
- `Professional Ethics`: `test_class_count_below_5` (train d/v `24/22`, test d/v `2/2`)
- `Right & Duty & Justice`: `test_class_count_below_5` (train d/v `23/21`, test d/v `3/3`)
- `Science & Techonology`: `test_class_count_below_5` (train d/v `25/23`, test d/v `1/1`)
- `Sports`: `missing_class` (train d/v `25/24`, test d/v `1/0`)
- `Transport`: `test_class_count_below_5` (train d/v `24/23`, test d/v `2/1`)

- mean AUROC across evaluated holdouts: `None`

### `filter_on` / `last_75` Source-Family x Prime

| Source family | Deontology | Virtue |
| --- | ---: | ---: |
| `ai_risk_dilemmas` | `9` | `8` |
| `daily_dilemmas` | `9` | `7` |
| `expert_written_collab` | `8` | `9` |

### `filter_on` / `last_75` Context x Prime

| Context | Deontology | Virtue |
| --- | ---: | ---: |
| `Animal & Environment` | `2` | `1` |
| `Bioethics & Healthcare` | `2` | `2` |
| `Business & Workplace` | `2` | `3` |
| `Education` | `3` | `4` |
| `Entertainment` | `1` | `1` |
| `Interpersonal relationship` | `5` | `4` |
| `Others` | `2` | `2` |
| `Professional Ethics` | `2` | `2` |
| `Right & Duty & Justice` | `3` | `3` |
| `Science & Techonology` | `1` | `1` |
| `Sports` | `1` | `0` |
| `Transport` | `2` | `1` |

### `filter_on` / `last_75` Response Source-Family Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `ai_risk_dilemmas` | `33` | `17` | `d=17, v=16` | `d=9, v=8` | `1.0` | `1.0` |
| `daily_dilemmas` | `34` | `16` | `d=17, v=17` | `d=9, v=7` | `1.0` | `0.9286` |
| `expert_written_collab` | `33` | `17` | `d=18, v=15` | `d=8, v=9` | `1.0` | `1.0` |

- mean AUROC across evaluated holdouts: `1.0`

### `filter_on` / `last_75` Response Context Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| _none_ |  |  |  |  |  |  |

Skipped holdouts:
- `Animal & Environment`: `test_class_count_below_5` (train d/v `24/23`, test d/v `2/1`)
- `Bioethics & Healthcare`: `test_class_count_below_5` (train d/v `24/22`, test d/v `2/2`)
- `Business & Workplace`: `test_class_count_below_5` (train d/v `24/21`, test d/v `2/3`)
- `Education`: `test_class_count_below_5` (train d/v `23/20`, test d/v `3/4`)
- `Entertainment`: `test_class_count_below_5` (train d/v `25/23`, test d/v `1/1`)
- `Interpersonal relationship`: `test_class_count_below_5` (train d/v `21/20`, test d/v `5/4`)
- `Others`: `test_class_count_below_5` (train d/v `24/22`, test d/v `2/2`)
- `Professional Ethics`: `test_class_count_below_5` (train d/v `24/22`, test d/v `2/2`)
- `Right & Duty & Justice`: `test_class_count_below_5` (train d/v `23/21`, test d/v `3/3`)
- `Science & Techonology`: `test_class_count_below_5` (train d/v `25/23`, test d/v `1/1`)
- `Sports`: `missing_class` (train d/v `25/24`, test d/v `1/0`)
- `Transport`: `test_class_count_below_5` (train d/v `24/23`, test d/v `2/1`)

- mean AUROC across evaluated holdouts: `None`

### `filter_on` / `last_25` Source-Family x Prime

| Source family | Deontology | Virtue |
| --- | ---: | ---: |
| `ai_risk_dilemmas` | `9` | `8` |
| `daily_dilemmas` | `9` | `7` |
| `expert_written_collab` | `8` | `9` |

### `filter_on` / `last_25` Context x Prime

| Context | Deontology | Virtue |
| --- | ---: | ---: |
| `Animal & Environment` | `2` | `1` |
| `Bioethics & Healthcare` | `2` | `2` |
| `Business & Workplace` | `2` | `3` |
| `Education` | `3` | `4` |
| `Entertainment` | `1` | `1` |
| `Interpersonal relationship` | `5` | `4` |
| `Others` | `2` | `2` |
| `Professional Ethics` | `2` | `2` |
| `Right & Duty & Justice` | `3` | `3` |
| `Science & Techonology` | `1` | `1` |
| `Sports` | `1` | `0` |
| `Transport` | `2` | `1` |

### `filter_on` / `last_25` Response Source-Family Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `ai_risk_dilemmas` | `33` | `17` | `d=17, v=16` | `d=9, v=8` | `0.9444` | `0.8194` |
| `daily_dilemmas` | `34` | `16` | `d=17, v=17` | `d=9, v=7` | `0.8571` | `0.8571` |
| `expert_written_collab` | `33` | `17` | `d=18, v=15` | `d=8, v=9` | `0.9722` | `0.8889` |

- mean AUROC across evaluated holdouts: `0.9246`

### `filter_on` / `last_25` Response Context Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| _none_ |  |  |  |  |  |  |

Skipped holdouts:
- `Animal & Environment`: `test_class_count_below_5` (train d/v `24/23`, test d/v `2/1`)
- `Bioethics & Healthcare`: `test_class_count_below_5` (train d/v `24/22`, test d/v `2/2`)
- `Business & Workplace`: `test_class_count_below_5` (train d/v `24/21`, test d/v `2/3`)
- `Education`: `test_class_count_below_5` (train d/v `23/20`, test d/v `3/4`)
- `Entertainment`: `test_class_count_below_5` (train d/v `25/23`, test d/v `1/1`)
- `Interpersonal relationship`: `test_class_count_below_5` (train d/v `21/20`, test d/v `5/4`)
- `Others`: `test_class_count_below_5` (train d/v `24/22`, test d/v `2/2`)
- `Professional Ethics`: `test_class_count_below_5` (train d/v `24/22`, test d/v `2/2`)
- `Right & Duty & Justice`: `test_class_count_below_5` (train d/v `23/21`, test d/v `3/3`)
- `Science & Techonology`: `test_class_count_below_5` (train d/v `25/23`, test d/v `1/1`)
- `Sports`: `missing_class` (train d/v `25/24`, test d/v `1/0`)
- `Transport`: `test_class_count_below_5` (train d/v `24/23`, test d/v `2/1`)

- mean AUROC across evaluated holdouts: `None`

### `filter_on` / `mid_50` Source-Family x Prime

| Source family | Deontology | Virtue |
| --- | ---: | ---: |
| `ai_risk_dilemmas` | `9` | `8` |
| `daily_dilemmas` | `9` | `7` |
| `expert_written_collab` | `8` | `9` |

### `filter_on` / `mid_50` Context x Prime

| Context | Deontology | Virtue |
| --- | ---: | ---: |
| `Animal & Environment` | `2` | `1` |
| `Bioethics & Healthcare` | `2` | `2` |
| `Business & Workplace` | `2` | `3` |
| `Education` | `3` | `4` |
| `Entertainment` | `1` | `1` |
| `Interpersonal relationship` | `5` | `4` |
| `Others` | `2` | `2` |
| `Professional Ethics` | `2` | `2` |
| `Right & Duty & Justice` | `3` | `3` |
| `Science & Techonology` | `1` | `1` |
| `Sports` | `1` | `0` |
| `Transport` | `2` | `1` |

### `filter_on` / `mid_50` Response Source-Family Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `ai_risk_dilemmas` | `33` | `17` | `d=17, v=16` | `d=9, v=8` | `1.0` | `1.0` |
| `daily_dilemmas` | `34` | `16` | `d=17, v=17` | `d=9, v=7` | `1.0` | `1.0` |
| `expert_written_collab` | `33` | `17` | `d=18, v=15` | `d=8, v=9` | `1.0` | `1.0` |

- mean AUROC across evaluated holdouts: `1.0`

### `filter_on` / `mid_50` Response Context Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| _none_ |  |  |  |  |  |  |

Skipped holdouts:
- `Animal & Environment`: `test_class_count_below_5` (train d/v `24/23`, test d/v `2/1`)
- `Bioethics & Healthcare`: `test_class_count_below_5` (train d/v `24/22`, test d/v `2/2`)
- `Business & Workplace`: `test_class_count_below_5` (train d/v `24/21`, test d/v `2/3`)
- `Education`: `test_class_count_below_5` (train d/v `23/20`, test d/v `3/4`)
- `Entertainment`: `test_class_count_below_5` (train d/v `25/23`, test d/v `1/1`)
- `Interpersonal relationship`: `test_class_count_below_5` (train d/v `21/20`, test d/v `5/4`)
- `Others`: `test_class_count_below_5` (train d/v `24/22`, test d/v `2/2`)
- `Professional Ethics`: `test_class_count_below_5` (train d/v `24/22`, test d/v `2/2`)
- `Right & Duty & Justice`: `test_class_count_below_5` (train d/v `23/21`, test d/v `3/3`)
- `Science & Techonology`: `test_class_count_below_5` (train d/v `25/23`, test d/v `1/1`)
- `Sports`: `missing_class` (train d/v `25/24`, test d/v `1/0`)
- `Transport`: `test_class_count_below_5` (train d/v `24/23`, test d/v `2/1`)

- mean AUROC across evaluated holdouts: `None`

## Response Diagnostic: `filter_off`

- viewport `full` row count: `60`
- viewport `last_75` row count: `60`
- viewport `last_25` row count: `60`
- viewport `mid_50` row count: `60`

### `filter_off` / `full` Source-Family x Prime

| Source family | Deontology | Virtue |
| --- | ---: | ---: |
| `ai_risk_dilemmas` | `10` | `10` |
| `daily_dilemmas` | `10` | `10` |
| `expert_written_collab` | `10` | `10` |

### `filter_off` / `full` Context x Prime

| Context | Deontology | Virtue |
| --- | ---: | ---: |
| `Animal & Environment` | `2` | `2` |
| `Bioethics & Healthcare` | `3` | `3` |
| `Business & Workplace` | `3` | `3` |
| `Education` | `4` | `4` |
| `Entertainment` | `1` | `1` |
| `Interpersonal relationship` | `6` | `6` |
| `Others` | `2` | `2` |
| `Professional Ethics` | `2` | `2` |
| `Right & Duty & Justice` | `3` | `3` |
| `Science & Techonology` | `1` | `1` |
| `Sports` | `1` | `1` |
| `Transport` | `2` | `2` |

### `filter_off` / `full` Response Source-Family Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `ai_risk_dilemmas` | `40` | `20` | `d=20, v=20` | `d=10, v=10` | `1.0` | `1.0` |
| `daily_dilemmas` | `40` | `20` | `d=20, v=20` | `d=10, v=10` | `1.0` | `1.0` |
| `expert_written_collab` | `40` | `20` | `d=20, v=20` | `d=10, v=10` | `1.0` | `1.0` |

- mean AUROC across evaluated holdouts: `1.0`

### `filter_off` / `full` Response Context Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `Interpersonal relationship` | `48` | `12` | `d=24, v=24` | `d=6, v=6` | `1.0` | `1.0` |

Skipped holdouts:
- `Animal & Environment`: `test_class_count_below_5` (train d/v `28/28`, test d/v `2/2`)
- `Bioethics & Healthcare`: `test_class_count_below_5` (train d/v `27/27`, test d/v `3/3`)
- `Business & Workplace`: `test_class_count_below_5` (train d/v `27/27`, test d/v `3/3`)
- `Education`: `test_class_count_below_5` (train d/v `26/26`, test d/v `4/4`)
- `Entertainment`: `test_class_count_below_5` (train d/v `29/29`, test d/v `1/1`)
- `Others`: `test_class_count_below_5` (train d/v `28/28`, test d/v `2/2`)
- `Professional Ethics`: `test_class_count_below_5` (train d/v `28/28`, test d/v `2/2`)
- `Right & Duty & Justice`: `test_class_count_below_5` (train d/v `27/27`, test d/v `3/3`)
- `Science & Techonology`: `test_class_count_below_5` (train d/v `29/29`, test d/v `1/1`)
- `Sports`: `test_class_count_below_5` (train d/v `29/29`, test d/v `1/1`)
- `Transport`: `test_class_count_below_5` (train d/v `28/28`, test d/v `2/2`)

- mean AUROC across evaluated holdouts: `1.0`

### `filter_off` / `last_75` Source-Family x Prime

| Source family | Deontology | Virtue |
| --- | ---: | ---: |
| `ai_risk_dilemmas` | `10` | `10` |
| `daily_dilemmas` | `10` | `10` |
| `expert_written_collab` | `10` | `10` |

### `filter_off` / `last_75` Context x Prime

| Context | Deontology | Virtue |
| --- | ---: | ---: |
| `Animal & Environment` | `2` | `2` |
| `Bioethics & Healthcare` | `3` | `3` |
| `Business & Workplace` | `3` | `3` |
| `Education` | `4` | `4` |
| `Entertainment` | `1` | `1` |
| `Interpersonal relationship` | `6` | `6` |
| `Others` | `2` | `2` |
| `Professional Ethics` | `2` | `2` |
| `Right & Duty & Justice` | `3` | `3` |
| `Science & Techonology` | `1` | `1` |
| `Sports` | `1` | `1` |
| `Transport` | `2` | `2` |

### `filter_off` / `last_75` Response Source-Family Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `ai_risk_dilemmas` | `40` | `20` | `d=20, v=20` | `d=10, v=10` | `1.0` | `1.0` |
| `daily_dilemmas` | `40` | `20` | `d=20, v=20` | `d=10, v=10` | `1.0` | `0.95` |
| `expert_written_collab` | `40` | `20` | `d=20, v=20` | `d=10, v=10` | `1.0` | `1.0` |

- mean AUROC across evaluated holdouts: `1.0`

### `filter_off` / `last_75` Response Context Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `Interpersonal relationship` | `48` | `12` | `d=24, v=24` | `d=6, v=6` | `1.0` | `1.0` |

Skipped holdouts:
- `Animal & Environment`: `test_class_count_below_5` (train d/v `28/28`, test d/v `2/2`)
- `Bioethics & Healthcare`: `test_class_count_below_5` (train d/v `27/27`, test d/v `3/3`)
- `Business & Workplace`: `test_class_count_below_5` (train d/v `27/27`, test d/v `3/3`)
- `Education`: `test_class_count_below_5` (train d/v `26/26`, test d/v `4/4`)
- `Entertainment`: `test_class_count_below_5` (train d/v `29/29`, test d/v `1/1`)
- `Others`: `test_class_count_below_5` (train d/v `28/28`, test d/v `2/2`)
- `Professional Ethics`: `test_class_count_below_5` (train d/v `28/28`, test d/v `2/2`)
- `Right & Duty & Justice`: `test_class_count_below_5` (train d/v `27/27`, test d/v `3/3`)
- `Science & Techonology`: `test_class_count_below_5` (train d/v `29/29`, test d/v `1/1`)
- `Sports`: `test_class_count_below_5` (train d/v `29/29`, test d/v `1/1`)
- `Transport`: `test_class_count_below_5` (train d/v `28/28`, test d/v `2/2`)

- mean AUROC across evaluated holdouts: `1.0`

### `filter_off` / `last_25` Source-Family x Prime

| Source family | Deontology | Virtue |
| --- | ---: | ---: |
| `ai_risk_dilemmas` | `10` | `10` |
| `daily_dilemmas` | `10` | `10` |
| `expert_written_collab` | `10` | `10` |

### `filter_off` / `last_25` Context x Prime

| Context | Deontology | Virtue |
| --- | ---: | ---: |
| `Animal & Environment` | `2` | `2` |
| `Bioethics & Healthcare` | `3` | `3` |
| `Business & Workplace` | `3` | `3` |
| `Education` | `4` | `4` |
| `Entertainment` | `1` | `1` |
| `Interpersonal relationship` | `6` | `6` |
| `Others` | `2` | `2` |
| `Professional Ethics` | `2` | `2` |
| `Right & Duty & Justice` | `3` | `3` |
| `Science & Techonology` | `1` | `1` |
| `Sports` | `1` | `1` |
| `Transport` | `2` | `2` |

### `filter_off` / `last_25` Response Source-Family Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `ai_risk_dilemmas` | `40` | `20` | `d=20, v=20` | `d=10, v=10` | `0.96` | `0.85` |
| `daily_dilemmas` | `40` | `20` | `d=20, v=20` | `d=10, v=10` | `0.91` | `0.9` |
| `expert_written_collab` | `40` | `20` | `d=20, v=20` | `d=10, v=10` | `0.96` | `0.85` |

- mean AUROC across evaluated holdouts: `0.9433`

### `filter_off` / `last_25` Response Context Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `Interpersonal relationship` | `48` | `12` | `d=24, v=24` | `d=6, v=6` | `1.0` | `1.0` |

Skipped holdouts:
- `Animal & Environment`: `test_class_count_below_5` (train d/v `28/28`, test d/v `2/2`)
- `Bioethics & Healthcare`: `test_class_count_below_5` (train d/v `27/27`, test d/v `3/3`)
- `Business & Workplace`: `test_class_count_below_5` (train d/v `27/27`, test d/v `3/3`)
- `Education`: `test_class_count_below_5` (train d/v `26/26`, test d/v `4/4`)
- `Entertainment`: `test_class_count_below_5` (train d/v `29/29`, test d/v `1/1`)
- `Others`: `test_class_count_below_5` (train d/v `28/28`, test d/v `2/2`)
- `Professional Ethics`: `test_class_count_below_5` (train d/v `28/28`, test d/v `2/2`)
- `Right & Duty & Justice`: `test_class_count_below_5` (train d/v `27/27`, test d/v `3/3`)
- `Science & Techonology`: `test_class_count_below_5` (train d/v `29/29`, test d/v `1/1`)
- `Sports`: `test_class_count_below_5` (train d/v `29/29`, test d/v `1/1`)
- `Transport`: `test_class_count_below_5` (train d/v `28/28`, test d/v `2/2`)

- mean AUROC across evaluated holdouts: `1.0`

### `filter_off` / `mid_50` Source-Family x Prime

| Source family | Deontology | Virtue |
| --- | ---: | ---: |
| `ai_risk_dilemmas` | `10` | `10` |
| `daily_dilemmas` | `10` | `10` |
| `expert_written_collab` | `10` | `10` |

### `filter_off` / `mid_50` Context x Prime

| Context | Deontology | Virtue |
| --- | ---: | ---: |
| `Animal & Environment` | `2` | `2` |
| `Bioethics & Healthcare` | `3` | `3` |
| `Business & Workplace` | `3` | `3` |
| `Education` | `4` | `4` |
| `Entertainment` | `1` | `1` |
| `Interpersonal relationship` | `6` | `6` |
| `Others` | `2` | `2` |
| `Professional Ethics` | `2` | `2` |
| `Right & Duty & Justice` | `3` | `3` |
| `Science & Techonology` | `1` | `1` |
| `Sports` | `1` | `1` |
| `Transport` | `2` | `2` |

### `filter_off` / `mid_50` Response Source-Family Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `ai_risk_dilemmas` | `40` | `20` | `d=20, v=20` | `d=10, v=10` | `1.0` | `1.0` |
| `daily_dilemmas` | `40` | `20` | `d=20, v=20` | `d=10, v=10` | `1.0` | `1.0` |
| `expert_written_collab` | `40` | `20` | `d=20, v=20` | `d=10, v=10` | `1.0` | `1.0` |

- mean AUROC across evaluated holdouts: `1.0`

### `filter_off` / `mid_50` Response Context Holdout

| Holdout | Train | Test | Train counts | Test counts | AUROC | BA |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| `Interpersonal relationship` | `48` | `12` | `d=24, v=24` | `d=6, v=6` | `1.0` | `1.0` |

Skipped holdouts:
- `Animal & Environment`: `test_class_count_below_5` (train d/v `28/28`, test d/v `2/2`)
- `Bioethics & Healthcare`: `test_class_count_below_5` (train d/v `27/27`, test d/v `3/3`)
- `Business & Workplace`: `test_class_count_below_5` (train d/v `27/27`, test d/v `3/3`)
- `Education`: `test_class_count_below_5` (train d/v `26/26`, test d/v `4/4`)
- `Entertainment`: `test_class_count_below_5` (train d/v `29/29`, test d/v `1/1`)
- `Others`: `test_class_count_below_5` (train d/v `28/28`, test d/v `2/2`)
- `Professional Ethics`: `test_class_count_below_5` (train d/v `28/28`, test d/v `2/2`)
- `Right & Duty & Justice`: `test_class_count_below_5` (train d/v `27/27`, test d/v `3/3`)
- `Science & Techonology`: `test_class_count_below_5` (train d/v `29/29`, test d/v `1/1`)
- `Sports`: `test_class_count_below_5` (train d/v `29/29`, test d/v `1/1`)
- `Transport`: `test_class_count_below_5` (train d/v `28/28`, test d/v `2/2`)

- mean AUROC across evaluated holdouts: `1.0`

## Interpretation

The comparison point is the earlier random dilemma-group holdout text baseline of `1.0` AUROC. Cue-side type holdouts do not produce any AUROC at or below `0.85`. Across response-side diagnostics, any viewport/holdout/filter cell at or below `0.85` is `False`. If every evaluated response cell stays at or above `0.95` (`False`), then neither dropping the strict copy filter nor shifting to later windows meaningfully lowers the deontology-vs-virtue text ceiling on the existing response data.

## Decision Surface

- any cue holdout <= `0.85`: `False`
- all cue holdouts >= `0.95`: `True`
- any evaluated response cell <= `0.85`: `False`
- all evaluated response cells >= `0.95`: `False`
- per user request, step 2 was executed regardless of the cue result.

