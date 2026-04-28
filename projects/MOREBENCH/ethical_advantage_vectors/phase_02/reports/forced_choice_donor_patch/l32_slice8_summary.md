# L32 donor-patching slice-8 summary

Date: 2026-04-28

## Setup

- Dataset: 8 forced-choice dilemmas, 4 option-order variants each.
- Conditions: `P_ethical_choice_01` and `P_exploit_choice_01` only.
- Pairing: exact same dilemma and option order.
- Patch operator: full residual interchange from donor condition to target condition.
- Write layer: L32.
- Sites tested: `options_end`, `prompt_end`.
- Additional span tested: `decision_tail`, spanning from the final option token through the final prompt token.
- Target metric: intended option-type flip.

## Results

| Site | Direction | Rows | Intended flips | Flip rate | Notes |
|---|---:|---:|---:|---:|---|
| `options_end` | ethical donor -> exploit target | 32 | 3 | 9.4% | All flips were `order_03`, `C -> D` |
| `options_end` | exploit donor -> ethical target | 32 | 0 | 0.0% | No option-type changes |
| `prompt_end` | ethical donor -> exploit target | 32 | 0 | 0.0% | No option-type changes |
| `prompt_end` | exploit donor -> ethical target | 32 | 0 | 0.0% | No option-type changes |
| `decision_tail` | ethical donor -> exploit target | 32 | 12 | 37.5% | Whole-span patch from options endpoint through answer instruction |
| `decision_tail` | exploit donor -> ethical target | 32 | 0 | 0.0% | No option-type changes |

All baselines were on-target: exploit targets chose `self_advantage`, ethical targets chose `ethical`.
There were no malformed generations and no skipped patch cases.

## Interpretation

Single-token donor interchange does not support a strong causal ethical-vs-exploit state at L32. The isolated endpoint patches are weak or null: `options_end` ethical->exploit flipped 3/32 cases, all in the same option ordering where `self_advantage=C` and `ethical=D`; `prompt_end` was 0/32 in both directions.

The whole-span `decision_tail` intervention changes the picture: ethical donor -> exploit target flips 12/32 cases (37.5%). This suggests any causal effect is distributed across the short pre-answer decision span rather than localized to a single endpoint token.

However, the effect is strongly asymmetric. Exploit donor -> ethical target remains 0/32 even with the whole span. Current best read: the ethical/default choice basin is more stable than the exploit-instruction basin; ethical-state donor information can partially rescue exploit targets, but exploit-state donor information does not overcome the model's ethical/default behavior under the ethical prompt.

## Artifact Pointers

- Options-end report: `report_54e5e13e8232_06e5e653`
- Prompt-end report: `report_2328167e3baf_7e158f90`
- Decision-tail report: `report_e938b9b36464_bbcab917`
- Workflow: `projects/MOREBENCH/ethical_advantage_vectors/phase_02/specs/forced_choice_donor_patching_workflow.py`
