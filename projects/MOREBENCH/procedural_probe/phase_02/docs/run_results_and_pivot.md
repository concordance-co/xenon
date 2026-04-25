# Phase 02 Results And Pivot

Latest completed raw-dimension run:

- run: `wr_df6945c0f649_7f46582c`
- report: `projects/MOREBENCH/procedural_probe/phase_02/reports/pipelines_v2/report_c4439498a3ab_ca5b71aa/report.md`
- examples generated: `500`
- examples kept for prompt+generated capture: `487`
- skipped length-finished generations: `13`

## Result Read

The raw rubric-dimension targets are not good headline mechanistic targets.

The label construction is highly imbalanced:

- `dominant_dimension_by_count`
  - `identifying`: `351`
  - `logical process`: `95`
  - `clear process`: `30`
  - `helpful outcome`: `24`
- `dominant_harmless_outcome_by_count`
  - single class: `no=500`

Best balanced accuracy by residual probe:

| target | prompt-end | generation-end | note |
| --- | ---: | ---: | --- |
| `dominant_dimension_by_count` | `0.2971` | `0.2976` | multiclass target is dominated by `identifying` |
| `dominant_identifying_by_count` | `0.5219` | `0.5301` | near text baseline; weak lift |
| `dominant_clear_process_by_count` | `0.5326` | `0.4955` | sparse positive class |
| `dominant_logical_process_by_count` | `0.5199` | `0.5214` | weak lift |
| `dominant_helpful_outcome_by_count` | `0.5717` | `0.6058` | best numeric result, but positive class is only `24/500` |

The best-looking helpful-outcome result should not be over-read. At the best
generation layer, accuracy is below a majority-style baseline and selectivity is
negative, so the balanced-accuracy bump is not a stable readout claim.

## Interpretation

This looks more like a label/probe-construction failure than evidence that the
model lacks moral-reasoning structure.

The issue is that raw rubric dimensions are grading families, not clean latent
variables. They mix:

- prompt-side dilemma coverage
- response-side reasoning process
- conclusion/helpfulness requirements
- safety or harm-avoidance penalties
- rubric authoring style and criterion-count artifacts

Counting dimensions and choosing a dominant family mostly asks the probe to
recover the shape of the evaluator rubric, not a model-internal deliberative
state.

## Remote-Origin Changes That Matter

The pulled `origin/main` additions are directly useful. Most important files:

- `methodology/archive/benchmark-to-latent-labels.md`
- `projects/MOREBENCH/phase_01/docs/01-latent-label-spec.md`
- `projects/MOREBENCH/phase_01/docs/01-confound-audit.md`
- `projects/MOREBENCH/phase_02/docs/02-augmentation-report.md`
- `projects/MOREBENCH/phase_02/outputs/theory_prompt_augmentation_examples.jsonl`
- `projects/MOREBENCH/phase_02/outputs/theory_control_augmentation_examples.jsonl`
- `projects/MOREBENCH/phase_02/outputs/theory_wording_variant_examples.jsonl`
- `projects/MOREBENCH/phase_02/outputs/action_locus_rewrite_pairs.jsonl`

Those changes support the same conclusion as the run: do not treat native rubric
dimensions as final mech-interp labels. Translate benchmark labels into latent
targets first.

## Recommended Pivot

Retire raw-dimension probes as headline phase-02 targets. Keep this run as a
negative control and diagnostic baseline.

Next runnable phase should focus on latent labels and matched augmentations:

1. Theory-conditioned reasoning mode
   - use the materialized theory prompt augmentations
   - compare direct theory prompts against neutral controls
   - split by `group_id`, not by individual prompt row
   - use wording variants as same-label robustness checks
   - probe prompt-end and generated-answer-end activations

2. Action-locus control state
   - use the matched advisor/agent rewrite pairs only as a smoke test
   - split by `pair_id`
   - treat current `10` pairs as insufficient for a serious result
   - expand before making a strong claim

3. Response-side labels from fresh generations
   - derive labels from generated responses, not rubric metadata
   - first labels: `tradeoff_engagement`, `commitment_style`,
     `refuses_or_hedges`, `helpfulness_invoked`,
     `harm_avoidance_invoked`, and `uncertainty_and_scope_calibration`
   - keep the labeler project-local until the LLM-judge abstraction is clearer

4. Controls
   - text baselines for every target
   - shuffled-label/selectivity controls for activation probes
   - grouped splits by base dilemma, theory group, or rewrite pair
   - nuisance tracking for source, type, topic, prompt length, and wrapper
   - same-label wording variants before treating a direction as stable

## Workflow Implication

Do not rerun the current raw-dimension workflow expecting a better answer. The
next workflow should be a new latent-label/augmentation workflow that consumes
the checked-in augmentation JSONL files and fresh generation artifacts.
