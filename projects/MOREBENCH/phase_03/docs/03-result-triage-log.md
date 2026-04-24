---
benchmark: morebench
phase: 03
version: v1
frozen_date: 2026-04-23
input_artifacts:
  - projects/MOREBENCH/phase_03/docs/03-analysis-plan.md
  - projects/MOREBENCH/phase_03/docs/03-experiment-specs.md
  - projects/MOREBENCH/phase_03/docs/03-controls-and-splits.md
  - projects/MOREBENCH/phase_03/docs/03-execution-targets.md
---

# MoReBench 03 Result Triage Log

## Experiment 1: Theory-Identity Prompt Readout

- execution date:
  `2026-04-23`
- workflow:
  `morebench_phase03_experiment01_theory_identity`
- execution note:
  the auxiliary `anchor_clause` span-capture branch stalled operationally during a reuse run, but the core readouts needed for triage completed and are sufficient for the verdict below
- completed evidence used for triage:
  - `text_baseline_1_f2ae3c29`
  - `probe_1_7282017a`
  - `transfer_probe_1_213eabe1`
  - `probe_1_e4e36c77`

### Metrics

- prompt-EOS theory-vs-control readout:
  best balanced accuracy `1.0` at layer `20`
- direct / wording-variant / anchor-only transfer:
  best balanced accuracy `1.0` across all three families at layers `20` and `44`
- named-theory clause localization:
  balanced accuracy `1.0` at every captured layer
- cheap semantic baseline:
  `anchor_text` bag-of-words logistic baseline balanced accuracy `1.0`

### Interpretation

The result is strong in a narrow readout sense and weak in a mechanistic-discovery sense.

What it establishes:

- the current phase-02 theory prompt family very strongly encodes `theory_identity`
- that encoding survives direct-vs-wording transfer
- the supposed `anchor_only` control was not a credible anti-shortcut family because the fixed per-theory anchor sentence remained
- the signal is localizable to the named theory clause on named rows

What it does **not** establish:

- a nontrivial framework-conditioned prompt state beyond explicit prompt semantics
- a compelling target for deeper mechanistic follow-up in the current prompt design

The decisive issue is the cheap baseline.
The `anchor_text` baseline alone classifies theory identity perfectly, so the current result is fully explainable by explicit semantic content in the prompt family.

### Verdict

- verdict:
  `AUGMENTATION_NEEDED`
- routing:
  hand the current `theory_identity` prompt family back to phase 02 for anti-shortcut repair before any prompt-side retry
- why this verdict rather than promotion:
  the result does **not** beat the cheap surface-semantic baseline named in the control philosophy, so this is a repair signal about the dataset family rather than a usable phase-03 finding

### Follow-on Action

Because `theory_identity` remains strategically important, it should now stay in the phase-02 repair loop:

- treat the legacy explicit-theory family as known-broken
- materialize harder factorial, alias-based, and description-based theory families
- run stronger prompt-side baseline preflight before any phase-03 retry
- treat `alias_only` as the best current prompt-side diagnostic family, but keep the retry gate closed until its held-out text baselines fall further
- treat `description_only` as a theory-priming family for generation-time persistence work rather than as a clean prompt-side retry family

If the goal is the strongest next phase-03 execution target, move to the response-side pilot and freeze path for:

- `theory_conditioned_generation_persistence`
- `tradeoff_engagement`
- `commitment_style`
- `helpfulness_invoked`
- `harm_avoidance_invoked`

## Experiment 2: Theory-Conditioned Generation Persistence

- execution dates:
  `2026-04-23` to `2026-04-24`
- primary workflows and replay runs:
  - `morebench_phase03_experiment02_theory_persistence`
  - `morebench_phase03_experiment02_behavior_broad`
  - `morebench_phase03_experiment02_behavior_broad_replay_capture`
  - `morebench_phase03_experiment02_benchmark_missing_replay_capture`
- key evidence used for triage:
  - full-sequence benchmark run:
    - `projects/MOREBENCH/phase_03/reports/experiment_02_manual_analysis/report.md`
    - `projects/MOREBENCH/phase_03/reports/experiment_02_extended_analysis/description_only_extended_metrics.json`
  - tail / transfer / residualization:
    - `projects/MOREBENCH/phase_03/reports/experiment_02_tail_residualization/tail_residualization.json`
    - `projects/MOREBENCH/phase_03/reports/experiment_02_tail_fraction_sweep/tail_fraction_sweep_summary.json`
    - `projects/MOREBENCH/phase_03/reports/experiment_02_family_transfer_analysis/multifamily_analysis.json`
    - `projects/MOREBENCH/phase_03/reports/experiment_02_family_transfer_tail_analysis/multifamily_tail_analysis.json`
  - behavior / PCA / contested-case readouts:
    - `projects/MOREBENCH/phase_03/reports/experiment_02_behavior_recommendation_analysis/report.md`
    - `projects/MOREBENCH/phase_03/reports/experiment_02_pca_geometry/report.md`
    - `projects/MOREBENCH/phase_03/reports/experiment_02_behavior_broad_llm_judged/report.md`
    - `projects/MOREBENCH/phase_03/reports/experiment_02_contested_override_controls/report.md`

### Sequence Of Findings

1. Full-sequence single-family theory-identity persistence was negative in the intended scientific sense.

- strict filtered capture kept `134/180` rows after copy filtering
- best text baseline balanced accuracy: `1.0`
- best probe balanced accuracy: `1.0`
- best layer: `0`
- extended AUROC / one-vs-rest metrics were also at ceiling

Interpretation:

- the original `description_only` full-sequence target was shortcut-dominated / control-insufficient
- any apparent readout success was fully matched by generated-text baselines

2. Tail-window and cross-family analyses were good and necessary follow-up steps, but they did not survive stronger controls.

What initially looked encouraging:

- tail-window and leave-one-family-out transfer produced positive raw probe-over-text deltas
- smaller tail fractions strengthened the raw effect
- early-mid and late layers both looked viable before stronger controls

What decided the issue:

- residualization largely collapsed the apparent tail/cross-family advantage
- the surviving effect size was not stable enough to support a phase-03 representational claim

Interpretation:

- the tail work was scientifically useful because it tested whether lexical domination from the opening paragraphs was hiding a real late-generation signal
- the answer was mostly no: the late-generation effect was still substantially text-correlated

3. Behavioral review was the most durable positive result of the experiment.

Canonical description-only review:

- `22/30` dilemma groups unanimous
- `8/30` genuine split groups

Broad `90`-dilemma manually judged run:

- `68/90` unanimous
- `22/90` split
- benchmark theory slice: `7/30` split
- public extension: `15/60` split

Prime-level split behavior:

- deontology defects from the majority in `9/20` majority-defined split groups
- utilitarian defects in `8/20`
- virtue ethics and generic in `4/20`
- contractarianism and contractualism in `3/20`

Interpretation:

- theory primes do change action-level behavior on a minority of dilemmas
- the live behavioral pattern is a hardness / default-deviation gradient, not a clean theory-taxonomy separation

4. PCA was useful diagnostically but not vindicating.

- prime centroids showed low-rank structure
- the geometry suggested plausible coarse clustering
- but the pre-registered deontic-vs-welfarist interpretation failed behaviorally
- override-status PCA on contested cases showed no clean visible separation

Interpretation:

- PCA helped reposition the hypothesis from `6`-way theory identity toward lower-rank structure
- it did not produce an independently corroborated representational target

5. The public-conflict extension was a methodological success.

- benchmark clarification: the theory split contains `150` rows but only `30` unique dilemma texts
- adding `60` structurally screened public dilemmas produced a real `90`-dilemma behavior substrate
- the extension yielded `15/60` genuine split groups under manual judgment, close to the theory slice's `7/30`

Interpretation:

- the extension was worthwhile even though the downstream probe target failed
- it validated structural contested-case selection as a useful benchmark-construction step

6. The final contested-case override/default-deviation framing failed under strong controls.

Key final control result for pooled `differs_from_generic`:

- strongest full-sequence baseline: `tfidf_char`, AUROC `0.6431`
- best raw probe: layer `4`, AUROC `0.6385`
- delta vs strongest baseline: `-0.0046`, CI `[-0.0982, 0.0978]`

Tail-window did not rescue it:

- strongest tail baseline: `tfidf_char`, AUROC `0.6986`
- best raw tail probe: AUROC `0.6789`
- delta: `-0.0197`, CI `[-0.1684, 0.112]`

Residualization did not rescue it:

- residualized full best delta: `-0.0126`
- residualized tail best delta: `-0.0479`

The `defect_from_majority` target was weaker still.

Interpretation:

- response-side action labels are surface-recoverable by construction from the generated text
- once strong char-level text baselines are included, the probe no longer beats the cheap baseline

### What Survived

- Level 1 behavioral finding:
  theory primes induce genuine action-level splits on a minority of dilemmas
- methodological finding:
  structurally screened public dilemmas are a workable way to enrich for contested cases
- methodology warning:
  the original char-n-gram clustering scorer was not adequate for action-equivalence judgment and had to be replaced by manual LLM judgment

### What Did Not Survive

- full-sequence single-family theory persistence as a representational claim
- tail-window cross-family reopening as a stable representational claim
- PCA-inspired deontic-vs-welfarist cluster hypothesis as a behavioral explanation
- contested-case override/default-deviation as a response-side representational claim

### Verdict

- verdict:
  `TRIVIAL_OR_NULL`
- routing:
  close Experiment 2 as a response-side representational target on the current substrate
- claim ceiling reached:
  `behavioral` plus `methodological`, not `representational`

### Why This Verdict

The decisive failure mode is structural rather than accidental.
For the final contested-case target, the label was action-based and the capture site was response-side generated activations.
The response text itself contains the advocated action, so strong char-level text baselines recover the target directly.
Once those baselines are run, the probe no longer clears a meaningful margin.

### Follow-on Action

- record the surviving phase-03 outputs as:
  - behavioral split-rate and prime-hardness findings
  - public-conflict extension methodology result
  - negative response-side representational result
- do **not** capture the `408` unanimous rows for override validation
- pause before any further MoReBench response-side capture
- only reopen this line if the question changes materially, for example:
  - prompt-side or pre-first-generated-token capture
  - a non-action response label that is not already explicit in the response text

### Addendum: Prompt-Side Cross-Language Reopening

- execution date:
  `2026-04-24`
- canonical workflow:
  `morebench_phase03_experiment02_cross_language_prompt_probe_full_capture`
- canonical capture artifact:
  `capture_1_2c011b403d39`
- canonical report:
  `projects/MOREBENCH/phase_03/reports/experiment_02_cross_language_prompt_probe_full/report.md`

#### Metrics

- prompt-text cross-language baseline mean AUROC:
  `0.6424`, grouped bootstrap CI `[0.6140, 0.6628]`
- prompt-length cross-language baseline mean AUROC:
  `0.5583`
- best prompt-final probe layer:
  `32`
- best-layer mean cross-language probe AUROC:
  `1.0`, grouped bootstrap CI `[1.0, 1.0]`
- best-layer delta vs prompt text:
  `+0.3576`, grouped bootstrap CI `[0.3372, 0.3860]`
- random-label control at best layer:
  mean `0.4992`, p95 `0.6733`, share `>= 0.80` = `0.0039`

#### Interpretation

This is the first defended Level 2 result in the program.

What it establishes:

- a prompt-final `deontology` vs `virtue_ethics` readout exists on the translated `English / Spanish / Simplified Chinese` prompt family
- that readout transfers across languages, including cross-script pairs
- the emergence shape is computation-like rather than lexical-shortcut-like:
  `L0 0.50 -> L4 0.73 -> L8 0.90 -> L16 0.9987 -> L32 1.0`
- the result is not explained by prompt-text or prompt-length baselines on the same cross-language splits

What materially strengthens the claim:

- the full-30 prompt family is already description-only rather than name-only
- follow-up prompt audit finds `0` framework-name hits across all `180` prompts
- the prompt `L32` direction is nearly orthogonal to the old response-side `description_only` directions, so this is not just a cleaner readout of the old response-side ceiling

#### Reproducibility Note

The original mixed local-transform / GPU-capture workflow failed at the capture handoff.
The canonical artifact for this result is therefore the recovered capture-only workflow output:

- `capture_1_2c011b403d39`
