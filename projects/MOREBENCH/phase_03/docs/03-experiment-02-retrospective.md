---
benchmark: morebench
phase: 03
experiment: 02
version: v1
frozen_date: 2026-04-24
input_artifacts:
  - projects/MOREBENCH/phase_03/reports/experiment_02_manual_analysis/report.md
  - projects/MOREBENCH/phase_03/reports/experiment_02_extended_analysis/description_only_extended_metrics.json
  - projects/MOREBENCH/phase_03/reports/experiment_02_behavior_recommendation_analysis/report.md
  - projects/MOREBENCH/phase_03/reports/experiment_02_behavior_broad_llm_judged/report.md
  - projects/MOREBENCH/phase_03/reports/experiment_02_contested_override_controls/report.md
  - projects/MOREBENCH/phase_03/docs/03-result-triage-log.md
---

# Experiment 02 Retrospective

This document records the motivation, sequence, intermediate decisions, useful detours, and final outcome of the phase-03 Experiment 2 line of work.

The short version is:

- the experiment generated real behavioral findings
- several intermediate analyses were worth doing and materially sharpened the question
- the final response-side representational claim did **not** survive the strongest controls
- we probably should have taken the hint earlier that MoReBench response-side action labels were structurally too text-recoverable for this capture setup

## Starting Motivation

Experiment 2 was the main theory question for phase 03:

- do theory-conditioned prompt families produce a generation-time residual-state signal that persists beyond the prompt text itself?
- if so, can we localize that signal in generated reasoning rather than merely reading explicit theory cues from the prompt?

The original target was `theory_conditioned_generation_persistence`.
The original hope was a Level 2 or Level 3 result:

- representational evidence that a theory-conditioned state persists into generation
- ideally localized to a late-generation band worth future causal follow-up

## What We Did, In Order

### 1. Full-sequence single-family run

We began with the canonical `description_only` generation batch and full-sequence generated-token capture.

Why this was a good first step:

- it was the most direct test of whether the primed theory family remained decodable from the generated response state
- it gave a baseline for later controls and family-transfer work

What happened:

- generated-text baseline hit ceiling
- probe hit ceiling
- best layer was `0`
- extended AUROC / one-vs-rest metrics also ceilinged

What we learned:

- full-sequence response-side theory identity was not a live mechanistic target in this form
- the readout was dominated by shortcut / lexical information in the generated text

This was the first major warning sign.

### 2. Tail-window and cross-family reopening attempts

Instead of stopping at the full-sequence ceilinged negative, we asked the right next question:

- was the opening portion of the response dominating the task so strongly that it hid a weaker but more interesting late-generation signal?

That led to:

- tail-window analyses
- cross-family transfer analyses
- leave-one-family-out comparisons
- tail-fraction sweeps

Why this was a good step:

- it directly tested the "full-sequence lexical domination" hypothesis
- it asked whether a signal survived in a more constrained viewport
- it also tested whether any signal generalized beyond one prompt family

What happened:

- raw tail-window and cross-family deltas looked meaningfully better
- smaller tail fractions looked stronger than broader windows
- layers around `8` and `44` often looked competitive before stronger controls

What we learned:

- tail-window analysis was worth doing; it did surface the only plausible reopening of the original negative
- but residualization mostly collapsed the result
- the apparent reopening was not robust enough to count as a representational win

This was the second major warning sign.

### 3. Behavior review on the original benchmark slice

We then pivoted from label recovery to actual recommendation behavior.

Why this was a good move:

- if theory-conditioned generation is not changing decisions, there is little reason to expect a meaningful action-relevant latent state
- behavior can reveal whether the benchmark actually contains live action-level contrast sets

What happened on the canonical `30` theory dilemmas:

- `22/30` groups were unanimous
- `8/30` showed genuine action-level splits

What we learned:

- there is more than lexical variation here
- but the benchmark does not exhibit five clean, stable theory policies
- the better picture is one broad default recommendation policy plus a minority of branch-point dilemmas

This was the first durable positive result.

### 4. PCA geometry pass

We then asked whether probing had been pointed at the wrong target.

Why this was a good move:

- PCA answers a different question from supervised probes
- it can reveal whether the activation geometry is low-rank, clustered, or dominated by other variation
- it helps avoid overfitting the analysis to the wrong hypothesis

What happened:

- prime centroids showed real low-rank structure
- the combined tail geometry suggested plausible coarse relations among theory families
- but capture coverage was incomplete, so this had to be interpreted cautiously

What we learned:

- probing for `6`-way theory identity was probably the wrong initial target shape
- a lower-rank or coarse-cluster hypothesis was more plausible than five independent framework directions

This was a useful reframing step, even though it did not survive later behavioral testing.

### 5. Broad public-dilemma extension

We then realized the benchmark theory split was not what we had mentally assumed.

Important correction:

- the theory split contains `150` rows but only `30` unique dilemma texts
- each dilemma is repeated across `5` native benchmark theory labels

Why the extension was a good step:

- it corrected the substrate misunderstanding
- it increased real dilemma diversity
- it was designed structurally rather than by theory-coded filtering
- it created a larger contrast-discovery substrate without pretending the original theory split already had `150` unique dilemmas

What we built:

- original `30` grouped theory dilemmas x `6` prime conditions = `180`
- plus `60` structurally screened public dilemmas x `6` = `360`
- total `540` generations over `90` dilemma groups

What we learned:

- the public extension itself was worth it
- it later produced `15/60` genuine split groups under manual judgment, close to the theory slice's `7/30`

This was a real methodological success.

### 6. Broad behavior scoring and scorer failure

The first automated broad-behavior scorer was wrong.

Why this mattered:

- it initially reported too few splits
- it mixed together polarity-opposite recommendations and split apart same-action paraphrases
- if left uncorrected, it would have sent us toward the wrong capture target and the wrong scientific conclusion

Why replacing it was a good step:

- it forced a return to the actual recommendations
- it prevented the benchmark from being distorted by a bad response clustering heuristic

What happened after manual LLM judgment:

- the true combined result was `22/90` split and `68/90` unanimous
- the public extension contributed `15/60` splits
- only `1/22` split matched the pre-registered deontic-vs-welfarist cluster hypothesis

What we learned:

- the coarse PCA cluster hypothesis was not behaviorally load-bearing
- theory-conditioned action changes were real, but sparse
- the live behavioral structure looked more like a hardness / default-deviation gradient than a stable framework taxonomy

This was one of the most important corrections in the whole project.

### 7. Contested-case capture repair

At this point we had a better behavioral target, but capture coverage was incomplete.

Why fixing capture coverage was a good step:

- it made the contested-case set genuinely usable
- it avoided a fake null caused by missing rows from the benchmark side

What we did:

- replay-captured the new public conflict groups
- audited old benchmark capture coverage
- replay-captured the missing benchmark conflict rows

What we learned:

- some earlier benchmark capture incompleteness was due to strict copy filtering, not the absence of useful contested cases
- after repair, the contested-case activation set was complete enough for the final override/default-deviation test

### 8. Override / default-deviation framing

We then tried the best relational response-side target we had found:

- not "which theory is active?"
- but "did the theory prime push the model away from its usual or reference action?"

Why this was a good step:

- it replaced a brittle content label with a relational target
- it matched the actual behavioral observations better than the original theory-family labels
- it was the strongest conceptual target the line produced

We tried two binary labels:

- `differs_from_generic`
- `defect_from_majority`

What happened on the first pass:

- pooled `differs_from_generic` looked mildly positive before stronger controls
- pooled `defect_from_majority` looked weaker
- deontology within-prime looked like the only plausible live subcase

Why this was still a good step:

- it gave the line the fairest possible response-side target before closure
- it tested whether a default-deviation signal existed even when theory-identity did not

### 9. Strong-control kill pass

We then did exactly what the benchmark-first discipline called for:

- stronger text baselines
- grouped bootstrap CIs
- fixed-layer reporting alongside exploratory best-of-8
- tail-window reruns
- residualized probes

This was the decisive step.

What happened:

- `char TF-IDF` met or exceeded the pooled probe
- tail-window did not rescue the signal
- residualization did not rescue the signal
- PCA did not show clean override-status geometry
- the deontology tail corner remained too wide-CI and too unstable to defend

What we learned:

- the final target was structurally too text-recoverable on response-side activations
- once the label is action-based and the capture is response-side, the advocated action is already in the text
- there is no robust representational margin left for the probe to claim

This was the final closure point.

## What Was Good About The Path We Took

Several steps were absolutely worth doing even though the final representational claim failed.

### We did not stop at the first ceilinged null.

The tail-window, family-transfer, and residualization passes meaningfully clarified whether the original negative was merely an artifact of a bad viewport.

### We checked behavior directly rather than assuming the label was behaviorally real.

That prevented the project from claiming framework structure where most dilemmas actually collapsed to the same action.

### We used unsupervised geometry to challenge the initial supervised target.

PCA did not save the line, but it was the right diagnostic step and it improved the target selection logic.

### We corrected dataset misunderstandings instead of quietly working around them.

The `150`-rows-versus-`30`-unique-dilemmas clarification mattered, and the public extension was the right fix.

### We caught a broken scorer before it propagated downstream.

This was important enough to count as a methodological finding in its own right.

### We repaired missing capture coverage instead of letting incompleteness masquerade as a negative result.

That made the final closure cleaner and more defensible.

### We actually ran the kill controls.

The strongest value of this line is not the null itself but that it is a disciplined null:

- strong baseline
- grouped bootstrap
- residualization
- viewport check
- repaired capture coverage

## Where We Probably Should Have Taken The Hint Earlier

Several warning signs were present before the final override-control pass.

### Hint 1: full-sequence theory persistence ceilinged immediately.

That was already telling us the response-side theory-identity target was shortcut-prone.

### Hint 2: the tail-window reopening mostly collapsed under residualization.

That was a stronger sign than we initially treated it as.

### Hint 3: behavior was sparse and family-sensitive.

Even the real split cases often failed to stabilize across families.
That limited how much clean framework-state structure the substrate was likely to support.

### Hint 4: the final action-based target lived in the response text itself.

In hindsight, this is the most important structural issue.
Once the label is "which action did the model recommend?" and the capture site is the generated response, the shortest path to the answer is the text itself.

## Final Surviving Results

### 1. Behavioral result

This survived.

- `22/90` genuine action-level split groups under manual judgment
- prime hardness gradient:
  - deontology `9/20`
  - utilitarian `8/20`
  - virtue ethics `4/20`
  - generic `4/20`
  - contractarianism `3/20`
  - contractualism `3/20`

Interpretation:

- theory primes can move behavior on a minority of dilemmas
- the effect is not a clean theory taxonomy

### 2. Methodological result

This survived.

- the structural public-conflict extension produced `15/60` split groups
- that is close to the benchmark theory slice's `7/30`

Interpretation:

- structural contested-case discovery worked as intended
- this is useful benchmark-construction knowledge even though the probe target failed

### 3. Negative representational result

This survived.

- response-side action-based labels do not beat strong cheap text baselines on this substrate
- `char TF-IDF` was the named cheap baseline that closed the case

Interpretation:

- on this benchmark, with this capture site, action-based response labels leak by construction

### 4. Scorer methodology note

This survived.

- naive response clustering by char-level overlap is not reliable for action-equivalence judgment
- polarity and paraphrase both break it

## Ultimate Result

Experiment 2 does **not** support a phase-03 representational or localized-representational claim on the current response-side setup.

The final disposition is:

- `behavioral`: yes
- `methodological`: yes
- `representational`: no
- `localized representational`: no
- `causal`: not licensed

## Recommendation For Future Reopening

Do not reopen the same response-side action-label line.

If MoReBench is revisited for a Level 2+ theory claim, the change must be structural, not cosmetic.

Plausible alternatives:

- prompt-side or pre-first-generated-token capture, before the action is verbalized
- a non-action response label that is not already explicit in the response text

What should **not** happen:

- more response-side capture on the same action-based target
- more attempts to rescue the deontology tail corner
- more work on unanimous-row capture for the killed override framing
