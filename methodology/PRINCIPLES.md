# Principles

Cross-cutting principles for mechanistic interpretability work inside the flywheel.

Small, stable, reusable, load-bearing. Referenced from every phase.

## 1. Behavioral sanity comes first

Do not jump to mechanism before establishing the task is real.

Before any probing, localization, or intervention:

- inspect real examples from each class or label family
- run the probe-target model on a small slice
- verify output parsing
- verify the task is being solved at a reasonable rate
- inspect failures manually

If the task is not behaviorally sane, stop and repair it first.

Behavioral sanity is distinct from data worthiness. Data can be real, rich, and important and still be a bad substrate for a specific model if the model can't do the task, outputs are malformed, labels are behaviorally incoherent, or the task is solved by shortcuts.

For response-side labels, the practical version: inspect generated model outputs, not just activations. An activation-only smoke test is not enough.

## 2. Evidence must climb a ladder

Mechanistic work builds a chain of evidence. Each level earns the next.

- **Level 1 — behavioral.** The model behaves sanely on the task.
- **Level 2 — representational.** A readout or probe detects the target variable.
- **Level 3 — localized representational.** The signal is localized to a span, section, token, layer, or position.
- **Level 4 — causal.** An intervention changes behavior beyond controls.
- **Level 5 — mechanistic.** A plausible computation path is identified.

Do not make Level 5 claims from Level 2 evidence. A high-AUROC probe supports a representational claim; it does not by itself support a causal or mechanistic claim.

The ladder is claim hygiene that runs alongside the flywheel. See `FLYWHEEL.md` for how they relate.

## 3. Read layer is not write layer

A readable signal is not automatically a writable signal.

Common pattern: later layers give the strongest probe AUROC; earlier layers give more causal leverage. Late layers may contain compressed summary states. Early or middle layers may still contain malleable computation.

Do not pick intervention sites only because they are easy to read.

## 4. Outcome-designed is not latent-designed

Labels in upstream data usually describe outcomes or grader judgments — was the response helpful, was the trade profitable, did the rubric pass. Mechanistic work needs labels closer to prompt structure, internal state, deliberative process, objective orientation, or intervention target.

The job is often to translate from outcome supervision to latent-oriented supervision. This happens in flywheel stage 2.

## 5. Source data is evidence, not proof of tractability

Start from the data as a crystallized artifact. Its shape encodes what someone thought mattered — what distinctions they expected to be meaningful, what behaviors they believed were worth surfacing. This is evidence of value.

It is not proof of mechanistic tractability. The correct posture: derive the implicit question from the data, then test whether the data supports that question cleanly.

For benchmarks: design choices carry information about the field's priors about what matters.

For agent traces: production choices — what was logged, what the agent was asked to do, how users complained — carry information about what the system's operators care about.

Both deserve the same skepticism about tractability.

## 6. Claim strength must match evidence strength

At the end of an analysis, separate what each method supports:

- behavior says the task is real
- probes say the variable is represented
- localization says where it emerges
- intervention says whether it is causally load-bearing
- attention or routing follow-up says something about mechanism

Always ask: what exactly has been shown? what has not yet been shown? what is still only an interpretation?

This principle is the reason every phase report has a claim boundary section.

## 7. Controls are mandatory

Good-looking results are not enough. Always ask:

- could a cheap surface baseline do this?
- could the model be using style instead of state?
- could source, length, or role tokens explain the effect?
- could prompt format alone recover the label?
- could a same-label control show the intervention is just destabilizing the model?

Controls should be planned before results, not added as a reaction to skepticism.

## 8. Nuisance-stratified cell size matters

Any analysis plan should check post-stratification sample size.

How many examples remain per target label after stratifying on the nuisance variables that actually need to be controlled? Do the resulting cells still support the planned probe or intervention?

If the answer is no, the correct response is to narrow the question, augment the data, or stop. Not: run the analysis anyway and hope regularization hides the problem.

## 9. Data is a seed set, not a prison

If the data cannot cleanly support the intended question, the answer is not always "give up."

It may need rewrites, matched pairs, counterbalancing, response generation, or synthetic augmentation. Augmentation should repair the experiment, not just enlarge it.

This is what flywheel stage 2 is for.

## 10. Labels leak by construction

Upstream labels — whether from graders, outcomes, or production bookkeeping — are usually applicable from surface semantics. Treat imported labels as surface-recoverable by default until the pipeline has actively broken the most plausible shortcut channels.

The goal is not to remove every easy feature blindly. It is to break surface-label correlation while preserving the semantic content a careful human reader would still recover.

## 11. One-split success is not transfer

A result that survives only one narrow slice should not be promoted as abstraction, transfer, or mechanism.

If a result looks unusually strong, clean, or easy, stress-test it against the strongest plausible shortcut explanation before promoting it up the evidence ladder.

## 12. Response-side probing requires active confound reduction

Instruction-following models produce label-adjacent vocabulary when primed with label-adjacent instructions. This is not a bug; it is the defining property of the models being probed. Any response-side label on an instruction-following benchmark will carry a lexical confound unless the experiment actively works against it.

Four complementary confound-reduction technique categories exist, and they stack:

- `viewport reduction` — probe only a portion of the response where the instruction-acknowledgment leakage is absent (tail window, conclusion span, non-header content)
- `training distribution variation` — train the probe across prompt formats, paraphrases, or prime variants so the learned direction must be format-invariant
- `lexical subspace subtraction` — residualize probe features against text-baseline predictions, or erase text-aligned subspaces, before probing
- `target reformulation` — shift from categorical identity to binary collapse, relational contrast, or behavioral extraction

No single technique reliably pushes text baselines off ceiling. Combinations are the norm, not the exception. Plan response-side probe work by pairing techniques from at least two categories before accepting any representational claim.
