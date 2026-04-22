# Review Proposal — Benchmark-to-Latent-Labels

> HISTORICAL NOTE
>
> This file is a historical review artifact that informed
> [benchmark-to-latent-labels.md](./benchmark-to-latent-labels.md).
> It should not be treated as a parallel canonical process document.

**Date:** 2026-04-22
**Status:** Proposal for integration into [benchmark-to-latent-labels.md](./benchmark-to-latent-labels.md)
**Intended reader:** Codex, merging this with its own thoughts into the main doc

This file collects structural and content revisions surfaced during a pressure-test of the current process document. It is organized so each section can be accepted, rejected, or modified independently rather than wholesale-replaced. Drop-in language is marked explicitly; discussion is kept separate.

Original doc has these sections: Purpose, Core Principle, Target Output, Recommended Workflow (steps 0–12), Practical Checklist, Common Failure Modes, Outputs, What Came Out Of This Process So Far, Suggested Reusable Template. References below are to those section names.

---

## 1. Summary of proposed changes

1. **Add a Required Inputs section** near the top. Several choices that gate which labels are meaningful are currently implicit: probe-target model, generation protocol, activation capture regime, research mode (correlational vs causal). A skill that is silent on these produces inconsistent outputs across users.
2. **Augment Core Principle with a second principle: benchmark-derived question posture.** Existence is evidence. A benchmark embodies an implicit mechanistic question through its design choices. Do not bring externally-generated question lists; derive the question from the benchmark and then validate it for mech-interp tractability.
3. **Insert a new stage between intake and label inventory: "Extract the implicit mechanistic question(s) from the benchmark."** This is the step that protects against getting lost in rubric mechanics without smuggling in external questions.
4. **Reorder within the workflow.** Move signal-location hypothesis and first-pass ontology selection *before* derivability validation. You should only pay labeling costs on ideas that survived selection.
5. **Add three quantitative audit stages** (cell-size / power check, surface-feature baseline, inter-label correlation) that the current doc does not include but that cheaply kill bad ideas.
6. **Add an ontology-freeze / pre-registration gate** between label work and any activation work. This is the wall that protects against post-hoc label rationalization.
7. **Add an explicit Gap List as the handoff contract** from this skill to the downstream data-augmentation skill.
8. **Rename the downstream skill** from "synthetic data" to `latent-label-data-augmentation`. Most of its value is in rewrites, minimal pairs, and counterbalancing, not whole-cloth generation. The "synthetic data" name will bias users toward the least artifact-safe technique.
9. **Add Stopping Criteria and Escalation Triggers sections.** The current doc has no gates for when to stop, when to abort, or when to hand off. These are the single biggest missing piece for reusability.
10. **Add a canonical cross-benchmark label vocabulary.** Even a small shared set (role framing, subject of action, information stance, stakes framing, considerations represented, commitment transition, harm-avoidance invoked, helpfulness invoked) would let a future researcher say "this benchmark supports 3 of the 8 canonical axes" instead of reinventing.
11. **Note a potential future third skill:** `latent-label-probe-design`. Current steps 9 and 11 of the workflow smuggle in probe-experiment-design work. For v1, keep it folded in. If this skill bloats, probe design is the natural split.
12. **Additions to Common Failure Modes and to the confound checklist.** Two failure modes and two confounds are missing; details below.

---

## 2. Existing process is not discarded

To be explicit about what the review affirms in the current doc, so this proposal is read as refinement not replacement:

- The grader-designed-vs-latent-designed framing in Core Principle is correct and load-bearing.
- The label-type taxonomy in step 2 (prompt-side structure / response-side process / objective orientation / outcome-rubric / nuisance) is a genuinely useful ontology and should become the canonical vocabulary across Skill A and Skill B.
- The five-level match rating in step 3 (direct / derived / validation-only / nuisance-only / not useful) is well chosen.
- Step 9 (where should signal live) is the step most mech-interp docs skip. It stays.
- The Common Failure Modes section captures real failures, not hypothetical ones.

All four sections above survive the revision as written or with minor edits.

---

## 3. Drop-in: new Required Inputs section

Insert between Purpose and Core Principle, or between Target Output and Recommended Workflow — either placement works.

> ### Required Inputs
>
> Before starting this process, record the following. Each of these choices determines which candidate labels are meaningful, so leaving them implicit causes silent inconsistency across runs.
>
> - `probe-target model(s)`
>   base / instruct / RLHFed. A label like "helpful-vs-harmless invocation" is meaningful for an RLHFed instruct model and may not exist as a clean internal dimension in a base model.
> - `generation protocol`
>   zero-shot / few-shot / CoT / system-prompted. This determines which response-side labels are even observable.
> - `activation capture regime`
>   prompt-only snapshot / generation-time / full trajectory. Labels that reference process or commitment only make sense with generation-time or trajectory capture.
> - `research mode`
>   correlational readout / causal intervention / both. Causal intervention requires matched pairs; correlational readout does not. This gates whether the benchmark is sufficient or whether Skill B is required.
> - `sampling parameters and seeds`
>   required for reproducibility of any response-side labels.

---

## 4. Drop-in: augmented Core Principle

Proposed replacement for the current Core Principle section.

> ### Core Principle
>
> Two complementary posture principles govern this process.
>
> **(a) Grader-designed ≠ latent-designed.** Many benchmarks are grader-, rubric-, preference-, or task-outcome-designed. Mech interp needs labels that correspond to prompt structure, internal state, deliberative process, control policy, or intervention target. The job is to translate from grader-designed supervision to latent-oriented supervision. This translation is not automatic.
>
> **(b) Existence is evidence.** A benchmark was built by someone with domain expertise who decided it was worth building. Its design choices — what was included, contrasted, annotated, omitted — embody an implicit mechanistic question, whether or not the authors stated it cleanly. The first analytical move is to recover that implicit question from the benchmark itself, not to bring externally-generated questions to it. A benchmark-first posture aligns the work with crystallized field wisdom and protects against benchmark-shopping, abstraction drift, and question-fishing.
>
> The two principles work together: derive the question from the benchmark, then audit whether the benchmark's native labels actually serve that question or merely serve grading of it.
>
> **Exception handling.** Some benchmarks were built primarily for task grading or leaderboard ranking rather than capability probing. Their implicit question may be real but not mechanistic (e.g., "does the model get the right answer?" vs "how does the model represent the problem?"). This is a useful finding, not a failure: it tells you the benchmark is better suited to Skill B-style augmentation (contrast sets that expose internal structure) than to direct labeling. The tractability check in Stage 3 surfaces this case.

---

## 5. Drop-in: Revised Workflow (replaces current steps 0–12)

The revision preserves almost all of the current content but reorders, splits, and adds gates. Changes from the current doc are flagged in brackets after the stage title.

> ### Recommended Workflow
>
> #### Stage 1. Benchmark intake *[was step 0; unchanged except renumbering]*
>
> Before touching labels, establish the benchmark frame. Record:
> - dataset size, public availability, splits and configs
> - prompt structure, response structure
> - whether labels apply to prompt, response, trajectory, or evaluator judgment
> - whether labels are human-authored, expert-authored, verifier-generated, or synthetic
> - whether a runnable harness exists
> - whether activations can be captured at prompt time, generation time, or both
>
> #### Stage 2. Extract the implicit mechanistic question(s) *[NEW]*
>
> Read the question off the benchmark's design. Why did the authors build *this* benchmark and not a different one? What internal capacity, representation, or control state are its design choices implicitly targeting?
>
> Output: 1–5 mechanistic question statements in "does the model represent / switch between / combine X?" form. Each statement is tagged with the benchmark evidence it was read from (which labels, which contrasts, which splits).
>
> This step is derivation, not invention. Do not generate questions from an external list. Do not skip it and go straight to label inventory — the label inventory will be harder to interpret without the implicit question in view.
>
> #### Stage 3. Tractability and fit check *[NEW; gate]*
>
> For each extracted question, verify:
> - Is it a mechanistic question (about representation, process, or control) rather than a task-outcome question (about correctness)?
> - Is it correlational or causal? If causal, does the benchmark contain matched pairs or would Skill B be required?
> - Given the benchmark's size and composition, what is the expected post-stratification cell count for the most plausible target axis? If this is below a chosen floor (e.g., 50), flag the question as needs-Skill-B or drop it.
>
> If all extracted questions fail this check, stop. The benchmark either needs augmentation first, or is not the right substrate. Do not proceed to labeling.
>
> #### Stage 4. Inventory existing labels *[was step 1; unchanged]*
>
> #### Stage 5. Classify the label type *[was step 2; split "outcome / rubric score" into "outcome" and "rubric score" — graders and outcomes diverge]*
>
> #### Stage 6. Rate match to plausible latent labels *[was step 3; unchanged]*
>
> #### Stage 7. Question × benchmark intersection *[was step 4, rephrased to fit benchmark-derived posture]*
>
> With the label inventory in hand, revisit the extracted mechanistic questions. For each question, list the native or derivable labels that could support it. Drop questions the benchmark cannot touch even in principle — park them, don't delete.
>
> This is the step where the benchmark-derived question meets the label inventory. It is no longer about "what questions do we care about" (Stage 2 did that) but about "how well does the benchmark's actual label set serve the question it implicitly asks."
>
> #### Stage 8. Signal-location hypothesis *[was step 9; moved earlier]*
>
> For each surviving question, state explicitly where in the forward / generation pass the signal should live: prompt-side representation, generation-time deliberation, commitment / action selection, final-token readout. This gates label selection: prompt-only labels cannot answer generation-time questions.
>
> #### Stage 9. Select minimal first-pass ontology *[was step 10; moved earlier]*
>
> Pick: 1–2 prompt-side structure labels, 1–2 response-side process labels, 1 objective-orientation contrast, tracked nuisance set. Postpone everything else explicitly, not ambiguously.
>
> #### Stage 10. Define success and kill criteria *[was step 11]*
>
> Per label family: indicator success, mechanistic success, control success (where applicable), required baselines, and a kill criterion that would cause the label family to be dropped.
>
> #### Stage 11. Specify labeling function *[NEW]*
>
> For each label, specify: labeling function (human / LLM with frozen prompt / regex / classifier), version, storage location, versioning/hash scheme. No ad-hoc labeling.
>
> #### Stage 12. Validate derivability *[was step 7; moved later]*
>
> Validate only the labels that survived Stage 9 against a small hand-labeled gold set. Report agreement against the labeling function. Revise or drop. Do not spend labeling budget on ideas that didn't survive selection.
>
> #### Stage 13. Surface-feature baseline and correlation audit *[NEW]*
>
> For each surviving label:
> - Train a simple surface-feature classifier (bag-of-words logistic regression or similar). If its accuracy is at or above the probe target, flag: any probe finding this label is trivial.
> - Compute inter-label and label-vs-nuisance correlation matrices. Collapse highly correlated labels.
>
> #### Stage 14. Confound audit *[was step 8; expanded checklist — see §10]*
>
> Co-occurrence between every nuisance and every target label. For each: stratify / regress out / hold out / demote to nuisance-only.
>
> #### Stage 15. Ontology freeze and pre-registration *[NEW; gate]*
>
> Commit the label spec to version control with a hash. Record: labels, labeling functions, success criteria, confound stratification plan. No further changes without opening a new version. This is the wall between labeling and activation work, and the primary defense against post-hoc rationalization of probe results.
>
> #### Stage 16. Gap list — handoff contract to Skill B *[was step 12, sharpened]*
>
> Enumerate what's missing and would be needed from `latent-label-data-augmentation`:
> - matched pairs for causal questions
> - response-side generations (if the benchmark is prompt-only)
> - paired framing variants (theory, role, person, length)
> - confound-counterbalanced subsets
> - whole-cloth synthetic cases to fill confound holes
>
> The gap list is the formal contract between Skill A and Skill B. If Skill A ends with an empty gap list and all questions supported, Skill B is not needed. If Skill A ends with a non-empty gap list, invoking Skill B is the next action.
>
> #### Stage 17. First experiment spec
>
> One-page spec for the highest-value label: probe-target model, prompt, activation capture point, probe family, split plan, baselines, success criterion. This either closes this skill or hands off to a future `latent-label-probe-design` skill.
>
> #### Stage 18. Parking lot
>
> Questions and labels deferred, each with the reason and the condition that would bring them back (e.g., "brought back if Skill B produces matched theory-paired cases").

---

## 6. Drop-in: Required Outputs (revises current Outputs section)

> Each run of this skill should produce, as a versioned artifact bundle:
>
> 1. Intake card
> 2. Extracted mechanistic questions with benchmark evidence
> 3. Tractability and fit-check report
> 4. Native-label inventory
> 5. Refined ontology (prompt-side, response-side, nuisance)
> 6. Signal-location hypothesis per question
> 7. Frozen label dataset (CSV/parquet with row ID, labels, labeler version, content hash)
> 8. Labeling function specs
> 9. Derivability validation report
> 10. Surface-feature baseline report
> 11. Inter-label and label-vs-nuisance correlation matrices
> 12. Confound audit and stratification plan
> 13. Success and kill criteria per label family
> 14. Gap list (Skill B contract)
> 15. First experiment spec
> 16. Parking lot
> 17. Audit log of decisions and their justifications

Item 17 (audit log) is what distinguishes a reproducible skill output from a one-off analysis.

---

## 7. Drop-in: Stopping Criteria (new section)

> ### Stopping Criteria
>
> This skill is done when all of the following hold:
>
> - Ontology frozen (Stage 15) with ≥1 label family validated for derivability.
> - Surface-feature baselines and correlation audit complete.
> - Confound stratification plan written and implementable with current data, or escalated to Skill B via the gap list.
> - At least one first-experiment spec written.
> - Gap list either empty or non-empty-with-handoff, and parking lot committed.
>
> Kill the benchmark (do not proceed) if:
>
> - No extracted mechanistic question survives the tractability check.
> - Post-stratification cell sizes are below floor for every candidate label, and Skill B cannot cheaply repair this.
> - All candidate labels resolve to case-specific rubric titles with no generalizable core.
> - The benchmark is entirely rubric-scored without raw prompt/response text to label against.
>
> Neither "we ran out of ideas" nor "the ontology is still imperfect" is a stopping condition. Imperfection is expected; the ontology freeze is a commitment to a concrete first pass.

---

## 8. Drop-in: Escalation Triggers to Skill B (new section)

> ### When to Escalate to latent-label-data-augmentation
>
> Escalate when any of the following hold after Stage 14:
>
> **Statistical**
> - Post-stratification cell size < floor (e.g., 50) for any target × nuisance combination needed to answer a question.
> - Target axis imbalanced beyond recoverable weighting (e.g., 95% one class).
> - Correlation between any target label and any nuisance variable exceeds the stratification threshold (e.g., 0.7).
>
> **Design**
> - A question is causal but no matched pairs exist in the benchmark.
> - A response-side label requires generations but the benchmark is prompt-only.
> - A theory / framing overlay is required but the benchmark is not paired across framings.
> - A label is well-defined but has too few positive examples to validate even with few-shot LLM labeling.
>
> **Quality**
> - Derivability is reliable only on a restricted subset (e.g., long cases only); rewrites would bring other subsets into scope.
> - The only way to rule out a confound is a counterfactual the benchmark lacks.
>
> **Do not escalate — stop instead — when:**
> - Post-augmentation, the question still requires data that cannot be constructed without introducing its own artifacts.
> - The labeling or generation cost exceeds the information value; a purpose-built benchmark is cheaper than repair.

---

## 9. Drop-in: Canonical cross-benchmark label vocabulary (new section)

A shared, benchmark-independent vocabulary makes this skill reusable rather than one-off. The vocabulary below is a minimal seed, not a final list.

> ### Canonical Label Axes (reusable across benchmarks)
>
> **Prompt-side structure**
> - `role_framing` — advisor / agent / observer / unspecified
> - `subject_of_action` — ai_system / human_actor / institutional_actor
> - `person_grammar` — first / second / third
> - `information_stance` — facts_stipulated / facts_uncertain / facts_contested
> - `stakes_framing` — life_safety / livelihood / relational / institutional / mixed
> - `length_bucket` — nuisance, benchmark-specific
> - `source_template` — nuisance, benchmark-specific
>
> **Response-side process**
> - `considerations_represented` — integer count of morally / epistemically / strategically live options named
> - `uncertainty_markers_present` — binary
> - `commitment_transition_token` — position in the generation where the model moves from analysis to recommendation
> - `refuses_or_hedges` — capability_refusal / value_refusal / epistemic_hedge / none
>
> **Response-side objective orientation**
> - `harm_avoidance_invoked` — binary, on the generation
> - `helpfulness_invoked` — binary, on the generation
>
> **Nuisance set (always track)**
> - source, length, template, person grammar, domain, annotator family, rubric family, ideology, model family
>
> For any new benchmark, first ask which of these canonical axes the benchmark supports, then add benchmark-specific axes on top. This inverts the default "every benchmark gets its own ontology" pattern.

---

## 10. Additions to Common Failure Modes

Two failure modes to add to the existing section:

> **Hidden inputs drift the ontology.** Which probe-target model, which generation protocol, which activation capture regime — if these are left implicit, two researchers running the process on the same benchmark will produce different ontologies for defensible reasons. Surface these as required inputs (§Required Inputs) so drift is visible rather than invisible.
>
> **Labels trained on the same data they are evaluated on.** A labeling function derived from a strong LLM, a probe trained on that label, and an evaluation on the same dilemmas will produce inflated apparent signal. Train/probe/eval splits must be set before ontology freeze and respected at every stage.

## 11. Additions to confound checklist (Stage 14)

The current confound list is good. Add:

- `model_family` — a label meaningful for RLHFed instruct models may not exist in base models; do not assume invariance.
- `probe_training_data_confound` — the probe learns a proxy through spurious correlation in the training split; check transfer to held-out distribution.
- `annotator_LLM_family` — if the labeling function is an LLM, its family and version is a nuisance that can correlate with apparent label structure.

## 12. Naming: Skill B

Recommend naming the downstream skill `latent-label-data-augmentation` (or `contrast-set-construction`), not `synthetic-data-generation`. Most of Skill B's value is in rewrites, minimal pairs, paired elicitations, and confound counterbalancing of existing data. "Synthetic data" is only one technique inside it, and the least artifact-safe one. Naming the skill after its weakest technique will bias users toward using it.

The existing `synthetic-data-generation` skill in the repo (surfaced in the environment) should be reviewed for whether it already covers this ground — if so, rename and rescope; if not, build alongside.

## 13. Note on Skill C (future, not now)

The current workflow smuggles probe-experiment-design work into Stages 8, 10, and 17. For v1, keep it folded in. If this skill bloats, probe-experiment-design is the clean split:

> **Skill C: `latent-label-probe-design`** — takes a Latent Label Spec (output of Skill A) and produces a concrete probe protocol: activation capture spec, train/test split design, stratification plan, baselines, success criteria, intervention design.

Don't build this now. Build Skill A clean, build Skill B when Skill A's gap list demands it, and split Skill C off only if A becomes unwieldy.

---

## 14. What NOT to change from the current doc

To be explicit to future readers:

- Keep the Purpose, Target Output, Practical Checklist, and What Came Out Of This Process So Far sections largely as-is. The Practical Checklist is a good companion to the revised workflow.
- Keep the five-level match rating.
- Keep the label-type taxonomy in the old step 2 (with the minor outcome/rubric split).
- Keep the Common Failure Modes section; only add the two modes in §10 above.
- Keep the signal-location taxonomy from the old step 9 (moved earlier, not rewritten).

---

## 15. Open questions for Codex's integration pass

1. Should the canonical label vocabulary (§9) live in this doc or in a separate `canonical-label-vocabulary.md` that both Skill A and Skill B import? I'd lean toward separate, because the vocabulary grows as more benchmarks are processed and shouldn't churn this doc.
2. Should the Required Inputs section (§3) be echoed in the Practical Checklist? Probably yes, as a first checkbox set.
3. Is there appetite for codifying Skill A and Skill B as actual `.skill` files in the repo now, or should this doc remain a methodology note until multiple benchmarks have run through it?
4. Does `benchmark-mech-interp` (the existing skill in the repo) already cover Skill A territory? If yes, this doc's revisions should flow into that skill's spec rather than creating a parallel structure.
