# Mech Interp Methodology Roster

**Date:** 2026-04-21

## Purpose

This document extracts the actual methodological approaches used across the replication-priorities papers and organizes them as a reusable roster.

The point is not to say only “this paper uses probes.”
The point is to be explicit about:

- exactly what kind of readout or intervention is being used
- what kind of labels or contrast structure it needs
- what artifacts it produces
- what kinds of benchmark substrates it is a good fit for
- what kinds of claims it can and cannot support

This should be read alongside:

- [mech-interp-replication-priorities-2026-04-20.md](mech-interp-replication-priorities-2026-04-20.md)
- [benchmark-to-mech-interp.md](benchmark-to-mech-interp.md)

The intended workflow is:

1. identify a benchmark with strong labels
2. identify which methodology family best fits those labels
3. choose an initial discovery recipe
4. escalate from descriptive readouts to stronger causal tests where possible

## Methodology Families

### 1. Linear Residual Probes

- What it is:
  Train a linear classifier or regressor directly on residual-stream activations, usually at one token position and one or more layers.
- Typical implementation details:
  Logistic regression or linear regression on normalized residual activations, often sweeping layers and token positions. Sometimes trained on the final prompt token, sometimes on claim tokens, sometimes on special-template tokens.
- What labels it needs:
  Binary or multiclass labels such as role, conflict outcome, honesty/deception, hallucinated/not, high-stakes/not, theory A vs theory B, or scalar rubric dimensions.
- What it produces:
  Probe weights, per-layer separability curves, calibration thresholds, and often direction vectors that can later be reused for steering or transfer analysis.
- Good fit for:
  Rubric dimensions, role labels, theory labels, deception labels, claim-level factuality labels, and any dataset with enough examples for supervised decoding.
- Strongest example papers:
  `Prompt Injection as Role Confusion`, `Who is In Charge?`, `Simple Factuality Probes`, `Detecting Strategic Deception`, `Detecting High-Stakes Interactions`.
- What kind of claim it supports:
  “This variable is linearly decodable at this layer/token.”
- Main limitation:
  A probe can read out information without proving the model uses it causally.

### 2. Difference-In-Means Directions

- What it is:
  Compute a direction by subtracting the mean activation for one condition from the mean activation for another.
- Typical implementation details:
  Average residual vectors across two classes, then use the resulting direction as a readout or steering vector. Often one of the fastest first-pass methods for contrastive data.
- What labels it needs:
  Paired or contrastive labels such as honest vs deceptive, pressured vs unpressured, theory A vs theory B, sycophantic vs non-sycophantic, high-stakes vs low-stakes.
- What it produces:
  A simple concept direction that can be scored, compared across layers/models, or used for activation steering.
- Good fit for:
  Contrastive datasets like MASK, SycophancyEval, Anthropic model-written evals, ManyIH-Bench-style hierarchy contrasts, and theory-labeled benchmark slices.
- Strongest example papers:
  `Valence-Arousal Subspace`, `Can You Trust an LLM with Your Life-Changing Decision?`, `Who is In Charge?` steering baseline, many persona/state papers.
- What kind of claim it supports:
  “These two conditions differ consistently in activation space along this direction.”
- Main limitation:
  Very sensitive to confounds if the two classes differ in style, format, or domain rather than only the intended variable.

### 3. Geometry Recovery With PCA Or Similarity Structure

- What it is:
  Recover low-dimensional structure from a family of concept directions or activation differences.
- Typical implementation details:
  Build many mean-difference vectors, run PCA or another factorization, then align the components with external human ratings or task attributes.
- What labels it needs:
  Many related concept labels rather than only two classes, such as dozens of emotions or multiple theory variants.
- What it produces:
  Low-dimensional axes, geometric embeddings, component loadings, and interpretable subspaces.
- Good fit for:
  Emotion sets, persona taxonomies, norm clusters, or hallucination/error taxonomies where concepts are related rather than isolated.
- Strongest example papers:
  `Valence-Arousal Subspace in LLMs`.
- What kind of claim it supports:
  “This family of concepts organizes into a structured latent geometry.”
- Main limitation:
  Geometry can be descriptive without proving the axes are functionally used by the model.

### 4. Regression Onto External Human Ratings

- What it is:
  Fit a regressor from internal directions/components onto external scalar ratings such as valence, arousal, or another human judgment.
- Typical implementation details:
  Recover concept vectors first, then use ridge regression or another simple model to align them to human-labeled scales.
- What labels it needs:
  External continuous ratings or ordered scales.
- What it produces:
  Interpretable scalar axes and quantitative alignment between internal geometry and human annotation spaces.
- Good fit for:
  Emotion data, risk scores, confidence/caution scales, or benchmark dimensions with human scalar ratings.
- Strongest example papers:
  `Valence-Arousal Subspace in LLMs`.
- What kind of claim it supports:
  “The internal subspace aligns with a meaningful human scale.”
- Main limitation:
  Alignment to ratings does not by itself show the model uses that variable causally during generation.

### 5. Residual Subtraction And Confound Projection

- What it is:
  Construct a concept vector while explicitly subtracting nuisance structure such as shared mean activity or neutral-style confounds.
- Typical implementation details:
  Compute residual activations after the concept is established, subtract cross-class means, and project out neutral or formatting directions to isolate the target concept.
- What labels it needs:
  Multi-class concept labels plus some explicit neutral or control condition.
- What it produces:
  Cleaner concept vectors that are less contaminated by topic/style and more tied to the intended semantic variable.
- Good fit for:
  Emotion concepts, persona concepts, therapy styles, or moral-framework contrasts where style confounds are likely.
- Strongest example papers:
  `Emotion concepts and their function in a large language model`.
- What kind of claim it supports:
  “This concept vector is not just picking up generic style/topic.”
- Main limitation:
  Depends on good controls; poor neutral controls can still leave major confounds.

### 6. Layer Sweep And Token-Position Sweep

- What it is:
  Evaluate the same decoding problem across many layers and many token positions to see where the signal first appears and where it stabilizes.
- Typical implementation details:
  Train/evaluate identical probes at each layer or on each token class, often plotting separability vs layer depth.
- What labels it needs:
  Any supervised label, but especially useful for sequence tasks where timing matters.
- What it produces:
  Localization curves over layer depth and token position, giving a first map of where a variable is represented.
- Good fit for:
  Role conflict, source attribution, deception onset, hallucination emergence, and multi-turn risk escalation.
- Strongest example papers:
  `Who is In Charge?`, `Detecting Strategic Deception`, `Emotion concepts`, `Role Confusion`.
- What kind of claim it supports:
  “This variable becomes linearly readable around these layers/tokens.”
- Main limitation:
  Readability does not prove that these are the write sites rather than read sites.

### 7. Claim Decomposition And Claim-Level Supervision

- What it is:
  Break long-form responses into atomic claims, then attach labels or evidence to each claim instead of scoring the whole answer at once.
- Typical implementation details:
  Use an auxiliary model or structured heuristic to split answers into claims, align claims to source spans or evidence, then train probes on the claim-token activations.
- What labels it needs:
  Claim-level factuality labels, evidence spans, or domain-specific support annotations.
- What it produces:
  Claim-level monitors, risk heatmaps, and much better localized supervision for groundedness studies.
- Good fit for:
  RAGTruth-style datasets, finance/legal report checking, citation-grounded generation, or any long-form benchmark with support labels.
- Strongest example papers:
  `Simple Factuality Probes`.
- What kind of claim it supports:
  “Specific claims become unsupported or risky here.”
- Main limitation:
  The decomposition pipeline can introduce errors or bleed benchmark artifacts into the labels.

### 8. Retrieval-Based Oracle Labeling

- What it is:
  Use a retrieval system or evidence verifier as an approximate oracle to produce labels for factuality/support rather than relying only on humans.
- Typical implementation details:
  Retrieve relevant evidence, verify each claim against it, and use those labels to supervise hidden-state probes.
- What labels it needs:
  Source corpora or knowledge bases, plus a verification step.
- What it produces:
  Scalable claim-level labels and domain-portable factuality harnesses.
- Good fit for:
  Finance, legal, health, RAG, citation-heavy tasks, and long-form analysis settings.
- Strongest example papers:
  `Simple Factuality Probes`.
- What kind of claim it supports:
  “This hidden-state signal tracks support/factuality as judged by the retrieval oracle.”
- Main limitation:
  Oracle quality caps label quality.

### 9. Logit Lens And Output Attribution

- What it is:
  Inspect intermediate representations by projecting them toward vocabulary logits or attributing final-token behavior back to internal activations.
- Typical implementation details:
  Apply logit lens at multiple layers or use attribution methods to estimate which components contribute to a final decision.
- What labels it needs:
  Usually paired with a known output decision or class label, but does not always require explicit supervision.
- What it produces:
  Layer-wise pictures of how a decision or token prediction is taking shape.
- Good fit for:
  Deception, role conflict, refusal/continuation competition, and situations where we care about how an output becomes likely over depth.
- Strongest example papers:
  `Can LLMs Lie?`, `Who is In Charge?`.
- What kind of claim it supports:
  “This intermediate state is already pushing toward this output.”
- Main limitation:
  Often descriptive and sensitive to interpretation.

### 10. Zero Ablation Of MLPs Or Attention Components

- What it is:
  Set selected MLP blocks, attention outputs, or heads to zero during inference to test whether behavior changes.
- Typical implementation details:
  Identify candidate components from prior analysis, ablate them at selected token positions/layers, then measure effect on the target behavior.
- What labels it needs:
  Behavioral metric or task success label; usually paired with a contrastive task.
- What it produces:
  Causal evidence that a component matters for the behavior.
- Good fit for:
  Deception, refusal, source attribution, and tasks where a specific behavior can be measured clearly.
- Strongest example papers:
  `Can LLMs Lie?`, `The Struggle Between Continuation and Refusal`.
- What kind of claim it supports:
  “These components are causally important for this behavior.”
- Main limitation:
  Zeroing can be a blunt intervention and may introduce off-manifold effects.

### 11. Head-Level Ablation And Head Ranking

- What it is:
  Evaluate individual attention heads or small head sets by ablating them and measuring behavior changes.
- Typical implementation details:
  Score heads individually or greedily, then test the smallest set whose removal changes the behavior significantly.
- What labels it needs:
  A measurable target behavior such as lying, refusal, or jailbreak success.
- What it produces:
  Ranked head lists, compact causal subsets, and more interpretable intervention targets than whole-layer ablations.
- Good fit for:
  Refusal circuits, deception planning, jailbreak mechanisms, hierarchy resolution.
- Strongest example papers:
  `Can LLMs Lie?`, `The Struggle Between Continuation and Refusal`.
- What kind of claim it supports:
  “A small head subset is doing a lot of the work.”
- Main limitation:
  Heads can be redundant and distributed; ranking one by one may miss interactions.

### 12. Attention Blocking Between Specific Token Classes

- What it is:
  Prevent attention flow from one token set to another to test whether information transfer along that path is necessary.
- Typical implementation details:
  Mask or zero attention links from, for example, subject tokens or intent tokens into dummy template tokens, then measure the behavioral effect.
- What labels it needs:
  A task with meaningful token classes and a measurable output behavior.
- What it produces:
  Stronger causal path claims than global ablation alone.
- Good fit for:
  Deception, source confusion, retrieval-vs-system conflicts, tool-output influence, template-token scratchpads.
- Strongest example papers:
  `Can LLMs Lie?`.
- What kind of claim it supports:
  “This information path is important for the target behavior.”
- Main limitation:
  Requires a good hypothesis about which token classes matter.

### 13. Contrastive Activation Steering

- What it is:
  Add or subtract a concept direction during inference to test whether behavior moves in a predicted direction.
- Typical implementation details:
  Construct a vector from differences, probe weights, or another concept representation, inject it at chosen layers/positions, and measure behavioral shift.
- What labels it needs:
  A contrastive concept or supervised signal from which the direction can be built.
- What it produces:
  Steering vectors and causal evidence that moving along the direction affects behavior.
- Good fit for:
  Emotion, honesty, caution, moral framework, source trust, harmfulness, or any label with a reasonably stable direction.
- Strongest example papers:
  `Valence-Arousal Subspace`, `Can LLMs Lie?`, `Truthfulness Spectrum`, `Emotion concepts`, `CDAS`.
- What kind of claim it supports:
  “This direction is not only readable; intervening on it changes behavior.”
- Main limitation:
  Steering can be lossy, off-manifold, or confounded by correlated features.

### 14. Cross-Domain Probe Transfer Matrices

- What it is:
  Train probes on one domain or task, test them on others, and build a transfer map.
- Typical implementation details:
  Pairwise train/test across datasets or label families, then compare accuracy, AUROC, or another metric.
- What labels it needs:
  Shared concept labels across multiple domains or task families.
- What it produces:
  Transfer matrices, portability estimates, and insight into which concept directions are general vs domain-specific.
- Good fit for:
  Truthfulness, factuality, groundedness, risk-awareness, or any benchmark family with multiple related domains.
- Strongest example papers:
  `Truthfulness Spectrum`, `Simple Factuality Probes`.
- What kind of claim it supports:
  “This concept transfers across these domains but not these others.”
- Main limitation:
  Transfer failure can reflect data mismatch, not only concept specificity.

### 15. Mahalanobis-Cosine Similarity For Probe Portability

- What it is:
  Compare probe directions using covariance-aware geometry rather than plain cosine similarity.
- Typical implementation details:
  Estimate the covariance structure of the representation space, then compute similarity in the whitened space and relate that to observed transfer performance.
- What labels it needs:
  Multiple trained probes over related concepts/domains.
- What it produces:
  A geometry-based transfer estimator.
- Good fit for:
  Probe banks, domain transfer analysis, and deciding whether a direction trained on one benchmark should move to another.
- Strongest example papers:
  `The Truthfulness Spectrum Hypothesis`.
- What kind of claim it supports:
  “These probe directions are likely to transfer because they are close in the right geometry.”
- Main limitation:
  Promising but still needs broader validation beyond the original paper’s setting.

### 16. Concept Erasure With LEACE And Stratified INLP

- What it is:
  Remove a concept direction from representations to test whether the model still succeeds or whether certain behaviors disappear.
- Typical implementation details:
  Use LEACE or INLP-style projection to erase the learned subspace, sometimes stratified by domain, then evaluate performance/behavior afterward.
- What labels it needs:
  A supervised concept and enough data to estimate the subspace robustly.
- What it produces:
  Domain-general vs domain-specific subspaces, and a stronger test of concept necessity than readout alone.
- Good fit for:
  Truthfulness, sycophancy, theory labels, or benchmark dimensions with clear supervision.
- Strongest example papers:
  `The Truthfulness Spectrum Hypothesis`.
- What kind of claim it supports:
  “This behavior depends on this subspace, and this part is general vs domain-specific.”
- Main limitation:
  Erasure can damage unrelated information if the concept is entangled.

### 17. Layer-Wise Activation Projection To Reference Models

- What it is:
  Compare one model’s activations against reference differences from other models, often base vs instruct vs fine-tuned variants.
- Typical implementation details:
  Compute residual differences between model variants, project one run onto those differences layer by layer, and see which reference manifold it aligns with over depth.
- What labels it needs:
  Usually model-variant comparisons rather than benchmark labels.
- What it produces:
  Internal drift curves and alignment/misalignment trajectories over layers.
- Good fit for:
  Post-finetune audit questions, alignment erosion, preference optimization studies, and model-family comparisons.
- Strongest example papers:
  `Re-Emergent Misalignment`.
- What kind of claim it supports:
  “This fine-tuned model drifts internally toward or away from the aligned reference over depth.”
- Main limitation:
  Comparative and geometric, but not always strongly causal.

### 18. Loss Geometry And Gradient Similarity

- What it is:
  Use gradients or loss surfaces to compare how similarly different models/processes treat the same examples.
- Typical implementation details:
  Measure gradient similarity, output-probability geometry, or loss alignment across model variants.
- What labels it needs:
  Behavioral targets or curated example sets, often harmful vs harmless.
- What it produces:
  Evidence that two variants process a behavior similarly or differently at the optimization/gradient level.
- Good fit for:
  Alignment erosion, preference optimization, and post-finetune audit work.
- Strongest example papers:
  `Re-Emergent Misalignment`.
- What kind of claim it supports:
  “These models are similar or divergent in how they treat this behavior.”
- Main limitation:
  Harder to turn directly into product-like monitors.

### 19. Residual-Difference SVD Or Low-Rank Shared-Direction Discovery

- What it is:
  Factorize residual differences between conditions/models to find shared low-rank directions.
- Typical implementation details:
  Compute residual difference matrices and run SVD or related factorization to find common latent axes.
- What labels it needs:
  Model variants or multiple related harmfulness settings.
- What it produces:
  Shared latent directions linking multiple behaviors or domains.
- Good fit for:
  Misalignment, safety-drift, harm transfer, and domain-adaptation audits.
- Strongest example papers:
  `Re-Emergent Misalignment`.
- What kind of claim it supports:
  “These seemingly different behaviors share a latent internal axis.”
- Main limitation:
  Shared direction does not automatically imply a manipulable or unique mechanism.

### 20. Signed SNIP-Style Targeted Weight Pruning

- What it is:
  Score weights for how much they contribute to harmful behavior vs benign capability, then prune a targeted subset.
- Typical implementation details:
  Use harmful-response data plus a preservation dataset, compute signed saliency scores, remove a tiny selected subset of weights, then test both harmfulness and utility.
- What labels it needs:
  Harmful/benign example sets and a capability-preservation dataset.
- What it produces:
  A pruned model, weight subsets, and evidence about whether harm depends on a compact shared subnetwork.
- Good fit for:
  Harm generation, refusal, emergent misalignment, and other open-weight safety questions.
- Strongest example papers:
  `Pruning a Tiny High-Harm Subnetwork from LLMs`.
- What kind of claim it supports:
  “This tiny weight subset is causally important for harmful generation.”
- Main limitation:
  Weight-level, not activation-first; harder to turn directly into a monitor.

### 21. Generation-Vs-Detection Dissociation Tests

- What it is:
  Test whether a model can still recognize/explain harmful content after an intervention that reduces harmful generation.
- Typical implementation details:
  Measure generation ability, detection ability, refusal ability, and explanation ability separately before and after intervention.
- What labels it needs:
  Distinct evaluation sets for generation, recognition, refusal, or explanation.
- What it produces:
  A dissociation result showing whether “knowing” and “doing” are separated by the intervention.
- Good fit for:
  Harmfulness, deception, refusal, and safety mechanism questions.
- Strongest example papers:
  `Pruning a Tiny High-Harm Subnetwork from LLMs`.
- What kind of claim it supports:
  “The model can still recognize the bad behavior even though it no longer executes it.”
- Main limitation:
  Requires well-designed evaluation splits to avoid ambiguous interpretation.

### 22. Probe Architecture Bakeoffs

- What it is:
  Compare multiple probe families rather than assuming a single linear probe is enough.
- Typical implementation details:
  Evaluate linear probes, MLP probes, attention-based probes, pooling strategies, context windows, and cascades.
- What labels it needs:
  Any supervised label with enough train/test data, especially where OOD behavior matters.
- What it produces:
  Performance comparisons, robustness curves, and deployment guidance.
- Good fit for:
  High-stakes detection, cyber misuse, long-context monitoring, or any product-facing benchmark-first work.
- Strongest example papers:
  `Detecting High-Stakes Interactions`, `Building Production-Ready Probes for Gemini`.
- What kind of claim it supports:
  “This probe family is more robust/effective for this deployment setting.”
- Main limitation:
  More engineering-heavy and less mechanistically explanatory on its own.

### 23. Cascade Systems

- What it is:
  Use a cheap activation-side monitor as a first pass and escalate flagged cases to a larger model or more expensive judge.
- Typical implementation details:
  Probe scores determine whether to route to a stronger evaluator or additional analysis pipeline.
- What labels it needs:
  A task label plus some notion of high-cost review benefit.
- What it produces:
  Practical monitor pipelines and cost-performance tradeoff curves.
- Good fit for:
  High-stakes triage, groundedness review, cyber misuse, counseling risk, finance/legal escalation.
- Strongest example papers:
  `Detecting High-Stakes Interactions`, `Building Production-Ready Probes for Gemini`.
- What kind of claim it supports:
  “A cheap white-box filter can reduce cost while preserving most of the value of a larger monitor.”
- Main limitation:
  Product-useful, but not always a deep mechanistic claim.

### 24. Long-Context And Distribution-Shift Stress Testing

- What it is:
  Evaluate whether probe directions or architectures survive when context is longer, multi-turn, jailbroken, or adversarially shifted.
- Typical implementation details:
  Test the same monitor under clean, long-context, multi-turn, jailbreak, and adaptive-red-team conditions.
- What labels it needs:
  Evaluation datasets that intentionally span deployment shifts.
- What it produces:
  OOD robustness estimates and deployment-readiness criteria.
- Good fit for:
  Any benchmark-first monitor likely to ship, especially agent, high-stakes, or retrieval products.
- Strongest example papers:
  `Building Production-Ready Probes for Gemini`.
- What kind of claim it supports:
  “This monitor survives or fails under realistic shift.”
- Main limitation:
  Critical for shipping, but not itself a discovery method unless paired with deeper analysis.

### 25. Architecture Search For Probe Design

- What it is:
  Search over probe architectures automatically rather than choosing one by hand.
- Typical implementation details:
  Use methods like AlphaEvolve or other search procedures to discover better long-context or multi-turn probe architectures.
- What labels it needs:
  A supervised task and enough evaluation bandwidth to score many candidate probes.
- What it produces:
  Better-performing probe architectures and sometimes interpretable design lessons.
- Good fit for:
  Product-facing benchmark-first work where robustness matters more than minimalism.
- Strongest example papers:
  `Building Production-Ready Probes for Gemini`.
- What kind of claim it supports:
  “This architecture is more effective for this monitoring problem.”
- Main limitation:
  More systems/engineering than mechanism explanation.

### 26. Path Patching

- What it is:
  Intervene on a specific activation path, not just a whole component, by swapping or patching signals from one run into another.
- Typical implementation details:
  Patch attention or residual paths between clean and attacked runs to localize the path carrying the critical information.
- What labels it needs:
  Paired examples where one run has the target behavior and another does not.
- What it produces:
  Much stronger causal localization than readout alone.
- Good fit for:
  Jailbreaks, source confusion, role conflict, theory contrasts, and any benchmark with matched conditions.
- Strongest example papers:
  `The Struggle Between Continuation and Refusal`; also highly relevant to `Role Confusion` style future work.
- What kind of claim it supports:
  “This path carries the decisive information for the behavior.”
- Main limitation:
  Requires carefully matched paired examples and good intervention hygiene.

### 27. Activation Scaling Of Candidate Components

- What it is:
  Multiply selected heads/features/components up or down to test whether the target behavior changes monotonically.
- Typical implementation details:
  Identify candidate components, rescale them during inference, then measure behavior or attack success rate.
- What labels it needs:
  Clear behavioral metric such as ASR, refusal, deception rate, or rubric dimension score.
- What it produces:
  Causal sensitivity curves and evidence that a component is not just correlated but functionally influential.
- Good fit for:
  Refusal circuits, harmfulness, emotion intensity, or any near-linear component hypothesis.
- Strongest example papers:
  `The Struggle Between Continuation and Refusal`.
- What kind of claim it supports:
  “Increasing this component pushes the model toward or away from this behavior.”
- Main limitation:
  Scaling can create off-distribution behavior if overused.

### 28. Distributed Interchange Interventions And Distribution Matching

- What it is:
  Learn steering interventions using interchange-based causal ideas plus distribution matching rather than only simple direction vectors.
- Typical implementation details:
  Use DII-style interventions and a distributional objective like JSD to find more faithful steering directions or subspaces.
- What labels it needs:
  A target behavior/concept and a way to compare the intervened distribution to a reference distribution.
- What it produces:
  Faithful steering vectors/subspaces and stronger causal control than naive contrastive steering.
- Good fit for:
  Behaviors where ordinary steering distorts utility too much, including refusal, backdoors, or fine-grained benchmark rubric dimensions.
- Strongest example papers:
  `Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions`.
- What kind of claim it supports:
  “We can steer this concept while preserving more of the surrounding behavior.”
- Main limitation:
  More complex and infrastructure-heavy than simple readout methods.

### 29. Synthetic Data Construction For Mechanistic Supervision

- What it is:
  Deliberately generate examples that isolate a latent variable so downstream probes and interventions are cleaner.
- Typical implementation details:
  Create synthetic role corpora, conflict prompts, emotion stories, or high-stakes scenarios, often with procedural controls to hold nuisance factors constant.
- What labels it needs:
  Often self-generated labels from the synthetic template itself.
- What it produces:
  Clean supervision sets, matched controls, and scalable training/eval harnesses.
- Good fit for:
  Role integrity, theory labels, high-stakes classification, emotion concepts, hierarchy resolution.
- Strongest example papers:
  `Role Confusion`, `Emotion concepts`, `Who is In Charge?`, `Detecting High-Stakes Interactions`.
- What kind of claim it supports:
  “We can isolate this variable cleanly enough to study it mechanistically.”
- Main limitation:
  Synthetic simplicity may not survive transfer to real-world data.

## Mapping Methodologies To Label Structures

### Best fit for rubric-rich benchmarks

- Linear residual probes
- multi-label probe families
- claim decomposition if outputs are long-form
- layer/token sweeps
- contrastive steering on single rubric dimensions
- long-context robustness testing if the use case is product-facing

Best benchmark examples:

- MoReBench
- CounselBench
- HealthBench
- PLawBench

### Best fit for theory or framework labels

- difference-in-means directions
- framework-conditioned probes
- PCA/subspace recovery across many related concepts
- activation steering
- patching between matched frameworks
- LEACE / INLP erasure for domain-general vs domain-specific structure

Best benchmark examples:

- MoReBench theory subset
- CBT-Bench
- MedSafetyBench

### Best fit for span or claim supervision

- claim decomposition
- token/span probes
- emergence-point tracing
- retrieval-oracle supervision
- hallucination-type transfer matrices

Best benchmark examples:

- RAGTruth
- FAVA
- FELM
- HALoGEN
- LegalBench-RAG

### Best fit for belief-vs-pressure or contrastive honesty data

- linear probes
- difference-in-means vectors
- patching between belief and pressured runs
- attention blocking
- head ablation
- cross-domain transfer mapping

Best benchmark examples:

- MASK
- SycophancyEval
- some Anthropic model-written eval slices

### Best fit for hierarchy, role, or source labels

- role/source probes
- layer sweeps
- path patching
- attention blocking
- synthetic-data warm-starts
- transfer from clean hierarchy data into real agent traces

Best benchmark examples:

- ManyIH-Bench
- AgentDojo
- Who is In Charge?
- BIPIA

### Best fit for multi-turn or trajectory benchmarks

- conversation-state probes
- turn-by-turn scoring
- drift analysis
- intervention timing experiments
- cascade systems
- long-context robustness testing

Best benchmark examples:

- AgentDojo
- HealthBench
- MACHIAVELLI
- tau-bench

## Practical Escalation Ladder

When a new benchmark looks promising, the most useful order is usually:

1. `descriptive readout`
   layer sweeps, token sweeps, simple probes
2. `geometry or transfer`
   compare layers, domains, models, transferability
3. `intervention`
   steering, erasure, ablation, patching
4. `deployment hardening`
   OOD testing, long-context testing, cascades, architecture bakeoffs

This helps us avoid jumping to strong causal claims too early, while still giving a path from benchmark to usable monitor.

## Bottom Line

The main methodological lesson from the replication set is that there is no single “probe method.”
There is a stack:

- `readout methods` for finding signals
- `geometry methods` for understanding structure and transfer
- `causal methods` for testing whether the signal matters
- `deployment methods` for figuring out whether the thing survives real use

That is exactly what we need for the benchmark-first program.

The next step is to cross-reference this roster with [benchmark-to-mech-interp.md](benchmark-to-mech-interp.md) so that for each top benchmark we can say:

- what labels it has
- what methodology families fit it best
- what the first experiment should be
- what stronger follow-up tests would look like
