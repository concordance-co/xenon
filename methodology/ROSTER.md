# Methodology Roster

Reusable catalog of methods for mechanistic interpretability work inside the flywheel.

For each method, explicit about:

- what kind of readout or intervention it is
- what labels or contrast structure it needs
- what artifacts it produces
- what kinds of data substrates it fits
- what claims it can and cannot support

Intended use:

1. identify the current flywheel stage and the question being asked
2. identify which methodology families fit the available labels
3. choose an initial recipe
4. escalate claims per the evidence ladder, not per the flywheel

Read alongside `FLYWHEEL.md` and `PRINCIPLES.md`.

---

## Methodology Families

### 1. Linear Residual Probes

- What it is: Train a linear classifier or regressor directly on residual-stream activations, usually at one token position and one or more layers.
- Typical implementation: Logistic regression or linear regression on normalized residual activations, often sweeping layers and token positions. Sometimes trained on the final prompt token, sometimes on claim tokens, sometimes on special-template tokens.
- Labels needed: Binary or multiclass labels — role, conflict outcome, honesty/deception, hallucinated/not, high-stakes/not, theory A vs theory B, or scalar rubric dimensions.
- What it produces: Probe weights, per-layer separability curves, calibration thresholds, direction vectors reusable for steering or transfer.
- Good fit for: Rubric dimensions, role labels, theory labels, deception labels, claim-level factuality labels, any dataset with enough examples for supervised decoding.
- Example papers: `Prompt Injection as Role Confusion`, `Who is In Charge?`, `Simple Factuality Probes`, `Detecting Strategic Deception`, `Detecting High-Stakes Interactions`.
- Supports: "This variable is linearly decodable at this layer/token."
- Limit: A probe can read out information without proving the model uses it causally.

### 2. Difference-In-Means Directions

- What it is: Compute a direction by subtracting the mean activation for one condition from the mean activation for another.
- Typical implementation: Average residual vectors across two classes, then use the resulting direction as a readout or steering vector. One of the fastest first-pass methods for contrastive data.
- Labels needed: Paired or contrastive labels — honest vs deceptive, pressured vs unpressured, theory A vs theory B, sycophantic vs non-sycophantic, high-stakes vs low-stakes.
- What it produces: A simple concept direction that can be scored, compared across layers/models, or used for activation steering.
- Good fit for: Contrastive datasets like MASK, SycophancyEval, Anthropic model-written evals, ManyIH-Bench-style hierarchy contrasts, theory-labeled slices.
- Example papers: `Valence-Arousal Subspace`, `Can You Trust an LLM with Your Life-Changing Decision?`, `Who is In Charge?` steering baseline, persona/state papers.
- Supports: "These two conditions differ consistently in activation space along this direction."
- Limit: Very sensitive to confounds if the two classes differ in style, format, or domain rather than only the intended variable.

### 3. Geometry Recovery With PCA Or Similarity Structure

- What it is: Recover low-dimensional structure from a family of concept directions or activation differences.
- Typical implementation: Build many mean-difference vectors, run PCA or another factorization, align components with external human ratings or task attributes.
- Labels needed: Many related concept labels rather than only two classes — dozens of emotions, multiple theory variants.
- What it produces: Low-dimensional axes, geometric embeddings, component loadings, interpretable subspaces.
- Good fit for: Emotion sets, persona taxonomies, norm clusters, hallucination/error taxonomies where concepts are related rather than isolated.
- Example papers: `Valence-Arousal Subspace in LLMs`.
- Supports: "This family of concepts organizes into a structured latent geometry."
- Limit: Geometry can be descriptive without proving the axes are functionally used by the model.

### 4. Regression Onto External Human Ratings

- What it is: Fit a regressor from internal directions/components onto external scalar ratings — valence, arousal, or another human judgment.
- Typical implementation: Recover concept vectors first, then use ridge regression or another simple model to align them to human-labeled scales.
- Labels needed: External continuous ratings or ordered scales.
- What it produces: Interpretable scalar axes and quantitative alignment between internal geometry and human annotation spaces.
- Good fit for: Emotion data, risk scores, confidence/caution scales, data dimensions with human scalar ratings.
- Example papers: `Valence-Arousal Subspace in LLMs`.
- Supports: "The internal subspace aligns with a meaningful human scale."
- Limit: Alignment to ratings does not by itself show the model uses that variable causally during generation.

### 5. Residual Subtraction And Confound Projection

- What it is: Construct a concept vector while explicitly subtracting nuisance structure — shared mean activity, neutral-style confounds.
- Typical implementation: Compute residual activations after the concept is established, subtract cross-class means, project out neutral or formatting directions to isolate the target concept.
- Labels needed: Multi-class concept labels plus some explicit neutral or control condition.
- What it produces: Cleaner concept vectors, less contaminated by topic/style, more tied to the intended semantic variable.
- Good fit for: Emotion concepts, persona concepts, therapy styles, moral-framework contrasts where style confounds are likely.
- Example papers: `Emotion concepts and their function in a large language model`.
- Supports: "This concept vector is not just picking up generic style/topic."
- Limit: Depends on good controls; poor neutral controls can still leave major confounds.

### 6. Layer Sweep And Token-Position Sweep

- What it is: Evaluate the same decoding problem across many layers and token positions to see where the signal first appears and where it stabilizes.
- Typical implementation: Train/evaluate identical probes at each layer or on each token class, often plotting separability vs layer depth.
- Labels needed: Any supervised label, especially useful for sequence tasks where timing matters.
- What it produces: Localization curves over layer depth and token position — a first map of where a variable is represented.
- Good fit for: Role conflict, source attribution, deception onset, hallucination emergence, multi-turn risk escalation.
- Example papers: `Who is In Charge?`, `Detecting Strategic Deception`, `Emotion concepts`, `Role Confusion`.
- Supports: "This variable becomes linearly readable around these layers/tokens."
- Limit: Readability does not prove these are the write sites rather than read sites.

### 7. Claim Decomposition And Claim-Level Supervision

- What it is: Break long-form responses into atomic claims, then attach labels or evidence to each claim instead of scoring the whole answer at once.
- Typical implementation: Use an auxiliary model or structured heuristic to split answers into claims, align claims to source spans or evidence, then train probes on the claim-token activations.
- Labels needed: Claim-level factuality labels, evidence spans, or domain-specific support annotations.
- What it produces: Claim-level monitors, risk heatmaps, much better localized supervision for groundedness studies.
- Good fit for: RAGTruth-style datasets, finance/legal report checking, citation-grounded generation, any long-form data with support labels.
- Example papers: `Simple Factuality Probes`.
- Supports: "Specific claims become unsupported or risky here."
- Limit: The decomposition pipeline can introduce errors or bleed artifacts into the labels.

### 8. Retrieval-Based Oracle Labeling

- What it is: Use a retrieval system or evidence verifier as an approximate oracle to produce labels for factuality/support rather than relying only on humans.
- Typical implementation: Retrieve relevant evidence, verify each claim against it, use those labels to supervise hidden-state probes.
- Labels needed: Source corpora or knowledge bases, plus a verification step.
- What it produces: Scalable claim-level labels and domain-portable factuality harnesses.
- Good fit for: Finance, legal, health, RAG, citation-heavy tasks, long-form analysis.
- Example papers: `Simple Factuality Probes`.
- Supports: "This hidden-state signal tracks support/factuality as judged by the retrieval oracle."
- Limit: Oracle quality caps label quality.

### 9. Logit Lens And Output Attribution

- What it is: Inspect intermediate representations by projecting them toward vocabulary logits or attributing final-token behavior back to internal activations.
- Typical implementation: Apply logit lens at multiple layers or use attribution methods to estimate which components contribute to a final decision.
- Labels needed: Usually paired with a known output decision or class label; does not always require explicit supervision.
- What it produces: Layer-wise pictures of how a decision or token prediction is taking shape.
- Good fit for: Deception, role conflict, refusal/continuation competition, situations where we care about how an output becomes likely over depth.
- Example papers: `Can LLMs Lie?`, `Who is In Charge?`.
- Supports: "This intermediate state is already pushing toward this output."
- Limit: Often descriptive and sensitive to interpretation.

### 10. Zero Ablation Of MLPs Or Attention Components

- What it is: Set selected MLP blocks, attention outputs, or heads to zero during inference to test whether behavior changes.
- Typical implementation: Identify candidate components from prior analysis, ablate at selected token positions/layers, measure effect on the target behavior.
- Labels needed: Behavioral metric or task success label; usually paired with a contrastive task.
- What it produces: Causal evidence that a component matters for the behavior.
- Good fit for: Deception, refusal, source attribution, tasks where a specific behavior can be measured clearly.
- Example papers: `Can LLMs Lie?`, `The Struggle Between Continuation and Refusal`.
- Supports: "These components are causally important for this behavior."
- Limit: Zeroing can be a blunt intervention and may introduce off-manifold effects.

### 11. Head-Level Ablation And Head Ranking

- What it is: Evaluate individual attention heads or small head sets by ablating them and measuring behavior changes.
- Typical implementation: Score heads individually or greedily, then test the smallest set whose removal changes the behavior significantly.
- Labels needed: A measurable target behavior — lying, refusal, jailbreak success.
- What it produces: Ranked head lists, compact causal subsets, more interpretable intervention targets than whole-layer ablations.
- Good fit for: Refusal circuits, deception planning, jailbreak mechanisms, hierarchy resolution.
- Example papers: `Can LLMs Lie?`, `The Struggle Between Continuation and Refusal`.
- Supports: "A small head subset is doing a lot of the work."
- Limit: Heads can be redundant and distributed; ranking one by one may miss interactions.

### 12. Attention Blocking Between Specific Token Classes

- What it is: Prevent attention flow from one token set to another to test whether information transfer along that path is necessary.
- Typical implementation: Mask or zero attention links from, for example, subject tokens or intent tokens into dummy template tokens, then measure the behavioral effect.
- Labels needed: A task with meaningful token classes and a measurable output behavior.
- What it produces: Stronger causal path claims than global ablation alone.
- Good fit for: Deception, source confusion, retrieval-vs-system conflicts, tool-output influence, template-token scratchpads.
- Example papers: `Can LLMs Lie?`.
- Supports: "This information path is important for the target behavior."
- Limit: Requires a good hypothesis about which token classes matter.

### 13. Contrastive Activation Steering

- What it is: Add or subtract a concept direction during inference to test whether behavior moves in a predicted direction.
- Typical implementation: Construct a vector from differences, probe weights, or another concept representation, inject at chosen layers/positions, measure behavioral shift.
- Labels needed: A contrastive concept or supervised signal from which the direction can be built.
- What it produces: Steering vectors and causal evidence that moving along the direction affects behavior.
- Good fit for: Emotion, honesty, caution, moral framework, source trust, harmfulness, any label with a reasonably stable direction.
- Example papers: `Valence-Arousal Subspace`, `Can LLMs Lie?`, `Truthfulness Spectrum`, `Emotion concepts`, `CDAS`.
- Supports: "This direction is not only readable; intervening on it changes behavior."
- Limit: Steering can be lossy, off-manifold, or confounded by correlated features.

### 14. Cross-Domain Probe Transfer Matrices

- What it is: Train probes on one domain or task, test them on others, build a transfer map.
- Typical implementation: Pairwise train/test across datasets or label families, compare accuracy, AUROC, or another metric.
- Labels needed: Shared concept labels across multiple domains or task families.
- What it produces: Transfer matrices, portability estimates, insight into which concept directions are general vs domain-specific.
- Good fit for: Truthfulness, factuality, groundedness, risk-awareness, any data family with multiple related domains.
- Example papers: `Truthfulness Spectrum`, `Simple Factuality Probes`.
- Supports: "This concept transfers across these domains but not these others."
- Limit: Transfer failure can reflect data mismatch, not only concept specificity.

### 15. Mahalanobis-Cosine Similarity For Probe Portability

- What it is: Compare probe directions using covariance-aware geometry rather than plain cosine similarity.
- Typical implementation: Estimate the covariance structure of the representation space, compute similarity in the whitened space, relate that to observed transfer performance.
- Labels needed: Multiple trained probes over related concepts/domains.
- What it produces: A geometry-based transfer estimator.
- Good fit for: Probe banks, domain transfer analysis, deciding whether a direction trained on one dataset should move to another.
- Example papers: `The Truthfulness Spectrum Hypothesis`.
- Supports: "These probe directions are likely to transfer because they are close in the right geometry."
- Limit: Promising but still needs broader validation beyond the original paper's setting.

### 16. Concept Erasure With LEACE And Stratified INLP

- What it is: Remove a concept direction from representations to test whether the model still succeeds or whether certain behaviors disappear.
- Typical implementation: Use LEACE or INLP-style projection to erase the learned subspace, sometimes stratified by domain, then evaluate performance/behavior afterward.
- Labels needed: A supervised concept and enough data to estimate the subspace robustly.
- What it produces: Domain-general vs domain-specific subspaces, stronger test of concept necessity than readout alone.
- Good fit for: Truthfulness, sycophancy, theory labels, data dimensions with clear supervision.
- Example papers: `The Truthfulness Spectrum Hypothesis`.
- Supports: "This behavior depends on this subspace, and this part is general vs domain-specific."
- Limit: Erasure can damage unrelated information if the concept is entangled.

### 17. Layer-Wise Activation Projection To Reference Models

- What it is: Compare one model's activations against reference differences from other models — base vs instruct vs fine-tuned variants.
- Typical implementation: Compute residual differences between model variants, project one run onto those differences layer by layer, see which reference manifold it aligns with over depth.
- Labels needed: Usually model-variant comparisons rather than data labels.
- What it produces: Internal drift curves and alignment/misalignment trajectories over layers.
- Good fit for: Post-finetune audit questions, alignment erosion, preference optimization studies, model-family comparisons.
- Example papers: `Re-Emergent Misalignment`.
- Supports: "This fine-tuned model drifts internally toward or away from the aligned reference over depth."
- Limit: Comparative and geometric, but not always strongly causal.

### 18. Loss Geometry And Gradient Similarity

- What it is: Use gradients or loss surfaces to compare how similarly different models/processes treat the same examples.
- Typical implementation: Measure gradient similarity, output-probability geometry, or loss alignment across model variants.
- Labels needed: Behavioral targets or curated example sets, often harmful vs harmless.
- What it produces: Evidence that two variants process a behavior similarly or differently at the optimization/gradient level.
- Good fit for: Alignment erosion, preference optimization, post-finetune audit work.
- Example papers: `Re-Emergent Misalignment`.
- Supports: "These models are similar or divergent in how they treat this behavior."
- Limit: Harder to turn directly into product-like monitors.

### 19. Residual-Difference SVD Or Low-Rank Shared-Direction Discovery

- What it is: Factorize residual differences between conditions/models to find shared low-rank directions.
- Typical implementation: Compute residual difference matrices and run SVD or related factorization to find common latent axes.
- Labels needed: Model variants or multiple related harmfulness settings.
- What it produces: Shared latent directions linking multiple behaviors or domains.
- Good fit for: Misalignment, safety-drift, harm transfer, domain-adaptation audits.
- Example papers: `Re-Emergent Misalignment`.
- Supports: "These seemingly different behaviors share a latent internal axis."
- Limit: Shared direction does not automatically imply a manipulable or unique mechanism.

### 20. Signed SNIP-Style Targeted Weight Pruning

- What it is: Score weights for how much they contribute to harmful behavior vs benign capability, then prune a targeted subset.
- Typical implementation: Use harmful-response data plus a preservation dataset, compute signed saliency scores, remove a tiny selected subset of weights, test both harmfulness and utility.
- Labels needed: Harmful/benign example sets and a capability-preservation dataset.
- What it produces: A pruned model, weight subsets, evidence about whether harm depends on a compact shared subnetwork.
- Good fit for: Harm generation, refusal, emergent misalignment, open-weight safety questions.
- Example papers: `Pruning a Tiny High-Harm Subnetwork from LLMs`.
- Supports: "This tiny weight subset is causally important for harmful generation."
- Limit: Weight-level, not activation-first; harder to turn directly into a monitor.

### 21. Generation-Vs-Detection Dissociation Tests

- What it is: Test whether a model can still recognize/explain harmful content after an intervention that reduces harmful generation.
- Typical implementation: Measure generation ability, detection ability, refusal ability, and explanation ability separately before and after intervention.
- Labels needed: Distinct evaluation sets for generation, recognition, refusal, or explanation.
- What it produces: A dissociation result showing whether "knowing" and "doing" are separated by the intervention.
- Good fit for: Harmfulness, deception, refusal, safety mechanism questions.
- Example papers: `Pruning a Tiny High-Harm Subnetwork from LLMs`.
- Supports: "The model can still recognize the bad behavior even though it no longer executes it."
- Limit: Requires well-designed evaluation splits to avoid ambiguous interpretation.

### 22. Probe Architecture Bakeoffs

- What it is: Compare multiple probe families rather than assuming a single linear probe is enough.
- Typical implementation: Evaluate linear probes, MLP probes, attention-based probes, pooling strategies, context windows, cascades.
- Labels needed: Any supervised label with enough train/test data, especially where OOD behavior matters.
- What it produces: Performance comparisons, robustness curves, deployment guidance.
- Good fit for: High-stakes detection, cyber misuse, long-context monitoring, any product-facing work.
- Example papers: `Detecting High-Stakes Interactions`, `Building Production-Ready Probes for Gemini`.
- Supports: "This probe family is more robust/effective for this deployment setting."
- Limit: More engineering-heavy and less mechanistically explanatory on its own.

### 23. Cascade Systems

- What it is: Use a cheap activation-side monitor as a first pass and escalate flagged cases to a larger model or more expensive judge.
- Typical implementation: Probe scores determine whether to route to a stronger evaluator or additional analysis pipeline.
- Labels needed: A task label plus some notion of high-cost review benefit.
- What it produces: Practical monitor pipelines and cost-performance tradeoff curves.
- Good fit for: High-stakes triage, groundedness review, cyber misuse, counseling risk, finance/legal escalation.
- Example papers: `Detecting High-Stakes Interactions`, `Building Production-Ready Probes for Gemini`.
- Supports: "A cheap white-box filter can reduce cost while preserving most of the value of a larger monitor."
- Limit: Product-useful, but not always a deep mechanistic claim.

### 24. Long-Context And Distribution-Shift Stress Testing

- What it is: Evaluate whether probe directions or architectures survive when context is longer, multi-turn, jailbroken, or adversarially shifted.
- Typical implementation: Test the same monitor under clean, long-context, multi-turn, jailbreak, and adaptive-red-team conditions.
- Labels needed: Evaluation datasets that intentionally span deployment shifts.
- What it produces: OOD robustness estimates and deployment-readiness criteria.
- Good fit for: Any monitor likely to ship, especially agent, high-stakes, or retrieval products.
- Example papers: `Building Production-Ready Probes for Gemini`.
- Supports: "This monitor survives or fails under realistic shift."
- Limit: Critical for shipping, but not itself a discovery method unless paired with deeper analysis.

### 25. Architecture Search For Probe Design

- What it is: Search over probe architectures automatically rather than choosing one by hand.
- Typical implementation: Use methods like AlphaEvolve or other search procedures to discover better long-context or multi-turn probe architectures.
- Labels needed: A supervised task and enough evaluation bandwidth to score many candidate probes.
- What it produces: Better-performing probe architectures and sometimes interpretable design lessons.
- Good fit for: Product-facing work where robustness matters more than minimalism.
- Example papers: `Building Production-Ready Probes for Gemini`.
- Supports: "This architecture is more effective for this monitoring problem."
- Limit: More systems/engineering than mechanism explanation.

### 26. Path Patching

- What it is: Intervene on a specific activation path by swapping or patching signals from one run into another.
- Typical implementation: Patch attention or residual paths between clean and attacked runs to localize the path carrying the critical information.
- Labels needed: Paired examples where one run has the target behavior and another does not.
- What it produces: Much stronger causal localization than readout alone.
- Good fit for: Jailbreaks, source confusion, role conflict, theory contrasts, any dataset with matched conditions.
- Example papers: `The Struggle Between Continuation and Refusal`; also highly relevant to `Role Confusion` style future work.
- Supports: "This path carries the decisive information for the behavior."
- Limit: Requires carefully matched paired examples and good intervention hygiene.

### 27. Activation Scaling Of Candidate Components

- What it is: Multiply selected heads/features/components up or down to test whether the target behavior changes monotonically.
- Typical implementation: Identify candidate components, rescale them during inference, measure behavior or attack success rate.
- Labels needed: Clear behavioral metric — ASR, refusal, deception rate, rubric dimension score.
- What it produces: Causal sensitivity curves and evidence that a component is not just correlated but functionally influential.
- Good fit for: Refusal circuits, harmfulness, emotion intensity, any near-linear component hypothesis.
- Example papers: `The Struggle Between Continuation and Refusal`.
- Supports: "Increasing this component pushes the model toward or away from this behavior."
- Limit: Scaling can create off-distribution behavior if overused.

### 28. Distributed Interchange Interventions And Distribution Matching

- What it is: Learn steering interventions using interchange-based causal ideas plus distribution matching rather than only simple direction vectors.
- Typical implementation: Use DII-style interventions and a distributional objective like JSD to find more faithful steering directions or subspaces.
- Labels needed: A target behavior/concept and a way to compare the intervened distribution to a reference distribution.
- What it produces: Faithful steering vectors/subspaces and stronger causal control than naive contrastive steering.
- Good fit for: Behaviors where ordinary steering distorts utility too much — refusal, backdoors, fine-grained rubric dimensions.
- Example papers: `Faithful Bi-Directional Model Steering via Distribution Matching and Distributed Interchange Interventions`.
- Supports: "We can steer this concept while preserving more of the surrounding behavior."
- Limit: More complex and infrastructure-heavy than simple readout methods.

### 29. Synthetic Data Construction For Mechanistic Supervision

- What it is: Deliberately generate examples that isolate a latent variable so downstream probes and interventions are cleaner.
- Typical implementation: Create synthetic role corpora, conflict prompts, emotion stories, high-stakes scenarios, often with procedural controls to hold nuisance factors constant.
- Labels needed: Often self-generated labels from the synthetic template itself.
- What it produces: Clean supervision sets, matched controls, scalable training/eval harnesses.
- Good fit for: Role integrity, theory labels, high-stakes classification, emotion concepts, hierarchy resolution.
- Example papers: `Role Confusion`, `Emotion concepts`, `Who is In Charge?`, `Detecting High-Stakes Interactions`.
- Supports: "We can isolate this variable cleanly enough to study it mechanistically."
- Limit: Synthetic simplicity may not survive transfer to real-world data.

This is flywheel stage 2 as a method family. Every subproject should expect to construct synth.

---

## Mapping Methodologies To Label Structures

### Rubric-rich substrates

- Linear residual probes
- Multi-label probe families
- Claim decomposition if outputs are long-form
- Layer/token sweeps
- Contrastive steering on single rubric dimensions
- Long-context robustness testing if product-facing

Example substrates: rubric-rich safety, legal, health, or policy benchmarks.

### Theory or framework labels

- Difference-in-means directions
- Framework-conditioned probes
- PCA/subspace recovery across many related concepts
- Activation steering
- Patching between matched frameworks
- LEACE / INLP erasure for domain-general vs domain-specific structure

Example substrates: theory-labeled dialogue, therapy, or safety benchmarks.

### Span or claim supervision

- Claim decomposition
- Token/span probes
- Emergence-point tracing
- Retrieval-oracle supervision
- Hallucination-type transfer matrices

Example substrates: RAGTruth, FAVA, FELM, HALoGEN, LegalBench-RAG.

### Belief-vs-pressure or contrastive honesty data

- Linear probes
- Difference-in-means vectors
- Patching between belief and pressured runs
- Attention blocking
- Head ablation
- Cross-domain transfer mapping

Example substrates: MASK, SycophancyEval, some Anthropic model-written eval slices.

### Hierarchy, role, or source labels

- Role/source probes
- Layer sweeps
- Path patching
- Attention blocking
- Synthetic-data warm-starts
- Transfer from clean hierarchy data into real agent traces

Example substrates: ManyIH-Bench, AgentDojo, Who is In Charge?, BIPIA. DX Terminal-style real-agent data fits here.

### Multi-turn or trajectory data

- Conversation-state probes
- Turn-by-turn scoring
- Drift analysis
- Intervention timing experiments
- Cascade systems
- Long-context robustness testing

Example substrates: AgentDojo, HealthBench, MACHIAVELLI, tau-bench, production agent traces.

---

## Evidence Escalation Ladder

This is claim hygiene, not flywheel order. It runs alongside the flywheel. The flywheel says what moves are available next; the ladder says what claims the current evidence earns.

When a candidate signal looks promising, the most useful escalation order is:

1. **Descriptive readout** — layer sweeps, token sweeps, simple probes.
2. **Geometry or transfer** — compare layers, domains, models, transferability.
3. **Intervention** — steering, erasure, ablation, patching.
4. **Deployment hardening** — OOD testing, long-context testing, cascades, architecture bakeoffs.

Mapping to `PRINCIPLES.md` evidence levels:

- Steps 1–2 typically earn Levels 2–3 (representational, localized representational).
- Step 3 is required for Level 4 (causal).
- Step 4 is a separate product-facing axis; it tests robustness, not depth of claim.

Do not skip levels. A probe AUROC does not buy a causal claim; an intervention effect does not buy a mechanistic claim without a plausible computation path.

---

## Bottom Line

There is no single "probe method." There is a stack:

- **Readout methods** for finding signals
- **Geometry methods** for understanding structure and transfer
- **Causal methods** for testing whether the signal matters
- **Deployment methods** for figuring out whether the thing survives real use

Every subproject should plan to touch multiple tiers before making strong claims.
